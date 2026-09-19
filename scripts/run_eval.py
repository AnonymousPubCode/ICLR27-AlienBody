#!/usr/bin/env python3
"""Batch evaluation script for AlienBody benchmark.

Usage:
    # Run GPT-4o on all test environments
    python scripts/run_eval.py --agent gpt-4o --family all --split test

    # Run random baseline on Family 1 dev set
    python scripts/run_eval.py --agent random --family 1 --split dev

    # Run with specific settings
    python scripts/run_eval.py --agent gpt-4o --modality text --prompt cot --familiar

    # Resume interrupted run
    python scripts/run_eval.py --agent gpt-4o --resume

    # E-C2 name-prior prereg runs: feed the frozen, pre-call-audited label
    # packs read from disk (never recomputed); one condition per run, with a
    # per-condition output directory
    python scripts/run_eval.py --agent vllm:qwen3.5-9b \
        --data-dir data/frozen_suites/replication_primary_v3 \
        --split replication_primary --family all \
        --label-pack-suite replication_primary_v3 \
        --label-pack-condition nonce_matched \
        --output results/name_prior_replication/nonce_matched

Two label paths exist and must not be mixed:
  * ``--label-pack-suite`` + ``--label-pack-condition`` (E-C2, preferred):
    loads ``data/label_packages/<suite>/<condition>/env_<env_id>.json`` --
    the exact bytes the pre-call audit approved (sha256-checked against the
    suite manifest).  A missing/mismatched pack aborts the run, and every
    trajectory records the pack path + sha256 it was fed.
  * ``--action-labels true|misleading`` (legacy, kept for backwards
    compatibility with earlier analyses): recomputes labels via
    ``alienbody.labels.labels_for``-equivalent logic; its ``misleading`` is the
    ``+1`` rotation, NOT the pre-registered per-environment derangement.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Fix Windows console encoding for Unicode characters (→, etc.)
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent.parent))

from alienbody.env.core import AlienBodyEnv
from alienbody.env.grid import EnvConfig
from alienbody.env.generator import validate_solvable
from alienbody.agents import RandomAgent, OracleAgent, run_episode, DONE_EXPLORING
from alienbody.eval import compute_all_metrics, aggregate_metrics
from alienbody.trajectory import TrajectoryWriter, get_completed_env_ids
from alienbody.labels import (
    CONDITIONS,
    DEFAULT_LABEL_PACK_DIR,
    LabelPack,
    LabelPackError,
    load_label_pack,
    resolve_pack_dir,
)


class MissingFamilyPoolError(RuntimeError):
    """A requested family's environment pool is absent from the data dir."""


def load_environments(data_dir: Path, family: str, split: str,
                      allow_missing: bool = False) -> list[EnvConfig]:
    """Load all environment configs for given family and split.

    A requested family whose pool directory (or env_*.json set) is missing is a
    hard error: previously it only warned + skipped, so a partial local suite
    produced a plausible SR and exit 0.  ``allow_missing=True`` restores the old
    warn-and-skip behaviour for deliberate partial-suite runs.
    """
    families = list(range(1, 7)) if family == "all" else [int(family)]
    configs = []
    missing: list[str] = []

    for fam in families:
        env_dir = data_dir / f"family{fam}" / split
        if not env_dir.exists():
            missing.append(f"{env_dir} (directory not found)")
            continue
        env_files = sorted(env_dir.glob("env_*.json"))
        if not env_files:
            missing.append(f"{env_dir} (no env_*.json)")
            continue
        for json_file in env_files:
            configs.append(EnvConfig.from_file(str(json_file)))

    if missing:
        detail = "; ".join(missing)
        if allow_missing:
            print(f"  Warning: {len(missing)} requested family pool(s) missing, "
                  f"skipping: {detail}")
        else:
            raise MissingFamilyPoolError(
                f"requested family pool(s) missing for family={family!r} "
                f"split={split!r}: {detail}.  Refusing to run: the SR would "
                f"silently cover fewer families than requested.  Point --data-dir "
                f"at a complete suite, or pass --allow-missing-family for a "
                f"deliberate partial run.")

    return configs


def create_agent(agent_name: str, config: EnvConfig, modality: str = "image",
                 prompt_variant: str = "minimal", familiar: bool = False,
                 assist_level: int = 0, vllm_base_url: str | None = None,
                 action_labels: list[str] | None = None,
                 tool: bool = False, code_tool: bool = False,
                 max_tokens: int | None = None,
                 induce_rounds: int = 4, induce_fallback: bool = True,
                 bon_n: int = 1):
    """Create an agent by name.

    Supported agents:
      Control:       random, oracle, systematic
      Algorithmic:   bayesian, memory
      RL (trained):  rl:<path>, meta-rl:<path>
      Method v3:     dsl-blind, dsl-propose+<model>
      LLM/VLM:       gpt-4o, gpt-4o-mini, claude-3.7, gemini-2.5-pro, ...
      vLLM:          vllm:qwen3.5-4b, vllm:qwen3.5-9b, ...
      Frameworks:    react+<model>, reflexion+<model>,
                     plan+<model>, monologue+<model>

    Assist levels (diagnostic ladder):
      0 = normal, 1 = familiar, 2 = summary, 3 = mapping given
    """
    name_lower = agent_name.lower()

    if name_lower == "random":
        return RandomAgent(n_actions=config.n_actions, seed=hash(config.env_id))

    if name_lower == "oracle":
        return OracleAgent(config)

    if name_lower == "systematic":
        from alienbody.agents import SystematicExplorer
        return SystematicExplorer(n_actions=config.n_actions)

    if name_lower == "fmb":
        from alienbody.agents.fmb_agent import FMBAgent
        return FMBAgent(config)

    if name_lower == "enum-verify":
        from alienbody.agents.enumerate_agent import EnumerateVerifyAgent
        return EnumerateVerifyAgent(config)

    if name_lower == "afmb" or name_lower.startswith("afmb:"):
        from alienbody.agents.afmb_agent import AFMBAgent
        mode = None
        if ":" in agent_name:
            mode = agent_name.split(":", 1)[1]
        return AFMBAgent(config, mode=mode)

    if name_lower == "dsl-blind":
        from alienbody.agents.dsl_agent import DSLBlindAgent
        return DSLBlindAgent(config)

    if name_lower.startswith("dsl-propose+"):
        from alienbody.agents.dsl_agent import DSLProposeAgent
        model_spec = agent_name.split("+", 1)[1]
        client = _make_model_client(model_spec, modality, prompt_variant, familiar)
        return DSLProposeAgent(config, client, induce_rounds=induce_rounds)

    if name_lower.startswith("dsl-catalog+"):
        from alienbody.agents.dsl_agent import DSLCatalogProposeAgent
        model_spec = agent_name.split("+", 1)[1]
        client = _make_model_client(model_spec, modality, prompt_variant, familiar)
        return DSLCatalogProposeAgent(config, client, induce_rounds=induce_rounds)

    # FMB with VLM induction: fmb-vlm+<model_spec>
    # e.g. fmb-vlm+gateway:gpt-4o, fmb-vlm+vllm:qwen3.5-4b
    if name_lower.startswith("fmb-vlm+"):
        from alienbody.agents.fmb_vlm import VLMFMBAgent
        from alienbody.agents.llm_agent import make_agent
        model_spec = agent_name[8:]  # part after "fmb-vlm+"
        induction_client = _make_model_client(model_spec, modality, prompt_variant, familiar)
        return VLMFMBAgent(config, induction_client, explore_agent_name="double")

    # FMB with active babbling: fmb-active+<model_spec>
    if name_lower.startswith("fmb-active+"):
        from alienbody.agents.fmb_active import ActiveBabblingAgent
        from alienbody.agents.llm_agent import make_agent
        model_spec = agent_name[11:]  # part after "fmb-active+"
        vlm_client = _make_model_client(model_spec, modality, prompt_variant, familiar)
        return ActiveBabblingAgent(config, vlm_client)

    # Inductive FMB: LLM proposal + simulator verification loop
    #   fmb-ind+<model_spec>  (e.g. fmb-ind+gateway:gpt-4o)
    if name_lower.startswith("fmb-ind+"):
        from alienbody.agents.fmb_inductive import InductiveFMBAgent
        model_spec = agent_name[8:]  # part after "fmb-ind+"
        induction_client = _make_model_client(model_spec, modality, prompt_variant, familiar)
        return InductiveFMBAgent(config, induction_client,
                                 induce_rounds=induce_rounds,
                                 fallback=induce_fallback)

    # FMB with trained LoRA: fmb-trained:<lora_path>
    if name_lower.startswith("fmb-trained:"):
        from alienbody.agents.fmb_trained import make_trained_fmb_agent
        lora_path = agent_name[12:]  # part after "fmb-trained:"
        return make_trained_fmb_agent(config, lora_path)

    # FMB with text-model LoRA or full model:
    #   fmb-text:<model_path>:<lora_path>[:cot|fullft]
    # e.g. fmb-text:models/Qwen3.5-4B:models/fmb_stage1_text_4b/final
    #      fmb-text:models/Qwen3.5-9B:models/f4_cot_2000/final:cot
    #      fmb-text:models/Qwen3.5-4B:models/f4_fullft_2000/final:fullft
    if name_lower.startswith("fmb-text:"):
        from alienbody.agents.fmb_text import TextFMBAgent
        parts = agent_name[9:].split(":")
        if len(parts) in (2, 3):
            prompt_format = parts[2].strip().lower() if len(parts) == 3 else None
            if prompt_format not in (None, "cot", "fullft", "multiobs"):
                raise ValueError("fmb-text 3rd segment must be cot, fullft, or multiobs")
            return TextFMBAgent(config, model_path=parts[0], lora_path=parts[1],
                                prompt_format=prompt_format)
        raise ValueError("fmb-text requires <model_path>:<lora_path>[:cot|fullft]")

    if name_lower == "bayesian":
        from alienbody.agents.algorithmic_agents import BayesianExplorer
        return BayesianExplorer(n_actions=config.n_actions)

    if name_lower == "memory":
        from alienbody.agents.algorithmic_agents import MemoryAugmentedExplorer
        return MemoryAugmentedExplorer(n_actions=config.n_actions)

    # Permission-matched, observation-only baselines (P0-B).
    # Constructed with n_actions ONLY -- not even the config object is passed,
    # so these agents cannot read the ground-truth action mapping.
    # ``obs_bayes`` is the alias used for the results directory name.
    if name_lower in ("psrl", "rmax", "obs-bayes", "obs_bayes"):
        from alienbody.agents.permission_matched import (
            PSRLAgent, RMaxAgent, ObsBayesAgent)
        agent_cls = {"psrl": PSRLAgent, "rmax": RMaxAgent,
                     "obs-bayes": ObsBayesAgent,
                     "obs_bayes": ObsBayesAgent}[name_lower]
        return agent_cls(config.n_actions)

    # RL agents (require a trained model path)
    if name_lower.startswith("rl:"):
        from alienbody.agents.rl_agent import SB3Agent
        return SB3Agent(model_path=agent_name[3:], n_actions=config.n_actions)

    if name_lower.startswith("meta-rl:"):
        from alienbody.agents.rl_agent import MetaRLAgent
        return MetaRLAgent(model_path=agent_name[8:], n_actions=config.n_actions)

    # Framework wrappers
    if name_lower.startswith("react+"):
        from alienbody.agents.llm_agent import make_agent
        from alienbody.agents.framework_agents import ReActAgent
        base = make_agent(agent_name[6:], modality=modality,
                          prompt_variant=prompt_variant, familiar=familiar)
        return ReActAgent(base)

    if name_lower.startswith("reflexion+"):
        from alienbody.agents.llm_agent import make_agent
        from alienbody.agents.framework_agents import ReflexionAgent
        base = make_agent(agent_name[10:], modality=modality,
                          prompt_variant=prompt_variant, familiar=familiar)
        return ReflexionAgent(base)

    if name_lower.startswith("plan+"):
        from alienbody.agents.llm_agent import make_agent
        from alienbody.agents.framework_agents import PlanAndExecuteAgent
        base = make_agent(agent_name[5:], modality=modality,
                          prompt_variant=prompt_variant, familiar=familiar)
        return PlanAndExecuteAgent(base)

    if name_lower.startswith("monologue+"):
        from alienbody.agents.llm_agent import make_agent
        from alienbody.agents.framework_agents import InnerMonologueAgent
        base = make_agent(agent_name[10:], modality=modality,
                          prompt_variant=prompt_variant, familiar=familiar)
        return InnerMonologueAgent(base)

    def _wrap_l3_tools(client, modality_local: str):
        from alienbody.prompts import PromptVariant
        if code_tool:
            from alienbody.agents.code_tool_agent import CodeToolAgent
            return CodeToolAgent(
                config, client, modality=modality_local,
                prompt_variant=PromptVariant(prompt_variant),
                response_max_tokens=max_tokens or 2048,
            )
        if tool:
            from alienbody.agents.tool_agent import ToolAgent
            return ToolAgent(config, client, modality=modality_local,
                             prompt_variant=PromptVariant(prompt_variant))
        return None

    # gateway API models (explicit prefix)
    if name_lower.startswith("gateway:"):
        from alienbody.agents.gateway_client import GatewayClient
        from alienbody.agents.llm_agent import LLMAgent
        from alienbody.prompts import PromptVariant
        gateway_model = agent_name[5:]
        action_mapping_list = list(config.action_mapping) if assist_level == 3 else None
        client = GatewayClient(model=gateway_model)
        wrapped = _wrap_l3_tools(client, modality)
        if wrapped is not None:
            return wrapped
        return LLMAgent(
            client=client, modality=modality,
            prompt_variant=PromptVariant(prompt_variant),
            familiar=familiar, n_actions=config.n_actions,
            assist_level=assist_level, action_mapping=action_mapping_list,
            action_labels=action_labels,
            response_max_tokens=max_tokens or 128,
        )

    # vLLM models (explicit prefix)
    if name_lower.startswith("vllm:"):
        from alienbody.agents.llm_agent import VLLMClient, LLMAgent
        from alienbody.prompts import PromptVariant
        vllm_model = agent_name[5:]
        action_mapping_list = list(config.action_mapping) if assist_level == 3 else None
        client_kwargs = {}
        if vllm_base_url:
            client_kwargs["base_url"] = vllm_base_url
        client = VLLMClient(model=vllm_model, **client_kwargs)
        wrapped = _wrap_l3_tools(client, modality)
        if wrapped is not None:
            return wrapped
        return LLMAgent(
            client=client, modality=modality,
            prompt_variant=PromptVariant(prompt_variant),
            familiar=familiar, n_actions=config.n_actions,
            assist_level=assist_level, action_mapping=action_mapping_list,
            action_labels=action_labels,
            response_max_tokens=max_tokens or 128,
        )

    # LLM/VLM agents (direct model names)
    try:
        from alienbody.agents.llm_agent import make_agent
        action_mapping = list(config.action_mapping) if assist_level == 3 else None
        kwargs = {}
        if vllm_base_url:
            kwargs["base_url"] = vllm_base_url
        agent = make_agent(agent_name, modality=modality,
                           prompt_variant=prompt_variant, familiar=familiar,
                           assist_level=assist_level, action_mapping=action_mapping,
                           bon_n=bon_n, **kwargs)
        return agent
    except (ImportError, ValueError) as e:
        raise ValueError(
            f"Unknown agent: {agent_name}. Options:\n"
            f"  Control:     random, oracle, systematic, bayesian, memory\n"
            f"  RL:          rl:<model_path>, meta-rl:<model_path>\n"
            f"  vLLM:        vllm:qwen3.5-4b, vllm:qwen3.5-9b, ...\n"
            f"  LLM/VLM:    gpt-4o, gpt-4o-mini, claude-3.7, gemini-2.5-pro, ...\n"
            f"  Frameworks:  react+<model>, reflexion+<model>, plan+<model>, monologue+<model>"
        ) from e


def _make_model_client(model_spec: str, modality: str = "image",
                       prompt_variant: str = "minimal", familiar: bool = False):
    """Create a ModelClient from a model spec string.

    Supports: gateway:<model>, vllm:<model>, or bare model names
    (gpt-4o, gemini-2.5-pro, etc.) resolved through make_agent's client.
    """
    spec_lower = model_spec.lower()

    if spec_lower.startswith("gateway:"):
        from alienbody.agents.gateway_client import GatewayClient
        return GatewayClient(model=model_spec[5:])

    if spec_lower.startswith("vllm:"):
        from alienbody.agents.llm_agent import VLLMClient
        return VLLMClient(model=model_spec[5:])

    # Bare model name: try gateway first, then vLLM
    try:
        from alienbody.agents.gateway_client import GatewayClient
        return GatewayClient(model=model_spec)
    except Exception:
        from alienbody.agents.llm_agent import VLLMClient
        return VLLMClient(model=model_spec)


def run_single(config: EnvConfig, agent_name: str, modality: str,
               prompt_variant: str, familiar: bool, assist_level: int = 0,
               vllm_base_url: str | None = None,
               action_labels: list[str] | None = None,
               tool: bool = False, code_tool: bool = False,
               max_tokens: int | None = None,
               induce_rounds: int = 4, induce_fallback: bool = True,
               bon_n: int = 1) -> dict:
    """Run a single episode and return trajectory with metrics."""
    env = AlienBodyEnv(config, render_mode="both")
    agent = create_agent(agent_name, config, modality, prompt_variant, familiar,
                         assist_level=assist_level, vllm_base_url=vllm_base_url,
                         action_labels=action_labels, tool=tool,
                         code_tool=code_tool,
                         max_tokens=max_tokens,
                         induce_rounds=induce_rounds, induce_fallback=induce_fallback,
                         bon_n=bon_n)

    trajectory = run_episode(env, agent)

    # Attach metadata
    trajectory["agent_name"] = agent_name
    trajectory["modality"] = modality
    trajectory["prompt_variant"] = prompt_variant
    trajectory["familiar"] = familiar
    trajectory["action_labels"] = action_labels
    trajectory["tool"] = tool
    trajectory["code_tool"] = code_tool

    # Compute metrics
    metrics = compute_all_metrics(trajectory)
    trajectory["metrics"] = metrics

    # Attach reasoning log if available
    if hasattr(agent, "get_reasoning_log"):
        trajectory["reasoning_log"] = agent.get_reasoning_log()

    # Attach tool-call log if available (A1 next_state experiment)
    if hasattr(agent, "get_tool_log"):
        trajectory["tool_log"] = agent.get_tool_log()
        trajectory["n_tool_calls"] = len(trajectory["tool_log"])

    # Attach induction log if available (Inductive FMB)
    if hasattr(agent, "get_induction_log"):
        trajectory["induction_log"] = agent.get_induction_log()
        log = trajectory["induction_log"]
        trajectory["n_induce_rounds"] = len([e for e in log if e.get("round", -1) >= 0])
        trajectory["induction_fallback"] = any(e.get("fallback") for e in log)
        trajectory["induction_verified"] = any(
            e.get("verified") for e in log
        )

    return trajectory


def main():
    parser = argparse.ArgumentParser(description="AlienBody Batch Evaluation")
    parser.add_argument("--agent", type=str, required=True,
                        help="Agent name: random, oracle, gpt-4o, claude-sonnet-4-20250514, etc.")
    parser.add_argument("--family", type=str, default="all",
                        help="Family: 1, 2, 3, 4, or all")
    parser.add_argument("--split", type=str, default="test",
                        help="Split: train, dev, test, secret")
    parser.add_argument("--modality", type=str, default="image",
                        choices=["image", "text", "both"])
    parser.add_argument("--prompt", type=str, default="minimal",
                        choices=["minimal", "cot", "explore_first", "expert_demo"])
    parser.add_argument("--familiar", action="store_true")
    parser.add_argument("--assist-level", type=int, default=0,
                        choices=[0, 1, 2, 3],
                        help="Diagnostic ladder: 0=normal, 1=familiar, 2=summary, 3=mapping given")
    parser.add_argument("--action-labels", type=str, default=None,
                        choices=["true", "misleading"],
                        help="Name-prior ablation: reveal button labels. "
                             "true = labels match the ground-truth mapping; "
                             "misleading = labels are a shifted permutation (all wrong)")
    parser.add_argument("--label-pack-suite", type=str, default=None,
                        metavar="SUITE",
                        help="E-C2: load every environment's name-prior labels "
                             "from the frozen, pre-call-audited pack files "
                             f"({DEFAULT_LABEL_PACK_DIR}/<suite>/<condition>/"
                             "env_<env_id>.json) instead of recomputing them. "
                             "This is the path the preregistered runs must use. "
                             "Requires --label-pack-condition; mutually exclusive "
                             "with --action-labels (the legacy recompute path).")
    parser.add_argument("--label-pack-condition", type=str, default=None,
                        choices=list(CONDITIONS),
                        help="Condition whose audited pack to load; requires "
                             "--label-pack-suite. anonymous = no label block; "
                             "true = ground truth; misleading = per-env "
                             "derangement; nonce_matched / synonym_true / "
                             "id_permuted as pre-registered.")
    parser.add_argument("--tool", action="store_true",
                        help="A1 clean-planning experiment: give the agent a "
                             "next_state(r,c,a) one-step transition oracle "
                             "(requires --assist-level 3; simulator queries "
                             "consume no env steps)")
    parser.add_argument("--code-tool", action="store_true",
                        help="L3 + Python REPL over next_state (write BFS). "
                             "Requires --assist-level 3. Mutually exclusive "
                             "with --tool.")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Max output tokens per model turn "
                             "(default 128; thinking models need >= 600; "
                             "code-tool default 2048)")
    parser.add_argument("--induce-rounds", type=int, default=4,
                        help="Induction rounds for fmb-ind/dsl agents")
    parser.add_argument("--bon", type=int, default=1,
                        help="Best-of-N: sample N candidates per phase-2 step and majority-vote")
    parser.add_argument("--induce-fallback", action="store_true", default=True,
                        help="Inductive FMB: fall back to permutation "
                             "enumeration when no proposal fully verifies "
                             "(default: on)")
    parser.add_argument("--no-induce-fallback", action="store_false",
                        dest="induce_fallback",
                        help="Disable the permutation-enumeration fallback")
    parser.add_argument("--n-envs", type=int, default=-1,
                        help="Max environments (-1 = all)")
    parser.add_argument("--n-workers", type=int, default=1,
                        help="Parallel workers (>1 only for API agents)")
    parser.add_argument("--vllm-base-url", type=str, default=None,
                        help="vLLM server base URL (e.g., http://localhost:8000/v1)")
    parser.add_argument("--vllm-host", type=str, default=None,
                        help="vLLM server host (overrides VLLM_HOST env)")
    parser.add_argument("--vllm-port", type=int, default=None,
                        help="vLLM server port (overrides VLLM_PORT env or default)")
    parser.add_argument("--data-dir", type=str, default="data/envs")
    parser.add_argument("--allow-missing-family", action="store_true",
                        help="Permit a requested family whose env pool is "
                             "absent (default: hard error).  Only for "
                             "deliberate partial-suite runs: the printed SR "
                             "then covers fewer families than requested.")
    parser.add_argument("--output", type=str, default="results")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-completed environments")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.tool and args.assist_level != 3:
        parser.error("--tool requires --assist-level 3 (clean-planning setting)")
    if args.code_tool and args.assist_level != 3:
        parser.error("--code-tool requires --assist-level 3")
    if args.tool and args.code_tool:
        parser.error("--tool and --code-tool are mutually exclusive")

    # E-C2 label-pack path: frozen audited packs read from disk (see the module
    # docstring).  The legacy --action-labels recompute path stays untouched.
    if args.label_pack_condition and not args.label_pack_suite:
        parser.error("--label-pack-condition requires --label-pack-suite")
    if args.label_pack_suite and not args.label_pack_condition:
        parser.error("--label-pack-suite requires --label-pack-condition "
                     f"<{'|'.join(CONDITIONS)}>")
    if args.label_pack_suite and args.action_labels:
        parser.error("--label-pack-suite and --action-labels are mutually "
                     "exclusive: --action-labels is the legacy recompute path "
                     "(true/misleading only, its 'misleading' is the +1 "
                     "rotation); the audited pack path selects its condition "
                     "with --label-pack-condition")
    if args.label_pack_suite and args.assist_level == 3:
        parser.error("--label-pack-suite requires --assist-level 0, 1 or 2: the "
                     "assist-level-3 prompt gives the action mapping and never "
                     "shows interface labels")
    if args.label_pack_suite:
        pack_root = resolve_pack_dir(DEFAULT_LABEL_PACK_DIR)
        suite_dir = pack_root / args.label_pack_suite
        if not suite_dir.is_dir():
            available = (sorted(p.name for p in pack_root.iterdir() if p.is_dir())
                         if pack_root.is_dir() else [])
            parser.error(f"unknown --label-pack-suite {args.label_pack_suite!r}: "
                         f"{suite_dir} not found"
                         + (f"; available suites: {', '.join(available)}"
                            if available else f" (pack root {pack_root} missing)"))
        if not (args.agent.lower().startswith("gateway:")
                or args.agent.lower().startswith("vllm:")):
            print(f"  WARNING: --label-pack-suite with agent {args.agent!r}: only "
                  f"gateway:/vllm: agents forward interface labels to a model; any "
                  f"other agent type ignores the pack labels.")

    data_dir = Path(__file__).parent.parent / args.data_dir
    output_dir = Path(__file__).parent.parent / args.output

    # Load environments.  A requested family whose pool is absent aborts the
    # run here (unless --allow-missing-family) instead of quietly evaluating a
    # smaller suite and exiting 0.
    try:
        configs = load_environments(data_dir, args.family, args.split,
                                    allow_missing=args.allow_missing_family)
    except MissingFamilyPoolError as exc:
        parser.error(str(exc))
    if not configs:
        parser.error(f"no environments found in {data_dir} "
                     f"(family={args.family!r}, split={args.split!r})")

    if args.n_envs > 0:
        configs = configs[:args.n_envs]

    # Output path — sanitize agent name for filesystem (Windows: no colons)
    agent_safe = args.agent.replace(":", "_").replace("/", "_").replace("+", "_")
    fam_str = f"family{args.family}" if args.family != "all" else "all"
    output_file = output_dir / agent_safe / f"{fam_str}_{args.split}.jsonl"

    # Resume: skip completed
    completed = set()
    if args.resume and output_file.exists():
        completed = get_completed_env_ids(output_file)
        print(f"  Resuming: {len(completed)} already completed")

    configs = [c for c in configs if c.env_id not in completed]

    # Resolve vLLM base URL
    vllm_base_url = args.vllm_base_url
    if vllm_base_url is None and args.vllm_host:
        port = args.vllm_port or 8000
        vllm_base_url = f"http://{args.vllm_host}:{port}/v1"

    print(f"{'='*60}")
    print(f"  AlienBody Evaluation")
    print(f"  Agent: {args.agent} | Modality: {args.modality} | Prompt: {args.prompt}")
    print(f"  Family: {args.family} | Split: {args.split} | Environments: {len(configs)}")
    if args.tool:
        print(f"  Tool: next_state simulator oracle (A1)")
    if args.code_tool:
        print(f"  Tool: Python REPL over next_state (code-tool)")
    if args.label_pack_suite:
        print(f"  Labels: {args.label_pack_condition} "
              f"(audited pack suite {args.label_pack_suite})")
    print(f"  Output: {output_file}")
    if vllm_base_url:
        print(f"  vLLM: {vllm_base_url}")
    print(f"{'='*60}")

    # E-C2: read every env's audited label pack from disk up front.  A missing
    # or mismatched pack aborts the run here -- before any episode -- instead
    # of silently falling back to recomputed labels.
    pack_labels: dict[str, LabelPack] | None = None
    if args.label_pack_suite:
        pack_labels = {}
        for config in configs:
            try:
                pack_labels[config.env_id] = load_label_pack(
                    args.label_pack_suite, args.label_pack_condition, config.env_id)
            except LabelPackError as exc:
                print(f"  ERROR: {exc}")
                sys.exit(2)
        n_labelled = sum(1 for pack in pack_labels.values() if pack.labels is not None)
        print(f"  Label packs: {len(pack_labels)} loaded from disk "
              f"({n_labelled} labelled, {len(pack_labels) - n_labelled} anonymous), "
              f"sha256-checked against {args.label_pack_suite}/manifest.json")

    # Name-prior ablation: per-config button labels
    def _labels_for(config: EnvConfig) -> list[str] | None:
        if pack_labels is not None:
            # E-C2: exactly the audited bytes read from disk (never recomputed)
            return pack_labels[config.env_id].labels
        if args.action_labels is None:
            return None
        mapping = list(config.action_mapping)
        if args.action_labels == "true":
            return mapping
        n = len(mapping)
        return [mapping[(i + 1) % n] for i in range(n)]  # shifted: all wrong

    # Run evaluation
    all_metrics = []
    start_time = time.time()

    with TrajectoryWriter(output_file) as writer:
        if args.n_workers > 1 and args.agent not in ("random", "oracle", "systematic"):
            # Parallel execution for API agents
            with ThreadPoolExecutor(max_workers=args.n_workers) as pool:
                futures = {
                    pool.submit(run_single, config, args.agent, args.modality,
                                args.prompt, args.familiar, args.assist_level,
                                vllm_base_url, _labels_for(config), args.tool,
                                args.code_tool, args.max_tokens, args.induce_rounds,
                                args.induce_fallback, args.bon): config
                    for config in configs
                }
                for i, future in enumerate(as_completed(futures)):
                    config = futures[future]
                    try:
                        traj = future.result()
                        if pack_labels is not None:
                            traj["label_pack"] = pack_labels[config.env_id].as_meta()
                        writer.write(traj)
                        all_metrics.append(traj["metrics"])
                        _print_progress(i + 1, len(configs), traj, start_time)
                    except Exception as e:
                        print(f"  ERROR on {config.env_id}: {e}")
        else:
            # Sequential execution
            for i, config in enumerate(configs):
                try:
                    traj = run_single(config, args.agent, args.modality,
                                      args.prompt, args.familiar, args.assist_level,
                                      vllm_base_url, _labels_for(config), args.tool,
                                      args.code_tool, args.max_tokens, args.induce_rounds,
                                      args.induce_fallback, args.bon)
                    if pack_labels is not None:
                        traj["label_pack"] = pack_labels[config.env_id].as_meta()
                    writer.write(traj)
                    all_metrics.append(traj["metrics"])
                    _print_progress(i + 1, len(configs), traj, start_time)
                except Exception as e:
                    print(f"  ERROR on {config.env_id}: {e}")

    # Summary
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")

    if all_metrics:
        agg = aggregate_metrics(all_metrics)
        print(f"  Episodes: {agg['n_episodes']} | Success: {agg['n_success']}")
        print(f"  CE:  {agg['ce_mean']:.3f}")
        print(f"  SR:  {agg['sr_pct']:.1f}%")
        print(f"  EC:  {agg['ec_mean']:.2f}")
        print(f"  CA:  {agg['ca_mean']:.2f}")
        print(f"  P1:  {agg['p1_mean']:.1f} | P2: {agg['p2_mean']:.1f}")
        print(f"  Time: {elapsed:.1f}s ({elapsed/len(all_metrics):.1f}s/episode)")

        # Save summary
        summary_path = output_file.with_suffix(".summary.json")
        with open(summary_path, "w") as f:
            json.dump({"args": vars(args), "metrics": agg, "elapsed": elapsed}, f, indent=2)
        print(f"  Summary: {summary_path}")
    else:
        print("  No results collected.")


def _print_progress(done: int, total: int, traj: dict, start_time: float):
    """Print progress line."""
    elapsed = time.time() - start_time
    eta = (elapsed / done) * (total - done) if done > 0 else 0
    status = "OK" if traj["success"] else "FAIL"
    ce = traj["metrics"]["ce"]
    print(f"  [{done}/{total}] {traj['env_id']} → {status} CE={ce:.3f} "
          f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s)", end="\r")
    if done == total:
        print()


if __name__ == "__main__":
    main()

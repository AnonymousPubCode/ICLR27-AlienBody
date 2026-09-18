"""Text-modality FMB agent with trained LoRA induction.

Uses a text LLM (Qwen3.5) + LoRA to induce ForwardModel from
structured text observations with F4 relational context.

Minimal wrapper around FMBAgent: replaces heuristic _induce_forward_model
with text-model inference, then uses the same BFS planner.
"""
from __future__ import annotations

import json
import re
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from alienbody.agents.fmb_agent import FMBAgent
from alienbody.schema import ForwardModel, ActionSchema, EffectType, ForwardModelSimulator

# ── Prompt (matches training format) ─────────────────────────────

SYSTEM_PROMPT = """You are an expert at reverse-engineering action systems from observations.
Given structured observations from an unknown action space, predict the action's effect schema.
Output a JSON object with the effect type and parameters.

Effect types:
- translate: {"effect_type": "translate", "params": {"direction": "up"|"down"|"left"|"right"|"forward"|"backward"}}
- rotate: {"effect_type": "rotate", "params": {"rotation": "cw"|"ccw"}}
- state_change: {"effect_type": "state_change", "params": {"change": "inc"|"dec"|"color_move"|"color_interact"}}
- relational: {"effect_type": "relational", "params": {"relation": "nearest_diff"|"nearest_same"|"toward_brightest"|"flee_same"}}
- composite: {"effect_type": "composite", "params": {"primitives": [...]}}
- temporal: {"effect_type": "temporal", "params": {"same_dir": 0, "diff_dir": 1}}
- noop: {"effect_type": "noop", "params": {}}

Output ONLY the JSON object, no other text."""


def build_text_induction_prompt(observations: dict[int, list[dict]], f4_contexts: dict[int, list[dict]]) -> str:
    """Build text prompt for schema induction from observations with F4 context."""
    lines = ["Here are the observations from Phase 1 exploration:\n"]

    for action in sorted(observations.keys()):
        records = observations[action]
        ctxs = f4_contexts.get(action, [{}] * len(records))
        lines.append(f"Action {action}: tested {len(records)} time(s)")

        for i, rec in enumerate(records):
            ctx = ctxs[i] if i < len(ctxs) else {}
            parts = [f"Position: ({rec['prev_pos'][0]}, {rec['prev_pos'][1]})"]

            if ctx:
                cell_color = ctx.get("cell_color", -1)
                color_name = ctx.get("cell_color_name", f"c{cell_color}")
                parts.append(f"Cell: {color_name}({cell_color})")
                if "nearest_diff_pos" in ctx:
                    nd = ctx["nearest_diff_pos"]
                    parts.append(f"NearestDiff: ({nd[0]},{nd[1]}) c={ctx.get('nearest_diff_color','?')}")
                if "nearest_same_pos" in ctx:
                    ns = ctx["nearest_same_pos"]
                    parts.append(f"NearestSame: ({ns[0]},{ns[1]}) c={ctx.get('nearest_same_color','?')}")
                if "brightest_pos" in ctx:
                    bp = ctx["brightest_pos"]
                    parts.append(f"Brightest: ({bp[0]},{bp[1]}) c={ctx.get('brightest_color','?')}")
                if "flee_step" in ctx:
                    parts.append(f"FleeStep: ({ctx['flee_step'][0]},{ctx['flee_step'][1]})")

            dr, dc = rec.get("dr", 0), rec.get("dc", 0)
            dist = abs(dr) + abs(dc)
            if dist > 2:
                move = f"TELEPORT ({dr:+d},{dc:+d})"
            elif dist > 0:
                dirs = {(-1,0):"up",(1,0):"down",(0,-1):"left",(0,1):"right"}
                move = f"moved {dirs.get((dr,dc), f'({dr:+d},{dc:+d})')}"
            else:
                move = "no move"
            parts.append(f"Effect: {move}")

            ddir = rec.get("ddir", 0)
            if ddir != 0:
                parts.append("rotated " + ("cw" if ddir in (1,-3) else "ccw"))
            dcolor = rec.get("dcolor", 0)
            if dcolor != 0:
                parts.append(f"color {dcolor:+d}")

            lines.append(f"  {'; '.join(parts)}")
        lines.append("")

    lines.append("Based on these observations, infer the schema for all actions.")
    lines.append("Output ONLY a JSON object mapping action indices to schemas.")
    return "\n".join(lines)


def parse_schema_response(response: str) -> ForwardModel:
    """Parse text model response into ForwardModel."""
    json_match = re.search(r'\{[^{}]*"effect_type"[^{}]*\}', response, re.DOTALL)
    if not json_match:
        json_match = re.search(r'\{.*\}', response, re.DOTALL)

    raw = {}
    if json_match:
        try:
            raw = json.loads(json_match.group(0))
            if not any(isinstance(v, dict) and "effect_type" in v for v in raw.values()):
                raw = {"0": raw}
        except json.JSONDecodeError:
            pass

    model = ForwardModel()
    for key, value in raw.items():
        try:
            action_id = int(key)
            effect_type = EffectType(value.get("effect_type", "noop"))
            params = value.get("params", {})
            model.actions[action_id] = ActionSchema(effect_type=effect_type, params=params)
        except (ValueError, KeyError):
            continue
    return model


def _extract_schema_dict(response: str) -> dict | None:
    """Robustly extract the schema JSON from a model response.

    Handles CoT-style ``<schema>{...}</schema>`` wrappers, bare JSON, and
    schemas whose ``params`` contain nested braces (which the simple
    ``{...}`` regex cannot span). Returns None when no valid dict with an
    ``effect_type`` key is found.
    """
    candidates = []
    # 1) CoT tag format: <schema>...</schema> (greedy: params may nest braces)
    m = re.search(r"<schema>\s*(\{.*\})\s*</schema>", response, re.DOTALL)
    if m:
        candidates.append(m.group(1))
    # 2) Bare JSON objects: try decoding at every '{' position
    for mm in re.finditer(r"\{", response):
        try:
            obj, _ = json.JSONDecoder().raw_decode(response[mm.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "effect_type" in obj:
            candidates.append(json.dumps(obj))
    for cand in candidates:
        try:
            d = json.loads(cand)
            if isinstance(d, dict) and "effect_type" in d:
                return d
        except json.JSONDecodeError:
            continue
    return None


def _format_cot_obs(rec: dict) -> str:
    """Build the CoT-training-style user observation text.

    Mirrors ``prep_solidify_experiments.format_cot_example`` observation
    lines exactly: no trailing instruction (instructions live in the CoT
    system prompt), f4 context lines only when present.
    """
    ps = rec["prev_state"]
    eff = rec["effect"]
    f4 = rec.get("f4_context", {})
    lines = [f"Action {rec['action']} at ({ps['pos'][0]},{ps['pos'][1]}) facing {['N', 'E', 'S', 'W'][ps['dir']]}"]
    lines.append(f"Effect: dr={eff['dr']}, dc={eff['dc']}, ddir={eff['ddir']}")
    if f4:
        lines.append(f"Cell color: c{f4.get('cell_color', '?')}")
        if f4.get("nearest_diff_pos"):
            lines.append(f"Nearest diff: ({f4['nearest_diff_pos'][0]},{f4['nearest_diff_pos'][1]}) c{f4.get('nearest_diff_color', '?')}")
        if f4.get("nearest_same_pos"):
            lines.append(f"Nearest same: ({f4['nearest_same_pos'][0]},{f4['nearest_same_pos'][1]}) c{f4.get('nearest_same_color', '?')}")
        if f4.get("brightest_pos"):
            lines.append(f"Brightest: ({f4['brightest_pos'][0]},{f4['brightest_pos'][1]}) c{f4.get('brightest_color', '?')}")
    return "\n".join(lines)


def _format_fullft_obs(rec: dict) -> str:
    """Build the fullft-training-style user observation text.

    Mirrors ``train_fmb_fullft.format_example`` but WITHOUT the
    ``Schema: {gt}`` line (that line is training-label leakage and does
    not exist at eval time).
    """
    ps = rec["prev_state"]
    eff = rec["effect"]
    f4 = rec.get("f4_context", {})
    parts = [f"Action {rec['action']} was executed.",
             f"Position: ({ps['pos'][0]}, {ps['pos'][1]})",
             f"Facing: {['N', 'E', 'S', 'W'][ps['dir']]}"]
    if f4:
        parts.append(f"Cell color: c{f4.get('cell_color', '?')}")
        if f4.get("nearest_diff_pos"):
            parts.append(f"Nearest diff: ({f4['nearest_diff_pos'][0]},{f4['nearest_diff_pos'][1]}) c{f4.get('nearest_diff_color', '?')}")
        if f4.get("nearest_same_pos"):
            parts.append(f"Nearest same: ({f4['nearest_same_pos'][0]},{f4['nearest_same_pos'][1]}) c{f4.get('nearest_same_color', '?')}")
        if f4.get("brightest_pos"):
            parts.append(f"Brightest: ({f4['brightest_pos'][0]},{f4['brightest_pos'][1]}) c{f4.get('brightest_color', '?')}")
    parts.append(f"Effect: dr={eff['dr']}, dc={eff['dc']}, ddir={eff['ddir']}")
    return "\n".join(parts)


# ── Text FMB Agent ───────────────────────────────────────────────

class TextFMBAgent(FMBAgent):
    """FMB with text-model schema induction (trained LoRA or full model)."""

    PROMPT_FORMATS = ("stage1", "cot", "fullft", "multiobs")

    def __init__(self, config, model_path: str, lora_path: str, cot: bool = False,
                 prompt_format: str | None = None):
        super().__init__(config)
        self._model_path = model_path
        self._lora_path = lora_path
        # cot=True is the legacy flag for prompt_format="cot"
        self._prompt_format = prompt_format or ("cot" if cot else "stage1")
        if self._prompt_format not in self.PROMPT_FORMATS:
            raise ValueError(f"prompt_format must be one of {self.PROMPT_FORMATS}")
        self._model = None
        self._tokenizer = None
        self._f4_contexts: dict[int, list[dict]] = {}

    @property
    def name(self) -> str:
        return f"TextFMB({self._lora_path.split('/')[-1]})"

    def reset(self):
        super().reset()
        self._f4_contexts = {}

    def _lazy_load_model(self):
        if self._model is not None:
            return
        import os
        print(f"Loading text model from {self._model_path}...")
        base = AutoModelForCausalLM.from_pretrained(
            self._model_path, torch_dtype=torch.bfloat16,
            trust_remote_code=True, device_map="auto")
        # Full fine-tuned checkpoints have no adapter_config.json
        if os.path.exists(os.path.join(self._lora_path, "adapter_config.json")):
            self._model = PeftModel.from_pretrained(base, self._lora_path)
        else:
            self._model = base
        self._model.eval()
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_path, trust_remote_code=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    def record_step(self, action: int, prev_pos: tuple, new_pos: tuple,
                    prev_dir: int = 0, new_dir: int = 0,
                    prev_color: int = 0, new_color: int = 0,
                    f4_ctx: dict | None = None):
        super().record_step(action, prev_pos, new_pos, prev_dir, new_dir, prev_color, new_color)
        # Compute F4 context from config.cell_colors if not provided
        if f4_ctx is None:
            f4_ctx = self._compute_f4_ctx(prev_pos)
        if f4_ctx:
            if action not in self._f4_contexts:
                self._f4_contexts[action] = []
            self._f4_contexts[action].append(f4_ctx)

    def _compute_f4_ctx(self, pos: tuple) -> dict:
        """Compute F4 context at a given position."""
        try:
            from alienbody.env.actions import _find_nearest_by_color
            from alienbody.env.renderer import COLOR_NAMES
            from alienbody.env.grid import DIRECTION_DELTAS, Position

            pr, pc = pos
            gs = self.config.grid_size
            if not (0 <= pr < gs and 0 <= pc < gs):
                return {}

            agent_cell_color = self.config.cell_colors[pr][pc]
            ctx = {"cell_color": agent_cell_color, "cell_color_name": COLOR_NAMES.get(agent_cell_color, f"c{agent_cell_color}")}

            nd = _find_nearest_by_color(Position(pr, pc), self.config, same=False, ref_color=agent_cell_color)
            if nd:
                ctx["nearest_diff_pos"] = [nd.row, nd.col]
                ctx["nearest_diff_color"] = self.config.cell_colors[nd.row][nd.col]

            ns = _find_nearest_by_color(Position(pr, pc), self.config, same=True, ref_color=agent_cell_color)
            if ns:
                ctx["nearest_same_pos"] = [ns.row, ns.col]
                ctx["nearest_same_color"] = self.config.cell_colors[ns.row][ns.col]

            obstacles = set()
            if hasattr(self.config, 'obstacles') and self.config.obstacles:
                obstacles = {(r, c) for r, c in self.config.obstacles}
            best_pos, best_color = None, -1
            for d_idx in range(4):
                delta = DIRECTION_DELTAS[d_idx]
                nb_r, nb_c = pr + delta[0], pc + delta[1]
                if 0 <= nb_r < gs and 0 <= nb_c < gs and (nb_r, nb_c) not in obstacles:
                    nc = self.config.cell_colors[nb_r][nb_c]
                    if nc > best_color:
                        best_color = nc
                        best_pos = (nb_r, nb_c)
            if best_pos:
                ctx["brightest_pos"] = list(best_pos)
                ctx["brightest_color"] = best_color

            if ns:
                flee_dr, flee_dc = pr - ns.row, pc - ns.col
                step = (1 if flee_dr > 0 else -1, 0) if abs(flee_dr) >= abs(flee_dc) else (0, 1 if flee_dc > 0 else -1)
                ctx["flee_step"] = list(step)

            return ctx
        except Exception:
            return {}

    def _induce_forward_model(self) -> ForwardModel:
        """Use trained text model for per-action induction (matching training format)."""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
        fmt_p = self._prompt_format
        if fmt_p == "cot":
            from scripts.prep_solidify_experiments import SYSTEM_PROMPT_COT as SYS
            max_new_tokens = 512
        elif fmt_p == "fullft":
            from scripts.train_fmb_fullft import SYSTEM_PROMPT as SYS
            max_new_tokens = 128
        elif fmt_p == "multiobs":
            from scripts.train_fmb_multiobs import (
                SYSTEM_PROMPT_MULTIOBS as SYS, build_multiobs_prompt)
            max_new_tokens = 256
        else:
            from scripts.train_fmb_stage1_text import format_example_text as fmt, SYSTEM_PROMPT as SYS
            max_new_tokens = 128

        self._lazy_load_model()

        model = ForwardModel()
        for action in range(self.n_actions):
            records = self._observations.get(action, [])
            if not records:
                model.actions[action] = ActionSchema(EffectType.NOOP)
                continue

            ctxs = self._f4_contexts.get(action, [])
            rec = records[-1]
            ctx = ctxs[-1] if ctxs and len(ctxs) >= len(records) else {}

            try:
                train_rec = {
                    "action": action,
                    "prev_state": {"pos": list(rec["prev_pos"]), "dir": rec.get("prev_dir", 0), "color": rec.get("prev_color", 0)},
                    "effect": {"dr": rec.get("dr", 0), "dc": rec.get("dc", 0), "ddir": rec.get("ddir", 0), "dcolor": rec.get("dcolor", 0)},
                    "f4_context": ctx,
                }
                if fmt_p in ("cot", "fullft"):
                    # Match training format exactly: raw im_start text with the
                    # training system prompt (trainers tokenized the raw string).
                    if fmt_p == "cot":
                        obs_text = _format_cot_obs(train_rec)
                    else:
                        obs_text = f"Observe:\n\n{_format_fullft_obs(train_rec)}"
                    text = (f"<|im_start|>system\n{SYS}<|im_end|>\n"
                            f"<|im_start|>user\n{obs_text}<|im_end|>\n"
                            f"<|im_start|>assistant\n")
                    inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
                else:
                    user_prompt = fmt(train_rec)
                    messages = [
                        {"role": "system", "content": SYS},
                        {"role": "user", "content": user_prompt},
                    ]
                    text = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)

                with torch.no_grad():
                    outputs = self._model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.1, do_sample=True, pad_token_id=self._tokenizer.eos_token_id)
                response = self._tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

                schema_dict = _extract_schema_dict(response)
                if schema_dict is not None:
                    try:
                        model.actions[action] = ActionSchema(
                            effect_type=EffectType(schema_dict.get("effect_type", "noop")),
                            params=schema_dict.get("params", {}))
                    except ValueError:
                        # e.g. "unknown": the model honestly cannot determine
                        # the schema (F4 relations are ambiguous from a single
                        # observation). Leave the action absent so the planner
                        # falls back instead of treating it as a known noop.
                        pass
                else:
                    model.actions[action] = ActionSchema(EffectType.NOOP)
            except Exception:
                model.actions[action] = ActionSchema(EffectType.NOOP)

        return model

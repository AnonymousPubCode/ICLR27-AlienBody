"""LLM/VLM Agent — unified wrapper for OpenAI, Anthropic, and Google APIs.

Supports both image (VLM) and text (LLM) modalities.
Handles conversation history, rate limiting, and response parsing.
"""
from __future__ import annotations

import base64
import io
import os
import time
from abc import ABC, abstractmethod
from typing import Optional

from alienbody.agents import Agent, DONE_EXPLORING
from alienbody.prompts import (
    PromptVariant, build_system_prompt, build_turn_prompt, parse_action_response,
    build_phase1_summary, _PHASE2_SUMMARY_HEADER,
)
from alienbody.env.renderer import image_to_png_bytes


class ModelClient(ABC):
    """Abstract API client for LLM/VLM providers."""

    @abstractmethod
    def complete(self, messages: list[dict], **kwargs) -> str:
        """Send messages and return model response text."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


class OpenAIClient(ModelClient):
    """OpenAI API client (GPT-4o, GPT-4o-mini, etc.)."""

    def __init__(self, model: str = "gpt-4o", api_key: str | None = None,
                 max_retries: int = 3, temperature: float = 0.0,
                 base_url: str | None = None):
        import openai
        self.model = model
        self._client = openai.OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"),
            base_url=base_url,
        )
        self.max_retries = max_retries
        self.temperature = temperature
        self.total_tokens = 0

    @property
    def name(self) -> str:
        return f"openai/{self.model}"

    def complete(self, messages: list[dict], **kwargs) -> str:
        for attempt in range(self.max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=kwargs.get("max_tokens", 256),
                )
                self.total_tokens += response.usage.total_tokens if response.usage else 0
                return response.choices[0].message.content or ""
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait = 2 ** attempt
                    time.sleep(wait)
                else:
                    raise RuntimeError(f"OpenAI API failed after {self.max_retries} retries: {e}")
        return ""


class AnthropicClient(ModelClient):
    """Anthropic API client (Claude 3.7 Sonnet, etc.)."""

    def __init__(self, model: str = "claude-sonnet-4-20250514", api_key: str | None = None,
                 max_retries: int = 3, temperature: float = 0.0):
        import anthropic
        self.model = model
        self._client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.max_retries = max_retries
        self.temperature = temperature
        self.total_tokens = 0

    @property
    def name(self) -> str:
        return f"anthropic/{self.model}"

    def complete(self, messages: list[dict], **kwargs) -> str:
        # Anthropic uses separate system param; extract from messages
        system_msg = ""
        user_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"] if isinstance(msg["content"], str) else msg["content"][0]["text"]
            else:
                user_messages.append(msg)

        for attempt in range(self.max_retries):
            try:
                response = self._client.messages.create(
                    model=self.model,
                    system=system_msg,
                    messages=user_messages,
                    temperature=self.temperature,
                    max_tokens=kwargs.get("max_tokens", 256),
                )
                self.total_tokens += response.usage.input_tokens + response.usage.output_tokens
                return response.content[0].text if response.content else ""
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise RuntimeError(f"Anthropic API failed: {e}")
        return ""


class GeminiClient(ModelClient):
    """Google Gemini API client."""

    def __init__(self, model: str = "gemini-2.5-pro", api_key: str | None = None,
                 max_retries: int = 3, temperature: float = 0.0):
        import google.generativeai as genai
        genai.configure(api_key=api_key or os.environ.get("GOOGLE_API_KEY"))
        self.model_name = model
        self._model = genai.GenerativeModel(model)
        self.max_retries = max_retries
        self.temperature = temperature
        self.total_tokens = 0

    @property
    def name(self) -> str:
        return f"gemini/{self.model_name}"

    def complete(self, messages: list[dict], **kwargs) -> str:
        # Convert to Gemini format
        contents = []
        for msg in messages:
            role = "user" if msg["role"] in ("user", "system") else "model"
            if isinstance(msg["content"], str):
                contents.append({"role": role, "parts": [msg["content"]]})
            elif isinstance(msg["content"], list):
                parts = []
                for part in msg["content"]:
                    if part["type"] == "text":
                        parts.append(part["text"])
                    elif part["type"] == "image_url":
                        # Decode base64 image
                        import google.generativeai as genai
                        data = part["image_url"]["url"].split(",")[1]
                        parts.append({"mime_type": "image/png", "data": data})
                contents.append({"role": role, "parts": parts})

        for attempt in range(self.max_retries):
            try:
                response = self._model.generate_content(
                    contents,
                    generation_config={"temperature": self.temperature,
                                       "max_output_tokens": kwargs.get("max_tokens", 256)},
                )
                return response.text or ""
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise RuntimeError(f"Gemini API failed: {e}")
        return ""


class VLLMClient(ModelClient):
    """vLLM OpenAI-compatible API client for local models (Qwen, etc.).

    vLLM serves an OpenAI-compatible endpoint. This client wraps it
    with the same interface, supporting both text and image modalities
    for multimodal models (e.g., Qwen3.5-VL).

    Usage:
        client = VLLMClient(model="qwen3.5-4b", base_url="http://localhost:8000/v1")
        response = client.complete([{"role": "user", "content": "..."}])
    """

    # Default ports per model size (can be overridden)
    DEFAULT_PORTS = {
        "qwen3.5-4b": 8000,
        "qwen3.5-9b": 8001,
        "qwen3.5-35b": 8002,
        "qwen3.5-397b": 8003,
        "qwen3.6-27b": 8004,
    }

    def __init__(self, model: str = "qwen3.5-4b",
                 base_url: str | None = None,
                 api_key: str = "EMPTY",
                 max_retries: int = 3, temperature: float = 0.0):
        import openai

        # Resolve base URL
        if base_url is None:
            host = os.environ.get("VLLM_HOST", "localhost")
            port = os.environ.get("VLLM_PORT") or self.DEFAULT_PORTS.get(
                model.lower().replace("_", "."), 8000
            )
            base_url = f"http://{host}:{port}/v1"

        self.model = model
        self._client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.max_retries = max_retries
        self.temperature = temperature
        self.total_tokens = 0

    @property
    def name(self) -> str:
        return f"vllm/{self.model}"

    def complete(self, messages: list[dict], **kwargs) -> str:
        # Convert multimodal content for vLLM (uses OpenAI vision format)
        vllm_messages = []
        for msg in messages:
            content = msg["content"]
            if isinstance(content, list):
                # Already in multimodal format — keep as-is
                vllm_content = content
            else:
                vllm_content = content
            vllm_messages.append({"role": msg["role"], "content": vllm_content})

        for attempt in range(self.max_retries):
            try:
                extra_body = {"skip_special_tokens": True}
                extra_body.update(kwargs.get("extra_body", {}))
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=vllm_messages,
                    temperature=kwargs.get("temperature", self.temperature),
                    max_tokens=kwargs.get("max_tokens", 256),
                    extra_body=extra_body,
                )
                self.total_tokens += response.usage.total_tokens if response.usage else 0
                content = response.choices[0].message.content or ""
                return content
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait = 2 ** attempt
                    time.sleep(wait)
                else:
                    raise RuntimeError(
                        f"vLLM API failed after {self.max_retries} retries: {e}"
                    )
        return ""


# ── LLM Agent ──────────────────────────────────────────────────────

class LLMAgent(Agent):
    """Unified LLM/VLM agent that works with any ModelClient.

    Supports image (VLM) and text (LLM) modalities.
    Manages conversation history for multi-turn interaction.
    """

    def __init__(
        self,
        client: ModelClient,
        modality: str = "image",           # "image", "text", or "both"
        prompt_variant: PromptVariant = PromptVariant.MINIMAL,
        familiar: bool = False,
        n_actions: int = 4,
        max_history_turns: int = 10,        # keep last N turns (lower for text, higher for image)
        assist_level: int = 0,             # diagnostic ladder: 0-3
        action_mapping: list[str] | None = None,  # for assist_level=3
        action_labels: list[str] | None = None,   # name-prior ablation
        response_max_tokens: int = 128,    # max output tokens per turn
                                            # (thinking models need >= 600)
        bon_n: int = 1,                    # Best-of-N: sample N candidates per
                                            # step (phase 2) and majority-vote
    ):
        self.client = client
        self.modality = modality
        self.prompt_variant = prompt_variant
        self.familiar = familiar
        self.n_actions = n_actions
        self.max_history_turns = max_history_turns
        self.assist_level = assist_level
        self.action_mapping = action_mapping
        self.response_max_tokens = response_max_tokens
        self.bon_n = bon_n

        self._messages: list[dict] = []
        self._system_prompt = build_system_prompt(
            prompt_variant, n_actions, familiar,
            assist_level=assist_level, action_mapping=action_mapping,
            action_labels=action_labels,
        )
        self._step = 0
        self._reasoning_log: list[str] = []  # store model reasoning for analysis
        self._phase1_history: list[dict] = []  # for summary assist (level 2)
        self._summary_injected = False  # track if summary was already injected

    @property
    def name(self) -> str:
        mod = self.modality[0].upper()  # I/T/B
        return f"{self.client.name}_{mod}_{self.prompt_variant.value}"

    def reset(self):
        self._messages = [{"role": "system", "content": self._system_prompt}]
        self._step = 0
        self._reasoning_log = []
        self._phase1_history = []
        self._summary_injected = False
        if hasattr(self, '_l2_action_counter'):
            del self._l2_action_counter
        if hasattr(self, '_l2_test_round'):
            del self._l2_test_round

    def act(self, observation: dict, info: dict | None = None) -> int:
        self._step += 1
        phase = observation.get("phase", 1)
        feedback = observation.get("feedback")

        # Level 3: mapping given — skip Phase 1, always execute
        if self.assist_level == 3 and phase == 1:
            return DONE_EXPLORING

        # Level 2: systematic multi-round testing in Phase 1
        # Tests each action N times to capture conditional effects
        if self.assist_level == 2 and phase == 1:
            if not hasattr(self, '_l2_action_counter'):
                self._l2_action_counter = 0
                self._l2_test_round = 0
                self._l2_rounds = getattr(self, '_l2_num_rounds', 1)  # default 1
            if info:
                self._phase1_history.append(info.get("last_record", {}))
            if self._l2_test_round < self._l2_rounds:
                action = self._l2_action_counter
                self._l2_action_counter += 1
                if self._l2_action_counter >= self.n_actions:
                    self._l2_action_counter = 0
                    self._l2_test_round += 1
                return action
            return DONE_EXPLORING

        # Build Phase 2 summary for level 2 (inject once at transition)
        phase2_summary = None
        if (self.assist_level == 2 and phase == 2
                and not self._summary_injected and self._phase1_history):
            summary_table = build_phase1_summary(self._phase1_history)
            phase2_summary = _PHASE2_SUMMARY_HEADER.format(summary_table=summary_table)
            self._summary_injected = True

        # Build user message
        turn_text = build_turn_prompt(
            self.prompt_variant, observation, phase, self._step, feedback,
            phase2_summary=phase2_summary,
        )

        # Build message content based on modality
        if self.modality in ("image", "both") and "image" in observation:
            content = self._build_image_message(observation["image"], turn_text)
        else:
            content = turn_text

        self._messages.append({"role": "user", "content": content})

        # Trim history if too long
        self._trim_history()

        # Call LLM — Best-of-N mode samples N candidates (phase 2) and
        # majority-votes on the parsed action; history keeps only the winner.
        if self.bon_n > 1 and phase == 2:
            from collections import Counter
            votes = []
            for _ in range(self.bon_n):
                response = self.client.complete(
                    self._messages,
                    max_tokens=self.response_max_tokens,
                    temperature=0.7,
                )
                if not response or not response.strip():
                    votes.append(0)
                    continue
                parsed = parse_action_response(response, self.n_actions)
                votes.append(0 if parsed == "done" else parsed)
            result = Counter(votes).most_common(1)[0][0]
            response = f"<bon n={self.bon_n} votes={votes} winner={result}>"
        else:
            response = self.client.complete(self._messages, max_tokens=self.response_max_tokens)

            # Guard against empty responses (e.g., thinking models that consume
            # output tokens for internal reasoning). Treat as action 0.
            if not response or not response.strip():
                response = "0"

            # Parse response
            result = parse_action_response(response, self.n_actions)

        self._reasoning_log.append(response)

        # Add assistant response to history
        self._messages.append({"role": "assistant", "content": response})

        if result == "done":
            return DONE_EXPLORING
        return result

    def _build_image_message(self, image, text: str) -> list[dict]:
        """Build multimodal message with image + text."""
        import numpy as np
        if isinstance(image, np.ndarray):
            png_bytes = image_to_png_bytes(image)
        else:
            png_bytes = image
        b64 = base64.b64encode(png_bytes).decode("utf-8")

        return [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": text},
        ]

    def _trim_history(self):
        """Keep system prompt + last N turns to prevent context overflow."""
        if len(self._messages) <= self.max_history_turns * 2 + 1:
            return
        # Keep system message + last N user/assistant pairs
        system = self._messages[0]
        recent = self._messages[-(self.max_history_turns * 2):]
        self._messages = [system] + recent

    def get_reasoning_log(self) -> list[str]:
        """Return all model responses for post-hoc analysis."""
        return list(self._reasoning_log)


# ── Convenience constructors ───────────────────────────────────────

def make_agent(
    model: str,
    modality: str = "image",
    prompt_variant: str = "minimal",
    familiar: bool = False,
    assist_level: int = 0,
    action_mapping: list[str] | None = None,
    bon_n: int = 1,
    **kwargs,
) -> LLMAgent:
    """Create an LLMAgent from a model name string.

    Args:
        model: Model name (e.g., "gpt-4o", "claude-sonnet-4-20250514",
               "vllm:qwen3.5-4b").
        modality: "image", "text", or "both".
        prompt_variant: "minimal", "cot", "explore_first", "expert_demo".
        familiar: If True, reveal that actions are scrambled directions.
        assist_level: Diagnostic ladder level (0-4).
            0 = normal, 1 = familiar, 2 = summary assist,
            3 = mapping given (requires action_mapping), 4 = oracle (use OracleAgent).
        action_mapping: Ground truth for assist_level=3 (e.g., ["left","up","right","down"]).
    """
    variant = PromptVariant(prompt_variant)

    # vLLM/local models: prefix with "vllm:"
    if model.lower().startswith("vllm:"):
        vllm_model = model[5:]
        client = VLLMClient(model=vllm_model, **kwargs)
        return LLMAgent(
            client=client, modality=modality, prompt_variant=variant,
            familiar=familiar, assist_level=assist_level, action_mapping=action_mapping,
            bon_n=bon_n,
        )

    if "gpt" in model.lower() or "o1" in model.lower() or "o3" in model.lower():
        client = OpenAIClient(model=model, **kwargs)
    elif "claude" in model.lower():
        client = AnthropicClient(model=model, **kwargs)
    elif "gemini" in model.lower():
        client = GeminiClient(model=model, **kwargs)
    else:
        # Try gateway API as fallback for all other models
        try:
            from alienbody.agents.gateway_client import GatewayClient
            client = GatewayClient(model=model, **kwargs)
        except (ImportError, ValueError):
            raise ValueError(f"Unknown model: {model}. Use 'gpt-*', 'claude-*', 'gemini-*', 'vllm:*', or a gateway model name.")

    return LLMAgent(
        client=client, modality=modality, prompt_variant=variant,
        familiar=familiar, assist_level=assist_level, action_mapping=action_mapping,
        bon_n=bon_n,
    )

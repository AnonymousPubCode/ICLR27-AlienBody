"""Optional gateway API client for AlienBody LLM experiments.

Synchronous wrapper around a configurable OpenAI-compatible / custom chat
proxy. Endpoints and credentials come from environment variables (see
``.env.example``) or an optional gitignored ``fuxi_secrets.local.py``.

Usage:
    from alienbody.agents.fuxi_client import FuxiClient
    client = FuxiClient(model="gpt-4.1")
    response = client.complete([{"role": "user", "content": "Hello"}])
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import string
import time
from pathlib import Path
from typing import Optional

import requests

from alienbody.agents.llm_agent import ModelClient

# ── Model → API version mapping ──────────────────────────────────

FUXI_MODELS = {
    "gpt-4o": {"api_version": "v2", "max_tokens": 16384},
    "gpt-4.1": {"api_version": "v2", "max_tokens": 16384},
    "gpt-5.1": {"api_version": "v1", "max_tokens": 10000},
    "claude-opus-4@20250514": {"api_version": "v2", "max_tokens": 32000},
    "gemini-2.5-pro": {"api_version": "v3", "max_tokens": 65535},
    "gemini-2.5-flash": {"api_version": "v3", "max_tokens": 65535},
    "gemini-3-pro-preview": {"api_version": "v1", "max_tokens": 65536},
    "gemini-3-flash-preview": {"api_version": "v1", "max_tokens": 65536},
    "deepseek-v4-pro": {"api_version": "v1", "max_tokens": 16000},
    "deepseek-v4-flash": {"api_version": "v1", "max_tokens": 16000},
    # DeepSeek via Leihuo (free!) — non-thinking and thinking variants
    "dsv4-lh": {"api_version": "leihuo", "max_tokens": 16000, "model_name": "deepseek-v4-pro"},
    "dsv4-lh-think": {"api_version": "leihuo", "max_tokens": 16000, "thinking_budget": 500, "model_name": "deepseek-v4-pro"},
    "dsv4-flash-lh": {"api_version": "leihuo", "max_tokens": 16000, "model_name": "deepseek-v4-flash"},
    "dsv4-flash-lh-think": {"api_version": "leihuo", "max_tokens": 16000, "thinking_budget": 500, "model_name": "deepseek-v4-flash"},
    # Gemini via Leihuo (OpenAI-compatible): must NOT send the `thinking`
    # field — disabled/enabled both break it (empty content). thinking_budget
    # = False marks "omit the field entirely" (three-state: None/0 = force
    # disabled, >0 = enabled with budget, False = omit).
    "gemini-3.1-pro-preview": {"api_version": "leihuo", "max_tokens": 65536,
                               "thinking_budget": False, "model_name": "gemini-3.1-pro-preview"},
    "gemini-3.7-flash": {"api_version": "leihuo", "max_tokens": 65536,
                         "thinking_budget": False, "model_name": "gemini-3.7-flash"},
    "gemini-3.1-flash-lite": {"api_version": "leihuo", "max_tokens": 65536,
                              "thinking_budget": False, "model_name": "gemini-3.1-flash-lite"},
    "kimi-k3": {"api_version": "leihuo", "max_tokens": 65536},
    "kimi-k2.5": {"api_version": "leihuo", "max_tokens": 65536},
    "glm-5.2": {"api_version": "leihuo", "max_tokens": 65536},
    "glm-5": {"api_version": "leihuo", "max_tokens": 65536},
    "deepseek-r1": {"api_version": "v2", "max_tokens": 16000},
    "deepseek-v3.1-250821": {"api_version": "v2", "max_tokens": 16000},
    "qwen-max": {"api_version": "v2", "max_tokens": 8192},
    # Claude via Leihuo (2026-09-07, from /v1/models): ONLY the Anthropic
    # /v1/messages endpoint accepts these — /v1/chat/completions returns
    # 403 "请勿使用端点". Route through api_version "leihuo-claude".
    "claude-opus-4-8-v4-pro": {"api_version": "leihuo-claude", "max_tokens": 65536,
                               "model_name": "claude-opus-4-8-v4-pro"},
    "claude-opus-4-8-v4-flash": {"api_version": "leihuo-claude", "max_tokens": 65536,
                                 "model_name": "claude-opus-4-8-v4-flash"},
    "claude-opus-4-6-v4-pro": {"api_version": "leihuo-claude", "max_tokens": 65536,
                               "model_name": "claude-opus-4-6-v4-pro"},
}

# ── API credentials (env vars / optional local secrets file) ─────
# Never hard-code keys in this file. For local use, either:
#   1) export FUXI_* / LEIHUO_* (see .env.example), or
#   2) place fuxi_secrets.local.py next to this module (gitignored).


def _load_api_configs() -> dict:
    """Build API_CONFIGS from environment, then overlay local secrets if present."""
    configs = {
        "v1": {
            "app_id": os.environ.get("FUXI_V1_APP_ID", ""),
            "app_key": os.environ.get("FUXI_V1_APP_KEY", ""),
            "project_id": os.environ.get("FUXI_V1_PROJECT_ID", ""),
            "bearer_app_key": os.environ.get("FUXI_V1_BEARER", ""),
            "end_point": os.environ.get(
                "FUXI_V1_ENDPOINT",
                "",
            ),
        },
        "v2": {
            "app_id": os.environ.get("FUXI_V2_APP_ID", ""),
            "app_key": os.environ.get("FUXI_V2_APP_KEY", ""),
            "project_id": os.environ.get("FUXI_V2_PROJECT_ID", ""),
            "end_point": os.environ.get(
                "FUXI_V2_ENDPOINT",
                "",
            ),
        },
        "v3": {
            "app_id": os.environ.get("FUXI_V3_APP_ID", ""),
            "app_key": os.environ.get("FUXI_V3_APP_KEY", ""),
            "project_id": os.environ.get("FUXI_V3_PROJECT_ID", ""),
            "end_point": os.environ.get(
                "FUXI_V3_ENDPOINT",
                "",
            ),
        },
        "leihuo": {
            "bearer_app_key": os.environ.get("LEIHUO_API_KEY", ""),
            "end_point": os.environ.get(
                "LEIHUO_ENDPOINT",
                "",
            ),
        },
        "leihuo-claude": {
            "bearer_app_key": os.environ.get(
                "LEIHUO_CLAUDE_API_KEY",
                os.environ.get("LEIHUO_API_KEY", ""),
            ),
            "end_point": os.environ.get(
                "LEIHUO_CLAUDE_ENDPOINT",
                "",
            ),
        },
    }
    local = Path(__file__).with_name("fuxi_secrets.local.py")
    if local.exists():
        ns: dict = {}
        exec(compile(local.read_text(encoding="utf-8"), str(local), "exec"), ns)
        overlay = ns.get("API_CONFIGS", {})
        for ver, cfg in overlay.items():
            configs.setdefault(ver, {}).update(
                {k: v for k, v in cfg.items() if v}
            )
    return configs


API_CONFIGS = _load_api_configs()


def _md5_sign(app_id: str, app_key: str) -> tuple[dict, str, str]:
    """Generate MD5 signature headers for v2/v3."""
    nonce = "".join(random.choices(string.ascii_letters + string.digits, k=10))
    timestamp = str(int(time.time()))
    raw = f"appId={app_id}&nonce={nonce}&timestamp={timestamp}&appkey={app_key}"
    sign = hashlib.md5(raw.encode()).hexdigest().upper()
    return nonce, timestamp, sign


class FuxiClient(ModelClient):
    """Synchronous Fuxi API client implementing the ModelClient interface."""

    def __init__(
        self,
        model: str = "gpt-4.1",
        temperature: float = 0.0,
        max_retries: int = 3,
    ):
        if model not in FUXI_MODELS:
            raise ValueError(
                f"Unknown model: {model}. Available: {list(FUXI_MODELS.keys())}"
            )
        self.model = model
        self._config = FUXI_MODELS[model]
        self._api_version = self._config["api_version"]
        self._api = API_CONFIGS[self._api_version]
        self._model_name = self._config.get("model_name", model)  # override for API call
        self.temperature = temperature
        self.max_retries = max_retries
        self.total_tokens = 0
        self._thinking_budget = self._config.get("thinking_budget", 0)

        # kimi models require temperature=1.0; thinking models also use T=1
        if model.startswith("kimi") or self._thinking_budget:
            self.temperature = 1.0

    @property
    def name(self) -> str:
        return f"fuxi/{self.model}"

    def complete(self, messages: list[dict], **kwargs) -> str:
        """Send messages and return model response text."""
        max_tokens = kwargs.get("max_tokens", 512)

        for attempt in range(self.max_retries):
            try:
                headers = self._build_headers()
                body = self._build_body(messages, max_tokens)
                endpoint = self._api["end_point"]

                resp = requests.post(
                    endpoint, headers=headers, json=body, timeout=60
                )

                if resp.status_code != 200:
                    raise RuntimeError(
                        f"HTTP {resp.status_code}: {resp.text[:200]}"
                    )

                data = resp.json()
                text, tokens = self._parse_response(data)
                self.total_tokens += tokens
                return text

            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise RuntimeError(
                        f"Fuxi API failed after {self.max_retries} retries: {e}"
                    )
        return ""

    def _build_headers(self) -> dict:
        if self._api_version in ("leihuo", "leihuo-claude"):
            headers = {
                "Authorization": f"Bearer {self._api['bearer_app_key']}",
                "Content-Type": "application/json",
            }
            if self._api_version == "leihuo-claude":
                headers["anthropic-version"] = "2023-06-01"
            return headers
        elif self._api_version == "v1":
            return {
                "Authorization": f"Bearer {self._api['bearer_app_key']}",
                "Content-Type": "application/json",
            }
        else:
            nonce, timestamp, sign = _md5_sign(
                self._api["app_id"], self._api["app_key"]
            )
            return {
                "appId": self._api["app_id"],
                "nonce": nonce,
                "timestamp": timestamp,
                "sign": sign,
                "version": self._api_version,
                "Content-Type": "application/json",
                "projectId": self._api["project_id"],
            }

    def _build_body(self, messages: list[dict], max_tokens: int) -> dict:
        if self._api_version == "leihuo-claude":
            # Anthropic Messages API: system prompt goes to the top-level
            # `system` field (role=system is rejected inside `messages`).
            sys_parts, conv = [], []
            for msg in self._flatten_messages(messages):
                if msg["role"] == "system":
                    c = msg["content"]
                    sys_parts.append(c if isinstance(c, str) else c[0].get("text", ""))
                else:
                    role = "assistant" if msg["role"] == "assistant" else "user"
                    conv.append({"role": role, "content": msg["content"]})
            body = {
                "model": self._model_name,
                "messages": conv,
                "max_tokens": max_tokens,
                "temperature": self.temperature,
            }
            if sys_parts:
                body["system"] = "\n\n".join(sys_parts)
            return body
        elif self._api_version == "leihuo":
            body = {
                "model": self._model_name,
                "messages": self._flatten_messages(messages),
                "max_tokens": max_tokens,
                "temperature": self.temperature,
                "stream": False,
            }
            # Add thinking config for models that support it.
            # Three states: >0 budget = enabled; 0/None = force disabled
            # (DeepSeek V4 defaults to reasoning and eats all max_tokens);
            # False = omit the field entirely (Gemini via Leihuo returns
            # empty content when the field is present in either state).
            if hasattr(self, '_thinking_budget') and self._thinking_budget:
                body["thinking"] = {"type": "enabled", "budget_tokens": self._thinking_budget}
            elif not hasattr(self, '_thinking_budget') or self._thinking_budget is not False:
                body["thinking"] = {"type": "disabled"}
            return body
        elif self._api_version == "v3":
            # Gemini format: convert messages → contents
            contents = []
            for msg in messages:
                role = msg.get("role", "user")
                if role == "assistant":
                    role = "model"
                elif role == "system":
                    role = "user"  # Gemini doesn't have system role
                content = msg["content"] if isinstance(msg["content"], str) else msg["content"][0].get("text", "")
                contents.append({"role": role, "parts": [{"text": content}]})
            return {
                "model": self.model,
                "contents": contents,
                "generationConfig": {"temperature": self.temperature},
            }
        elif self._api_version == "v1":
            body = {
                "model": self._model_name,
                "messages": self._flatten_messages(messages),
                "max_tokens": max_tokens,
                "temperature": self.temperature,
            }
            if "gpt-5.1" not in self.model:
                body["stream"] = False
            return body
        else:  # v2
            return {
                "model": self.model,
                "messages": self._flatten_messages(messages),
                "maxTokens": max_tokens,
                "temperature": self.temperature,
            }

    def _flatten_messages(self, messages: list[dict]) -> list[dict]:
        """Keep multimodal content intact (image_url parts included).

        Previously this stripped image parts from list content, which
        silently degraded every 'image'-modality evaluation to text-only.
        The gateways (v1/v2/leihuo) accept OpenAI-format multimodal
        content; text-only messages (plain strings) pass through unchanged.
        """
        flat = []
        for msg in messages:
            content = msg["content"]
            if isinstance(content, list):
                # Preserve text + image parts for vision-capable models
                flat.append({"role": msg["role"], "content": content})
            else:
                flat.append({"role": msg["role"], "content": content})
        return flat

    def _parse_response(self, data: dict) -> tuple[str, int]:
        """Parse response and return (text, total_tokens)."""
        if self._api_version == "leihuo-claude":
            # Anthropic Messages format: content = [{type: text, text: ...}]
            blocks = data.get("content", [])
            text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
            if not text:
                raise RuntimeError(f"Leihuo-claude no content: {json.dumps(data)[:200]}")
            usage = data.get("usage", {})
            tokens = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
            return text, tokens
        elif self._api_version == "leihuo":
            # Standard OpenAI format
            choices = data.get("choices", [])
            if not choices:
                raise RuntimeError(f"Leihuo no choices: {json.dumps(data)[:200]}")
            text = choices[0].get("message", {}).get("content", "")
            usage = data.get("usage", {})
            tokens = usage.get("total_tokens", 0)
            return text, tokens
        elif self._api_version == "v3":
            # Gemini format
            if "candidates" in data and data["candidates"]:
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                return text, 0  # Gemini v3 doesn't always return token counts
            raise RuntimeError(f"Gemini response error: {json.dumps(data)[:200]}")

        elif self._api_version == "v1":
            # OpenAI-compatible format
            choices = data.get("choices", [])
            if not choices:
                raise RuntimeError(f"v1 no choices: {json.dumps(data)[:200]}")
            text = choices[0].get("message", {}).get("content", "")
            usage = data.get("usage", {})
            tokens = usage.get("total_tokens", 0)
            return text, tokens

        else:  # v2
            detail = data.get("detail")
            if not detail:
                raise RuntimeError(f"v2 no detail: {json.dumps(data)[:200]}")
            choices = detail.get("choices", [])
            if not choices:
                raise RuntimeError(f"v2 no choices: {json.dumps(data)[:200]}")
            text = choices[0]["message"]["content"]
            usage = detail.get("usage", {})
            tokens = usage.get("totalTokens", 0)
            return text, tokens

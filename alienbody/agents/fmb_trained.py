"""Trained LoRA model client — wraps Stage 1 model as ModelClient.

Compatible with VLMFMBAgent and ActiveBabblingAgent.
Loads base Qwen3-VL + LoRA adapter for inference.
"""
from __future__ import annotations

from typing import Optional

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

from alienbody.agents.llm_agent import ModelClient


class TrainedLoraClient(ModelClient):
    """ModelClient wrapping a LoRA-trained Qwen3-VL model for inference."""

    def __init__(
        self,
        base_model_path: str,
        lora_path: str,
        device: str = "cuda:0",
        max_tokens: int = 256,
    ):
        self._base_path = base_model_path
        self._lora_path = lora_path
        self._max_tokens = max_tokens

        print(f"Loading base model from {base_model_path}...")
        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )

        print(f"Loading LoRA adapter from {lora_path}...")
        from peft import PeftModel
        self._model = PeftModel.from_pretrained(self._model, lora_path)
        self._model.eval()

        self._processor = AutoProcessor.from_pretrained(
            base_model_path, trust_remote_code=True,
        )

        self.total_tokens = 0

    @property
    def name(self) -> str:
        return f"FMB-Stage1({self._lora_path.split('/')[-1]})"

    def complete(self, messages: list[dict], **kwargs) -> str:
        """Generate text from messages. Handles text-only prompts."""
        max_tokens = kwargs.get("max_tokens", self._max_tokens)

        # Build text prompt from messages
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, list):
                # Multimodal: extract text
                content = " ".join(
                    p.get("text", "") for p in content if p.get("type") == "text"
                )
            prompt_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        prompt_parts.append("<|im_start|>assistant\n")

        text = "\n".join(prompt_parts)

        # Tokenize
        inputs = self._processor(
            text=[text],
            return_tensors="pt",
            padding=True,
        ).to(self._model.device)

        # Generate
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self._processor.tokenizer.eos_token_id,
            )

        # Decode only the generated part
        input_len = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][input_len:]
        response = self._processor.decode(generated_ids, skip_special_tokens=True)

        self.total_tokens += len(generated_ids)
        return response.strip()


def make_trained_fmb_agent(config, lora_path: str, base_model: str = None):
    """Create a VLMFMBAgent using a trained LoRA model.

    Usage:
        from alienbody.agents.fmb_trained import make_trained_fmb_agent
        agent = make_trained_fmb_agent(config, "models/fmb_stage1/final")
    """
    from alienbody.agents.fmb_vlm import VLMFMBAgent

    if base_model is None:
        # Auto-detect base model from adapter config
        import json as _json, os as _os
        adapter_cfg_path = _os.path.join(lora_path, "adapter_config.json")
        if _os.path.exists(adapter_cfg_path):
            with open(adapter_cfg_path) as _f:
                adapter_cfg = _json.load(_f)
            base_model = adapter_cfg.get("base_model_name_or_path",
                                          "/project/model/Qwen3-VL-4B-Instruct")
        else:
            base_model = "/project/model/Qwen3-VL-4B-Instruct"

    client = TrainedLoraClient(
        base_model_path=base_model,
        lora_path=lora_path,
    )

    return VLMFMBAgent(config, induction_client=client, explore_agent_name="double")

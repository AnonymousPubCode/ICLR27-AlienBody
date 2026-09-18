"""Debug vLLM token count discrepancy."""
import sys
sys.path.insert(0, "/root/alienbody/code")

from alienbody.agents.llm_agent import VLLMClient
from alienbody.prompts import build_system_prompt, build_turn_prompt, PromptVariant
from alienbody.env.core import AlienBodyEnv
from alienbody.env.grid import EnvConfig

# Simulate LLMAgent
sys_prompt = build_system_prompt(PromptVariant.MINIMAL, n_actions=4)
messages = [{"role": "system", "content": sys_prompt}]

config = EnvConfig.from_file("data/envs/family1/test/env_000.json")
env = AlienBodyEnv(config, render_mode="text")
obs, info = env.reset()

turn_text = build_turn_prompt(PromptVariant.MINIMAL, obs, phase=1, step=1)
messages.append({"role": "user", "content": turn_text})

print(f"System prompt: {len(sys_prompt)} chars")
print(f"Turn text: {len(turn_text)} chars")
for i, m in enumerate(messages):
    c = m["content"]
    clen = len(c) if isinstance(c, str) else len(str(c))
    print(f"  [{i}] role={m['role']}, content_len={clen}")

client = VLLMClient(model="/project/model/Qwen3.5-4B", base_url="http://localhost:8000/v1")
try:
    resp = client.complete(messages, max_tokens=10)
    print(f"Response: {resp[:100]}...")
    print(f"Total tokens: {client.total_tokens}")
except Exception as e:
    print(f"Error: {e}")

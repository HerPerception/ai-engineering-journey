import os
import anthropic

client = anthropic.Anthropic()
model = os.environ.get("LLM_MODEL", "claude-sonnet-5")

response = client.messages.create(
    model=model,
    max_tokens=256,
    messages=[{"role": "user", "content": "What is a neural network?"}]
)
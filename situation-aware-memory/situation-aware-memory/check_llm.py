"""Tests your real LLM configuration. Prints the model, endpoint and the raw reply; never prints keys."""
from memcore.llm import OpenAIChat
c = OpenAIChat()
print("model:", c.model, "| endpoint:", c._client.base_url)
out = c.complete_json('Return exactly this JSON and nothing else: {"ok": true}')
print("type:", type(out).__name__)
print("raw reply:", str(out)[:300])

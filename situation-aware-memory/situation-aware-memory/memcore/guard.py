"""Fail-fast wrapper: stops a run when the API is clearly broken instead of scoring it as abstentions."""
from __future__ import annotations

import json

HINT = ("The LLM API is not returning usable JSON replies. Check the key/token, the endpoint (OPENAI_BASE_URL / GITHUB_MODELS_ENDPOINT), "
        "the model name (PIPELINE_MODEL) and your credit or rate limit. Run python check_llm.py to see the raw reply.")


def _usable(out) -> bool:
    if not isinstance(out, str):
        return False
    try:
        d = json.loads(out)
    except json.JSONDecodeError:
        return False
    return isinstance(d, dict) and len(d) > 0


class GuardedLLM:
    def __init__(self, inner, max_bad: int = 3):
        self.inner, self.max_bad, self.bad, self.last = inner, max_bad, 0, ""

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def complete_json(self, prompt: str) -> str:
        try:
            out = self.inner.complete_json(prompt)
        except AttributeError as e:
            raise RuntimeError(f"{HINT} (underlying error: {e})") from e
        if _usable(out):
            self.bad = 0
        else:
            self.bad += 1
            self.last = str(out)[:200]
            if self.bad >= self.max_bad:
                raise RuntimeError(f"{self.bad} unusable replies in a row. Last reply: {self.last!r}. {HINT}")
        return out

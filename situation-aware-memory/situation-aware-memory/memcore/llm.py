"""LLM access behind one tiny interface so everything is testable offline.

`LLMClient.complete_json(prompt)` returns the raw model text (expected JSON).
Use `ScriptedLLM` in tests, `OpenAIChat` for real runs.
"""
from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Callable, Protocol


def _load_env_file() -> None:
    """Load a nearby .env when python-dotenv is not installed."""
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.is_file():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


class LLMClient(Protocol):
    def complete_json(self, prompt: str) -> str: ...


class ScriptedLLM:
    """Test double. `responder` maps a prompt to a raw response string, or pass
    a fixed list of responses to return in order. Records every prompt."""

    def __init__(self, responder: Callable[[str], str] | list[str]):
        self._responder = responder
        self._i = 0
        self.prompts: list[str] = []

    def complete_json(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if callable(self._responder):
            return self._responder(prompt)
        out = self._responder[self._i]
        self._i += 1
        return out


class OpenAIChat:
    """Real backend. Mirrors the paper: temperature 0, JSON mode. The `openai`
    package is imported lazily so the rest of the project needs no install."""

    def __init__(self, model: str | None = None, temperature: float = 0.0):
        from openai import OpenAI  # lazy

        _load_env_file()
        api_key = os.environ.get("GITHUB_TOKEN") or os.environ.get("GITHUB_MODELS_API_KEY")
        base_url = os.environ.get("OPENAI_BASE_URL")
        using_github_models = bool(api_key)
        using_openrouter = False
        if using_github_models:
            base_url = base_url or os.environ.get("GITHUB_MODELS_ENDPOINT",
                                                   "https://models.github.ai/inference")
        else:
            api_key = os.environ.get("OPENROUTER_API_KEY")
        if api_key and not using_github_models:
            base_url = base_url or "https://openrouter.ai/api/v1"
            using_openrouter = True
        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key and api_key.startswith("sk-or-"):
                base_url = base_url or "https://openrouter.ai/api/v1"
                using_openrouter = True

        placeholder_values = {
            "your_new_rotated_key",
            "replace_with_your_rotated_key",
            "replace_with_your_key",
        }
        if not api_key or api_key.strip() in placeholder_values:
            raise RuntimeError(
                "No valid API key was configured. Put a real key in .env: "
                "GITHUB_TOKEN=github_pat_... or GITHUB_MODELS_API_KEY=... or "
                "OPENROUTER_API_KEY=sk-or-v1-... or "
                "OPENAI_API_KEY=sk-..."
            )

        self._client = OpenAI(api_key=api_key, base_url=base_url)
        default_model = ("openai/gpt-4o-mini" if using_openrouter or using_github_models
                         else "gpt-4o-mini")
        self.model = (model or os.environ.get("PIPELINE_MODEL")
                      or (os.environ.get("GITHUB_MODELS_MODEL") if using_github_models else None)
                      or default_model)
        self.temperature = temperature

    def complete_json(self, prompt: str) -> str:
        kwargs = {} if self.model.startswith("o") and self.model[1:2].isdigit() \
            else {"temperature": self.temperature}
        resp = self._client.chat.completions.create(
            model=self.model, response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}], **kwargs,
        )
        if isinstance(resp, str):
            return resp
        if isinstance(resp, dict):
            choices = resp.get("choices") or []
            if choices:
                message = choices[0].get("message") or {}
                return message.get("content") or "{}"
            return json.dumps(resp)
        return resp.choices[0].message.content or "{}"


class CountingLLM:
    """Wraps any client and counts calls (a cost / latency proxy)."""

    def __init__(self, inner: LLMClient):
        self.inner, self.calls = inner, 0

    def complete_json(self, prompt: str) -> str:
        self.calls += 1
        return self.inner.complete_json(prompt)

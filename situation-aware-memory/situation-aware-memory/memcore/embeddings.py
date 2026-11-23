"""Dense (semantic) retrieval behind an `Embedder` interface.

- `HashingEmbedder`: offline, deterministic bag-of-words + character n-gram
  hashing. NOT truly semantic (it cannot match "boss" with "manager"); it
  exists so the dense/hybrid code path runs and is testable without a model.
- `OpenAIEmbedder`: real embeddings for real runs (needs OPENAI_API_KEY).
"""
from __future__ import annotations

import hashlib
import math
from typing import Protocol

from .memory import MemoryStore
from .retrieval import TOP_K, Hit, tokenize


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def _h(feature: str) -> int:
    return int.from_bytes(hashlib.md5(feature.encode()).digest()[:8], "big")


class HashingEmbedder:
    name = "hashing"

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        toks = tokenize(text)
        feats = list(toks)
        for t in toks:
            p = f"#{t}#"
            feats += [p[i:i + 3] for i in range(len(p) - 2)]
        for f in feats:
            h = _h(f)
            v[h % self.dim] += 1.0 if (h >> 40) & 1 else -1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]


class OpenAIEmbedder:
    name = "openai"

    def __init__(self, model: str | None = None, batch: int = 256):
        from .llm import OpenAIChat  # reuse its .env / OpenRouter / OpenAI configuration

        chat = OpenAIChat()
        self._client, self.batch = chat._client, batch
        on_openrouter = "openrouter" in str(chat._client.base_url)
        self.model = model or ("openai/text-embedding-3-small" if on_openrouter
                               else "text-embedding-3-small")

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch):
            resp = self._client.embeddings.create(model=self.model, input=texts[i:i + self.batch])
            for d in resp.data:
                norm = math.sqrt(sum(x * x for x in d.embedding)) or 1.0
                out.append([x / norm for x in d.embedding])
        return out


class DenseRetriever:
    name = "dense"

    def __init__(self, store: MemoryStore, embedder: Embedder):
        self._records = store.all()
        self.embedder = embedder
        self._vecs = embedder.embed([r.text for r in self._records]) if self._records else []

    def search(self, query: str, top_k: int = TOP_K) -> list[Hit]:
        if not self._records:
            return []
        q = self.embedder.embed([query])[0]
        scored = [(sum(a * b for a, b in zip(q, v)), i) for i, v in enumerate(self._vecs)]
        scored.sort(key=lambda x: -x[0])
        return [Hit(rank=r + 1, score=s, record=self._records[i])
                for r, (s, i) in enumerate(scored[:top_k])]

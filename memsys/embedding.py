"""Embedding backends.

The encoder is fixed for the whole run and never fine-tuned (design doc 5.1).
`HashingEmbedder` is a dependency-free deterministic fallback so the package runs
and tests anywhere; swap in `SentenceTransformerEmbedder` for real experiments.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Iterable, Protocol

_WORD = re.compile(r"[a-z0-9]+|[一-鿿]")


class Embedder(Protocol):
    dim: int
    name: str

    def encode(self, texts: list[str]) -> list[list[float]]: ...


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    return sum(x * y for x, y in zip(a, b))  # both sides are L2-normalized


def _l2(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n else v


class HashingEmbedder:
    """Hashed bag of word-unigrams/bigrams + char 4-grams.

    Deterministic across processes (blake2b, not the salted builtin hash), needs
    no model download. Good enough for dedup thresholds and smoke tests; not a
    substitute for a real sentence encoder in the actual experiments.
    """

    def __init__(self, dim: int = 512):
        self.dim = dim
        self.name = f"hashing-{dim}"
        self._cache: dict[str, list[float]] = {}

    def _features(self, text: str) -> Iterable[tuple[str, float]]:
        t = text.lower()
        words = _WORD.findall(t)
        for w in words:
            yield ("w:" + w, 1.0)
        for a, b in zip(words, words[1:]):
            yield (f"b:{a}_{b}", 0.7)
        squashed = " ".join(words)
        for i in range(len(squashed) - 3):
            yield ("c:" + squashed[i : i + 4], 0.3)

    def _encode_one(self, text: str) -> list[float]:
        hit = self._cache.get(text)
        if hit is not None:
            return hit
        vec = [0.0] * self.dim
        for feat, w in self._features(text):
            h = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(h[:4], "little") % self.dim
            sign = 1.0 if h[4] & 1 else -1.0
            vec[idx] += sign * w
        vec = _l2(vec)
        if len(self._cache) < 100_000:
            self._cache[text] = vec
        return vec

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [self._encode_one(t) for t in texts]


class SentenceTransformerEmbedder:  # pragma: no cover - optional dependency
    """Real encoder for experiments, e.g. 'BAAI/bge-m3' or a gte model."""

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str | None = None, batch_size: int = 32):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, device=device)
        self.dim = self._model.get_sentence_embedding_dimension()
        self.name = model_name
        self.batch_size = batch_size
        self._cache: dict[str, list[float]] = {}

    def encode(self, texts: list[str]) -> list[list[float]]:
        todo = [t for t in texts if t not in self._cache]
        if todo:
            vecs = self._model.encode(
                todo, batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=False
            )
            for t, v in zip(todo, vecs):
                self._cache[t] = [float(x) for x in v]
        return [self._cache[t] for t in texts]


def default_embedder() -> Embedder:
    return HashingEmbedder()

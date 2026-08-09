"""Token counting. Uses tiktoken when available, otherwise a chars/4 heuristic.

The heuristic is only used for budget accounting; it is consistent across all
memory systems, so the equal-token-budget comparison (design doc 1.1) stays fair
either way.
"""

from __future__ import annotations

import math
from functools import lru_cache

_ENCODER = None
_ENCODER_TRIED = False


def _encoder():
    global _ENCODER, _ENCODER_TRIED
    if not _ENCODER_TRIED:
        _ENCODER_TRIED = True
        try:  # pragma: no cover - depends on optional dep
            import tiktoken

            _ENCODER = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _ENCODER = None
    return _ENCODER


@lru_cache(maxsize=8192)
def count_tokens(text: str) -> int:
    if not text:
        return 0
    enc = _encoder()
    if enc is not None:  # pragma: no cover - depends on optional dep
        return len(enc.encode(text))
    return max(1, math.ceil(len(text) / 4))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Best-effort truncation to a token budget (used for trajectory rendering)."""
    if max_tokens <= 0:
        return ""
    if count_tokens(text) <= max_tokens:
        return text
    enc = _encoder()
    if enc is not None:  # pragma: no cover
        return enc.decode(enc.encode(text)[:max_tokens])
    return text[: max_tokens * 4]

"""Benchmark adapters: turn a live agent loop into `Episode` objects.

The core of `memsys` is benchmark-agnostic -- everything downstream of an
`Episode` is shared. An adapter is therefore only responsible for two things:

1. running the agent against a benchmark with a memory block injected, and
2. packaging the resulting rollouts as an `Episode`.

Adapters are imported lazily so the zero-dependency core stays importable
without ALFWorld / TextWorld / openai installed.
"""

from __future__ import annotations

__all__ = ["alfworld", "webshop"]

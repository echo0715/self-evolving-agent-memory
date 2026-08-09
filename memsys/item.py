"""The unified memory item (design doc 1).

All four types share this envelope so retrieval, dedup, budget packing, eviction and
logging are written exactly once.

`support` / `refute` live here rather than inside `content` because verification is a
write MECHANISM, not a property of the rule format -- once it applies to every type,
its bookkeeping belongs to every type.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from typing import Any

from .schemas import get_type, laplace_confidence


def new_id() -> str:
    return uuid.uuid4().hex[:12]


@dataclass
class MemoryItem:
    type: str
    content: dict[str, Any]

    id: str = field(default_factory=new_id)
    retrieval_key: str = ""
    embedding: list[float] | None = None
    scope: dict[str, str] = field(default_factory=dict)

    # ---- provenance (needed by the Memory Source experiments) ----
    source_task_ids: list[str] = field(default_factory=list)
    source_outcome: str = "success"  # success | failure | mixed
    writer_model: str = ""
    created_at_step: int = 0
    updated_at_step: int = 0
    version: int = 1
    superseded_by: str | None = None

    # ---- verification bookkeeping (uniform across types) ----
    support: int = 0
    refute: int = 0

    # ---- retrieval usage statistics ----
    n_retrieved: int = 0
    n_retrieved_success: int = 0

    # ---- misc per-type bookkeeping (counterevidence, failed-step counters, ...) ----
    stats: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        spec = get_type(self.type)
        self.content = spec.normalize(self.content)
        if not self.retrieval_key:
            self.retrieval_key = spec.retrieval_key(self.content)

    # -- derived --
    @property
    def spec(self) -> type:
        return get_type(self.type)

    @property
    def alive(self) -> bool:
        return self.superseded_by is None

    def confidence(self) -> float:
        return laplace_confidence(self.support, self.refute)

    def n_observations(self) -> int:
        return self.support + self.refute

    def reset_evidence(self) -> None:
        """A refined entry is a new claim: its old evidence no longer applies."""
        self.support = self.refute = 0
        self.stats.pop("counterevidence", None)

    def refresh_key(self) -> None:
        """Call after mutating content; invalidates the embedding."""
        self.retrieval_key = self.spec.retrieval_key(self.content)
        self.embedding = None

    def render(self, verbose: bool = False) -> str:
        out = self.spec.render(self.content, verbose=verbose)
        if verbose and self.n_observations():
            out += f"\n   [conf={self.confidence():.2f} support={self.support} refute={self.refute}]"
        return out

    def utility(self) -> float:
        """(hits+1)/(uses+2) -- design doc 5.4"""
        return (self.n_retrieved_success + 1) / (self.n_retrieved + 2)

    def priority(self) -> float:
        return self.utility() * (1.0 + math.log(1.0 + self.n_retrieved))

    # -- persistence --
    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "embedding"}

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryItem":
        d = dict(d)
        d.pop("embedding", None)
        return cls(**d)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        head = self.retrieval_key[:60].replace("\n", " ")
        return f"<{self.type}:{self.id} v{self.version} {head!r}>"

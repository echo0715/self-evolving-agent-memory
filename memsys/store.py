"""Memory store: indexing, scoped retrieval, token-budget packing, eviction.

Shared verbatim by all four memory types -- that sharing is what keeps the
Memory Content comparison controlled (design doc 0).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable

from .config import MemoryConfig
from .embedding import Embedder, cosine, default_embedder
from .item import MemoryItem
from .tokens import count_tokens


@dataclass
class Scored:
    item: MemoryItem
    score: float
    similarity: float


class MemoryStore:
    def __init__(self, embedder: Embedder | None = None, config: MemoryConfig | None = None):
        self.embedder = embedder or default_embedder()
        self.config = config or MemoryConfig()
        self._items: dict[str, MemoryItem] = {}

    # ------------------------------------------------------------------ basics
    def __len__(self) -> int:
        return sum(1 for i in self._items.values() if i.alive)

    def __contains__(self, item_id: str) -> bool:
        return item_id in self._items

    def get(self, item_id: str) -> MemoryItem | None:
        return self._items.get(item_id)

    def items(self, type: str | None = None, include_dead: bool = False) -> list[MemoryItem]:
        return [
            i
            for i in self._items.values()
            if (include_dead or i.alive) and (type is None or i.type == type)
        ]

    def _ensure_embedding(self, item: MemoryItem) -> None:
        if item.embedding is None:
            item.embedding = self.embedder.encode([item.retrieval_key])[0]

    def add(self, item: MemoryItem) -> MemoryItem:
        self._ensure_embedding(item)
        self._items[item.id] = item
        return item

    def remove(self, item_id: str) -> None:
        self._items.pop(item_id, None)

    def supersede(self, old_id: str, new_id: str) -> None:
        """Soft delete: keeps the row for provenance analysis (design doc 1)."""
        old = self._items.get(old_id)
        if old is not None:
            old.superseded_by = new_id

    def touch(self, item: MemoryItem, step: int) -> None:
        item.updated_at_step = step
        item.version += 1
        item.refresh_key()
        self._ensure_embedding(item)

    # --------------------------------------------------------------- retrieval
    def _scope_ok(self, item: MemoryItem, scope: dict[str, str] | None) -> bool:
        if not scope:
            return True
        for k in self.config.scope_filter_keys:
            want = scope.get(k)
            if want is not None and item.scope.get(k) != want:
                return False
        return True

    def search(
        self,
        query: str,
        type: str | None = None,
        scope: dict[str, str] | None = None,
        k: int | None = None,
        candidates: Iterable[MemoryItem] | None = None,
    ) -> list[Scored]:
        pool = list(candidates) if candidates is not None else self.items(type=type)
        pool = [i for i in pool if self._scope_ok(i, scope)]
        if not pool:
            return []
        qv = self.embedder.encode([query])[0]
        lam = self.config.utility_lambda
        out = []
        for it in pool:
            self._ensure_embedding(it)
            sim = cosine(qv, it.embedding)
            score = sim + lam * it.utility() if lam else sim
            out.append(Scored(it, score, sim))
        out.sort(key=lambda s: s.score, reverse=True)
        return out[: (k or self.config.k_pool)]

    def nearest(
        self, key: str, type: str, scope: dict[str, str] | None = None, exclude: set[str] | None = None
    ) -> Scored | None:
        """Nearest same-type neighbour, used for the dedup / merge decision."""
        pool = [i for i in self.items(type=type) if not exclude or i.id not in exclude]
        hits = self.search(key, type=type, scope=scope, k=1, candidates=pool)
        return hits[0] if hits else None

    # ------------------------------------------------------------ budget pack
    def pack(
        self,
        scored: list[Scored],
        budget_tokens: int | None = None,
        max_items: int | None = None,
        verbose: bool = False,
    ) -> tuple[list[MemoryItem], str, int]:
        """Greedily fill the injection block by score (design doc 1.1).

        Returns (selected_items, rendered_block, token_count). Items that do not
        fit are dropped, never truncated -- a half-rendered rule is worse than none.
        """
        budget = self.config.injection_budget_tokens if budget_tokens is None else budget_tokens
        cap = max_items if max_items is not None else self.config.equal_item_count

        chosen: list[MemoryItem] = []
        header_cost = 0
        used = 0
        for s in scored:
            if cap is not None and len(chosen) >= cap:
                break
            body = s.item.render(verbose=verbose)
            if not chosen:
                header_cost = count_tokens(s.item.spec.block_header()) + 1
            cost = count_tokens(body) + 2  # newline + numbering slack
            if used + cost + header_cost > budget:
                continue
            chosen.append(s.item)
            used += cost
        if not chosen:
            return [], "", 0
        block = render_block(chosen, verbose=verbose)
        return chosen, block, count_tokens(block)

    # ------------------------------------------------------------- maintenance
    def record_usage(self, items: Iterable[MemoryItem], task_succeeded: bool) -> None:
        for it in items:
            it.n_retrieved += 1
            if task_succeeded:
                it.n_retrieved_success += 1

    def enforce_capacity(self, max_items: int | None = None) -> list[MemoryItem]:
        """Evict lowest-priority items; never-retrieved items go first, oldest first.

        Returns the evicted items so the caller can log them.
        """
        cap = self.config.max_items if max_items is None else max_items
        live = self.items()
        if len(live) <= cap:
            return []

        def key(it: MemoryItem):
            if it.n_retrieved == 0:
                return (0, float(it.created_at_step))
            return (1, it.priority())

        live.sort(key=key)
        evicted = live[: len(live) - cap]
        for it in evicted:
            self.remove(it.id)
        return evicted

    # ------------------------------------------------------------- persistence
    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for it in self._items.values():
                f.write(json.dumps(it.to_dict(), ensure_ascii=False) + "\n")

    def load(self, path: str) -> None:
        self._items.clear()
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    it = MemoryItem.from_dict(json.loads(line))
                    self._items[it.id] = it

    def summary(self) -> dict:
        live = self.items()
        by_type: dict[str, int] = {}
        for it in live:
            by_type[it.type] = by_type.get(it.type, 0) + 1
        return {
            "n_live": len(live),
            "n_total_rows": len(self._items),
            "by_type": by_type,
            "mean_version": (sum(i.version for i in live) / len(live)) if live else 0.0,
        }


def render_block(items: list[MemoryItem], verbose: bool = False) -> str:
    """Render selected items as the memory block injected into the agent prompt."""
    if not items:
        return ""
    lines = [items[0].spec.block_header()]
    numbered = items[0].type in ("reflection", "rule", "raw")
    for i, it in enumerate(items):
        body = it.render(verbose=verbose)
        if it.type == "rule":
            lines.append(f"- {body}")
        elif numbered:
            lines.append(f"{i+1}. {body}")
        else:
            lines.append(body)
    return "\n".join(lines)

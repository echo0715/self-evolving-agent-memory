"""Benchmark-agnostic rollout / episode containers.

Nothing here knows about ALFWorld, WebShop or AppWorld. A benchmark adapter only
has to produce `Episode` objects; everything downstream is shared.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from .tokens import truncate_to_tokens

ALL_SUCCESS = "all_success"
ALL_FAILURE = "all_failure"
MIXED = "mixed"


@dataclass
class Step:
    """One agent turn. `thought` may be empty for non-ReAct agents."""

    action: str
    observation: str = ""
    thought: str = ""

    def render(self, idx: int | None = None, max_obs_chars: int = 400) -> str:
        head = f"[{idx}] " if idx is not None else ""
        parts = []
        if self.thought:
            parts.append(f"{head}think: {self.thought}")
            head = ""
        obs = self.observation
        if len(obs) > max_obs_chars:
            obs = obs[:max_obs_chars] + " ...<truncated>"
        parts.append(f"{head}> {self.action}")
        if obs:
            parts.append(f"  {obs}")
        return "\n".join(parts)


@dataclass
class Rollout:
    rollout_id: str
    steps: list[Step] = field(default_factory=list)
    reward: float = 0.0
    success: bool = False
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.steps)

    def render(self, max_tokens: int = 1200, max_obs_chars: int = 400) -> str:
        body = "\n".join(
            s.render(i + 1, max_obs_chars=max_obs_chars) for i, s in enumerate(self.steps)
        )
        tag = "SUCCESS" if self.success else "FAILURE"
        head = f"--- rollout {self.rollout_id} ({tag}, reward={self.reward:g}, {len(self.steps)} steps) ---"
        if self.error:
            body += f"\n[terminated: {self.error}]"
        return head + "\n" + truncate_to_tokens(body, max_tokens)

    def raw_text(self) -> str:
        """Un-truncated concatenation, used for evidence substring checks."""
        chunks = []
        for s in self.steps:
            if s.thought:
                chunks.append(s.thought)
            chunks.append(s.action)
            if s.observation:
                chunks.append(s.observation)
        if self.error:
            chunks.append(self.error)
        return "\n".join(chunks)


@dataclass
class Episode:
    """All rollouts the agent produced for one task instance."""

    task_id: str
    instruction: str
    rollouts: list[Rollout] = field(default_factory=list)
    scope: dict[str, str] = field(default_factory=dict)  # {"env": "alfworld", "task_type": ...}
    step_index: int = 0  # which evolving step this episode is (design doc 7)
    meta: dict[str, Any] = field(default_factory=dict)

    # ---- convenience ----
    def successes(self) -> list[Rollout]:
        return [r for r in self.rollouts if r.success]

    def failures(self) -> list[Rollout]:
        return [r for r in self.rollouts if not r.success]

    @property
    def any_success(self) -> bool:
        return any(r.success for r in self.rollouts)

    @property
    def success_rate(self) -> float:
        return (sum(1 for r in self.rollouts if r.success) / len(self.rollouts)) if self.rollouts else 0.0

    def outcome(self) -> str:
        """Rollout-filtering bucket (design doc 6.3)."""
        if not self.rollouts:
            return ALL_FAILURE
        n_ok = sum(1 for r in self.rollouts if r.success)
        if n_ok == len(self.rollouts):
            return ALL_SUCCESS
        if n_ok == 0:
            return ALL_FAILURE
        return MIXED

    def best_success(self) -> Rollout | None:
        ok = self.successes()
        return min(ok, key=len) if ok else None  # shortest success = cleanest exemplar

    def worst_failure(self) -> Rollout | None:
        bad = self.failures()
        return max(bad, key=len) if bad else None

    def raw_text(self) -> str:
        return "\n".join(r.raw_text() for r in self.rollouts)

    def render(self, rollouts: Iterable[Rollout] | None = None, max_tokens_each: int = 1200) -> str:
        rs = list(rollouts) if rollouts is not None else self.rollouts
        return "\n\n".join(r.render(max_tokens=max_tokens_each) for r in rs)


_WS = re.compile(r"\s+")
_WORD = re.compile(r"\w+")


def _norm(s: str) -> str:
    return _WS.sub(" ", s.lower()).strip()


def evidence_supported(evidence: str, haystack: str, mode: str = "fuzzy", min_ratio: float = 0.7) -> bool:
    """Check that a writer-produced `evidence` string really comes from the trajectory.

    This is the main hallucination guard for Reflection memory (design doc 2.1).
    mode: "strict" (normalized substring), "fuzzy" (substring or token containment), "off".
    """
    if mode == "off":
        return True
    e, h = _norm(evidence), _norm(haystack)
    if not e:
        return False
    if e in h:
        return True
    if mode == "strict":
        return False
    toks = set(_WORD.findall(e))
    if not toks:
        return False
    hit = sum(1 for t in toks if t in h)
    return hit / len(toks) >= min_ratio

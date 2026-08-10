"""Evolving-phase driver + the JSONL log from design doc 7.

Benchmark-agnostic: it consumes `Episode` objects from anywhere (a live agent
loop, or replayed rollouts from disk) and drives retrieve -> observe -> log.
The evaluation phase must use `frozen()`, which makes writes impossible rather
than merely discouraged.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from .config import MemoryConfig
from .episode import Episode
from .systems import MemorySystem, Retrieval, WriteReport


class RunLogger:
    def __init__(self, path: str | None = None):
        self.path = path
        self.records: list[dict] = []
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            open(path, "w").close()

    def log(self, record: dict) -> None:
        self.records.append(record)
        if self.path:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    # -- analysis helpers for the Memory Source tables --
    def op_distribution(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.records:
            for op in r.get("writer_ops", []):
                out[op["op"]] = out.get(op["op"], 0) + 1
        return out

    def growth_curve(self) -> list[tuple[int, int]]:
        return [(r["step"], r["store_size"]) for r in self.records]


@dataclass
class StepRecord:
    step: int
    task_id: str
    rollout_rewards: list[float]
    retrieved_ids: list[str]
    injected_tokens: int
    writer_ops: list[dict]
    rejected: list[dict]
    evicted: list[str]
    writer_prompt_tokens: int
    writer_completion_tokens: int
    writer_calls: int
    store_size: int
    outcome: str
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return self.__dict__


def _usage_snapshot(system: MemorySystem) -> tuple[int, int, int]:
    """(prompt_tokens, completion_tokens, n_calls) across every writer involved."""
    writers = []
    if getattr(system, "writer", None) is not None:
        writers.append(system.writer)
    for sub in getattr(system, "systems", {}).values():
        if getattr(sub, "writer", None) is not None:
            writers.append(sub.writer)
    seen, p, c, n = set(), 0, 0, 0
    for w in writers:
        if id(w.llm) in seen:
            continue
        seen.add(id(w.llm))
        u = w.llm.usage
        p += u.prompt_tokens
        c += u.completion_tokens
        n += u.n_calls
    return p, c, n


def _store_size(system: MemorySystem) -> int:
    if getattr(system, "store", None) is not None:
        return len(system.store)
    return sum(len(s.store) for s in getattr(system, "systems", {}).values())


@contextmanager
def frozen(system: MemorySystem):
    """Evaluation phase: retrieval works, every write raises."""

    def _blocked(*a, **kw):
        raise RuntimeError("memory is frozen during evaluation (design doc 0)")

    targets = [system] + list(getattr(system, "systems", {}).values())
    saved = [(t, t.observe) for t in targets]
    for t, _ in saved:
        t.observe = _blocked  # type: ignore[method-assign]
    try:
        yield system
    finally:
        for t, fn in saved:
            t.observe = fn  # type: ignore[method-assign]


class Evolver:
    """Runs the evolving loop over a stream of episodes."""

    def __init__(self, system: MemorySystem, config: MemoryConfig | None = None, logger: RunLogger | None = None):
        self.system = system
        self.config = config or system.config
        self.logger = logger or RunLogger()
        self.step = 0

    def retrieve(self, instruction: str, scope: dict[str, str] | None = None) -> Retrieval:
        return self.system.retrieve(instruction, scope=scope)

    def step_once(self, episode: Episode, retrieval: Retrieval | None = None) -> WriteReport:
        episode.step_index = self.step
        if retrieval is None:
            retrieval = (
                self.retrieve(episode.instruction, episode.scope)
                if self.config.retrieve_during_evolving
                else Retrieval()
            )
        p0, c0, n0 = _usage_snapshot(self.system)
        report = self.system.observe(episode, retrieval)
        p1, c1, n1 = _usage_snapshot(self.system)

        self.logger.log(
            StepRecord(
                step=self.step,
                task_id=episode.task_id,
                rollout_rewards=[r.reward for r in episode.rollouts],
                retrieved_ids=retrieval.ids(),
                injected_tokens=retrieval.tokens,
                writer_ops=report.applied,
                rejected=report.rejected,
                evicted=report.evicted,
                writer_prompt_tokens=p1 - p0,
                writer_completion_tokens=c1 - c0,
                writer_calls=n1 - n0,
                store_size=_store_size(self.system),
                outcome=episode.outcome(),
                extra=report.extra,
            ).to_dict()
        )
        self.step += 1
        return report

    def run(
        self,
        episodes: Iterable[Episode],
        on_step: Callable[[Episode, WriteReport], None] | None = None,
    ) -> list[WriteReport]:
        out = []
        for ep in episodes:
            rep = self.step_once(ep)
            out.append(rep)
            if on_step:
                on_step(ep, rep)
        self.flush()
        return out

    def flush(self) -> None:
        """Induce over the trailing partial batch, which the cadence would otherwise skip.

        The resulting ops are logged like any other step. They must be: the
        induction writer may APPEND consolidated entries and DELETE the ones they
        subsume, and `store.remove()` is a hard delete. Dropping this report on
        the floor -- as this method used to -- left the final store unexplainable
        from the log (four appends recorded, two entries actually present) and
        made `op_distribution()` undercount every delete and induced append.
        """
        for sys_ in [self.system] + list(getattr(self.system, "systems", {}).values()):
            if not (sys_.policy.batch_induction and getattr(sys_, "_buffer", None)):
                continue
            p0, c0, n0 = _usage_snapshot(self.system)
            report = sys_.run_batch_induction()
            p1, c1, n1 = _usage_snapshot(self.system)
            self.logger.log(
                StepRecord(
                    step=self.step,
                    task_id=f"__flush_induction__:{sys_.type_name}",
                    rollout_rewards=[],
                    retrieved_ids=[],
                    injected_tokens=0,
                    writer_ops=report.applied,
                    rejected=report.rejected,
                    evicted=report.evicted,
                    writer_prompt_tokens=p1 - p0,
                    writer_completion_tokens=c1 - c0,
                    writer_calls=n1 - n0,
                    store_size=_store_size(self.system),
                    outcome="flush_induction",
                    extra=report.extra,
                ).to_dict()
            )
            self.step += 1

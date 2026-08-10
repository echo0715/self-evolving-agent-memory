"""The memory systems: retrieval + write policy on top of a shared store.

`MemorySystem` implements EVERY mechanism -- online write, dedup/merge,
verification, refinement, confidence and utility deletion, cross-task batch
induction -- and runs whichever ones `WritePolicy` enables. The subclasses differ
only in their content type and writer:

    ReflectionSystem / RuleSystem / SkillSystem   -- type + writer, nothing else
    RawTrajectorySystem                           -- the baseline: no LLM writer
    CompositeSystem                               -- the "with all" row

Holding the policy fixed across the three types is what makes the Memory Content
table a comparison of CONTENT. Varying it (see NATIVE_POLICIES) is a separate
write-mechanism ablation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from .config import MemoryConfig, WritePolicy
from .embedding import cosine
from .episode import ALL_FAILURE, ALL_SUCCESS, MIXED, Episode
from .item import MemoryItem
from .llm import LLMClient
from .store import MemoryStore, Scored
from .writers import (
    APPEND,
    DELETE,
    FOLLOWED_BAD,
    FOLLOWED_OK,
    REVISE,
    BaseWriter,
    Proposal,
    ReflectionWriter,
    RuleWriter,
    SkillWriter,
    WriteOp,
)

_OUTCOME_MAP = {ALL_SUCCESS: "success", ALL_FAILURE: "failure", MIXED: "mixed"}


@dataclass
class Retrieval:
    items: list[MemoryItem] = field(default_factory=list)
    block: str = ""
    tokens: int = 0
    scored: list[Scored] = field(default_factory=list)

    def ids(self) -> list[str]:
        return [i.id for i in self.items]


@dataclass
class WriteReport:
    applied: list[dict] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)
    evicted: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for a in self.applied:
            out[a["op"]] = out.get(a["op"], 0) + 1
        return out

    def merge_in(self, other: "WriteReport") -> "WriteReport":
        self.applied += other.applied
        self.rejected += other.rejected
        self.evicted += other.evicted
        for k, v in other.extra.items():
            self.extra[k] = v
        return self


# ==========================================================================
class MemorySystem:
    type_name: str = ""

    def __init__(
        self,
        store: MemoryStore | None = None,
        writer: BaseWriter | None = None,
        config: MemoryConfig | None = None,
    ):
        self.config = config or (store.config if store else MemoryConfig())
        self.store = store or MemoryStore(config=self.config)
        self.writer = writer
        self._n_observed = 0
        self._buffer: list[Episode] = []

    @property
    def policy(self) -> WritePolicy:
        return self.config.policy_for(self.type_name)

    # ------------------------------------------------------------- retrieval
    def retrieve(
        self, instruction: str, scope: dict[str, str] | None = None, budget_tokens: int | None = None
    ) -> Retrieval:
        scored = self.store.search(instruction, type=self.type_name, scope=scope)
        items, block, tokens = self.store.pack(scored, budget_tokens=budget_tokens)
        return Retrieval(items=items, block=block, tokens=tokens, scored=scored)

    # ----------------------------------------------------------------- write
    def observe(self, episode: Episode, retrieval: Retrieval | None = None) -> WriteReport:
        """One evolving step. Runs exactly the mechanisms the policy enables."""
        retrieved = list(retrieval.items) if retrieval else []
        self.store.record_usage(retrieved, episode.any_success)
        pol = self.policy
        report = WriteReport()

        # (1) online extraction from this episode
        if pol.online_write and self.writer is not None:
            proposal = self.writer.propose(episode, retrieved)
            report.merge_in(self.apply(proposal.ops, episode))
            report.rejected += proposal.rejected

        # (2) verification of the entries that were actually injected
        if pol.verify and self.writer is not None and retrieved:
            report.merge_in(self._verify(episode, retrieved))

        # (3) refinement / (4) confidence deletion, on the verified entries
        if (pol.refine or pol.delete_on_low_confidence) and self.writer is not None:
            report.merge_in(self._refine_and_prune(episode, retrieved))

        # (5) utility-based deletion
        if pol.utility_deletion:
            report.merge_in(self._utility_prune())

        # type-specific extras (currently: skill step refinement, off by default)
        report.merge_in(self.post_observe(episode, retrieved))

        # (6) cross-task batch induction
        self._buffer.append(episode)
        if pol.batch_induction and pol.batch_every > 0 and (self._n_observed + 1) % pol.batch_every == 0:
            report.merge_in(self.run_batch_induction())

        self._n_observed += 1
        report.evicted += [e.id for e in self.store.enforce_capacity()]
        return report

    def post_observe(self, episode: Episode, retrieved: list[MemoryItem]) -> WriteReport:
        return WriteReport()

    # ------------------------------------------------ (2) verification pass
    def _verify(self, episode: Episode, retrieved: list[MemoryItem]) -> WriteReport:
        rep = WriteReport()
        live = [r for r in retrieved if r.id in self.store]
        if not live:
            return rep
        verdicts = self.writer.judge(episode, live)
        by_id = {r.id: r for r in live}
        for v in verdicts:
            item = by_id.get(v.item_id)
            if item is None:
                continue
            if v.verdict == FOLLOWED_OK:
                item.support += 1
            elif v.verdict == FOLLOWED_BAD:
                item.refute += 1
                item.stats.setdefault("counterevidence", []).append(
                    f"[task {episode.task_id}] {v.reason}"
                )
            else:
                continue  # not_applicable / violated say nothing about the entry itself
            item.updated_at_step = episode.step_index
        rep.extra[f"{self.type_name}_verdicts"] = [(v.item_id, v.verdict) for v in verdicts]
        return rep

    # ------------------------------- (3)(4) refinement + confidence deletion
    def _refine_and_prune(self, episode: Episode, retrieved: list[MemoryItem]) -> WriteReport:
        rep = WriteReport()
        cfg, pol = self.config, self.policy
        for item in list(retrieved):
            if item.id not in self.store:
                continue
            conf = item.confidence()
            if (
                pol.delete_on_low_confidence
                and conf < cfg.delete_confidence
                and item.n_observations() >= cfg.delete_min_observations
            ):
                self.store.remove(item.id)
                rep.applied.append({"op": DELETE, "id": item.id, "reason": f"confidence {conf:.2f} too low"})
                continue
            if pol.refine and item.refute >= cfg.refute_threshold and conf < cfg.refine_confidence:
                op = self.writer.refine(item, item.stats.get("counterevidence", []), episode)
                if op is not None:
                    rep.merge_in(self.apply([op], episode))
        return rep

    # --------------------------------------------- (5) utility-based deletion
    def _utility_prune(self) -> WriteReport:
        rep = WriteReport()
        cfg = self.config
        for it in list(self.store.items(type=self.type_name)):
            if (
                it.n_retrieved >= cfg.utility_delete_min_retrieved
                and it.n_retrieved_success / it.n_retrieved < cfg.utility_delete_success_rate
            ):
                self.store.remove(it.id)
                rep.applied.append({"op": DELETE, "id": it.id, "reason": f"utility {it.utility():.2f} below floor"})
        return rep

    # -------------------------------------------- (6) cross-task consolidation
    def run_batch_induction(self, episodes: list[Episode] | None = None) -> WriteReport:
        rep = WriteReport()
        batch = episodes if episodes is not None else self._buffer
        if not batch or self.writer is None:
            if episodes is None:
                self._buffer = []
            return rep
        clusters = self.cluster(batch)
        rep.extra[f"{self.type_name}_batch"] = {"n_episodes": len(batch), "n_clusters": len(clusters)}
        for name, eps in clusters.items():
            existing = self.store.items(type=self.type_name)
            proposal: Proposal = self.writer.induce(eps, existing, cluster_name=name)
            rep.merge_in(self.apply(proposal.ops, eps[0]))
            rep.rejected += proposal.rejected
        if episodes is None:
            self._buffer = []
        return rep

    def cluster(self, episodes: list[Episode]) -> dict[str, list[Episode]]:
        """Group a batch by task type when the scope provides one, else greedily by
        instruction embedding similarity. Identical for every memory type."""
        key = self.config.cluster_key
        if all(ep.scope.get(key) for ep in episodes):
            out: dict[str, list[Episode]] = {}
            for ep in episodes:
                out.setdefault(ep.scope[key], []).append(ep)
            return out

        vecs = self.store.embedder.encode([ep.instruction for ep in episodes])
        centroids: list[list[float]] = []
        groups: list[list[Episode]] = []
        for ep, v in zip(episodes, vecs):
            best_i, best_sim = -1, -1.0
            for i, c in enumerate(centroids):
                s = cosine(v, c)
                if s > best_sim:
                    best_i, best_sim = i, s
            if best_i >= 0 and best_sim >= self.config.cluster_sim:
                groups[best_i].append(ep)
            else:
                centroids.append(v)
                groups.append([ep])
        return {f"cluster_{i}": g for i, g in enumerate(groups)}

    # ------------------------------------------------------------ op applier
    def apply(self, ops: Sequence[WriteOp], episode: Episode) -> WriteReport:
        rep = WriteReport()
        for op in ops:
            if op.op == DELETE:
                if op.target_id and op.target_id in self.store:
                    self.store.remove(op.target_id)
                    rep.applied.append({"op": DELETE, "id": op.target_id, "reason": op.reason})
                else:
                    # A DELETE naming an entry that is not in the store used to
                    # vanish silently. Record it: a writer that mostly emits
                    # unresolvable ids is malfunctioning, and the only evidence
                    # is the gap between ops proposed and ops applied.
                    rep.rejected.append(
                        {"content": None, "errors": [f"DELETE target not in store: {op.target_id!r}"]}
                    )
                continue

            if op.op == REVISE:
                target = self.store.get(op.target_id) if op.target_id else None
                if target is None or op.content is None:
                    rep.rejected.append({"content": op.content, "errors": ["REVISE target not found"]})
                    continue
                target.content = op.content
                target.reset_evidence()  # a reformulated entry is a new claim
                self.store.touch(target, episode.step_index)
                rep.applied.append({"op": REVISE, "id": target.id, "reason": op.reason})
                continue

            # APPEND (with dedup -> merge)
            if op.content is None:
                continue
            item = self._make_item(op.content, episode)
            near = self.store.nearest(item.retrieval_key, type=self.type_name, scope=episode.scope)
            tau = self.config.tau_dup.get(self.type_name, 0.85)
            if near is not None and near.similarity >= tau:
                rep.merge_in(self._merge(near.item, item, episode, near.similarity))
                continue
            self.store.add(item)
            rep.applied.append({"op": APPEND, "id": item.id, "reason": op.reason})
        return rep

    def _merge(self, existing: MemoryItem, incoming: MemoryItem, episode: Episode, sim: float) -> WriteReport:
        rep = WriteReport()
        if not self.policy.merge_on_duplicate or self.writer is None:
            rep.rejected.append(
                {"content": incoming.content, "errors": [f"duplicate of {existing.id} (sim={sim:.3f})"]}
            )
            return rep
        merged_content = self.writer.merge(existing, incoming.content, episode)
        if merged_content is None:
            rep.rejected.append({"content": incoming.content, "errors": ["merge call failed"]})
            return rep

        merged = self._make_item(merged_content, episode)
        merged.source_task_ids = sorted(set(existing.source_task_ids) | set(incoming.source_task_ids))
        merged.created_at_step = existing.created_at_step
        merged.version = existing.version + 1
        merged.n_retrieved = existing.n_retrieved
        merged.n_retrieved_success = existing.n_retrieved_success
        merged.stats = dict(existing.stats)
        # verification evidence is additive and type-neutral
        merged.support = existing.support + incoming.support
        merged.refute = existing.refute + incoming.refute
        if existing.source_outcome != incoming.source_outcome:
            merged.source_outcome = "mixed"

        self.store.add(merged)
        self.store.supersede(existing.id, merged.id)
        rep.applied.append({"op": "MERGE", "id": merged.id, "reason": f"merged {existing.id} (sim={sim:.3f})"})
        return rep

    def _make_item(self, content: dict, episode: Episode) -> MemoryItem:
        return MemoryItem(
            type=self.type_name,
            content=content,
            scope=dict(episode.scope),
            source_task_ids=[episode.task_id],
            source_outcome=_OUTCOME_MAP.get(episode.outcome(), "mixed"),
            writer_model=getattr(self.writer.llm, "model", "") if self.writer else "",
            created_at_step=episode.step_index,
            updated_at_step=episode.step_index,
        )

    def summary(self) -> dict:
        return {"type": self.type_name, "policy": self.policy.describe(), **self.store.summary()}


# ==========================================================================
class ReflectionSystem(MemorySystem):
    """Design doc 2. Content type only -- mechanisms come from the policy."""

    type_name = "reflection"

    def __init__(self, llm: LLMClient | None = None, store=None, config=None, writer=None):
        config = config or (store.config if store else MemoryConfig())
        writer = writer or (ReflectionWriter(llm, config) if llm else None)
        super().__init__(store=store, writer=writer, config=config)


class RuleSystem(MemorySystem):
    """Design doc 3. Content type only -- mechanisms come from the policy."""

    type_name = "rule"

    def __init__(self, llm: LLMClient | None = None, store=None, config=None, writer=None):
        config = config or (store.config if store else MemoryConfig())
        writer = writer or (RuleWriter(llm, config) if llm else None)
        super().__init__(store=store, writer=writer, config=config)


class SkillSystem(MemorySystem):
    """Design doc 4. Content type + one optional extra: failing-step refinement.

    That extra is OFF by default because no other type has an analogue, so enabling
    it reintroduces the asymmetry the policy split exists to remove. It also needs
    `episode.meta["failed_skill_steps"] = {skill_id: step_idx}` from the harness.
    """

    type_name = "skill"

    def __init__(self, llm: LLMClient | None = None, store=None, config=None, writer=None):
        config = config or (store.config if store else MemoryConfig())
        writer = writer or (SkillWriter(llm, config) if llm else None)
        super().__init__(store=store, writer=writer, config=config)

    def post_observe(self, episode: Episode, retrieved: list[MemoryItem]) -> WriteReport:
        rep = WriteReport()
        if not self.config.skill_step_refinement or self.writer is None:
            return rep
        if not retrieved or episode.any_success:
            return rep
        hint = episode.meta.get("failed_skill_steps") or {}
        for skill in retrieved:
            if skill.id not in self.store:
                continue
            idx = hint.get(skill.id)
            if idx is None:
                continue
            counters = skill.stats.setdefault("failed_steps", {})
            k = str(idx)
            counters[k] = counters.get(k, 0) + 1
            skill.stats.setdefault("failure_evidence", []).append(
                f"[task {episode.task_id}] step {int(idx) + 1} failed"
            )
            if counters[k] >= self.config.skill_failed_step_threshold:
                op = self.writer.refine_step(skill, int(idx), skill.stats.get("failure_evidence", []), episode)
                if op is not None:
                    rep.merge_in(self.apply([op], episode))
                    counters[k] = 0
        return rep


# ==========================================================================
class RawTrajectorySystem(MemorySystem):
    """Design doc 1: the baseline. Append-only, successes only, no LLM writer.

    Being writer-free, it cannot run verification/refinement/induction no matter what
    the policy says -- it is a baseline, not a peer in the mechanism comparison.
    Utility-based deletion still applies, since that needs no LLM.
    """

    type_name = "raw"

    def __init__(self, store=None, config=None, max_trajectory_tokens: int = 900):
        config = config or (store.config if store else MemoryConfig())
        super().__init__(store=store, writer=None, config=config)
        self.max_trajectory_tokens = max_trajectory_tokens

    def observe(self, episode: Episode, retrieval: Retrieval | None = None) -> WriteReport:
        retrieved = list(retrieval.items) if retrieval else []
        self.store.record_usage(retrieved, episode.any_success)
        rep = WriteReport()
        r = episode.best_success()
        if self.policy.online_write and r is not None:
            content = {
                "instruction": episode.instruction,
                "trajectory": r.render(max_tokens=self.max_trajectory_tokens),
                "reward": r.reward,
            }
            item = self._make_item(content, episode)
            near = self.store.nearest(item.retrieval_key, type=self.type_name, scope=episode.scope)
            if near is not None and near.similarity >= self.config.tau_dup.get("raw", 0.98):
                rep.rejected.append({"content": None, "errors": [f"duplicate of {near.item.id}"]})
            else:
                self.store.add(item)
                rep.applied.append({"op": APPEND, "id": item.id, "reason": "successful trajectory"})
        if self.policy.utility_deletion:
            rep.merge_in(self._utility_prune())
        self._n_observed += 1
        rep.evicted += [e.id for e in self.store.enforce_capacity()]
        return rep


# ==========================================================================
class CompositeSystem(MemorySystem):
    """The "With all" row (doc 5.5): three independent stores, split budget.

    Writer cost is ~3x a single system; `writer_usage()` exposes that so the ablation
    can be reported honestly.
    """

    type_name = "composite"

    def __init__(self, systems: dict[str, MemorySystem], config: MemoryConfig | None = None):
        self.systems = systems
        self.config = config or next(iter(systems.values())).config
        self.store = None  # type: ignore[assignment]
        self.writer = None
        self._n_observed = 0
        self._buffer = []

    @property
    def policy(self) -> WritePolicy:
        return self.config.policy

    def retrieve(
        self, instruction: str, scope: dict[str, str] | None = None, budget_tokens: int | None = None
    ) -> Retrieval:
        total = self.config.injection_budget_tokens if budget_tokens is None else budget_tokens
        out = Retrieval()
        blocks = []
        for name, sys_ in self.systems.items():
            share = self.config.composite_split.get(name, 1.0 / len(self.systems))
            r = sys_.retrieve(instruction, scope=scope, budget_tokens=int(total * share))
            out.items += r.items
            out.scored += r.scored
            if r.block:
                blocks.append(r.block)
        out.block = "\n\n".join(blocks)
        out.tokens = _count(out.block)
        return out

    def observe(self, episode: Episode, retrieval: Retrieval | None = None) -> WriteReport:
        rep = WriteReport()
        for name, sys_ in self.systems.items():
            sub_items = [i for i in (retrieval.items if retrieval else []) if i.type == sys_.type_name]
            r = sys_.observe(episode, Retrieval(items=sub_items))
            rep.merge_in(r)
            rep.extra[f"{name}_counts"] = r.counts()
        self._n_observed += 1
        return rep

    def writer_usage(self) -> dict:
        out = {}
        for name, sys_ in self.systems.items():
            if sys_.writer is not None:
                out[name] = sys_.writer.llm.usage.to_dict()
        return out

    def summary(self) -> dict:
        return {"type": self.type_name, "subsystems": {n: s.summary() for n, s in self.systems.items()}}


def _count(text: str) -> int:
    from .tokens import count_tokens

    return count_tokens(text)


SYSTEMS: dict[str, type[MemorySystem]] = {
    ReflectionSystem.type_name: ReflectionSystem,
    RuleSystem.type_name: RuleSystem,
    SkillSystem.type_name: SkillSystem,
    RawTrajectorySystem.type_name: RawTrajectorySystem,
}


def build_system(
    kind: str,
    llm: LLMClient | None = None,
    config: MemoryConfig | None = None,
    store: MemoryStore | None = None,
) -> MemorySystem:
    """Factory: build_system("reflection", llm) / build_system("all", llm)."""
    config = config or MemoryConfig()
    if kind in ("all", "composite"):
        subs = {
            n: build_system(n, llm, config, MemoryStore(config=config))
            for n in ("reflection", "rule", "skill")
        }
        return CompositeSystem(subs, config=config)
    if kind not in SYSTEMS:
        raise KeyError(f"unknown system {kind!r}; known: {sorted(SYSTEMS) + ['all']}")
    cls = SYSTEMS[kind]
    store = store or MemoryStore(config=config)
    if cls is RawTrajectorySystem:
        return cls(store=store, config=config)
    return cls(llm=llm, store=store, config=config)

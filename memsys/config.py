"""Configuration: content type and write mechanism are two INDEPENDENT axes.

The original design doc attached a different write mechanism to each memory type
(rule got a verification loop, skill got cross-task batch induction, reflection got
neither). That confounds the Memory Content comparison: if skill wins, it could be
because procedural content is better, or merely because its writer saw 25 episodes
at once while the reflection writer saw one.

So every mechanism lives in `WritePolicy` and applies uniformly to all types. The
main Memory Content table should hold the policy FIXED across types
(`WritePolicy.full()` or `.minimal()`), and the per-type asymmetry is available as
`NATIVE_POLICIES` for a separate write-mechanism ablation.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any


@dataclass
class WritePolicy:
    """Which write mechanisms are enabled. Applies to any memory type."""

    #: per-episode extraction from the current rollouts
    online_write: bool = True
    #: LLM merge when a candidate duplicates an existing entry (else drop it)
    merge_on_duplicate: bool = True
    #: check the entry's `evidence` really occurs in the trajectory
    grounding_check: bool = True
    #: after each episode, judge the entries that were injected (support/refute)
    verify: bool = True
    #: rewrite an entry once counterevidence accumulates
    refine: bool = True
    #: drop entries whose confidence collapses
    delete_on_low_confidence: bool = True
    #: drop entries that get retrieved often but rarely coincide with success
    utility_deletion: bool = True
    #: periodic cross-task consolidation over a buffered batch
    batch_induction: bool = True
    batch_every: int = 25
    #: max entries the writer may append per episode
    n_max: int = 2

    @classmethod
    def full(cls, **kw) -> "WritePolicy":
        """Every mechanism on. Recommended default for the Memory Content table."""
        return cls(**kw)

    @classmethod
    def minimal(cls, **kw) -> "WritePolicy":
        """Online extraction + append/merge only. The other end of the fair comparison."""
        base = dict(
            verify=False,
            refine=False,
            delete_on_low_confidence=False,
            utility_deletion=False,
            batch_induction=False,
        )
        base.update(kw)
        return cls(**base)

    def describe(self) -> str:
        on = [f for f in ("online_write", "merge_on_duplicate", "grounding_check", "verify",
                          "refine", "delete_on_low_confidence", "utility_deletion",
                          "batch_induction") if getattr(self, f)]
        return f"n_max={self.n_max} [{', '.join(on) or 'nothing'}]"


#: The asymmetric per-type mechanisms of the original design doc. Keep ONLY for the
#: write-mechanism ablation -- using this in the Memory Content table is the confound.
NATIVE_POLICIES: dict[str, WritePolicy] = {
    "reflection": WritePolicy.minimal(n_max=3),
    "rule": WritePolicy.minimal(n_max=2, verify=True, refine=True, delete_on_low_confidence=True),
    "skill": WritePolicy.minimal(n_max=1, batch_induction=True, batch_every=25, utility_deletion=True),
}


@dataclass
class MemoryConfig:
    # ---- write mechanism (uniform unless policy_by_type overrides it) ----
    policy: WritePolicy = field(default_factory=WritePolicy)
    policy_by_type: dict[str, WritePolicy] = field(default_factory=dict)

    # ---- injection (design doc 1.1) ----
    injection_budget_tokens: int = 1500  # B; 2500 for AppWorld
    equal_item_count: int | None = None  # set to e.g. 3 for the equal-#items ablation

    # ---- retrieval (5.1) ----
    k_pool: int = 10
    utility_lambda: float = 0.0  # score = cos + lambda * utility
    scope_filter_keys: tuple[str, ...] = ("env",)  # hard filter keys (5.2)
    retrieve_during_evolving: bool = True  # (5.3)

    # ---- capacity (5.4) ----
    max_items: int = 200  # 300 for AppWorld

    # ---- dedup thresholds ----
    # Per-type because retrieval keys differ in length and vocabulary; calibrate these
    # so the observed duplicate RATE is comparable across types, not the raw number.
    tau_dup: dict[str, float] = field(
        default_factory=lambda: {"reflection": 0.85, "rule": 0.88, "skill": 0.90, "raw": 0.98}
    )

    # ---- verification / refinement thresholds (uniform across types) ----
    refute_threshold: int = 2
    refine_confidence: float = 0.5
    delete_confidence: float = 0.3
    delete_min_observations: int = 4

    # ---- utility-based deletion (uniform across types) ----
    utility_delete_min_retrieved: int = 5
    utility_delete_success_rate: float = 0.20

    # ---- batch induction clustering (uniform across types) ----
    cluster_key: str = "task_type"  # scope key to cluster by, when present
    cluster_sim: float = 0.60  # greedy embedding threshold otherwise

    # ---- grounding ----
    evidence_check: str = "fuzzy"  # "strict" | "fuzzy" | "off"

    # ---- skill-only extra, OFF by default: it has no analogue for the other types,
    # so enabling it reintroduces an asymmetry (needs episode.meta["failed_skill_steps"]).
    skill_step_refinement: bool = False
    skill_failed_step_threshold: int = 2

    # ---- composite "with all" (5.5) ----
    composite_split: dict[str, float] = field(
        default_factory=lambda: {"reflection": 0.3, "rule": 0.2, "skill": 0.5}
    )

    # ---- rendering ----
    render_evidence: bool = False  # 2.4: evidence is provenance, not injected context
    max_rollout_tokens: int = 1200  # per-rollout budget when shown to the writer

    seed: int = 0

    # -- helpers --
    def policy_for(self, type_name: str) -> WritePolicy:
        return self.policy_by_type.get(type_name, self.policy)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def replace(self, **kw) -> "MemoryConfig":
        return dataclasses.replace(self, **kw)

    def uniform(self, policy: WritePolicy) -> "MemoryConfig":
        """Same mechanism for every type -- the fair setting."""
        return dataclasses.replace(self, policy=policy, policy_by_type={})

    def native(self) -> "MemoryConfig":
        """The design doc's original per-type mechanisms (ablation only)."""
        return dataclasses.replace(self, policy_by_type=dict(NATIVE_POLICIES))

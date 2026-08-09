"""Memory writers: the LLM-facing half of each memory system.

Every mechanism -- online extraction, merge, verification, refinement, cross-task
batch induction -- is implemented ONCE on `BaseWriter` and is available to every
memory type. Subclasses supply only what is intrinsic to their content format: the
extraction instruction, the induction instruction, and the schema.

Writers never touch the store; applying ops is the system's job.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .config import MemoryConfig, WritePolicy
from .episode import ALL_FAILURE, ALL_SUCCESS, Episode, Rollout, evidence_supported
from .item import MemoryItem
from .llm import LLMClient
from .schemas import ContentType, ReflectionContent, RuleContent, SkillContent, get_type

APPEND, REVISE, DELETE = "APPEND", "REVISE", "DELETE"

# verdicts of the verification pass (type-neutral)
NOT_APPLICABLE = "not_applicable"
FOLLOWED_OK = "followed_success"
FOLLOWED_BAD = "followed_failure"
VIOLATED = "violated"
VERDICTS = (NOT_APPLICABLE, FOLLOWED_OK, FOLLOWED_BAD, VIOLATED)


@dataclass
class WriteOp:
    op: str
    content: dict[str, Any] | None = None
    target_id: str | None = None
    reason: str = ""

    def to_dict(self) -> dict:
        return {"op": self.op, "target_id": self.target_id, "reason": self.reason}


@dataclass
class Proposal:
    ops: list[WriteOp] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)  # [{"content":..., "errors":[...]}]
    raw: str = ""


@dataclass
class Verdict:
    item_id: str
    verdict: str
    reason: str = ""


# --------------------------------------------------------------------------
# JSON extraction
# --------------------------------------------------------------------------
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json(text: str) -> Any:
    """Pull a JSON value out of a model response, tolerating fences and prose."""
    if not text:
        return None
    for candidate in [text] + _FENCE.findall(text):
        candidate = candidate.strip()
        try:
            return json.loads(candidate)
        except Exception:
            pass
    for opener, closer in (("{", "}"), ("[", "]")):
        i, j = text.find(opener), text.rfind(closer)
        if 0 <= i < j:
            try:
                return json.loads(text[i : j + 1])
            except Exception:
                continue
    return None


def parse_ops(text: str) -> list[WriteOp]:
    data = extract_json(text)
    if data is None:
        return []
    if isinstance(data, dict):
        raw_ops = data.get("ops")
        if raw_ops is None:
            raw_ops = [{"op": APPEND, "content": data}] if data else []
    elif isinstance(data, list):
        raw_ops = data
    else:
        return []

    ops: list[WriteOp] = []
    for o in raw_ops:
        if not isinstance(o, dict):
            continue
        name = str(o.get("op", APPEND)).upper()
        if name not in (APPEND, REVISE, DELETE):
            name = APPEND
        content = o.get("content")
        if content is None and name == APPEND:
            content = {k: v for k, v in o.items() if k not in ("op", "target_id", "reason")}
        ops.append(
            WriteOp(
                op=name,
                content=content if isinstance(content, dict) else None,
                target_id=o.get("target_id") or None,
                reason=str(o.get("reason", "")),
            )
        )
    return ops


def build_prompt(
    role: str, inputs: str, task: str, schema: str, constraints: list[str], output: str
) -> str:
    cons = "\n".join(f"  - {c}" for c in constraints)
    return (
        f"[ROLE]\n{role}\n\n"
        f"[INPUT]\n{inputs}\n\n"
        f"[TASK]\n{task}\n\n"
        f"[SCHEMA]\n{schema}\n\n"
        f"[CONSTRAINTS]\n{cons}\n\n"
        f"[OUTPUT]\n{output}"
    )


OUTPUT_SPEC = (
    'Reply with JSON only, no prose:\n'
    '{"ops": [{"op": "APPEND"|"REVISE"|"DELETE", "target_id": "<id, REVISE/DELETE only>", '
    '"content": {<schema fields>}, "reason": "<short>"}]}\n'
    'An empty list is a valid and often correct answer.'
)

ENTRIES_HEADER = "MEMORY ENTRIES THAT WERE SHOWN TO THE AGENT:"


# --------------------------------------------------------------------------
# Base writer -- owns every mechanism
# --------------------------------------------------------------------------
class BaseWriter:
    type_name: str = ""
    #: what this memory type is called in prompts
    noun: str = "memory entry"
    noun_plural: str = "memory entries"

    def __init__(self, llm: LLMClient, config: MemoryConfig | None = None):
        self.llm = llm
        self.config = config or MemoryConfig()

    # ------------------------------------------------------------- accessors
    @property
    def spec(self) -> type[ContentType]:
        return get_type(self.type_name)

    @property
    def policy(self) -> WritePolicy:
        return self.config.policy_for(self.type_name)

    @property
    def n_max(self) -> int:
        return self.policy.n_max

    # ------------------------------------------------------- shared prompting
    def _role(self, episode: Episode) -> str:
        env = episode.scope.get("env", "the environment")
        return (
            f"You are a memory writer for an agent operating in {env}. You distill reusable "
            f"experience from the agent's own rollouts into {self.type_name} memory."
        )

    def select_rollouts(self, episode: Episode) -> list[Rollout]:
        """Identical rollout selection for every memory type.

        The writers must see the SAME evidence, otherwise the Memory Content
        comparison measures input differences rather than content differences.
        """
        o = episode.outcome()
        if o == ALL_SUCCESS:
            r = episode.best_success()
            return [r] if r else []
        if o == ALL_FAILURE:
            r = episode.worst_failure()
            return [r] if r else []
        return [r for r in (episode.best_success(), episode.worst_failure()) if r]

    def _render_memory(self, retrieved: list[MemoryItem]) -> str:
        if not retrieved:
            return "EXISTING MEMORY: (empty)"
        lines = ["EXISTING MEMORY (id -> entry):"]
        for it in retrieved:
            lines.append(f"  {it.id}: {it.render(verbose=True)}")
        return "\n".join(lines)

    def _render_rollouts(self, episode: Episode, rollouts: list[Rollout]) -> str:
        return (
            f"TASK: {episode.instruction}\n"
            f"SCOPE: {json.dumps(episode.scope, ensure_ascii=False)}\n\n"
            + episode.render(rollouts, max_tokens_each=self.config.max_rollout_tokens)
        )

    def _inputs(self, episode: Episode, retrieved: list[MemoryItem]) -> str:
        return (
            self._render_rollouts(episode, self.select_rollouts(episode))
            + "\n\n"
            + self._render_memory(retrieved)
        )

    def _task(self, episode: Episode) -> str:  # per-type extraction instruction
        raise NotImplementedError

    def _induce_task(self, n: int) -> str:  # per-type induction instruction
        raise NotImplementedError

    def _constraints(self, episode: Episode) -> list[str]:
        base = [
            "Entries must transfer to other task instances; never encode one-off instance names.",
            "Do not restate anything already present in EXISTING MEMORY. If an existing entry is "
            "wrong, emit a REVISE op with its target_id instead of a near-duplicate APPEND.",
            f"Emit at most {self.n_max} entries. Prefer fewer. An empty list is correct when the "
            "rollouts contain nothing transferable.",
        ]
        if self.spec.grounding_field:
            base.append(
                f"`{self.spec.grounding_field}` must be copied VERBATIM from the trajectory above. "
                "Entries whose evidence cannot be found in the trajectory are discarded."
            )
        return base + self._type_constraints(episode)

    def _type_constraints(self, episode: Episode) -> list[str]:
        return []

    # ------------------------------------------------- (1) online extraction
    def propose(self, episode: Episode, retrieved: list[MemoryItem]) -> Proposal:
        user = build_prompt(
            role=self._role(episode),
            inputs=self._inputs(episode, retrieved),
            task=self._task(episode),
            schema=self.spec.schema_block(),
            constraints=self._constraints(episode),
            output=OUTPUT_SPEC,
        )
        resp = self.llm.complete(system=self._role(episode), user=user, tag=f"{self.type_name}.write")
        ops = parse_ops(resp.text)[: self.n_max + 2]  # slack for REVISE/DELETE ops
        return self.validate(ops, episode, raw=resp.text)

    # --------------------------------------------------------- (2) validation
    def validate(
        self,
        ops: list[WriteOp],
        episode: Episode,
        raw: str = "",
        haystack: str | None = None,
        n_max: int | None = None,
    ) -> Proposal:
        cap = self.n_max if n_max is None else n_max
        hay = episode.raw_text() if haystack is None else haystack
        good, bad = [], []
        n_append = 0
        for op in ops:
            if op.op == DELETE:
                if op.target_id:
                    good.append(op)
                else:
                    bad.append({"content": None, "errors": ["DELETE without target_id"]})
                continue
            if op.content is None:
                bad.append({"content": None, "errors": [f"{op.op} without content"]})
                continue
            content = {k: v for k, v in op.content.items() if k in self.spec.writer_fields}
            content = self.spec.normalize(content)
            errors = self.spec.validate(content) + self._grounding_errors(content, hay)
            if op.op == REVISE and not op.target_id:
                errors.append("REVISE without target_id")
            if op.op == APPEND:
                if n_append >= cap:
                    errors.append(f"exceeds n_max={cap}")
                else:
                    n_append += 1
            if errors:
                bad.append({"content": content, "errors": errors})
                continue
            op.content = content
            good.append(op)
        return Proposal(ops=good, rejected=bad, raw=raw)

    def _grounding_errors(self, content: dict, haystack: str) -> list[str]:
        """The same hallucination guard for every type (policy.grounding_check)."""
        gf = self.spec.grounding_field
        if not gf or not self.policy.grounding_check:
            return []
        ev = content.get(gf, "")
        if ev and not evidence_supported(ev, haystack, mode=self.config.evidence_check):
            return [f"{gf} not found in trajectory (hallucination guard)"]
        return []

    # -------------------------------------------------------------- (3) merge
    def merge(self, existing: MemoryItem, incoming: dict, episode: Episode) -> dict | None:
        user = build_prompt(
            role=self._role(episode),
            inputs=(
                f"ENTRY A (already in memory):\n{json.dumps(existing.content, ensure_ascii=False, indent=2)}\n\n"
                f"ENTRY B (new candidate):\n{json.dumps(incoming, ensure_ascii=False, indent=2)}"
            ),
            task=(
                "A and B overlap. Produce ONE merged entry that covers both: keep the more general "
                "formulation, preserve the union of the evidence/conditions, and drop redundancy. "
                "Do not invent anything not present in A or B."
            ),
            schema=self.spec.schema_block(),
            constraints=["Output exactly one merged entry.", "Never widen a condition beyond what A and B support."],
            output='Reply with JSON only: {"content": {<schema fields>}}',
        )
        resp = self.llm.complete(system=self._role(episode), user=user, tag=f"{self.type_name}.merge")
        data = extract_json(resp.text) or {}
        content = data.get("content") if isinstance(data, dict) else None
        if not isinstance(content, dict):
            content = data if isinstance(data, dict) else None
        if not isinstance(content, dict):
            return None
        content = self.spec.normalize({k: v for k, v in content.items() if k in self.spec.writer_fields})
        # The merge prompt contains no trajectory, so any `evidence` the model writes here
        # would be ungrounded. Carry A's (or B's) over instead of trusting the model.
        gf = self.spec.grounding_field
        if gf:
            content[gf] = existing.content.get(gf) or incoming.get(gf, "")
        return content if not self.spec.validate(content) else None

    # ------------------------------------------------------- (4) verification
    def judge(self, episode: Episode, items: list[MemoryItem]) -> list[Verdict]:
        """Did each injected entry apply, was it followed, did it help?

        Type-neutral by construction: the entries are rendered through their own
        schema, the question is the same for lessons, rules and procedures.
        """
        if not items:
            return []
        listing = "\n".join(f"  {it.id}: {it.render()}" for it in items)
        user = build_prompt(
            role=self._role(episode),
            inputs=self._render_rollouts(episode, self.select_rollouts(episode))
            + f"\n\n{ENTRIES_HEADER}\n{listing}",
            task=(
                f"For each {self.noun} decide, from the trajectory alone:\n"
                f"  {NOT_APPLICABLE}   - its condition never arose in these rollouts\n"
                f"  {FOLLOWED_OK}      - it applied, the agent acted consistently with it, task succeeded\n"
                f"  {FOLLOWED_BAD}     - it applied, the agent acted consistently with it, and it led to "
                "or failed to prevent the failure\n"
                f"  {VIOLATED}         - it applied and the agent did NOT act consistently with it\n"
                "Judge only what the trajectory shows. Default to not_applicable when unsure."
            ),
            schema="item_id: str\nverdict: str\nreason: str  # <=1 sentence, cite the step number",
            # Do not repeat ENTRIES_HEADER here: the listing must be the only occurrence
            # so the section can be located unambiguously.
            constraints=["Return exactly one verdict per entry listed above, and no others."],
            output='Reply with JSON only: {"verdicts": [{"item_id": ..., "verdict": ..., "reason": ...}]}',
        )
        resp = self.llm.complete(system=self._role(episode), user=user, tag=f"{self.type_name}.judge")
        data = extract_json(resp.text) or {}
        rows = data.get("verdicts", []) if isinstance(data, dict) else data
        valid = {it.id for it in items}
        out = []
        for row in rows if isinstance(rows, list) else []:
            if not isinstance(row, dict):
                continue
            iid = row.get("item_id") or row.get("rule_id") or row.get("id")
            v = str(row.get("verdict", "")).strip()
            if iid in valid and v in VERDICTS:
                out.append(Verdict(iid, v, str(row.get("reason", ""))))
        return out

    # --------------------------------------------------------- (5) refinement
    def refine(self, item: MemoryItem, counterevidence: list[str], episode: Episode) -> WriteOp | None:
        ce = "\n".join(f"  - {c}" for c in counterevidence) or "  (none recorded)"
        user = build_prompt(
            role=self._role(episode),
            inputs=(
                f"ENTRY (id={item.id}, support={item.support}, refute={item.refute}, "
                f"confidence={item.confidence():.2f}):\n  {item.render()}\n\nCOUNTEREVIDENCE:\n{ce}"
            ),
            task=(
                f"This {self.noun} has been contradicted. Choose exactly one:\n"
                "  (a) narrow its applicability condition so the counterevidence no longer falls under it\n"
                "  (b) add or extend an exception / fallback that excludes those cases\n"
                "  (c) declare it invalid\n"
                "Pick (c) only if no narrowing keeps it useful."
            ),
            schema=self.spec.schema_block(),
            constraints=[
                "Do not weaken the content into vague advice; narrow the scope instead.",
                "Change as little as possible beyond what the counterevidence forces.",
            ],
            output='Reply with JSON only: {"decision": "revise"|"invalid", "content": {<schema fields>}, "reason": "..."}',
        )
        resp = self.llm.complete(system=self._role(episode), user=user, tag=f"{self.type_name}.refine")
        data = extract_json(resp.text)
        if not isinstance(data, dict):
            return None
        decision = str(data.get("decision", "")).lower()
        if decision.startswith("invalid") or decision == "delete":
            return WriteOp(op=DELETE, target_id=item.id, reason=str(data.get("reason", "refuted")))
        content = data.get("content")
        if not isinstance(content, dict):
            return None
        content = self.spec.normalize({k: v for k, v in content.items() if k in self.spec.writer_fields})
        # Same reasoning as merge(): this prompt has no trajectory, so the refined entry
        # inherits the original evidence rather than whatever the model invents.
        gf = self.spec.grounding_field
        if gf:
            content[gf] = item.content.get(gf, "")
        if self.spec.validate(content):
            return None
        return WriteOp(op=REVISE, target_id=item.id, content=content, reason=str(data.get("reason", "refined")))

    # ------------------------------------------------- (6) batch induction
    def induce(
        self, episodes: list[Episode], existing: list[MemoryItem], cluster_name: str = ""
    ) -> Proposal:
        """Cross-task consolidation over a buffered batch. Available to every type."""
        if not episodes:
            return Proposal()
        anchor = episodes[0]
        blocks = [self._render_rollouts(ep, self.select_rollouts(ep)) for ep in episodes]
        cap = max(self.n_max, 4)  # consolidation legitimately touches more entries
        user = build_prompt(
            role=self._role(anchor),
            inputs=(
                f"TASK CLUSTER: {cluster_name or 'unnamed'} ({len(episodes)} tasks)\n\n"
                + "\n\n====\n\n".join(blocks)
                + "\n\n"
                + self._render_memory(existing)
            ),
            task=self._induce_task(len(episodes)),
            schema=self.spec.schema_block(),
            constraints=[
                c for c in self._constraints(anchor) if not c.startswith("Emit at most")
            ]
            + [f"Emit at most {cap} ops."],
            output=OUTPUT_SPEC,
        )
        resp = self.llm.complete(system=self._role(anchor), user=user, tag=f"{self.type_name}.induce")
        haystack = "\n".join(ep.raw_text() for ep in episodes)
        return self.validate(parse_ops(resp.text), anchor, raw=resp.text, haystack=haystack, n_max=cap)


# --------------------------------------------------------------------------
# Reflection (design doc 2)
# --------------------------------------------------------------------------
_REFLECTION_MODES = {
    "from_failure": (
        "The rollout below FAILED. Locate the FIRST step that made the failure unrecoverable "
        "(not the last step where it became visible) and write the transferable lesson for it."
    ),
    "from_success": (
        "The rollout below SUCCEEDED. Write down only the NON-TRIVIAL decisions: things a "
        "competent agent could plausibly have gotten wrong. Skip anything obvious from the task text."
    ),
    "from_contrast": (
        "One rollout succeeded and one failed on the SAME task. Align them, find the FIRST step "
        "where they diverge, and explain what makes the successful branch correct there."
    ),
}


class ReflectionWriter(BaseWriter):
    type_name = ReflectionContent.name
    noun = "lesson"
    noun_plural = "lessons"

    def mode(self, episode: Episode) -> str:
        o = episode.outcome()
        return {ALL_SUCCESS: "from_success", ALL_FAILURE: "from_failure"}.get(o, "from_contrast")

    def _task(self, episode: Episode) -> str:
        m = self.mode(episode)
        return f"{_REFLECTION_MODES[m]}\nSet outcome_tag to \"{m}\" on every entry you emit."

    def _induce_task(self, n: int) -> str:
        return (
            f"Consolidate lessons across all {n} tasks in this batch, not from any single rollout:\n"
            "  1. where several tasks taught the same thing in instance-specific words, emit ONE "
            "general lesson (REVISE an existing entry if it already covers it)\n"
            "  2. mine the FAILED rollouts for lessons only visible across tasks: recurring failure "
            "modes that look like one-offs in isolation\n"
            "  3. DELETE existing lessons this batch contradicts"
        )

    def _type_constraints(self, episode: Episode) -> list[str]:
        return ["`lesson` must be imperative and actionable, not a description of what happened."]


# --------------------------------------------------------------------------
# Rule (design doc 3)
# --------------------------------------------------------------------------
class RuleWriter(BaseWriter):
    type_name = RuleContent.name
    noun = "rule"
    noun_plural = "rules"

    def _task(self, episode: Episode) -> str:
        return (
            "Abstract the rollouts into atomic operating principles of the form "
            "`when <trigger> -> <do/avoid directive>`.\n"
            "Before emitting each rule, self-check its trigger: estimate on what fraction of steps "
            "the trigger would fire. If it is close to 100% the trigger is too generic -- rewrite it "
            "or drop the rule."
        )

    def _induce_task(self, n: int) -> str:
        return (
            f"Induce rules that hold across all {n} tasks in this batch, not from any single rollout:\n"
            "  1. prefer a rule supported by several tasks over one read off a single trajectory\n"
            "  2. where the batch shows a rule's trigger firing in cases it should not cover, REVISE "
            "that rule to narrow the trigger or add an exception\n"
            "  3. DELETE existing rules this batch contradicts"
        )

    def _type_constraints(self, episode: Episode) -> list[str]:
        return [
            "`trigger` must be a predicate the agent can evaluate ON THE SPOT from the current "
            "observation or the last action's feedback. Conditions like \"when the task is hard\" "
            "are invalid.",
            "`directive` must name an action in the environment's action space, or a constraint on "
            "choosing among actions. Exactly ONE action per rule -- split compound advice.",
        ]


# --------------------------------------------------------------------------
# Procedural skill (design doc 4)
# --------------------------------------------------------------------------
class SkillWriter(BaseWriter):
    type_name = SkillContent.name
    noun = "procedure"
    noun_plural = "procedures"

    def _task(self, episode: Episode) -> str:
        return (
            "Extract at most one reusable procedure (SOP) from the rollouts: the ordered steps, what "
            "must hold before starting, how to verify each key step, and what to do when "
            "verification fails. Use the failed rollout, if present, to fill in `verification` and "
            "`fallback`."
        )

    def _induce_task(self, n: int) -> str:
        return (
            f"Induce procedures across all {n} tasks in this batch, not from any single rollout:\n"
            "  1. extract the common skeleton shared by the SUCCESSFUL rollouts -> `steps`\n"
            "  2. mine the FAILED rollouts for what goes wrong and where -> `verification` and "
            "`fallback` (this is the main value of the failures)\n"
            "  3. where an existing procedure already covers a skeleton, REVISE it instead of "
            "appending a near-duplicate"
        )

    def _type_constraints(self, episode: Episode) -> list[str]:
        return [
            "Steps must use placeholders (<obj>, <recep>, <entity_id>) instead of the concrete "
            "instance names seen in these rollouts.",
            "3 to 12 steps. If the procedure needs more, it is two skills -- emit the more reusable one.",
            "Every entry needs at least one `verification` item; a procedure with no success check "
            "is not reusable.",
        ]

    def propose(self, episode: Episode, retrieved: list[MemoryItem]) -> Proposal:
        # A procedure needs at least one working path; this is a property of procedural
        # content, not a mechanism, and the Memory Source study is designed to expose it.
        if not episode.successes():
            return Proposal(ops=[], rejected=[], raw="")
        return super().propose(episode, retrieved)

    # -- skill-only extra, gated by config.skill_step_refinement (off by default) --
    def refine_step(
        self, skill: MemoryItem, failed_step_idx: int, evidence: list[str], episode: Episode
    ) -> WriteOp | None:
        steps = skill.content.get("steps", [])
        step_txt = steps[failed_step_idx] if 0 <= failed_step_idx < len(steps) else "<unknown>"
        ev = "\n".join(f"  - {e}" for e in evidence) or "  (none recorded)"
        user = build_prompt(
            role=self._role(episode),
            inputs=(
                f"SKILL (id={skill.id}):\n{skill.render()}\n\n"
                f"FAILING STEP #{failed_step_idx + 1}: {step_txt}\n\nFAILURE EVIDENCE:\n{ev}"
            ),
            task=(
                "This step has failed repeatedly. Rewrite the skill so the failure is handled: either "
                "reformulate that step, or add a `verification` and `fallback` entry for it. Keep the "
                "rest of the procedure unchanged."
            ),
            schema=self.spec.schema_block(),
            constraints=["Change as little as possible outside the failing step."],
            output='Reply with JSON only: {"content": {<schema fields>}, "reason": "..."}',
        )
        resp = self.llm.complete(system=self._role(episode), user=user, tag="skill.refine_step")
        data = extract_json(resp.text)
        if not isinstance(data, dict):
            return None
        content = data.get("content") if isinstance(data.get("content"), dict) else data
        content = self.spec.normalize({k: v for k, v in content.items() if k in self.spec.writer_fields})
        content["evidence"] = skill.content.get("evidence", "")
        if self.spec.validate(content):
            return None
        return WriteOp(op=REVISE, target_id=skill.id, content=content, reason=str(data.get("reason", "step refined")))


WRITERS: dict[str, type[BaseWriter]] = {
    ReflectionWriter.type_name: ReflectionWriter,
    RuleWriter.type_name: RuleWriter,
    SkillWriter.type_name: SkillWriter,
}

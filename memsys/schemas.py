"""Content schemas for the three memory systems (+ the raw-trajectory baseline).

A ContentType owns ONLY what is intrinsic to the content format: its fields, their
validity, the text that gets embedded, and how it renders into the agent prompt.
Write mechanisms (verification, refinement, batch induction) live in WritePolicy and
apply to all types alike -- see config.py for why.

All three real types carry an `evidence` field: the trajectory snippet that motivated
the entry. It is provenance, checked against the real trajectory by the same
grounding guard for every type, and it is NOT injected into the agent prompt, so it
costs nothing against the injection budget.
"""

from __future__ import annotations

from typing import Any

from .tokens import count_tokens

OUTCOME_TAGS = ("from_success", "from_failure", "from_contrast")

#: Cap on the `evidence` field, which is the grounding anchor rather than
#: injected context (`MemoryConfig.render_evidence` is False, so raising this
#: gives no arm more context at inference -- it only changes how much the writer
#: may quote to prove an entry came from a real trajectory).
#:
#: 80 is calibrated for ALFWorld's and WebShop's evidence, which is a sentence of
#: observation text. AppWorld's is *Python code and API documentation*, where a
#: single faithful quote runs past 80 tokens routinely, and the cap then rejects
#: most of what the writer produces -- unevenly across content types, because a
#: type that writes rarely (skill, which only writes from successful episodes)
#: can be rejected down to an empty store while a chatty one still accumulates.
#: That is a benchmark-calibration artifact masquerading as a content-type
#: result, so it is settable per run and recorded in config.json.
MAX_EVIDENCE_TOKENS = 80


def set_max_evidence_tokens(n: int) -> None:
    """Set the evidence cap for this process. Apply it to every arm in a run."""
    global MAX_EVIDENCE_TOKENS
    if n <= 0:
        raise ValueError("max evidence tokens must be positive")
    MAX_EVIDENCE_TOKENS = int(n)


def _s(v: Any) -> str:
    return "" if v is None else str(v).strip()


def _slist(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        v = [v]
    return [str(x).strip() for x in v if _s(x)]


class ContentType:
    name: str = ""
    #: token cap on the INJECTED rendering (what competes for the budget)
    max_tokens: int = 200
    #: fields the writer is allowed to set; anything else it emits is dropped
    writer_fields: tuple[str, ...] = ()
    #: field checked against the trajectory by the grounding guard, if any
    grounding_field: str | None = "evidence"

    @classmethod
    def normalize(cls, content: dict) -> dict:
        raise NotImplementedError

    @classmethod
    def validate(cls, content: dict) -> list[str]:
        raise NotImplementedError

    @classmethod
    def retrieval_key(cls, content: dict) -> str:
        raise NotImplementedError

    @classmethod
    def render(cls, content: dict, **kw) -> str:
        raise NotImplementedError

    @classmethod
    def block_header(cls) -> str:
        return ""

    @classmethod
    def schema_block(cls) -> str:
        """The YAML skeleton shown to the writer in its prompt."""
        raise NotImplementedError

    # -- shared checks, identical for every type --
    @classmethod
    def _common_checks(cls, content: dict) -> list[str]:
        errs = []
        n = count_tokens(cls.render(content))  # injected form only
        if n > cls.max_tokens:
            errs.append(f"content too long: {n} tokens > {cls.max_tokens}")
        gf = cls.grounding_field
        if gf is not None:
            ev = content.get(gf)
            if not ev:
                errs.append(f"missing field: {gf}")
            elif count_tokens(str(ev)) > MAX_EVIDENCE_TOKENS:
                errs.append(f"{gf} too long: > {MAX_EVIDENCE_TOKENS} tokens")
        return errs

    @classmethod
    def evidence_line(cls, content: dict) -> str:
        gf = cls.grounding_field
        return _s(content.get(gf)) if gf else ""


# --------------------------------------------------------------------------
# Reflection (design doc 2)
# --------------------------------------------------------------------------
class ReflectionContent(ContentType):
    name = "reflection"
    max_tokens = 120
    writer_fields = ("situation", "lesson", "rationale", "evidence", "outcome_tag")

    @classmethod
    def normalize(cls, content: dict) -> dict:
        return {
            "situation": _s(content.get("situation")),
            "lesson": _s(content.get("lesson")),
            "rationale": _s(content.get("rationale")),
            "evidence": _s(content.get("evidence")),
            "outcome_tag": _s(content.get("outcome_tag")) or "from_success",
        }

    @classmethod
    def validate(cls, content: dict) -> list[str]:
        errs = []
        for f in ("situation", "lesson", "rationale"):
            if not content.get(f):
                errs.append(f"missing field: {f}")
        if content.get("outcome_tag") not in OUTCOME_TAGS:
            errs.append(f"outcome_tag must be one of {OUTCOME_TAGS}")
        return errs + cls._common_checks(content)

    @classmethod
    def retrieval_key(cls, content: dict) -> str:
        return f"{content.get('situation','')} {content.get('lesson','')}".strip()

    @classmethod
    def render(cls, content: dict, verbose: bool = False, **kw) -> str:
        out = f"[when {content.get('situation','')}] {content.get('lesson','')}"
        if content.get("rationale"):
            out += f"\n   why: {content['rationale']}"
        if verbose and content.get("evidence"):
            out += f"\n   evidence: {content['evidence']}"
        return out

    @classmethod
    def block_header(cls) -> str:
        return "## Past experience (lessons)"

    @classmethod
    def schema_block(cls) -> str:
        return (
            "situation:   str   # the situation this applies to, 1 sentence\n"
            "lesson:      str   # transferable lesson, imperative, 1-2 sentences\n"
            "rationale:   str   # why it holds (causal), 1-2 sentences\n"
            "evidence:    str   # a VERBATIM snippet from the trajectory, <=2 lines\n"
            "outcome_tag: str   # from_success | from_failure | from_contrast"
        )


# --------------------------------------------------------------------------
# Rule (design doc 3)
# --------------------------------------------------------------------------
class RuleContent(ContentType):
    name = "rule"
    max_tokens = 60  # 50-token budget for trigger+directive+exception, plus slack
    writer_fields = ("trigger", "directive", "polarity", "exception", "evidence")

    @classmethod
    def normalize(cls, content: dict) -> dict:
        pol = _s(content.get("polarity")).lower() or "do"
        if pol not in ("do", "avoid"):
            pol = "avoid" if pol.startswith(("dont", "don't", "never", "no")) else "do"
        return {
            "trigger": _s(content.get("trigger")),
            "directive": _s(content.get("directive")),
            "polarity": pol,
            "exception": _s(content.get("exception")) or None,
            "evidence": _s(content.get("evidence")),
        }

    @classmethod
    def validate(cls, content: dict) -> list[str]:
        errs = []
        if not content.get("trigger"):
            errs.append("missing field: trigger")
        if not content.get("directive"):
            errs.append("missing field: directive")
        # one rule = one directive (design doc 3.1)
        d = content.get("directive", "").lower()
        for conj in (" and then ", "; then ", " then ", " and also "):
            if conj in d:
                errs.append("directive contains multiple actions; split into separate rules")
                break
        return errs + cls._common_checks(content)

    @classmethod
    def retrieval_key(cls, content: dict) -> str:
        return f"{content.get('trigger','')} {content.get('directive','')}".strip()

    @classmethod
    def render(cls, content: dict, verbose: bool = False, **kw) -> str:
        verb = "DO" if content.get("polarity", "do") == "do" else "AVOID"
        out = f"When {content.get('trigger','')}, {verb} {content.get('directive','')}."
        if content.get("exception"):
            out += f" (unless {content['exception']})"
        if verbose and content.get("evidence"):
            out += f"\n   evidence: {content['evidence']}"
        return out

    @classmethod
    def block_header(cls) -> str:
        return "## Rules"

    @classmethod
    def schema_block(cls) -> str:
        return (
            "trigger:   str        # \"the search page shows no item matching all constraints\"\n"
            "                      # MUST be decidable from the current observation\n"
            "directive: str        # exactly ONE action or action-constraint\n"
            "polarity:  str        # do | avoid\n"
            "exception: str|null   # when the rule does NOT apply\n"
            "evidence:  str        # a VERBATIM snippet from the trajectory that motivates it"
        )


# --------------------------------------------------------------------------
# Procedural skill (design doc 4)
# --------------------------------------------------------------------------
class SkillContent(ContentType):
    name = "skill"
    max_tokens = 400
    writer_fields = (
        "name", "trigger", "preconditions", "steps", "verification", "fallback", "evidence",
    )

    @classmethod
    def normalize(cls, content: dict) -> dict:
        return {
            "name": _s(content.get("name")).replace(" ", "_"),
            "trigger": _s(content.get("trigger")),
            "preconditions": _slist(content.get("preconditions")),
            "steps": _slist(content.get("steps")),
            "verification": _slist(content.get("verification")),
            "fallback": _slist(content.get("fallback")),
            "evidence": _s(content.get("evidence")),
        }

    @classmethod
    def validate(cls, content: dict) -> list[str]:
        errs = []
        if not content.get("name"):
            errs.append("missing field: name")
        if not content.get("trigger"):
            errs.append("missing field: trigger")
        n = len(content.get("steps", []))
        if not 3 <= n <= 12:
            errs.append(f"steps must have 3..12 entries, got {n}")
        if not content.get("verification"):
            errs.append("missing field: verification")
        return errs + cls._common_checks(content)

    @classmethod
    def retrieval_key(cls, content: dict) -> str:
        head = " ".join(content.get("steps", [])[:3])
        return f"{content.get('name','')} {content.get('trigger','')} {head}".strip()

    @classmethod
    def render(cls, content: dict, verbose: bool = False, **kw) -> str:
        steps = content.get("steps", [])
        ver = content.get("verification", [])
        fb = content.get("fallback", [])
        aligned_v = len(ver) == len(steps)
        aligned_f = len(fb) == len(steps)

        lines = [f"### {content.get('name','')}", f"Use when: {content.get('trigger','')}"]
        if content.get("preconditions"):
            lines.append("Preconditions: " + "; ".join(content["preconditions"]))
        lines.append("Steps:")
        for i, s in enumerate(steps):
            row = f"  {i+1}. {s}"
            if aligned_v and ver[i]:
                row += f"   [check: {ver[i]}]"
            if aligned_f and fb[i]:
                row += f"   [if fails: {fb[i]}]"
            lines.append(row)
        if ver and not aligned_v:
            lines.append("Verify: " + "; ".join(ver))
        if fb and not aligned_f:
            lines.append("Fallback: " + "; ".join(fb))
        if verbose and content.get("evidence"):
            lines.append(f"   evidence: {content['evidence']}")
        return "\n".join(lines)

    @classmethod
    def block_header(cls) -> str:
        return "## Applicable procedures"

    @classmethod
    def schema_block(cls) -> str:
        return (
            "name:          str        # verb phrase, snake_case\n"
            "trigger:       str        # task-level condition for using this procedure\n"
            "preconditions: [str]      # state that must hold before executing\n"
            "steps:         [str]      # 3..12 ordered steps, use <obj>/<recep> placeholders,\n"
            "                          # never concrete instance names\n"
            "verification:  [str]      # how to confirm each key step succeeded\n"
            "fallback:      [str]      # recovery when verification fails\n"
            "evidence:      str        # a VERBATIM snippet from a trajectory this came from"
        )


# --------------------------------------------------------------------------
# Raw trajectory baseline (design doc 1) - append-only, no LLM writer
# --------------------------------------------------------------------------
class RawContent(ContentType):
    name = "raw"
    max_tokens = 4000
    writer_fields = ("instruction", "trajectory", "reward")
    grounding_field = None  # the item IS the trajectory

    @classmethod
    def normalize(cls, content: dict) -> dict:
        return {
            "instruction": _s(content.get("instruction")),
            "trajectory": _s(content.get("trajectory")),
            "reward": float(content.get("reward", 0.0) or 0.0),
        }

    @classmethod
    def validate(cls, content: dict) -> list[str]:
        errs = []
        if not content.get("instruction"):
            errs.append("missing field: instruction")
        if not content.get("trajectory"):
            errs.append("missing field: trajectory")
        return errs

    @classmethod
    def retrieval_key(cls, content: dict) -> str:
        return content.get("instruction", "")

    @classmethod
    def render(cls, content: dict, verbose: bool = False, **kw) -> str:
        return f"Task: {content.get('instruction','')}\n{content.get('trajectory','')}"

    @classmethod
    def block_header(cls) -> str:
        return "## Similar solved tasks"

    @classmethod
    def schema_block(cls) -> str:
        return "instruction: str\ntrajectory: str\nreward: float"


CONTENT_TYPES: dict[str, type[ContentType]] = {
    ReflectionContent.name: ReflectionContent,
    RuleContent.name: RuleContent,
    SkillContent.name: SkillContent,
    RawContent.name: RawContent,
}


def get_type(name: str) -> type[ContentType]:
    if name not in CONTENT_TYPES:
        raise KeyError(f"unknown memory type {name!r}; known: {sorted(CONTENT_TYPES)}")
    return CONTENT_TYPES[name]


def laplace_confidence(support: int, refute: int) -> float:
    """(support + 1) / (support + refute + 2)  -- design doc 3.3, now used for all types."""
    return (support + 1) / (support + refute + 2)

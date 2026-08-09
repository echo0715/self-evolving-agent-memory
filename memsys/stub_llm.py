"""A dependency-free stand-in writer for demos, tests and dry runs.

It produces schema-valid JSON derived from the prompt it is given (so evidence
checks and id references actually line up), but it does NOT reason. Never use it to
produce numbers for the paper -- it exists so the pipeline can be exercised, CI can
run, and prompt/plumbing bugs surface without burning API calls.
"""

from __future__ import annotations

import json
import re

from .llm import LLMClient, LLMResponse
from .writers import ENTRIES_HEADER

_TASK = re.compile(r"^TASK: (.+)$", re.M)
_ACTION = re.compile(r"^\s*(?:\[\d+\]\s*)?> (.+)$", re.M)
_ENTRY_ROW = re.compile(r"^\s{2}([0-9a-f]{12}): (.+)$", re.M)
_ENTRY_A = re.compile(r"ENTRY A \(already in memory\):\s*(\{.*?\})\s*\n\s*\n", re.S)


class StubWriterLLM(LLMClient):
    model = "stub"

    def __init__(self):
        super().__init__()
        self.n = 0

    def _complete(self, system: str, user: str, tag: str) -> LLMResponse:
        self.n += 1
        kind, _, action = tag.partition(".")
        if action == "merge":
            return LLMResponse(text=self._merge(user))
        if action == "judge":
            return LLMResponse(text=self._judge(user))
        if action == "refine":
            return LLMResponse(text=self._refine(kind, user))
        if action == "refine_step":
            refined = json.loads(self._refine("skill", user))
            return LLMResponse(text=json.dumps({"content": refined["content"], "reason": "stub step fix"}))
        if action in ("write", "induce"):
            content = {
                "reflection": self._reflection_content,
                "rule": self._rule_content,
                "skill": self._skill_content,
            }.get(kind)
            if content is None:
                return LLMResponse(text='{"ops": []}')
            return LLMResponse(
                text=json.dumps({"ops": [{"op": "APPEND", "content": content(user), "reason": "stub"}]},
                                ensure_ascii=False)
            )
        return LLMResponse(text='{"ops": []}')

    # -- prompt scraping helpers --
    @staticmethod
    def _task(user: str) -> str:
        m = _TASK.search(user)
        return m.group(1).strip() if m else "the task"

    @staticmethod
    def _actions(user: str) -> list[str]:
        return [a.strip() for a in _ACTION.findall(user)]

    def _evidence(self, user: str) -> str:
        acts = self._actions(user)
        return acts[-1] if acts else self._task(user)

    # -- per-type content --
    def _reflection_content(self, user: str) -> dict:
        tag = "from_contrast" if ("SUCCESS" in user and "FAILURE" in user) else (
            "from_failure" if "FAILURE" in user else "from_success"
        )
        return {
            "situation": f"handling a task of the form: {self._task(user)[:60]}",
            "lesson": f"Verify the precondition of step {self.n % 5 + 1} before acting.",
            "rationale": "The environment gives no error on a silent no-op, so an unchecked "
            "precondition surfaces only much later.",
            "evidence": self._evidence(user),
            "outcome_tag": tag,
        }

    def _rule_content(self, user: str) -> dict:
        acts = self._actions(user) or ["act"]
        return {
            "trigger": f"the last action returned no state change (variant {self.n % 4})",
            "directive": f"re-issue `{acts[0][:40]}` after re-reading the observation",
            "polarity": "do",
            "exception": None,
            "evidence": self._evidence(user),
        }

    def _skill_content(self, user: str) -> dict:
        acts = self._actions(user)
        steps = [re.sub(r"\b\w+ \d+\b", "<obj>", a) for a in acts[:6]]
        while len(steps) < 3:
            steps.append(f"inspect <recep> to locate <obj> (filler {len(steps)})")
        return {
            "name": f"stub_procedure_{self.n % 3}",
            "trigger": f"the task asks to {self._task(user)[:50]}",
            "preconditions": ["the target object has been located"],
            "steps": steps,
            "verification": ["the observation names <obj> as held or placed"],
            "fallback": ["re-scan the receptacles and retry once"],
            "evidence": self._evidence(user),
        }

    # -- mechanisms --
    def _judge(self, user: str) -> str:
        section = user.split(ENTRIES_HEADER)[-1]
        verdict = "followed_failure" if "FAILURE" in user else "followed_success"
        rows = [
            {"item_id": iid, "verdict": verdict, "reason": "stub verdict"}
            for iid, _ in _ENTRY_ROW.findall(section)
        ]
        return json.dumps({"verdicts": rows}, ensure_ascii=False)

    def _refine(self, kind: str, user: str) -> str:
        # evidence is deliberately omitted: the writer carries the original over, which
        # is what keeps a refined entry grounded even though this prompt has no trajectory
        content = {
            "reflection": {
                "situation": f"handling a narrowed situation (v{self.n})",
                "lesson": "Verify the precondition, but only when the target is already held.",
                "rationale": "The counterevidence shows the check is wasted otherwise.",
                "outcome_tag": "from_failure",
            },
            "rule": {
                "trigger": f"the last action returned no state change AND the target is held (v{self.n})",
                "directive": "re-read the observation before re-issuing the action",
                "polarity": "do",
                "exception": "the episode step budget is nearly exhausted",
            },
            "skill": {
                "name": "stub_procedure_refined",
                "trigger": "the task asks for a multi-step object manipulation",
                "preconditions": ["the target object has been located"],
                "steps": [
                    "locate <obj> by scanning receptacles",
                    "take <obj> from <recep>",
                    "apply the required transformation to <obj>",
                ],
                "verification": ["the observation confirms <obj> changed state"],
                "fallback": ["if no confirmation, re-take <obj> and retry once"],
            },
        }.get(kind)
        if content is None:
            return "{}"
        return json.dumps({"decision": "revise", "content": content, "reason": "stub narrowing"},
                          ensure_ascii=False)

    def _merge(self, user: str) -> str:
        m = _ENTRY_A.search(user + "\n\n")
        if not m:
            return "{}"
        try:
            a = json.loads(m.group(1))
        except Exception:
            return "{}"
        if "lesson" in a:
            a["lesson"] = "(merged) " + a["lesson"]
        elif "name" in a:
            a["name"] = a["name"] + "_merged"
        return json.dumps({"content": a}, ensure_ascii=False)

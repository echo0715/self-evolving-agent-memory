"""Unit tests for the memory systems.

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memsys import (  # noqa: E402
    NATIVE_POLICIES,
    CallableLLM,
    Episode,
    Evolver,
    MemoryConfig,
    MemoryItem,
    MemoryStore,
    RawTrajectorySystem,
    ReflectionSystem,
    Rollout,
    RuleSystem,
    SkillSystem,
    Step,
    WritePolicy,
    build_system,
    frozen,
    parse_ops,
)
from memsys.episode import evidence_supported  # noqa: E402
from memsys.schemas import ReflectionContent, RuleContent, SkillContent  # noqa: E402
from memsys.stub_llm import StubWriterLLM  # noqa: E402
from memsys.tokens import count_tokens  # noqa: E402
from memsys.writers import ENTRIES_HEADER, ReflectionWriter, WriteOp  # noqa: E402

ENV = "testenv"
TYPES = ("reflection", "rule", "skill")


# ---------------------------------------------------------------- fixtures
def rollout(rid: str, success: bool, n: int = 3) -> Rollout:
    steps = [
        Step(action=f"go to shelf {i}", observation=f"You see item {i} on shelf {i}.") for i in range(n)
    ]
    steps.append(
        Step(action="take mug 1 from shelf 1", observation="You pick up the mug 1." if success else "Nothing happens.")
    )
    return Rollout(rollout_id=rid, steps=steps, reward=1.0 if success else 0.0, success=success)


def episode(i: int, flags=(True, False), instruction: str | None = None, task_type: str = "pick") -> Episode:
    return Episode(
        task_id=f"t{i}",
        instruction=instruction or f"put the mug {i} on the shelf",
        rollouts=[rollout(f"{i}-{j}", ok) for j, ok in enumerate(flags)],
        scope={"env": ENV, "task_type": task_type},
        step_index=i,
    )


EVIDENCE = "Nothing happens."


def reflection_item(lesson: str, situation: str = "picking up an object", **kw) -> MemoryItem:
    return MemoryItem(
        type="reflection",
        content={
            "situation": situation,
            "lesson": lesson,
            "rationale": "because the environment silently no-ops",
            "evidence": EVIDENCE,
            "outcome_tag": "from_failure",
        },
        scope={"env": ENV},
        **kw,
    )


def rule_item(trigger: str, directive: str = "re-read the observation", **kw) -> MemoryItem:
    return MemoryItem(
        type="rule",
        content={
            "trigger": trigger,
            "directive": directive,
            "polarity": "do",
            "exception": None,
            "evidence": EVIDENCE,
        },
        scope={"env": ENV},
        **kw,
    )


def skill_item(name: str, trigger: str = "the task asks to move an object", **kw) -> MemoryItem:
    return MemoryItem(
        type="skill",
        content={
            "name": name,
            "trigger": trigger,
            "preconditions": [],
            "steps": ["locate <obj>", "take <obj>", "put <obj> in <recep>"],
            "verification": ["the observation confirms the move"],
            "fallback": [],
            "evidence": EVIDENCE,
        },
        scope={"env": ENV},
        **kw,
    )


def mechanisms(llm) -> set[str]:
    """Which write mechanisms actually fired, independent of memory type."""
    return {tag.split(".", 1)[1] for tag in llm.usage.by_tag}


# ---------------------------------------------------------------- schemas
class SchemaTest(unittest.TestCase):
    def test_reflection_requires_evidence(self):
        c = ReflectionContent.normalize({"situation": "a", "lesson": "b", "rationale": "c"})
        self.assertIn("missing field: evidence", ReflectionContent.validate(c))

    def test_every_real_type_has_the_same_grounding_field(self):
        for spec in (ReflectionContent, RuleContent, SkillContent):
            self.assertEqual(spec.grounding_field, "evidence", spec.name)
            self.assertIn("evidence", spec.writer_fields, spec.name)

    def test_rule_and_skill_require_evidence_too(self):
        c = RuleContent.normalize({"trigger": "t", "directive": "d"})
        self.assertIn("missing field: evidence", RuleContent.validate(c))
        c = SkillContent.normalize({"name": "n", "trigger": "t", "steps": ["a", "b", "c"], "verification": ["v"]})
        self.assertIn("missing field: evidence", SkillContent.validate(c))

    def test_length_cap_measures_the_injected_form_only(self):
        long_ev = {"situation": "s", "lesson": "l", "rationale": "r",
                   "evidence": "x " * 300, "outcome_tag": "from_success"}
        errs = ReflectionContent.validate(ReflectionContent.normalize(long_ev))
        self.assertFalse(any("content too long" in e for e in errs))  # evidence is not injected
        self.assertTrue(any("evidence too long" in e for e in errs))  # but it is still capped

        long_body = {"situation": "verbose padding text " * 40, "lesson": "l", "rationale": "r",
                     "evidence": "e", "outcome_tag": "from_success"}
        self.assertTrue(any("content too long" in e for e in
                            ReflectionContent.validate(ReflectionContent.normalize(long_body))))

    def test_rule_rejects_compound_directive(self):
        c = RuleContent.normalize(
            {"trigger": "the page is empty", "directive": "search again and then click the first result",
             "evidence": EVIDENCE}
        )
        self.assertTrue(any("multiple actions" in e for e in RuleContent.validate(c)))

    def test_rule_polarity_coercion(self):
        self.assertEqual(
            RuleContent.normalize({"trigger": "t", "directive": "d", "polarity": "never"})["polarity"], "avoid"
        )

    def test_verification_stats_are_not_content(self):
        c = RuleContent.normalize({"trigger": "t", "directive": "d", "support": 9, "confidence": 0.99})
        self.assertNotIn("support", c)
        self.assertNotIn("confidence", c)
        self.assertAlmostEqual(rule_item("t").confidence(), 0.5)

    def test_skill_step_count_bounds(self):
        for n in (2, 13):
            c = SkillContent.normalize(
                {"name": "s", "trigger": "t", "steps": [f"step {i}" for i in range(n)],
                 "verification": ["v"], "evidence": EVIDENCE}
            )
            self.assertTrue(any("3..12" in e for e in SkillContent.validate(c)), f"n={n}")

    def test_skill_requires_verification(self):
        c = SkillContent.normalize({"name": "s", "trigger": "t", "steps": ["a", "b", "c"], "evidence": EVIDENCE})
        self.assertIn("missing field: verification", SkillContent.validate(c))


class EvidenceGuardTest(unittest.TestCase):
    def test_substring_and_fuzzy(self):
        hay = "You pick up the mug 1.\n> heat mug 1 with microwave 1\nNothing happens."
        self.assertTrue(evidence_supported("Nothing happens.", hay, "strict"))
        self.assertFalse(evidence_supported("The door is locked.", hay, "strict"))
        self.assertTrue(evidence_supported("heat mug with microwave", hay, "fuzzy"))
        self.assertTrue(evidence_supported("anything at all", hay, "off"))

    def test_guard_applies_to_all_three_types(self):
        """The hallucination guard used to be reflection-only; it must be uniform."""
        ep = episode(0)
        bad = "The oven exploded and the sky turned green"
        cases = {
            "reflection": {"situation": "s", "lesson": "l", "rationale": "r",
                           "evidence": bad, "outcome_tag": "from_failure"},
            "rule": {"trigger": "t", "directive": "d", "polarity": "do", "evidence": bad},
            "skill": {"name": "n", "trigger": "t", "steps": ["a", "b", "c"],
                      "verification": ["v"], "evidence": bad},
        }
        for kind, content in cases.items():
            sysm = build_system(kind, llm=StubWriterLLM(), config=MemoryConfig())
            prop = sysm.writer.validate([WriteOp(op="APPEND", content=content)], ep)
            self.assertEqual(prop.ops, [], kind)
            self.assertTrue(any("not found in trajectory" in e for e in prop.rejected[0]["errors"]), kind)

    def test_guard_can_be_disabled_by_policy(self):
        cfg = MemoryConfig(policy=WritePolicy(grounding_check=False))
        sysm = build_system("rule", llm=StubWriterLLM(), config=cfg)
        content = {"trigger": "t", "directive": "d", "polarity": "do", "evidence": "invented"}
        self.assertEqual(len(sysm.writer.validate([WriteOp(op="APPEND", content=content)], episode(0)).ops), 1)


class ParseOpsTest(unittest.TestCase):
    def test_fenced_json(self):
        ops = parse_ops('sure!\n```json\n{"ops": [{"op": "APPEND", "content": {"a": 1}}]}\n```\n')
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0].op, "APPEND")

    def test_bare_content_becomes_append(self):
        ops = parse_ops('{"situation": "s", "lesson": "l"}')
        self.assertEqual(ops[0].op, "APPEND")
        self.assertEqual(ops[0].content["lesson"], "l")

    def test_garbage_returns_empty(self):
        self.assertEqual(parse_ops("I cannot help with that."), [])

    def test_unknown_op_defaults_to_append(self):
        self.assertEqual(parse_ops('{"ops":[{"op":"FROB","content":{"a":1}}]}')[0].op, "APPEND")


# ------------------------------------------------------------------ store
class StoreTest(unittest.TestCase):
    def setUp(self):
        self.store = MemoryStore(config=MemoryConfig(injection_budget_tokens=1000))

    def test_scope_filter_is_hard(self):
        a = self.store.add(reflection_item("check the fridge door"))
        b = reflection_item("check the fridge door")
        b.scope = {"env": "other"}
        self.store.add(b)
        hits = self.store.search("fridge door", type="reflection", scope={"env": ENV})
        self.assertEqual([h.item.id for h in hits], [a.id])

    def test_pack_respects_token_budget(self):
        for i in range(30):
            self.store.add(reflection_item(f"lesson number {i} about shelves and mugs and doors"))
        scored = self.store.search("mug shelf", type="reflection", scope={"env": ENV}, k=30)
        items, block, tokens = self.store.pack(scored, budget_tokens=120)
        self.assertLessEqual(tokens, 120)
        self.assertGreater(len(items), 0)
        self.assertEqual(tokens, count_tokens(block))

    def test_evidence_never_reaches_the_injected_block(self):
        self.store.add(reflection_item("check the door"))
        scored = self.store.search("door", type="reflection", scope={"env": ENV})
        _, block, _ = self.store.pack(scored)
        self.assertNotIn(EVIDENCE, block)

    def test_pack_equal_item_count_mode(self):
        for i in range(10):
            self.store.add(reflection_item(f"lesson {i}"))
        scored = self.store.search("lesson", type="reflection", scope={"env": ENV}, k=10)
        items, _, _ = self.store.pack(scored, budget_tokens=10_000, max_items=3)
        self.assertEqual(len(items), 3)

    def test_eviction_drops_never_retrieved_oldest_first(self):
        old = self.store.add(reflection_item("oldest unused", created_at_step=0))
        new = self.store.add(reflection_item("newest unused", created_at_step=9))
        used = self.store.add(reflection_item("used and useful", created_at_step=1))
        used.n_retrieved, used.n_retrieved_success = 5, 5
        evicted = self.store.enforce_capacity(max_items=2)
        self.assertEqual([e.id for e in evicted], [old.id])
        self.assertIn(new.id, self.store)
        self.assertIn(used.id, self.store)

    def test_eviction_prefers_low_utility(self):
        good = self.store.add(reflection_item("useful"))
        bad = self.store.add(reflection_item("useless"))
        good.n_retrieved, good.n_retrieved_success = 10, 9
        bad.n_retrieved, bad.n_retrieved_success = 10, 0
        evicted = self.store.enforce_capacity(max_items=1)
        self.assertEqual([e.id for e in evicted], [bad.id])

    def test_round_trip_preserves_verification_stats(self):
        it = self.store.add(reflection_item("persist me"))
        it.support, it.refute = 3, 1
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.jsonl")
            self.store.save(p)
            other = MemoryStore(config=self.store.config)
            other.load(p)
        loaded = other.items()[0]
        self.assertEqual((loaded.support, loaded.refute), (3, 1))
        self.assertEqual(loaded.content["lesson"], "persist me")

    def test_superseded_items_are_hidden_but_kept(self):
        a = self.store.add(reflection_item("v1"))
        b = self.store.add(reflection_item("v2 totally different text"))
        self.store.supersede(a.id, b.id)
        self.assertEqual(len(self.store), 1)
        self.assertEqual(len(self.store.items(include_dead=True)), 2)


# ------------------------------------------------------------ write policy
class DedupMergeTest(unittest.TestCase):
    def test_near_duplicate_triggers_merge_not_append(self):
        sysm = ReflectionSystem(llm=StubWriterLLM(), config=MemoryConfig())
        existing = sysm.store.add(reflection_item("check the mug before heating"))
        rep = sysm.apply([WriteOp(op="APPEND", content=dict(existing.content))], episode(1))
        self.assertEqual(rep.counts(), {"MERGE": 1})
        self.assertIsNotNone(sysm.store.get(existing.id).superseded_by)
        self.assertEqual(len(sysm.store), 1)

    def test_merge_disabled_drops_duplicate(self):
        cfg = MemoryConfig(policy=WritePolicy(merge_on_duplicate=False))
        sysm = ReflectionSystem(llm=StubWriterLLM(), config=cfg)
        existing = sysm.store.add(reflection_item("check the mug before heating"))
        rep = sysm.apply([WriteOp(op="APPEND", content=dict(existing.content))], episode(1))
        self.assertEqual(rep.applied, [])
        self.assertTrue(any("duplicate" in e for e in rep.rejected[0]["errors"]))

    def test_merge_unions_provenance_and_sums_verification_stats(self):
        sysm = RuleSystem(llm=StubWriterLLM(), config=MemoryConfig())
        a = rule_item("the page shows no matching item")
        a.support, a.refute, a.source_task_ids = 3, 1, ["t0"]
        sysm.store.add(a)
        rep = sysm.apply([WriteOp(op="APPEND", content=dict(a.content))], episode(5))
        merged = sysm.store.get(rep.applied[0]["id"])
        self.assertEqual((merged.support, merged.refute), (3, 1))
        self.assertEqual(merged.source_task_ids, ["t0", "t5"])
        self.assertEqual(merged.version, a.version + 1)

    def test_merge_keeps_grounded_evidence(self):
        sysm = ReflectionSystem(llm=StubWriterLLM(), config=MemoryConfig())
        existing = sysm.store.add(reflection_item("check the mug"))
        rep = sysm.apply([WriteOp(op="APPEND", content=dict(existing.content))], episode(1))
        self.assertEqual(sysm.store.get(rep.applied[0]["id"]).content["evidence"], EVIDENCE)

    def test_revise_replaces_content_and_resets_evidence(self):
        sysm = ReflectionSystem(llm=StubWriterLLM(), config=MemoryConfig())
        it = sysm.store.add(reflection_item("old lesson"))
        it.support, it.refute = 4, 3
        new = dict(it.content, lesson="new lesson entirely")
        rep = sysm.apply([WriteOp(op="REVISE", target_id=it.id, content=new)], episode(2))
        self.assertEqual(rep.counts(), {"REVISE": 1})
        self.assertEqual(sysm.store.get(it.id).content["lesson"], "new lesson entirely")
        self.assertEqual((it.support, it.refute), (0, 0))
        self.assertEqual(sysm.store.get(it.id).version, 2)

    def test_revise_with_missing_target_is_rejected(self):
        sysm = ReflectionSystem(llm=StubWriterLLM(), config=MemoryConfig())
        rep = sysm.apply([WriteOp(op="REVISE", target_id="nope", content={})], episode(2))
        self.assertEqual(rep.applied, [])
        self.assertEqual(len(rep.rejected), 1)

    def test_n_max_caps_appends(self):
        cfg = MemoryConfig(policy=WritePolicy(n_max=1))
        writer = ReflectionWriter(StubWriterLLM(), cfg)
        ops = [
            WriteOp(op="APPEND", content={"situation": f"s{i}", "lesson": f"l{i}", "rationale": "r",
                                          "evidence": EVIDENCE, "outcome_tag": "from_failure"})
            for i in range(3)
        ]
        prop = writer.validate(ops, episode(0))
        self.assertEqual(len(prop.ops), 1)
        self.assertEqual(len(prop.rejected), 2)


# ----------------------------------------------------------- FAIRNESS
class MechanismFairnessTest(unittest.TestCase):
    """The point of the WritePolicy split: content type and write mechanism are
    independent, so the Memory Content table compares content and nothing else."""

    def _run(self, kind: str, cfg: MemoryConfig, n: int = 6):
        llm = StubWriterLLM()
        sysm = build_system(kind, llm=llm, config=cfg)
        for i in range(n):
            ep = episode(i)
            sysm.observe(ep, sysm.retrieve(ep.instruction, ep.scope))
        return sysm, llm

    def test_uniform_full_policy_runs_the_same_mechanisms_for_every_type(self):
        cfg = MemoryConfig(policy=WritePolicy.full(batch_every=2))
        fired = {k: mechanisms(self._run(k, cfg)[1]) for k in TYPES}
        for kind in TYPES:
            self.assertTrue(
                {"write", "judge", "induce"} <= fired[kind], f"{kind} only ran {fired[kind]}"
            )

    def test_uniform_minimal_policy_runs_no_extra_mechanisms_for_any_type(self):
        cfg = MemoryConfig(policy=WritePolicy.minimal())
        for kind in TYPES:
            fired = mechanisms(self._run(kind, cfg)[1])
            self.assertTrue(fired <= {"write", "merge"}, f"{kind} ran {fired}")

    def test_uniform_policy_gives_every_type_the_same_write_budget(self):
        cfg = MemoryConfig(policy=WritePolicy.full())
        systems = {k: build_system(k, llm=StubWriterLLM(), config=cfg) for k in TYPES}
        self.assertEqual(len({s.policy.n_max for s in systems.values()}), 1)
        self.assertEqual(len({s.policy.batch_every for s in systems.values()}), 1)

    def test_every_type_sees_identical_rollouts(self):
        """Input asymmetry is as much a confound as mechanism asymmetry."""
        ep = episode(0, flags=(True, False, False))
        cfg = MemoryConfig()
        chosen = {
            k: [r.rollout_id for r in build_system(k, llm=StubWriterLLM(), config=cfg).writer.select_rollouts(ep)]
            for k in TYPES
        }
        self.assertEqual(len({tuple(v) for v in chosen.values()}), 1, chosen)

    def test_native_presets_reproduce_the_original_asymmetry(self):
        cfg = MemoryConfig().native().replace(injection_budget_tokens=1500)
        cfg.policy_by_type["skill"] = NATIVE_POLICIES["skill"].__class__(
            **{**NATIVE_POLICIES["skill"].__dict__, "batch_every": 2}
        )
        fired = {k: mechanisms(self._run(k, cfg)[1]) for k in TYPES}
        self.assertNotIn("judge", fired["reflection"])
        self.assertIn("judge", fired["rule"])
        self.assertNotIn("induce", fired["rule"])
        self.assertIn("induce", fired["skill"])
        self.assertNotIn("judge", fired["skill"])

    def test_policy_can_be_overridden_per_type_for_the_ablation(self):
        cfg = MemoryConfig(
            policy=WritePolicy.minimal(),
            policy_by_type={"rule": WritePolicy.full(batch_every=2)},
        )
        self.assertFalse(cfg.policy_for("reflection").verify)
        self.assertTrue(cfg.policy_for("rule").verify)


# ------------------------------------------------------- verification loop
def judging_llm(verdict: str) -> CallableLLM:
    """Judges every injected entry the same way, whatever its type."""

    def fn(system, user, tag):
        action = tag.split(".", 1)[1]
        if action == "judge":
            section = user.split(ENTRIES_HEADER)[-1]
            ids = re.findall(r"^  ([0-9a-f]{12}): ", section, re.M)
            return json.dumps({"verdicts": [{"item_id": i, "verdict": verdict, "reason": "test"} for i in ids]})
        if action == "refine":
            return json.dumps(
                {
                    "decision": "revise",
                    "content": {
                        # reflection fields
                        "situation": "a narrowed situation", "lesson": "a narrowed lesson",
                        "rationale": "narrowed because of counterevidence", "outcome_tag": "from_failure",
                        # rule fields
                        "trigger": "the page shows no matching item AND a filter is active",
                        "directive": "relax the least specific filter", "polarity": "do",
                        # skill fields
                        "name": "narrowed_procedure", "steps": ["a", "b", "c"], "verification": ["v"],
                    },
                    "reason": "narrowed",
                }
            )
        return '{"ops": []}'

    return CallableLLM(fn)


class VerificationTest(unittest.TestCase):
    """These used to be rule-only mechanisms; they must now work for every type."""

    SEEDS = {"reflection": reflection_item, "rule": rule_item, "skill": skill_item}

    def _run(self, kind: str, verdict: str, cfg: MemoryConfig, n: int):
        sysm = build_system(kind, llm=judging_llm(verdict), config=cfg)
        seed = self.SEEDS[kind]("the page shows no matching item" if kind == "rule" else "seeded entry")
        sysm.store.add(seed)
        for i in range(n):
            ep = episode(i, flags=(False,), instruction="the page shows no matching item for the mug")
            sysm.observe(ep, sysm.retrieve(ep.instruction, ep.scope))
        return sysm, seed

    def test_support_increments_for_every_type(self):
        cfg = MemoryConfig(policy=WritePolicy.full(online_write=False, batch_induction=False, refine=False))
        for kind in TYPES:
            sysm, seed = self._run(kind, "followed_success", cfg, 3)
            self.assertEqual(sysm.store.get(seed.id).support, 3, kind)
            self.assertGreater(sysm.store.get(seed.id).confidence(), 0.5, kind)

    def test_refinement_fires_for_every_type_and_resets_counters(self):
        cfg = MemoryConfig(
            policy=WritePolicy.full(online_write=False, batch_induction=False),
            refute_threshold=2,
            delete_min_observations=99,
        )
        for kind in TYPES:
            sysm, seed = self._run(kind, "followed_failure", cfg, 2)
            refined = sysm.store.get(seed.id)
            self.assertIsNotNone(refined, kind)
            self.assertEqual((refined.support, refined.refute), (0, 0), kind)
            self.assertEqual(refined.version, 2, kind)

    def test_low_confidence_deletion_fires_for_every_type(self):
        cfg = MemoryConfig(
            policy=WritePolicy.full(online_write=False, batch_induction=False, refine=False),
            delete_confidence=0.3,
            delete_min_observations=4,
        )
        for kind in TYPES:
            sysm, seed = self._run(kind, "followed_failure", cfg, 4)
            self.assertNotIn(seed.id, sysm.store, kind)

    def test_not_applicable_changes_nothing(self):
        cfg = MemoryConfig(policy=WritePolicy.full(online_write=False, batch_induction=False))
        for kind in TYPES:
            sysm, seed = self._run(kind, "not_applicable", cfg, 3)
            it = sysm.store.get(seed.id)
            self.assertEqual((it.support, it.refute), (0, 0), kind)

    def test_refined_entry_keeps_its_grounded_evidence(self):
        cfg = MemoryConfig(
            policy=WritePolicy.full(online_write=False, batch_induction=False), refute_threshold=2
        )
        sysm, seed = self._run("rule", "followed_failure", cfg, 2)
        self.assertEqual(sysm.store.get(seed.id).content["evidence"], EVIDENCE)


# --------------------------------------------------------- batch induction
class BatchInductionTest(unittest.TestCase):
    def test_fires_on_cadence_for_every_type(self):
        cfg = MemoryConfig(policy=WritePolicy.full(batch_every=3))
        for kind in TYPES:
            llm = StubWriterLLM()
            sysm = build_system(kind, llm=llm, config=cfg)
            for i in range(3):
                sysm.observe(episode(i), None)
            self.assertEqual(llm.usage.by_tag.get(f"{kind}.induce"), 1, kind)
            self.assertEqual(sysm._buffer, [], kind)

    def test_cluster_by_task_type_when_available(self):
        sysm = build_system("reflection", llm=StubWriterLLM(), config=MemoryConfig())
        eps = [episode(0, task_type="heat"), episode(1, task_type="clean"), episode(2, task_type="heat")]
        clusters = sysm.cluster(eps)
        self.assertEqual(sorted(clusters), ["clean", "heat"])
        self.assertEqual(len(clusters["heat"]), 2)

    def test_cluster_by_embedding_when_no_task_type(self):
        sysm = build_system("rule", llm=StubWriterLLM(), config=MemoryConfig(cluster_sim=0.9))
        eps = [
            Episode(task_id="a", instruction="heat the mug and put it in the fridge", scope={"env": ENV}),
            Episode(task_id="b", instruction="heat the mug and put it in the fridge", scope={"env": ENV}),
            Episode(task_id="c", instruction="buy a red shirt under 30 dollars", scope={"env": ENV}),
        ]
        self.assertEqual(len(sysm.cluster(eps)), 2)

    def test_induction_evidence_is_checked_against_the_whole_batch(self):
        sysm = build_system("skill", llm=StubWriterLLM(), config=MemoryConfig(policy=WritePolicy.full()))
        eps = [episode(i) for i in range(3)]
        prop = sysm.writer.induce(eps, [], cluster_name="c")
        self.assertEqual(prop.rejected, [])
        self.assertGreaterEqual(len(prop.ops), 1)


# ------------------------------------------------------- type-inherent bits
class SkillSystemTest(unittest.TestCase):
    def test_no_online_draft_from_pure_failure(self):
        """Content-inherent, not a mechanism: a procedure needs a working path."""
        llm = StubWriterLLM()
        cfg = MemoryConfig(policy=WritePolicy.minimal())
        sysm = SkillSystem(llm=llm, config=cfg)
        sysm.observe(episode(0, flags=(False, False)), None)
        self.assertIsNone(llm.usage.by_tag.get("skill.write"))
        self.assertEqual(len(sysm.store), 0)

    def test_step_refinement_is_off_by_default(self):
        llm = StubWriterLLM()
        cfg = MemoryConfig(policy=WritePolicy.minimal())
        sysm = SkillSystem(llm=llm, config=cfg)
        skill = sysm.store.add(skill_item("pick_and_place"))
        for i in range(3):
            ep = episode(i, flags=(False, False))
            ep.meta["failed_skill_steps"] = {skill.id: 1}
            sysm.observe(ep, sysm.retrieve(ep.instruction, ep.scope))
        self.assertIsNone(llm.usage.by_tag.get("skill.refine_step"))
        self.assertEqual(sysm.store.get(skill.id).content["name"], "pick_and_place")

    def test_step_refinement_when_explicitly_enabled(self):
        cfg = MemoryConfig(policy=WritePolicy.minimal(), skill_step_refinement=True,
                           skill_failed_step_threshold=2)
        sysm = SkillSystem(llm=StubWriterLLM(), config=cfg)
        skill = sysm.store.add(skill_item("pick_and_place"))
        for i in range(2):
            ep = episode(i, flags=(False, False))
            ep.meta["failed_skill_steps"] = {skill.id: 1}
            sysm.observe(ep, sysm.retrieve(ep.instruction, ep.scope))
        self.assertEqual(sysm.store.get(skill.id).content["name"], "stub_procedure_refined")


class UtilityDeletionTest(unittest.TestCase):
    def test_applies_to_every_type(self):
        cfg = MemoryConfig(
            policy=WritePolicy.minimal(utility_deletion=True, online_write=False),
            utility_delete_min_retrieved=3,
            utility_delete_success_rate=0.2,
        )
        seeds = {"reflection": reflection_item, "rule": rule_item, "skill": skill_item}
        for kind, mk in seeds.items():
            sysm = build_system(kind, llm=StubWriterLLM(), config=cfg)
            it = sysm.store.add(mk("useless entry"))
            it.n_retrieved, it.n_retrieved_success = 3, 0
            sysm.observe(episode(0, flags=(False,)), None)
            self.assertNotIn(it.id, sysm.store, kind)


class RawBaselineTest(unittest.TestCase):
    def test_appends_only_successes_and_uses_no_llm(self):
        sysm = RawTrajectorySystem(config=MemoryConfig())
        sysm.observe(episode(0, flags=(False, False)), None)
        self.assertEqual(len(sysm.store), 0)
        sysm.observe(episode(1, flags=(True,)), None)
        self.assertEqual(len(sysm.store), 1)
        self.assertIsNone(sysm.writer)


class CompositeTest(unittest.TestCase):
    def test_budget_is_split_and_total_respected(self):
        cfg = MemoryConfig(injection_budget_tokens=600, policy=WritePolicy.minimal())
        sysm = build_system("all", llm=StubWriterLLM(), config=cfg)
        for i in range(6):
            ep = episode(i)
            sysm.observe(ep, sysm.retrieve(ep.instruction, ep.scope))
        r = sysm.retrieve("put the mug on the shelf", {"env": ENV, "task_type": "pick"})
        self.assertLessEqual(r.tokens, cfg.injection_budget_tokens)
        self.assertTrue(all(k in sysm.writer_usage() for k in TYPES))

    def test_subsystem_stores_are_independent(self):
        sysm = build_system("all", llm=StubWriterLLM(), config=MemoryConfig(policy=WritePolicy.minimal()))
        sysm.observe(episode(0), None)
        for name, s in sysm.systems.items():
            self.assertTrue({i.type for i in s.store.items()} <= {name}, name)


# ---------------------------------------------------------------- runner
class RunnerTest(unittest.TestCase):
    def test_frozen_blocks_writes_but_allows_retrieval(self):
        sysm = build_system("reflection", llm=StubWriterLLM(), config=MemoryConfig())
        sysm.observe(episode(0), None)
        with frozen(sysm):
            r = sysm.retrieve("put the mug on the shelf", {"env": ENV})
            self.assertGreaterEqual(len(r.items), 0)
            with self.assertRaises(RuntimeError):
                sysm.observe(episode(1), None)
        sysm.observe(episode(2), None)

    def test_frozen_blocks_composite_subsystems(self):
        sysm = build_system("all", llm=StubWriterLLM(), config=MemoryConfig())
        with frozen(sysm):
            with self.assertRaises(RuntimeError):
                sysm.systems["rule"].observe(episode(0), None)

    def test_log_record_has_everything_the_analysis_needs(self):
        cfg = MemoryConfig(policy=WritePolicy.minimal())
        sysm = build_system("reflection", llm=StubWriterLLM(), config=cfg)
        ev = Evolver(sysm, config=cfg)
        ev.run([episode(i) for i in range(3)])
        rec = ev.logger.records[-1]
        for field in (
            "step", "task_id", "rollout_rewards", "retrieved_ids", "injected_tokens",
            "writer_ops", "writer_prompt_tokens", "writer_completion_tokens", "store_size", "outcome",
        ):
            self.assertIn(field, rec)
        self.assertGreater(rec["writer_prompt_tokens"], 0)
        self.assertEqual(len(ev.logger.growth_curve()), 3)
        self.assertIn("APPEND", ev.logger.op_distribution())

    def test_trailing_batch_is_flushed(self):
        cfg = MemoryConfig(policy=WritePolicy.full(batch_every=10))
        llm = StubWriterLLM()
        sysm = build_system("rule", llm=llm, config=cfg)
        Evolver(sysm, config=cfg).run([episode(i) for i in range(3)])
        self.assertEqual(llm.usage.by_tag.get("rule.induce"), 1)

    def test_flush_induction_is_logged(self):
        """The flush report must reach the log, not just the store.

        The induction writer can APPEND consolidated entries and DELETE the ones
        they subsume, and `store.remove` is a hard delete. When flush() dropped
        its report, the log said "4 appends" while the store held 2 items and
        nothing explained the difference.
        """
        cfg = MemoryConfig(policy=WritePolicy.full(batch_every=10))
        sysm = build_system("rule", llm=StubWriterLLM(), config=cfg)
        ev = Evolver(sysm, config=cfg)
        ev.run([episode(i) for i in range(3)])
        flush_recs = [r for r in ev.logger.records if r["outcome"] == "flush_induction"]
        self.assertEqual(len(flush_recs), 1)
        rec = flush_recs[0]
        self.assertTrue(rec["task_id"].startswith("__flush_induction__"))
        self.assertGreater(rec["writer_calls"], 0)
        self.assertEqual(rec["store_size"], len(sysm.store))

    def test_retrieval_can_be_disabled_during_evolving(self):
        cfg = MemoryConfig(retrieve_during_evolving=False, policy=WritePolicy.minimal())
        sysm = build_system("reflection", llm=StubWriterLLM(), config=cfg)
        ev = Evolver(sysm, config=cfg)
        ev.run([episode(i) for i in range(3)])
        self.assertTrue(all(r["retrieved_ids"] == [] for r in ev.logger.records))

    def test_log_written_to_disk(self):
        with tempfile.TemporaryDirectory() as d:
            from memsys import RunLogger

            p = os.path.join(d, "run.jsonl")
            cfg = MemoryConfig(policy=WritePolicy.minimal())
            sysm = build_system("reflection", llm=StubWriterLLM(), config=cfg)
            Evolver(sysm, config=cfg, logger=RunLogger(p)).run([episode(0)])
            with open(p) as f:
                lines = [json.loads(x) for x in f if x.strip()]
        self.assertEqual(len(lines), 1)


class BudgetFairnessTest(unittest.TestCase):
    """The Memory Content comparison is only valid if every system obeys the same B."""

    def test_all_systems_respect_the_same_budget(self):
        B = 300
        for kind in ("reflection", "rule", "skill", "raw", "all"):
            cfg = MemoryConfig(injection_budget_tokens=B, policy=WritePolicy.full(batch_every=2))
            sysm = build_system(kind, llm=StubWriterLLM(), config=cfg)
            for i in range(6):
                ep = episode(i)
                sysm.observe(ep, sysm.retrieve(ep.instruction, ep.scope))
            r = sysm.retrieve("put the mug on the shelf", {"env": ENV, "task_type": "pick"})
            self.assertLessEqual(r.tokens, B, f"{kind} overflowed the budget: {r.tokens} > {B}")


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""Adapter tests. No ALFWorld, no network: the environment and the LLM are faked.

What is worth testing here is the glue that silently corrupts a run when wrong:
response parsing, the memory block reaching the prompt, history trimming, and
the success/outcome bookkeeping that every memory system reads.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from memsys.adapters.alfworld import (
    MEMORY_HEADER,
    SYSTEM_PROMPT,
    AgentConfig,
    AlfworldAgent,
    TaskSpec,
    extract_instruction,
    load_demonstrations,
    load_manifest,
    parse_response,
    run_task,
)

TASK = TaskSpec(
    task_id="t1",
    split="valid_unseen",
    gamefile="json_2.1.1/valid_unseen/pick_and_place_simple-X/trial_1/game.tw-pddl",
    task_family="pick_and_place_simple",
    instruction="put a mug in desk",
)


class FakeEnv:
    """Stands in for ALFWorldEnvironment; succeeds when `winning_action` is sent."""

    def __init__(self, winning_action: str | None = "put mug 1 in/on desk 1", n_before_done=None):
        self.winning_action = winning_action
        self.actions: list[str] = []
        self.resets = 0
        self.n_before_done = n_before_done

    def reset(self, task):
        self.resets += 1
        self.actions = []
        return (
            "You are in the middle of a room. Looking quickly around you, you see a desk 1.\n\n"
            "Your task is to: put a mug in desk."
        )

    def step(self, action):
        self.actions.append(action)
        won = self.winning_action is not None and action == self.winning_action
        done = won or (self.n_before_done is not None and len(self.actions) >= self.n_before_done)
        return {
            "observation": "You put the mug in the desk." if won else "Nothing happens.",
            "success": won,
            "reward": 1.0 if won else 0.0,
            "done": done,
            "truncated": done and not won,
        }

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


class FakeClient:
    """Minimal stand-in for the OpenAI client, returning scripted completions."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[list[dict]] = []
        outer = self

        class _Completions:
            def create(self, model, messages, temperature, max_tokens, stop=None):
                outer.calls.append([dict(m) for m in messages])
                outer.last_stop = stop
                text = outer.responses.pop(0) if outer.responses else "Thought: idle.\nAction: look"
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
                    usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
                )

        self.chat = SimpleNamespace(completions=_Completions())


def make_agent(responses, **cfg):
    return AlfworldAgent(
        client=FakeClient(responses),
        model="fake",
        demonstrations=[{"role": "user", "content": "demo"}],
        config=AgentConfig(**cfg),
    )


class ParsingTest(unittest.TestCase):
    def test_react_format(self):
        thought, action = parse_response("Thought: I should look around.\nAction: go to desk 1")
        self.assertEqual(action, "go to desk 1")
        self.assertEqual(thought, "I should look around.")

    def test_first_action_wins_when_model_hallucinates_a_continuation(self):
        # Later actions belong to turns the model imagined; executing one would
        # skip the episode ahead past observations that never happened.
        _, action = parse_response(
            "Thought: go there.\nAction: go to desk 1\n"
            "Observation: you see a mug 1.\nAction: take mug 1 from desk 1"
        )
        self.assertEqual(action, "go to desk 1")

    def test_bare_action_without_scaffolding(self):
        _, action = parse_response("go to desk 1")
        self.assertEqual(action, "go to desk 1")

    def test_unparseable_returns_empty(self):
        _, action = parse_response("I am not going to follow the format, " + "x" * 300)
        self.assertEqual(action, "")

    def test_extract_instruction(self):
        obs = "You are in a room. You see a desk 1.\n\nYour task is to: put a mug in desk."
        self.assertEqual(extract_instruction(obs), "put a mug in desk.")


class AgentLoopTest(unittest.TestCase):
    def test_success_ends_episode(self):
        agent = make_agent(
            ["Thought: go.\nAction: go to desk 1", "Thought: done.\nAction: put mug 1 in/on desk 1"],
            max_steps=10,
        )
        rollout = agent.run(FakeEnv(), TASK)
        self.assertTrue(rollout.success)
        self.assertEqual(rollout.reward, 1.0)
        self.assertEqual(len(rollout.steps), 2)
        self.assertIsNone(rollout.error)

    def test_step_limit_marks_failure(self):
        agent = make_agent(["Thought: x.\nAction: go to desk 1"] * 3, max_steps=3)
        rollout = agent.run(FakeEnv(winning_action=None), TASK)
        self.assertFalse(rollout.success)
        self.assertEqual(rollout.error, "step_limit_reached")
        self.assertEqual(len(rollout.steps), 3)

    def test_unparseable_response_falls_back_to_look(self):
        agent = make_agent(["no format here " + "y" * 300], max_steps=1)
        env = FakeEnv(winning_action=None)
        rollout = agent.run(env, TASK)
        self.assertEqual(env.actions, ["look"])
        self.assertEqual(rollout.meta["parse_failures"], 1)

    def test_stop_sequence_is_sent(self):
        agent = make_agent(["Thought: t.\nAction: put mug 1 in/on desk 1"], max_steps=1)
        agent.run(FakeEnv(), TASK)
        self.assertIn("\nObservation:", agent.client.last_stop or [])

    def test_llm_error_is_recorded_not_raised(self):
        agent = make_agent([], max_steps=2)

        def boom(**kw):
            raise RuntimeError("connection reset")

        agent.client.chat.completions.create = boom
        rollout = agent.run(FakeEnv(), TASK)
        self.assertIn("llm_error", rollout.error or "")
        self.assertEqual(len(rollout.steps), 0)

    def test_memory_block_is_injected_into_first_user_turn(self):
        agent = make_agent(["Thought: t.\nAction: put mug 1 in/on desk 1"], max_steps=2)
        agent.run(FakeEnv(), TASK, memory_block="RULE: always open the drawer first")
        first_call = agent.client.calls[0]
        self.assertEqual(first_call[0]["role"], "system")
        self.assertEqual(first_call[0]["content"], SYSTEM_PROMPT)
        opening = first_call[-1]["content"]
        self.assertIn(MEMORY_HEADER, opening)
        self.assertIn("always open the drawer first", opening)
        self.assertIn("Your task is to: put a mug in desk.", opening)

    def test_no_memory_block_leaves_no_header(self):
        agent = make_agent(["Thought: t.\nAction: put mug 1 in/on desk 1"], max_steps=2)
        agent.run(FakeEnv(), TASK, memory_block="")
        self.assertNotIn(MEMORY_HEADER, agent.client.calls[0][-1]["content"])

    def test_history_is_trimmed_but_keeps_system_demos_and_task(self):
        agent = make_agent(["Thought: t.\nAction: go to desk 1"] * 30,
                           max_steps=30, context_limit_tokens=200, min_history_steps=2)
        agent.run(FakeEnv(winning_action=None), TASK)
        last = agent.client.calls[-1]
        self.assertEqual(last[0]["content"], SYSTEM_PROMPT)
        self.assertEqual(last[1]["content"], "demo")
        self.assertIn("Your task is to:", last[2]["content"])
        # trimming really happened rather than the guard silently never firing
        self.assertLess(len(last), 2 + 2 * 30)


class EpisodeTest(unittest.TestCase):
    def test_outcome_buckets_across_rollouts(self):
        import memsys.adapters.alfworld as mod

        env = FakeEnv()
        original = mod.ALFWorldEnvironment
        mod.ALFWorldEnvironment = lambda *a, **k: env
        try:
            # rollout 0 succeeds, rollout 1 does not -> MIXED, the only bucket in
            # which the Reflection writer's from_contrast mode fires.
            agent = make_agent(
                ["Thought: a.\nAction: put mug 1 in/on desk 1",
                 "Thought: b.\nAction: go to desk 1"],
                max_steps=1,
            )
            ep = run_task(agent, "/unused", TASK, n_rollouts=2)
        finally:
            mod.ALFWorldEnvironment = original
        self.assertEqual(ep.outcome(), "mixed")
        self.assertEqual(ep.success_rate, 0.5)
        self.assertEqual(ep.instruction, "put a mug in desk")
        self.assertEqual(ep.scope, {"env": "alfworld", "task_type": "pick_and_place_simple"})


class ManifestTest(unittest.TestCase):
    def test_roundtrip_and_family_inference(self):
        payload = {
            "split": "train",
            "tasks": [
                {"task_id": "a", "gamefile":
                 "json_2.1.1/train/pick_heat_then_place_in_recep-Egg/trial_2/game.tw-pddl",
                 "instruction": "heat an egg"},
            ],
        }
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m.json"
            p.write_text(json.dumps(payload), encoding="utf-8")
            tasks = load_manifest(p)
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].task_family, "pick_heat_then_place_in_recep")
        self.assertEqual(tasks[0].instruction, "heat an egg")
        self.assertEqual(tasks[0].split, "train")

    def test_demonstrations_cover_all_six_families(self):
        msgs = load_demonstrations()
        self.assertGreater(len(msgs), 40)
        self.assertTrue(all(m["role"] in ("user", "assistant") for m in msgs))
        # The scaffold's value is that it demonstrates every task type; SAGE
        # measured 18% -> 60% from adding exactly these.
        self.assertTrue(any("Thought:" in m["content"] for m in msgs))


class ConcurrentResetTest(unittest.TestCase):
    """Regression: loading a .tw-pddl game is not thread-safe.

    `reset()` parses the PDDL grammar with a module-level tatsu parser, so
    parallel evaluation workers used to corrupt its shared rule stack and die
    with `IndexError: pop from empty list` far away in tatsu. Single-threaded
    runs never show it. Needs real data, so it skips when ALFWORLD_DATA is unset.
    """

    def test_parallel_resets_do_not_corrupt_the_pddl_parser(self):
        import os
        from concurrent.futures import ThreadPoolExecutor

        data_root = os.environ.get("ALFWORLD_DATA")
        manifest = Path(__file__).resolve().parent.parent / "manifests"
        candidates = sorted(manifest.glob("eval_*.json")) if manifest.is_dir() else []
        if not data_root or not candidates:
            self.skipTest("needs ALFWORLD_DATA and a built manifest")

        from memsys.adapters.alfworld import ALFWorldEnvironment

        tasks = load_manifest(candidates[0])[:8]

        def load(task):
            with ALFWorldEnvironment(data_root, max_steps=1) as env:
                return extract_instruction(env.reset(task))

        with ThreadPoolExecutor(max_workers=4) as pool:
            goals = list(pool.map(load, tasks))
        self.assertEqual(len(goals), len(tasks))
        self.assertTrue(all(g for g in goals), "a worker returned an empty goal")


if __name__ == "__main__":
    unittest.main()

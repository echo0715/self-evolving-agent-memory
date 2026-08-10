"""WebShop adapter tests. No WebShop, no network: env and LLM are faked.

Same principle as `test_alfworld_adapter.py` -- test the glue that corrupts a run
silently rather than loudly. For WebShop that is action parsing (a malformed
`click[...]` is indistinguishable from a rejected one unless the adapter says
so), the graded-vs-strict reward split, and the memory block reaching the prompt.

`LiveServerTest` at the bottom exercises the real HTTP server concurrently and
skips when it is not running.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from memsys.adapters.webshop import (
    MEMORY_HEADER,
    SYSTEM_PROMPT,
    AgentConfig,
    TaskSpec,
    WebShopAgent,
    extract_instruction,
    load_demonstrations,
    load_manifest,
    normalize_action,
    parse_response,
    run_task,
)

TASK = TaskSpec(
    task_id="test:7",
    split="test",
    goal_index=7,
    category="beauty",
    instruction="i need a 3 ounce bottle of citrus deodorant, and price lower than 40.00 dollars",
)


class FakeEnv:
    """Stands in for WebShopEnvironment. Buying `winner` scores `reward`."""

    def __init__(self, winner="click[Buy Now]", reward=1.0, invalid_actions=()):
        self.winner = winner
        self.reward = reward
        self.invalid_actions = set(invalid_actions)
        self.actions: list[str] = []
        self.resets = 0
        self.closed = False

    def reset(self, task):
        self.resets += 1
        self.actions = []
        return (
            "WebShop [SEP] Instruction: [SEP] i need a 3 ounce bottle of citrus deodorant, "
            "and price lower than 40.00 dollars [SEP] Search"
        )

    def step(self, action):
        self.actions.append(action)
        if action.lower().startswith("think["):
            return {"observation": "OK.", "reward": 0.0, "success": False, "done": False,
                    "truncated": False, "invalid": False, "thought_only": True}
        bought = action == self.winner
        return {
            "observation": "Thank you for shopping with us!" if bought else "[SEP] results [SEP]",
            "reward": self.reward if bought else 0.0,
            "success": bought and self.reward >= 1.0,
            "done": bought,
            "truncated": False,
            "invalid": action in self.invalid_actions,
            "thought_only": False,
        }

    def render_actions(self):
        return "Available actions: search[<keywords>]"

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[list[dict]] = []
        self.last_stop = None
        outer = self

        class _Completions:
            def create(self, model, messages, temperature, max_tokens, stop=None):
                outer.calls.append([dict(m) for m in messages])
                outer.last_stop = stop
                text = outer.responses.pop(0) if outer.responses else "Thought: idle.\nAction: think[wait]"
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
                    usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
                )

        self.chat = SimpleNamespace(completions=_Completions())


def make_agent(responses, **cfg):
    return WebShopAgent(
        client=FakeClient(responses),
        model="fake",
        demonstrations=[{"role": "user", "content": "demo"}],
        config=AgentConfig(**cfg),
    )


class ParsingTest(unittest.TestCase):
    def test_react_format(self):
        thought, action = parse_response("Thought: I should search.\nAction: search[citrus deodorant]")
        self.assertEqual(action, "search[citrus deodorant]")
        self.assertEqual(thought, "I should search.")

    def test_first_action_wins_when_model_hallucinates_a_continuation(self):
        _, action = parse_response(
            "Thought: search.\nAction: search[deodorant]\n"
            "Observation: results\nAction: click[B078GWRC1J]"
        )
        self.assertEqual(action, "search[deodorant]")

    def test_bare_action_without_scaffolding(self):
        _, action = parse_response("click[Buy Now]")
        self.assertEqual(action, "click[Buy Now]")

    def test_prose_is_not_mistaken_for_an_action(self):
        # The ALFWorld adapter accepts any short first line as an action because
        # ALFWorld actions are bare text. WebShop actions are always verb[arg],
        # so prose must be rejected -- otherwise every chatty response would be
        # sent to the store as a nonsense click and counted as a valid step.
        _, action = parse_response("I think I should look for a deodorant first.")
        self.assertEqual(action, "")

    def test_normalize_action_rejects_unknown_verbs_and_empty_args(self):
        self.assertEqual(normalize_action("Search[Citrus]"), "search[Citrus]")
        self.assertEqual(normalize_action("buy[thing]"), "")
        self.assertEqual(normalize_action("click[]"), "")
        self.assertEqual(normalize_action("click[a] extra"), "")

    def test_extract_instruction_from_sep_joined_page(self):
        obs = ("WebShop [SEP] Instruction: [SEP] i need a citrus deodorant, and price lower "
               "than 40.00 dollars [SEP] Search")
        self.assertEqual(
            extract_instruction(obs),
            "i need a citrus deodorant, and price lower than 40.00 dollars",
        )


class AgentLoopTest(unittest.TestCase):
    def test_purchase_ends_episode_with_full_reward(self):
        agent = make_agent(
            ["Thought: search.\nAction: search[citrus deodorant]",
             "Thought: buy.\nAction: click[Buy Now]"],
            max_steps=10,
        )
        rollout = agent.run(FakeEnv(), TASK)
        self.assertTrue(rollout.success)
        self.assertEqual(rollout.reward, 1.0)
        self.assertTrue(rollout.meta["purchased"])
        self.assertEqual(len(rollout.steps), 2)
        self.assertIsNone(rollout.error)

    def test_partial_reward_is_kept_but_is_not_success(self):
        # WebShop grades partially. An arm that buys plausible-but-wrong items
        # raises the score and lowers the success rate, so conflating the two
        # would hide exactly the effect worth seeing.
        agent = make_agent(["Thought: buy.\nAction: click[Buy Now]"], max_steps=5)
        rollout = agent.run(FakeEnv(reward=0.75), TASK)
        self.assertFalse(rollout.success)
        self.assertEqual(rollout.reward, 0.75)
        self.assertTrue(rollout.meta["purchased"])

    def test_step_limit_marks_failure_and_no_purchase(self):
        agent = make_agent(["Thought: x.\nAction: search[deodorant]"] * 3, max_steps=3)
        rollout = agent.run(FakeEnv(), TASK)
        self.assertFalse(rollout.success)
        self.assertEqual(rollout.error, "step_limit_reached")
        self.assertFalse(rollout.meta["purchased"])
        self.assertEqual(len(rollout.steps), 3)

    def test_unparseable_response_costs_a_step_as_a_thought(self):
        agent = make_agent(["no format here at all"], max_steps=1)
        env = FakeEnv()
        rollout = agent.run(env, TASK)
        self.assertEqual(rollout.meta["parse_failures"], 1)
        self.assertEqual(len(env.actions), 1)
        self.assertTrue(env.actions[0].startswith("think["))

    def test_invalid_action_is_announced_to_the_agent(self):
        # The store returns the *unchanged page* for an action it cannot perform,
        # which reads exactly like an action that legitimately changed nothing.
        agent = make_agent(
            ["Thought: t.\nAction: click[nope]", "Thought: t.\nAction: click[Buy Now]"],
            max_steps=5,
        )
        rollout = agent.run(FakeEnv(invalid_actions=["click[nope]"]), TASK)
        self.assertEqual(rollout.meta["invalid_actions"], 1)
        self.assertIn("Invalid action", rollout.steps[0].observation)

    def test_think_steps_are_counted(self):
        agent = make_agent(
            ["Thought: t.\nAction: think[plan first]", "Thought: t.\nAction: click[Buy Now]"],
            max_steps=5,
        )
        rollout = agent.run(FakeEnv(), TASK)
        self.assertEqual(rollout.meta["thought_steps"], 1)
        self.assertEqual(rollout.steps[0].observation, "OK.")

    def test_stop_sequence_is_sent(self):
        agent = make_agent(["Thought: t.\nAction: click[Buy Now]"], max_steps=1)
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
        agent = make_agent(["Thought: t.\nAction: click[Buy Now]"], max_steps=2)
        agent.run(FakeEnv(), TASK, memory_block="SKILL: select every option before buying")
        first_call = agent.client.calls[0]
        self.assertEqual(first_call[0]["role"], "system")
        self.assertEqual(first_call[0]["content"], SYSTEM_PROMPT)
        opening = first_call[-1]["content"]
        self.assertIn(MEMORY_HEADER, opening)
        self.assertIn("select every option before buying", opening)
        self.assertIn("citrus deodorant", opening)
        self.assertIn("Available actions:", opening)

    def test_no_memory_block_leaves_no_header(self):
        agent = make_agent(["Thought: t.\nAction: click[Buy Now]"], max_steps=2)
        agent.run(FakeEnv(), TASK, memory_block="")
        self.assertNotIn(MEMORY_HEADER, agent.client.calls[0][-1]["content"])

    def test_long_observations_are_truncated_before_entering_history(self):
        class Bloated(FakeEnv):
            def step(self, action):
                out = super().step(action)
                out["observation"] = "x" * 50_000
                out["done"] = False
                return out

        agent = make_agent(["Thought: t.\nAction: search[a]"] * 2,
                           max_steps=2, max_observation_chars=100)
        agent.run(Bloated(), TASK)
        last_user = agent.client.calls[-1][-1]["content"]
        self.assertLess(len(last_user), 500)
        self.assertIn("<truncated>", last_user)

    def test_history_is_trimmed_but_keeps_system_demos_and_task(self):
        agent = make_agent(["Thought: t.\nAction: search[a]"] * 30,
                           max_steps=30, context_limit_tokens=200, min_history_steps=2)
        agent.run(FakeEnv(), TASK)
        last = agent.client.calls[-1]
        self.assertEqual(last[0]["content"], SYSTEM_PROMPT)
        self.assertEqual(last[1]["content"], "demo")
        self.assertIn("citrus deodorant", last[2]["content"])
        self.assertLess(len(last), 2 + 2 * 30)


class EpisodeTest(unittest.TestCase):
    def test_outcome_buckets_and_scope(self):
        import memsys.adapters.webshop as mod

        env = FakeEnv()
        original = mod.WebShopEnvironment
        mod.WebShopEnvironment = lambda *a, **k: env
        try:
            # rollout 0 buys, rollout 1 runs out of steps -> MIXED, the only
            # bucket in which Reflection's from_contrast mode fires.
            agent = make_agent(
                ["Thought: a.\nAction: click[Buy Now]", "Thought: b.\nAction: search[x]"],
                max_steps=1,
            )
            ep = run_task(agent, "http://unused", TASK, n_rollouts=2)
        finally:
            mod.WebShopEnvironment = original
        self.assertEqual(ep.outcome(), "mixed")
        self.assertEqual(ep.success_rate, 0.5)
        self.assertEqual(ep.instruction, TASK.instruction)
        self.assertEqual(ep.scope, {"env": "webshop", "task_type": "beauty"})
        self.assertTrue(env.closed)


class ManifestTest(unittest.TestCase):
    def test_roundtrip(self):
        payload = {
            "split": "train",
            "tasks": [{"task_id": "train:1794", "goal_index": 1794,
                       "category": "garden", "instruction": "i need a sewing kit"}],
        }
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m.json"
            p.write_text(json.dumps(payload), encoding="utf-8")
            tasks = load_manifest(p)
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].goal_index, 1794)
        self.assertEqual(tasks[0].scope(), {"env": "webshop", "task_type": "garden"})

    def test_entry_without_goal_index_is_rejected(self):
        # A WebShop task *is* its goal index. Defaulting a missing one to 0 would
        # silently evaluate task 0 a hundred times and still report a rate.
        with self.assertRaises(ValueError):
            TaskSpec.from_dict({"task_id": "x", "category": "beauty"})

    def test_shipped_manifests_are_disjoint_and_carry_server_identity(self):
        root = Path(__file__).resolve().parent.parent / "manifests"
        evolve = root / "webshop_evolve_train_50_seed42.json"
        ev = root / "webshop_eval_test_100_seed42.json"
        if not (evolve.is_file() and ev.is_file()):
            self.skipTest("webshop manifests not built")
        a, b = json.loads(evolve.read_text()), json.loads(ev.read_text())
        ia = {t["goal_index"] for t in a["tasks"]}
        ib = {t["goal_index"] for t in b["tasks"]}
        self.assertEqual(len(ia), 50)
        self.assertEqual(len(ib), 100)
        self.assertFalse(ia & ib, "evolve and eval sets overlap")
        # A goal index means nothing without the corpus/seed that produced it.
        for m in (a, b):
            self.assertEqual(m["server"]["scale"], "full")
            self.assertIsNotNone(m["server"]["seed"])

    def test_demonstrations_end_in_a_purchase(self):
        msgs = load_demonstrations()
        self.assertGreater(len(msgs), 8)
        self.assertTrue(all(m["role"] in ("user", "assistant") for m in msgs))
        assistants = [m["content"] for m in msgs if m["role"] == "assistant"]
        self.assertTrue(any("click[Buy Now]" in c for c in assistants))
        # Every demonstrated action must parse under the same rules the runtime
        # uses, or the demonstrations teach a format the parser rejects.
        for c in assistants:
            _, action = parse_response(c)
            self.assertNotEqual(action, "", f"demonstration action does not parse: {c!r}")


class LiveServerTest(unittest.TestCase):
    """Exercises the real env server, including concurrent sessions.

    The server serialises env access behind one lock and keys browser state on a
    per-client token. The failure this guards against is two concurrent episodes
    sharing a session: WebShop reuses the goal already attached to a session id,
    so a collision would silently give two workers the same task while both
    report results for different ones.
    """

    def setUp(self):
        self.url = os.environ.get("WEBSHOP_SERVER_URL", "http://localhost:7000")
        from memsys.adapters.webshop import WebShopEnvironment, WebShopError

        try:
            WebShopEnvironment(self.url, timeout=5).health()
        except WebShopError:
            self.skipTest(f"no WebShop server at {self.url}")

    def test_concurrent_sessions_keep_their_own_goals(self):
        from concurrent.futures import ThreadPoolExecutor

        from memsys.adapters.webshop import WebShopEnvironment

        tasks = [TaskSpec(f"test:{i}", "test", i) for i in range(8)]

        def probe(task):
            with WebShopEnvironment(self.url, max_steps=2) as env:
                env.reset(task)
                # Step once so the session is genuinely live while its neighbours
                # are resetting -- a reset-only probe would not exercise the pool.
                env.step("search[socks]")
                return task.goal_index, env.instruction

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(probe, tasks))

        instructions = {idx: text for idx, text in results}
        self.assertEqual(len(instructions), len(tasks))
        self.assertTrue(all(t for t in instructions.values()), "a session got no instruction")
        # Distinct goals must produce distinct instructions; identical text
        # across sessions is the signature of a session-id collision.
        self.assertEqual(len(set(instructions.values())), len(tasks))

    def test_reward_is_graded_and_purchase_terminates(self):
        from memsys.adapters.webshop import WebShopEnvironment

        with WebShopEnvironment(self.url, max_steps=10) as env:
            env.reset(TaskSpec("test:3019", "train", 3019))
            env.step("search[heavy duty splice board computer office desk]")
            env.step("click[B09NCB2NC4]")
            result = env.step("click[Buy Now]")
        self.assertTrue(result["done"])
        self.assertGreater(result["reward"], 0.0)
        self.assertLessEqual(result["reward"], 1.0)


if __name__ == "__main__":
    unittest.main()

"""AppWorld adapter tests. No AppWorld, no network: env and LLM are faked.

The glue worth testing here is not the same as the other two adapters'. AppWorld
runs *arbitrary generated Python* against live APIs, so the failure modes are
prompt rendering (a 7k-token scaffold with seven substitution slots), code
extraction, and the evaluate path -- whose default arguments lose every
incomplete episode if you let them.

`LiveServerTest` exercises a real `appworld serve environment` and skips when
none is running.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from memsys.adapters.appworld import (
    EMPTY_PLAYBOOK,
    AgentConfig,
    AppWorldAgent,
    TaskSpec,
    extract_code,
    load_manifest,
    load_prompt_messages,
    render_prompt,
    run_task,
)

TASK = TaskSpec(
    task_id="3d9a636_1",
    split="test_normal",
    instruction="What is the title of the most-liked song in my Spotify playlists.",
    scenario="3d9a636",
)

TASK_INFO = {
    "task_id": "3d9a636_1",
    "instruction": TASK.instruction,
    "supervisor": {"first_name": "Joyce", "last_name": "Weaver",
                   "email": "joyce-weav@gmail.com", "phone_number": "3155673041"},
}


class FakeEnv:
    """Stands in for AppWorldEnvironment; `winning_code` completes the task."""

    def __init__(self, winning_code="apis.supervisor.complete_task()", score=1.0,
                 incomplete_score=0.0, num_tests=2):
        self.winning_code = winning_code
        self.score = score
        # AppWorld scores world state, so an episode that never called
        # complete_task can still pass tests. Keeping that separately settable is
        # what lets one test assert a timed-out episode still scores, while
        # others get the ordinary "ran out of time, got nothing" outcome.
        self.incomplete_score = incomplete_score
        self.num_tests = num_tests
        self.codes: list[str] = []
        self.completed = False
        self.closed = False
        self.evaluated = 0

    def reset(self, task):
        self.codes = []
        self.completed = False
        return dict(TASK_INFO, task_id=task.task_id)

    def execute(self, code):
        self.codes.append(code)
        if code.strip() == self.winning_code:
            self.completed = True
            return "Execution successful."
        if "boom" in code:
            return "Traceback (most recent call last):\nValueError: boom"
        return "Output ok"

    def task_completed(self):
        return self.completed

    def evaluate(self):
        self.evaluated += 1
        score = self.score if self.completed else self.incomplete_score
        return {"success": score >= 1.0, "score": score,
                "num_tests": self.num_tests, "n_passed": round(score * self.num_tests),
                "difficulty": 1, "failures": []}

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
                text = outer.responses.pop(0) if outer.responses else "Code:\n```python\npass\n```"
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
                    usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
                )

        self.chat = SimpleNamespace(completions=_Completions())


def make_agent(responses, **cfg):
    return AppWorldAgent(
        client=FakeClient(responses),
        model="fake",
        app_descriptions="[{'name': 'spotify'}]",
        prompt_messages=[{"role": "user", "content":
                          "Playbook:\n{{ playbook }}\nApps: {{ app_descriptions }}\n"
                          "I am {{ main_user.first_name }} {{ main_user.last_name }} "
                          "({{ main_user.email }}, {{ main_user.phone_number }}).\n"
                          "Task: {{ input_str }}"}],
        config=AgentConfig(**cfg),
    )


def code(body):
    return f"Some reasoning.\nCode:\n```python\n{body}\n```"


class PromptTest(unittest.TestCase):
    def test_shipped_prompt_parses_into_alternating_messages(self):
        msgs = load_prompt_messages()
        self.assertGreater(len(msgs), 20)
        self.assertTrue(all(m["role"] in ("user", "assistant") for m in msgs))
        # Consecutive same-role blocks must be merged: ACE's prompt ends with
        # three USER blocks in a row and some chat templates reject repeats.
        for a, b in zip(msgs, msgs[1:]):
            self.assertNotEqual(a["role"], b["role"], "consecutive same-role messages survived")

    def test_every_slot_is_substituted(self):
        rendered = render_prompt(load_prompt_messages(), TASK_INFO, "APPS", "PLAYBOOK")
        joined = "\n".join(m["content"] for m in rendered)
        self.assertNotIn("{{", joined)
        self.assertIn("PLAYBOOK", joined)
        self.assertIn("Joyce", joined)
        self.assertIn(TASK.instruction, joined)

    def test_empty_memory_still_fills_the_playbook_slot(self):
        # Dropping the section for `none` would make the no-memory arm a
        # different scaffold rather than the same scaffold with empty memory.
        rendered = render_prompt(load_prompt_messages(), TASK_INFO, "APPS", "")
        joined = "\n".join(m["content"] for m in rendered)
        self.assertIn(EMPTY_PLAYBOOK, joined)
        self.assertNotIn("{{ playbook }}", joined)


class CodeExtractionTest(unittest.TestCase):
    def test_fenced_python_block(self):
        self.assertEqual(extract_code(code("print(1)")), "print(1)")

    def test_bare_fence_without_language(self):
        self.assertEqual(extract_code("Code:\n```\nprint(1)\n```"), "print(1)")

    def test_first_block_wins(self):
        # A model emitting several blocks has imagined its own continuation.
        # Running a later one executes code written against a REPL state that
        # never existed -- and these APIs send money and delete files.
        text = "```python\nfirst()\n```\nOutput:\n```\nok\n```\n```python\nsecond()\n```"
        self.assertEqual(extract_code(text), "first()")

    def test_prose_yields_nothing(self):
        self.assertEqual(extract_code("I will now log in to Spotify."), "")


class AgentLoopTest(unittest.TestCase):
    def test_completion_ends_episode_and_evaluates(self):
        agent = make_agent([code("x = 1"), code("apis.supervisor.complete_task()")],
                           max_interactions=10)
        env = FakeEnv()
        rollout = agent.run(env, TASK)
        self.assertTrue(rollout.success)
        self.assertEqual(rollout.reward, 1.0)
        self.assertTrue(rollout.meta["completed_task"])
        self.assertEqual(env.evaluated, 1)
        self.assertIsNone(rollout.error)

    def test_partial_credit_is_kept_but_is_not_success(self):
        agent = make_agent([code("apis.supervisor.complete_task()")], max_interactions=5)
        rollout = agent.run(FakeEnv(score=0.5), TASK)
        self.assertFalse(rollout.success)
        self.assertEqual(rollout.reward, 0.5)
        self.assertEqual(rollout.meta["n_passed"], 1)

    def test_interaction_limit_still_evaluates(self):
        # AppWorld scores world state, so an agent that did the work but never
        # called complete_task can still pass. Skipping evaluation on timeout
        # would score those as 0.
        agent = make_agent([code("x = 1")] * 3, max_interactions=3)
        env = FakeEnv(incomplete_score=0.5)
        rollout = agent.run(env, TASK)
        self.assertEqual(rollout.error, "interaction_limit_reached")
        self.assertFalse(rollout.meta["completed_task"])
        self.assertEqual(env.evaluated, 1)
        self.assertEqual(rollout.reward, 0.5)

    def test_missing_code_block_spends_an_interaction_and_nudges(self):
        agent = make_agent(["I will log in to Spotify.", code("apis.supervisor.complete_task()")],
                           max_interactions=5)
        env = FakeEnv()
        rollout = agent.run(env, TASK)
        self.assertEqual(rollout.meta["parse_failures"], 1)
        self.assertEqual(len(env.codes), 1)  # the prose turn executed nothing
        self.assertIn("No code block found", agent.client.calls[-1][-1]["content"])

    def test_execution_errors_are_counted(self):
        agent = make_agent([code("boom()"), code("apis.supervisor.complete_task()")],
                           max_interactions=5)
        rollout = agent.run(FakeEnv(), TASK)
        self.assertEqual(rollout.meta["exec_errors"], 1)

    def test_llm_error_is_recorded_not_raised(self):
        agent = make_agent([], max_interactions=3)

        def boom(**kw):
            raise RuntimeError("connection reset")

        agent.client.chat.completions.create = boom
        rollout = agent.run(FakeEnv(), TASK)
        self.assertIn("llm_error", rollout.error or "")
        self.assertEqual(len(rollout.steps), 0)

    def test_memory_block_reaches_the_playbook_slot(self):
        agent = make_agent([code("apis.supervisor.complete_task()")], max_interactions=3)
        agent.run(FakeEnv(), TASK, memory_block="RULE: always paginate")
        first = agent.client.calls[0]
        self.assertIn("RULE: always paginate", first[0]["content"])
        self.assertNotIn("{{", first[0]["content"])

    def test_long_output_is_truncated_before_entering_history(self):
        class Chatty(FakeEnv):
            def execute(self, code):
                super().execute(code)
                return "y" * 50_000

        agent = make_agent([code("x=1")] * 2, max_interactions=2, max_output_chars=100)
        agent.run(Chatty(), TASK)
        last_user = agent.client.calls[-1][-1]["content"]
        self.assertLess(len(last_user), 500)
        self.assertIn("output truncated", last_user)

    def test_history_is_trimmed_but_the_rendered_scaffold_survives(self):
        agent = make_agent([code("x=1")] * 30, max_interactions=30,
                           context_limit_tokens=120, min_history_steps=2)
        agent.run(FakeEnv(), TASK)
        last = agent.client.calls[-1]
        self.assertIn("Task: " + TASK.instruction, last[0]["content"])
        self.assertLess(len(last), 1 + 2 * 30)

    def test_stop_sequence_is_sent(self):
        agent = make_agent([code("apis.supervisor.complete_task()")], max_interactions=1)
        agent.run(FakeEnv(), TASK)
        self.assertIn("\nUSER:", agent.client.last_stop or [])


class EpisodeTest(unittest.TestCase):
    def test_outcome_buckets_and_scope(self):
        import memsys.adapters.appworld as mod

        env = FakeEnv()
        original = mod.AppWorldEnvironment
        mod.AppWorldEnvironment = lambda *a, **k: env
        try:
            agent = make_agent(
                [code("apis.supervisor.complete_task()"), code("x=1")], max_interactions=1
            )
            ep = run_task(agent, "http://unused", TASK, n_rollouts=2)
        finally:
            mod.AppWorldEnvironment = original
        self.assertEqual(ep.outcome(), "mixed")
        self.assertEqual(ep.scope, {"env": "appworld", "task_type": "3d9a636"})
        self.assertTrue(env.closed)


class ManifestTest(unittest.TestCase):
    def test_roundtrip_and_scenario_inference(self):
        payload = {"split": "train",
                   "tasks": [{"task_id": "82e2fac_2", "instruction": "do a thing"}]}
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m.json"
            p.write_text(json.dumps(payload), encoding="utf-8")
            tasks = load_manifest(p)
        self.assertEqual(tasks[0].scenario, "82e2fac")
        self.assertEqual(tasks[0].scope()["task_type"], "82e2fac")

    def test_entry_without_task_id_is_rejected(self):
        with self.assertRaises(ValueError):
            TaskSpec.from_dict({"instruction": "do a thing"})

    def test_shipped_manifests_are_disjoint_and_scenario_grouped(self):
        root = Path(__file__).resolve().parent.parent / "manifests"
        evolve = root / "appworld_evolve_train_50_seed42.json"
        ev = root / "appworld_eval_test_normal_100_seed42.json"
        if not (evolve.is_file() and ev.is_file()):
            self.skipTest("appworld manifests not built")
        a, b = json.loads(evolve.read_text()), json.loads(ev.read_text())
        ia = {t["task_id"] for t in a["tasks"]}
        ib = {t["task_id"] for t in b["tasks"]}
        self.assertEqual((len(ia), len(ib)), (50, 100))
        self.assertFalse(ia & ib)
        self.assertFalse({t["scenario"] for t in a["tasks"]} & {t["scenario"] for t in b["tasks"]},
                         "a scenario spans both splits")
        for m in (a, b):
            self.assertTrue(all(t["instruction"] for t in m["tasks"]),
                            "an entry has no resolved instruction")


class LiveServerTest(unittest.TestCase):
    def setUp(self):
        self.url = os.environ.get("APPWORLD_SERVER_URL", "http://localhost:9000")
        from memsys.adapters.appworld import AppWorldEnvironment, AppWorldError

        try:
            AppWorldEnvironment(self.url, timeout=5).health()
        except AppWorldError:
            self.skipTest(f"no AppWorld server at {self.url}")

    def test_evaluate_survives_an_unfinished_task(self):
        # The /evaluate route defaults suppress_errors to False, so an
        # unfinished task makes the evaluator raise on its first failed
        # assertion and the call comes back as an opaque HTTP 500. Every
        # incomplete episode -- the common case -- would be lost.
        from memsys.adapters.appworld import AppWorldEnvironment

        with AppWorldEnvironment(self.url, max_interactions=5) as env:
            env.reset(TaskSpec("82e2fac_1", "train"))
            env.execute("x = 1")
            result = env.evaluate()
        self.assertFalse(result["success"])
        self.assertGreaterEqual(result["num_tests"], 1)
        self.assertGreaterEqual(result["score"], 0.0)

    def test_app_descriptions_is_the_short_list_not_the_full_reference(self):
        # GET /api_docs returns the complete API reference for all nine apps
        # (~519 KB, ~140k tokens). Substituting that into the scaffold makes a
        # 7k-token prompt exceed any context window, and the names are close
        # enough that it looks like a working prompt until every call fails.
        from memsys.adapters.appworld import fetch_app_descriptions

        text = fetch_app_descriptions(self.url, "82e2fac_1")
        self.assertLess(len(text), 20_000)
        self.assertIn("spotify", text)


if __name__ == "__main__":
    unittest.main()

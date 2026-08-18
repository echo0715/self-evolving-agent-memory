"""ScienceWorld adapter: a ReAct agent loop that produces `Episode` objects.

Scaffold provenance (this matters, and it differs from the other benchmarks)
---------------------------------------------------------------------------
ALFWorld and WebShop take their system prompt and demonstrations verbatim from
the SAGE harness, so their `none` baselines can be checked against a number
somebody else measured. **SAGE has no ScienceWorld**, so the prompt below and
the two demonstrations in `scienceworld_examples.json` were written for this
study. The consequence is worth stating plainly rather than discovering later:
arm-vs-arm comparisons are unaffected -- every arm shares this exact scaffold
and the memory block is still the only thing that differs -- but the absolute
value of the `none` baseline is only meaningful *within* this study and should
not be compared against published ScienceWorld numbers, which use different
prompts, step limits and often a much larger model.

The demonstrations are built from `env.get_gold_action_sequence()` on **train**
variations (`find-non-living-thing` and `test-conductivity`, variation 0 of
each), with observations copied verbatim from the simulator and the Thought
lines written by hand. Train variations only: a demo drawn from a test
variation would put a worked solution to an evaluation task in every arm's
prompt, including `none`.

Three simulator behaviours drive the code below
-----------------------------------------------
1. **A wrong `focus on` scores -100 and ends the episode instantly.** Not a
   penalty to recover from -- `done` comes back True on that step. This is the
   dominant failure mode for a small model, so the system prompt warns about it
   explicitly and `Rollout.meta` counts it separately from ordinary failure.
2. **`envStepLimit` does not terminate the episode.** Constructing the env with
   `envStepLimit=N` and stepping past N leaves `done` False forever, so the step
   cap is enforced here, the same way the ALFWorld adapter does it.
3. **An out-of-range variation index is not an error.** `load()` accepts it and
   `reset()` returns `"ERROR: ... exceeds the total number of variations"` as
   the observation, which is indistinguishable downstream from an agent that
   walked into a wall. Both the manifest builder and `reset()` check for it.

Splits are the simulator's own: `get_variations_train/dev/test()` partition each
task's variation indices, and they are a property of the *currently loaded*
task -- reading them after loading a different task silently returns the wrong
range, which is how (3) is usually triggered.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from ..episode import Episode, Rollout, Step
from ..tokens import count_tokens

HERE = Path(__file__).resolve().parent
DEFAULT_DEMOS = HERE / "scienceworld_examples.json"

#: ScienceWorld's 30 task names, which are also the `task_type` retrieval scope
#: key and what batch induction clusters on (MemoryConfig.cluster_key). Listed
#: rather than queried so a manifest can be validated without booting a JVM.
TASK_NAMES = (
    "boil", "change-the-state-of-matter-of", "chemistry-mix",
    "chemistry-mix-paint-secondary-color", "chemistry-mix-paint-tertiary-color",
    "find-animal", "find-living-thing", "find-non-living-thing", "find-plant",
    "freeze", "grow-fruit", "grow-plant", "identify-life-stages-1",
    "identify-life-stages-2", "inclined-plane-determine-angle",
    "inclined-plane-friction-named-surfaces", "inclined-plane-friction-unnamed-surfaces",
    "lifespan-longest-lived", "lifespan-longest-lived-then-shortest-lived",
    "lifespan-shortest-lived", "measure-melting-point-known-substance",
    "measure-melting-point-unknown-substance", "melt",
    "mendelian-genetics-known-plant", "mendelian-genetics-unknown-plant",
    "power-component", "power-component-renewable-vs-nonrenewable-energy",
    "test-conductivity", "test-conductivity-of-unknown-substances", "use-thermometer",
)

#: The 26 action templates the simulator accepts, from `get_possible_actions()`.
#: Given verbatim because ScienceWorld's parser is exact-match on the template:
#: "grab X" and "take X" are both rejected where "pick up X" works, and a model
#: left to guess spends most of its step budget on rejected phrasings.
ACTION_TEMPLATES = """activate OBJ
close OBJ
connect OBJ to OBJ
deactivate OBJ
disconnect OBJ
dunk OBJ in OBJ
eat OBJ
flush OBJ
focus on OBJ
go OBJ
inventory
look around
look at OBJ
look in OBJ
mix OBJ
move OBJ to OBJ
open OBJ
pick up OBJ
pour OBJ in OBJ
put down OBJ
read OBJ
task
use OBJ on OBJ
wait
wait1"""

SYSTEM_PROMPT = f"""Interact with a simulated science laboratory to solve a task. You are an intelligent agent in a house with several rooms (kitchen, bathroom, workshop, art studio, greenhouse, bedroom, living room, hallway, outside), and your target is to perform actions that complete the task goal. At the beginning of your interactions you will be given the task description and a description of the room you start in.

For each of your turns you will be given the observation from the last turn. You should first think about the current condition and plan your future actions, and then output your action for this turn. Your output must strictly follow this format:"Thought: your thoughts.\nAction: your next action".

The available actions are exactly these templates, where OBJ is an object name:
{ACTION_TEMPLATES}

Important rules:
- Doors start closed. "open door to kitchen" then "go to kitchen".
- Use "look around" to see what is in a room before acting on it.
- "focus on OBJ" marks the object the task is about. It is scored, and focusing on the WRONG object ends the task immediately with no score. Only focus once you are sure, and only on the thing the task actually asks about.
- Some processes need simulator time to take effect (heating, cooling, a bulb lighting, a plant growing). Use "wait1" and then look again rather than assuming nothing happened.
- If the environment says an action is not understood, the phrasing was wrong -- re-read the templates above and use the exact object name the room description gave you.
- Keep your Thought to at most two sentences, then give the Action. Never end a turn without an Action line.

Your response should use the following format:

Thought: <your thoughts>
Action: <your next action>"""

MEMORY_HEADER = (
    "**Archived memory (distilled from your own past episodes in this environment):**"
)

#: Returned as the observation when `load()` was given a variation the task does
#: not have. See module docstring (3).
_VARIATION_ERROR = "exceeds the total number of variations"


def _ensure_java_on_path() -> None:
    """Put the interpreter's own `bin/` on PATH if `java` is not findable.

    ScienceWorld is a Scala simulator: `ScienceWorldEnv()` shells out to `java`
    through py4j. The JDK lives inside the conda env (see
    scripts/setup_scienceworld.sh), but running `$ENV/bin/python` directly does
    NOT put `$ENV/bin` on PATH -- only `conda activate` does. The failure is
    `FileNotFoundError: [Errno 2] No such file or directory: 'java'` raised from
    inside py4j's `launch_gateway`, which reads as a py4j problem rather than a
    PATH one. Same class of trap as `ninja` in scripts/serve_qwen.sh; fixing it
    here means every entry point works without a wrapper.
    """
    if shutil.which("java"):
        return
    bindir = Path(sys.executable).resolve().parent
    if (bindir / "java").exists():
        os.environ["PATH"] = f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}"


# ============================================================ task manifests
@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    split: str
    task_name: str
    variation: int
    task_family: str = ""
    #: The task description, baked in by scripts/build_scienceworld_manifests.py.
    #: Retrieval happens before the agent acts, but ScienceWorld only reveals the
    #: description after load()+reset(), so resolving it at build time is what
    #: lets every memory system retrieve on the real instruction rather than on
    #: "boil/21". Unlike ALFWorld the description is NOT part of the opening
    #: observation, so the agent is shown it separately.
    instruction: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "split": self.split,
            "task_name": self.task_name,
            "variation": self.variation,
            "task_family": self.task_family,
            "instruction": self.instruction,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any], default_split: str = "train") -> "TaskSpec":
        task_name = str(d.get("task_name") or d.get("task") or "")
        if task_name not in TASK_NAMES:
            raise ValueError(f"manifest entry has no known ScienceWorld task name: {d}")
        variation = int(d["variation"])
        return cls(
            task_id=str(d.get("task_id") or f"{task_name}/{variation}"),
            split=str(d.get("split") or default_split),
            task_name=task_name,
            variation=variation,
            task_family=str(d.get("task_family") or task_name),
            instruction=str(d.get("instruction") or ""),
        )


def load_manifest(path: str | Path) -> list[TaskSpec]:
    """Read a manifest (`{"tasks": [...]}`) or a bare list."""
    p = Path(path)
    value = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(value, list):
        return [TaskSpec.from_dict(item) for item in value]
    if not isinstance(value, dict) or not isinstance(value.get("tasks"), list):
        raise ValueError(f"unsupported manifest format: {p}")
    split = str(value.get("split") or "train")
    return [TaskSpec.from_dict(item, default_split=split) for item in value["tasks"]]


# ============================================================== environment
class ScienceWorldEnvironment:
    """One ScienceWorld simulator (one JVM), reused across tasks.

    Deliberately NOT one env per task, which is what the ALFWorld adapter does:
    there a fresh env costs a PDDL parse, here it costs a JVM launch plus a
    socket handshake (seconds), and `load()` fully re-initialises the world
    anyway. One long-lived instance per process is the difference between a JVM
    per episode and a JVM per worker.
    """

    def __init__(self, max_steps: int = 100):
        self.max_steps = int(max_steps)
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self._env: Any = None
        self._steps = 0

    def _ensure_env(self) -> Any:
        if self._env is None:
            _ensure_java_on_path()
            from scienceworld import ScienceWorldEnv

            # envStepLimit is passed for completeness but is NOT what stops an
            # episode -- see module docstring (2); self.max_steps is.
            self._env = ScienceWorldEnv("", envStepLimit=self.max_steps)
        return self._env

    def reset(self, task: TaskSpec) -> str:
        env = self._ensure_env()
        env.load(task.task_name, int(task.variation), "")
        observation, _info = env.reset()
        observation = str(observation)
        if _VARIATION_ERROR in observation:
            # Refuse rather than return it: downstream this string is just a bad
            # observation, and the episode would be recorded as an agent failure.
            raise ValueError(
                f"{task.task_id}: variation {task.variation} does not exist for task "
                f"{task.task_name!r} -- the manifest is stale or was built against "
                f"another task's variation list ({observation.strip()})"
            )
        self._steps = 0
        return observation

    def task_description(self) -> str:
        return str(self._ensure_env().get_task_description()).strip()

    def step(self, action: str) -> dict[str, Any]:
        if self._env is None:
            raise RuntimeError("reset() must be called before step()")
        observation, _reward, done, info = self._env.step(str(action).strip())
        self._steps += 1
        score = int(info.get("score", 0))
        native_done = bool(done)
        truncated = self._steps >= self.max_steps and not native_done
        return {
            "observation": str(observation),
            # A -100 score is the simulator's "you did something that makes the
            # task unachievable" signal. Clipping at 0 is the usual convention
            # for reporting; the raw value is kept so the failure is countable.
            "score": score,
            "reward": max(0, score) / 100.0,
            "success": score >= 100,
            "task_failed": score < 0,
            "done": native_done or truncated,
            "truncated": truncated,
        }

    def close(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            except Exception:  # noqa: BLE001 - a dead JVM must not fail a sweep
                pass
            finally:
                self._env = None

    def __enter__(self) -> "ScienceWorldEnvironment":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


# =================================================================== agent
from .alfworld import parse_response  # noqa: E402,F401  (identical ReAct format)


def load_demonstrations(path: str | Path | None = None) -> list[dict[str, str]]:
    """The worked ReAct trajectories, flattened into chat messages."""
    p = Path(path) if path else DEFAULT_DEMOS
    entries = json.loads(p.read_text(encoding="utf-8"))
    messages: list[dict[str, str]] = []
    for entry in entries:
        messages.extend(
            {"role": m["role"], "content": m["content"]} for m in entry["example"]
        )
    return messages


#: Same rationale as the ALFWorld adapter: stop the model before it can invent
#: an environment response. A memory entry distilled from a hallucinated
#: observation is grounded in nothing while still passing an evidence check
#: against the model's own text.
STOP_SEQUENCES = ["\nObservation:", "\nObservation :"]


@dataclass
class AgentConfig:
    #: 100 is ScienceWorld's own default budget and what published baselines
    #: use. It is double ALFWorld's 50 because several task families (grow-fruit,
    #: mendelian-genetics) have gold paths of 90-180 actions -- at 50 those
    #: families would be unreachable for every arm, which would look like a
    #: memory result and would not be one.
    max_steps: int = 100
    #: 256, same as ALFWorld, and the "at most two sentences" line in
    #: SYSTEM_PROMPT is what makes that survivable here. Measured on three eval
    #: tasks x 14 steps, before that line existed: 256 truncated the completion
    #: mid-Thought on 21% of turns, and a truncated turn carries no `Action:` at
    #: all, so the agent fell back to `look around` and burned the step. Raising
    #: the cap to 512 cut it to 3%; adding the brevity instruction and KEEPING
    #: 256 cut it to 0%. The brevity fix is preferred over the bigger budget
    #: because the cap is also what bounds context growth -- SAGE lost a whole
    #: run to prompts reaching 129k tokens with 63.6% of calls dying on HTTP 400.
    max_tokens: int = 256
    temperature: float = 0.7
    #: leave headroom under the server's --max-model-len before trimming history
    context_limit_tokens: int = 28000
    #: never trim below this many recent step pairs
    min_history_steps: int = 6
    #: ScienceWorld room descriptions are long and repetitive; an untrimmed
    #: 100-step episode is far larger than an ALFWorld one at the same step
    #: count, so observations are capped per turn.
    max_observation_chars: int = 1500


class ScienceWorldAgent:
    """ReAct agent over `ScienceWorldEnvironment`, with one memory block injected."""

    def __init__(
        self,
        client: Any,
        model: str,
        demonstrations: Sequence[dict[str, str]] | None = None,
        config: AgentConfig | None = None,
    ):
        self.client = client
        self.model = model
        self.demos = list(demonstrations if demonstrations is not None else load_demonstrations())
        self.config = config or AgentConfig()

    # ------------------------------------------------------------ prompting
    def _opening_user_turn(self, task_desc: str, observation: str, memory_block: str) -> str:
        parts = []
        if memory_block.strip():
            parts.append(f"{MEMORY_HEADER}\n{memory_block.strip()}")
        parts.append(
            "Now, it's your turn to solve a new task.\n"
            f"Task: {task_desc.strip()}\n\n{observation.strip()}"
        )
        return "\n\n".join(parts)

    def _clip(self, observation: str) -> str:
        obs = observation.strip()
        limit = self.config.max_observation_chars
        return obs if len(obs) <= limit else obs[:limit] + " ...<truncated>"

    def _trim(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """Drop the oldest step pairs if the conversation approaches the window."""
        head = 1 + len(self.demos) + 1  # system + demos + opening user turn
        total = sum(count_tokens(m["content"]) for m in messages)
        while total > self.config.context_limit_tokens and (
            len(messages) - head > 2 * self.config.min_history_steps
        ):
            dropped = messages[head : head + 2]
            del messages[head : head + 2]
            total -= sum(count_tokens(m["content"]) for m in dropped)
        return messages

    # ----------------------------------------------------------------- loop
    def run(
        self,
        env: ScienceWorldEnvironment,
        task: TaskSpec,
        memory_block: str = "",
        rollout_id: str = "r0",
        temperature: float | None = None,
    ) -> Rollout:
        initial_observation = observation = env.reset(task)
        # Prefer the manifest's text so the key written to memory is byte-identical
        # to the key retrieval used; fall back to asking the simulator.
        task_desc = task.instruction or env.task_description()
        messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(self.demos)
        messages.append(
            {"role": "user",
             "content": self._opening_user_turn(task_desc, self._clip(observation), memory_block)}
        )

        steps: list[Step] = []
        success, reward, error = False, 0.0, None
        score = 0
        parse_failures = 0
        not_understood = 0
        task_failed = False
        prompt_tokens = completion_tokens = 0

        for _ in range(self.config.max_steps):
            self._trim(messages)
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.config.temperature if temperature is None else temperature,
                    max_tokens=self.config.max_tokens,
                    stop=STOP_SEQUENCES,
                )
            except Exception as exc:  # noqa: BLE001
                # Record rather than die: a truncated episode is data, a crashed
                # sweep is not. `error` propagates into the Episode.
                error = f"llm_error: {type(exc).__name__}: {exc}"
                break
            usage = getattr(resp, "usage", None)
            prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
            completion_tokens += getattr(usage, "completion_tokens", 0) or 0
            text = resp.choices[0].message.content or ""

            thought, action = parse_response(text)
            if not action:
                parse_failures += 1
                # "look around" is always valid and costs one step, so a format
                # slip degrades the episode instead of aborting it.
                action = "look around"

            result = env.step(action)
            observation = result["observation"]
            if "no known action matches" in observation.lower():
                not_understood += 1
            steps.append(Step(action=action, observation=self._clip(observation), thought=thought))

            messages.append({"role": "assistant", "content": text})
            messages.append(
                {"role": "user", "content": f"Observation: {self._clip(observation)}"}
            )

            score = result["score"]
            reward = result["reward"]
            if result["task_failed"]:
                task_failed = True
            if result["done"]:
                success = result["success"]
                if result["truncated"]:
                    error = error or "step_limit_reached"
                elif task_failed:
                    # Distinct from running out of steps: the episode is over
                    # because an action made the goal unreachable, which is the
                    # dominant ScienceWorld failure mode and the one a memory
                    # system has the clearest shot at preventing.
                    error = error or "task_failed_by_action"
                break
        else:
            error = error or "step_limit_reached"

        return Rollout(
            rollout_id=rollout_id,
            steps=steps,
            reward=reward,
            success=success,
            error=error,
            meta={
                "task_id": task.task_id,
                "task_family": task.task_family,
                "task_name": task.task_name,
                "variation": task.variation,
                "initial_observation": initial_observation,
                "score": score,
                "task_failed_by_action": task_failed,
                "n_steps": len(steps),
                "parse_failures": parse_failures,
                # Kept under the ALFWorld name so summarize.py and the episode
                # rows stay uniform across benchmarks; here it counts actions the
                # simulator's parser rejected outright.
                "nothing_happens": not_understood,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "injected_memory_tokens": count_tokens(memory_block),
            },
        )


# ================================================================ episodes
def run_task(
    agent: ScienceWorldAgent,
    task: TaskSpec,
    memory_block: str = "",
    n_rollouts: int = 1,
    temperature: float | None = None,
    env: ScienceWorldEnvironment | None = None,
) -> Episode:
    """Run `n_rollouts` independent attempts at one task and package them.

    `env` is threaded through rather than created here so a caller that runs many
    tasks in one process pays for one JVM instead of one per task. When it is
    omitted a private environment is created and closed, which is what the
    manifest builder and the tests want.
    """
    owned = env is None
    env = env or ScienceWorldEnvironment(max_steps=agent.config.max_steps)
    try:
        rollouts: list[Rollout] = [
            agent.run(
                env, task, memory_block=memory_block,
                rollout_id=f"{task.task_id}#{i}", temperature=temperature,
            )
            for i in range(n_rollouts)
        ]
    finally:
        if owned:
            env.close()
    instruction = task.instruction or rollouts[0].meta.get("task_desc", "") or task.task_id
    return Episode(
        task_id=task.task_id,
        instruction=instruction,
        rollouts=rollouts,
        scope={"env": "scienceworld", "task_type": task.task_family},
        meta={"split": task.split, "task_name": task.task_name, "variation": task.variation},
    )

"""ALFWorld adapter: a ReAct agent loop that produces `Episode` objects.

Scaffold provenance (this matters for comparability)
----------------------------------------------------
The system prompt and the six worked demonstrations are taken verbatim from the
SAGE harness (`SAGE/repos/MemRL/memrl/agent/prompts.py` and
`configs/alfworld/alfworld_examples.json`). SAGE measured a no-evolution
Qwen3.5-9B baseline of 58-60% on 50 `valid_unseen` tasks with exactly this
scaffold, and it also recorded what happens when you change it: ACE's generic
"you are an analysis expert" prompt with zero demonstrations scored **18%** on
the same model and tasks. A 40-point swing from prompt wording alone dwarfs any
memory effect we are trying to measure, so the scaffold is held fixed and the
memory block is the *only* thing that differs between arms.

The environment wrapper mirrors `sage_harness.alfworld.environment`: it registers
one exact compiled `game.tw-pddl` with TextWorld rather than going through any
framework's random sampler, so a manifest task is the same task on every run.

Action protocol: standard ALFWorld ReAct. The agent is NOT shown the admissible
action list -- it emits free text and the environment answers "Nothing happens."
on a miss. This is what SAGE's baseline numbers were measured under.
"""

from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from ..episode import Episode, Rollout, Step
from ..tokens import count_tokens

HERE = Path(__file__).resolve().parent
DEFAULT_DEMOS = HERE / "alfworld_examples.json"

#: Verbatim from SAGE/repos/MemRL/memrl/agent/prompts.py. Do not reword: the
#: 58-60% reference baseline is a property of this exact text.
SYSTEM_PROMPT = """Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish.
For each of your turn, you will be given the observation of the last turn. You should first think about the current condition and plan for your future actions, and then output your action in this turn. Your output must strictly follow this format:"Thought: your thoughts.\nAction: your next action".

The available actions are:
1. go to {recep}
2. take {obj} from {recep}
3. move {obj} to {recep}
4. open {recep}
5. close {recep}
6. use {obj}
7. clean {obj} with {recep}
8. heat {obj} with {recep}
9. cool {obj} with {recep}
where {obj} and {recep} correspond to objects and receptacles.
After your each turn, the environment will give you immediate feedback based on which you plan your next few steps. if the envrionment output "Nothing happened", that means the previous action is invalid and you should try more options.

Your response should use the following format:

Thought: <your thoughts>
Action: <your next action>"""

MEMORY_HEADER = (
    "**Archived memory (distilled from your own past episodes in this environment):**"
)

# ALFWorld's six task families, as they appear in the compiled task directory
# name. `task_family` becomes the memsys retrieval scope key `task_type`, which
# is also what batch induction clusters on (MemoryConfig.cluster_key).
TASK_FAMILIES = (
    "pick_and_place_simple",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "look_at_obj_in_light",
    "pick_two_obj_and_place",
)


# ============================================================ task manifests
@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    split: str
    gamefile: str
    task_family: str = ""
    #: The "Your task is to: ..." goal, baked in by scripts/build_manifests.py.
    #: Memory retrieval happens BEFORE the rollout, but the goal is only revealed
    #: by env.reset(), so it has to be resolved ahead of time or every system
    #: would have to retrieve on a weak proxy key (the gamefile path).
    instruction: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "task_id": self.task_id,
            "split": self.split,
            "gamefile": self.gamefile,
            "task_family": self.task_family,
            "instruction": self.instruction,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any], default_split: str = "train") -> "TaskSpec":
        gamefile = str(d.get("gamefile") or d.get("game_file") or d.get("path") or "")
        if not gamefile:
            raise ValueError(f"manifest entry has no gamefile: {d}")
        return cls(
            task_id=str(d.get("task_id") or gamefile),
            split=str(d.get("split") or default_split),
            gamefile=gamefile,
            task_family=str(d.get("task_family") or _family_from_path(gamefile)),
            instruction=str(d.get("instruction") or ""),
        )


def _family_from_path(gamefile: str) -> str:
    """ALFWorld encodes the task family in the trial directory name."""
    for part in Path(gamefile).parts:
        for fam in TASK_FAMILIES:
            if part.startswith(fam):
                return fam
    return ""


def load_manifest(path: str | Path) -> list[TaskSpec]:
    """Read a SAGE-format manifest (`{"tasks": [...]}`) or a bare list."""
    p = Path(path)
    value = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(value, list):
        return [TaskSpec.from_dict(item) for item in value]
    if not isinstance(value, dict) or not isinstance(value.get("tasks"), list):
        raise ValueError(f"unsupported manifest format: {p}")
    split = str(value.get("split") or "train")
    return [TaskSpec.from_dict(item, default_split=split) for item in value["tasks"]]


# ============================================================== environment
#: Serialises game registration AND the first reset(), because both touch
#: process-global state that is not thread-safe:
#:
#:   * `textworld.gym.register_game` writes to a global registry dict;
#:   * `reset()` is what actually loads the game, and for a .tw-pddl that means
#:     `textworld.envs.pddl.logic.GameLogic` parsing the PDDL grammar with a
#:     MODULE-LEVEL tatsu parser. Two threads parsing at once interleave on its
#:     shared `_rule_stack` and fail as `IndexError: pop from empty list` or
#:     `TypeError: 'NoneType' object is not iterable` -- deep inside tatsu, with
#:     nothing in the traceback pointing at concurrency.
#:
#: Parsing costs a second or two against a 60s+ episode, so serialising it is
#: cheap; the alternative is process-level parallelism for a rounding error.
_REGISTER_LOCK = threading.Lock()


class ALFWorldEnvironment:
    """One exact compiled ALFWorld gamefile, with binary success as the reward."""

    def __init__(self, data_root: str | Path, max_steps: int = 50):
        self.data_root = Path(data_root).expanduser().resolve()
        self.max_steps = int(max_steps)
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self._env: Any = None
        self._task: TaskSpec | None = None
        self._steps = 0
        self._observation = ""

    @staticmethod
    def _first(value: Any) -> Any:
        """TextWorld's batch API returns length-1 sequences for batch_size=1."""
        if isinstance(value, (list, tuple)):
            return value[0]
        try:
            if getattr(value, "shape", None) and len(value):
                return value[0]
        except TypeError:
            pass
        return value

    def resolve_gamefile(self, task: TaskSpec) -> Path:
        gamefile = Path(os.path.expandvars(task.gamefile)).expanduser()
        if not gamefile.is_absolute():
            gamefile = self.data_root / gamefile
        gamefile = gamefile.resolve()
        if not gamefile.is_file():
            raise FileNotFoundError(f"ALFWorld gamefile does not exist: {gamefile}")
        return gamefile

    def reset(self, task: TaskSpec) -> str:
        import textworld
        import textworld.gym
        from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos

        self.close()
        gamefile = self.resolve_gamefile(task)
        request_infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile"])
        with _REGISTER_LOCK:
            env_id = textworld.gym.register_game(
                str(gamefile),
                request_infos=request_infos,
                batch_size=1,
                asynchronous=False,
                max_episode_steps=self.max_steps,
                # shuffle=False keeps object numbering deterministic across arms,
                # so every arm really sees the identical task instance.
                wrappers=[AlfredDemangler(shuffle=False), AlfredInfos()],
            )
            self._env = textworld.gym.make(env_id)
            observations, _ = self._env.reset()
        self._task = task
        self._steps = 0
        self._observation = str(self._first(observations))
        return self._observation

    def step(self, action: str) -> dict[str, Any]:
        if self._env is None:
            raise RuntimeError("reset() must be called before step()")
        observations, _, dones, infos = self._env.step([str(action).strip()])
        self._steps += 1
        self._observation = str(self._first(observations))
        won = bool(self._first(infos.get("won", False)))
        native_done = bool(self._first(dones))
        truncated = self._steps >= self.max_steps and not native_done
        return {
            "observation": self._observation,
            "success": won,
            "reward": 1.0 if won else 0.0,
            "done": native_done or truncated,
            "truncated": truncated,
        }

    def close(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            finally:
                self._env = None

    def __enter__(self) -> "ALFWorldEnvironment":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


# =================================================================== agent
_TASK_RE = re.compile(r"your task is to:\s*(.+)", re.IGNORECASE)
_ACTION_RE = re.compile(r"Action:\s*(.+?)(?:\n|$)", re.IGNORECASE | re.DOTALL)
_THOUGHT_RE = re.compile(r"Thought:\s*(.+?)(?:\n\s*Action:|$)", re.IGNORECASE | re.DOTALL)


def extract_instruction(observation: str) -> str:
    """The 'Your task is to: ...' line -- the retrieval key for every system."""
    m = _TASK_RE.search(observation)
    return m.group(1).strip() if m else observation.strip().split("\n")[-1].strip()


def parse_response(text: str) -> tuple[str, str]:
    """Return (thought, action). action is "" when the format was not followed.

    The FIRST `Action:` wins, not the last. When a model emits several, it has
    hallucinated its own continuation ("Action: go to desk 1 / Observation: ... /
    Action: take mug 1") -- the later ones are steps it imagines taking after
    observations that never happened, so executing one would skip the episode
    ahead. STOP_SEQUENCES makes this rare; this rule handles the remainder.
    """
    text = str(text or "")
    action_m = _ACTION_RE.search(text)
    action = action_m.group(1).strip() if action_m else ""
    thought_m = _THOUGHT_RE.search(text)
    thought = thought_m.group(1).strip() if thought_m else ""
    if not action and not thought:
        # Some responses are a bare action with no ReAct scaffolding at all.
        first = text.strip().split("\n")[0].strip()
        if first and len(first) < 120:
            action = first
    return thought, action


def load_demonstrations(path: str | Path | None = None) -> list[dict[str, str]]:
    """The six worked ReAct trajectories, flattened into chat messages."""
    p = Path(path) if path else DEFAULT_DEMOS
    entries = json.loads(p.read_text(encoding="utf-8"))
    messages: list[dict[str, str]] = []
    for entry in entries:
        messages.extend(
            {"role": m["role"], "content": m["content"]} for m in entry["example"]
        )
    return messages


#: Cut generation off before the model can invent an environment response.
#: Hallucinated observations are not just wasted tokens here: the trajectory is
#: what the memory writers read, and an entry distilled from an imagined
#: observation would be grounded in nothing while still passing an evidence
#: check against the model's own text.
STOP_SEQUENCES = ["\nObservation:", "\nObservation :"]


@dataclass
class AgentConfig:
    max_steps: int = 50
    # 256 is a hard cap on context growth, and it is deliberate. SAGE lost a whole
    # run to this: with max_tokens=2048 and no history management, Qwen filled the
    # ceiling every turn, prompts reached 129k tokens and 63.6% of calls died with
    # HTTP 400 -- silently, because the agent swallowed the error and kept going.
    # A ReAct Thought+Action needs far less than 256.
    max_tokens: int = 256
    temperature: float = 0.7  # Qwen3.5 non-thinking recommendation
    #: leave headroom under the server's --max-model-len before trimming history
    context_limit_tokens: int = 28000
    #: never trim below this many recent step pairs
    min_history_steps: int = 6


class AlfworldAgent:
    """ReAct agent over `ALFWorldEnvironment`, with one memory block injected."""

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
    def _opening_user_turn(self, observation: str, memory_block: str) -> str:
        parts = []
        if memory_block.strip():
            parts.append(f"{MEMORY_HEADER}\n{memory_block.strip()}")
        parts.append(f"Now, it's your turn to solve a new task.\n{observation.strip()}")
        return "\n\n".join(parts)

    def _trim(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """Drop the oldest step pairs if the conversation approaches the window.

        With max_tokens=256 this should essentially never fire; it exists so that
        a pathological episode degrades instead of returning HTTP 400 forever.
        """
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
        env: ALFWorldEnvironment,
        task: TaskSpec,
        memory_block: str = "",
        rollout_id: str = "r0",
        temperature: float | None = None,
    ) -> Rollout:
        initial_observation = observation = env.reset(task)
        messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(self.demos)
        messages.append(
            {"role": "user", "content": self._opening_user_turn(observation, memory_block)}
        )

        steps: list[Step] = []
        success, reward, error = False, 0.0, None
        parse_failures = 0
        nothing_happens = 0
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
                # Record it rather than dying: a truncated episode is data, a
                # crashed sweep is not. `error` propagates into the Episode.
                error = f"llm_error: {type(exc).__name__}: {exc}"
                break
            usage = getattr(resp, "usage", None)
            prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
            completion_tokens += getattr(usage, "completion_tokens", 0) or 0
            text = resp.choices[0].message.content or ""

            thought, action = parse_response(text)
            if not action:
                parse_failures += 1
                # "look" is always valid and costs one step, so a format slip
                # degrades the episode instead of aborting it.
                action = "look"

            result = env.step(action)
            observation = result["observation"]
            if "nothing happens" in observation.lower():
                nothing_happens += 1
            steps.append(Step(action=action, observation=observation, thought=thought))

            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": f"Observation: {observation.strip()}"})

            if result["done"]:
                success, reward = result["success"], result["reward"]
                if result["truncated"]:
                    error = error or "step_limit_reached"
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
                "initial_observation": initial_observation,
                "n_steps": len(steps),
                "parse_failures": parse_failures,
                "nothing_happens": nothing_happens,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "injected_memory_tokens": count_tokens(memory_block),
            },
        )


# ================================================================ episodes
def run_task(
    agent: AlfworldAgent,
    data_root: str | Path,
    task: TaskSpec,
    memory_block: str = "",
    n_rollouts: int = 1,
    temperature: float | None = None,
) -> Episode:
    """Run `n_rollouts` independent attempts at one task and package them.

    n_rollouts > 1 is what makes `Episode.outcome()` able to return MIXED, which
    is the only case in which the Reflection writer's `from_contrast` mode fires.
    """
    rollouts: list[Rollout] = []
    with ALFWorldEnvironment(data_root, max_steps=agent.config.max_steps) as env:
        for i in range(n_rollouts):
            rollouts.append(
                agent.run(
                    env, task, memory_block=memory_block,
                    rollout_id=f"{task.task_id}#{i}", temperature=temperature,
                )
            )
    # Prefer the manifest's instruction so the key written to memory is byte-identical
    # to the key retrieval used; fall back to parsing the initial observation.
    instruction = task.instruction or extract_instruction(rollouts[0].meta["initial_observation"])
    return Episode(
        task_id=task.task_id,
        instruction=instruction,
        rollouts=rollouts,
        scope={"env": "alfworld", "task_type": task.task_family},
        meta={"split": task.split, "gamefile": task.gamefile},
    )

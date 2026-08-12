"""AppWorld adapter: a code-writing ReAct agent that produces `Episode` objects.

Same contract as the ALFWorld and WebShop adapters -- TaskSpec/manifest in,
`Episode` out -- so the memory systems are unchanged. What differs here changes
how the results must be read.

**The agent writes Python, not actions.** Each turn it emits a ```python block
that runs in a stateful REPL against ~9 apps' REST APIs, and the environment
returns stdout. Variables persist across turns. A task ends when the agent calls
`apis.supervisor.complete_task()`, not when the environment decides.

**Scaffold provenance.** The prompt in `appworld_react_prompt.txt` is ACE's
published AppWorld ReAct generator prompt, copied verbatim from
`SAGE/repos/ACE/ace-appworld/experiments/prompts/`. It already contains a
`{{ playbook }}` slot -- ACE's own name for injected context -- so the memory
block goes exactly where that harness put its context, rather than somewhere this
study invented. This is the ALFWorld situation (byte-identical to SAGE) rather
than the WebShop one (written here). The slot is filled for *every* arm,
including `none`, which gets an explicit "no entries yet" placeholder: keeping
the surrounding 7k-token scaffold byte-identical is what makes the arms
comparable, and a missing section is a prompt change like any other.

**Two metrics again, and they are AppWorld's own.** `/evaluate` runs the task's
unit tests. `success` (all tests pass) is Task Goal Completion, the strict
number; `passes / num_tests` is the graded one. As on WebShop they can move
apart, so both are reported.

**One task per server process.** `appworld serve environment` keeps a single
module-level `world`, so a server can host exactly one live task. Concurrency
comes from a pool of servers with exclusive leases, not from threads sharing one.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ..episode import Episode, Rollout, Step
from ..tokens import count_tokens

HERE = Path(__file__).resolve().parent
DEFAULT_PROMPT = HERE / "appworld_react_prompt.txt"

SCOPE_ENV = "appworld"

#: What `none` puts in the `{{ playbook }}` slot. The slot cannot simply be
#: dropped: the paragraph introducing it is part of ACE's prompt, and removing
#: either would make the no-memory arm a different scaffold rather than the same
#: scaffold with an empty memory.
EMPTY_PLAYBOOK = "(No playbook entries yet -- rely on the instructions and the worked example above.)"

MEMORY_HEADER = ""  # AppWorld's scaffold has its own header ("### PLAYBOOK BEGIN")


# ============================================================ task manifests
@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    split: str
    instruction: str = ""
    #: AppWorld task ids are `<scenario>_<variant>`; tasks sharing a scenario are
    #: variations of one request. Used as the memsys `task_type` scope key, which
    #: `MemoryConfig.cluster_key` clusters on during batch induction.
    #:
    #: Deliberately NOT the task's required apps, even though that would be the
    #: more useful grouping: `required_apps.json` lives under `ground_truth/`, and
    #: putting it in the scope key would hand every memory system a part of the
    #: solution path for free. Difficulty is reported instead, post-hoc, from the
    #: evaluator -- see `Rollout.meta["difficulty"]`.
    scenario: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "split": self.split,
            "instruction": self.instruction,
            "scenario": self.scenario,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any], default_split: str = "train") -> "TaskSpec":
        task_id = str(d.get("task_id") or "")
        if not task_id:
            raise ValueError(f"manifest entry has no task_id: {d}")
        return cls(
            task_id=task_id,
            split=str(d.get("split") or default_split),
            instruction=str(d.get("instruction") or ""),
            scenario=str(d.get("scenario") or task_id.split("_")[0]),
        )

    def scope(self) -> dict[str, str]:
        return {"env": SCOPE_ENV, "task_type": self.scenario}


def load_manifest(path: str | Path) -> list[TaskSpec]:
    p = Path(path)
    value = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(value, list):
        return [TaskSpec.from_dict(item) for item in value]
    if not isinstance(value, dict) or not isinstance(value.get("tasks"), list):
        raise ValueError(f"unsupported manifest format: {p}")
    split = str(value.get("split") or "train")
    return [TaskSpec.from_dict(item, default_split=split) for item in value["tasks"]]


# ============================================================== environment
class AppWorldError(RuntimeError):
    pass


class AppWorldEnvironment:
    """Client for one `appworld serve environment` process.

    The server holds a single module-level `world`, so an instance of this class
    owns its server for the duration of an episode. `run_task` leases a URL from
    a pool and returns it afterwards; nothing else may touch that URL meanwhile.
    """

    def __init__(
        self,
        base_url: str,
        max_interactions: int = 30,
        timeout: float = 600.0,
        experiment_name: str = "memsys",
    ):
        self.base_url = str(base_url).rstrip("/")
        self.max_interactions = int(max_interactions)
        if self.max_interactions <= 0:
            raise ValueError("max_interactions must be positive")
        self.timeout = float(timeout)
        self.experiment_name = experiment_name
        self.task_id: str | None = None
        self.task: dict[str, Any] = {}
        self._steps = 0

    # -- transport --
    def _call(self, route: str, payload: dict | None = None, method: str = "POST") -> Any:
        data = json.dumps(payload or {}).encode("utf-8") if method == "POST" else None
        req = Request(self.base_url + route, data=data,
                      headers={"Content-Type": "application/json"}, method=method)
        try:
            with urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            raise AppWorldError(f"{route} -> HTTP {exc.code}: {detail[:2000]}") from exc
        except URLError as exc:
            raise AppWorldError(f"{route} -> {exc}") from exc
        return body.get("output", body) if isinstance(body, dict) else body

    def health(self) -> dict:
        return self._call("/", method="GET")

    def show_app_descriptions(self) -> str:
        """The short app list the demonstration's `{{ app_descriptions }}` slot wants.

        This must come from executing `apis.api_docs.show_app_descriptions()`,
        not from the server's `GET /api_docs`. That route returns the *complete*
        API reference for all nine apps -- 519 KB, ~140k tokens -- which silently
        turns a 7k-token scaffold into one that cannot fit in the context window
        at all. The names are similar enough that the mistake looks like a
        working prompt right up until every request fails.
        """
        return self.execute("print(apis.api_docs.show_app_descriptions())")

    def reset(self, task: TaskSpec) -> dict[str, Any]:
        self.task = self._call("/initialize", {
            "task_id": task.task_id,
            "experiment_name": self.experiment_name,
            "max_interactions": self.max_interactions,
            # The agent is an untrusted LLM writing arbitrary Python. Let the
            # environment neutralise unsafe syntax rather than abort the episode:
            # a killed episode is a missing data point, a null-patched one is a
            # failure we can read.
            "raise_on_unsafe_syntax": False,
            "null_patch_unsafe_execution": True,
        })
        self.task_id = task.task_id
        self._steps = 0
        return self.task

    def execute(self, code: str) -> str:
        if self.task_id is None:
            raise RuntimeError("reset() must be called before execute()")
        self._steps += 1
        out = self._call("/execute", {"task_id": self.task_id, "code": code})
        return out if isinstance(out, str) else json.dumps(out)

    def task_completed(self) -> bool:
        return bool(self._call("/task_completed", {"task_id": self.task_id}))

    def evaluate(self) -> dict[str, Any]:
        """Run the task's unit tests.

        `suppress_errors=True` is required, not optional: the route defaults it
        to False, and an unfinished task makes the evaluator raise on the first
        failed assertion, which surfaces as an opaque HTTP 500. Every incomplete
        episode -- the common case -- would be lost.
        """
        out = self._call("/evaluate", {"task_id": self.task_id, "suppress_errors": True})
        passes = out.get("passes") or []
        failures = out.get("failures") or []
        n = int(out.get("num_tests") or (len(passes) + len(failures)) or 0)
        return {
            # Task Goal Completion: every test passes.
            "success": bool(out.get("success")),
            # The graded number: fraction of the task's tests that pass.
            "score": (len(passes) / n) if n else 0.0,
            "num_tests": n,
            "n_passed": len(passes),
            "difficulty": out.get("difficulty"),
            "failures": [f.get("requirement", "") for f in failures],
        }

    def close(self) -> None:
        if self.task_id is not None:
            try:
                self._call("/close", {"task_id": self.task_id})
            except AppWorldError:
                pass  # a dead server must not mask the episode's own outcome
            finally:
                self.task_id = None

    def __enter__(self) -> "AppWorldEnvironment":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


# =================================================================== prompt
_ROLE_SPLIT = re.compile(r"^(USER|ASSISTANT):\s*$", re.M)
_CODE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.S)


def load_prompt_messages(path: str | Path | None = None) -> list[dict[str, str]]:
    """Parse ACE's transcript-style prompt file into chat messages.

    The file is a literal USER:/ASSISTANT: transcript. Consecutive same-role
    blocks are merged: the tail of ACE's prompt is three USER blocks in a row,
    and some chat templates reject or silently reorder repeated roles.
    """
    text = Path(path or DEFAULT_PROMPT).read_text(encoding="utf-8")
    parts = _ROLE_SPLIT.split(text)
    roles, bodies = parts[1::2], parts[2::2]
    messages: list[dict[str, str]] = []
    for role, body in zip(roles, bodies):
        role = "user" if role == "USER" else "assistant"
        body = body.strip("\n")
        if messages and messages[-1]["role"] == role:
            messages[-1]["content"] += "\n\n" + body
        else:
            messages.append({"role": role, "content": body})
    return messages


def render_prompt(
    messages: Sequence[dict[str, str]],
    task: dict[str, Any],
    app_descriptions: str,
    playbook: str,
) -> list[dict[str, str]]:
    """Substitute ACE's `{{ ... }}` slots. `playbook` is the memory block."""
    supervisor = task.get("supervisor") or {}
    subs = {
        "{{ playbook }}": playbook.strip() or EMPTY_PLAYBOOK,
        "{{ app_descriptions }}": app_descriptions,
        "{{ input_str }}": str(task.get("instruction") or ""),
        "{{ main_user.first_name }}": str(supervisor.get("first_name") or ""),
        "{{ main_user.last_name }}": str(supervisor.get("last_name") or ""),
        "{{ main_user.email }}": str(supervisor.get("email") or ""),
        "{{ main_user.phone_number }}": str(supervisor.get("phone_number") or ""),
    }
    out = []
    for m in messages:
        content = m["content"]
        for k, v in subs.items():
            content = content.replace(k, v)
        out.append({"role": m["role"], "content": content})
    return out


def extract_code(text: str) -> str:
    """Return the first fenced code block, or "" if the format was not followed.

    The FIRST block wins, for the same reason the other two adapters take the
    first action: a model that emits several has hallucinated its own
    continuation past outputs that never happened, and running a later block
    would execute code written against an imagined REPL state. Unlike a wrong
    ALFWorld action, that can mutate the world irreversibly -- these APIs send
    money and delete files.
    """
    m = _CODE_RE.search(str(text or ""))
    return m.group(1).strip() if m else ""


#: Stop before the model can invent the environment's reply and keep going.
STOP_SEQUENCES = ["\nUSER:", "\nOutput:\n```"]


@dataclass
class AgentConfig:
    #: AppWorld's own default is 50. 30 is a deliberate cost cut: the scaffold is
    #: ~7k tokens before any history, Qwen3.5 is a hybrid model so vLLM disables
    #: prefix caching and re-prefills the whole conversation every turn, and cost
    #: therefore grows quadratically in interactions. Recorded in config.json;
    #: results are not comparable to a 50-interaction run.
    max_interactions: int = 30
    #: Far above the other adapters' 256: a turn must fit a whole code block.
    max_tokens: int = 768
    temperature: float = 0.7
    #: The server is served with --max-model-len 32768.
    context_limit_tokens: int = 26000
    min_history_steps: int = 3
    #: One `print(show_api_descriptions(...))` can return tens of thousands of
    #: characters. Truncating per-output is what keeps a single exploratory call
    #: from consuming the whole window.
    max_output_chars: int = 4000


class AppWorldAgent:
    """Code-writing ReAct agent over `AppWorldEnvironment`."""

    def __init__(
        self,
        client: Any,
        model: str,
        app_descriptions: str,
        prompt_messages: Sequence[dict[str, str]] | None = None,
        config: AgentConfig | None = None,
    ):
        self.client = client
        self.model = model
        # Resolved once by `fetch_app_descriptions` and passed in, rather than
        # fetched per episode: obtaining it costs one REPL execution, and doing
        # that inside an episode would spend one of the agent's interactions on
        # something the scaffold is supposed to already know.
        self.app_descriptions = app_descriptions
        self.prompt = list(prompt_messages if prompt_messages is not None else load_prompt_messages())
        self.config = config or AgentConfig()

    def _trim(self, messages: list[dict[str, str]], head: int) -> list[dict[str, str]]:
        """Drop the oldest interaction pairs when the window fills.

        Unlike the other two adapters this fires routinely -- the scaffold alone
        is ~7k tokens and REPL outputs are large -- so `head` protects the whole
        rendered prompt, not just a system message.
        """
        total = sum(count_tokens(m["content"]) for m in messages)
        while total > self.config.context_limit_tokens and (
            len(messages) - head > 2 * self.config.min_history_steps
        ):
            dropped = messages[head : head + 2]
            del messages[head : head + 2]
            total -= sum(count_tokens(m["content"]) for m in dropped)
        return messages

    def _output_turn(self, output: str) -> str:
        out = output.strip()
        if len(out) > self.config.max_output_chars:
            out = out[: self.config.max_output_chars] + "\n...<output truncated>"
        return f"Output:\n```\n{out}\n```"

    def run(
        self,
        env: AppWorldEnvironment,
        task: TaskSpec,
        memory_block: str = "",
        rollout_id: str = "r0",
        temperature: float | None = None,
    ) -> Rollout:
        task_info = env.reset(task)
        messages = render_prompt(
            self.prompt, task_info, self.app_descriptions, memory_block
        )
        head = len(messages)

        steps: list[Step] = []
        error = None
        parse_failures = exec_errors = 0
        prompt_tokens = completion_tokens = 0
        completed = False

        for _ in range(self.config.max_interactions):
            self._trim(messages, head)
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.config.temperature if temperature is None else temperature,
                    max_tokens=self.config.max_tokens,
                    stop=STOP_SEQUENCES,
                )
            except Exception as exc:  # noqa: BLE001
                error = f"llm_error: {type(exc).__name__}: {exc}"
                break
            usage = getattr(resp, "usage", None)
            prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
            completion_tokens += getattr(usage, "completion_tokens", 0) or 0
            text = resp.choices[0].message.content or ""

            code = extract_code(text)
            if not code:
                parse_failures += 1
                # Nudge rather than abort, and spend the interaction: an episode
                # that degrades is data, one that dies is not.
                messages.append({"role": "assistant", "content": text})
                messages.append({"role": "user", "content":
                                 "Output:\n```\nNo code block found. Reply with a single "
                                 "```python ... ``` block.\n```"})
                steps.append(Step(action="", observation="no code block", thought=text[:400]))
                continue

            try:
                output = env.execute(code)
            except AppWorldError as exc:
                error = f"env_error: {exc}"
                break
            if "Traceback (most recent call last)" in output or "Error:" in output:
                exec_errors += 1
            steps.append(Step(action=code, observation=output, thought=text.split("Code:")[0].strip()))

            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": self._output_turn(output)})

            try:
                if env.task_completed():
                    completed = True
                    break
            except AppWorldError as exc:
                error = f"env_error: {exc}"
                break
        else:
            error = error or "interaction_limit_reached"

        # Evaluate regardless of how the episode ended: AppWorld scores the world
        # state, so an agent that did the work but never called complete_task can
        # still pass tests, and one that called it immediately can still fail.
        result = {"success": False, "score": 0.0, "num_tests": 0, "n_passed": 0,
                  "difficulty": None, "failures": []}
        if error is None or error == "interaction_limit_reached":
            try:
                result = env.evaluate()
            except AppWorldError as exc:
                error = error or f"eval_error: {exc}"

        return Rollout(
            rollout_id=rollout_id,
            steps=steps,
            reward=result["score"],
            success=result["success"],
            error=error,
            meta={
                "task_id": task.task_id,
                "task_family": task.scenario,
                "instruction": task_info.get("instruction", ""),
                "n_steps": len(steps),
                "parse_failures": parse_failures,
                "exec_errors": exec_errors,
                # Whether the agent declared itself done, as opposed to running
                # out of interactions. The AppWorld analogue of WebShop's
                # `purchased`: it separates "tried and got it wrong" from "never
                # finished", which are different failures.
                "completed_task": completed,
                "num_tests": result["num_tests"],
                "n_passed": result["n_passed"],
                "difficulty": result["difficulty"],
                "failures": result["failures"][:5],
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "injected_memory_tokens": count_tokens(memory_block),
            },
        )


# ================================================================ episodes
def run_task(
    agent: AppWorldAgent,
    base_url: str,
    task: TaskSpec,
    memory_block: str = "",
    n_rollouts: int = 1,
    temperature: float | None = None,
    timeout: float = 600.0,
    experiment_name: str = "memsys",
) -> Episode:
    """`experiment_name` must be unique per concurrent arm.

    AppWorld writes each task's working databases under
    `experiments/outputs/<experiment_name>/tasks/<task_id>/`, and prepares that
    tree on every `/initialize`. Two arm processes sharing a name therefore race
    on the same directories -- one arm's setup removes what another is about to
    use, and `os.makedirs(..., exist_ok=True)` fails with a FileNotFoundError
    naming a path that plainly should exist. Intermittent, and it kills the arm.
    """
    rollouts: list[Rollout] = []
    with AppWorldEnvironment(
        base_url, max_interactions=agent.config.max_interactions, timeout=timeout,
        experiment_name=experiment_name,
    ) as env:
        for i in range(n_rollouts):
            rollouts.append(
                agent.run(env, task, memory_block=memory_block,
                          rollout_id=f"{task.task_id}#{i}", temperature=temperature)
            )
    instruction = task.instruction or rollouts[0].meta.get("instruction", "")
    return Episode(
        task_id=task.task_id,
        instruction=instruction,
        rollouts=rollouts,
        scope=task.scope(),
        meta={"split": task.split, "scenario": task.scenario},
    )


def fetch_app_descriptions(base_url: str, task_id: str, timeout: float = 600.0,
                           experiment_name: str = "memsys") -> str:
    """Resolve the `{{ app_descriptions }}` slot once, on a throwaway world.

    Initialises the given task purely to get a REPL, runs one call and closes.
    Callers should do this at startup while the server pool is idle -- the server
    hosts a single world at a time, so this briefly occupies one.
    """
    with AppWorldEnvironment(base_url, timeout=timeout, experiment_name=experiment_name) as env:
        env.reset(TaskSpec(task_id, "train"))
        return env.show_app_descriptions()


def default_base_url() -> str:
    return os.environ.get("APPWORLD_SERVER_URL", "http://localhost:9000")

"""WebShop adapter: a ReAct agent loop that produces `Episode` objects.

Structurally this mirrors `adapters/alfworld.py` -- same TaskSpec/manifest shape,
same agent contract, same `run_task` -> `Episode` packaging -- so the memory
systems see one interface and the Memory Content comparison is unchanged. Three
things genuinely differ, and each one changes how results must be read.

**The environment lives in another process.** WebShop is pinned to a 2022
dependency set (python 3.8, `flask==2.1.2`, `spacy==3.3`, `typing_extensions<4.6`)
that cannot coexist with memsys's `sentence-transformers`/`openai`. So the
catalogue is served by `scripts/webshop_server.py` over HTTP and this module is
a client. That also means the 1.18M-product catalogue is loaded once for a whole
sweep instead of once per worker.

**Reward is graded, not binary.** `get_reward` scores attribute, option, type and
price matches, so a rollout can score 0.67 by buying a nearly-right product.
WebShop's literature reports two numbers and so does this adapter: *score* (mean
reward x100) and *success rate* (fraction with reward == 1.0). They can move in
opposite directions -- an arm that buys plausible-but-wrong items faster raises
score and lowers success rate -- so reporting only one of them hides the effect.
`Rollout.success` is the strict one; `Rollout.reward` carries the graded value,
which is also what the memory writers see.

**The scaffold has no external reference point.** The ALFWorld prompt is byte-
identical to SAGE's, which pins that arm's baseline to an independently measured
58-60%. Nothing equivalent exists here: the prompt and demonstration below were
written for this study. So the absolute number is not comparable to published
WebShop results, and the `none` arm is the only reference that means anything.
Every arm shares this scaffold exactly; only the memory block differs.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ..episode import Episode, Rollout, Step
from ..tokens import count_tokens

HERE = Path(__file__).resolve().parent
DEFAULT_DEMOS = HERE / "webshop_examples.json"

SYSTEM_PROMPT = """You are shopping on a text-based online store to satisfy a customer instruction. Your goal is to buy the single product that best matches every requirement in the instruction, including its attributes, its options (size, colour, flavour, count, ...) and its price limit.

For each turn you are given the current page and the actions available on it. First think about what the page tells you and what to do next, then output exactly one action. Your output must strictly follow this format:"Thought: your thoughts.\nAction: your next action".

The available actions are:
1. search[keywords] -- only on the search page. Use short keywords describing the product, not the size, colour or price.
2. click[button] -- click one of the listed buttons. Product ids such as B07XYZ1234 are buttons on a results page; option values such as "blue" or "3 ounce (pack of 1)" are buttons on a product page.
3. click[Buy Now] -- purchase the product currently shown. This ends the episode and is scored, so select the matching options first.
4. think[your reasoning] -- reason without touching the store. The store replies "OK." and the page does not change.

Notes on scoring: the reward is partial. You are credited for matching the product type, the requested attributes, the selected options and the price limit, so buying a close product is much better than buying nothing. Every episode ends in a purchase or in nothing at all -- if you are running out of turns, buy the best candidate you have seen.

Your response should use the following format:

Thought: <your thoughts>
Action: <your next action>"""

#: Same header as the ALFWorld adapter, so the injected block is presented
#: identically in both benchmarks and any difference in effect is not a
#: difference in framing.
MEMORY_HEADER = (
    "**Archived memory (distilled from your own past episodes in this environment):**"
)

#: WebShop's goal `category` -- the coarse department, e.g. "beauty", "garden".
#: This becomes the memsys retrieval scope key `task_type` and the key batch
#: induction clusters on (`MemoryConfig.cluster_key`), playing the role ALFWorld's
#: six task families play there. It is a weaker grouping: ALFWorld families are
#: *procedures* ("heat then place"), while these are *product departments*, and
#: two tasks in "beauty" may need completely different action sequences. Expect
#: less transfer from clustering here, and do not read that as a property of the
#: memory type.
SCOPE_ENV = "webshop"


# ============================================================ task manifests
@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    split: str
    #: Index into the server's shuffled goal list. This is the whole task
    #: identity -- WebShop has no per-task file. It is only meaningful together
    #: with the server's (corpus, human_goals, seed) triple, which is why
    #: `run_webshop.py` records the server's /health output next to the results.
    goal_index: int
    category: str = ""
    instruction: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "split": self.split,
            "goal_index": self.goal_index,
            "category": self.category,
            "instruction": self.instruction,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any], default_split: str = "train") -> "TaskSpec":
        if "goal_index" not in d and "session" not in d:
            raise ValueError(f"manifest entry has no goal_index: {d}")
        idx = int(d.get("goal_index", d.get("session")))
        return cls(
            task_id=str(d.get("task_id") or f"{d.get('split', default_split)}:{idx}"),
            split=str(d.get("split") or default_split),
            goal_index=idx,
            category=str(d.get("category") or ""),
            instruction=str(d.get("instruction") or ""),
        )

    def scope(self) -> dict[str, str]:
        return {"env": SCOPE_ENV, "task_type": self.category}


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
class WebShopError(RuntimeError):
    pass


class WebShopEnvironment:
    """Client for one agent session against `scripts/webshop_server.py`.

    Each instance owns a `client` token; the server keys a `SimBrowser` on it and
    every browser shares the one loaded catalogue. Two instances never collide
    even on the same goal index, which matters because the frozen evaluation
    phase runs many of these at once.
    """

    def __init__(self, base_url: str, max_steps: int = 15, timeout: float = 120.0):
        self.base_url = str(base_url).rstrip("/")
        self.max_steps = int(max_steps)
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self.timeout = float(timeout)
        self.client = uuid.uuid4().hex[:12]
        self._steps = 0
        self._open = False
        self.instruction = ""
        self.available: dict[str, Any] = {}

    # -- transport --
    def _call(self, route: str, payload: dict | None = None) -> dict:
        data = json.dumps(payload or {}).encode("utf-8")
        req = Request(
            self.base_url + route, data=data, headers={"Content-Type": "application/json"}
        )
        try:
            with urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except HTTPError as exc:  # the server hands back its own traceback
            body = exc.read().decode("utf-8", "replace")
            raise WebShopError(f"{route} -> HTTP {exc.code}: {body[:2000]}") from exc
        except URLError as exc:
            raise WebShopError(f"{route} -> {exc}") from exc

    def health(self) -> dict:
        return self._call("/health")

    def reset(self, task: TaskSpec) -> str:
        r = self._call("/reset", {"session": int(task.goal_index), "client": self.client})
        self._steps = 0
        self._open = True
        self.instruction = r.get("instruction", "")
        self.available = r.get("available_actions", {})
        return r["observation"]

    def step(self, action: str) -> dict[str, Any]:
        if not self._open:
            raise RuntimeError("reset() must be called before step()")
        action = str(action).strip()
        # `think[...]` never reaches the store. Handling it here rather than
        # server-side keeps the environment a faithful WebShop -- the step budget
        # is what the agent spends, and a thought costs one step just like an
        # action, which is what stops the agent from thinking the clock away.
        if action.lower().startswith("think["):
            self._steps += 1
            return {
                "observation": "OK.",
                "reward": 0.0,
                "success": False,
                "done": self._steps >= self.max_steps,
                "truncated": self._steps >= self.max_steps,
                "invalid": False,
                "thought_only": True,
            }
        r = self._call("/step", {"client": self.client, "action": action})
        self._steps += 1
        self.available = r.get("available_actions", {})
        reward = float(r.get("reward") or 0.0)
        native_done = bool(r.get("done"))
        truncated = self._steps >= self.max_steps and not native_done
        return {
            "observation": r["observation"],
            "reward": reward,
            # Strict WebShop success: a purchase that satisfies everything.
            # Partial credit lives in `reward` and is reported separately.
            "success": native_done and reward >= 1.0,
            "done": native_done or truncated,
            "truncated": truncated,
            "invalid": bool(r.get("invalid")),
            "thought_only": False,
        }

    def render_actions(self) -> str:
        """The clickable list, as one compact line appended to each observation.

        WebShop's `text` observation mode flattens the page to ` [SEP] `-joined
        strings, which erases the distinction between a product title and a
        button. Without this line an agent cannot tell that `blue-dinosaur` is
        selectable while the words next to it are not, and option selection --
        which the reward function grades directly -- becomes guesswork. This is
        the one place the adapter adds to what WebShop emits, and it is added
        identically for every arm.
        """
        clickables = [c for c in self.available.get("clickables", []) if c != "search"]
        parts = []
        if self.available.get("has_search_bar"):
            parts.append("search[<keywords>]")
        parts.extend(f"click[{c}]" for c in clickables)
        return "Available actions: " + (", ".join(parts) if parts else "(none)")

    def close(self) -> None:
        if self._open:
            try:
                self._call("/close", {"client": self.client})
            except WebShopError:
                pass  # a dead server must not mask the episode's own outcome
            finally:
                self._open = False

    def __enter__(self) -> "WebShopEnvironment":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


# =================================================================== agent
_ACTION_RE = re.compile(r"Action:\s*(.+?)(?:\n|$)", re.IGNORECASE | re.DOTALL)
_THOUGHT_RE = re.compile(r"Thought:\s*(.+?)(?:\n\s*Action:|$)", re.IGNORECASE | re.DOTALL)
_VERB_RE = re.compile(r"^(search|click|think)\s*\[(.*)\]\s*$", re.IGNORECASE | re.DOTALL)
_INSTRUCTION_RE = re.compile(r"Instruction:\s*(.+)", re.IGNORECASE)


def extract_instruction(observation: str) -> str:
    """The customer instruction -- the retrieval key for every memory system."""
    text = observation.replace(" [SEP] ", "\n")
    m = _INSTRUCTION_RE.search(text)
    if m:
        return m.group(1).strip().split("\n")[0].strip()
    return text.strip().split("\n")[0].strip()


def normalize_action(action: str) -> str:
    """Canonicalise a parsed action, or return "" if it is not a WebShop action.

    Kept separate from `parse_response` so a format slip (no verb at all) is
    distinguishable from a well-formed action the store happens to reject.
    """
    m = _VERB_RE.match(str(action or "").strip())
    if not m:
        return ""
    verb, arg = m.group(1).lower(), m.group(2).strip()
    if not arg:
        return ""
    return f"{verb}[{arg}]"


def parse_response(text: str) -> tuple[str, str]:
    """Return (thought, action); action is "" when the format was not followed.

    The FIRST `Action:` wins. When a model emits several it has hallucinated its
    own continuation past observations that never happened, so a later one would
    skip the episode ahead. STOP_SEQUENCES makes this rare; this covers the rest.
    """
    text = str(text or "")
    action_m = _ACTION_RE.search(text)
    raw = action_m.group(1).strip() if action_m else ""
    thought_m = _THOUGHT_RE.search(text)
    thought = thought_m.group(1).strip() if thought_m else ""
    action = normalize_action(raw)
    if not action:
        # Some responses are a bare action with no ReAct scaffolding at all.
        for line in text.strip().split("\n"):
            action = normalize_action(line)
            if action:
                break
    return thought, action


def load_demonstrations(path: str | Path | None = None) -> list[dict[str, str]]:
    p = Path(path) if path else DEFAULT_DEMOS
    entries = json.loads(p.read_text(encoding="utf-8"))
    messages: list[dict[str, str]] = []
    for entry in entries:
        messages.extend({"role": m["role"], "content": m["content"]} for m in entry["example"])
    return messages


#: Stop before the model can invent the store's reply. Hallucinated observations
#: are not merely wasted tokens: the trajectory is what the memory writers read,
#: and an entry distilled from an imagined page would be grounded in nothing
#: while still passing an evidence check against the model's own text.
STOP_SEQUENCES = ["\nObservation:", "\nObservation :", "\nAvailable actions:"]


@dataclass
class AgentConfig:
    #: WebShop's conventional horizon. An episode is search -> browse -> select
    #: options -> buy, so 15 is generous for a direct solution and tight enough
    #: that wandering costs the purchase.
    max_steps: int = 15
    #: Hard cap on context growth, as in the ALFWorld adapter: SAGE lost a run to
    #: max_tokens=2048 with no history management -- prompts reached 129k tokens
    #: and 63.6% of calls died with HTTP 400, silently, because the agent
    #: swallowed the error and kept going.
    max_tokens: int = 256
    temperature: float = 0.7  # Qwen3.5 non-thinking recommendation
    context_limit_tokens: int = 28000
    min_history_steps: int = 4
    #: Truncate a single page before it enters the conversation. A results page
    #: with long titles can run past 1,500 tokens, and 15 of those overflow the
    #: window on their own.
    max_observation_chars: int = 3000


class WebShopAgent:
    """ReAct agent over `WebShopEnvironment`, with one memory block injected."""

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
    def _opening_user_turn(self, observation: str, actions: str, memory_block: str) -> str:
        parts = []
        if memory_block.strip():
            parts.append(f"{MEMORY_HEADER}\n{memory_block.strip()}")
        parts.append(f"Now, it's your turn to solve a new task.\n{observation.strip()}\n{actions}")
        return "\n\n".join(parts)

    def _observation_turn(self, observation: str, actions: str) -> str:
        obs = observation.strip()
        if len(obs) > self.config.max_observation_chars:
            obs = obs[: self.config.max_observation_chars] + " ...<truncated>"
        return f"Observation: {obs}\n{actions}"

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
        env: WebShopEnvironment,
        task: TaskSpec,
        memory_block: str = "",
        rollout_id: str = "r0",
        temperature: float | None = None,
    ) -> Rollout:
        initial_observation = observation = env.reset(task)
        messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(self.demos)
        messages.append({
            "role": "user",
            "content": self._opening_user_turn(observation, env.render_actions(), memory_block),
        })

        steps: list[Step] = []
        success, reward, error = False, 0.0, None
        parse_failures = invalid_actions = thoughts = 0
        prompt_tokens = completion_tokens = 0
        purchased = False

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
                # A no-op that costs one step, so a format slip degrades the
                # episode instead of aborting it. There is no WebShop equivalent
                # of ALFWorld's "look", so re-reading the page is the closest
                # harmless action.
                action = "think[I could not produce a valid action; re-read the page.]"

            try:
                result = env.step(action)
            except WebShopError as exc:
                error = f"env_error: {exc}"
                break

            if result["invalid"]:
                invalid_actions += 1
            if result["thought_only"]:
                thoughts += 1
            observation = result["observation"]
            if result["invalid"]:
                # The store returns the unchanged page for an action it could not
                # perform, which reads to the agent exactly like an action that
                # legitimately changed nothing. Say so.
                observation = "Invalid action. The page did not change.\n" + observation
            steps.append(Step(action=action, observation=observation, thought=thought))

            messages.append({"role": "assistant", "content": text})
            messages.append({
                "role": "user",
                "content": self._observation_turn(observation, env.render_actions()),
            })

            if result["done"]:
                reward = result["reward"]
                success = result["success"]
                purchased = not result["truncated"]
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
                "task_family": task.category,
                "goal_index": task.goal_index,
                "initial_observation": initial_observation,
                "n_steps": len(steps),
                "parse_failures": parse_failures,
                "invalid_actions": invalid_actions,
                "thought_steps": thoughts,
                # An episode that never buys scores 0 for a different reason than
                # one that buys the wrong thing, and only one of those is a
                # memory problem. Keeping them apart is what makes the score /
                # success-rate split readable.
                "purchased": purchased,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "injected_memory_tokens": count_tokens(memory_block),
            },
        )


# ================================================================ episodes
def run_task(
    agent: WebShopAgent,
    base_url: str,
    task: TaskSpec,
    memory_block: str = "",
    n_rollouts: int = 1,
    temperature: float | None = None,
    timeout: float = 120.0,
) -> Episode:
    """Run `n_rollouts` independent attempts at one task and package them.

    n_rollouts > 1 is what lets `Episode.outcome()` return MIXED, the only case
    in which the Reflection writer's `from_contrast` mode fires.
    """
    rollouts: list[Rollout] = []
    with WebShopEnvironment(base_url, max_steps=agent.config.max_steps, timeout=timeout) as env:
        for i in range(n_rollouts):
            rollouts.append(
                agent.run(
                    env, task, memory_block=memory_block,
                    rollout_id=f"{task.task_id}#{i}", temperature=temperature,
                )
            )
    # Prefer the manifest instruction so the key written to memory is byte-
    # identical to the key retrieval used; fall back to parsing the first page.
    instruction = task.instruction or extract_instruction(rollouts[0].meta["initial_observation"])
    return Episode(
        task_id=task.task_id,
        instruction=instruction,
        rollouts=rollouts,
        scope=task.scope(),
        meta={"split": task.split, "goal_index": task.goal_index},
    )


def default_base_url() -> str:
    return os.environ.get("WEBSHOP_SERVER_URL", "http://localhost:7000")

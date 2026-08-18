"""Mind2Web adapter: offline action prediction that produces `Episode` objects.

Same contract as the ALFWorld, WebShop, AppWorld and SpreadsheetBench adapters --
TaskSpec/manifest in, `Episode` out -- so the memory systems are unchanged. Four
things differ, and each one changes how the results must be read.

**There is no environment.** Mind2Web is an *offline* benchmark: every step ships
a frozen DOM snapshot, the ground-truth operation, and a candidate element pool.
The agent predicts the next action; nothing is executed and the page never
changes in response. Previous actions in the prompt are the *annotator's*, not
the agent's -- teacher forcing -- so an early mistake cannot cascade, and a step
is scored independently of every other step. That is the benchmark's design
(`src/action_prediction/metric.py:evaluate_dataset_llm` upstream), and it makes
these numbers not comparable in kind to ALFWorld's or WebShop's, where an agent
lives with its own errors.

**One memsys "task" is one Mind2Web *step*, not one annotation.** This is the
one deliberate departure and it needs justifying, because it is the difference
between a usable experiment and a degenerate one. Whole-task success ("every
step correct") runs at 0-2% for GPT-3.5/GPT-4 in the paper, so at annotation
granularity essentially every episode would be `all_failure`: `raw` and `skill`
would write nothing at all (both need a successful rollout), and the study would
reduce to the failure-only regime that RESULTS_ALFWORLD.md §11 already measured
as harmful. At step granularity the success signal is the benchmark's own
headline metric, Step SR, which lands in the 20-40% band -- the same range
WebShop's episodes occupy. Manifests are still *drawn* at annotation level and
expanded in order, so a "50-task" run means 50 annotations' worth of steps and
the memory a step sees includes what earlier steps of its own task wrote.

**The element decision is a tournament, not a single choice.** Upstream shows the
model 5 candidates at a time out of the top-k ranked pool, re-queues each round's
winner, and repeats until one survives. Ported verbatim, including the parts that
look like bugs: a round whose answer is "A. None of the above" contributes no
survivor, `final_prediction` is overwritten by every later round rather than
being decided at the end, and a step whose ground-truth element is not in the
top-k is scored 0 without the model ever being called. Changing any of these
would silently redefine the metric.

**The scaffold is upstream's, byte-for-byte where it is published.** The system
prompt and the 3-shot demonstrations come from
`src/action_prediction/llm_prompt.json`; the DOM pruning and serialisation are a
port of `src/data_utils/dom_utils.py`; scoring is a port of `postprocess_action_llm`
and `calculate_f1`. RUN_ALFWORLD.md §5 records a 40-point swing on ALFWorld from
scaffold wording alone, so the memory block is the *only* thing this file adds to
the prompt, and it is appended ahead of the question in the final user turn.
"""

from __future__ import annotations

import copy
import json
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from lxml import etree

from ..episode import Episode, Rollout, Step
from ..tokens import count_tokens

SCOPE_ENV = "mind2web"

MEMORY_HEADER = (
    "**Archived memory (distilled from your own past episodes on other websites):**"
)


# ==================================================== port of dom_utils.py
# OSU-NLP-Group/Mind2Web, `src/data_utils/dom_utils.py`. Behaviour-identical;
# the only changes are explicit (never shared-mutable) id_mapping arguments.

SALIENT_ATTRIBUTES = {
    "alt", "aria_description", "aria_label", "aria_role", "input_checked",
    "input_value", "label", "name", "option_selected", "placeholder", "role",
    "text_value", "title", "type", "value",
}


def clean_text(text: str | None) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def get_descendants(node, max_depth: int, current_depth: int = 0) -> list:
    if current_depth > max_depth:
        return []
    out = []
    for child in node:
        out.append(child)
        out.extend(get_descendants(child, max_depth, current_depth + 1))
    return out


def prune_tree(dom_tree, candidate_set, max_depth=5, max_children=50, max_sibling=3):
    """Keep only the candidates, their ancestors, shallow descendants and near siblings."""
    nodes_to_keep: set[str] = set()
    for candidate_id in candidate_set:
        found = dom_tree.xpath(f'//*[@backend_node_id="{candidate_id}"]')
        if not found:
            # Upstream indexes [0] unguarded. A missing id means the candidate
            # pool and the snapshot disagree, which does happen in the released
            # data; skipping keeps the step scoreable instead of killing the run.
            continue
        candidate_node = found[0]
        nodes_to_keep.add(candidate_node.attrib["backend_node_id"])
        nodes_to_keep.update(
            x.attrib.get("backend_node_id", "") for x in candidate_node.xpath("ancestor::*")
        )
        nodes_to_keep.update(
            [x.attrib.get("backend_node_id", "")
             for x in get_descendants(candidate_node, max_depth)][:max_children]
        )
        parent = candidate_node.getparent()
        if parent is not None:
            siblings = [x for x in parent.getchildren() if x.tag != "text"]
            idx = siblings.index(candidate_node)
            nodes_to_keep.update(
                x.attrib.get("backend_node_id", "")
                for x in siblings[max(0, idx - max_sibling): idx + max_sibling + 1]
            )
    new_tree = copy.deepcopy(dom_tree)
    for node in new_tree.xpath("//*")[::-1]:
        if node.tag != "text":
            is_keep = node.attrib.get("backend_node_id", "") in nodes_to_keep
            is_candidate = node.attrib.get("backend_node_id", "") in candidate_set
        else:
            parent = node.getparent()
            is_keep = parent is not None and parent.attrib.get("backend_node_id", "") in nodes_to_keep
            is_candidate = parent is not None and parent.attrib.get("backend_node_id", "") in candidate_set
        if not is_keep and node.getparent() is not None:
            node.getparent().remove(node)
        else:
            if not is_candidate or node.tag == "text":
                node.attrib.pop("backend_node_id", None)
            if (
                len(node.attrib) == 0
                and not any(x.tag == "text" for x in node.getchildren())
                and node.getparent() is not None
                and node.tag != "text"
                and len(node.getchildren()) <= 1
            ):
                for child in node.getchildren():
                    node.addprevious(child)
                node.getparent().remove(node)
    return new_tree


def get_attribute_repr(node, max_value_length: int = 5, max_length: int = 20) -> None:
    attr_values_set: set[str] = set()
    attr_values = ""
    for attr in ("role", "aria_role", "type", "alt", "aria_description", "aria_label",
                 "label", "title", "name", "text_value", "value", "placeholder",
                 "input_checked", "input_value", "option_selected", "class"):
        if attr in node.attrib and node.attrib[attr] is not None:
            value = node.attrib[attr].lower()
            if value in ("hidden", "none", "presentation", "null", "undefined") or value.startswith("http"):
                continue
            value = " ".join([v for v in value.split() if len(v) < 15][:max_value_length])
            if value and value not in attr_values_set:
                attr_values_set.add(value)
                attr_values += value + " "
    uid = node.attrib.get("backend_node_id", "")
    node.attrib.clear()
    if uid:
        node.attrib["id"] = uid
    if attr_values:
        node.attrib["meta"] = " ".join(attr_values.split()[:max_length])


_HTML_ESCAPES = (
    ("&quot;", '"'), ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"), ("&nbsp;", " "),
    ("&ndash;", "-"), ("&rsquo;", "'"), ("&lsquo;", "'"), ("&ldquo;", '"'),
    ("&rdquo;", '"'), ("&#39;", "'"), ("&#40;", "("), ("&#41;", ")"),
)


def get_tree_repr(tree, max_value_length=5, max_length=20, id_mapping=None,
                  keep_html_brackets=False) -> tuple[str, dict]:
    id_mapping = {} if id_mapping is None else id_mapping
    tree = etree.fromstring(tree) if isinstance(tree, str) else copy.deepcopy(tree)
    for node in tree.xpath("//*"):
        if node.tag != "text":
            if "backend_node_id" in node.attrib:
                if node.attrib["backend_node_id"] not in id_mapping:
                    id_mapping[node.attrib["backend_node_id"]] = len(id_mapping)
                node.attrib["backend_node_id"] = str(id_mapping[node.attrib["backend_node_id"]])
            get_attribute_repr(node, max_value_length, max_length)
        else:
            node.text = " ".join((node.text or "").split()[:max_length])
    tree_repr = etree.tostring(tree, encoding="unicode")
    tree_repr = tree_repr.replace('"', " ")
    tree_repr = tree_repr.replace("meta= ", "").replace("id= ", "id=").replace(" >", ">")
    tree_repr = re.sub(r"<text>(.*?)</text>", r"\1", tree_repr)
    if not keep_html_brackets:
        tree_repr = tree_repr.replace("/>", "$/$>")
        tree_repr = re.sub(r"</(.+?)>", r")", tree_repr)
        tree_repr = re.sub(r"<(.+?)>", r"(\1", tree_repr)
        tree_repr = tree_repr.replace("$/$", ")")
    for k, v in _HTML_ESCAPES:
        tree_repr = tree_repr.replace(k, v)
    return re.sub(r"\s+", " ", tree_repr).strip(), id_mapping


def format_input_multichoice(
    sample: dict, candidate_ids: Sequence[str], gt: str | int = -1,
    previous_k: int = 5, keep_html_brackets: bool = True,
) -> tuple[str, str, str, list[list[str]]]:
    """Port of `dataloader.format_input_multichoice`.

    Returns (page context, question, target answer, choices), where `choices[i]`
    is `[backend_node_id, short element rendering]` for option chr(66+i).
    """
    dom_tree = etree.fromstring(sample["cleaned_html"])
    dom_tree = prune_tree(dom_tree, list(candidate_ids))
    tree_repr, id_mapping = get_tree_repr(
        dom_tree, id_mapping={}, keep_html_brackets=keep_html_brackets
    )
    candidate_nodes = dom_tree.xpath("//*[@backend_node_id]")
    choices = []
    for node in candidate_nodes:
        choices.append([
            node.attrib["backend_node_id"],
            " ".join(get_tree_repr(
                node, id_mapping=id_mapping, keep_html_brackets=keep_html_brackets
            )[0].split()[:10]),
        ])
    gt_idx = id_mapping.get(gt, -1)
    seq_input = (
        "Based on the HTML webpage above, try to complete the following task:\n"
        f"Task: {sample['confirmed_task']}\n"
        "Previous actions:\n"
    )
    prev = sample.get("previous_actions") or []
    seq_input += ("".join(f"{a}\n" for a in prev[-previous_k:])) if prev else "None\n"
    seq_input += (
        "What should be the next action? Please select from the following choices "
        "(If the correct action is not in the page above, please select A. 'None of the above'):\n\n"
        "A. None of the above\n"
    )
    for idx, choice in enumerate(choices):
        seq_input += f"{chr(66 + idx)}. {choice[1]}\n"
    if gt_idx == -1:
        seq_target = "A."
    else:
        op = sample["operation"]["op"]
        seq_target = f"{chr(65 + gt_idx + 1)}.\nAction: {op}"
        if op != "CLICK":
            seq_target += f"\nValue: {sample['operation']['value']}"
    return tree_repr, seq_input, seq_target, choices


# ================================================== port of metric.py scoring
def postprocess_action_llm(text: str) -> tuple[str, str]:
    """Port of `ActionEvaluatorMultiChoice.postprocess_action_llm`.

    An unparseable answer degrades to "A" (none of the above) rather than
    raising -- upstream's behaviour, and it keeps a format slip costing one step
    instead of the run.
    """
    text = text.strip()
    m = re.search(r"Answer: (A|B|C|D|E|F)", text)
    selected_option = m.group(1) if m is not None else "A"
    m = re.search(r"Action: (CLICK|SELECT|TYPE)", text)
    action = m.group(1) if m is not None else ""
    m = re.search(r"Value: (.*)$", text, re.MULTILINE)
    value = m.group(1) if m is not None else ""
    return selected_option, (action.strip() + " " + value.strip())


def calculate_f1(pred: str, label: str) -> float:
    """Token-set F1 between two action strings ("CLICK", "TYPE new york", ...)."""
    pred_set, label_set = set(pred.strip().split()), set(label.strip().split())
    if not pred_set and not label_set:
        return 1.0
    if not pred_set or not label_set:
        return 0.0
    tp = len(pred_set & label_set)
    if tp == 0:
        return 0.0
    precision, recall = tp / len(pred_set), tp / len(label_set)
    return 2 * precision * recall / (precision + recall)


def target_action_string(sample: dict) -> str:
    """The label side of `calculate_f1`, built exactly as upstream does."""
    op = sample["operation"]["op"]
    value = sample["operation"]["value"]
    return (op + " " + value).strip() if op != "CLICK" else op


# ================================================================ task specs
@dataclass(frozen=True)
class TaskSpec:
    """One Mind2Web *step*: the memsys episode unit (see the module docstring)."""

    task_id: str          # f"{split}:{annotation_id}:{step_index}"
    split: str
    annotation_id: str
    action_uid: str
    step_index: int
    n_steps: int          # steps in the parent annotation
    website: str = ""
    domain: str = ""
    subdomain: str = ""
    instruction: str = ""  # confirmed_task, identical across a task's steps

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id, "split": self.split,
            "annotation_id": self.annotation_id, "action_uid": self.action_uid,
            "step_index": self.step_index, "n_steps": self.n_steps,
            "website": self.website, "domain": self.domain,
            "subdomain": self.subdomain, "instruction": self.instruction,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any], default_split: str = "train") -> "TaskSpec":
        split = str(d.get("split") or default_split)
        annotation_id = str(d["annotation_id"])
        step_index = int(d.get("step_index", 0))
        return cls(
            task_id=str(d.get("task_id") or f"{split}:{annotation_id}:{step_index}"),
            split=split, annotation_id=annotation_id,
            action_uid=str(d.get("action_uid") or ""), step_index=step_index,
            n_steps=int(d.get("n_steps", 0)), website=str(d.get("website") or ""),
            domain=str(d.get("domain") or ""), subdomain=str(d.get("subdomain") or ""),
            instruction=str(d.get("instruction") or ""),
        )

    def scope(self) -> dict[str, str]:
        # `domain` is the family axis, as `task_type` is on ALFWorld: it is the
        # coarsest label that a transferable web procedure could plausibly key on.
        return {"env": SCOPE_ENV, "task_type": self.domain}


def load_manifest(path: str | Path) -> list[TaskSpec]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(value, list):
        return [TaskSpec.from_dict(item) for item in value]
    if not isinstance(value, dict) or not isinstance(value.get("tasks"), list):
        raise ValueError(f"unsupported manifest format: {path}")
    split = str(value.get("split") or "train")
    return [TaskSpec.from_dict(item, default_split=split) for item in value["tasks"]]


class StepStore:
    """Reads the per-annotation step payloads written by `build_mind2web_manifests.py`.

    The released splits are 0.6 GB JSON shards holding full page snapshots, which
    neither an evaluation worker nor a 50-episode evolving loop can re-read. The
    builder extracts exactly the fields the prompt needs (cleaned HTML, candidate
    pools already merged with the released ranker's ranks, the operation, and the
    annotator's action strings) into one small file per annotation; this reads
    them back, with a tiny cache because a task's steps arrive consecutively.
    """

    def __init__(self, root: str | Path, cache_size: int = 4):
        self.root = Path(root)
        self.cache_size = cache_size
        self._cache: dict[str, list[dict]] = {}
        self._order: list[str] = []

    def steps(self, split: str, annotation_id: str) -> list[dict]:
        key = f"{split}/{annotation_id}"
        if key not in self._cache:
            path = self.root / split / f"{annotation_id}.json"
            if not path.is_file():
                raise FileNotFoundError(
                    f"no step cache at {path}; run scripts/build_mind2web_manifests.py"
                )
            self._cache[key] = json.loads(path.read_text(encoding="utf-8"))["steps"]
            self._order.append(key)
            while len(self._order) > self.cache_size:
                self._cache.pop(self._order.pop(0), None)
        return self._cache[key]

    def sample(self, task: TaskSpec) -> dict:
        return self.steps(task.split, task.annotation_id)[task.step_index]


# ==================================================================== agent
DEFAULT_PROMPT_PATH = Path(__file__).with_name("mind2web_prompt.json")


def load_prompt_template(path: str | Path | None = None) -> list[dict[str, str]]:
    """Upstream's `llm_prompt.json`: system turn + 3 demonstrations + an empty user turn."""
    p = Path(path) if path else DEFAULT_PROMPT_PATH
    messages = json.loads(p.read_text(encoding="utf-8"))
    if not messages or messages[-1]["role"] != "user":
        raise ValueError(f"{p}: expected the template to end with an empty user turn")
    return messages


@dataclass
class AgentConfig:
    #: Candidate pool depth, as ranked by the released DeBERTa candidate
    #: generator. The paper uses 50 for most models and 10 for GPT-4; 10 costs
    #: ~3 LLM calls per step against ~13, which is the whole reason a sweep of
    #: this benchmark is affordable here.
    top_k: int = 50
    #: Upstream generates 50 new tokens: the answer is a letter plus at most two
    #: short lines. Anything larger only buys runaway rationales.
    max_tokens: int = 64
    #: Greedy. The task is multiple choice with a single scored answer, and with
    #: temperature 0 the *only* thing that differs between arms is the memory
    #: block -- ALFWorld's sampling noise floor does not exist here.
    temperature: float = 0.0
    previous_k: int = 5
    #: Truncate the serialised page before it enters the prompt. Upstream feeds
    #: its own truncation through a tokenizer during fine-tuning; for the LLM
    #: path it does not truncate at all, and a handful of pages serialise past
    #: 30k tokens, which would fail the 32k-context server outright.
    max_context_tokens: int = 12000
    seed: int = 42


class Mind2WebAgent:
    """Multi-choice action predictor over cached Mind2Web steps."""

    def __init__(self, client: Any, model: str,
                 prompt_template: Sequence[dict[str, str]] | None = None,
                 config: AgentConfig | None = None):
        self.client = client
        self.model = model
        self.template = [dict(m) for m in (prompt_template or load_prompt_template())]
        self.config = config or AgentConfig()

    # ------------------------------------------------------------ prompting
    def _messages(self, context: str, question: str, memory_block: str) -> list[dict[str, str]]:
        if count_tokens(context) > self.config.max_context_tokens:
            from ..tokens import truncate_to_tokens

            context = truncate_to_tokens(context, self.config.max_context_tokens) + " ...(truncated)"
        body = f"'''\n{context}\n'''\n\n{question}"
        if memory_block.strip():
            body = f"{MEMORY_HEADER}\n{memory_block.strip()}\n\n{body}"
        messages = [dict(m) for m in self.template]
        messages[-1]["content"] = body
        return messages

    # ----------------------------------------------------------- one sample
    def predict_step(self, sample: dict, memory_block: str = "", rng: random.Random | None = None) -> dict:
        """Run upstream's tournament for one step and score it.

        Returns a dict with the prediction, the three per-step metrics, and the
        bookkeeping the Episode carries (token usage, number of LLM calls).
        """
        cfg = self.config
        rng = rng or random.Random(cfg.seed)
        out = {
            "element_acc": 0, "action_f1": 0.0, "step_acc": 0, "n_calls": 0,
            "prompt_tokens": 0, "completion_tokens": 0, "predicted": None,
            "skipped": False, "error": None,
            "target": target_action_string(sample),
            "target_repr": sample.get("action_repr", ""),
        }
        pos = [c for c in sample["pos_candidates"] if c.get("rank", 10**9) < cfg.top_k]
        pos_ids = [c["backend_node_id"] for c in pos]
        if not pos_ids:
            # Upstream scores these 0 without calling the model: the ranker never
            # surfaced the right element, so the reader cannot separate "the LLM
            # chose wrong" from "the LLM was never shown it". Recorded, not hidden.
            out["skipped"] = True
            return out
        neg_ids = [c["backend_node_id"] for c in sample["neg_candidates"]
                   if c.get("rank", 10**9) < cfg.top_k]
        all_candidates = pos_ids + neg_ids
        rng.shuffle(all_candidates)

        final_prediction = None
        while len(all_candidates) > 1:
            candidate_ids, all_candidates = all_candidates[:5], all_candidates[5:]
            try:
                context, question, _, choices = format_input_multichoice(
                    sample, candidate_ids, -1, previous_k=cfg.previous_k,
                    keep_html_brackets=True,
                )
            except Exception as exc:  # noqa: BLE001 - malformed snapshot, not our bug
                out["error"] = f"format_error: {type(exc).__name__}: {exc}"
                break
            try:
                resp = self.client.chat.completions.create(
                    model=self.model, messages=self._messages(context, question, memory_block),
                    temperature=cfg.temperature, max_tokens=cfg.max_tokens,
                )
            except Exception as exc:  # noqa: BLE001
                # Record rather than die, as in the other adapters: a truncated
                # episode is data, a crashed sweep is not.
                out["error"] = f"llm_error: {type(exc).__name__}: {exc}"
                break
            out["n_calls"] += 1
            usage = getattr(resp, "usage", None)
            out["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
            out["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0
            text = resp.choices[0].message.content or ""
            pred_letter, pred_action = postprocess_action_llm(text)
            if pred_letter != "A":
                idx = ord(pred_letter) - ord("B")
                if 0 <= idx < len(choices):
                    all_candidates.append(choices[idx][0])
                    final_prediction = (choices[idx][0], pred_action, choices[idx][1])
                # An out-of-range letter is upstream's IndexError branch: the
                # round contributes no survivor and the previous prediction stands.

        if final_prediction is not None:
            out["predicted"] = {
                "backend_node_id": final_prediction[0], "action": final_prediction[1],
                "element": final_prediction[2],
            }
            out["element_acc"] = 1 if final_prediction[0] in pos_ids else 0
            out["action_f1"] = calculate_f1(final_prediction[1], out["target"])
            out["step_acc"] = 1 if (out["element_acc"] == 1 and out["action_f1"] == 1) else 0
        return out

    # ---------------------------------------------------------------- rollout
    def run(self, task: TaskSpec, sample: dict, memory_block: str = "",
            rollout_id: str = "r0") -> Rollout:
        # Seed the candidate shuffle from the task id as a *string*: `random.Random`
        # hashes str seeds with sha512, so this is stable across processes, while
        # `hash()` is randomised per interpreter by PYTHONHASHSEED. That matters
        # more here than it looks -- the shuffle decides which five candidates
        # meet in each tournament round, so an unstable seed would make two arms
        # differ by shuffle noise on top of the memory block, and would stop the
        # `none` baseline from being reusable at all.
        result = self.predict_step(
            sample, memory_block=memory_block,
            rng=random.Random(f"{self.config.seed}/{task.task_id}/{rollout_id}"),
        )
        pred = result["predicted"]
        if result["skipped"]:
            action = "(no prediction: ground-truth element outside the candidate pool)"
        elif pred is None:
            action = "Answer: A. (none of the above)"
        else:
            action = f"Answer: {pred['element']} -> {pred['action']}"
        verdict = "CORRECT" if result["step_acc"] else "INCORRECT"
        observation = (
            f"{verdict}. The annotated action was: {result['target_repr'] or result['target']}"
            f" (element_acc={result['element_acc']}, action_f1={result['action_f1']:.2f})"
        )
        step = Step(action=action, observation=observation,
                    thought=f"Step {task.step_index + 1} of {task.n_steps} on {task.website}")
        return Rollout(
            rollout_id=rollout_id, steps=[step],
            reward=float(result["step_acc"]), success=bool(result["step_acc"]),
            error=result["error"],
            meta={
                "element_acc": result["element_acc"], "action_f1": result["action_f1"],
                "step_acc": result["step_acc"], "skipped_no_pos_candidate": result["skipped"],
                "n_llm_calls": result["n_calls"], "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "injected_memory_tokens": count_tokens(memory_block),
                "parse_failures": 0, "nothing_happens": 0,
            },
        )


def run_task(agent: Mind2WebAgent, store: StepStore, task: TaskSpec,
             memory_block: str = "", n_rollouts: int = 1) -> Episode:
    """One step = one Episode. `n_rollouts > 1` only matters at temperature > 0."""
    sample = dict(store.sample(task))
    sample["confirmed_task"] = task.instruction or sample.get("confirmed_task", "")
    rollouts = [
        agent.run(task, sample, memory_block=memory_block, rollout_id=f"{task.task_id}#{i}")
        for i in range(n_rollouts)
    ]
    # The retrieval key is the annotation's task description plus how far in we
    # are: every step of one annotation shares the task text, and without the
    # step marker the memory system cannot tell "start the search" from "confirm
    # the booking". The written key is the same string, so retrieval and storage
    # agree by construction.
    instruction = task.instruction
    if task.n_steps > 1:
        instruction = f"{instruction} (step {task.step_index + 1} of {task.n_steps})"
    return Episode(
        task_id=task.task_id, instruction=instruction, rollouts=rollouts,
        scope=task.scope(),
        meta={"split": task.split, "annotation_id": task.annotation_id,
              "website": task.website, "domain": task.domain,
              "step_index": task.step_index, "n_steps": task.n_steps},
    )


def aggregate_metrics(rows: Iterable[dict]) -> dict[str, float]:
    """Benchmark-level metrics from per-step rows (as written to eval.jsonl).

    Micro averages over steps, plus Mind2Web's task success rate: an annotation
    counts only if *every* one of its steps was predicted exactly right.
    """
    rows = list(rows)
    if not rows:
        return {}
    by_task: dict[str, list[int]] = {}
    for r in rows:
        by_task.setdefault(r.get("annotation_id", r["task_id"]), []).append(int(r["step_acc"]))
    n = len(rows)
    return {
        "n_steps": n,
        "n_tasks": len(by_task),
        "element_acc": sum(r["element_acc"] for r in rows) / n,
        "action_f1": sum(r["action_f1"] for r in rows) / n,
        "step_success_rate": sum(r["step_acc"] for r in rows) / n,
        "task_success_rate": sum(1 for v in by_task.values() if all(v)) / len(by_task),
        "skipped_no_pos_candidate": sum(1 for r in rows if r.get("skipped")) / n,
    }

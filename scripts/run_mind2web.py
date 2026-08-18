#!/usr/bin/env python
"""Run one memsys arm end to end on Mind2Web: evolve, then frozen evaluation.

    python scripts/run_mind2web.py --arm rule --policy minimal \
        --evolve-manifest manifests/mind2web_evolve_train_50_seed42.json \
        --eval-manifest   manifests/mind2web_eval_test_task_100_seed42.json \
        --cache-root /gpfs/radev/scratch/cohan/jw3278/mind2web_cache

Same shape as `run_alfworld.py` / `run_webshop.py` -- one process = one arm =
one (memory type, write policy) cell -- with three benchmark-specific
differences, all of which follow from Mind2Web being an *offline* benchmark:

*The episode unit is a step.* Manifest entries are Mind2Web steps, drawn at
annotation level and expanded in order (`memsys/adapters/mind2web.py` explains
why annotation-level success is unusable here). A "50-task" evolving run is
therefore ~380 episodes, and `--evolve-limit` counts steps, not annotations.

*Evaluation fans out over threads, not processes.* There is no environment and
no PDDL backend -- a worker only reads a cached JSON page and calls the server --
so the ALFWorld process-pool workaround buys nothing here.

*Two success rates are reported.* `eval_success_rate` is Mind2Web's step success
rate, the metric this study's paired tests run on; `task_success_rate` (every
step of an annotation correct) is the benchmark's strict number and is reported
alongside because it is what the Mind2Web paper's headline table shows.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memsys import (  # noqa: E402
    Episode,
    Evolver,
    MemoryConfig,
    MemoryStore,
    OpenAIChatClient,
    OpenAIResponsesClient,
    RunLogger,
    WritePolicy,
    build_system,
    frozen,
)
from memsys.adapters.mind2web import (  # noqa: E402
    AgentConfig,
    Mind2WebAgent,
    StepStore,
    TaskSpec,
    aggregate_metrics,
    load_manifest,
    run_task,
)

ARMS = ("none", "raw", "reflection", "rule", "skill", "all")


def build_config(args) -> MemoryConfig:
    policy = WritePolicy.full() if args.policy == "full" else WritePolicy.minimal()
    policy.n_max = args.n_max
    policy.batch_every = args.batch_every
    return MemoryConfig(
        policy=policy,
        injection_budget_tokens=args.injection_budget,
        max_items=args.max_items,
        seed=args.seed,
    )


def build_store(args, config: MemoryConfig) -> MemoryStore:
    if args.embedder == "hashing":
        from memsys import HashingEmbedder

        return MemoryStore(embedder=HashingEmbedder(), config=config)
    from memsys import SentenceTransformerEmbedder

    # CPU on purpose: both GPUs are held by vLLM at 0.90 utilisation, and an
    # auto-placed encoder either OOMs or steals KV cache from the server it is
    # talking to.
    return MemoryStore(
        embedder=SentenceTransformerEmbedder(args.embedding_model, device=args.embedding_device),
        config=config,
    )


def make_agent(args, base_url: str) -> Mind2WebAgent:
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key="EMPTY", timeout=args.timeout, max_retries=3)
    return Mind2WebAgent(
        client=client, model=args.model,
        config=AgentConfig(
            top_k=args.top_k, max_tokens=args.max_tokens, temperature=args.temperature,
            previous_k=args.previous_k, max_context_tokens=args.max_context_tokens,
            seed=args.seed,
        ),
    )


def _dotenv(name: str) -> str | None:
    """Read one key from the repo's .env, which is gitignored and holds the
    gateway credential. Real environment variables win over the file."""
    if os.environ.get(name):
        return os.environ[name]
    path = Path(__file__).resolve().parent.parent / ".env"
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        if k.strip() == name:
            return v.strip().strip("'\"")
    return None


def build_writer_llm(args, writer_url: str):
    """The memory writer's client, deliberately independent of the agent's:
    `--writer-model` is the independent variable of the Memory Writing Model
    study, while the actor stays on `--model` (the local Qwen) so any delta is
    attributable to what was written rather than to who acted. Same contract as
    `run_alfworld.py:build_writer_llm`."""
    model = args.writer_model or args.model
    key = "EMPTY"
    if args.writer_api_key_env:
        key = _dotenv(args.writer_api_key_env)
        if not key:
            # Failing here is the point: an unauthenticated writer 401s on every
            # call, and `parse_ops` turns that into an empty store rather than an
            # error, so the arm would finish and report a plausible number.
            raise SystemExit(
                f"--writer-api-key-env {args.writer_api_key_env!r} is set but no such key "
                f"is in the environment or .env")
    if args.writer_api == "responses":
        return OpenAIResponsesClient(
            model, base_url=writer_url, api_key=key,
            temperature=0.0, max_tokens=args.writer_max_tokens, timeout=args.timeout,
            reasoning_effort=args.writer_reasoning_effort,
        )
    return OpenAIChatClient(
        model, base_url=writer_url, api_key=key,
        temperature=0.0, max_tokens=args.writer_max_tokens, timeout=args.timeout,
    )


def episode_row(ep: Episode, seconds: float) -> dict:
    r = ep.rollouts[0]
    return {
        "task_id": ep.task_id,
        "annotation_id": ep.meta.get("annotation_id", ""),
        "instruction": ep.instruction,
        "task_family": ep.scope.get("task_type", ""),
        "website": ep.meta.get("website", ""),
        "step_index": ep.meta.get("step_index", 0),
        "n_steps": ep.meta.get("n_steps", 0),
        "outcome": ep.outcome(),
        "success_rate": ep.success_rate,
        "any_success": ep.any_success,
        "element_acc": r.meta.get("element_acc", 0),
        "action_f1": r.meta.get("action_f1", 0.0),
        "step_acc": r.meta.get("step_acc", 0),
        "skipped": bool(r.meta.get("skipped_no_pos_candidate")),
        "n_llm_calls": r.meta.get("n_llm_calls", 0),
        "error": r.error,
        "rollouts": [
            {
                "rollout_id": x.rollout_id, "success": x.success, "reward": x.reward,
                "n_steps": len(x.steps), "error": x.error,
                **{k: x.meta.get(k) for k in
                   ("prompt_tokens", "completion_tokens", "injected_memory_tokens",
                    "parse_failures", "nothing_happens")},
            }
            for x in ep.rollouts
        ],
        "seconds": round(seconds, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--policy", choices=("minimal", "full"), default="minimal")
    ap.add_argument("--evolve-manifest")
    ap.add_argument("--eval-manifest", required=True)
    ap.add_argument("--cache-root", default=os.environ.get(
        "MIND2WEB_CACHE", "/gpfs/radev/scratch/cohan/jw3278/mind2web_cache"))
    ap.add_argument("--out", required=True)

    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--agent-base-url", default="http://localhost:8000/v1")
    ap.add_argument("--writer-base-url", default=None,
                    help="defaults to --agent-base-url; point elsewhere for a remote writer")
    ap.add_argument("--writer-model", default=None,
                    help="memory-writing model; defaults to --model (the agent's). Set this "
                         "to vary the writer independently of the actor, which is the point "
                         "of the Memory Writing Model study")
    ap.add_argument("--writer-api", choices=("chat", "responses"), default="chat",
                    help="wire protocol for the writer endpoint. vLLM speaks 'chat'; the "
                         "Perplexity gateway serving openai/gpt-5.6-* speaks only 'responses'")
    ap.add_argument("--writer-api-key-env", default=None,
                    help="environment variable holding the writer API key (also read from "
                         "the repo .env). Local vLLM needs none")
    ap.add_argument("--writer-reasoning-effort", default=None,
                    help="reasoning effort for a 'responses' writer, e.g. low/medium/high")
    ap.add_argument("--timeout", type=float, default=180.0)

    ap.add_argument("--n-rollouts", type=int, default=1)
    ap.add_argument("--top-k", type=int, default=50,
                    help="candidate pool depth; the paper uses 50, and 10 for GPT-4")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--previous-k", type=int, default=5)
    ap.add_argument("--max-context-tokens", type=int, default=12000)
    ap.add_argument("--writer-max-tokens", type=int, default=1024)

    ap.add_argument("--injection-budget", type=int, default=1500)
    ap.add_argument("--max-items", type=int, default=200)
    ap.add_argument("--n-max", type=int, default=2)
    ap.add_argument("--batch-every", type=int, default=25)
    ap.add_argument("--embedder", choices=("hashing", "st"), default="st")
    ap.add_argument("--embedding-model", default="BAAI/bge-large-en-v1.5")
    ap.add_argument("--embedding-device", default="cpu")

    ap.add_argument("--eval-workers", type=int, default=8)
    ap.add_argument("--evolve-limit", type=int, default=0, help="0 = all STEPS in manifest")
    ap.add_argument("--eval-limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume-store", default=None)
    ap.add_argument("--evolve-step-offset", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    config = build_config(args)
    writer_url = args.writer_base_url or args.agent_base_url

    system = None
    if args.arm != "none":
        llm = build_writer_llm(args, writer_url)
        system = build_system(args.arm, llm=llm, config=config, store=build_store(args, config))
        if args.resume_store:
            store = getattr(system, "store", None)
            if store is None:
                raise SystemExit(f"--resume-store given but arm {args.arm!r} has no store")
            store.load(args.resume_store)
            print(f"[resume] loaded {len(store)} live entries from {args.resume_store}", flush=True)

    agent = make_agent(args, args.agent_base_url)
    steps = StepStore(args.cache_root)

    (out / "config.json").write_text(json.dumps({
        "args": vars(args),
        "policy": asdict(config.policy),
        "memory_config": {k: v for k, v in config.to_dict().items() if k != "policy_by_type"},
    }, indent=2, default=str) + "\n", encoding="utf-8")

    # ------------------------------------------------------------- evolving
    evolve_rows = []
    if system is not None and args.evolve_manifest:
        tasks = load_manifest(args.evolve_manifest)
        if args.evolve_limit:
            tasks = tasks[: args.evolve_limit]
        logger = RunLogger(str(out / "evolve_log.jsonl"))
        ev = Evolver(system, config=config, logger=logger)
        ev.step = args.evolve_step_offset
        n_ok = 0
        with (out / "evolve_episodes.jsonl").open("w", encoding="utf-8") as fh:
            for i, task in enumerate(tasks):
                t0 = time.time()
                # Retrieve first, act second, write third -- the entries handed to
                # step_once are exactly the ones the agent saw.
                ret = ev.retrieve(task.instruction, scope=task.scope())
                ep = run_task(agent, steps, task, memory_block=ret.block,
                              n_rollouts=args.n_rollouts)
                ev.step_once(ep, ret)
                row = episode_row(ep, time.time() - t0)
                row["step"] = args.evolve_step_offset + i
                evolve_rows.append(row)
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                n_ok += int(bool(ep.any_success))
                store = getattr(system, "store", None)
                print(f"[evolve {i+1}/{len(tasks)}] {row['outcome']:<12} "
                      f"ok={n_ok} ea={row['element_acc']} f1={row['action_f1']:.2f} "
                      f"store={len(store) if store is not None else '-'} "
                      f"{row['seconds']:.0f}s {task.task_id}", flush=True)
        ev.flush()
        if getattr(system, "store", None) is not None:
            system.store.save(str(out / "store.jsonl"))

    # ----------------------------------------------------------- evaluation
    eval_tasks = load_manifest(args.eval_manifest)
    if args.eval_limit:
        eval_tasks = eval_tasks[: args.eval_limit]

    # Retrieval happens here, in the parent, under frozen(): evaluation must not
    # write. Each worker then receives a finished memory block.
    ctx = frozen(system) if system is not None else _nullcontext()
    with ctx:
        payloads = []
        for task in eval_tasks:
            block = ""
            if system is not None:
                block = system.retrieve(task.instruction, scope=task.scope()).block
            payloads.append((task, block))

    # Threads, not processes: a worker parses a cached JSON page and calls the
    # server. There is no environment to corrupt and no CUDA state to inherit.
    fh = (out / "eval.jsonl").open("w", encoding="utf-8")
    eval_rows = []

    def _run_eval(payload):
        task, block = payload
        t0 = time.time()
        ep = run_task(agent, StepStore(args.cache_root), task,
                      memory_block=block, n_rollouts=1)
        return episode_row(ep, time.time() - t0)

    with ThreadPoolExecutor(max_workers=args.eval_workers) as pool:
        futures = [pool.submit(_run_eval, p) for p in payloads]
        for fut in as_completed(futures):
            row = fut.result()
            eval_rows.append(row)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            if len(eval_rows) % 25 == 0 or len(eval_rows) == len(payloads):
                done = sum(r["step_acc"] for r in eval_rows)
                print(f"[eval {len(eval_rows)}/{len(eval_tasks)}] "
                      f"step_sr={done/len(eval_rows):.3f}", flush=True)
    fh.close()

    # -------------------------------------------------------------- summary
    metrics = aggregate_metrics(eval_rows)
    evolve_metrics = aggregate_metrics(evolve_rows) if evolve_rows else {}
    summary = {
        "arm": args.arm,
        "policy": args.policy,
        "model": args.model,
        # The actor and the writer are separate variables; recording only one of
        # them makes two runs that differ in the writer indistinguishable in the
        # summary. `none`/`raw` have no writer LLM at all, hence the empty string.
        "writer_model": getattr(
            getattr(getattr(system, "writer", None), "llm", None), "model", ""),
        "top_k": args.top_k,
        # Read by scripts/summarize.py for the table header, and by nothing else.
        "benchmark": "Mind2Web",
        "eval_split": Path(args.eval_manifest).stem.replace("mind2web_eval_", ""),
        # The generic summarizer's optional partial-credit column. On Mind2Web
        # that is element accuracy: step success additionally requires the
        # operation *and* its value to match exactly, so the two separate
        # "found the right element" from "issued the right action on it".
        "eval_score": metrics.get("element_acc", 0.0),
        "eval_n": len(eval_rows),
        "eval_success": sum(r["step_acc"] for r in eval_rows),
        # The paired tests in summarize_mind2web.py run on this: one Bernoulli
        # per step, exactly as `any_success` is one per task on ALFWorld.
        "eval_success_rate": metrics.get("step_success_rate", 0.0),
        "eval_element_acc": metrics.get("element_acc", 0.0),
        "eval_action_f1": metrics.get("action_f1", 0.0),
        "eval_task_success_rate": metrics.get("task_success_rate", 0.0),
        "eval_n_annotations": metrics.get("n_tasks", 0),
        "eval_skipped_no_pos_candidate": metrics.get("skipped_no_pos_candidate", 0.0),
        "eval_by_family": _by_family(eval_rows),
        "evolve_n": len(evolve_rows),
        "evolve_total": args.evolve_step_offset + len(evolve_rows),
        "resumed_from": args.resume_store,
        "evolve_success_rate": evolve_metrics.get("step_success_rate"),
        "evolve_element_acc": evolve_metrics.get("element_acc"),
        "evolve_outcomes": _counts(r["outcome"] for r in evolve_rows),
        "store": system.store.summary() if getattr(system, "store", None) is not None else None,
        "writer_usage": (
            system.writer.llm.usage.to_dict()
            if getattr(system, "writer", None) is not None else None
        ),
        "mean_injected_tokens": _mean(
            r["rollouts"][0].get("injected_memory_tokens") or 0 for r in eval_rows
        ),
        "mean_llm_calls_per_step": _mean(r["n_llm_calls"] for r in eval_rows),
        "wall_seconds": round(time.time() - t_start, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n",
                                      encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))


def _counts(values):
    out: dict[str, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return out


def _mean(values):
    vals = list(values)
    return round(sum(vals) / len(vals), 1) if vals else 0.0


def _by_family(rows):
    agg: dict[str, list[int]] = {}
    for r in rows:
        agg.setdefault(r["task_family"] or "unknown", []).append(int(r["step_acc"]))
    return {k: {"n": len(v), "success": sum(v), "rate": sum(v) / len(v)} for k, v in agg.items()}


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *_):
        return False


if __name__ == "__main__":
    main()

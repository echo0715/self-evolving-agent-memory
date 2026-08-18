#!/usr/bin/env python
"""Run one memsys arm end to end on ScienceWorld: evolve, then frozen evaluation.

    python scripts/run_scienceworld.py --arm rule --policy minimal \
        --evolve-manifest manifests/scienceworld_evolve_train_50_seed42.json \
        --eval-manifest   manifests/scienceworld_eval_test_100_seed42.json

One process = one arm = one (memory type, write policy) cell of the Memory
Content table. Arms are independent, so several may run concurrently against
different vLLM servers.

Two phases, with different parallelism rules:

*Evolving* is strictly sequential. Memory written after episode i is retrieved by
episode i+1, so the loop is inherently ordered -- parallelising it would not just
be unsafe, it would measure a different system.

*Evaluation* runs under `frozen()`, where every write raises and the store is
read-only, so tasks are independent and run on a process pool.

Processes, not threads, and one simulator per worker: ScienceWorld is a Scala
simulator behind py4j, so every `ScienceWorldEnvironment` owns a JVM. The pool
initializer builds exactly one per worker and `run_task` reuses it across tasks
-- the alternative is a JVM launch per episode, which costs seconds each and
would put `--eval-workers` JVMs *plus* their churn on the node at once.

Scoring follows the benchmark: `score` is 0-100, 100 means the task was
completed, and a negative score means the agent took an action that made the
goal unreachable (almost always focusing on the wrong object). `success` is
score == 100 and is what the Memory Content table reports; the graded mean
score is carried alongside it because on this benchmark partial credit is the
majority of the signal.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from multiprocessing import get_context
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memsys import (  # noqa: E402
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
from memsys.adapters.scienceworld import (  # noqa: E402
    AgentConfig,
    ScienceWorldAgent,
    ScienceWorldEnvironment,
    TaskSpec,
    load_manifest,
    run_task,
)

ARMS = ("none", "raw", "reflection", "rule", "skill", "all")

# ---- evaluation worker state (one per process, built once by the initializer)
_WORKER: dict = {}


def _init_worker(spec: dict) -> None:
    from openai import OpenAI

    _WORKER["agent"] = ScienceWorldAgent(
        client=OpenAI(base_url=spec["base_url"], api_key="EMPTY",
                      timeout=spec["timeout"], max_retries=3),
        model=spec["model"],
        config=AgentConfig(max_steps=spec["max_steps"], max_tokens=spec["max_tokens"],
                           temperature=spec["temperature"]),
    )
    # One JVM per worker, alive for the worker's whole life. Never closed
    # explicitly: the pool tears the process down, and py4j's shutdown races
    # with interpreter finalisation noisily enough to look like a real error.
    _WORKER["env"] = ScienceWorldEnvironment(max_steps=spec["max_steps"])


def _run_eval_task(payload: tuple[dict, str]) -> dict:
    """Run one frozen-evaluation episode. The memory block is already resolved."""
    task_dict, memory_block = payload
    t0 = time.time()
    ep = run_task(
        _WORKER["agent"], TaskSpec.from_dict(task_dict),
        memory_block=memory_block, n_rollouts=1, env=_WORKER["env"],
    )
    return episode_row(ep, time.time() - t0)


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

    # device="cpu" on purpose: both GPUs are held by vLLM at 0.90 memory
    # utilisation, so an auto-placed encoder either OOMs or steals KV cache from
    # the server it is trying to talk to.
    return MemoryStore(
        embedder=SentenceTransformerEmbedder(args.embedding_model, device=args.embedding_device),
        config=config,
    )


def make_agent(args, base_url: str) -> ScienceWorldAgent:
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key="EMPTY", timeout=args.timeout, max_retries=3)
    return ScienceWorldAgent(
        client=client,
        model=args.model,
        config=AgentConfig(
            max_steps=args.max_steps,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        ),
    )


def _dotenv(name: str) -> str | None:
    """Read one key from the repo's .env, which is gitignored and holds any
    gateway credential. Real environment variables win over the file."""
    import os

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
    study, while the actor stays fixed at `--model`."""
    model = args.writer_model or args.model
    key = "EMPTY"
    if args.writer_api_key_env:
        key = _dotenv(args.writer_api_key_env)
        if not key:
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


def episode_row(ep, seconds: float) -> dict:
    r0 = ep.rollouts[0]
    return {
        "task_id": ep.task_id,
        "instruction": ep.instruction,
        "task_family": ep.scope.get("task_type", ""),
        "outcome": ep.outcome(),
        "success_rate": ep.success_rate,
        "any_success": ep.any_success,
        # ScienceWorld-specific and the reason this is not the ALFWorld row:
        # binary success throws away most of the signal here, and
        # `task_failed_by_action` separates "ran out of steps" from "killed the
        # episode with a wrong focus", which are different problems for memory.
        "score": max((x.meta.get("score", 0) for x in ep.rollouts), default=0),
        "task_failed_by_action": any(x.meta.get("task_failed_by_action") for x in ep.rollouts),
        "rollouts": [
            {
                "rollout_id": x.rollout_id,
                "success": x.success,
                "reward": x.reward,
                "score": x.meta.get("score", 0),
                "n_steps": len(x.steps),
                "error": x.error,
                **{k: x.meta.get(k) for k in
                   ("parse_failures", "nothing_happens", "task_failed_by_action",
                    "prompt_tokens", "completion_tokens", "injected_memory_tokens")},
            }
            for x in ep.rollouts
        ],
        "seconds": round(seconds, 1),
        "_r0_steps": len(r0.steps),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--policy", choices=("minimal", "full"), default="full")
    ap.add_argument("--evolve-manifest")
    ap.add_argument("--eval-manifest", required=True)
    ap.add_argument("--out", required=True, help="output directory for this arm")

    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--agent-base-url", default="http://localhost:8000/v1")
    ap.add_argument("--writer-base-url", default=None)
    ap.add_argument("--writer-model", default=None)
    ap.add_argument("--writer-api", choices=("chat", "responses"), default="chat")
    ap.add_argument("--writer-api-key-env", default=None)
    ap.add_argument("--writer-reasoning-effort", default=None)
    ap.add_argument("--timeout", type=float, default=180.0)

    ap.add_argument("--n-rollouts", type=int, default=1, help="rollouts per evolving task")
    # 100 is ScienceWorld's own default and what published baselines use; see
    # AgentConfig.max_steps for why halving it would silently exclude families.
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--writer-max-tokens", type=int, default=1024)

    ap.add_argument("--injection-budget", type=int, default=1500)
    ap.add_argument("--max-items", type=int, default=200)
    ap.add_argument("--n-max", type=int, default=2)
    ap.add_argument("--batch-every", type=int, default=25)
    ap.add_argument("--embedder", choices=("hashing", "st"), default="st")
    ap.add_argument("--embedding-model", default="BAAI/bge-large-en-v1.5")
    ap.add_argument("--embedding-device", default="cpu", help="keep off the GPUs vLLM owns")

    ap.add_argument("--eval-workers", type=int, default=4)
    ap.add_argument("--evolve-limit", type=int, default=0, help="0 = all tasks in manifest")
    ap.add_argument("--eval-limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--success-only-writes", action="store_true")
    ap.add_argument("--failure-only-writes", action="store_true")
    ap.add_argument("--evolve-until-successes", type=int, default=0)
    ap.add_argument("--evolve-until-failures", type=int, default=0)
    ap.add_argument("--resume-store", default=None,
                    help="continue a previous run's store.jsonl instead of starting empty")
    ap.add_argument("--evolve-step-offset", type=int, default=0,
                    help="start the Evolver's step counter here, so a resumed run's "
                         "batch-induction cadence lines up with an uninterrupted one")
    args = ap.parse_args()

    if args.success_only_writes and args.failure_only_writes:
        raise SystemExit("--success-only-writes and --failure-only-writes are mutually exclusive")

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
        n_success = 0
        n_fail = 0
        # One simulator for the whole sequential phase, same reasoning as the
        # eval workers: `load()` re-initialises the world, so a JVM per episode
        # would buy nothing.
        env = ScienceWorldEnvironment(max_steps=args.max_steps)
        try:
            with (out / "evolve_episodes.jsonl").open("w", encoding="utf-8") as fh:
                for i, task in enumerate(tasks):
                    t0 = time.time()
                    # Retrieve first, act second, write third: the entries handed
                    # to step_once are exactly the ones the agent saw, which is
                    # what makes the verification pass attributable.
                    ret = ev.retrieve(
                        task.instruction, scope={"env": "scienceworld", "task_type": task.task_family}
                    )
                    ep = run_task(agent, task, memory_block=ret.block,
                                  n_rollouts=args.n_rollouts, env=env)
                    if args.success_only_writes:
                        written = bool(ep.any_success)
                    elif args.failure_only_writes:
                        written = not ep.any_success
                    else:
                        written = True
                    if written:
                        row_step = ev.step
                        ev.step_once(ep, ret)
                    else:
                        row_step = None
                    row = episode_row(ep, time.time() - t0)
                    row["step"] = row_step
                    row["written"] = written
                    row["task_index"] = i
                    evolve_rows.append(row)
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    fh.flush()
                    n_success += int(bool(ep.any_success))
                    n_fail += int(not ep.any_success)
                    store = getattr(system, "store", None)
                    print(f"[evolve {i+1}/{len(tasks)}] {row['outcome']:<12} "
                          f"score={row['score']:>4} {'W' if written else '.'} "
                          f"succ={n_success} fail={n_fail} "
                          f"store={len(store) if store is not None else '-'} "
                          f"{row['seconds']:.0f}s {task.task_id}", flush=True)
                    if args.evolve_until_successes and n_success >= args.evolve_until_successes:
                        print(f"[evolve] reached {n_success} successes after {i+1} tasks", flush=True)
                        break
                    if args.evolve_until_failures and n_fail >= args.evolve_until_failures:
                        print(f"[evolve] reached {n_fail} failures after {i+1} tasks", flush=True)
                        break
                else:
                    if args.evolve_until_successes:
                        print(f"[evolve] WARNING: manifest exhausted at {n_success} successes "
                              f"(<{args.evolve_until_successes}) after {len(tasks)} tasks", flush=True)
                    if args.evolve_until_failures:
                        print(f"[evolve] WARNING: manifest exhausted at {n_fail} failures "
                              f"(<{args.evolve_until_failures}) after {len(tasks)} tasks", flush=True)
        finally:
            env.close()
        ev.flush()
        if getattr(system, "store", None) is not None:
            system.store.save(str(out / "store.jsonl"))

    # ----------------------------------------------------------- evaluation
    eval_tasks = load_manifest(args.eval_manifest)
    if args.eval_limit:
        eval_tasks = eval_tasks[: args.eval_limit]

    # Retrieval happens HERE, in the parent, under frozen(): the store lives in
    # this process and evaluation must not write to it. Each worker then receives
    # a finished memory block and needs no access to the memory system at all.
    ctx = frozen(system) if system is not None else _nullcontext()
    with ctx:
        payloads = []
        for task in eval_tasks:
            block = ""
            if system is not None:
                block = system.retrieve(
                    task.instruction, scope={"env": "scienceworld", "task_type": task.task_family}
                ).block
            payloads.append((task.to_dict(), block))

    fh = (out / "eval.jsonl").open("w", encoding="utf-8")
    eval_rows = []
    agent_spec = {
        "base_url": args.agent_base_url, "model": args.model, "timeout": args.timeout,
        "max_steps": args.max_steps, "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }
    mp_ctx = get_context("spawn")  # fork would inherit the parent's loaded torch/JVM state
    with ProcessPoolExecutor(
        max_workers=args.eval_workers, mp_context=mp_ctx,
        initializer=_init_worker, initargs=(agent_spec,),
    ) as pool:
        # as_completed, not map: map yields strictly in submission order, so one
        # slow 100-step episode at position 0 suppresses every progress line
        # behind it and a healthy run looks hung. Analysis keys on task_id.
        futures = [pool.submit(_run_eval_task, p) for p in payloads]
        for fut in as_completed(futures):
            row = fut.result()
            eval_rows.append(row)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"[eval {len(eval_rows)}/{len(eval_tasks)}] "
                  f"{'OK ' if row['any_success'] else 'FAIL'} score={row['score']:>4} "
                  f"{row['seconds']:.0f}s {row['task_id']}", flush=True)
    fh.close()

    # -------------------------------------------------------------- summary
    n_ok = sum(1 for r in eval_rows if r["any_success"])
    summary = {
        "arm": args.arm,
        "policy": args.policy,
        "model": args.model,
        "writer_model": getattr(
            getattr(getattr(system, "writer", None), "llm", None), "model", ""),
        "eval_n": len(eval_rows),
        "eval_success": n_ok,
        "eval_success_rate": n_ok / len(eval_rows) if eval_rows else 0.0,
        # The graded number the benchmark is usually reported on. Kept beside
        # completion rather than instead of it: completion is what the Memory
        # Content table compares across benchmarks, and mean score is where most
        # of ScienceWorld's signal actually lives.
        "eval_mean_score": _mean(max(0, r["score"]) for r in eval_rows),
        "eval_task_failed_by_action": sum(1 for r in eval_rows if r["task_failed_by_action"]),
        "eval_by_family": _by_family(eval_rows),
        "eval_mean_score_by_family": _score_by_family(eval_rows),
        "evolve_n": len(evolve_rows),
        "evolve_written": sum(1 for r in evolve_rows if r.get("written", True)),
        "evolve_successes": sum(1 for r in evolve_rows if r.get("any_success")),
        "evolve_failures": sum(1 for r in evolve_rows if not r.get("any_success")),
        "evolve_mean_score": _mean(max(0, r["score"]) for r in evolve_rows),
        "evolve_task_failed_by_action": sum(
            1 for r in evolve_rows if r.get("task_failed_by_action")),
        "success_only_writes": args.success_only_writes,
        "failure_only_writes": args.failure_only_writes,
        "evolve_budget": (
            {"kind": "successes", "target": args.evolve_until_successes}
            if args.evolve_until_successes else
            {"kind": "failures", "target": args.evolve_until_failures}
            if args.evolve_until_failures else {"kind": "tasks", "target": len(evolve_rows)}
        ),
        "evolve_total": args.evolve_step_offset + sum(
            1 for r in evolve_rows if r.get("written", True)
        ),
        "resumed_from": args.resume_store,
        "evolve_success_rate": (
            sum(r["success_rate"] for r in evolve_rows) / len(evolve_rows) if evolve_rows else None
        ),
        "evolve_outcomes": _counts(r["outcome"] for r in evolve_rows),
        "store": system.store.summary() if getattr(system, "store", None) is not None else None,
        "writer_usage": (
            system.writer.llm.usage.to_dict()
            if getattr(system, "writer", None) is not None else None
        ),
        "mean_injected_tokens": _mean(
            r["rollouts"][0].get("injected_memory_tokens") or 0 for r in eval_rows
        ),
        "wall_seconds": round(time.time() - t_start, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))


def _counts(values):
    out = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return out


def _mean(values):
    vals = list(values)
    return round(sum(vals) / len(vals), 1) if vals else 0.0


def _by_family(rows):
    agg: dict[str, list[int]] = {}
    for r in rows:
        agg.setdefault(r["task_family"] or "unknown", []).append(1 if r["any_success"] else 0)
    return {k: {"n": len(v), "success": sum(v), "rate": sum(v) / len(v)} for k, v in agg.items()}


def _score_by_family(rows):
    agg: dict[str, list[int]] = {}
    for r in rows:
        agg.setdefault(r["task_family"] or "unknown", []).append(max(0, r["score"]))
    return {k: round(sum(v) / len(v), 1) for k, v in agg.items()}


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *_):
        return False


if __name__ == "__main__":
    main()

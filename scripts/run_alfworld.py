#!/usr/bin/env python
"""Run one memsys arm end to end on ALFWorld: evolve, then frozen evaluation.

    python scripts/run_alfworld.py --arm rule --policy full \
        --evolve-manifest manifests/evolve_train_50_seed42.json \
        --eval-manifest   manifests/eval_valid_unseen_100_seed42.json

One process = one arm = one (memory type, write policy) cell of the Memory
Content table. Arms are independent, so several may run concurrently against
different vLLM servers.

Two phases, with different parallelism rules:

*Evolving* is strictly sequential. Memory written after episode i is retrieved by
episode i+1 (design doc 5.3), so the loop is inherently ordered -- parallelising
it would not just be unsafe, it would measure a different system.

*Evaluation* runs under `frozen()`, where every write raises and the store is
read-only, so tasks are independent and run on a thread pool.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from multiprocessing import get_context
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memsys import (  # noqa: E402
    Episode,
    Evolver,
    MemoryConfig,
    MemoryStore,
    OpenAIChatClient,
    RunLogger,
    WritePolicy,
    build_system,
    frozen,
)
from memsys.adapters.alfworld import (  # noqa: E402
    AgentConfig,
    AlfworldAgent,
    TaskSpec,
    load_manifest,
    run_task,
)

ARMS = ("none", "raw", "reflection", "rule", "skill", "all")

# ---- evaluation worker state (one per process, built once by the initializer)
_WORKER: dict = {}


def _init_worker(spec: dict) -> None:
    from openai import OpenAI

    _WORKER["data_root"] = spec["data_root"]
    _WORKER["agent"] = AlfworldAgent(
        client=OpenAI(base_url=spec["base_url"], api_key="EMPTY",
                      timeout=spec["timeout"], max_retries=3),
        model=spec["model"],
        config=AgentConfig(max_steps=spec["max_steps"], max_tokens=spec["max_tokens"],
                           temperature=spec["temperature"]),
    )


def _run_eval_task(payload: tuple[dict, str]) -> dict:
    """Run one frozen-evaluation episode. The memory block is already resolved."""
    task_dict, memory_block = payload
    t0 = time.time()
    ep = run_task(
        _WORKER["agent"], _WORKER["data_root"], TaskSpec.from_dict(task_dict),
        memory_block=memory_block, n_rollouts=1,
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
        # Fine for tests; the README is explicit that reported numbers should not
        # use it, so it stays opt-in.
        from memsys import HashingEmbedder

        return MemoryStore(embedder=HashingEmbedder(), config=config)
    from memsys import SentenceTransformerEmbedder

    # device="cpu" on purpose. sentence-transformers defaults to CUDA when it is
    # visible, and both GPUs are already held by vLLM at 0.90 memory utilisation,
    # so an auto-placed encoder either OOMs or steals KV-cache from the server it
    # is trying to talk to. The store holds a few hundred short strings; CPU is
    # far below the noise floor of a 50-step episode.
    return MemoryStore(
        embedder=SentenceTransformerEmbedder(args.embedding_model, device=args.embedding_device),
        config=config,
    )


def make_agent(args, base_url: str) -> AlfworldAgent:
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key="EMPTY", timeout=args.timeout, max_retries=3)
    return AlfworldAgent(
        client=client,
        model=args.model,
        config=AgentConfig(
            max_steps=args.max_steps,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        ),
    )


def episode_row(ep: Episode, seconds: float) -> dict:
    return {
        "task_id": ep.task_id,
        "instruction": ep.instruction,
        "task_family": ep.scope.get("task_type", ""),
        "outcome": ep.outcome(),
        "success_rate": ep.success_rate,
        "any_success": ep.any_success,
        "rollouts": [
            {
                "rollout_id": r.rollout_id,
                "success": r.success,
                "reward": r.reward,
                "n_steps": len(r.steps),
                "error": r.error,
                **{k: r.meta.get(k) for k in
                   ("parse_failures", "nothing_happens", "prompt_tokens",
                    "completion_tokens", "injected_memory_tokens")},
            }
            for r in ep.rollouts
        ],
        "seconds": round(seconds, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--policy", choices=("minimal", "full"), default="full")
    ap.add_argument("--evolve-manifest")
    ap.add_argument("--eval-manifest", required=True)
    ap.add_argument("--data-root", default=os.environ.get(
        "ALFWORLD_DATA", "/gpfs/radev/scratch/cohan/jw3278/alfworld_data"))
    ap.add_argument("--out", required=True, help="output directory for this arm")

    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--agent-base-url", default="http://localhost:8000/v1")
    ap.add_argument("--writer-base-url", default=None,
                    help="defaults to --agent-base-url; point elsewhere to split GPUs")
    ap.add_argument("--timeout", type=float, default=180.0)

    ap.add_argument("--n-rollouts", type=int, default=1, help="rollouts per evolving task")
    ap.add_argument("--max-steps", type=int, default=50)
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
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    config = build_config(args)
    writer_url = args.writer_base_url or args.agent_base_url

    # The "none" arm has no store, no writer and no evolving phase: it measures
    # the model + scaffold alone. Without it no delta is interpretable, because
    # ALFWorld's run-to-run noise floor at n=100 is several points.
    system = None
    if args.arm != "none":
        llm = OpenAIChatClient(
            args.model, base_url=writer_url, api_key="EMPTY",
            temperature=0.0, max_tokens=args.writer_max_tokens, timeout=args.timeout,
        )
        system = build_system(args.arm, llm=llm, config=config, store=build_store(args, config))

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
        with (out / "evolve_episodes.jsonl").open("w", encoding="utf-8") as fh:
            for i, task in enumerate(tasks):
                t0 = time.time()
                # Retrieve first, act second, write third: the entries handed to
                # step_once are exactly the ones the agent saw, which is what makes
                # the verification pass (support/refute) attributable.
                ret = ev.retrieve(
                    task.instruction, scope={"env": "alfworld", "task_type": task.task_family}
                )
                ep = run_task(agent, args.data_root, task,
                              memory_block=ret.block, n_rollouts=args.n_rollouts)
                ev.step_once(ep, ret)
                row = episode_row(ep, time.time() - t0)
                row["step"] = i
                evolve_rows.append(row)
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                store = getattr(system, "store", None)
                print(f"[evolve {i+1}/{len(tasks)}] {row['outcome']:<12} "
                      f"store={len(store) if store is not None else '-'} "
                      f"{row['seconds']:.0f}s {task.task_id}", flush=True)
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
                    task.instruction, scope={"env": "alfworld", "task_type": task.task_family}
                ).block
            payloads.append((task.to_dict(), block))

    # Processes, not threads. ALFWorld's PDDL backend keeps per-module state that
    # two threads corrupt: loading a game parses the grammar with a module-level
    # tatsu parser, and stepping goes through textworld's PDDL layer, which blew
    # up in `_gather_infos` as `TypeError: 'NoneType' object is not subscriptable`
    # once evaluation ran 2-wide. Neither traceback mentions concurrency, and a
    # sequential run never reproduces it. Separate processes give each worker its
    # own copy of all of it, which removes the whole failure class rather than
    # guessing which global to lock next.
    fh = (out / "eval.jsonl").open("w", encoding="utf-8")
    eval_rows = []
    agent_spec = {
        "base_url": args.agent_base_url, "model": args.model, "timeout": args.timeout,
        "max_steps": args.max_steps, "max_tokens": args.max_tokens,
        "temperature": args.temperature, "data_root": args.data_root,
    }
    mp_ctx = get_context("spawn")  # fork would inherit the parent's loaded CUDA/torch state
    with ProcessPoolExecutor(
        max_workers=args.eval_workers, mp_context=mp_ctx,
        initializer=_init_worker, initargs=(agent_spec,),
    ) as pool:
        # as_completed, not map: map yields strictly in submission order, so one
        # slow 50-step episode at position 0 suppresses every progress line behind
        # it and a healthy run becomes indistinguishable from a hung one.
        # Analysis keys on task_id, so completion order costs nothing.
        futures = [pool.submit(_run_eval_task, p) for p in payloads]
        for fut in as_completed(futures):
            row = fut.result()
            eval_rows.append(row)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"[eval {len(eval_rows)}/{len(eval_tasks)}] "
                  f"{'OK ' if row['any_success'] else 'FAIL'} {row['seconds']:.0f}s {row['task_id']}",
                  flush=True)
    fh.close()

    # -------------------------------------------------------------- summary
    n_ok = sum(1 for r in eval_rows if r["any_success"])
    summary = {
        "arm": args.arm,
        "policy": args.policy,
        "model": args.model,
        "eval_n": len(eval_rows),
        "eval_success": n_ok,
        "eval_success_rate": n_ok / len(eval_rows) if eval_rows else 0.0,
        "eval_by_family": _by_family(eval_rows),
        "evolve_n": len(evolve_rows),
        "evolve_success_rate": (
            sum(r["success_rate"] for r in evolve_rows) / len(evolve_rows) if evolve_rows else None
        ),
        "evolve_outcomes": _counts(r["outcome"] for r in evolve_rows),
        # `is not None`, not truthiness: MemoryStore defines __len__, so an empty
        # store is falsy and would be reported as "no store at all" -- hiding the
        # exact case worth seeing (a system that ended up with nothing).
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


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *_):
        return False


if __name__ == "__main__":
    main()

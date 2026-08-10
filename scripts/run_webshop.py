#!/usr/bin/env python
"""Run one memsys arm end to end on WebShop: evolve, then frozen evaluation.

    python scripts/run_webshop.py --arm rule --policy full \
        --evolve-manifest manifests/webshop_evolve_train_50_seed42.json \
        --eval-manifest   manifests/webshop_eval_test_100_seed42.json \
        --out $MEMSYS_RESULTS_ROOT/webshop/full/rule_full

One process = one arm = one (memory type, write policy) cell of the Memory
Content table. Arms are independent and may run concurrently against different
vLLM servers.

Two phases, with different parallelism rules -- identical in intent to
`run_alfworld.py`:

*Evolving* is strictly sequential. Memory written after episode i is retrieved by
episode i+1, so the loop is inherently ordered; parallelising it would measure a
different system.

*Evaluation* runs under `frozen()`, where every write raises, so tasks are
independent and fan out. Unlike ALFWorld this uses **threads, not processes**:
the environment is not in this process at all, it is behind
`scripts/webshop_server.py`. A worker here only makes HTTP calls to that server
and to vLLM, both IO-bound, so threads are the right tool and there is no
thread-unsafe PDDL backend to hide from. Concurrency safety lives in the server,
which serialises env access behind one lock.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from urllib.request import urlopen

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
from memsys.adapters.webshop import (  # noqa: E402
    SCOPE_ENV,
    AgentConfig,
    TaskSpec,
    WebShopAgent,
    load_manifest,
    run_task,
)

ARMS = ("none", "raw", "reflection", "rule", "skill", "all")


def server_health(url: str) -> dict:
    with urlopen(url.rstrip("/") + "/health", timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def check_servers(urls: list[str], manifest_path: str) -> list[dict]:
    """Refuse to run unless every server *is* the one the manifest was built for.

    A WebShop task is an integer index into a shuffled goal list, and what that
    integer means depends on the corpus, on `human_goals`, and on the RNG seed --
    product prices are drawn at load time and each goal's "price lower than X
    dollars" clause is sampled from them. Point this at a server with a different
    triple and every arm still runs to completion, still reports a success rate,
    and is silently solving different tasks. Cheap check, unrecoverable failure.
    """
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    want = manifest.get("server") or {}
    healths = []
    for url in urls:
        h = server_health(url)
        healths.append(h)
        for key in ("scale", "corpus", "index", "n_goals", "human_goals", "seed"):
            if key in want and want[key] != h.get(key):
                raise SystemExit(
                    f"server {url} does not match the manifest: {key}={h.get(key)!r} "
                    f"but manifest was built against {want[key]!r}"
                )
    return healths


class RoundRobin:
    """Hand out server URLs so concurrent episodes spread over the servers."""

    def __init__(self, urls: list[str]):
        self.urls = list(urls)
        self._i = 0
        self._lock = threading.Lock()

    def next(self) -> str:
        with self._lock:
            url = self.urls[self._i % len(self.urls)]
            self._i += 1
            return url


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


def make_agent(args, base_url: str) -> WebShopAgent:
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key="EMPTY", timeout=args.timeout, max_retries=3)
    return WebShopAgent(
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
        # WebShop's own headline number. `any_success` is the strict rate
        # (reward == 1.0); `score` is the graded reward the benchmark also
        # reports, and the two can move in opposite directions.
        "score": max((r.reward for r in ep.rollouts), default=0.0),
        "rollouts": [
            {
                "rollout_id": r.rollout_id,
                "success": r.success,
                "reward": r.reward,
                "n_steps": len(r.steps),
                "error": r.error,
                **{k: r.meta.get(k) for k in
                   ("parse_failures", "invalid_actions", "thought_steps", "purchased",
                    "prompt_tokens", "completion_tokens", "injected_memory_tokens")},
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
    ap.add_argument("--server", action="append", default=None,
                    help="WebShop env server URL; repeat to spread load (default $WEBSHOP_SERVER_URL)")
    ap.add_argument("--out", required=True, help="output directory for this arm")

    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--agent-base-url", default="http://localhost:8000/v1")
    ap.add_argument("--writer-base-url", default=None,
                    help="defaults to --agent-base-url; point elsewhere to split GPUs")
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--env-timeout", type=float, default=180.0)

    ap.add_argument("--n-rollouts", type=int, default=1, help="rollouts per evolving task")
    ap.add_argument("--max-steps", type=int, default=15)
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

    ap.add_argument("--eval-workers", type=int, default=8)
    ap.add_argument("--evolve-limit", type=int, default=0, help="0 = all tasks in manifest")
    ap.add_argument("--eval-limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    servers = args.server or [os.environ.get("WEBSHOP_SERVER_URL", "http://localhost:7000")]
    healths = check_servers(servers, args.eval_manifest)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    config = build_config(args)
    writer_url = args.writer_base_url or args.agent_base_url

    # The "none" arm has no store, no writer and no evolving phase: it measures
    # the model + scaffold alone. Without it no delta is interpretable.
    system = None
    if args.arm != "none":
        llm = OpenAIChatClient(
            args.model, base_url=writer_url, api_key="EMPTY",
            temperature=0.0, max_tokens=args.writer_max_tokens, timeout=args.timeout,
        )
        system = build_system(args.arm, llm=llm, config=config, store=build_store(args, config))

    agent = make_agent(args, args.agent_base_url)
    pool = RoundRobin(servers)

    (out / "config.json").write_text(json.dumps({
        "args": vars(args),
        "servers": healths,
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
                # step_once are exactly the ones the agent saw, which is what
                # makes the verification pass attributable.
                ret = ev.retrieve(task.instruction, scope=task.scope())
                ep = run_task(agent, pool.next(), task, memory_block=ret.block,
                              n_rollouts=args.n_rollouts, timeout=args.env_timeout)
                ev.step_once(ep, ret)
                row = episode_row(ep, time.time() - t0)
                row["step"] = i
                evolve_rows.append(row)
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                store = getattr(system, "store", None)
                print(f"[evolve {i+1}/{len(tasks)}] {row['outcome']:<12} "
                      f"score={row['score']:.2f} "
                      f"store={len(store) if store is not None else '-'} "
                      f"{row['seconds']:.0f}s {task.task_id}", flush=True)
        ev.flush()
        if getattr(system, "store", None) is not None:
            system.store.save(str(out / "store.jsonl"))

    # ----------------------------------------------------------- evaluation
    eval_tasks = load_manifest(args.eval_manifest)
    if args.eval_limit:
        eval_tasks = eval_tasks[: args.eval_limit]

    # Retrieval happens HERE, in the parent, under frozen(): evaluation must not
    # write to the store. Each worker then receives a finished memory block and
    # needs no access to the memory system at all.
    ctx = frozen(system) if system is not None else _nullcontext()
    with ctx:
        payloads = []
        for task in eval_tasks:
            block = ""
            if system is not None:
                block = system.retrieve(task.instruction, scope=task.scope()).block
            payloads.append((task, block))

    def _run_one(payload):
        task, block = payload
        t0 = time.time()
        ep = run_task(agent, pool.next(), task, memory_block=block,
                      n_rollouts=1, timeout=args.env_timeout)
        return episode_row(ep, time.time() - t0)

    fh = (out / "eval.jsonl").open("w", encoding="utf-8")
    eval_rows = []
    with ThreadPoolExecutor(max_workers=args.eval_workers) as tp:
        # as_completed, not map: map yields in submission order, so one slow
        # episode at position 0 suppresses every progress line behind it and a
        # healthy run becomes indistinguishable from a hung one. Analysis keys on
        # task_id, so completion order costs nothing.
        futures = [tp.submit(_run_one, p) for p in payloads]
        for fut in as_completed(futures):
            row = fut.result()
            eval_rows.append(row)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"[eval {len(eval_rows)}/{len(eval_tasks)}] "
                  f"{'OK  ' if row['any_success'] else 'FAIL'} score={row['score']:.2f} "
                  f"{row['seconds']:.0f}s {row['task_id']}", flush=True)
    fh.close()

    # -------------------------------------------------------------- summary
    n_ok = sum(1 for r in eval_rows if r["any_success"])
    summary = {
        "benchmark": "WebShop",
        "arm": args.arm,
        "policy": args.policy,
        "model": args.model,
        "eval_split": eval_tasks[0].split if eval_tasks else "",
        "eval_n": len(eval_rows),
        "eval_success": n_ok,
        "eval_success_rate": n_ok / len(eval_rows) if eval_rows else 0.0,
        # WebShop reports both; see episode_row.
        "eval_score": _mean(r["score"] for r in eval_rows),
        "eval_purchase_rate": _mean(
            1.0 if (r["rollouts"][0].get("purchased")) else 0.0 for r in eval_rows
        ),
        "eval_by_family": _by_family(eval_rows),
        "evolve_n": len(evolve_rows),
        "evolve_success_rate": (
            sum(r["success_rate"] for r in evolve_rows) / len(evolve_rows) if evolve_rows else None
        ),
        "evolve_score": _mean(r["score"] for r in evolve_rows) if evolve_rows else None,
        "evolve_outcomes": _counts(r["outcome"] for r in evolve_rows),
        # `is not None`, not truthiness: MemoryStore defines __len__, so an empty
        # store is falsy and would be reported as "no store at all" -- hiding the
        # exact case worth seeing.
        "store": system.store.summary() if getattr(system, "store", None) is not None else None,
        "writer_usage": (
            system.writer.llm.usage.to_dict()
            if getattr(system, "writer", None) is not None else None
        ),
        "mean_injected_tokens": _mean(
            r["rollouts"][0].get("injected_memory_tokens") or 0 for r in eval_rows
        ),
        "mean_steps": _mean(r["rollouts"][0].get("n_steps") or 0 for r in eval_rows),
        "mean_invalid_actions": _mean(
            r["rollouts"][0].get("invalid_actions") or 0 for r in eval_rows
        ),
        "wall_seconds": round(time.time() - t_start, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))


def _counts(values):
    out: dict[str, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return out


def _mean(values):
    vals = list(values)
    return round(sum(vals) / len(vals), 3) if vals else 0.0


def _by_family(rows):
    agg: dict[str, list[tuple[int, float]]] = {}
    for r in rows:
        agg.setdefault(r["task_family"] or "unknown", []).append(
            (1 if r["any_success"] else 0, r["score"])
        )
    return {
        k: {
            "n": len(v),
            "success": sum(s for s, _ in v),
            "rate": sum(s for s, _ in v) / len(v),
            "score": round(sum(sc for _, sc in v) / len(v), 3),
        }
        for k, v in agg.items()
    }


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *_):
        return False


if __name__ == "__main__":
    main()

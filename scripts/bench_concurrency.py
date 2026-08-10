#!/usr/bin/env python
"""Measure real ALFWorld episode throughput vs concurrency, before committing hours.

    python scripts/bench_concurrency.py --levels 1,4,8 --episodes 8

Guessing this wrong is expensive in both directions. SAGE measured a 3-5x
per-episode *slowdown* from running agents concurrently against a Qwen3.5
server -- it is a hybrid Gated-DeltaNet model, so vLLM disables prefix caching
and every ReAct step re-prefills the whole conversation. But their contexts had
grown to ~100k tokens; ours are capped near 15k, which is a different regime.

So: measure. The number that matters is episodes/minute (throughput), not
seconds/episode (latency) -- latency is expected to degrade, and that is fine
as long as total throughput rises.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memsys.adapters.alfworld import (  # noqa: E402
    TaskSpec,
    load_manifest,
)
# Same worker machinery as the real runner. Processes, not threads: ALFWorld's
# PDDL backend is not thread-safe, and a threaded version of this benchmark dies
# inside tatsu with `IndexError: pop from empty list`.
from run_alfworld import _init_worker, _run_eval_task  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifests/eval_valid_unseen_100_seed42.json")
    ap.add_argument("--data-root", default="/gpfs/radev/scratch/cohan/jw3278/alfworld_data")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--levels", default="1,4,8")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--offset", type=int, default=50, help="skip tasks used elsewhere")
    args = ap.parse_args()

    all_tasks = load_manifest(args.manifest)
    spec = {
        "base_url": args.base_url, "model": args.model, "timeout": 180.0,
        "max_steps": args.max_steps, "max_tokens": 256, "temperature": 0.7,
        "data_root": args.data_root,
    }
    mp_ctx = get_context("spawn")

    print(f"{'conc':>5} {'episodes':>9} {'wall_s':>8} {'ep/min':>8} {'med_s':>7} {'ok':>4}")
    best = (0.0, 0)
    for level in [int(x) for x in args.levels.split(",")]:
        # A distinct task slice per level: re-running the same tasks would let a
        # lucky draw of short episodes masquerade as a throughput win.
        tasks = (all_tasks * 4)[args.offset : args.offset + args.episodes]
        args.offset += args.episodes
        payloads = [(t.to_dict(), "") for t in tasks]

        t0 = time.time()
        with ProcessPoolExecutor(
            max_workers=level, mp_context=mp_ctx,
            initializer=_init_worker, initargs=(spec,),
        ) as pool:
            rows = list(pool.map(_run_eval_task, payloads))
        wall = time.time() - t0
        rate = 60.0 * len(rows) / wall
        n_ok = sum(1 for r in rows if r["any_success"])
        print(f"{level:>5} {len(rows):>9} {wall:>8.0f} {rate:>8.2f} "
              f"{statistics.median(r['seconds'] for r in rows):>7.0f} {n_ok:>4}", flush=True)
        if rate > best[0]:
            best = (rate, level)

    print(f"\nbest throughput: {best[0]:.2f} episodes/min at concurrency {best[1]}")


if __name__ == "__main__":
    main()

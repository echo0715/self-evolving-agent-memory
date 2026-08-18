#!/usr/bin/env python
"""Build frozen ScienceWorld manifests with the task description resolved.

    python scripts/build_scienceworld_manifests.py --out manifests \
        --evolve-split train --evolve-count 100 \
        --eval-split test --eval-count 100 --seed 42

Four properties, three of them shared with the other benchmarks' builders and
one forced by ScienceWorld's shape.

**Splits are the simulator's own.** `get_variations_train/dev/test()` partition
each task's variation indices; evolving draws from `train`, evaluation from
`test`, so the sets are disjoint by construction. No id-split file is involved.
Note these lists are a property of the *currently loaded task* -- reading them
after loading a different task returns another task's index range, which is the
usual way a stale manifest gets built (`memsys/adapters/scienceworld.py`
docstring, point 3). This script always re-loads before reading.

**Round-robin across the 30 tasks, not a flat permutation.** Variation counts
per task span 4 to 405: `find-animal` has 150 train variations and `boil` has
14. A prefix of one flat permutation over all (task, variation) pairs would
therefore be ~11x more `use-thermometer` than `boil`, and "evolve on 100 tasks"
would mostly mean "evolve on the three biggest families". Selection instead
permutes each task's variations with the seed, permutes the task order with the
same seed, and then takes one variation per task per lap. So 100 evolving tasks
is 3-4 variations of each of the 30 families, and `task_type` -- the retrieval
scope key and the batch-induction cluster key -- is close to uniform.

**Nested prefixes.** Because selection is a prefix of *one* round-robin order,
the 25-task set is a prefix of the 50-task set is a prefix of the 100-task set.
That is what makes "evolve on 25 / 50 / 100" a comparison of amount of
experience rather than three unrelated draws, and it is why `--evolve-skip`
emits positions [skip, count) for a continuation.

**Resolved instructions.** Each entry carries its task description. Retrieval
happens before the agent acts, but ScienceWorld only reveals the description
after `load()` + `reset()`, so it is resolved here. Unlike ALFWorld the
description is not part of the opening observation either, so this pass is what
lets the agent be *shown* the goal at all. Resolving is the slow part of this
script (one reset per task, no LLM involved).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memsys.adapters.scienceworld import (  # noqa: E402
    TASK_NAMES,
    TaskSpec,
    _ensure_java_on_path,
)

SPLIT_GETTERS = {
    "train": "get_variations_train",
    "dev": "get_variations_dev",
    "test": "get_variations_test",
}


def variation_pools(env, split: str) -> dict[str, list[int]]:
    """`{task_name: [variation, ...]}` for one split, in the simulator's order.

    The `load()` before each read is not redundant: the variation lists belong
    to the loaded task, so reading them in a loop without re-loading returns the
    previous task's indices for every task after the first.
    """
    getter = SPLIT_GETTERS[split]
    pools = {}
    for name in TASK_NAMES:
        env.load(name, 0, "")
        pools[name] = [int(v) for v in getattr(env, getter)()]
    return pools


def round_robin_order(pools: dict[str, list[int]], seed: int) -> list[tuple[str, int]]:
    """One seeded order over (task, variation), balanced across tasks.

    Each task's variations are shuffled with the seed and the task order is
    shuffled with the same seed; then one variation is drawn per task per lap.
    Tasks that run out drop off, so the tail is drawn from the biggest families
    only -- which is unavoidable, and irrelevant at the sizes this study uses
    (100 tasks over 30 families is 4 laps, and the smallest family has 4
    train variations).
    """
    rng = random.Random(seed)
    shuffled = {}
    for name in sorted(pools):
        vs = list(pools[name])
        rng.shuffle(vs)
        shuffled[name] = vs
    order_of_tasks = sorted(shuffled)
    rng.shuffle(order_of_tasks)

    out: list[tuple[str, int]] = []
    lap = 0
    while len(out) < sum(len(v) for v in shuffled.values()):
        drew = False
        for name in order_of_tasks:
            if lap < len(shuffled[name]):
                out.append((name, shuffled[name][lap]))
                drew = True
        if not drew:
            break
        lap += 1
    return out


def resolve(env, pairs: list[tuple[str, int]], split: str) -> list[TaskSpec]:
    tasks: list[TaskSpec] = []
    for i, (name, var) in enumerate(pairs, 1):
        env.load(name, var, "")
        obs, _ = env.reset()
        desc = str(env.get_task_description()).strip()
        if "exceeds the total number of variations" in str(obs) or not desc or desc == "unknown":
            raise SystemExit(
                f"{name}/{var}: simulator returned no usable task description "
                f"(obs={str(obs)[:120]!r}) -- refusing to write a manifest whose "
                f"entries would look like agent failures at run time")
        tasks.append(TaskSpec(
            task_id=f"{name}/{var}", split=split, task_name=name,
            variation=var, task_family=name, instruction=desc,
        ))
        if i % 25 == 0 or i == len(pairs):
            print(f"  resolved {i}/{len(pairs)}", flush=True)
    return tasks


def write_manifest(path: Path, split: str, seed: int, available: int,
                   tasks: list[TaskSpec], skip: int = 0) -> None:
    ids = [t.task_id for t in tasks]
    families: dict[str, int] = {}
    for t in tasks:
        families[t.task_family] = families.get(t.task_family, 0) + 1
    payload = {
        "schema_version": 1,
        "benchmark": "ScienceWorld",
        "split": split,
        "seed": seed,
        "selection": ("prefix_of_seeded_round_robin_over_tasks" if not skip
                      else f"positions_[{skip},{skip + len(tasks)})_of_seeded_round_robin"),
        "skip": skip,
        "available_count": available,
        "selected_count": len(tasks),
        "selected_by_task": families,
        "task_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "tasks": [t.to_dict() for t in tasks],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(tasks)} tasks -> {path}")
    print(f"  sha256={payload['task_ids_sha256'][:16]}  "
          f"tasks_covered={len(families)}/30  per_task={sorted(set(families.values()))}")


def build(env, split: str, count: int, seed: int, out: Path, skip: int = 0) -> list[TaskSpec]:
    pools = variation_pools(env, split)
    order = round_robin_order(pools, seed)
    if count > len(order):
        raise SystemExit(f"{split} has only {len(order)} (task, variation) pairs, asked for {count}")
    selected = order[skip:count]
    print(f"[{split}] resolving descriptions for {len(selected)} of {len(order)} pairs"
          f"{f' (positions {skip}..{count})' if skip else ''}...")
    tasks = resolve(env, selected, split)
    write_manifest(out, split, seed, len(order), tasks, skip=skip)
    return tasks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="manifests")
    ap.add_argument("--evolve-split", default="train", choices=tuple(SPLIT_GETTERS))
    ap.add_argument("--evolve-count", type=int, default=100)
    ap.add_argument("--evolve-skip", type=int, default=0,
                    help="emit evolve positions [skip, count) -- continues an existing run")
    ap.add_argument("--eval-split", default="test", choices=tuple(SPLIT_GETTERS))
    ap.add_argument("--eval-count", type=int, default=100)
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    _ensure_java_on_path()
    from scienceworld import ScienceWorldEnv

    out = Path(args.out)
    env = ScienceWorldEnv("", envStepLimit=100)
    try:
        suffix = (f"_{args.evolve_count}" if not args.evolve_skip
                  else f"_{args.evolve_skip}to{args.evolve_count}")
        evolve = build(env, args.evolve_split, args.evolve_count, args.seed,
                       out / f"scienceworld_evolve_{args.evolve_split}{suffix}_seed{args.seed}.json",
                       skip=args.evolve_skip)
        if args.no_eval:
            return
        ev = build(env, args.eval_split, args.eval_count, args.seed,
                   out / f"scienceworld_eval_{args.eval_split}_{args.eval_count}_seed{args.seed}.json")
    finally:
        env.close()

    # The splits are disjoint by construction, but a builder that silently
    # stopped being true would be invisible in the results, so assert it.
    overlap = {t.task_id for t in evolve} & {t.task_id for t in ev}
    if overlap:
        raise SystemExit(f"evolve and eval overlap on {sorted(overlap)[:10]}")
    print(f"OK: {len(evolve)} evolve / {len(ev)} eval, disjoint")


if __name__ == "__main__":
    main()

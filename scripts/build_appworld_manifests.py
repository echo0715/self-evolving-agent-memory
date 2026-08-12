#!/usr/bin/env python
"""Build frozen AppWorld manifests with the instruction resolved.

    python scripts/build_appworld_manifests.py --root $APPWORLD_ROOT \
        --out manifests --evolve-count 50 --eval-count 100 --seed 42

Reads the dataset split files and each task's `specs.json` straight off disk, so
it needs neither a running environment server nor the `appworld` interpreter.

Three properties, matching the other two benchmarks' builders:

**Split by AppWorld's own splits.** Evolving draws from `train` (90 tasks),
evaluation from `test_normal` (168), so the sets are disjoint by construction.

**Nested prefixes.** Selection is a prefix of one seeded permutation, so
"evolve on 50 vs 100" compares amount of experience, not two unrelated draws.
`--evolve-skip 50 --evolve-count 100` emits positions [50, 100) of that same
permutation -- the tasks the first 50-episode run did *not* see -- so a resumed
store continues the sequence instead of repeating it.

**`train` runs out at 90, so the continuation overflows into `dev`.** AppWorld's
train split holds 90 tasks (30 scenarios); positions [50, 100) therefore need 10
more than exist. `--evolve-overflow-split dev` appends dev's own seeded scenario
permutation after train's is exhausted. dev is disjoint from `test_normal`, so
the evaluation set stays untouched; the cost is that a 100-task evolving set
spans two splits, which is recorded in the manifest as `splits`.

**Grouped by scenario, not split within it.** AppWorld task ids are
`<scenario>_<variant>` and the three variants of a scenario are near-identical
requests over the same world. Sampling tasks independently would put variant 1 in
the evolving set and variant 2 in the evaluation set of a *different* split --
harmless here because the splits are already disjoint, but within a split it
would leak. Selection is therefore over whole scenarios, which also keeps the
memsys `task_type` scope key (the scenario) meaningful rather than half-present.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memsys.adapters.appworld import TaskSpec, load_manifest  # noqa: E402

SPLITS = ("train", "dev", "test_normal", "test_challenge")


def read_split(root: Path, split: str) -> list[str]:
    path = root / "data" / "datasets" / f"{split}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"no such split file: {path}")
    return [line.strip() for line in path.read_text(encoding="utf-8").split() if line.strip()]


def read_instruction(root: Path, task_id: str) -> str:
    specs = root / "data" / "tasks" / task_id / "specs.json"
    if not specs.is_file():
        raise FileNotFoundError(f"no specs.json for {task_id}: {specs}")
    # specs.json only -- never ground_truth/. That directory holds the solution,
    # the required apps and the expected answer, and anything read from it would
    # end up in a retrieval key or a scope value.
    return str(json.loads(specs.read_text(encoding="utf-8"))["instruction"]).strip()


def ordered_tasks(root: Path, split: str, seed: int) -> list[str]:
    """Every task id of `split`, in the seeded scenario permutation's order.

    Whole scenarios in shuffled order, variants sorted within a scenario. Every
    selection in this file is a slice of this list, which is what makes
    `[0, 50)` and `[50, 100)` two halves of one sequence rather than two draws.
    """
    task_ids = read_split(root, split)
    by_scenario: dict[str, list[str]] = {}
    for tid in task_ids:
        by_scenario.setdefault(tid.split("_")[0], []).append(tid)

    scenarios = sorted(by_scenario)
    random.Random(seed).shuffle(scenarios)  # one permutation; every count is a prefix
    return [tid for scenario in scenarios for tid in sorted(by_scenario[scenario])]


def build(root: Path, split: str, count: int, seed: int, out: Path,
          skip: int = 0, overflow_splits: tuple[str, ...] = ()) -> list[TaskSpec]:
    order = [(tid, split) for tid in ordered_tasks(root, split, seed)]
    available = len(order)
    for extra in overflow_splits:
        order += [(tid, extra) for tid in ordered_tasks(root, extra, seed)]

    selected = order[skip:count]
    if len(selected) < count - skip:
        raise ValueError(
            f"{'+'.join((split,) + overflow_splits)} has only {len(order)} tasks, "
            f"asked for positions [{skip}, {count})")

    tasks = [
        TaskSpec(task_id=tid, split=src, instruction=read_instruction(root, tid),
                 scenario=tid.split("_")[0])
        for tid, src in selected
    ]
    n_scenarios = len({tid.split("_")[0] for tid, src in order if src == split})
    write_manifest(out, split, seed, available, n_scenarios, tasks,
                   skip=skip, overflow_splits=overflow_splits, ordered_count=len(order))
    return tasks


def write_manifest(
    path: Path, split: str, seed: int, available: int, n_scenarios: int, tasks: list[TaskSpec],
    skip: int = 0, overflow_splits: tuple[str, ...] = (), ordered_count: int | None = None,
) -> None:
    ids = [t.task_id for t in tasks]
    payload = {
        "schema_version": 2,
        "benchmark": "AppWorld",
        # The top-level split is the manifest's *default*; every task also carries
        # its own, which is what a continuation spilling into `dev` relies on.
        "split": split,
        "splits": sorted({t.split for t in tasks}),
        "seed": seed,
        "selection": ("prefix_of_seeded_permutation_of_scenarios" if not skip
                      else f"positions_[{skip},{skip + len(tasks)})_of_seeded_permutation_of_scenarios"),
        "skip": skip,
        "overflow_splits": list(overflow_splits),
        "available_count": available,
        "available_in_order": ordered_count if ordered_count is not None else available,
        "available_scenarios": n_scenarios,
        "selected_count": len(tasks),
        "selected_scenarios": len({t.scenario for t in tasks}),
        "task_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "tasks": [t.to_dict() for t in tasks],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(tasks)} tasks ({payload['selected_scenarios']} scenarios) -> {path}")
    print(f"  sha256={payload['task_ids_sha256'][:16]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.environ.get(
        "APPWORLD_ROOT", "/gpfs/radev/scratch/cohan/jw3278/appworld_root"))
    ap.add_argument("--out", default="manifests")
    ap.add_argument("--evolve-split", default="train", choices=SPLITS)
    ap.add_argument("--evolve-count", type=int, default=50)
    ap.add_argument("--evolve-skip", type=int, default=0,
                    help="emit evolve positions [skip, count) -- continues an existing run")
    ap.add_argument("--evolve-overflow-split", action="append", default=None, choices=SPLITS,
                    help="append this split's own seeded permutation once --evolve-split runs "
                         "out; train holds only 90 tasks, so [50, 100) needs `dev`")
    ap.add_argument("--eval-split", default="test_normal", choices=SPLITS)
    ap.add_argument("--eval-count", type=int, default=100)
    ap.add_argument("--no-eval", action="store_true", help="skip rebuilding the eval manifest")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    overflow = tuple(args.evolve_overflow_split or ())
    if args.eval_split in overflow:
        raise SystemExit(f"--evolve-overflow-split {args.eval_split} would evolve on the eval split")
    label = "+".join((args.evolve_split,) + overflow)
    suffix = (f"_{args.evolve_count}" if not args.evolve_skip
              else f"_{args.evolve_skip}to{args.evolve_count}")
    evolve = build(root, args.evolve_split, args.evolve_count, args.seed,
                   Path(args.out) / f"appworld_evolve_{label}{suffix}_seed{args.seed}.json",
                   skip=args.evolve_skip, overflow_splits=overflow)
    ev = load_manifest(Path(args.out) / f"appworld_eval_{args.eval_split}_{args.eval_count}_seed{args.seed}.json") \
        if args.no_eval else \
        build(root, args.eval_split, args.eval_count, args.seed,
              Path(args.out) / f"appworld_eval_{args.eval_split}_{args.eval_count}_seed{args.seed}.json")

    overlap = {t.task_id for t in evolve} & {t.task_id for t in ev}
    if overlap:
        raise SystemExit(f"evolve and eval sets overlap on {sorted(overlap)[:10]}")
    print(f"OK: {len(evolve)} evolve / {len(ev)} eval, disjoint")


if __name__ == "__main__":
    main()

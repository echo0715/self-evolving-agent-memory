#!/usr/bin/env python
"""Build frozen SpreadsheetBench manifests with every task field resolved.

    python scripts/build_spreadsheetbench_manifests.py \
        --data-root $SPREADSHEETBENCH_ROOT \
        --id-split-dir $SKILLOPT/data/spreadsheetbench_id_split \
        --out manifests --evolve-count 50 --eval-count 100 --seed 42

    # continue an existing 50-episode run to 100, leaving the eval set alone
    python scripts/build_spreadsheetbench_manifests.py \
        --evolve-skip 50 --evolve-count 100 --evolve-overflow-split val --no-eval

Reads `dataset.json` and the test-case directories straight off disk; no server
and no benchmark package are needed. The manifest is self-contained, so the
sibling SkillOpt checkout is a one-time dependency at build time only.

Five properties, matching the other three benchmarks' builders:

**Split by SkillOpt's published id split.** `data/spreadsheetbench_id_split`
carries the 400-task Verified subset partitioned train=80 / val=40 / test=280.
Evolving draws from train, evaluation from test, so the sets are disjoint by
construction and the task selection is the one the rest of the SAGE ecosystem
uses on this benchmark.

**Files and metadata come from `all_data_912_v0.1`, not from Verified 400.**
Every id in the split exists in both archives, but only the 912 release ships
three test cases per task; Verified 400 ships one. Three cases is upstream's own
protocol and is what makes `reward` graded and hardcoding worthless -- with one
case, `reward` collapses onto `success` and an agent can read the preview and
write literal answers. The two releases are never mixed: Verified 400 revised
some instructions *and* their golden workbooks, so pairing its instruction text
with the 912 answer files would grade a task against the wrong answer.
`--data-root` may still be pointed at Verified 400, and the adapter reads its
naming convention, but then only one case exists per task.

**Nested prefixes.** Every selection is a slice of one seeded permutation, so
"evolve on 50 vs 100" compares amount of experience, not two unrelated draws.
`--evolve-skip 50 --evolve-count 100` emits positions [50, 100) -- the tasks the
first 50-episode run did *not* see -- so a resumed store continues the sequence
instead of repeating it.

**`train` runs out at 80, so the continuation overflows into `val`.** Positions
[50, 100) need 20 more tasks than train holds. `--evolve-overflow-split val`
appends val's own seeded permutation after train's is exhausted. val is disjoint
from test, so the evaluation set stays untouched; the cost is that a 100-task
evolving set spans two splits, recorded in the manifest as `splits`.

**Grouped by source workbook, and cross-set overlap is removed.**
SpreadsheetBench sheet-level ids are `<workbook>-<question>`, and tasks sharing
the leading number are different questions over the *same* spreadsheet. The
permutation is over whole groups so a group is never split mid-way. Beyond that,
each set excludes groups already claimed by the other: evaluation drops tasks
whose workbook was used for evolving, and a continuation built against a frozen
eval manifest (`--exclude-groups-from`) drops tasks whose workbook is in it.
Exclusion happens *before* slicing, and is a no-op on the first 50 positions, so
the nested-prefix property survives it. `--allow-group-leak` keeps them, for
reproducing a run built without the guard.
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

from memsys.adapters.spreadsheetbench import TaskSpec, find_test_cases  # noqa: E402

#: `rest` is not one of SkillOpt's splits -- it is everything in the 912 release
#: that the 400-task Verified id split does not cover. It exists for one job: an
#: outcome budget that needs more evolving tasks than the split can supply (100
#: *successes* costs 420-480 tasks at this benchmark's 21-24% evolve-time
#: success rate, against 297 selectable in train+val+test). It is only ever
#: legitimate as the last overflow pool, never as an evaluation split, and the
#: group filter still applies -- so the frozen eval set stays disjoint from it by
#: task id and by source workbook, which is the property that matters.
SPLITS = ("train", "val", "test", "rest")


def read_dataset(data_root: Path) -> dict[str, dict]:
    path = data_root / "dataset.json"
    if not path.is_file():
        raise FileNotFoundError(f"no dataset.json under --data-root: {path}")
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {str(r["id"]): r for r in rows}


def read_id_split(id_split_dir: Path, split: str, dataset: dict[str, dict] | None = None
                  ) -> list[str]:
    if split == "rest":
        # Everything in the release that the published split does not cover. Not
        # a split of SkillOpt's; see the SPLITS comment.
        if dataset is None:
            raise ValueError("the 'rest' pool needs the dataset to subtract the split from")
        covered = {tid for s in ("train", "val", "test")
                   for tid in read_id_split(id_split_dir, s)}
        return sorted(set(dataset) - covered)
    path = id_split_dir / split / "items.json"
    if not path.is_file():
        raise FileNotFoundError(f"no such id split file: {path}")
    return [str(item["id"]) for item in json.loads(path.read_text(encoding="utf-8"))]


def group_of(task_id: str) -> str:
    """The source workbook. Cell-level ids are bare post numbers and stand alone."""
    return task_id.split("-")[0] if "-" in task_id else task_id


def manifest_groups(path: str | Path) -> set[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    tasks = payload["tasks"] if isinstance(payload, dict) else payload
    return {group_of(str(t["task_id"])) for t in tasks}


def ordered_tasks(ids: list[str], seed: int) -> list[str]:
    """`ids` in the seeded group permutation's order.

    Whole groups in shuffled order, questions sorted within a group. Every
    selection in this file is a slice of this list, which is what makes
    `[0, 50)` and `[50, 100)` two halves of one sequence rather than two draws.
    """
    by_group: dict[str, list[str]] = {}
    for tid in ids:
        by_group.setdefault(group_of(tid), []).append(tid)
    groups = sorted(by_group)
    random.Random(seed).shuffle(groups)
    return [tid for g in groups for tid in sorted(by_group[g])]


def make_spec(task_id: str, split: str, row: dict) -> TaskSpec:
    return TaskSpec.from_dict({**row, "task_id": task_id, "split": split}, default_split=split)


def build(
    data_root: Path,
    dataset: dict[str, dict],
    pools: list[tuple[str, list[str]]],
    count: int,
    seed: int,
    out: Path,
    skip: int = 0,
    excluded_groups: set[str] | None = None,
    allow_leak: bool = False,
) -> list[TaskSpec]:
    """`pools` is `[(split, ids), ...]`, concatenated in order after permuting each."""
    order: list[tuple[str, str]] = []
    for split, ids in pools:
        missing = [t for t in ids if t not in dataset]
        if missing:
            raise SystemExit(
                f"{len(missing)} ids from the {split} split are absent from "
                f"{data_root/'dataset.json'} (e.g. {missing[:5]}) -- wrong --data-root?"
            )
        order += [(tid, split) for tid in ordered_tasks(ids, seed)]

    excluded = set(excluded_groups or ())
    dropped = [tid for tid, _ in order if group_of(tid) in excluded] if not allow_leak else []
    if not allow_leak and excluded:
        order = [(tid, s) for tid, s in order if group_of(tid) not in excluded]

    selected = order[skip:count]
    if len(selected) < count - skip:
        raise ValueError(
            f"{'+'.join(s for s, _ in pools)} yields only {len(order)} selectable tasks, "
            f"asked for positions [{skip}, {count}) ({len(dropped)} dropped for group overlap)"
        )
    tasks = [make_spec(tid, src, dataset[tid]) for tid, src in selected]

    # Resolve the test cases now rather than discovering at run time that a task
    # has none: a missing directory would otherwise surface mid-sweep as a
    # zero-score episode indistinguishable from an agent that failed.
    n_cases = {}
    for t in tasks:
        cases = find_test_cases(data_root / t.spreadsheet_path)
        if not cases:
            raise SystemExit(f"task {t.task_id}: no test cases under "
                             f"{data_root / t.spreadsheet_path}")
        n_cases[t.task_id] = len(cases)

    write_manifest(out, pools[0][0], seed, len(order), tasks, n_cases, dropped, data_root,
                   skip=skip, overflow_splits=tuple(s for s, _ in pools[1:]))
    return tasks


def write_manifest(
    path: Path, split: str, seed: int, available: int, tasks: list[TaskSpec],
    n_cases: dict[str, int], dropped: list[str], data_root: Path,
    skip: int = 0, overflow_splits: tuple[str, ...] = (),
) -> None:
    ids = [t.task_id for t in tasks]
    counts: dict[str, int] = {}
    for t in tasks:
        counts[t.task_type] = counts.get(t.task_type, 0) + 1
    case_hist: dict[str, int] = {}
    for n in n_cases.values():
        case_hist[str(n)] = case_hist.get(str(n), 0) + 1
    payload = {
        "schema_version": 1,
        "benchmark": "SpreadsheetBench",
        # The top-level split is the manifest's *default*; every task also
        # carries its own, which is what a continuation spilling into `val`
        # relies on.
        "split": split,
        "splits": sorted({t.split for t in tasks}),
        "seed": seed,
        "selection": ("prefix_of_seeded_permutation_of_source_workbook_groups" if not skip
                      else f"positions_[{skip},{skip + len(tasks)})_of_seeded_"
                           "permutation_of_source_workbook_groups"),
        "skip": skip,
        "overflow_splits": list(overflow_splits),
        "data_root": str(data_root),
        "available_in_order": available,
        "available_groups": len({group_of(t) for t in ids}),
        "selected_count": len(tasks),
        "selected_by_task_type": counts,
        "test_cases_per_task": case_hist,
        "dropped_for_group_overlap": dropped,
        "task_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "tasks": [{**t.to_dict(), "n_cases": n_cases[t.task_id]} for t in tasks],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(tasks)} tasks -> {path}")
    print(f"  splits={payload['splits']}  task_type={counts}  cases_per_task={case_hist}")
    if dropped:
        print(f"  dropped {len(dropped)} for source-workbook overlap: {dropped}")
    print(f"  sha256={payload['task_ids_sha256'][:16]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=os.environ.get(
        "SPREADSHEETBENCH_ROOT",
        "/gpfs/radev/scratch/cohan/jw3278/spreadsheetbench_root/all_data_912_v0.1"))
    ap.add_argument("--id-split-dir", default=os.environ.get(
        "SPREADSHEETBENCH_ID_SPLIT",
        "/gpfs/radev/home/jw3278/project/SAGE/repos/SkillOpt/data/spreadsheetbench_id_split"))
    ap.add_argument("--out", default="manifests")
    ap.add_argument("--evolve-split", default="train", choices=SPLITS)
    ap.add_argument("--evolve-count", type=int, default=50)
    ap.add_argument("--evolve-skip", type=int, default=0,
                    help="emit evolve positions [skip, count) -- continues an existing run")
    ap.add_argument("--evolve-overflow-split", action="append", default=None, choices=SPLITS,
                    help="append this split's permutation once the evolve split runs out")
    ap.add_argument("--exclude-groups-from", default=None,
                    help="manifest whose source workbooks are excluded from the evolve pool; "
                         "use the frozen eval manifest when building a continuation")
    ap.add_argument("--eval-split", default="test", choices=SPLITS)
    ap.add_argument("--eval-count", type=int, default=100)
    ap.add_argument("--no-eval", action="store_true", help="do not rebuild the eval manifest")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--allow-group-leak", action="store_true",
                    help="keep tasks whose source workbook is used by the other set")
    args = ap.parse_args()

    if args.eval_split == "rest":
        # `rest` is un-curated relative to the Verified subset the study evaluates
        # on; evaluating there would change what the numbers mean.
        raise SystemExit("--eval-split rest is not allowed: `rest` is an evolving-only pool")

    data_root = Path(args.data_root).expanduser().resolve()
    id_split_dir = Path(args.id_split_dir).expanduser().resolve()
    dataset = read_dataset(data_root)
    out_dir = Path(args.out)

    pools = [(args.evolve_split, read_id_split(id_split_dir, args.evolve_split, dataset))]
    for extra in args.evolve_overflow_split or []:
        pools.append((extra, read_id_split(id_split_dir, extra, dataset)))

    excluded = manifest_groups(args.exclude_groups_from) if args.exclude_groups_from else set()
    span = (f"_{args.evolve_count}" if not args.evolve_skip
            else f"_{args.evolve_skip}to{args.evolve_count}")
    name = "+".join(s for s, _ in pools)
    evolve = build(data_root, dataset, pools, args.evolve_count, args.seed,
                   out_dir / f"spreadsheetbench_evolve_{name}{span}_seed{args.seed}.json",
                   skip=args.evolve_skip, excluded_groups=excluded,
                   allow_leak=args.allow_group_leak)

    if args.no_eval:
        if args.exclude_groups_from:
            ev = {t.task_id for t in evolve}
            frozen = json.loads(Path(args.exclude_groups_from).read_text(encoding="utf-8"))
            fids = {str(t["task_id"]) for t in frozen["tasks"]}
            if ev & fids:
                raise SystemExit(f"continuation overlaps the frozen eval set on {sorted(ev & fids)}")
            shared = {group_of(t) for t in ev} & manifest_groups(args.exclude_groups_from)
            if shared and not args.allow_group_leak:
                raise SystemExit(f"continuation shares source workbooks {sorted(shared)}")
            print(f"OK: {len(evolve)} evolve tasks, disjoint from "
                  f"{args.exclude_groups_from} by task and by source workbook")
        return

    evolve_groups = {group_of(t.task_id) for t in evolve}
    ev = build(data_root, dataset, [(args.eval_split, read_id_split(id_split_dir, args.eval_split))],
               args.eval_count, args.seed,
               out_dir / f"spreadsheetbench_eval_{args.eval_split}_{args.eval_count}"
                         f"_seed{args.seed}.json",
               excluded_groups=evolve_groups, allow_leak=args.allow_group_leak)

    overlap = {t.task_id for t in evolve} & {t.task_id for t in ev}
    if overlap:
        raise SystemExit(f"evolve and eval sets overlap on {sorted(overlap)[:10]}")
    shared = evolve_groups & {group_of(t.task_id) for t in ev}
    if shared and not args.allow_group_leak:
        raise SystemExit(f"evolve and eval share source workbooks {sorted(shared)}")
    print(f"OK: {len(evolve)} evolve / {len(ev)} eval, disjoint"
          + (f" (group leak allowed: {sorted(shared)})" if shared else ""))


if __name__ == "__main__":
    main()

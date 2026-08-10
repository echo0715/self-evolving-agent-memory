#!/usr/bin/env python
"""Build frozen ALFWorld manifests with the goal text resolved.

    python scripts/build_manifests.py \
        --data-root $ALFWORLD_DATA --out manifests \
        --evolve-split train --evolve-count 50 \
        --eval-split valid_unseen --eval-count 100 --seed 42

Two properties matter and neither is free:

**Nested prefixes.** Selection is a prefix of one seeded full permutation, not
`random.sample(k)`. So the 50-task evolve set is a prefix of the 100-task one,
which makes "evolve on 50 vs 100 vs 150" (the columns of plan.md's Memory
Content table) a comparison of *amount*, not of two disjoint task draws.

**Resolved instructions.** Each entry carries the "Your task is to: ..." goal.
Retrieval happens before the agent acts, but ALFWorld only reveals the goal on
`reset()`, so resolving it here is what lets every memory system retrieve on the
real instruction instead of a filename. This pass opens every game once, which
is the slow part of this script (no LLM involved).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memsys.adapters.alfworld import (  # noqa: E402
    ALFWorldEnvironment,
    TaskSpec,
    _family_from_path,
    extract_instruction,
)

SPLIT_DIRS = {
    "train": "train",
    "valid_seen": "valid_seen",
    "valid_unseen": "valid_unseen",
}


def discover(data_root: Path, split: str) -> list[TaskSpec]:
    base = data_root / "json_2.1.1" / SPLIT_DIRS[split]
    gamefiles = sorted(base.glob("**/game.tw-pddl"), key=lambda p: p.as_posix())
    if not gamefiles:
        raise FileNotFoundError(f"no ALFWorld games below {base}")
    return [
        TaskSpec(
            task_id=f"{SPLIT_DIRS[split]}:{g.relative_to(data_root).as_posix()}",
            split=SPLIT_DIRS[split],
            gamefile=g.relative_to(data_root).as_posix(),
            task_family=_family_from_path(g.relative_to(data_root).as_posix()),
        )
        for g in gamefiles
    ]


def resolve_instructions(data_root: Path, tasks: list[TaskSpec]) -> list[TaskSpec]:
    out = []
    env = ALFWorldEnvironment(data_root, max_steps=1)
    try:
        for i, t in enumerate(tasks, 1):
            obs = env.reset(t)
            out.append(
                TaskSpec(t.task_id, t.split, t.gamefile, t.task_family, extract_instruction(obs))
            )
            if i % 25 == 0 or i == len(tasks):
                print(f"  resolved {i}/{len(tasks)}", flush=True)
    finally:
        env.close()
    return out


def write_manifest(path: Path, split: str, seed: int, available: int, tasks: list[TaskSpec]) -> None:
    ids = [t.task_id for t in tasks]
    payload = {
        "schema_version": 2,
        "benchmark": "ALFWorld",
        "data_version": "json_2.1.1",
        "split": SPLIT_DIRS[split],
        "seed": seed,
        "selection": "prefix_of_seeded_permutation_of_lexicographically_sorted_paths",
        "available_count": available,
        "selected_count": len(tasks),
        "task_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "tasks": [t.to_dict() for t in tasks],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    families: dict[str, int] = {}
    for t in tasks:
        families[t.task_family] = families.get(t.task_family, 0) + 1
    print(f"wrote {len(tasks)} tasks -> {path}")
    print(f"  sha256={payload['task_ids_sha256'][:16]}  families={families}")


def build(data_root: Path, split: str, count: int, seed: int, out: Path) -> None:
    all_tasks = discover(data_root, split)
    if count > len(all_tasks):
        raise ValueError(f"{split} has only {len(all_tasks)} tasks, asked for {count}")
    order = list(all_tasks)
    random.Random(seed).shuffle(order)  # one permutation; every count is a prefix
    selected = order[:count]
    print(f"[{split}] resolving goal text for {count} of {len(all_tasks)} tasks...")
    selected = resolve_instructions(data_root, selected)
    write_manifest(out, split, seed, len(all_tasks), selected)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", default="manifests")
    ap.add_argument("--evolve-split", default="train")
    ap.add_argument("--evolve-count", type=int, default=50)
    ap.add_argument("--eval-split", default="valid_unseen")
    ap.add_argument("--eval-count", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    out = Path(args.out)
    build(data_root, args.evolve_split, args.evolve_count, args.seed,
          out / f"evolve_{args.evolve_split}_{args.evolve_count}_seed{args.seed}.json")
    build(data_root, args.eval_split, args.eval_count, args.seed,
          out / f"eval_{args.eval_split}_{args.eval_count}_seed{args.seed}.json")


if __name__ == "__main__":
    main()

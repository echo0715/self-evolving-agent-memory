#!/usr/bin/env python
"""Build frozen WebShop manifests with the goal text resolved.

    python scripts/build_webshop_manifests.py --server http://localhost:7000 \
        --out manifests --evolve-count 50 --eval-count 100 --seed 42

Mirrors `scripts/build_manifests.py` (ALFWorld) in the two properties that make
those manifests usable, and adds one that is specific to WebShop.

**Nested prefixes.** Selection is a prefix of one seeded permutation of the
split's index range, not `random.sample(k)`. So the 50-task evolve set is a
prefix of the 100-task one, and "evolve on 50 vs 100 vs 150" compares *amount of
experience* rather than two unrelated task draws.

**Resolved instructions.** Each entry carries the customer instruction text.
Retrieval runs before the agent acts, but WebShop only reveals the instruction on
`reset()`, so resolving it here is what lets every memory system retrieve on the
real instruction instead of on a goal index.

**Split by index range.** WebShop has no named splits; the convention its
ecosystem settled on partitions the shuffled goal list by position -- goals
[0, 500) are test, [500, 1500) are validation, [1500, ...) are train. Evolving
draws from train and evaluation from test, so the two sets are disjoint by
construction, the way ALFWorld's `train` / `valid_unseen` are.

A manifest is only meaningful for a server with the same (corpus, human_goals,
seed) triple, because those decide what goal index N *is*: product prices are
drawn randomly at load, and each goal's "price lower than X dollars" clause is
sampled from them. The server's /health is recorded in the manifest, and
`run_webshop.py` refuses to run against a server that does not match.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memsys.adapters.webshop import TaskSpec  # noqa: E402

#: Positional split boundaries of the shuffled goal list.
TEST_RANGE = (0, 500)
VAL_RANGE = (500, 1500)
TRAIN_START = 1500

SPLITS = {"test": TEST_RANGE, "valid": VAL_RANGE, "train": None}


def get(server: str, route: str) -> dict:
    with urlopen(server.rstrip("/") + route, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_goals(server: str, start: int, stop: int) -> list[dict]:
    """Page through /goals -- the full range is thousands of entries."""
    out: list[dict] = []
    step = 500
    for lo in range(start, stop, step):
        out.extend(get(server, f"/goals?start={lo}&count={min(step, stop - lo)}")["goals"])
    return out


def split_range(split: str, n_goals: int) -> tuple[int, int]:
    if split == "train":
        if n_goals <= TRAIN_START:
            raise ValueError(
                f"server has only {n_goals} goals; the train split starts at {TRAIN_START}. "
                "This is what a small-corpus server looks like -- check /health."
            )
        return TRAIN_START, n_goals
    lo, hi = SPLITS[split]
    if n_goals < hi:
        raise ValueError(f"server has only {n_goals} goals; {split} split needs {hi}")
    return lo, hi


def build(server: str, split: str, count: int, seed: int, out: Path, health: dict) -> list[TaskSpec]:
    n_goals = int(health["n_goals"])
    lo, hi = split_range(split, n_goals)
    indices = list(range(lo, hi))
    if count > len(indices):
        raise ValueError(f"{split} has only {len(indices)} goals, asked for {count}")
    random.Random(seed).shuffle(indices)  # one permutation; every count is a prefix
    selected = sorted(indices[:count])

    print(f"[{split}] resolving {count} of {len(indices)} goals in [{lo}, {hi})...", flush=True)
    goals = {g["index"]: g for g in fetch_goals(server, lo, hi)}
    tasks = []
    for idx in selected:
        g = goals.get(idx)
        if g is None:
            raise RuntimeError(f"server did not return goal {idx}")
        instruction = str(g.get("instruction_text") or "").strip()
        if not instruction:
            raise RuntimeError(f"goal {idx} has no instruction text")
        tasks.append(
            TaskSpec(
                task_id=f"{split}:{idx}",
                split=split,
                goal_index=idx,
                category=str(g.get("category") or ""),
                instruction=instruction,
            )
        )
    write_manifest(out, split, seed, len(indices), tasks, health)
    return tasks


def write_manifest(
    path: Path, split: str, seed: int, available: int, tasks: list[TaskSpec], health: dict
) -> None:
    ids = [t.task_id for t in tasks]
    payload = {
        "schema_version": 2,
        "benchmark": "WebShop",
        "split": split,
        "seed": seed,
        "selection": "prefix_of_seeded_permutation_of_the_split_index_range",
        "available_count": available,
        "selected_count": len(tasks),
        "task_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        # The identity of the server these indices refer to. Without it a
        # manifest is a list of integers with no meaning.
        "server": {k: health.get(k) for k in
                   ("scale", "corpus", "index", "n_products", "n_goals", "human_goals", "seed")},
        "tasks": [t.to_dict() for t in tasks],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    cats: dict[str, int] = {}
    for t in tasks:
        cats[t.category] = cats.get(t.category, 0) + 1
    print(f"wrote {len(tasks)} tasks -> {path}")
    print(f"  sha256={payload['task_ids_sha256'][:16]}  categories={dict(sorted(cats.items()))}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default="http://localhost:7000")
    ap.add_argument("--out", default="manifests")
    ap.add_argument("--evolve-split", default="train")
    ap.add_argument("--evolve-count", type=int, default=50)
    ap.add_argument("--eval-split", default="test")
    ap.add_argument("--eval-count", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    health = get(args.server, "/health")
    print(f"[server] {json.dumps(health)}")
    if health.get("scale") != "full":
        print("[server] WARNING: not the full corpus; goal indices will not match a full-corpus run",
              file=sys.stderr)

    out = Path(args.out)
    evolve = build(args.server, args.evolve_split, args.evolve_count, args.seed,
                   out / f"webshop_evolve_{args.evolve_split}_{args.evolve_count}_seed{args.seed}.json",
                   health)
    ev = build(args.server, args.eval_split, args.eval_count, args.seed,
               out / f"webshop_eval_{args.eval_split}_{args.eval_count}_seed{args.seed}.json",
               health)

    overlap = {t.goal_index for t in evolve} & {t.goal_index for t in ev}
    if overlap:
        raise SystemExit(f"evolve and eval sets overlap on {sorted(overlap)[:10]}")
    print(f"OK: {len(evolve)} evolve / {len(ev)} eval, disjoint")


if __name__ == "__main__":
    main()

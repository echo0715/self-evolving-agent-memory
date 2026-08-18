#!/usr/bin/env python
"""Build frozen Mind2Web manifests and the compact per-step cache they point at.

    python scripts/build_mind2web_manifests.py \
        --data-root /gpfs/radev/scratch/cohan/jw3278/mind2web_data \
        --out manifests --cache-root /gpfs/radev/scratch/cohan/jw3278/mind2web_cache \
        --evolve-split train --evolve-count 50 \
        --eval-split test_task --eval-count 100 --seed 42

Three properties matter, and none of them is free:

**Nested prefixes, at annotation level.** Selection is a prefix of one seeded
permutation of the split's sorted `annotation_id`s, exactly as on ALFWorld -- so
"evolve on 50 / 100 / 150" compares *amount of experience*, not three unrelated
draws. The permutation is over annotations, never over steps: a step is only
meaningful after the steps before it, and the evolving loop is order-dependent.

**One manifest entry per step.** The memsys episode unit on this benchmark is a
Mind2Web *step* (see `memsys/adapters/mind2web.py` for why annotation-level
success is unusable here), so the 50 selected annotations expand into ~380
entries, emitted in annotation order and then step order.

**A cache, because the released shards cannot be read at run time.** Each split
is 0.6 GB JSON shards carrying `raw_html` and hundreds of candidates per step.
This writes one small file per selected annotation holding only what the prompt
needs -- `cleaned_html`, the operation, the annotator's action strings, and the
candidate pools already merged with the released ranker's ranks and truncated to
`--cache-top-k`. Without that, every evaluation worker would re-parse gigabytes.

`scores_all_data.pkl` (the released DeBERTa candidate-generation output) is
required: `rank` is what defines the top-k pool, and a pool built any other way
would not be this benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SPLIT_FILES = {
    "train": "data/train/train_*.json",
    "test_task": "test/test_task/*.json",
    "test_website": "test/test_website/*.json",
    "test_domain": "test/test_domain/*.json",
}


def shard_paths(data_root: Path, split: str) -> list[Path]:
    pattern = SPLIT_FILES[split]
    paths = sorted(data_root.glob(pattern), key=lambda p: (len(p.name), p.name))
    if not paths:
        raise FileNotFoundError(
            f"no shards for split {split!r} below {data_root} (expected {pattern}). "
            "The test splits ship as a password-protected test.zip: unzip it with "
            "the password published in the upstream README."
        )
    return paths


def load_scores(data_root: Path) -> tuple[dict, dict]:
    p = data_root / "scores_all_data.pkl"
    if not p.is_file():
        raise FileNotFoundError(f"missing {p}: download it from osunlp/Mind2Web")
    blob = pickle.loads(p.read_bytes())
    return blob["scores"], blob["ranks"]


def index_split(data_root: Path, split: str) -> dict[str, Path]:
    """annotation_id -> shard, without keeping any page HTML in memory."""
    index: dict[str, Path] = {}
    for path in shard_paths(data_root, split):
        records = json.loads(path.read_text(encoding="utf-8"))
        for rec in records:
            index[rec["annotation_id"]] = path
        print(f"  indexed {len(records):4d} annotations from {path.name}", flush=True)
        del records
    return index


def build(data_root: Path, split: str, count: int, seed: int, out_manifest: Path,
          cache_root: Path, cache_top_k: int, skip: int = 0) -> None:
    scores, ranks = load_scores(data_root)
    print(f"[{split}] indexing shards ...", flush=True)
    index = index_split(data_root, split)
    order = sorted(index)                      # deterministic before shuffling
    random.Random(seed).shuffle(order)         # one permutation; every count is a prefix
    selected = order[skip:count]
    if count > len(order):
        raise ValueError(f"{split} has only {len(order)} annotations, asked for {count}")
    wanted = set(selected)
    print(f"[{split}] extracting {len(selected)} of {len(order)} annotations"
          f"{f' (positions {skip}..{count})' if skip else ''} ...", flush=True)

    cache_dir = cache_root / split
    cache_dir.mkdir(parents=True, exist_ok=True)
    payloads: dict[str, dict] = {}
    for path in sorted({index[a] for a in selected}, key=lambda p: p.name):
        records = json.loads(path.read_text(encoding="utf-8"))
        for rec in records:
            if rec["annotation_id"] not in wanted:
                continue
            steps = []
            for i, action in enumerate(rec["actions"]):
                sample_id = f"{rec['annotation_id']}_{action['action_uid']}"
                s_scores = scores.get(sample_id, {})
                s_ranks = ranks.get(sample_id, {})

                def keep(cands):
                    out = []
                    for c in cands:
                        nid = c["backend_node_id"]
                        rank = s_ranks.get(nid)
                        if rank is None or rank >= cache_top_k:
                            continue
                        out.append({"backend_node_id": nid, "tag": c.get("tag", ""),
                                    "rank": int(rank), "score": float(s_scores.get(nid, 0.0))})
                    return sorted(out, key=lambda c: c["rank"])

                steps.append({
                    "action_uid": action["action_uid"],
                    "cleaned_html": action["cleaned_html"],
                    "operation": action["operation"],
                    "pos_candidates": keep(action["pos_candidates"]),
                    "neg_candidates": keep(action["neg_candidates"]),
                    # Teacher forcing: the prompt's history is the annotator's
                    # actions, never the agent's own predictions.
                    "previous_actions": rec["action_reprs"][:i],
                    "action_repr": rec["action_reprs"][i],
                    "confirmed_task": rec["confirmed_task"],
                })
            (cache_dir / f"{rec['annotation_id']}.json").write_text(
                json.dumps({"annotation_id": rec["annotation_id"], "steps": steps},
                           ensure_ascii=False), encoding="utf-8")
            payloads[rec["annotation_id"]] = {
                "website": rec["website"], "domain": rec["domain"],
                "subdomain": rec["subdomain"], "confirmed_task": rec["confirmed_task"],
                "n_steps": len(steps),
                "n_steps_with_pos": sum(1 for s in steps if s["pos_candidates"]),
            }
        print(f"  extracted from {path.name} ({len(payloads)}/{len(selected)} done)", flush=True)
        del records

    tasks = []
    for annotation_id in selected:                 # permutation order, not shard order
        meta = payloads[annotation_id]
        for step_index in range(meta["n_steps"]):
            tasks.append({
                "task_id": f"{split}:{annotation_id}:{step_index}",
                "split": split, "annotation_id": annotation_id,
                "step_index": step_index, "n_steps": meta["n_steps"],
                "website": meta["website"], "domain": meta["domain"],
                "subdomain": meta["subdomain"], "instruction": meta["confirmed_task"],
            })
    payload = {
        "schema_version": 2,
        "benchmark": "Mind2Web",
        "data_version": "osunlp/Mind2Web@17ece8e",
        "split": split,
        "seed": seed,
        "selection": ("prefix_of_seeded_permutation_of_sorted_annotation_ids" if not skip
                      else f"positions_[{skip},{skip + len(selected)})_of_seeded_permutation"),
        "skip": skip,
        "unit": "step",
        "available_annotations": len(order),
        "selected_annotations": len(selected),
        "selected_count": len(tasks),
        "cache_root": str(cache_root),
        "cache_top_k": cache_top_k,
        "annotation_ids_sha256": hashlib.sha256("\n".join(selected).encode()).hexdigest(),
        "tasks": tasks,
    }
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
    domains: dict[str, int] = {}
    for t in tasks:
        domains[t["domain"]] = domains.get(t["domain"], 0) + 1
    no_pos = sum(meta["n_steps"] - meta["n_steps_with_pos"] for meta in payloads.values())
    print(f"wrote {len(selected)} annotations / {len(tasks)} steps -> {out_manifest}")
    print(f"  sha256={payload['annotation_ids_sha256'][:16]}  domains={domains}")
    # Steps whose ground-truth element the ranker never surfaced are scored 0
    # without the model being called. Print the number now rather than letting it
    # surface as an unexplained ceiling on step SR later.
    print(f"  steps with no positive candidate in top-{cache_top_k}: {no_pos} "
          f"({no_pos / max(1, len(tasks)):.1%}) -- these are automatic zeros")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", default="manifests")
    ap.add_argument("--cache-root", required=True)
    ap.add_argument("--evolve-split", default="train", choices=list(SPLIT_FILES))
    ap.add_argument("--evolve-count", type=int, default=50)
    ap.add_argument("--evolve-skip", type=int, default=0)
    ap.add_argument("--eval-split", default="test_task", choices=list(SPLIT_FILES))
    ap.add_argument("--eval-count", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache-top-k", type=int, default=50,
                    help="candidates kept per step; must be >= any --top-k used at run time")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--evolve-only", action="store_true")
    args = ap.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    cache_root = Path(args.cache_root).expanduser().resolve()
    out = Path(args.out)
    if not args.eval_only:
        suffix = (f"_{args.evolve_count}" if not args.evolve_skip
                  else f"_{args.evolve_skip}to{args.evolve_count}")
        build(data_root, args.evolve_split, args.evolve_count, args.seed,
              out / f"mind2web_evolve_{args.evolve_split}{suffix}_seed{args.seed}.json",
              cache_root, args.cache_top_k, skip=args.evolve_skip)
    if not args.evolve_only:
        build(data_root, args.eval_split, args.eval_count, args.seed,
              out / f"mind2web_eval_{args.eval_split}_{args.eval_count}_seed{args.seed}.json",
              cache_root, args.cache_top_k)


if __name__ == "__main__":
    main()

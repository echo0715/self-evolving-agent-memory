#!/usr/bin/env python
"""Assemble the Mind2Web evolving-budget curve: each arm at 50 / 100 / 150 / 200.

    python scripts/summarize_mind2web_curve.py \
        --root /gpfs/radev/scratch/cohan/jw3278/memsys_results/mind2web \
        --host r818u33n04 --out RESULTS_MIND2WEB_CURVE.md

`scripts/summarize.py` compares arms *within* one evolving budget. This compares
one arm *across* budgets, which needs two things the generic summarizer has no
reason to do:

**Two baselines, not one.** Two no-memory runs of this benchmark on the same GPU,
same server, same seed and temperature 0 disagree on ~1.4% of steps -- vLLM's
reductions depend on batch composition, which temperature does not pin down.
A single baseline therefore fixes the reference somewhere inside a ~1-point band
by luck of the draw. Every delta here is reported as a range across the available
no-memory replicas, and the p-value quoted is the *worst* (largest) of them, so
an arm only looks significant if it beats both draws.

**The floor as a row.** The replica-vs-replica comparison is printed in the same
table as the arms, in the same units. An arm whose delta does not clear that row
has not been shown to do anything.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

BUDGETS = (50, 100, 150, 200)
ARMS = ("raw", "reflection", "rule", "skill")


def load(path: Path) -> dict[str, int] | None:
    if not path.is_file():
        return None
    return {r["task_id"]: int(r["step_acc"]) for r in map(json.loads, path.open())}


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact binomial test on the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def compare(arm: dict[str, int], refs: list[dict[str, int]]) -> tuple[str, str, str]:
    """Delta range, b/c against the first ref, and the least favourable p."""
    deltas, ps, bc = [], [], ""
    for i, ref in enumerate(refs):
        shared = [t for t in arm if t in ref]
        if not shared:
            continue
        b = sum(1 for t in shared if ref[t] and not arm[t])
        c = sum(1 for t in shared if not ref[t] and arm[t])
        deltas.append(100 * (sum(arm[t] for t in shared) - sum(ref[t] for t in shared))
                      / len(shared))
        ps.append(mcnemar_exact(b, c))
        if i == 0:
            bc = f"{b}/{c}"
    if not deltas:
        return "-", "-", "-"
    span = (f"{deltas[0]:+.1f}" if len(deltas) == 1
            else f"{min(deltas):+.1f}..{max(deltas):+.1f}")
    return span, bc, f"{max(ps):.3f}"


def arm_dir(root: Path, arm: str, budget: int, host: str) -> Path:
    """Where an (arm, budget) cell's evaluation lives.

    The 50 point is the 2026-08-14 sweep's memory re-scored on the current
    serving stack -- same store.jsonl, no further evolving -- so that it is
    comparable to the legs after it rather than to a baseline collected on
    another day.
    """
    if budget == 50:
        return root / f"minimal_e50_{host}" / f"{arm}_minimal"
    return root / f"minimal_e{budget}" / f"{arm}_minimal"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--host", required=True, help="node tag used for the same-stack runs")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.root)
    refs = [d for d in (load(root / f"_baseline_none_{args.host}_r{i}/eval.jsonl")
                        for i in (1, 2)) if d]
    if not refs:
        raise SystemExit(f"no no-memory replicas under {root} for host {args.host}")

    lines = [
        "# Mind2Web -- step success rate vs evolving budget",
        "",
        f"Eval: the frozen 100 `test_task` annotations (838 steps), top-50 pool, "
        f"temperature 0. All runs on `{args.host}`.",
        "",
        "Deltas are ranges across "
        f"{len(refs)} independent no-memory replica{'s' if len(refs) > 1 else ''}; "
        "the p-value is the least favourable of them.",
        "",
        "| arm | budget | evolve steps | step SR | Δ vs none | b/c | McNemar p | store | mem tok |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    for i, ref in enumerate(refs, 1):
        sr = sum(ref.values()) / len(ref)
        span, bc, p = compare(ref, [refs[0]] if i > 1 else refs[1:2] or [ref])
        lines.append(f"| **none (replica {i})** | — | — | {100 * sr:.1f}% | "
                     f"{span if len(refs) > 1 else '—'} | {bc or '—'} | "
                     f"{p if len(refs) > 1 else '—'} | — | 0 |")

    for arm in ARMS:
        for budget in BUDGETS:
            d = arm_dir(root, arm, budget, args.host)
            outcomes = load(d / "eval.jsonl")
            if outcomes is None:
                lines.append(f"| {arm} | {budget} | — | *not run* | — | — | — | — | — |")
                continue
            summary = json.loads((d / "summary.json").read_text())
            if budget == 50:
                # The 50 cell is an evaluation-only re-scoring, so its own
                # summary reports 0 evolving steps. The experience behind that
                # store is in the run that built it.
                origin = root / "minimal" / f"{arm}_minimal" / "summary.json"
                if origin.is_file():
                    summary["evolve_total"] = json.loads(origin.read_text())["evolve_total"]
            span, bc, p = compare(outcomes, refs)
            store = summary.get("store") or {}
            sr = sum(outcomes.values()) / len(outcomes)
            lines.append(
                f"| {arm} | {budget} | {summary.get('evolve_total', '—')} | {100 * sr:.1f}% | "
                f"{span} | {bc} | {p} | {store.get('n_live', '—')} | "
                f"{summary.get('mean_injected_tokens', '—')} |")

    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()

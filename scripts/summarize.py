#!/usr/bin/env python
"""Build the Memory Content table from a sweep directory, with paired tests.

    python scripts/summarize.py --root /path/to/memsys_results/full

Every arm evaluates the *same* ordered task list, so arm-vs-baseline is a paired
comparison and McNemar's exact test applies. This is not a formality: SAGE
measured a ±3-6 point run-to-run noise floor on 50 ALFWorld tasks, and reported
a -10 point delta there that was p=0.383 -- i.e. nothing. A raw delta column on
its own invites exactly that misreading, so the table carries b/c and p.
"""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value.

    b = baseline succeeded, arm failed;  c = baseline failed, arm succeeded.
    Concordant pairs carry no information and are excluded by construction.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2 * tail)


def load_arm(d: Path) -> dict | None:
    summary_path, eval_path = d / "summary.json", d / "eval.jsonl"
    if not summary_path.is_file() or not eval_path.is_file():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    outcomes = {}
    for line in eval_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            outcomes[row["task_id"]] = bool(row["any_success"])
    summary["_outcomes"] = outcomes
    summary["_dir"] = d.name
    summary["_ops"] = _write_ops(d / "evolve_log.jsonl")
    return summary


def _write_ops(path: Path) -> dict[str, int]:
    """Tally the write operations, so the store size is explainable.

    Worth surfacing: an APPEND must pass schema validation and the grounding
    check, while a DELETE only needs a target id. During batch induction the
    writer proposes "delete these, append the consolidation" -- and a
    consolidated entry's evidence is synthesised across episodes, so it often
    fails grounding. The deletes land, the append does not, and the store
    shrinks. A success-rate column alone would never show that.
    """
    ops: dict[str, int] = {}
    if not path.is_file():
        return ops
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        for op in rec.get("writer_ops", []):
            ops[op.get("op", "?")] = ops.get(op.get("op", "?"), 0) + 1
        if rec.get("rejected"):
            ops["rejected"] = ops.get("rejected", 0) + len(rec["rejected"])
    return ops


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--baseline", default="none", help="arm name used as the reference")
    ap.add_argument("--out", default=None, help="also write the markdown here")
    args = ap.parse_args()

    root = Path(args.root)
    arms = [a for a in (load_arm(d) for d in sorted(root.iterdir()) if d.is_dir()) if a]
    if not arms:
        print(f"no completed arms under {root}")
        return

    base = next((a for a in arms if a["arm"] == args.baseline), None)
    benchmark = arms[0].get("benchmark", "ALFWorld")
    split = arms[0].get("eval_split") or ("valid_unseen" if benchmark == "ALFWorld" else "test")
    # WebShop, AppWorld and SpreadsheetBench grade partially, so they report a
    # score alongside the strict success rate; ALFWorld has no such column and
    # must not grow an empty one. The two can move in opposite directions -- an
    # agent that buys plausible-but-wrong items faster raises score and lowers
    # success rate, and a SpreadsheetBench agent that hardcodes the one test case
    # it can see scores 1/3 on every task it "solves" -- which is exactly why the
    # column is added rather than substituted.
    has_score = any(a.get("eval_score") is not None for a in arms)
    score_head = " score |" if has_score else ""
    score_sep = "---|" if has_score else ""
    lines = [
        f"# {benchmark} Memory Content -- {root.name}",
        "",
        f"model: `{arms[0]['model']}`  |  eval: {arms[0]['eval_n']} tasks "
        f"({split})  |  evolve: {max(a['evolve_n'] for a in arms)} tasks",
        "",
        "| arm | policy | eval success | rate |" + score_head
        + " delta | b/c | McNemar p | store | mem tok | writer calls |",
        "|---|---|---|---|" + score_sep + "---|---|---|---|---|---|",
    ]
    for a in sorted(arms, key=lambda x: (x["policy"], x["arm"])):
        delta = bc = pval = "-"
        if base is not None and a is not base:
            shared = set(a["_outcomes"]) & set(base["_outcomes"])
            b = sum(1 for t in shared if base["_outcomes"][t] and not a["_outcomes"][t])
            c = sum(1 for t in shared if not base["_outcomes"][t] and a["_outcomes"][t])
            delta = f"{100 * (a['eval_success_rate'] - base['eval_success_rate']):+.1f}"
            bc = f"{b}/{c}"
            pval = f"{mcnemar_exact(b, c):.3f}"
        store = a.get("store") or {}
        usage = a.get("writer_usage") or {}
        score_cell = ""
        if has_score:
            s = a.get("eval_score")
            score_cell = f" {100 * s:.1f} |" if s is not None else " - |"
        lines.append(
            f"| {a['arm']} | {a['policy']} | {a['eval_success']}/{a['eval_n']} | "
            f"{100 * a['eval_success_rate']:.1f}% |" + score_cell
            + f" {delta} | {bc} | {pval} | "
            f"{store.get('n_live', '-')} | {a.get('mean_injected_tokens', '-')} | "
            f"{usage.get('n_calls', '-')} |"
        )

    lines += ["", "`b/c`: b = baseline solved it and the arm did not; c = the reverse.",
              "Concordant tasks carry no signal, so only b and c enter the test.", ""]

    # Per-family breakdown: an aggregate can hide a real effect confined to one
    # ALFWorld task family (the six differ a lot in difficulty). For WebShop the
    # grouping is the product department, which is a weaker cut -- two "beauty"
    # tasks can need entirely different action sequences -- so read it as
    # exploratory rather than as ALFWorld's procedure families. SpreadsheetBench
    # is weaker still: it labels only cell-level and sheet-level, so the two
    # columns are close to a coin flip on procedure. AppWorld reports difficulty
    # here instead, its ~34 scenarios being far too fine to read.
    families = sorted({f for a in arms for f in (a.get("eval_by_family") or {})})
    if families:
        lines += ["## By task family", "",
                  "| arm | policy | " + " | ".join(families) + " |",
                  "|---|---|" + "---|" * len(families)]
        for a in sorted(arms, key=lambda x: (x["policy"], x["arm"])):
            cells = []
            for f in families:
                s = (a.get("eval_by_family") or {}).get(f)
                cells.append(f"{s['success']}/{s['n']}" if s else "-")
            lines.append(f"| {a['arm']} | {a['policy']} | " + " | ".join(cells) + " |")
        lines.append("")

    text = "\n".join(lines)
    print(text)
    target = Path(args.out) if args.out else root / "RESULTS.md"
    target.write_text(text + "\n", encoding="utf-8")
    print(f"\nwrote {target}")


if __name__ == "__main__":
    main()

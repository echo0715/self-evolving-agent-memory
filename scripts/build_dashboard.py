#!/usr/bin/env python3
"""Build RESULTS_DASHBOARD.html from every run on scratch.

Walks $MEMSYS_RESULTS_ROOT, reads each arm's summary.json and eval.jsonl,
recomputes the paired McNemar test against that sweep's own `none` draw, and
emits a single self-contained HTML page: an axis x benchmark matrix where every
cell opens the full arm table behind it.

    python scripts/build_dashboard.py [--out RESULTS_DASHBOARD.html]
"""
from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
import sys
from datetime import datetime, timezone

ROOT = os.environ.get(
    "MEMSYS_RESULTS_ROOT", "/gpfs/radev/scratch/cohan/jw3278/memsys_results"
)

BENCH_ROOT = {
    "ALFWorld": "",
    "WebShop": "webshop",
    "AppWorld": "appworld",
    "SpreadsheetBench": "spreadsheetbench",
    "Mind2Web": "mind2web",
    "ScienceWorld": "scienceworld",
}
NESTED = {v for v in BENCH_ROOT.values() if v}
BENCHES = list(BENCH_ROOT)
ARMS = ["raw", "reflection", "rule", "skill"]
SKIP = ("smoke", "aborted", "contaminated", "_hwcheck", "evalfix", "_slices", "_baseline")

# The metric each benchmark's headline rate actually is.
METRIC = {
    "ALFWorld": "task success",
    "WebShop": "strict success (reward = 1.0)",
    "AppWorld": "task goal completion",
    "SpreadsheetBench": "all 3 test cases pass",
    "Mind2Web": "step success rate",
    "ScienceWorld": "task success",
}
# Measured run-to-run churn, in points of the headline rate. `note` says how.
FLOOR = {
    "ALFWorld": (5.0, "two no-memory runs landed 58 and 57 while disagreeing on 35/100 tasks"),
    "WebShop": (4.0, "never replicated; bounded from the discordant counts (sqrt 15-18)"),
    "AppWorld": (5.0, "the baseline was run twice: 20 and 25, disagreeing on 23/100"),
    "SpreadsheetBench": (4.7, "three identical-memory replicates all scored 27 while flipping 22/100; a second such pair — skill under the gpt-5.6 writer, whose four entries are character-identical at e50 and e100 — scored 23 and 27. The empty-store draws below span 13 to 23, so the write-up's 16.3 baseline mean is itself a floor estimate"),
    "Mind2Web": (1.1, "five no-memory draws span 31.5-32.6% and disagree on ~1.4% of steps"),
    "ScienceWorld": (5.0, "the two no-memory draws are only 2 points apart (33 and 31) but disagree on 30/100 tasks, so the per-task churn is ALFWorld-sized even where the rates nearly agree"),
}

# ------------------------------------------------------------------ axis table
GROUPS = [
    ("Stream length", "How many distinct evolving tasks the memory was built from, one pass each.", [
        ("e10", "10 evolve", "10 distinct tasks"),
        ("e25", "25 evolve", "25 distinct tasks"),
        ("e50", "50 evolve", "50 distinct tasks"),
        ("e75", "75 evolve", "75 distinct tasks"),
        ("e100", "100 evolve", "resumes the 50-task store"),
        ("e150", "150 evolve", "resumes the 100-task store"),
        ("e200", "200 evolve", "resumes the 150-task store"),
    ]),
    ("Repetition at fixed budget", "150 evolving episodes every time; only the number of distinct tasks changes.", [
        ("r30x5", "30 x 5", "30 tasks, five passes"),
        ("r50x2", "50 x 2", "50 tasks, two passes"),
        ("r50x3", "50 x 3", "50 tasks, three passes"),
        ("r75x2", "75 x 2", "75 tasks, two passes"),
    ]),
    ("Outcome budget", "Budget the stream by outcome instead of by attempt, and show the writer only one sign of evidence.", [
        ("succ100", "success-budget 100", "100 solved episodes, failures discarded"),
        ("fail100", "failure-budget 100", "100 failed episodes, successes discarded"),
    ]),
    ("Writer model: gpt-5.6-terra", "Actor stays Qwen3.5-9B; only the model that writes memory changes. Responses API via the Perplexity gateway, writer budget 1024 tokens.", [
        ("gpt_e25", "gpt-5.6, 25", "minimal policy only"),
        ("gpt_e50", "gpt-5.6, 50", "minimal policy only"),
        ("gpt_e100", "gpt-5.6, 100", "minimal policy only"),
        ("gpt_e150", "gpt-5.6, 150", "minimal policy only"),
    ]),
    ("Writer model: gemini-3.7-flash", "Same swap, same actor. Writer budget 4096 tokens rather than 1024: measured on the real prompt, this model spends up to 1214 completion tokens on a skill entry and the JSON is cut off at the old cap.", [
        ("gem_e50", "gemini-3.7, 50", "minimal policy only"),
        ("gem_e100", "gemini-3.7, 100", "minimal policy only"),
    ]),
    ("Writer model: kimi-k3", "Same swap, same actor, same 4096-token budget (it spends 1048-2363). Reads against the 400-token content cap more than any other writer -- most cells here reject more entries than they accept.", [
        ("kimi_e50", "kimi-k3, 50", "minimal policy only"),
        ("kimi_e100", "kimi-k3, 100", "minimal policy only"),
    ]),
]
AXES = {k: (lbl, sub) for _, _, rows in GROUPS for k, lbl, sub in rows}

CONDMAP = {
    "": "e50", "e10": "e10", "e25": "e25", "e30": "e30", "e50": "e50", "e75": "e75",
    "e100": "e100", "e150": "e150", "e200": "e200",
    "x2": "r50x2", "x3": "r50x3", "e75_x2": "r75x2",
    "e30_x2": "r30x2", "e30_x3": "r30x3", "e30_x4": "r30x4", "e30_x5": "r30x5",
    "succ100": "succ100", "ok100": "succ100", "fail100": "fail100",
}


# tag suffix -> (writer name for the tables, axis prefix for the matrix). A run
# that differs only in the writer lives in its own directory under the same mode
# name, so the suffix is the only thing separating it from the Qwen stream-length
# cells -- an unrecognised suffix would silently land on top of them.
WRITERS = {
    "_gpt56terra": ("gpt-5.6-terra", "gpt"),
    "_gemini37f": ("gemini-3.7-flash", "gem"),
    "_kimik3": ("kimi-k3", "kimi"),
}


def axis_of(cond):
    writer, prefix, c = "qwen", "", re.sub(r"_cont$", "", cond)
    for suffix, (name, pre) in WRITERS.items():
        if suffix in c:
            writer, prefix, c = name, pre, c.replace(suffix, "")
            break
    for pol in ("minimal", "full"):
        if c == pol or c.startswith(pol + "_"):
            rest = c[len(pol):].lstrip("_")
            axis = CONDMAP.get(rest)
            if axis and prefix:
                # Only the budgets a writer chain actually ran get a cell; an
                # axis with no column (e.g. a repetition leg) stays unmapped
                # rather than colliding with the Qwen one.
                axis = f"{prefix}_{axis}" if f"{prefix}_{axis}" in AXES else None
            return axis, pol, writer
    return None, None, None


# ------------------------------------------------------------------ statistics
def mcnemar_p(b, c):
    """Exact two-sided binomial test on the discordant pairs."""
    n, k = b + c, min(b, c)
    if n == 0:
        return 1.0
    pk = math.comb(n, k) * 0.5 ** n
    return min(1.0, sum(math.comb(n, i) * 0.5 ** n
                        for i in range(n + 1)
                        if math.comb(n, i) * 0.5 ** n <= pk * (1 + 1e-9)))


_outcomes: dict[str, dict] = {}


def outcomes(path):
    if path not in _outcomes:
        got, f = {}, os.path.join(path, "eval.jsonl")
        if os.path.isfile(f):
            for line in open(f):
                line = line.strip()
                if line:
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    got[d["task_id"]] = bool(d.get("any_success"))
        _outcomes[path] = got
    return _outcomes[path]


def fail_reasons(path):
    """Per-run count of the eval-sandbox errors worth flagging."""
    f = os.path.join(path, "eval.jsonl")
    if not os.path.isfile(f):
        return 0, 0
    bad = tot = 0
    for line in open(f):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        tot += 1
        for ro in d.get("rollouts", []):
            fr = str(ro.get("fail_reason") or "")
            if "partially initialized" in fr or "circular import" in fr:
                bad += 1
                break
    return bad, tot


# --------------------------------------------------------------------- collect
def collect():
    runs, replicates = [], []
    for bench, sub in BENCH_ROOT.items():
        root = os.path.join(ROOT, sub) if sub else ROOT
        if not os.path.isdir(root):
            continue
        for cond in sorted(os.listdir(root)):
            cpath = os.path.join(root, cond)
            if not os.path.isdir(cpath) or cond in NESTED:
                continue
            if cond.startswith("_baseline_none"):
                s = os.path.join(cpath, "summary.json")
                if os.path.isfile(s):
                    d = json.load(open(s))
                    if (d.get("eval_n") or 0) >= 50:
                        replicates.append({
                            "bench": bench, "label": cond, "path": cpath,
                            "n": d["eval_n"], "success": d["eval_success"],
                            "rate": d["eval_success_rate"],
                        })
                continue
            if any(s in cond for s in SKIP):
                continue
            axis, policy, writer = axis_of(cond)
            for armdir in sorted(os.listdir(cpath)):
                spath = os.path.join(cpath, armdir, "summary.json")
                if not os.path.isfile(spath):
                    continue
                try:
                    d = json.load(open(spath))
                except json.JSONDecodeError:
                    continue
                wu, st = d.get("writer_usage") or {}, d.get("store") or {}
                bad, tot = fail_reasons(os.path.join(cpath, armdir))
                runs.append({
                    "bench": bench, "cond": cond, "axis": axis, "writer": writer,
                    "policy": d.get("policy") or policy,
                    "arm": d.get("arm") or armdir.rsplit("_", 1)[0],
                    "path": os.path.join(cpath, armdir),
                    "n": d.get("eval_n"), "success": d.get("eval_success"),
                    "rate": d.get("eval_success_rate"),
                    "score": d.get("eval_score", d.get("eval_mean_score")),
                    "evolve_n": d.get("evolve_n"),
                    "evolve_total": d.get("evolve_total"),
                    "evolve_sr": d.get("evolve_success_rate"),
                    "budget": d.get("evolve_budget"),
                    "store": st.get("n_live"), "rows": st.get("n_total_rows"),
                    "writer_calls": wu.get("n_calls"),
                    "writer_ptok": wu.get("prompt_tokens"),
                    "writer_ctok": wu.get("completion_tokens"),
                    "inj": d.get("mean_injected_tokens"),
                    "wall_h": round(d["wall_seconds"] / 3600, 2) if d.get("wall_seconds") else None,
                    "stub": (d.get("eval_n") or 0) < 50,
                    "sandbox_errors": bad,
                })
    return runs, replicates


def annotate(runs):
    """Attach delta / b / c / p to every arm, paired against its own sweep's baseline."""
    default, by_cond = {}, {}
    for r in runs:
        if r["arm"] == "none" and not r["stub"]:
            by_cond[(r["bench"], r["cond"])] = r
            default.setdefault(r["bench"], r)
    for bench in BENCHES:  # prefer the canonical 50-task sweep's draw
        for r in runs:
            if r["bench"] == bench and r["arm"] == "none" and r["cond"] == "minimal":
                default[bench] = r
    for r in runs:
        base = by_cond.get((r["bench"], r["cond"])) or default.get(r["bench"])
        r["base_rate"] = base["rate"] if base else None
        r["base_success"] = base["success"] if base else None
        if not base or r["arm"] == "none" or r["stub"]:
            r.update(delta=None, b=None, c=None, p=None)
            continue
        bo, ao = outcomes(base["path"]), outcomes(r["path"])
        common = [k for k in bo if k in ao]
        b = sum(1 for k in common if bo[k] and not ao[k])
        c = sum(1 for k in common if ao[k] and not bo[k])
        r.update(b=b, c=c, p=mcnemar_p(b, c),
                 delta=round((r["rate"] - base["rate"]) * 100, 1))
    return default


def noise_floor(runs, replicates, default):
    """Every distinct no-memory draw per benchmark, and every pairwise gap."""
    out, per_bench = [], {}
    for bench in BENCHES:
        draws, seen = [], set()
        for r in runs:
            if r["bench"] != bench or r["arm"] != "none" or r["stub"]:
                continue
            sig = hash(tuple(sorted(outcomes(r["path"]).items())))
            if sig in seen:
                continue
            seen.add(sig)
            draws.append({"label": f"sweep {r['cond']}", "path": r["path"],
                          "success": r["success"], "n": r["n"]})
        for rep in replicates:
            if rep["bench"] != bench:
                continue
            sig = hash(tuple(sorted(outcomes(rep["path"]).items())))
            if sig in seen:
                continue
            seen.add(sig)
            draws.append({"label": rep["label"], "path": rep["path"],
                          "success": rep["success"], "n": rep["n"]})
        # arms whose store came out empty are byte-identical to no-memory
        for r in runs:
            if r["bench"] == bench and not r["stub"] and (r["store"] == 0 or (r["inj"] == 0 and r["arm"] != "none")):
                sig = hash(tuple(sorted(outcomes(r["path"]).items())))
                if sig in seen:
                    continue
                seen.add(sig)
                draws.append({"label": f"{r['arm']}/{r['policy']} @ {r['cond']}",
                              "path": r["path"], "success": r["success"], "n": r["n"],
                              "empty": True})
        per_bench[bench] = [{k: v for k, v in d.items() if k != "path"} for d in draws]
        for i in range(len(draws)):
            for j in range(i + 1, len(draws)):
                a, bb = outcomes(draws[i]["path"]), outcomes(draws[j]["path"])
                common = [k for k in a if k in bb]
                if not common:
                    continue
                out.append({
                    "bench": bench, "a": draws[i]["label"], "b": draws[j]["label"],
                    "a_succ": draws[i]["success"], "b_succ": draws[j]["success"],
                    "n": len(common),
                    "disagree": sum(1 for k in common if a[k] != bb[k]),
                })
    return out, per_bench


# ---------------------------------------------------------------------- prose
# One reading per cell. Sourced from the RESULTS_*.md write-ups where one
# exists; otherwise stated from the numbers and marked as not yet written up.
NOTES = {
 ("ALFWorld", "e10"): "Not in RESULTS_ALFWORLD.md. Ten episodes are enough for raw/minimal to reach 81.0% (+23.0, p = 0.0002) off a seven-entry store \u2014 above its own 25-, 50- and 100-task selves. rule is already 5 points below baseline here and never durably recovers.",
 ("ALFWorld", "e25"): "Not in RESULTS_ALFWORLD.md. skill/minimal peaks here at 88.0% from a six-entry store — its highest score anywhere in the study, and 9 points above its own 50-task self.",
 ("ALFWorld", "e50"): "The headline run. raw and skill/minimal both clear the floor decisively; rule/full is the study's one case of significant harm (-12.0, p = 0.043). Every LLM-writer arm is worse under full than minimal — a claim §6 later narrows to \"at 50 episodes\".",
 ("ALFWorld", "e75"): "First leg of the 75 x 2 chain, from an empty store. Only raw is significant against baseline under both policies; skill is significant here and loses it on the second pass.",
 ("ALFWorld", "e100"): "Doubling the stream overturns the write-policy claim: full's deficit halves for reflection and skill and flips sign for rule (+15.0, p = 0.017, the largest single 50 -> 100 move). raw is the only arm whose rate does not move.",
 ("ALFWorld", "e150"): "raw/minimal reaches 83.0%, the best Qwen-written result in the chain. rule's apparent monotone recovery stops here — from e100 onward it is indistinguishable from having no memory.",
 ("ALFWorld", "e200"): "reflection and rule only, the two arms that still looked like they were moving. Holm over the four tests: nothing survives. reflection/full misses by 0.0014.",
 ("ALFWorld", "r30x5"): "skill is missing — a deleted numpy in envs/memsys-alfworld killed its eval workers on 2026-08-17; reflection and rule were recovered by re-running eval from their stores. raw at +20.0 is the only arm clear of the floor.",
 ("ALFWorld", "r50x2"): "Second pass over the identical 50-task manifest. raw's store barely grows: every write attempt that repeats a solved task is rejected as an exact duplicate.",
 ("ALFWorld", "r50x3"): "No arm's epoch 1 -> 3 change is significant and none improves monotonically. By epoch 3 every one of raw's write attempts is a duplicate rejection (40/40); the writer arms register zero duplicate rejections in any epoch and grow linearly.",
 ("ALFWorld", "r75x2"): "The third arrangement of a 150-episode budget. Against 150-distinct and 50 x 3, sixteen paired tests, none significant, smallest p = 0.150 — how the episodes are arranged does not measurably matter.",
 ("ALFWorld", "succ100"): "The study's one large positive result for an LLM-written memory: rule/minimal at 84.0%, +26.0, p = 1e-5, statistically indistinguishable from raw. The same intervention drives skill/minimal below baseline. Two variables move at once (amount of successful experience, and the filter itself).",
 ("ALFWorld", "fail100"): "No arm beats baseline. Three separate mechanisms prevent failure-only memory from existing: skill cannot write without a successful rollout (zero writer calls in 235 tasks), full's utility floor deletes everything down to 2 entries, and where a store does survive it lands 26-38 points below its succ100 twin.",
 ("ALFWorld", "gpt_e50"): "Swapping only the writer model lifts rule from 53.0% to 77.0% (+24.0) and reflection to 81.0% (+16.0); skill does not move. The better writer also writes less and never proposes a duplicate. $1.14 for the sweep.",
 ("ALFWorld", "gpt_e100"): "Not in RESULTS_ALFWORLD.md. The gpt-writer chain continued to 100: skill reaches 83.0% and every arm stays clear of the floor, which the Qwen-writer chain never managed at this budget.",
 ("ALFWorld", "gpt_e150"): "Not in RESULTS_ALFWORLD.md. All four arms clear the floor — skill 85.0%, raw 83.0%, reflection 78.0%. This is the only cell in the study where every content type beats baseline at once.",
 ("SpreadsheetBench", "e10"): "Not written up. Ten episodes buy most of what fifty do: reflection 22/100 (+9.0, p = 0.049) against 27/100 at e50, and rule +8.0 from a six-entry store. Every arm is still within 2 sigma of the 4.7-point floor, so this cell separates nothing.",
 ("Mind2Web", "e10"): "Not in RESULTS_MIND2WEB.md. The benchmark\u0027s negative sign is present from the first 10 annotations (79 steps): no arm is above the 31.5% baseline, reflection is 3.3 points under it (p = 0.013), and rule ties it exactly.",
 ("ScienceWorld", "e10"): "Not written up. All four arms land below the 33/100 baseline, the same shape as 25, 50 and 100 \u2014 no leg of this benchmark has memory helping. skill ends with an empty store and 0 injected tokens, so its 31/100 is an extra no-memory draw, not a result.",
 ("WebShop", "e25"): "Not in RESULTS_WEBSHOP.md. Everything sits within 5 points of the 29.0% baseline in both directions.",
 ("WebShop", "e50"): "No arm reaches p < 0.05. The near-null is two real effects cancelling: every minimal arm is a better shopper than the baseline and every one finishes fewer episodes, because reading ~1000 tokens of memory costs steps against a 15-step horizon.",
 ("WebShop", "e100"): "Five of eight arms declined, and the study's only significant result is raw/full collapsing 16 points to below baseline (p = 0.002). Phase 2 was also a harder draw — all eight arms succeeded less often during evolving.",
 ("WebShop", "e150"): "raw/full went back up, 37 -> 21 -> 31, which retracts the collapse. At 150 no arm is more than 4 points from baseline and every arm's graded score is below it.",
 ("WebShop", "r50x2"): "raw/full begins a monotone decline under repetition that the diversity axis never showed.",
 ("WebShop", "r50x3"): "Mean rate 27.25 against 28.25 for 150 distinct tasks — not one arm differs by more than its own paired SD. reflection reaches 78 entries from 50 tasks seen three times, more than from 150 distinct tasks seen once.",
 ("WebShop", "succ100"): "The most expensive condition and the worst: 335-417 tasks per arm to bank 100 successes, and the mean lands below the no-memory baseline. skill/minimal writes 38 entries here against 4 at 50 attempts and posts its lowest score anywhere (24/100) — its small store was never selectivity, it was starvation.",
 ("WebShop", "gpt_e50"): "Not in RESULTS_WEBSHOP.md. Unlike ALFWorld, a frontier writer buys nothing here — all three arms sit within 3 points of baseline. raw is absent by construction (no writer).",
 ("AppWorld", "e25"): "Not in RESULTS_APPWORLD.md. raw/minimal at +10.0 is the only arm outside the +/-5 floor, and only against the first of two baseline draws.",
 ("AppWorld", "e50"): "Read the noise floor first: the baseline was run twice and landed 20.0 and 25.0. rule/minimal and skill/minimal look like +5.0 against rep1 and are exactly +0.0 against rep2. raw/full (+11.0, p = 0.052) is the only candidate effect.",
 ("AppWorld", "e100"): "Two arms now clear +/-5 against both baselines — raw/full (+16.0 / +11.0) and rule/minimal (+14.0 / +9.0) — but no arm's own 50 -> 100 change is significant, so this is not a measurement of \"more experience helps\".",
 ("SpreadsheetBench", "e25"): "Not in RESULTS_SPREADSHEETBENCH.md. skill/minimal at 29/100 is the top score of the whole benchmark, from a three-entry store.",
 ("SpreadsheetBench", "e50"): "Memory helps here by roughly 12-14 points at the top end, but the ordering among content types does not survive: the top four span three of the four types, separated by 2 points against a 4.7-point floor.",
 ("SpreadsheetBench", "e75"): "Not in RESULTS_SPREADSHEETBENCH.md. Note skill/full ends with an empty store and 0 injected tokens, so its 23/100 is a no-memory draw, not a result.",
 ("SpreadsheetBench", "e100"): "Fifty more episodes bought nothing. skill/full stopped writing after episode 50 and its three entries are character-identical across all three legs — which is what turned this chain into a true replicate and pinned the noise floor at 4.7 points.",
 ("SpreadsheetBench", "e150"): "raw/full climbs 18 -> 21 -> 27, the only arm monotone across three legs, from the least structured content type. reflection/full and rule/full both died mid-run when the batch-induction prompt outgrew a 32k window; recovered on 65k.",
 ("SpreadsheetBench", "e200"): "Contaminated, and never written up. 10-34% of eval tasks in six of eight cells failed with a partially-initialised openpyxl (a circular import inside the eval sandbox), which is why three cells read 0/100. Only the two skill cells are clean. Re-run before quoting anything here.",
 ("SpreadsheetBench", "r30x5"): "Not written up. skill is missing (shared env with the ALFWorld numpy incident). reflection/minimal at 29/100 ties the benchmark's best score.",
 ("SpreadsheetBench", "r50x2"): "Not written up. Everything lands 8-13 points above the 13/100 baseline, i.e. 1.7-2.8 sigma, which is where every SpreadsheetBench arm lands.",
 ("SpreadsheetBench", "r50x3"): "Not written up. skill/full at 28/100 is a benchmark best from a five-entry store; raw/minimal at 14/100 is the worst cell of the three repetition legs.",
 ("SpreadsheetBench", "r75x2"): "Not written up. reflection/full 29/100 and skill/minimal 28/100 are joint-best, from stores of 91 and 5 entries respectively — the clearest illustration that store size predicts nothing here.",
 ("SpreadsheetBench", "succ100"): "The comparison the failure leg could not make. skill goes from an empty store to a budget-filling one (2359 of 2500 tokens) purely by flipping the sign of the filter, and full's deletion machinery falls silent — reflection/full issued zero DELETEs in 100 successful episodes against 93 in 100 failed ones.",
 ("SpreadsheetBench", "fail100"): "skill writes nothing from failure and both its cells are therefore extra no-memory draws — 16/100 and 20/100 against the study's 13/100 baseline, which showed 13 was the low draw and made every delta in this document ~3 points optimistic.",
 ("SpreadsheetBench", "gpt_e50"): "Not written up. Swapping the writer does not repeat the ALFWorld story: rule goes 17 -> 24/100 (+11.0, p = 0.019) and skill 20 -> 23, but reflection drops 27 -> 17, giving up the best Qwen-written cell in the sweep. The raw row is a byte-identical copy of the Qwen leg \u2014 raw uses no writer, so this axis cannot move it.",
 ("SpreadsheetBench", "gpt_e100"): "Not written up. skill/minimal reads 27/100 (+14.0, p = 0.004) \u2014 but its four entries are character-identical to the gpt_e50 store that scored 23/100, so the two cells are a replicate pair and the 4-point gap between them is churn, not learning. Nothing else clears the floor.",
 ("SpreadsheetBench", "gpt_e150"): "Not written up. skill/minimal at 30/100 (+17.0, p = 0.0005) is the highest score anywhere on this benchmark, from five entries and one extra write since e50; reflection and rule sit at 21 and 20 with 95 and 83 entries. Whatever is working here, it is not store size.",
 ("Mind2Web", "e50"): "Six cells, six negative deltas, none significant. Around 90% of the damage is the model picking a different element, not a wrong operation; 12.2% of eval steps are unreachable because the ground-truth element is outside the ranker's top 50.",
 ("Mind2Web", "e100"): "raw, reflection and rule are all at the 200-entry cap by this point, so everything past here measures eviction rather than accumulation.",
 ("Mind2Web", "e150"): "reflection reads p = 0.031 here and p = 0.142 one leg later with a store pinned at capacity — the argument for not quoting any single cell.",
 ("Mind2Web", "e200"): "Across a 3.7x increase in evolving experience the largest excursion any arm makes is 1.5 points, and none is monotone. raw's live entry set is identical at 100, 150 and 200, so its last two legs are the same configuration evaluated twice.",
 ("Mind2Web", "gpt_e50"): "Not in RESULTS_MIND2WEB.md. A frontier writer does not rescue this benchmark either: reflection is 3.2 points below baseline (p = 0.020) and the other two are inside the floor.",
 ("ScienceWorld", "e25"): "Not written up anywhere. Every arm lands below the 33/100 baseline.",
 ("ScienceWorld", "e50"): "Not written up. rule/minimal at 38/100 is the only cell above baseline and it does not clear the floor.",
 ("AppWorld", "gpt_e50"): "Not written up. The writer swap moves this benchmark: rule/minimal 25 -> 35/100, +15.0 against the 20 baseline draw and +10.0 against the 25 draw (p = 0.003), and reflection 22 -> 29. skill 25 -> 29 stays inside the 5-point floor. raw and none are byte-identical copies of the Qwen leg \u2014 neither uses a writer, so this axis cannot move them.",
 ("AppWorld", "gpt_e100"): "Not written up. reflection holds at 28/100 where the Qwen-written chain collapsed to 17 \u2014 the writer swap removes that collapse rather than adding a gain (its own 50 -> 100 change is p = 1.000). rule does the opposite of its Qwen twin, 35 -> 22 against 25 -> 34, and that drop is one of the few paired 50 -> 100 changes in the study that is real: b/c 19/6, p = 0.015. Fifty more episodes made this writer\u0027s rule memory worse.",
 ("ALFWorld", "gem_e50"): "Not written up. Every arm clears the floor and beats raw: reflection 85.0% (+27.0), skill 79.0% (+21.0), rule 77.0% (+19.0), all p < 0.002. reflection does it from a SEVEN-entry store against the Qwen writer\u0027s 39 and 546 injected tokens against its 1022 -- on this benchmark the better writer writes less and wins by more.",
 ("ALFWorld", "gem_e100"): "Not written up, and the high-water mark of the entire study: reflection/minimal 91.0%, +33.0 over the no-memory baseline, b/c 3/36, p = 1e-8, from a twelve-entry store. All three writer arms are significant and all three beat raw, which only the gpt-5.6 e150 cell had managed before. reflection is also monotone across its own chain (85 -> 91), which most cells in this study are not.",
 ("ALFWorld", "kimi_e50"): "Not written up. The weakest ALFWorld writer of the four: reflection +6.0 (p = 0.392) and rule +9.0 (p = 0.163) both miss significance, and skill lands at 57.0%, one point BELOW having no memory at all. The skill number is not a memory result -- 34 writer calls produced 2 accepted entries against 26 rejections (23 `content too long`, 10 `missing field`), so the arm evaluates a two-entry store. kimi-k3 writes procedures the 400-token content cap will not admit.",
 ("AppWorld", "gem_e50"): "Not written up. Nothing clears the +/-5 floor: rule +9.0 (p = 0.136) is the best of it. The stores are the smallest anywhere on this benchmark -- 8, 5 and 1 entries, 176-663 injected tokens -- and the writer logged ZERO schema rejections, so this is the model declining to write rather than being filtered. Compare kimi_e50, which wrote 51 rule entries into the same slot and scored the same.",
 ("AppWorld", "gem_e100"): "Not written up. skill/minimal 30.0% (+10.0, p = 0.064) is the largest delta this writer produces on AppWorld and it still does not reach significance, from a six-entry store against reflection\u0027s 17 and rule\u0027s 12. Four writers have now been run at this budget and only gpt-5.6\u0027s e50 rule has ever cleared the floor here.",
 ("AppWorld", "kimi_e50"): "Not written up. rule/minimal +7.0 (p = 0.210) is the best cell and misses. skill is 20/100 -- exactly the baseline, b/c 9/9 -- because 13 writer calls yielded 2 accepted entries; like ALFWorld kimi_e50, that arm measures an almost-empty store, not skill memory.",
 ("AppWorld", "kimi_e100"): "Not written up. The clearest store-size null in the study: rule ends with NINETY live entries, the largest store on this benchmark by a factor of three, and scores 22/100 (+2.0, p = 0.839) -- the same as gemini\u0027s twelve-entry rule (24/100) and worse than its own six-entry skill (26.0%). Writing more did not help, and the 90 entries are what SURVIVED 32 rejections.",
 ("SpreadsheetBench", "gem_e50"): "Not written up. rule/minimal 26/100 (+13.0, p = 0.015) is the only cell past the 4.7-point floor, and the third consecutive lift of this one arm as the writer improves: Qwen 17, gpt-5.6 24, gemini 26 -- from an eleven-entry store. reflection and skill sit at 20 each.",
 ("SpreadsheetBench", "gem_e100"): "Not written up. The e50 ordering does not survive fifty more episodes: rule falls 26 -> 18 and skill rises 20 -> 24 (+11.0, p = 0.019), which is the same swap between those two arms that the gpt-5.6 chain made. Against a 4.7-point floor neither move should be read from the point estimates alone.",
 ("SpreadsheetBench", "kimi_e50"): "IN FLIGHT -- do not quote. reflection died at evolve 24/50 (three consecutive 300s gateway timeouts raise, killing the arm) and is being re-run at a 900s timeout; `none` and `raw` are not seeded until the leg completes, so this cell has no baseline and no delta column yet. What is already visible: skill wrote NOTHING in 50 episodes (9 calls, 0 accepted, empty store, 0 injected tokens), so its 23/100 is an extra no-memory draw rather than a result.",
 ("ScienceWorld", "gpt_e25"): "Not written up. Head of the gpt-5.6 chain, and it has to exist: the e50 leg resumes this store, so the chain cannot inherit the Qwen writer\u0027s. Same shape as everything else on this benchmark \u2014 all four arms below the 33/100 baseline. Only raw (-9.0) is outside the 5-point floor, and raw uses no writer.",
 ("ScienceWorld", "gpt_e50"): "Not written up. reflection/minimal at 34/100 (+1.0) is the only arm above baseline and is well inside the floor. rule drops to 26 from 31 at e25 while its store more than doubles to 44 entries.",
 ("ScienceWorld", "gpt_e100"): "Not written up. A frontier writer does not rescue this benchmark. reflection and rule improve on their Qwen-written twins (26 -> 34 and 24 -> 29) while skill drops 26 -> 22, and not one arm beats raw \u2014 which uses no writer at all. skill at 22/100 (-11.0, p = 0.052) is the only excursion past the 5-point floor and it is downward; its own 50 -> 100 change is not significant (p = 0.167). Chain finished 2026-08-18.",
 ("ScienceWorld", "e100"): "Not written up. raw is the only arm at or above baseline (+3, p = 0.71) and every structured arm loses 7-9 points. Chain finished 2026-08-17.",
}

DOCS = {
    "ALFWorld": "RESULTS_ALFWORLD.md", "WebShop": "RESULTS_WEBSHOP.md",
    "AppWorld": "RESULTS_APPWORLD.md", "SpreadsheetBench": "RESULTS_SPREADSHEETBENCH.md",
    "Mind2Web": "RESULTS_MIND2WEB.md", "ScienceWorld": None,
}


# ----------------------------------------------------------------- page assembly
def build_cells(runs):
    cells = {}
    for r in runs:
        if r["arm"] == "none" or r["stub"] or not r["axis"] or r["axis"] not in AXES:
            continue
        cells.setdefault((r["bench"], r["axis"]), []).append(r)
    for key, arms in cells.items():
        arms.sort(key=lambda r: (r["policy"] != "minimal", ARMS.index(r["arm"]) if r["arm"] in ARMS else 9))
    return cells


def esc(s):
    return html.escape(str(s), quote=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="RESULTS_DASHBOARD.html",
                    help="standalone page to open in a browser")
    ap.add_argument("--fragment",
                    help="also write a head-less copy for hosts that supply their own skeleton")
    args = ap.parse_args()

    runs, replicates = collect()
    default = annotate(runs)
    floor_pairs, floor_draws = noise_floor(runs, replicates, default)
    cells = build_cells(runs)

    payload = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "root": ROOT,
        "benches": BENCHES,
        "arms": ARMS,
        "policies": ["minimal", "full"],
        "groups": [{"title": t, "blurb": b,
                    "rows": [{"key": k, "label": l, "sub": s} for k, l, s in rows]}
                   for t, b, rows in GROUPS],
        "metric": METRIC,
        "floor": {k: {"pts": v[0], "note": v[1]} for k, v in FLOOR.items()},
        "docs": DOCS,
        "baselines": {b: {"success": r["success"], "n": r["n"], "rate": r["rate"],
                          "cond": r["cond"], "path": r["path"]}
                      for b, r in default.items()},
        "notes": {f"{b}|{a}": n for (b, a), n in NOTES.items()},
        "cells": {f"{b}|{a}": [
            {k: r[k] for k in ("arm", "policy", "rate", "score", "delta", "b", "c", "p",
                               "store", "rows", "inj", "writer_calls", "writer_ptok",
                               "writer_ctok", "evolve_n", "evolve_total", "evolve_sr",
                               "n", "success", "wall_h", "cond", "path", "sandbox_errors",
                               "budget")}
            for r in arms] for (b, a), arms in cells.items()},
        "replicates": replicates,
        "floor_pairs": floor_pairs,
        "floor_draws": floor_draws,
        # counted over the cells the grid actually shows, so the unfiltered
        # figures agree with what the filter recomputes
        "n_runs": sum(len(a) for a in cells.values()),
        "n_evals": sum(r["n"] or 0 for a in cells.values() for r in a),
    }

    body = render(payload)
    out = os.path.abspath(args.out)
    with open(out, "w") as fh:
        fh.write(
            '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width,initial-scale=1">\n'
            "</head>\n<body>\n" + body + "\n</body>\n</html>\n"
        )
    print(f"{payload['n_runs']} arm runs across {len(cells)} cells -> {out}")
    if args.fragment:
        frag = os.path.abspath(args.fragment)
        with open(frag, "w") as fh:
            fh.write(body)
        print(f"fragment -> {frag}")


def render(p):
    data = json.dumps(p, separators=(",", ":")).replace("</", "<\\/")
    return TEMPLATE.replace("__DATA__", data)


TEMPLATE = r"""<title>Memory Ablation Atlas</title>
<style>
:root{
  --ground:#F3F4F1; --surface:#FCFCFA; --sunk:#EAEBE6; --rule:#DBDDD5; --rule-2:#C6C9BF;
  --ink:#171A15; --ink-2:#4C534A; --ink-3:#7C8479;
  --pos:#0A7A52; --neg:#BE3C1E; --mid:#8C9289;
  --accent:#84611A; --accent-soft:#EFE7D3;
  --flag:#9A5B08; --flag-soft:#F7ECDA;
  --shadow:0 1px 2px rgba(20,24,18,.06),0 8px 24px -12px rgba(20,24,18,.16);
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --ground:#121410; --surface:#1B1E19; --sunk:#171A15; --rule:#2E332B; --rule-2:#3D4339;
  --ink:#E9EBE3; --ink-2:#A5ADA0; --ink-3:#767E71;
  --pos:#25A170; --neg:#DC6242; --mid:#7A827A;
  --accent:#C9A24E; --accent-soft:#2B2618;
  --flag:#D79542; --flag-soft:#2E2415;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 28px -14px rgba(0,0,0,.7);
}}
:root[data-theme="dark"]{
  --ground:#121410; --surface:#1B1E19; --sunk:#171A15; --rule:#2E332B; --rule-2:#3D4339;
  --ink:#E9EBE3; --ink-2:#A5ADA0; --ink-3:#767E71;
  --pos:#25A170; --neg:#DC6242; --mid:#7A827A;
  --accent:#C9A24E; --accent-soft:#2B2618;
  --flag:#D79542; --flag-soft:#2E2415;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 28px -14px rgba(0,0,0,.7);
}
*{box-sizing:border-box}
body{
  margin:0;background:var(--ground);color:var(--ink);
  font-family:"Charter","Iowan Old Style","Source Serif 4",Georgia,"Times New Roman",serif;
  font-size:16px;line-height:1.5;-webkit-font-smoothing:antialiased;
}
.mono,code,td.num,th.num{font-family:ui-monospace,"SF Mono","JetBrains Mono",Menlo,Consolas,monospace;font-variant-numeric:tabular-nums}
.sans{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif}
h1,h2,h3{text-wrap:balance;margin:0;font-weight:600;letter-spacing:-.01em}
a{color:var(--accent)}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px;border-radius:3px}

.wrap{max-width:1440px;margin:0 auto;padding:0 clamp(16px,3vw,40px) 96px}

/* ---- masthead ---- */
header{border-bottom:1px solid var(--rule);margin-bottom:32px;padding:clamp(32px,5vw,56px) 0 24px}
.eyebrow{font-family:ui-monospace,monospace;font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:var(--ink-3)}
h1{font-size:clamp(30px,4.4vw,46px);line-height:1.05;margin:10px 0 12px}
.lede{max-width:64ch;color:var(--ink-2);font-size:17px}
.meta{display:flex;flex-wrap:wrap;gap:8px 28px;margin-top:22px;font-family:ui-monospace,monospace;font-size:12px;color:var(--ink-3)}
.meta b{color:var(--ink-2);font-weight:500}

/* ---- key figures ---- */
.figs{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:28px 0 36px}
.fig{background:var(--surface);border:1px solid var(--rule);border-radius:6px;padding:16px 18px}
.fig .k{font-family:ui-monospace,monospace;font-size:10.5px;letter-spacing:.12em;text-transform:uppercase;color:var(--ink-3)}
.fig .v{font-family:ui-monospace,monospace;font-size:26px;font-variant-numeric:tabular-nums;margin-top:6px;letter-spacing:-.02em}
.fig .d{font-size:13px;color:var(--ink-2);margin-top:4px;line-height:1.35}

/* ---- filter bar ---- */
.filters{display:flex;flex-wrap:wrap;align-items:center;gap:10px 26px;margin:0 0 14px}
.fgroup{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.fgroup > .lab{font-family:ui-monospace,monospace;font-size:10.5px;letter-spacing:.12em;text-transform:uppercase;color:var(--ink-3)}
.chips{display:flex;gap:6px;flex-wrap:wrap}
button.chip{
  font-family:ui-monospace,monospace;font-size:12px;line-height:1;padding:6px 11px;cursor:pointer;
  background:var(--surface);color:var(--ink-2);border:1px solid var(--rule-2);border-radius:999px;
  transition:background .12s ease,color .12s ease,border-color .12s ease;
}
button.chip:hover{border-color:var(--accent);color:var(--ink)}
button.chip[aria-pressed="true"]{background:var(--accent-soft);border-color:var(--accent);color:var(--ink);font-weight:600}
button.chip[aria-pressed="true"]::before{content:"✓ ";color:var(--accent)}
button.chip.lone{cursor:default}
button.reset{
  font-family:ui-monospace,monospace;font-size:11px;background:none;border:0;color:var(--accent);
  cursor:pointer;text-decoration:underline;text-underline-offset:3px;padding:4px 2px;
}
.fcount{font-family:ui-monospace,monospace;font-size:11px;color:var(--ink-3);margin-left:auto}

/* ---- matrix ---- */
.matrix-shell{overflow-x:auto;border:1px solid var(--rule);border-radius:6px;background:var(--surface);box-shadow:var(--shadow)}
table.matrix{border-collapse:separate;border-spacing:0;width:100%;min-width:900px}
table.matrix th,table.matrix td{border-bottom:1px solid var(--rule);text-align:left;vertical-align:top}
table.matrix thead th{
  position:sticky;top:0;z-index:3;background:var(--surface);
  font-family:ui-monospace,monospace;font-size:11px;letter-spacing:.08em;text-transform:uppercase;
  color:var(--ink-2);font-weight:500;padding:14px 12px;border-bottom:1px solid var(--rule-2);white-space:nowrap;
}
table.matrix thead th .bl{display:block;font-size:10px;letter-spacing:.04em;text-transform:none;color:var(--ink-3);margin-top:3px;white-space:normal}
tr.grouprow td{background:var(--sunk);padding:14px 14px 12px;border-bottom:1px solid var(--rule-2)}
tr.grouprow .gt{font-family:ui-monospace,monospace;font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:var(--accent)}
tr.grouprow .gb{font-size:13.5px;color:var(--ink-2);max-width:80ch;margin-top:3px}
th.axis{padding:10px 14px;font-weight:500;width:196px;min-width:196px;background:var(--surface);position:sticky;left:0;z-index:2}
th.axis .al{font-size:15px;display:block}
th.axis .as{font-family:ui-monospace,monospace;font-size:10.5px;color:var(--ink-3);display:block;margin-top:2px;line-height:1.35;white-space:normal;font-weight:400}
td.cell{padding:0}

button.tile{
  display:flex;flex-direction:column;width:100%;height:100%;min-height:82px;text-align:left;padding:9px 12px 10px;
  background:none;border:0;border-left:1px solid var(--rule);color:inherit;font:inherit;cursor:pointer;
  transition:background .12s ease;
}
button.tile .strip{margin-top:auto}
button.tile:hover{background:var(--sunk)}
button.tile[aria-pressed="true"]{background:var(--accent-soft);box-shadow:inset 2px 0 0 var(--accent)}
button.tile.empty{cursor:default;color:var(--ink-3)}
button.tile.empty:hover{background:none}
button.tile.filtered{cursor:default;color:var(--ink-3);opacity:.55}
button.tile.filtered:hover{background:none}
.tile .dot{font-family:ui-monospace,monospace;font-size:10.5px;color:var(--ink-3);font-style:italic}
.tile .top{display:flex;align-items:baseline;gap:6px;min-width:0}
.tile .best{font-family:ui-monospace,monospace;font-size:19px;font-variant-numeric:tabular-nums;letter-spacing:-.02em;line-height:1.1;white-space:nowrap}
.tile .who{font-family:ui-monospace,monospace;font-size:10.5px;color:var(--ink-3);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.tile .dash{font-family:ui-monospace,monospace;font-size:17px;color:var(--ink-3)}
.tile .sub{font-family:ui-monospace,monospace;font-size:10px;color:var(--ink-3);margin-top:2px}
.pos{color:var(--pos)}.neg{color:var(--neg)}.nul{color:var(--ink-2)}

/* strip: one tick per arm on a shared +/-30pt scale, floor band behind */
.strip{display:block;position:relative;height:22px;margin-top:8px;border-radius:2px;background:var(--sunk);overflow:hidden}
.strip .band{position:absolute;top:0;bottom:0;background:var(--rule);opacity:.9}
.strip .zero{position:absolute;top:0;bottom:0;width:1px;background:var(--rule-2)}
.strip .tick{position:absolute;top:3px;height:16px;width:3px;border-radius:1.5px;border:1px solid var(--surface)}

.flagchip{display:inline-block;font-family:ui-monospace,monospace;font-size:9.5px;letter-spacing:.06em;
  text-transform:uppercase;color:var(--flag);background:var(--flag-soft);border-radius:3px;padding:1px 5px;margin-left:5px}

/* ---- detail ---- */
#detail{margin-top:34px;scroll-margin-top:16px}
.panel{background:var(--surface);border:1px solid var(--rule);border-radius:6px;box-shadow:var(--shadow);overflow:hidden}
.panel-head{padding:20px clamp(16px,2.4vw,26px);border-bottom:1px solid var(--rule);display:flex;flex-wrap:wrap;gap:14px 28px;align-items:flex-start;justify-content:space-between}
.panel-head h2{font-size:24px;line-height:1.15}
.panel-head .ctx{font-family:ui-monospace,monospace;font-size:11.5px;color:var(--ink-3);margin-top:6px;line-height:1.6}
.panel-head .ctx b{color:var(--ink-2);font-weight:500}
.panel-body{padding:0 clamp(16px,2.4vw,26px) 24px}
.note{max-width:72ch;color:var(--ink);font-size:15.5px;margin:20px 0 4px;padding-left:14px;border-left:2px solid var(--accent)}
.warn{max-width:72ch;font-size:14px;color:var(--flag);background:var(--flag-soft);border-radius:4px;padding:11px 14px;margin:18px 0 0}
.tblwrap{overflow-x:auto;margin-top:20px}
table.arms{border-collapse:collapse;width:100%;font-size:13.5px;min-width:760px}
table.arms th{font-family:ui-monospace,monospace;font-size:10px;letter-spacing:.08em;text-transform:uppercase;
  color:var(--ink-3);font-weight:500;text-align:right;padding:0 10px 7px;border-bottom:1px solid var(--rule-2);white-space:nowrap}
table.arms th:first-child,table.arms th.l{text-align:left}
table.arms td{padding:7px 10px;border-bottom:1px solid var(--rule);text-align:right;white-space:nowrap}
table.arms td.l{text-align:left}
table.arms tr.base td{color:var(--ink-2);border-bottom:1px solid var(--rule-2)}
table.arms tr.polhead td{font-family:ui-monospace,monospace;font-size:10px;letter-spacing:.1em;text-transform:uppercase;color:var(--accent);padding-top:14px;border-bottom:0}
.sig{font-weight:600}
.armname{font-family:ui-monospace,monospace;font-size:12.5px}
.paths{margin-top:20px;font-family:ui-monospace,monospace;font-size:11px;color:var(--ink-3);word-break:break-all;line-height:1.7}

/* ---- floor section ---- */
section.floor{margin-top:44px}
section.floor h2{font-size:22px}
section.floor p.intro{max-width:70ch;color:var(--ink-2);margin:10px 0 0;font-size:15.5px}
.floorgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:14px;margin-top:20px;align-items:start}
.fcard{background:var(--surface);border:1px solid var(--rule);border-radius:6px;padding:15px 17px 16px}
.fcard h3{font-size:16px;display:flex;align-items:baseline;justify-content:space-between;gap:10px}
.fcard h3 em{font-family:ui-monospace,monospace;font-size:12px;color:var(--ink-2);font-weight:400;font-style:normal;white-space:nowrap}
.fcard .draws{font-family:ui-monospace,monospace;font-size:12.5px;margin-top:9px;color:var(--ink);display:flex;flex-wrap:wrap;gap:5px;align-items:baseline}
.fcard .draws b{font-weight:600}
.fcard .draws s{text-decoration:none;color:var(--ink-3);font-size:11px}
.fcard .fn{font-size:13px;color:var(--ink-2);margin-top:9px;line-height:1.45}
.fcard ul{list-style:none;margin:10px 0 0;padding:9px 0 0;border-top:1px solid var(--rule);font-family:ui-monospace,monospace;font-size:11px;color:var(--ink-3);display:grid;gap:3px}
.fcard li b{color:var(--ink-2);font-weight:500}

.legend{display:flex;flex-wrap:wrap;gap:8px 22px;margin:18px 0 0;font-family:ui-monospace,monospace;font-size:11px;color:var(--ink-3);align-items:center}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;vertical-align:-1px;margin-right:5px}
footer{margin-top:56px;padding-top:20px;border-top:1px solid var(--rule);font-family:ui-monospace,monospace;font-size:11px;color:var(--ink-3);line-height:1.8}
@media (prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}
</style>

<div class="wrap">
<header>
  <div class="eyebrow">Self-evolving agent memory &middot; ablation grid</div>
  <h1>Memory Ablation Atlas</h1>
  <p class="lede">Every evolving-memory run measured so far, laid out as experiment axis against benchmark.
  Each cell opens the arm-by-arm table behind it &mdash; success rate, paired delta against that sweep&rsquo;s own
  no-memory draw, McNemar exact <em>p</em>, store size, injected tokens and writer cost.</p>
  <div class="meta" id="meta"></div>
</header>

<div class="figs" id="figs"></div>

<div class="filters" id="filters"></div>

<div class="matrix-shell"><table class="matrix" id="matrix"></table></div>
<div class="legend">
  <span><i style="background:var(--pos)"></i>helps, outside the floor</span>
  <span><i style="background:var(--neg)"></i>hurts, outside the floor</span>
  <span><i style="background:var(--mid)"></i>inside the noise floor</span>
  <span><i style="background:var(--flag)"></i>flagged run</span>
  <span>&mdash; never run &nbsp;&middot;&nbsp; <em>n hidden</em> ran, but filtered out</span>
  <span>ticks = one arm each, on a shared &plusmn;30 pt scale; shaded band = that benchmark&rsquo;s floor</span>
</div>

<div id="detail"></div>

<section class="floor">
  <h2>How big is nothing?</h2>
  <p class="intro">Before any delta above means anything, it has to clear the distance between two runs of the
  <em>same</em> configuration. These are every distinct no-memory draw in the archive &mdash; dedicated baseline
  replicates, plus arms that finished with an empty store and injected zero tokens, whose prompt was therefore
  byte-identical to the <code>none</code> arm. The scores are what nothing at all looks like.</p>
  <div class="floorgrid" id="floorgrid"></div>
</section>

<footer id="footer"></footer>
</div>

<script id="payload" type="application/json">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById('payload').textContent);
const $ = (t, c, h) => { const e = document.createElement(t); if (c) e.className = c; if (h != null) e.innerHTML = h; return e; };
const pct = v => v == null ? '—' : (v * 100).toFixed(1) + '%';
const sgn = v => (v > 0 ? '+' : v < 0 ? '−' : '±') + Math.abs(v).toFixed(1);
const glyph = v => v > 0 ? '▲ ' : v < 0 ? '▼ ' : '';
const pfmt = v => v == null ? '—' : v < 0.0001 ? '<0.0001' : v < 0.001 ? v.toFixed(5) : v.toFixed(3);
const num = v => v == null ? '—' : (Math.round(v) === v ? v : v.toFixed(1));
const tone = (d, floor) => d == null ? 'nul' : d > floor ? 'pos' : d < -floor ? 'neg' : 'nul';
const SCALE = 30;   // strip half-range, points

/* ---------- masthead ---------- */
document.getElementById('meta').innerHTML = [
  ['actor model', 'Qwen3.5-9B (vLLM)'], ['seed', '42'], ['rollouts / task', '1'],
  ['content types', 'raw · reflection · rule · skill'],
  ['write policies', 'minimal · full'], ['generated', D.generated],
].map(([k, v]) => `<span>${k} <b>${v}</b></span>`).join('');

const cellKeys = Object.keys(D.cells);
const nAxes = D.groups.reduce((n, g) => n + g.rows.length, 0);

/* ---------- filter state ---------- */
const sel = { arm: new Set(D.arms), pol: new Set(D.policies) };
const keeps = a => sel.arm.has(a.arm) && sel.pol.has(a.policy);
const armsIn = key => (D.cells[key] || []).filter(keeps);
const filtering = () => sel.arm.size < D.arms.length || sel.pol.size < D.policies.length;

const fbar = document.getElementById('filters');
function group(which, title, values) {
  const g = $('div', 'fgroup', `<span class="lab">${title}</span>`);
  const chips = $('div', 'chips');
  values.forEach(v => {
    const c = $('button', 'chip', v);
    c.type = 'button';
    c.addEventListener('click', () => {
      const s = sel[which];
      if (s.has(v)) { if (s.size === 1) return; s.delete(v); } else { s.add(v); }
      refresh();
    });
    chips.appendChild(c);
    c.dataset.which = which; c.dataset.val = v;
  });
  g.appendChild(chips);
  return g;
}
fbar.appendChild(group('arm', 'content type', D.arms));
fbar.appendChild(group('pol', 'write policy', D.policies));
const reset = $('button', 'reset', 'show all');
reset.type = 'button';
reset.addEventListener('click', () => { sel.arm = new Set(D.arms); sel.pol = new Set(D.policies); refresh(); });
fbar.appendChild(reset);
fbar.appendChild($('span', 'fcount'));

function renderFilters() {
  fbar.querySelectorAll('button.chip').forEach(c => {
    const s = sel[c.dataset.which], on = s.has(c.dataset.val);
    c.setAttribute('aria-pressed', on ? 'true' : 'false');
    c.classList.toggle('lone', on && s.size === 1);
    c.title = on && s.size === 1 ? 'at least one must stay selected' : '';
  });
  reset.style.visibility = filtering() ? 'visible' : 'hidden';
}

/* ---------- key figures, recomputed on the selection ---------- */
function renderFigs() {
  const arms = cellKeys.flatMap(k => armsIn(k).map(a => ({ ...a, bench: k.split('|')[0] })));
  const live = cellKeys.filter(k => armsIn(k).length).length;
  const wins = arms.filter(a => a.delta > D.floor[a.bench].pts);
  const harms = arms.filter(a => a.delta < -D.floor[a.bench].pts);
  const scope = filtering()
    ? `${[...sel.arm].join(' · ')} under ${[...sel.pol].join(' + ')}`
    : `${cellKeys.length} filled cells across ${D.benches.length} benchmarks and ${nAxes} axes`;
  document.getElementById('figs').innerHTML = [
    [filtering() ? 'runs in the selection' : 'runs in the grid', arms.length, scope],
    ['episodes evaluated', arms.reduce((s, a) => s + (a.n || 0), 0).toLocaleString(),
     'every arm re-runs the same frozen evaluation set'],
    ['clear the floor upward', wins.length,
     `${arms.length ? (100 * wins.length / arms.length).toFixed(0) : 0}% beat their baseline by more than run-to-run churn`],
    ['clear it downward', harms.length, 'arm runs that measurably hurt'],
  ].map(([k, v, d]) => `<div class="fig"><div class="k">${k}</div><div class="v">${v}</div><div class="d">${d}</div></div>`).join('');
  fbar.querySelector('.fcount').textContent =
    filtering() ? `${arms.length} of ${D.n_runs} runs · ${live} of ${cellKeys.length} cells` : '';
}

/* ---------- matrix ---------- */
const tbl = document.getElementById('matrix');
const thead = $('thead');
const hr = $('tr');
hr.appendChild($('th', 'axis', '<span class="al" style="font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--ink-3)">experiment axis</span>'));
D.benches.forEach(b => {
  const base = D.baselines[b];
  hr.appendChild($('th', null, `${b}<span class="bl">no memory ${base ? (base.n > 200 ? (100 * base.rate).toFixed(1) + '%' : base.success + '/' + base.n) : '—'} · floor ±${D.floor[b].pts}</span>`));
});
thead.appendChild(hr); tbl.appendChild(thead);

const tbody = $('tbody');
const tiles = [];
let current = null;
D.groups.forEach(g => {
  const gr = $('tr', 'grouprow');
  const gtd = $('td'); gtd.colSpan = D.benches.length + 1;
  gtd.innerHTML = `<div class="gt">${g.title}</div><div class="gb">${g.blurb}</div>`;
  gr.appendChild(gtd); tbody.appendChild(gr);

  g.rows.forEach(row => {
    const tr = $('tr');
    tr.appendChild($('th', 'axis', `<span class="al">${row.label}</span><span class="as">${row.sub}</span>`));
    D.benches.forEach(b => {
      const td = $('td', 'cell');
      const key = b + '|' + row.key;
      const btn = $('button', 'tile');
      btn.type = 'button';
      if (!D.cells[key] || !D.cells[key].length) {
        btn.className = 'tile empty';
        btn.innerHTML = '<span class="dash">—</span>';
        btn.disabled = true;
        btn.setAttribute('aria-label', `${row.label} on ${b}: not run`);
      } else {
        btn.addEventListener('click', () => select(key));
        tiles.push({ key, btn, bench: b, label: row.label });
      }
      td.appendChild(btn); tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
});
tbl.appendChild(tbody);

function renderTiles() {
  tiles.forEach(({ key, btn, bench, label }) => {
    const arms = armsIn(key);
    if (!arms.length) {
      btn.className = 'tile filtered';
      btn.disabled = true;
      btn.removeAttribute('aria-pressed');
      const n = D.cells[key].length;
      btn.innerHTML = `<span class="dot">${n} hidden</span>`;
      btn.setAttribute('aria-label',
        `${label} on ${bench}: ran ${D.cells[key].map(a => a.arm + '/' + a.policy).join(', ')}, none matching the current filter`);
      btn.title = `ran ${D.cells[key].map(a => a.arm + '/' + a.policy).join(', ')} — filtered out`;
      return;
    }
    const fl = D.floor[bench].pts;
    const flagged = arms.filter(suspect);
    // headline the largest clean effect; a contaminated or empty-store run is
    // never allowed to stand for the cell
    const pool = arms.filter(a => !suspect(a));
    const best = (pool.length ? pool : arms)
      .reduce((m, a) => (m == null || Math.abs(a.delta) > Math.abs(m.delta)) ? a : m, null);
    const pols = [...new Set(arms.map(a => a.policy))];
    btn.className = 'tile';
    btn.disabled = false;
    btn.title = '';
    btn.innerHTML =
      `<span class="top"><span class="best ${tone(best.delta, fl)}">${glyph(best.delta)}${sgn(best.delta)}</span>` +
      `<span class="who">${best.arm}/${best.policy.slice(0, 3)}</span></span>` +
      `<span class="sub">${arms.length} arm${arms.length > 1 ? 's' : ''} · ${pols.join('+')}` +
      (flagged.length ? `<span class="flagchip">${flagged.length} flagged</span>` : '') + `</span>` +
      strip(arms, fl);
    btn.setAttribute('aria-pressed', key === current ? 'true' : 'false');
    btn.setAttribute('aria-label',
      `${label} on ${bench}: ${arms.length} arms, best ${sgn(best.delta)} points from ${best.arm}/${best.policy}`);
  });
}

/* a run whose numbers do not measure memory: a contaminated eval sandbox, or a
   store that came out empty and left the prompt identical to the none arm */
function suspect(a) { return a.sandbox_errors > 0 || a.store === 0; }

function strip(arms, floor) {
  const x = d => 50 + 50 * Math.max(-1, Math.min(1, d / SCALE));
  const bw = 50 * floor / SCALE;
  let s = `<span class="strip"><span class="band" style="left:${50 - bw}%;width:${2 * bw}%"></span><span class="zero" style="left:50%"></span>`;
  arms.forEach(a => {
    if (a.delta == null) return;
    const hue = suspect(a) ? 'flag' : tone(a.delta, floor) === 'pos' ? 'pos'
      : tone(a.delta, floor) === 'neg' ? 'neg' : 'mid';
    s += `<span class="tick" style="left:calc(${x(a.delta)}% - 2px);background:var(--${hue})"></span>`;
  });
  return s + '</span>';
}

/* ---------- detail ---------- */
function select(key) {
  current = key;
  renderTiles();
  renderDetail();
  const panel = document.getElementById('detail');
  if (panel.scrollIntoView) panel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function renderDetail() {
  const host = document.getElementById('detail');
  if (!current) { host.innerHTML = ''; return; }
  const key = current;
  const [bench, axis] = key.split('|');
  const all = D.cells[key];
  const arms = armsIn(key);
  const fl = D.floor[bench].pts;
  const base = D.baselines[bench];
  const row = D.groups.flatMap(g => g.rows).find(r => r.key === axis);
  const big = base.n > 200;

  const budget = all[0].budget;
  const tasks = all[0].evolve_total || all[0].evolve_n;
  const ctx = [
    ['benchmark', `${bench} · ${D.metric[bench]} · n = ${base.n}`],
    ['no-memory baseline', big ? `${(100 * base.rate).toFixed(1)}%` : `${base.success}/${base.n} (${(100 * base.rate).toFixed(1)}%)`],
    ['noise floor', `±${fl} pts — ${D.floor[bench].note}`],
    ['evolving budget', budget && budget.kind !== 'tasks' ? `${budget.target} ${budget.kind}` : `${tasks} episodes`],
    ['write-up', D.docs[bench] ? D.docs[bench] : 'none yet'],
    ...(arms.length < all.length
      ? [['showing', `${arms.length} of ${all.length} arms — ${all.length - arms.length} hidden by the filter`]]
      : []),
  ].map(([k, v]) => `<b>${k}</b> ${v}`).join('<br>');

  const flagged = arms.filter(a => a.sandbox_errors > 0);
  const empty = arms.filter(a => a.store === 0);
  let warn = '';
  if (flagged.length) warn += `<div class="warn"><b>Eval-sandbox contamination.</b> ${flagged.map(a => `${a.arm}/${a.policy} ${a.sandbox_errors}/${a.n}`).join(', ')} tasks failed with a partially-initialised <code>openpyxl</code> (circular import inside the eval sandbox), not with a wrong answer. Those cells are not measurements of memory.</div>`;
  if (empty.length) warn += `<div class="warn"><b>Empty store.</b> ${empty.map(a => `${a.arm}/${a.policy}`).join(', ')} finished with zero live entries and injected 0 tokens, so the prompt was byte-identical to the no-memory arm. Read those rows as extra baseline draws, not as results.</div>`;

  let rows = '';
  let lastPol = null;
  rows += `<tr class="base"><td class="l armname">none</td><td class="l">—</td><td class="num">${big ? (100 * base.rate).toFixed(1) + '%' : base.success + '/' + base.n}</td><td class="num">—</td><td class="num">—</td><td class="num">—</td><td class="num">—</td><td class="num">0</td><td class="num">—</td><td class="num">—</td></tr>`;
  arms.forEach(a => {
    if (a.policy !== lastPol) {
      rows += `<tr class="polhead"><td colspan="10">WritePolicy.${a.policy}()</td></tr>`;
      lastPol = a.policy;
    }
    const t = tone(a.delta, fl);
    const sig = a.p != null && a.p < 0.05 ? ' sig' : '';
    rows += `<tr>
      <td class="l armname">${a.arm}${a.sandbox_errors > 0 ? '<span class="flagchip">' + a.sandbox_errors + ' bad</span>' : a.store === 0 ? '<span class="flagchip">empty</span>' : ''}</td>
      <td class="l">${a.policy}</td>
      <td class="num">${big ? (100 * a.rate).toFixed(1) + '%' : a.success + '/' + a.n}</td>
      <td class="num ${t}${sig}">${glyph(a.delta)}${sgn(a.delta)}</td>
      <td class="num">${a.b}/${a.c}</td>
      <td class="num${sig}">${pfmt(a.p)}</td>
      <td class="num">${num(a.store)}</td>
      <td class="num">${num(a.inj)}</td>
      <td class="num">${a.writer_calls == null ? '—' : a.writer_calls}</td>
      <td class="num">${a.evolve_sr == null ? '—' : (100 * a.evolve_sr).toFixed(0) + '%'}</td>
    </tr>`;
  });

  const note = D.notes[key];
  const paths = [...new Set(all.map(a => a.path.replace(/\/[^/]+$/, '')))];
  const table = arms.length ? `
      <div class="tblwrap"><table class="arms">
        <thead><tr>
          <th class="l">arm</th><th class="l">policy</th><th>rate</th><th>&Delta; vs none</th>
          <th>b / c</th><th>McNemar p</th><th>store</th><th>inj. tok</th><th>writer calls</th><th>evolve SR</th>
        </tr></thead><tbody>${rows}</tbody>
      </table></div>` : `
      <p class="note" style="border-color:var(--rule-2);color:var(--ink-2)">This cell ran
      ${all.map(a => a.arm + '/' + a.policy).join(', ')}, none of which matches the current filter.</p>`;

  host.innerHTML = `<div class="panel">
    <div class="panel-head">
      <div><h2>${row.label} <span style="color:var(--ink-3);font-weight:400">on</span> ${bench}</h2>
      <div class="ctx">${ctx}</div></div>
    </div>
    <div class="panel-body">
      ${note ? `<p class="note">${note}</p>` : ''}
      ${warn}
      ${table}
      <div class="paths">raw outputs &middot; ${paths.join('<br>raw outputs &middot; ')}</div>
    </div>
  </div>`;
}

/* ---------- noise floor cards ---------- */
document.getElementById('floorgrid').innerHTML = D.benches.map(b => {
  const pairs = D.floor_pairs.filter(p => p.bench === b && p.disagree > 0);
  const draws = (D.floor_draws[b] || []).slice().sort((x, y) => x.success - y.success);
  const f = D.floor[b];
  const big = draws.length && draws[0].n > 200;
  const val = d => big ? (100 * d.success / d.n).toFixed(1) + '%' : String(d.success);
  // always report spread and SD in points of the headline rate
  const pts = draws.map(d => 100 * d.success / d.n);
  const mean = pts.reduce((s, v) => s + v, 0) / (pts.length || 1);
  const sd = pts.length > 1
    ? Math.sqrt(pts.reduce((s, v) => s + (v - mean) ** 2, 0) / (pts.length - 1)) : null;
  const chips = draws.map(d =>
    `<span title="${d.label}"><b>${val(d)}</b>${d.empty ? '<s>&thinsp;empty store</s>' : ''}</span>`
  ).join('<span style="color:var(--ink-3)">·</span>');
  const spread = draws.length > 1
    ? `${draws.length} draws of ${draws[0].n} · spread ${(Math.max(...pts) - Math.min(...pts)).toFixed(1)} pts · SD ${sd.toFixed(1)}`
    : 'one draw only — the floor here is inherited, not measured';
  const items = pairs.length
    ? pairs.slice(0, 6).map(p => `<li><b>${p.a_succ}</b> vs <b>${p.b_succ}</b> &nbsp;—&nbsp; ${p.disagree}/${p.n} flipped</li>`).join('')
    : '';
  return `<div class="fcard">
    <h3>${b}<em>&plusmn;${f.pts} pts assumed</em></h3>
    <div class="draws">${chips}</div>
    <div class="fn">${spread}. ${f.note.charAt(0).toUpperCase() + f.note.slice(1)}.</div>
    ${items ? `<ul>${items}</ul>` : ''}
  </div>`;
}).join('');

/* ---------- footer ---------- */
document.getElementById('footer').innerHTML = `
Deltas are paired: every arm evaluates the identical ordered task list as the <code>none</code> draw shipped with
its own sweep, so McNemar&rsquo;s exact test applies. <em>b</em> = baseline solved it and the arm did not; <em>c</em> = the reverse.
p is the two-sided exact binomial test on the discordant pairs, recomputed here from <code>eval.jsonl</code> rather than copied
from the write-ups. No correction is applied across the ${cellKeys.length} cells &mdash; with well over 200 arm-vs-baseline tests
on the page, treat anything under ~10 points as unresolved regardless of its p.<br>
Generated by <code>scripts/build_dashboard.py</code> from <code>${D.root}</code> on ${D.generated}.`;

/* ---------- one entry point for every state change ---------- */
function refresh() {
  renderFilters();
  renderFigs();
  renderTiles();
  renderDetail();
}

/* open on the study's anchor run: the 50-task ALFWorld sweep */
current = D.cells['ALFWorld|e50'] ? 'ALFWorld|e50' : (tiles[0] && tiles[0].key) || null;
refresh();
</script>
"""

if __name__ == "__main__":
    sys.exit(main())

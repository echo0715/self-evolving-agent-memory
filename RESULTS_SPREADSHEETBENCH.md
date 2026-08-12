# SpreadsheetBench — Memory Content results

Qwen3.5-9B, 50 evolving episodes, 100 evaluation tasks, seed 42.
Adapter and protocol: [RUN_SPREADSHEETBENCH.md](RUN_SPREADSHEETBENCH.md).

---

## Read this first: the noise floor was not measured

**Every delta below is computed against a single `none` run.** The baseline was
run once, at 13/100, and reused for both write policies. On AppWorld two
identical baseline runs scored 20.0% and 25.0% and disagreed on 23 of 100 tasks
— a ±5 point floor that erased every arm in that study. Nothing here rules out
the same thing.

There is one near-replicate in the data, and it is not reassuring. The `raw` arm
has **no LLM writer**: `verify`, `refine` and `batch_induction` cannot fire for
it, so `minimal` and `full` differ only in two usage-statistic deletion flags.
Those two runs:

| | evolve success | eval success | tasks where they disagree |
|---|---|---|---|
| `raw` minimal | 14/50 | 19/100 | — |
| `raw` full | 13/50 | 18/100 | **15** |

One point apart in aggregate, and 15 of 100 individual task outcomes flipped.
Task-level agreement between any two configurations in this study ranges from 9
to 29 disagreements out of 100.

**No task was solved by every arm.** 51 of 100 tasks were solved by at least one
configuration; zero were solved by all nine. Each memory arm gains 8–20 tasks
the baseline missed and loses 3–6 the baseline got.

Treat the tables below as a description of what these particular runs did, not
as a ranking of memory types. The single most valuable next run is a second
`none` baseline (~30 min).

---

## What was run

| | |
|---|---|
| model | `Qwen/Qwen3.5-9B` (non-thinking), vLLM, 2×A100 |
| data | `all_data_912_v0.1`, 3 test cases per task |
| evolving | 50 tasks, SkillOpt id split `train`, seed 42 |
| evaluation | 100 tasks, SkillOpt id split `test`, seed 42, disjoint by task and by source workbook |
| agent | 30 turns max, fenced-block `bash` / `write_file` protocol |
| embedder | `BAAI/bge-large-en-v1.5` on CPU |
| injection budget | 2500 tokens |

A task counts as solved only if **all three test cases pass**. The agent sees
case 1; its `solution.py` is re-executed against cases 2 and 3.

## `WritePolicy.minimal()` — append + merge only

| arm | eval success | rate | score | delta | b/c | McNemar p | store | mem tok | writer calls |
|---|---|---|---|---|---|---|---|---|---|
| none | 13/100 | 13.0% | 18.0 | — | — | — | — | 0 | — |
| raw | 19/100 | 19.0% | 23.0 | +6.0 | 6/12 | 0.238 | 14 | 2085 | — |
| reflection | **27/100** | **27.0%** | **34.3** | **+14.0** | 6/20 | **0.009** | 47 | 1024 | 60 |
| rule | 17/100 | 17.0% | 21.7 | +4.0 | 4/8 | 0.388 | 32 | 456 | 54 |
| skill | 20/100 | 20.0% | 24.3 | +7.0 | 3/10 | 0.092 | 5 | 1692 | 13 |

## `WritePolicy.full()` — every mechanism on

| arm | eval success | rate | score | delta | b/c | McNemar p | store | mem tok | writer calls |
|---|---|---|---|---|---|---|---|---|---|
| none | 13/100 | 13.0% | 18.0 | — | — | — | — | 0 | — |
| raw | 18/100 | 18.0% | 24.7 | +5.0 | 5/10 | 0.302 | 8 | 2086 | — |
| reflection | 23/100 | 23.0% | 29.3 | +10.0 | 6/16 | 0.052 | 30 | 997 | 141 |
| rule | 26/100 | 26.0% | 29.7 | +13.0 | 5/18 | **0.011** | 17 | 537 | 112 |
| skill | **27/100** | **27.0%** | **32.7** | **+14.0** | 5/19 | **0.007** | 3 | 771 | 46 |

`b/c`: b = baseline solved it and the arm did not; c = the reverse.
`score` = mean fraction of test cases passed, ×100.

---

## The ranking is not stable across policy

| arm | minimal | full | swing |
|---|---|---|---|
| raw | 19.0% | 18.0% | −1 |
| reflection | **27.0%** | 23.0% | −4 |
| rule | 17.0% | **26.0%** | **+9** |
| skill | 20.0% | **27.0%** | **+7** |

Under `minimal`, reflection leads and rule is last. Under `full`, rule and skill
lead and reflection is third. A write-mechanism ablation is supposed to move
arms, so this is not automatically an artefact — but a +9 swing is larger than
the `raw` near-replicate's own instability only by a little, and the two
observations cannot be separated without a repeat.

**What is consistent** is the direction. On both graded metrics, all eight
memory configurations sit above the baseline, with no exceptions:

| arm / policy | success | score | cell match |
|---|---|---|---|
| none | 13.0% | 18.0 | 0.379 |
| rule / minimal | 17.0% | 21.7 | 0.415 |
| raw / minimal | 19.0% | 23.0 | 0.432 |
| reflection / full | 23.0% | 29.3 | 0.478 |
| skill / minimal | 20.0% | 24.3 | 0.483 |
| rule / full | 26.0% | 29.7 | 0.497 |
| raw / full | 18.0% | 24.7 | 0.509 |
| skill / full | 27.0% | 32.7 | 0.527 |
| reflection / minimal | 27.0% | 34.3 | 0.540 |

`cell match` is the fraction of graded cells matching gold — a diagnostic, never
a reported score, but it is the least discretised signal available and it orders
the arms almost the same way `score` does. Eight of eight above baseline on a
metric that does not depend on the all-three-cases threshold is the strongest
claim this study supports: **memory helped; which kind of memory helped most is
not resolved.**

## Skill writes almost nothing, and that is structural

| arm / policy | store | writer calls | breakdown |
|---|---|---|---|
| reflection / minimal | 47 | 60 | write 50, merge 10 |
| rule / minimal | 32 | 54 | write 50, merge 4 |
| skill / minimal | **5** | **13** | write 13 |
| reflection / full | 30 | 141 | write 50, judge 46, merge 32, refine 9, induce 4 |
| rule / full | 17 | 112 | write 50, judge 49, merge 9, induce 4 |
| skill / full | **3** | 46 | write 12, judge 30, induce 4 |

A procedural skill needs one working path, so skill cannot write from an
all-failure episode. With 34–41 of 50 evolving episodes ending in all-failure,
its writer was invoked 12–13 times against reflection's and rule's 50. This is
the irreducible asymmetry the README says should be stated rather than
engineered away, and a 13% baseline amplifies it.

That a 3-entry store ties for the best `full` result is the single most
suspicious number in this document. Either three procedures generalise
unusually well on this benchmark, or it is a favourable draw. One run cannot
tell them apart.

## Failure modes

Counts over the 100 evaluation tasks, `none` arm:

| reason | n | meaning |
|---|---|---|
| `eval-mismatch` | 77 | ran, produced a workbook, wrong values |
| `output-not-found` | 7 | never wrote the output file |
| `exec-error` | 2 | `solution.py` crashed on case 2 or 3 |
| `no-solution-py-for-other-cases` | 1 | edited case 1 by hand |

The benchmark is essentially all `eval-mismatch`: 97% of episodes produced a
`solution.py`, and the plumbing failures memory might plausibly fix
(`output-not-found`, `exec-error`) account for 10 tasks total. Memory arms cut
those to 2–8 but the aggregate is dominated by getting the *values* right.

Two arms show a plumbing effect worth noting: `rule / full` drove
`output-not-found` from 7 to 2 and had the highest `wrote_solution_rate` (0.99),
and `skill / full` had the lowest mean bash errors (0.64 vs the baseline's 0.96).
Both are consistent with procedural memory fixing the mechanics rather than the
spreadsheet reasoning — but both are single-run observations on small counts.

## Cost

| arm / policy | wall | writer prompt tok | writer completion tok |
|---|---|---|---|
| none | 28 min | — | — |
| raw / full | 80 min | — | — |
| skill / full | 75 min | 176k | 18k |
| rule / full | 89 min | 353k | 44k |
| reflection / full | 91 min | 405k | 54k |

Wall time is not comparable across arms — four arms shared two vLLM servers.
The writer token columns are, and they are the real compute difference:
`full` costs reflection and rule 2.3–2.4× their `minimal` writer tokens, for the
verify/refine/induce loop.

## What to run next, in order

1. **A second `none` baseline.** ~30 minutes. Nothing above is interpretable
   without it. If the floor is ±5, only reflection/minimal, rule/full and
   skill/full survive, and none of the between-arm differences do.
2. **A repeat of `skill / full`.** The 3-entry store tying for best is either
   the most interesting result here or the clearest noise artefact, and it is
   cheap to check.
3. **Raise the baseline before comparing memory types.** At 13/100, floor
   effects compress every arm into a 14-point band. Either more agent turns or
   a stronger model would separate the types better than more evolving episodes.

## Reproducing

```bash
bash scripts/setup_spreadsheetbench.sh
python scripts/build_spreadsheetbench_manifests.py
bash scripts/serve_qwen.sh 0 8000 --background
bash scripts/serve_qwen.sh 1 8001 --background
bash scripts/run_spreadsheetbench_sweep.sh minimal
bash scripts/run_spreadsheetbench_sweep.sh full
```

Raw outputs: `$MEMSYS_RESULTS_ROOT/spreadsheetbench/{minimal,full}/<arm>_<policy>/`
— `summary.json`, `eval.jsonl`, `evolve_episodes.jsonl`, `evolve_log.jsonl`,
`store.jsonl`, and per-task `solution.py` + predicted workbooks.

Manifest fingerprints (`task_ids_sha256`, first 16):
evolve `654ef6ea3a1d748e`, eval `5637cf201e1948d9`.

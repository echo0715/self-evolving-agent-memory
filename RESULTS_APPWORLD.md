# AppWorld Memory Content study — Qwen3.5-9B

Run 2026-08-10. 50 evolving tasks (`train`), 100 evaluation tasks (`test_normal`,
disjoint), 1 rollout per task, 30-interaction horizon, seed 42. Model served
locally by vLLM; the same model writes memory and acts.

Extended 2026-08-12 to 100 evolving episodes per arm, evaluated on the same
frozen 100 tasks — §5. The headline below is the 50-episode run; §5 reports both
budgets side by side.

Raw outputs: `/gpfs/radev/scratch/cohan/jw3278/memsys_results/appworld/`.
Reproduce with [RUN_APPWORLD.md](RUN_APPWORLD.md).

## Read this first: nothing here is resolvable

The no-memory baseline was run **twice**, identically — same 100 tasks, same
code, no memory either time:

| run | TGC | score |
|---|---|---|
| `none` rep1 | 20.0% | 61.1 |
| `none` rep2 | **25.0%** | 65.8 |

The two runs disagree on **23 of 100 tasks** (b/c = 9/14, McNemar p = 0.405).
**The run-to-run noise floor on this benchmark is ±5 points of TGC**, which is
the size of nearly every effect measured below.

Testing every arm against *both* baselines:

| arm | TGC | vs rep1 | p | vs rep2 | p |
|---|---|---|---|---|---|
| `none` rep2 | 25.0 | +5.0 | 0.405 | — | — |
| raw / minimal | 26.0 | +6.0 | 0.345 | +1.0 | 1.000 |
| reflection / minimal | 22.0 | +2.0 | 0.824 | −3.0 | 0.648 |
| rule / minimal | 25.0 | +5.0 | 0.424 | **+0.0** | 1.000 |
| skill / minimal | 25.0 | +5.0 | 0.424 | **+0.0** | 1.000 |
| **raw / full** | **31.0** | **+11.0** | **0.052** | +6.0 | 0.392 |
| reflection / full | 21.0 | +1.0 | 1.000 | −4.0 | 0.541 |
| rule / full | 23.0 | +3.0 | 0.690 | −2.0 | 0.851 |
| skill / full | 19.0 | −1.0 | 1.000 | −6.0 | 0.345 |

**No arm beats both baselines.** `rule/minimal` and `skill/minimal`, which look
like +5.0 against rep1, are exactly +0.0 against rep2 — they are
indistinguishable from re-running the no-memory arm.

Had only rep1 been run, this file would have reported "all four `minimal` arms
beat baseline by +2 to +6", which reads as a consistent positive effect and is
not one. The replicate cost 68 minutes on servers that were otherwise idle. It is
the most valuable measurement in the run, and it should be the first thing done
on any new benchmark rather than the last.

`RESULTS_ALFWORLD.md` already carried this warning second-hand — SAGE measured
54 / 58 / 60% across three identical ALFWorld baselines. Nobody had measured it
here.

## Results

`success`/TGC is Task Goal Completion (every unit test for the task passes);
`score` is the graded fraction of tests passed. Deltas below are against rep1,
the baseline the sweep itself produced; read them against the ±5 floor above.

| arm | policy | TGC | score | done% | store | inj. tok | writer calls | wall |
|---|---|---|---|---|---|---|---|---|
| none | — | 20.0% / 25.0% | 61.1 / 65.8 | 91 / 87 | — | 0 | — | 68 min |
| raw | minimal | 26.0% | 63.3 | 89 | 14 | 1935 | — | 130 min |
| reflection | minimal | 22.0% | 64.5 | 92 | 39 | 1083 | 88 | 181 min |
| rule | minimal | 25.0% | 66.4 | 90 | 24 | 441 | 75 | 147 min |
| skill | minimal | 25.0% | 64.7 | 91 | 5 | 1471 | 14 | 165 min |
| raw | full | **31.0%** | **69.0** | 92 | 14 | 1907 | — | 130 min |
| reflection | full | 21.0% | 64.8 | 91 | 21 | 979 | 176 | 167 min |
| rule | full | 23.0% | 65.2 | 90 | 13 | 448 | 147 | 149 min |
| skill | full | 19.0% | 63.6 | 92 | 5 | 1606 | 59 | 124 min |

Every arm completes ~90% of its episodes (the agent calls
`apis.supervisor.complete_task()` rather than running out of interactions) and
uses ~17–18 of its 30 interactions. Unlike WebShop, the horizon is **not** the
binding constraint here, and injected memory does not push episodes into
timeouts: `done%` moves by 5 points across the whole table, in no consistent
direction.

## 1. The one candidate effect, and why it is only a candidate

`raw/full` is the only arm outside the noise floor against rep1: **+11.0 points,
p = 0.052**, with b/c = 8/19 — it lost 8 tasks and won 19. Against rep2 it is
+6.0 at p = 0.392.

Taking the two baselines' mean (22.5%) as the reference puts it at +8.5. That is
the largest effect in the run and the only one worth a follow-up, but a
borderline p against one reference and a null against another is not a result.
It needs ≥3 seeds of both the arm and the baseline.

It is also consistent with the other two benchmarks in *ordering*:

| benchmark | best arm | Δ | p | outside a measured noise floor? |
|---|---|---|---|---|
| ALFWorld | raw / full | +22.0 | 0.0002 | yes (SAGE: ±3–6) |
| WebShop | raw / full | +8.0 | 0.096 | **unknown — never measured** |
| AppWorld | raw / full | +11.0 | 0.052 | no (+6.0 vs rep2) |

`raw` is the top arm on all three benchmarks. Only ALFWorld's effect is clearly
real. WebShop's needs the same replicate treatment applied here before anyone
leans on it.

What makes the ordering interesting rather than incidental: **`raw` is the only
arm with no LLM writer.** It stores successful trajectories verbatim, and the
sole `full`-policy mechanism touching it is utility deletion, which is
bookkeeping. Every arm that invokes a writer to abstract, generalise or
consolidate ranks below it on all three benchmarks. That is a negative claim
about abstraction, not a positive one about any memory type, and it is the only
pattern here that three benchmarks agree on.

## 2. The write-mechanism axis is not resolvable either

| arm | TGC minimal → full | score | writer calls |
|---|---|---|---|
| raw | 26.0 → 31.0 (+5.0) | 63.3 → 69.0 (+5.7) | 0 → 0 |
| reflection | 22.0 → 21.0 (−1.0) | 64.5 → 64.8 (+0.3) | 88 → 176 |
| rule | 25.0 → 23.0 (−2.0) | 66.4 → 65.2 (−1.2) | 75 → 147 |
| skill | 25.0 → 19.0 (−6.0) | 64.7 → 63.6 (−1.1) | 14 → 59 |

The direction matches ALFWorld's headline — the writer-free arm gains under
`full`, the writer arms lose while roughly doubling their writer bill — and
contradicts WebShop, where `full` raised the score for all four arms. But every
delta here except `skill`'s −6.0 is inside ±5, so **this run neither confirms nor
refutes the ALFWorld finding.** Reporting it as a reproduction would be wrong.

## 3. Compression reproduces, its benefit does not

`skill` again collapses to a tiny store — **5 procedures** from 14 writer calls,
matching ALFWorld's 4 and WebShop's 4 from comparable budgets. What it kept is
strikingly uniform:

```
authenticate_and_filter_received_payments
authenticate_and_list_directory_contents
authenticate_and_refund_sent_payment_request
authenticate_and_find_playlist_by_title
authenticate_and_delete_messages_by_sender
```

Every one begins with authentication: the writer independently discovered that
"read the password from the supervisor app → log in → then act" is the universal
AppWorld prefix. That is a real regularity of the benchmark, correctly induced.

It bought nothing. `skill/minimal` is +0.0 against rep2, and `skill/full` is the
worst arm in the run. As on WebShop, extreme compression finds something true and
general, and the agent does not convert it into completed tasks.

## 4. Corrections to earlier readings of this run

Recorded because they were wrong in an instructive way, and each was produced by
reading rep1 alone:

- **"`rule` gains entirely on the hard tier (+5 on difficulty_3)."** Baseline
  difficulty_3 is 2/40 in rep1 and **10/40** in rep2. That tier is the least
  stable part of the benchmark; no per-difficulty reading in this run survives
  its swing. The per-difficulty table below is kept for completeness and should
  not be interpreted.
- **"The ALFWorld write-mechanism finding reproduces on AppWorld."** See §2 —
  the deltas are inside the noise floor.

| arm | policy | difficulty_1 (30) | difficulty_2 (30) | difficulty_3 (40) |
|---|---|---|---|---|
| none rep1 | — | 11 | 7 | 2 |
| none rep2 | — | 9 | 6 | **10** |
| raw | minimal | 15 | 8 | 3 |
| reflection | minimal | 12 | 6 | 4 |
| rule | minimal | 12 | 6 | 7 |
| skill | minimal | 14 | 7 | 4 |
| raw | full | 15 | 7 | 9 |
| reflection | full | 12 | 4 | 5 |
| rule | full | 10 | 7 | 6 |
| skill | full | 8 | 5 | 6 |

## 5. Doubling the evolving budget: 50 → 100 episodes

Run 2026-08-12. Each arm resumed **its own** 50-episode `store.jsonl` and evolved
the next 50 tasks of the same seeded permutation
(`manifests/appworld_evolve_train+dev_50to100_seed42.json`), then re-ran the same
frozen 100-task `test_normal` evaluation. The baselines are unchanged — `none`
has no memory, so its evaluation does not depend on the evolving budget.

**The evolving set spills into `dev`.** AppWorld's `train` split holds 90 tasks
and the first run spent 50, so positions [50, 100) are 40 `train` + 10 `dev`.
`dev` is disjoint from `test_normal`, so the evaluation set is untouched, but a
100-episode AppWorld store has seen two splits where ALFWorld's and WebShop's saw
one. Outputs: `memsys_results/appworld/{full,minimal}_e100/`.

| arm | policy | ep | TGC | score | vs rep1 (20.0) | p | vs rep2 (25.0) | p | vs own e50 | p |
|---|---|---|---|---|---|---|---|---|---|---|
| raw | minimal | 50 | 26.0 | 63.3 | +6.0 | 0.345 | +1.0 | 1.000 | — | — |
| raw | minimal | 100 | 23.0 | 65.4 | +3.0 | 0.648 | −2.0 | 0.868 | −3.0 | 0.720 |
| reflection | minimal | 50 | 22.0 | 64.5 | +2.0 | 0.824 | −3.0 | 0.648 | — | — |
| reflection | minimal | 100 | 17.0 | 62.7 | −3.0 | 0.664 | −8.0 | 0.185 | −5.0 | 0.405 |
| **rule** | **minimal** | 50 | 25.0 | 66.4 | +5.0 | 0.424 | +0.0 | 1.000 | — | — |
| **rule** | **minimal** | **100** | **34.0** | **69.5** | **+14.0** | **0.013** | **+9.0** | 0.150 | +9.0 | 0.188 |
| skill | minimal | 50 | 25.0 | 64.7 | +5.0 | 0.424 | +0.0 | 1.000 | — | — |
| skill | minimal | 100 | 23.0 | 65.8 | +3.0 | 0.607 | −2.0 | 0.832 | −2.0 | 0.839 |
| raw | full | 50 | 31.0 | 69.0 | +11.0 | 0.052 | +6.0 | 0.392 | — | — |
| **raw** | **full** | **100** | **36.0** | 68.5 | **+16.0** | **0.002** | **+11.0** | 0.080 | +5.0 | 0.405 |
| reflection | full | 50 | 21.0 | 64.8 | +1.0 | 1.000 | −4.0 | 0.541 | — | — |
| reflection | full | 100 | 27.0 | 65.0 | +7.0 | 0.189 | +2.0 | 0.845 | +6.0 | 0.286 |
| rule | full | 50 | 23.0 | 65.2 | +3.0 | 0.690 | −2.0 | 0.851 | — | — |
| rule | full | 100 | 26.0 | 64.6 | +6.0 | 0.286 | +1.0 | 1.000 | +3.0 | 0.690 |
| skill | full | 50 | 19.0 | 63.6 | −1.0 | 1.000 | −6.0 | 0.345 | — | — |
| skill | full | 100 | 23.0 | 66.1 | +3.0 | 0.678 | −2.0 | 0.845 | +4.0 | 0.503 |

**Two arms now clear the ±5 floor against both baselines** — `raw/full` (+16.0 /
+11.0) and `rule/minimal` (+14.0 / +9.0). At 50 episodes only `raw/full` did, and
only against one. Against the two baselines' mean (22.5) they are +13.5 and
+11.5.

Three things keep this from being a result:

- **No arm's own 50 → 100 change is significant.** The best is `rule/minimal` at
  +9.0, p = 0.188; `raw/full` is +5.0, p = 0.405. So "more experience helps" is
  not what this measures. What moved is *which* arms land high, and each arm
  moved by about the noise floor.
- **The two winners are in different policy columns.** `raw` wins under `full`
  and loses 3 points under `minimal`; `rule` does the reverse, gaining 9 under
  `minimal` and 3 under `full`. There is no mechanism that predicts both.
- **`rule/minimal` was +0.0 against rep2 at 50 episodes.** Its entire effect
  appears in the second half, from an arm that was previously indistinguishable
  from re-running `none`. That is exactly the shape §4 warns about.

`raw/full` is the one thing that survives across both budgets: +11.0 then +16.0
against rep1, +6.0 then +11.0 against rep2, monotone in the evolving budget, and
still the top arm. It remains the study's only candidate effect, now with a
second data point rather than a confirmation. It still needs ≥3 seeds.

Bookkeeping for the second 50 episodes:

| arm | policy | store 50 → 100 | writer calls (2nd 50) | inj. tok | evolve success 1st/2nd 50 |
|---|---|---|---|---|---|
| raw | minimal | 14 → 26 | — | 1925 | 30% / 24% |
| reflection | minimal | 39 → 70 | 84 | 1035 | 34% / 22% |
| rule | minimal | 24 → 39 | 84 | 455 | 22% / 20% |
| skill | minimal | 5 → 9 | 19 | 2290 | 28% / 32% |
| raw | full | 14 → 27 | — | 1911 | 30% / 36% |
| reflection | full | 21 → 42 | 187 | 923 | 30% / 32% |
| rule | full | 13 → 15 | 175 | 515 | 32% / 36% |
| skill | full | 5 → 6 | 86 | 1687 | 22% / 28% |

Stores roughly double under `minimal` and grow far less under `full`, which is
utility deletion doing its job — `rule/full` added 2 entries net across 50
episodes and 175 writer calls. **`skill` still compresses to almost nothing** (5
→ 6 under `full`, 5 → 9 under `minimal`): a second 50 episodes did not change
what §3 found. Injected-token budgets are unchanged, so no arm's gain is a
side-effect of getting more context.

## 6. Three silent failures found while building this

All three produced results that looked normal. They are documented in
[RUN_APPWORLD.md](RUN_APPWORLD.md) §5 with the fixes; the point here is that each
would have corrupted the table without raising anything.

1. **The writer's JSON did not parse, and nothing said so.** Asked to quote a
   trajectory verbatim into `evidence`, the model emits JSON it cannot escape —
   AppWorld trajectories are Python code and API docs full of quotes and
   backslashes, and the escaping is dropped mid-string. `parse_ops` returns `[]`
   for unparseable text, so the arm logged writer calls, **zero ops and zero
   rejections**, and finished with an empty store — identical to `none` while
   still reporting a success rate. Fixed with constrained decoding
   (`response_format={"type":"json_object"}`), off by default elsewhere.
2. **The grounding-evidence cap rejected almost every write.**
   `MAX_EVIDENCE_TOKENS = 80` is calibrated for ALFWorld/WebShop observation
   prose; AppWorld evidence is code, and the dominant rejection reason was
   "evidence too long". It bit hardest on the arm that writes least often
   (`skill`, which only writes from successful episodes), so it would have read
   as a content-type result. Raised to 300 uniformly for AppWorld; evidence is
   provenance, not injected context, so no arm gained context from this.
3. **All arms shared one AppWorld experiment namespace.** Concurrent arms raced
   on the same `experiments/outputs/<name>/tasks/` tree; one arm's setup removed
   directories another was using, surfacing as `FileNotFoundError` from
   `os.makedirs(..., exist_ok=True)`. Each arm now gets `memsys_<arm>_<policy>`.

## Threats to validity

- **n = 100, and the measured noise floor is ±5 points.** At 50 episodes nothing
  except `raw/full` exceeds it against even one baseline, and nothing exceeds it
  against both; at 100, `raw/full` and `rule/minimal` do (§5). Effects below ~10
  points are not measurable in this design.
- **The 100-episode evolving set spans two splits.** `train` holds 90 tasks, so
  positions [50, 100) are 40 `train` + 10 `dev`. Disjoint from `test_normal` in
  both cases, but not the single-split design ALFWorld and WebShop use at 100.
- **Two baseline runs is a floor estimate, not a distribution.** ±5 comes from a
  single pair; the true spread could be wider.
- **The 30-interaction horizon is below AppWorld's default 50**, chosen for cost
  (the scaffold is ~6.6k tokens, Qwen3.5 is hybrid so vLLM disables prefix
  caching, and every turn re-prefills). Absolute numbers are not comparable to
  50-interaction results. Unlike WebShop the horizon does not appear to bind —
  ~90% of episodes finish voluntarily at ~17 of 30 interactions.
- **The scaffold is ACE's, the study's is not.** The prompt is ACE's published
  AppWorld ReAct generator prompt verbatim, including its `{{ playbook }}` slot,
  which is where the memory block goes. Good provenance, but this is a 9B model
  where ACE used frontier models, so the absolute TGC is not comparable to ACE's.
- **`skill` and `raw` write only from successful episodes**, and only ~28% of
  evolving episodes succeed (14–17 of 50). Both arms therefore learn from ~15
  episodes, not 50 — a much smaller effective budget than `reflection` (88 writer
  calls) or `rule` (75).
- **Writer model = actor model**, so content quality and consumption ability are
  confounded.
- **Scenario is a weak cluster key.** `task_type` is the scenario prefix, and
  `required_apps` — the natural grouping — is deliberately not used because it
  lives under `ground_truth/` and would leak part of the solution path.

## What to run next

1. **≥3 seeds of the baseline, of `raw/full`, and now of `rule/minimal` at 100
   episodes.** The two candidate effects, and the only way to settle either. The
   100-episode run (§5) doubled the evidence for `raw/full` without confirming
   it, and raised a second candidate whose own 50 → 100 change is p = 0.188.
2. **The same replicate treatment for WebShop.** Its `raw/full` (+8.0, p = 0.096)
   has never been checked against a measured noise floor, and this run shows what
   that check can do to a table.
3. **A larger evaluation set.** At ±5 noise and effects of ~5–10 points, n = 100
   is under-powered by roughly 3×.

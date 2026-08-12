# WebShop Memory Content study — Qwen3.5-9B

Run 2026-08-10. 50 evolving tasks (`train`, goal indices ≥ 1500), 100 evaluation
tasks (`test`, goal indices < 500, disjoint), 1 rollout per task, 15-step
horizon, seed 42. Full 1.18M-product corpus, 12,087 human-written instructions.
Model served locally by vLLM; the same model writes memory and acts.

Raw outputs: `/gpfs/radev/scratch/cohan/jw3278/memsys_results/webshop/{minimal,full}/`.
Reproduce with [RUN_WEBSHOP.md](RUN_WEBSHOP.md).

Read this alongside [RESULTS_ALFWORLD.md](RESULTS_ALFWORLD.md). Same memory
systems, same model, same write policies, same author — and on the two headline
questions the two benchmarks disagree. That disagreement is the most useful thing
in this file.

## Scaffold validation, and its limit

The no-memory baseline scored **29.0% success / 59.8 score**, inside the band
published WebShop agents occupy (roughly 28–40% success, 60–66 score for
ReAct-style LLM agents). 86% of its episodes end in a purchase, it uses 6.9 of
its 15 steps, and it issues 0.35 invalid actions per episode.

**This is a sanity check, not a provenance match, and the difference matters.**
The ALFWorld scaffold is byte-identical to SAGE's, which pins that baseline to an
independently measured 58–60% and makes it a genuine canary. Nothing equivalent
was available here: MemRL's `web_shopping` task is a stub and RAGEN's WebShop
prompt carries RL-specific hand-holding hints, so the system prompt was written
for this study and the two demonstrations were recorded by replaying real
trajectories through the live store (both verified to end at reward 1.0). Landing
in the published band means the harness is not broken. It does **not** license
comparing 29.0% against any published number. The `none` arm is the only
reference here that means anything.

## Results

Every arm evaluates the identical ordered task list, so comparisons against the
baseline are paired and McNemar's exact test applies. `b` = baseline solved it
and the arm did not; `c` = the reverse. `none` has no store and no writer, so it
is policy-independent and appears once.

WebShop grades partially: **score** is the mean graded reward ×100, **rate** is
the fraction scoring exactly 1.0. Both are reported because they move apart.

| arm | policy | rate | Δ | score | Δ | b/c | McNemar p | store | inj. tok | writer calls |
|---|---|---|---|---|---|---|---|---|---|---|
| none | — | 29.0% | — | 59.8 | — | — | — | — | 0 | — |
| raw | minimal | 30.0% | +1.0 | 58.1 | −1.7 | 8/9 | 1.000 | 16 | 1177 | — |
| reflection | minimal | 28.0% | −1.0 | 51.2 | −8.6 | 10/9 | 1.000 | 18 | 1117 | 58 |
| rule | minimal | 31.0% | +2.0 | 57.0 | −2.8 | 6/8 | 0.791 | 31 | 468 | 58 |
| skill | minimal | 32.0% | +3.0 | 55.9 | −3.9 | 6/9 | 0.607 | 4 | 969 | 20 |
| raw | full | **37.0%** | **+8.0** | **62.0** | +2.2 | 5/13 | 0.096 | 17 | 1261 | — |
| reflection | full | 25.0% | −4.0 | 52.8 | −7.0 | 10/6 | 0.454 | 13 | 1004 | 126 |
| rule | full | 35.0% | +6.0 | 58.8 | −1.0 | 6/12 | 0.238 | 7 | 294 | 111 |
| skill | full | 28.0% | −1.0 | 61.6 | +1.8 | 7/6 | 1.000 | 1 | 249 | 51 |

**No arm reaches p < 0.05.** `raw/full` is the strongest at +8.0 points and
p = 0.096, and it is the only arm whose discordant counts are lopsided enough to
be suggestive (it lost 5 tasks and won 13). Everything else is churn.

This is the first thing to say plainly: on ALFWorld the same code produced
`raw` +17.0 (p = 0.009) and `skill/minimal` +21.0 (p = 0.0002). Here the largest
effect is half that size and does not clear significance at n = 100.

## 1. The near-null is two real effects cancelling

The aggregate hides the mechanism completely. Decomposing every episode into
*did the agent buy anything* and *how good was what it bought*:

| arm | policy | purchase rate | rate given purchase | reward given purchase | mean steps |
|---|---|---|---|---|---|
| none | — | 86% | 33.7% | 69.6 | 6.94 |
| raw | minimal | 82% | 36.6% | 70.9 | 6.55 |
| reflection | minimal | 77% | 36.4% | 66.5 | 7.82 |
| rule | minimal | 83% | 37.3% | 68.7 | 6.98 |
| skill | minimal | 76% | **42.1%** | **73.5** | 8.28 |

Under `minimal`, **every arm is a better shopper than the baseline and every arm
finishes fewer episodes.** Success-given-purchase rises by 2.7 to 8.4 points
while the purchase rate falls by 3 to 10 points, and the two roughly cancel.

The lost episodes are not errors. Of the non-purchases — 14, 18, 23, 17 and 24
across the five arms — **every single one is a 15-step timeout**: 14/14, 18/18,
23/23, 17/17, 24/24, with zero LLM or environment failures. The step budget is
where the memory benefit goes.

`skill/minimal` is the clearest case. It is the best shopper in the entire run
(42.1% success given purchase, against the baseline's 33.7%) and the second-worst
at finishing (76%), because it spends 8.28 of its 15 steps against the baseline's
6.94. Its net is +3.0 points at p = 0.607.

**The 15-step horizon is therefore a confound, not a neutral setting.** It is
tight enough that reading ~1,000 tokens of injected memory and acting on it costs
the purchase that the memory was about to improve. ALFWorld's 50-step horizon has
no equivalent squeeze, which is one candidate explanation for why the same
systems look so different there. The control is to re-run at a longer horizon
(25 steps) and see whether the arms separate; that ablation has **not** been run.

## 2. More write mechanism helped here — the opposite of ALFWorld

ALFWorld's headline was that `WritePolicy.minimal()` dominated `full()` for every
content type that uses an LLM writer, losing 7 to 15 points each. On WebShop the
sign flips:

| arm | rate minimal → full | score minimal → full | purchase rate | mean steps | writer calls |
|---|---|---|---|---|---|
| raw | 30.0% → **37.0%** (+7) | 58.1 → 62.0 (**+3.9**) | 82% → 86% | 6.55 → 6.04 | none → none |
| reflection | 28.0% → 25.0% (−3) | 51.2 → 52.8 (**+1.6**) | 77% → 82% | 7.82 → 7.26 | 58 → 126 |
| rule | 31.0% → **35.0%** (+4) | 57.0 → 58.8 (**+1.8**) | 83% → 84% | 6.98 → 6.70 | 58 → 111 |
| skill | 32.0% → 28.0% (−4) | 55.9 → 61.6 (**+5.7**) | 76% → 86% | 8.28 → 6.62 | 20 → 51 |

`full` raises the score for **all four** arms, raises the purchase rate for
**all four**, and lowers mean steps for **all four**. That is exactly the tax
described in §1 being paid down: the extra mechanisms are all forms of pruning —
utility deletion drops entries that get retrieved but rarely coincide with
success, batch induction consolidates, delete-on-low-confidence removes the
rest — and a pruned store is cheaper to act on. `rule`'s store falls 31 → 7 and
its injected tokens 468 → 294; `skill`'s falls 4 → 1 and 969 → 249.

The strict success rate splits 2–2, so this is a score-and-efficiency result, not
a clean win. But the direction is consistent across four arms on three separate
measures, and it directly contradicts what the same code did on ALFWorld.

**The ALFWorld conclusion should not be stated as a property of the write
policy.** On a benchmark with a tight step budget and a graded reward, the
pruning mechanisms earn their keep.

## 3. Compression helps until it consolidates away the tactics

`skill` is the sharpest content-type story, in both directions.

`skill/minimal` keeps **4** live skills, and they are tactical recovery
procedures — what to do when the page does not offer what you need:

```
filter_product_by_variant
  scan the available <entity_type> options (e.g., size, color, quantity)
  if the requested value is not visible, click 'back to search'
  return to the search results and refine the query
```

That is not in the system prompt or the demonstrations, and it is the arm with
the best success-given-purchase in the run (42.1%).

`skill/full` consolidates the store to **1** skill. It is correct, complete, and
useless:

```
search_and_select_exact_size_product
  search using descriptive keywords excluding price and size constraints
  identify a listing matching the product type and within the price limit
  click the listing to view details
  locate the size or variant selector
  select the option matching the requirement
  verify the selection and the price
  proceed to purchase
```

This is a faithful restatement of the scaffold the agent already has — the system
prompt plus two demonstrations say exactly this. Its success-given-purchase is
**32.6%, below the 33.7% baseline**: the agent pays 249 tokens to be told what it
already knew, and gains nothing. The score still rises to 61.6 because the store
is small enough to stop costing steps (purchase rate back to 86%).

ALFWorld found batch induction consolidating a *false* belief into an
authoritative procedure that took a 43% task family to 0%. WebShop shows the
other failure mode of the same mechanism: consolidation into something **true but
vacuous**. Both are the induction step discarding specifics, which is what it is
for; the question the two runs raise together is whether the specifics were the
value all along.

## 4. Rule memory is instance-level again, and it still helped

`rule` writes the same shape of entry it wrote on ALFWorld — reactions keyed to
specific pages rather than transferable principles:

- *"the product page shows a 'Rapid Charger' or 'Fast Charging' option → …"*
- *"the product page shows selectable light count options (e.g., …) → …"*
- *"the search page shows no item matching the specific size constraint → …"*

On ALFWorld this content never beat the baseline and was significantly harmful
under `full` (46.0%, p = 0.043). Here `rule/full` is the second-strongest arm at
+6.0 points, with the smallest injected block of any arm that helps (294 tokens),
after batch induction cut the store from 31 entries to 7.

Instance-level rules transfer badly between ALFWorld's six procedures. WebShop's
100 evaluation tasks are all the same procedure — search, browse, select options,
buy — differing in product, so a rule about "the page offers a size option that
does not match" applies broadly. The content type did not change; the benchmark's
task structure did.

## 5. Reflection is the weakest content type on both benchmarks

`reflection` is the only arm that is below baseline on both metrics under both
policies: 28.0%/51.2 and 25.0%/52.8 against 29.0%/59.8. Its
success-given-purchase under `full` is 30.5%, the only value in the run below the
baseline's 33.7%, and it has the worst reward-given-purchase in the study (64.4).
It also has the highest writer bill under `full` (126 calls).

On ALFWorld it never reached significance in either direction. On WebShop it is
consistently, if not significantly, harmful. Two benchmarks is not a pattern, but
it is the one content type that has produced nothing positive anywhere.

## 6. Per-category breakdown

WebShop's grouping is the product department, which is a much weaker cut than
ALFWorld's six task families — two "beauty" tasks can need entirely different
action sequences, where "pick_heat_then_place" names a procedure. Read this as
exploratory.

| arm | policy | beauty (21) | electronics (20) | fashion (19) | garden (23) | grocery (17) |
|---|---|---|---|---|---|---|
| none | — | 7 | 4 | 8 | 4 | 6 |
| raw | minimal | 5 | 5 | 9 | 5 | 6 |
| reflection | minimal | 8 | 5 | 7 | 4 | 4 |
| rule | minimal | 6 | 6 | 8 | 3 | 8 |
| skill | minimal | 9 | 6 | 7 | 5 | 5 |
| raw | full | **10** | 3 | 9 | **7** | 8 |
| reflection | full | 5 | 4 | 7 | 4 | 5 |
| rule | full | **10** | 6 | 6 | 5 | 8 |
| skill | full | 8 | 5 | 5 | 3 | 7 |

`raw/full`'s +8 comes from `beauty` (+3) and `garden` (+3) while it *loses* a
point on `electronics`. No arm improves `electronics`, the hardest department for
the baseline (4/20).

## 6. Doubling the evolving experience helped nothing, and broke the best arm

Each arm resumed its own 50-episode store and evolved the *next* 50 tasks of the
same seeded permutation (`manifests/webshop_evolve_train_50to100_seed42.json`),
then re-ran the identical 100-task evaluation. This is the "evolve on 50 / 100"
axis of `plan.md`, run as a continuation rather than a fresh 100-episode run, so
the memory the first 50 produced is what the second 50 build on.

`p` is the paired test of each arm against **its own 50-episode self**, which is
the comparison that isolates the added experience.

| arm | policy | e50 | e100 | Δ | p | score 50→100 | store | survived/deleted/new |
|---|---|---|---|---|---|---|---|---|
| raw | minimal | 30.0% | 31.0% | +1.0 | 1.000 | 58.1 → 57.6 | 16 → 28 | 16/0/12 |
| reflection | minimal | 28.0% | 25.0% | −3.0 | 0.664 | 51.2 → 53.6 | 18 → 38 | 14/4/24 |
| rule | minimal | 31.0% | 25.0% | −6.0 | 0.210 | 57.0 → 56.2 | 31 → 52 | 25/6/27 |
| skill | minimal | 32.0% | 29.0% | −3.0 | 0.607 | 55.9 → 54.9 | 4 → 5 | 3/1/2 |
| **raw** | **full** | **37.0%** | **21.0%** | **−16.0** | **0.002** | 62.0 → 52.0 | 17 → 18 | 13/4/5 |
| reflection | full | 25.0% | 28.0% | +3.0 | 0.581 | 52.8 → 57.9 | 13 → 24 | 6/7/18 |
| rule | full | 35.0% | 31.0% | −4.0 | 0.424 | 58.8 → 56.4 | 7 → 3 | 0/7/3 |
| skill | full | 28.0% | 30.0% | +2.0 | 0.727 | 61.6 → 59.0 | 1 → 1 | 0/1/1 |

**Five of eight arms declined, and the only significant result in the entire
WebShop study is a 16-point collapse.** At evolve-50 every `minimal` arm sat at
or above the 29.0% baseline; at evolve-100 three of four are below it.

`raw/full` is the case worth understanding. It was the best arm in the study
(37.0%, +8.0 over baseline) and fell to 21.0%, *below* baseline, at p = 0.002 —
b/c = 21/5, so it lost 21 tasks it had been solving. Its store barely changed
size, 17 → 18, but a quarter of it turned over: 4 of the original trajectories
were deleted and 5 admitted from phase 2. `raw/minimal` — same content type, same
tasks, same new experience, no deletion mechanism — went 30.0 → 31.0 and kept all
16 original entries.

That contrast is suggestive but it is not a clean deletion story, and the table
says so: `reflection/full` deleted 7 of its 13 entries and *improved* by 3.0.
What separates the arms better is net direction — stores that shrank or churned
without replacement declined (`raw/full`, `rule/full` 7 → 3 with **zero**
survivors), stores that grew did not.

**Regression to the mean is at least as good an explanation for most rows.** The
two arms furthest above baseline at e50 (`raw/full` 37.0, `rule/full` 35.0) fell
the most; the two furthest below (`reflection/full` 25.0, `skill/full` 28.0)
rose. With one seed and no measured noise floor, that accounts for everything
here except `raw/full`, whose 16 points are too large for it.

### The axis is confounded: phase 2 was a worse 50 tasks

| arm | policy | phase-1 successes | phase-2 successes |
|---|---|---|---|
| raw | minimal | 16/50 | 12/50 |
| reflection | minimal | 15/50 | 10/50 |
| rule | minimal | 21/50 | 9/50 |
| skill | minimal | 18/50 | 12/50 |
| raw | full | 19/50 | **8/50** |
| reflection | full | 19/50 | 12/50 |
| rule | full | 15/50 | 10/50 |
| skill | full | 14/50 | 11/50 |

Every arm succeeded less often in phase 2 — 8 of 8, by ~35% on average. So this
column is not "more experience", it is "more experience, less of it successful",
which matters directly for `raw` and `skill` because they only write from
successful episodes (`raw`'s store grew by exactly its 12 phase-2 successes).

Two readings this run cannot separate: the phase-2 draw is simply harder, or the
accumulated memory is not helping during evolving either. Note the arms entered
phase 2 *with* memory and still did worse than they had from cold, which is not
what a working memory predicts. Distinguishing them needs a no-memory control run
over the phase-2 tasks, which `none` does not provide because it has no evolving
phase.

## Threats to validity

- **n = 100, single seed, and the effects are small.** Nothing here is
  significant at 0.05. The strongest result (`raw/full`, +8.0) is p = 0.096 and
  would need roughly 300 tasks to resolve at this effect size. Read the paired
  tests, not the deltas.
- **The noise floor was never measured on this benchmark.** Everything above is
  computed against a *single* no-memory run. The AppWorld study
  ([RESULTS_APPWORLD.md](RESULTS_APPWORLD.md)) later ran its baseline twice under
  identical conditions and got 20.0% and 25.0% — a ±5 point spread that
  disagreed on 23 of 100 tasks, and that erased every arm in that table,
  including two that looked like +5.0 against the first run and were +0.0 against
  the second. No claim here should be trusted above ~5 points until the same
  replicate is run for WebShop. It costs one baseline run.
- **The 15-step horizon confounds the content comparison** (§1). Every arm's
  benefit is partly eaten by timeouts, and the arms are affected unequally —
  `skill/minimal` loses 10 points of purchase rate, `rule/minimal` 3. Until the
  longer-horizon control is run, "content type X is better" cannot be separated
  from "content type X is cheaper to read".
- **The scaffold has no external reference.** Unlike ALFWorld, the baseline
  cannot be checked against an independently measured number, only against a
  published range. If the scaffold is systematically weak or strong, every arm
  moves with it.
- **One rollout per evolving task**, so `Episode.outcome()` is only ever
  `all_success` or `all_failure`; Reflection's `from_contrast` mode never fired
  and skill could not write from mixed outcomes.
- **Writer model = actor model.** Qwen3.5-9B writes the memory it later consumes,
  so content quality and consumption ability are confounded.
- **Product departments are a weak cluster key.** `MemoryConfig.cluster_key` is
  `task_type`, which here is the department. Batch induction therefore clusters
  over a grouping with little procedural coherence, which may be part of why
  consolidation produced a generic skill (§3).

## What to run next

1. **A second no-memory baseline run.** Now the highest-value measurement, and
   the cheapest: one run, ~25 minutes. On AppWorld the equivalent replicate moved
   the reference by 5 points and nullified the entire table. Until it exists,
   every delta above is uncalibrated.
2. **The 25-step horizon control.** The one confound that touches every number
   above. If the arms separate, §1's mechanism is confirmed and the content
   comparison becomes interpretable.
3. **≥ 3 seeds on `raw/full` and `rule/full`.** The two arms with lopsided
   discordant counts are the only candidates for a real effect.
4. **`equal_item_count`** to separate rule's small injected block (294 tokens)
   from its content, the same control ALFWorld's §5 called for.

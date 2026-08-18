# WebShop Memory Content study — Qwen3.5-9B

Run 2026-08-10, extended to 100 evolving episodes 2026-08-11 and to 150 on
2026-08-13. 50 / 100 / 150 evolving tasks (`train`, goal indices ≥ 1500), 100
evaluation tasks (`test`, goal indices < 500, disjoint), 1 rollout per task,
15-step horizon, seed 42. Full 1.18M-product corpus, 12,087 human-written
instructions. Model served locally by vLLM; the same model writes memory and
acts.

Sections 1–5 below describe the 50-episode run and are unchanged. Section 7
carries all three legs of the *diversity* axis (50 → 100 → 150 distinct tasks)
and revises the strongest claim the 100-episode leg made — see **"the collapse
did not hold"**. Section 8 adds the *repetition* axis (the same 50 tasks, three
times) and compares the two at equal episode budget. Section 9 adds the
*success-budget* axis — evolve until 100 episodes have succeeded, discarding
failures — and compares all four conditions.

Raw outputs, one directory per condition:
`/gpfs/radev/scratch/cohan/jw3278/memsys_results/webshop/{minimal,full}{,_e100,_e150,_x2,_x3,_ok100}/`.
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


| arm        | policy  | rate      | Δ        | score    | Δ    | b/c  | McNemar p | store | inj. tok | writer calls |     |
| ---------- | ------- | --------- | -------- | -------- | ---- | ---- | --------- | ----- | -------- | ------------ | --- |
| none       | —       | 29.0%     | —        | 59.8     | —    | —    | —         | —     | 0        | —            |     |
| raw        | minimal | 30.0%     | +1.0     | 58.1     | −1.7 | 8/9  | 1.000     | 16    | 1177     | —            |     |
| reflection | minimal | 28.0%     | −1.0     | 51.2     | −8.6 | 10/9 | 1.000     | 18    | 1117     | 58           |     |
| rule       | minimal | 31.0%     | +2.0     | 57.0     | −2.8 | 6/8  | 0.791     | 31    | 468      | 58           |     |
| skill      | minimal | 32.0%     | +3.0     | 55.9     | −3.9 | 6/9  | 0.607     | 4     | 969      | 20           |     |
| raw        | full    | **37.0%** | **+8.0** | **62.0** | +2.2 | 5/13 | 0.096     | 17    | 1261     |              | —   |
| reflection | full    | 25.0%     | −4.0     | 52.8     | −7.0 | 10/6 | 0.454     | 13    | 1004     | 126          |     |
| rule       | full    | 35.0%     | +6.0     | 58.8     | −1.0 | 6/12 | 0.238     | 7     | 294      | 111          |     |
| skill      | full    | 28.0%     | −1.0     | 61.6     | +1.8 | 7/6  | 1.000     | 1     | 249      | 51           |     |


**No arm reaches p < 0.05.** `raw/full` is the strongest at +8.0 points and
p = 0.096, and it is the only arm whose discordant counts are lopsided enough to
be suggestive (it lost 5 tasks and won 13). Everything else is churn.

This is the first thing to say plainly: on ALFWorld the same code produced
`raw` +17.0 (p = 0.009) and `skill/minimal` +21.0 (p = 0.0002). Here the largest
effect is half that size and does not clear significance at n = 100.

## 1. The near-null is two real effects cancelling

The aggregate hides the mechanism completely. Decomposing every episode into
*did the agent buy anything* and *how good was what it bought*:


| arm        | policy  | purchase rate | rate given purchase | reward given purchase | mean steps |
| ---------- | ------- | ------------- | ------------------- | --------------------- | ---------- |
| none       | —       | 86%           | 33.7%               | 69.6                  | 6.94       |
| raw        | minimal | 82%           | 36.6%               | 70.9                  | 6.55       |
| reflection | minimal | 77%           | 36.4%               | 66.5                  | 7.82       |
| rule       | minimal | 83%           | 37.3%               | 68.7                  | 6.98       |
| skill      | minimal | 76%           | **42.1%**           | **73.5**              | 8.28       |


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

### The same decomposition at 150 episodes, across both policies

At 50 episodes this trade-off was only demonstrated under `minimal`. At 150 it
holds for seven of eight arms under both:

| arm | policy | purchase rate | rate given purchase | reward given purchase | mean steps |
|---|---|---|---|---|---|
| none | — | 86% | 33.7% | 69.6 | 6.94 |
| raw | minimal | 85% | 35.3% | 67.1 | 6.42 |
| reflection | minimal | 73% | 35.6% | 70.5 | 7.67 |
| rule | minimal | 74% | 35.1% | 67.6 | 7.72 |
| skill | minimal | 78% | **38.5%** | 70.3 | 8.00 |
| raw | full | 82% | 37.8% | 70.4 | 6.98 |
| reflection | full | 79% | 31.6% | 65.7 | 7.58 |
| rule | full | 70% | **38.6%** | 68.7 | 8.04 |
| skill | full | **87%** | 35.6% | 68.0 | 6.95 |

Seven of eight arms shop better than the baseline and seven of eight finish less
often. `reflection/full` is the sole arm below baseline on quality (31.6%),
consistent with §5.

The two extremes make the mechanism unusually legible. **`rule/full` is the best
shopper in the run** (38.6% success given purchase) and the worst finisher (70%),
spending 8.04 of its 15 steps — net 27/100, below baseline. **`skill/full` is the
only arm that buys more often than the baseline** (87%) at essentially baseline
step cost (6.95), and it is the best arm at 31/100. Their stores are 10 entries
against 2, and their injected blocks 372 tokens against 538 — so this is not
about block size but about what the block makes the agent *do*. Memory that
provokes extra exploration converts a step budget into shopping quality it then
cannot cash in.

## 2. More write mechanism helped here — the opposite of ALFWorld

ALFWorld's headline was that `WritePolicy.minimal()` dominated `full()` for every
content type that uses an LLM writer, losing 7 to 15 points each. On WebShop the
sign flips:


| arm        | rate minimal → full    | score minimal → full   | purchase rate | mean steps  | writer calls |
| ---------- | ---------------------- | ---------------------- | ------------- | ----------- | ------------ |
| raw        | 30.0% → **37.0%** (+7) | 58.1 → 62.0 (**+3.9**) | 82% → 86%     | 6.55 → 6.04 | none → none  |
| reflection | 28.0% → 25.0% (−3)     | 51.2 → 52.8 (**+1.6**) | 77% → 82%     | 7.82 → 7.26 | 58 → 126     |
| rule       | 31.0% → **35.0%** (+4) | 57.0 → 58.8 (**+1.8**) | 83% → 84%     | 6.98 → 6.70 | 58 → 111     |
| skill      | 32.0% → 28.0% (−4)     | 55.9 → 61.6 (**+5.7**) | 76% → 86%     | 8.28 → 6.62 | 20 → 51      |


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


| arm        | policy  | beauty (21) | electronics (20) | fashion (19) | garden (23) | grocery (17) |
| ---------- | ------- | ----------- | ---------------- | ------------ | ----------- | ------------ |
| none       | —       | 7           | 4                | 8            | 4           | 6            |
| raw        | minimal | 5           | 5                | 9            | 5           | 6            |
| reflection | minimal | 8           | 5                | 7            | 4           | 4            |
| rule       | minimal | 6           | 6                | 8            | 3           | 8            |
| skill      | minimal | 9           | 6                | 7            | 5           | 5            |
| raw        | full    | **10**      | 3                | 9            | **7**       | 8            |
| reflection | full    | 5           | 4                | 7            | 4           | 5            |
| rule       | full    | **10**      | 6                | 6            | 5           | 8            |
| skill      | full    | 8           | 5                | 5            | 3           | 7            |


`raw/full`'s +8 comes from `beauty` (+3) and `garden` (+3) while it *loses* a
point on `electronics`. No arm improves `electronics`, the hardest department for
the baseline (4/20).

## 7. More evolving experience helped nothing — and the one "significant" result did not hold

Each arm resumed its own 50-episode store and evolved the *next* 50 tasks of the
same seeded permutation (`manifests/webshop_evolve_train_50to100_seed42.json`),
then re-ran the identical 100-task evaluation. This is the "evolve on 50 / 100"
axis of `plan.md`, run as a continuation rather than a fresh 100-episode run, so
the memory the first 50 produced is what the second 50 build on.

`p` is the paired test of each arm against **its own 50-episode self**, which is
the comparison that isolates the added experience.


| arm        | policy   | e50       | e100      | Δ         | p         | score 50→100 | store   | survived/deleted/new |
| ---------- | -------- | --------- | --------- | --------- | --------- | ------------ | ------- | -------------------- |
| raw        | minimal  | 30.0%     | 31.0%     | +1.0      | 1.000     | 58.1 → 57.6  | 16 → 28 | 16/0/12              |
| reflection | minimal  | 28.0%     | 25.0%     | −3.0      | 0.664     | 51.2 → 53.6  | 18 → 38 | 14/4/24              |
| rule       | minimal  | 31.0%     | 25.0%     | −6.0      | 0.210     | 57.0 → 56.2  | 31 → 52 | 25/6/27              |
| skill      | minimal  | 32.0%     | 29.0%     | −3.0      | 0.607     | 55.9 → 54.9  | 4 → 5   | 3/1/2                |
| **raw**    | **full** | **37.0%** | **21.0%** | **−16.0** | **0.002** | 62.0 → 52.0  | 17 → 18 | 13/4/5               |
| reflection | full     | 25.0%     | 28.0%     | +3.0      | 0.581     | 52.8 → 57.9  | 13 → 24 | 6/7/18               |
| rule       | full     | 35.0%     | 31.0%     | −4.0      | 0.424     | 58.8 → 56.4  | 7 → 3   | 0/7/3                |
| skill      | full     | 28.0%     | 30.0%     | +2.0      | 0.727     | 61.6 → 59.0  | 1 → 1   | 0/1/1                |


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

### The third leg: the collapse did not hold

A third leg carried every arm to 150 evolving episodes, resuming each 100-episode
store over `manifests/webshop_evolve_train_100to150_seed42.json` and re-running
the identical evaluation. **`raw/full` went back up.**

| arm | policy | e50 | e100 | e150 | score 50 → 100 → 150 | store 50 → 100 → 150 |
|---|---|---|---|---|---|---|
| raw | minimal | 30.0% | 31.0% | 30.0% | 58.1 → 57.6 → 57.0 | 16 → 28 → 40 |
| reflection | minimal | 28.0% | 25.0% | 26.0% | 51.2 → 53.6 → 51.4 | 18 → 38 → 68 |
| rule | minimal | 31.0% | 25.0% | 26.0% | 57.0 → 56.2 → **50.1** | 31 → 52 → 79 |
| skill | minimal | 32.0% | 29.0% | 30.0% | 55.9 → 54.9 → 54.9 | 4 → 5 → 6 |
| **raw** | **full** | **37.0%** | **21.0%** | **31.0%** | 62.0 → 52.0 → 57.7 | 17 → 18 → 29 |
| reflection | full | 25.0% | 28.0% | 25.0% | 52.8 → 57.9 → 51.9 | 13 → 24 → 35 |
| rule | full | 35.0% | 31.0% | 27.0% | 58.8 → 56.4 → **48.1** | 7 → 3 → 10 |
| skill | full | 28.0% | 30.0% | **31.0%** | 61.6 → 59.0 → 59.2 | 1 → 1 → 2 |

`raw/full` oscillates **37 → 21 → 31** while its store only ever grows,
17 → 18 → 29. The section above spent four paragraphs on whether the e100 drop
was a deletion effect — 4 entries deleted, 5 admitted, against `raw/minimal`
which kept all 16 and did not fall. **That reading is now much harder to
sustain.** A deletion that destroyed 16 points of performance does not undo
itself in the next 50 episodes without the deleted entries coming back. The
simpler account is the one this document already offered for the other seven
rows and then exempted `raw/full` from: it is noise, and the exemption was
granted only because 16 points seemed too large. Across three legs `raw/full`'s
spread is 16 points with no monotone driver, which is better evidence about the
benchmark's variance than about deletion.

**At 150, no arm is more than 4 points from baseline in either direction**, and
the largest paired p-value in the table is 0.48 — everything is null:

| arm | policy | rate | Δ | score | b/c | McNemar p | store | inj. tok |
|---|---|---|---|---|---|---|---|---|
| none | — | 29.0% | — | 59.8 | — | — | — | 0 |
| raw | minimal | 30.0% | +1.0 | 57.0 | 7/8 | 1.000 | 40 | 1129 |
| reflection | minimal | 26.0% | −3.0 | 51.4 | 10/7 | 0.629 | 68 | 1102 |
| rule | minimal | 26.0% | −3.0 | 50.1 | 10/7 | 0.629 | 79 | 475 |
| skill | minimal | 30.0% | +1.0 | 54.9 | 8/9 | 1.000 | 6 | 1419 |
| raw | full | 31.0% | +2.0 | 57.7 | 7/9 | 0.804 | 29 | 1142 |
| reflection | full | 25.0% | −4.0 | 51.9 | 11/7 | 0.481 | 35 | 1015 |
| rule | full | 27.0% | −2.0 | 48.1 | 9/7 | 0.804 | 10 | 372 |
| skill | full | **31.0%** | +2.0 | **59.2** | 8/10 | 0.815 | 2 | 538 |

Three things the third leg establishes that two legs could not:

1. **Every arm's graded score is now below baseline** — all eight, 48.1 to 59.2
   against 59.8. At e50 two arms were above it (`raw/full` 62.0, `skill/full`
   61.6). The score advantage §2 reported for `full` has not survived; what
   survives is that `full` still beats `minimal` on score for three of four
   content types (skill +4.3, raw +0.7, reflection +0.5, rule −2.0).
2. **`rule` decays monotonically on score under both policies** — 57.0 → 56.2 →
   50.1 and 58.8 → 56.4 → 48.1 — the only arm with a consistent direction across
   three legs on either metric. Its `minimal` store grew 31 → 52 → 79. This is
   the one trend in the WebShop study that more episodes made clearer rather
   than muddier, and it points the same way as §3: accumulation without useful
   consolidation costs the graded reward even when it leaves the strict rate
   alone.
3. **`skill/full` is now the best arm**, at 31.0% and 59.2 — the only score
   within a point of baseline — on a store of **2 entries** and 538 injected
   tokens. §3 called `skill/full`'s single consolidated skill "true but
   vacuous"; at 150 it holds two, and it is the only configuration that is at or
   above baseline on both metrics at every leg it was measured.

### The axis is confounded: phase 2 was a worse 50 tasks


| arm        | policy  | phase-1 successes | phase-2 successes |
| ---------- | ------- | ----------------- | ----------------- |
| raw        | minimal | 16/50             | 12/50             |
| reflection | minimal | 15/50             | 10/50             |
| rule       | minimal | 21/50             | 9/50              |
| skill      | minimal | 18/50             | 12/50             |
| raw        | full    | 19/50             | **8/50**          |
| reflection | full    | 19/50             | 12/50             |
| rule       | full    | 15/50             | 10/50             |
| skill      | full    | 14/50             | 11/50             |


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

## 8. Repetition vs diversity: 150 episodes spent either way is the same

§7 spent 150 evolving episodes on 150 *distinct* tasks. This section spends the
same 150 on the *same 50 tasks, three times* — each epoch resumes the store the
previous one left behind and re-runs the identical frozen 100-task evaluation.
Everything that differs between epoch k and k+1 is memory state, not the task
draw. That makes the two chains a controlled pair: equal episode budget, equal
evaluation, and only task diversity varying.

On epoch 2 the nearest neighbour of a task is usually the agent's own epoch-1
memory of *that same task*, which for `raw` is a near-verbatim replay of its own
trajectory. That is the phenomenon under test.

| arm | policy | epoch 1 | epoch 2 | epoch 3 | score 1 → 2 → 3 | store 1 → 2 → 3 |
|---|---|---|---|---|---|---|
| raw | minimal | 30.0% | 33.0% | 29.0% | 58.1 → 57.6 → 58.4 | 16 → 23 → 27 |
| reflection | minimal | 28.0% | 27.0% | 27.0% | 51.2 → 54.3 → 54.6 | 18 → 47 → **78** |
| rule | minimal | 31.0% | 31.0% | 27.0% | 57.0 → 56.8 → 54.2 | 31 → 39 → 45 |
| skill | minimal | 32.0% | 32.0% | 30.0% | 55.9 → 56.1 → 54.7 | 4 → 5 → 6 |
| raw | full | **37.0%** | 30.0% | **26.0%** | 62.0 → 57.6 → **51.4** | 17 → 22 → 27 |
| reflection | full | 25.0% | 30.0% | 25.0% | 52.8 → 57.5 → 51.0 | 13 → 32 → 33 |
| rule | full | 35.0% | 24.0% | 25.0% | 58.8 → 52.4 → 52.8 | 7 → 2 → 6 |
| skill | full | 28.0% | 28.0% | 29.0% | 61.6 → 57.4 → 55.3 | 1 → 2 → 3 |

### The controlled comparison

Both chains end at 150 episodes and evaluate the same 100 tasks, so the
comparison is paired arm by arm:

| arm | policy | 50 tasks × 3 | 150 distinct | diff | b/c | SD |
|---|---|---|---|---|---|---|
| raw | minimal | 29 | 30 | +1 | 7/8 | 3.9 |
| reflection | minimal | 27 | 26 | −1 | 9/8 | 4.1 |
| rule | minimal | 27 | 26 | −1 | 10/9 | 4.4 |
| skill | minimal | 30 | 30 | 0 | 7/7 | 3.7 |
| raw | full | 26 | 31 | +5 | 5/10 | 3.9 |
| reflection | full | 25 | 25 | 0 | 5/5 | 3.2 |
| rule | full | 25 | 27 | +2 | 7/9 | 4.0 |
| skill | full | 29 | 31 | +2 | 4/6 | 3.2 |
| **mean** | | **27.25** | **28.25** | **+1.0** | | |

**Not one arm differs by more than its own paired SD.** Mean rate 27.25 against
28.25; mean graded score 54.0 against 53.8. **At a fixed episode budget, on this
benchmark, it does not matter whether the episodes are 150 different tasks or 50
tasks seen three times.** Both land about 1.5 points below the 29.0% no-memory
baseline and about 6 points below its score.

That is a stronger negative than either chain alone. §7 could only say more
episodes did not help; together the two say the *composition* of those episodes
does not matter either — which is what one would expect if the evolving phase is
contributing little on WebShop regardless of what it is fed.

### Two things the repetition axis shows that the diversity axis could not

**`raw/full` decays monotonically under repetition: 37 → 30 → 26**, and on score
too, 62.0 → 57.6 → 51.4. Both metrics, three points, one direction. Compare §7,
where the same arm on new tasks went 37 → 21 → 31 with no direction at all. The
contrast is informative about §7's collapse: on the diversity axis `raw/full`
oscillates over a 16-point range, and on the repetition axis it slides steadily.
Only the second looks like a mechanism, and the mechanism it suggests is
straightforward — `raw` stores trajectories, re-seeing a task retrieves the
agent's own prior attempt at it, and re-reading your own mediocre attempt is not
useful. Its store grows 17 → 22 → 27 on 50 tasks it has already solved or failed.

**Reflection accumulates *faster* on repeated tasks than on new ones.** Its
`minimal` store reaches **78 entries from 50 distinct tasks seen three times**,
against **68 from 150 distinct tasks seen once**. Re-reflecting on a task the
agent has already reflected on produces a new entry more often than reflecting on
a task it has never seen. The dedup and merge checks are evidently not keyed on
anything that recognises "I have already written this task up" — the second and
third passes differ in trajectory detail, and that is enough to get past them.
This is the clearest evidence in the study that reflection's growth is not
tracking new information.

`skill` is the counter-example on both counts, and the same one §7 found: 4 → 5 →
6 entries under `minimal`, 1 → 2 → 3 under `full`, and `skill/full` is the only
arm in either chain whose rate does not decline (28 → 28 → 29). Whatever its
grounding and dedup checks are keyed on, they do recognise a repeat.

## 9. A memory built from 100 successes, with failures discarded

§7 and §8 both hold the number of *attempts* fixed and let the amount of
successful experience float. This section inverts that: keep evolving until
**100 episodes have succeeded**, and let only those 100 reach the memory system.
Failed episodes are still attempted — the agent runs them, and they still shape
what it retrieves next — but they are discarded before `observe`, so they produce
no writer call, no utility signal, and do not advance the induction cadence.

It cost 335 to 417 tasks per arm to buy 100 successes (24.0% to 29.9% evolving
success rate), from a 600-task pool; no arm exhausted it.

| arm | policy | tasks | succ % | rate | Δ | score | b/c | McNemar p | store | inj. tok | writer calls |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none | — | — | — | 29.0% | — | 59.8 | — | — | — | 0 | — |
| raw | minimal | 360 | 27.8 | **32.0%** | +3.0 | 55.5 | 6/9 | 0.607 | **100** | 1180 | — |
| reflection | minimal | 356 | 28.1 | 26.0% | −3.0 | 53.0 | 10/7 | 0.629 | 55 | 984 | 126 |
| rule | minimal | 335 | 29.9 | 29.0% | +0.0 | 50.1 | 10/10 | 1.000 | 36 | 495 | 143 |
| skill | minimal | 357 | 28.0 | **24.0%** | −5.0 | 52.8 | 11/6 | 0.332 | **38** | 1380 | 103 |
| raw | full | 343 | 29.2 | 27.0% | −2.0 | 56.1 | 9/7 | 0.804 | **100** | 1181 | — |
| reflection | full | 394 | 25.4 | 26.0% | −3.0 | 52.1 | 11/8 | 0.648 | 66 | 987 | 282 |
| rule | full | 417 | 24.0 | 25.0% | −4.0 | 48.8 | 10/6 | 0.454 | 9 | 502 | 323 |
| skill | full | 372 | 26.9 | 26.0% | −3.0 | 54.7 | 11/8 | 0.648 | **48** | 1399 | 227 |

**Nothing reaches significance, and the mean is the lowest of any condition
tried.** Across the four ways of spending an evolving budget, all evaluated on
the same frozen 100 tasks:

| arm | policy | 50 attempts | 150 attempts | 50×3 epochs | **100 successes** |
|---|---|---|---|---|---|
| raw | minimal | 30 | 30 | 29 | 32 |
| reflection | minimal | 28 | 26 | 27 | 26 |
| rule | minimal | 31 | 26 | 27 | 29 |
| skill | minimal | 32 | 30 | 30 | **24** |
| raw | full | **37** | 31 | 26 | 27 |
| reflection | full | 25 | 25 | 25 | 26 |
| rule | full | **35** | 27 | 25 | 25 |
| skill | full | 28 | 31 | 29 | 26 |
| **mean** | | **30.75** | 28.25 | 27.25 | **26.88** |

The condition with the *most* successful experience and the *most* distinct
tasks — 100 successes drawn from ~350 tasks — is the worst of the four, and the
only one where the mean sits clearly below the 29.0% no-memory baseline. The
cheapest condition, 50 attempts, is the only one above it.

Three arms fell far enough from their 50-attempt selves to be worth naming:
`raw/full` 37 → 27 (b/c 4/14, SD 4.2), `rule/full` 35 → 25 (3/13, SD 4.0),
`skill/minimal` 32 → 24 (4/12, SD 4.0) — all about 2σ. But the first two were
the two highest values in the 50-attempt run, so §7's regression-to-the-mean
caveat applies to them exactly as before. `skill/minimal` is the one that is
not explained that way, and it has a mechanism.

### Restricting writes to successes made `skill` write *more*, not less

`skill` has been the study's minimal-store arm everywhere: 4 entries at 50
attempts, 6 after three epochs, 1–3 under `full`. §8 read that as its grounding
and dedup checks being unusually selective. **Here it writes 38 entries under
`minimal` and 48 under `full`** — an order of magnitude more, in the one
condition that fed it *only* successes.

Both readings are right, and together they say something sharper. `skill` can
only write from a successful episode, so its store size was never a measure of
how selective it is — it was a measure of **how few successes it was being
offered**. Fifty attempts yield ~15 successes; three epochs over the same 50
tasks yield repeats, which dedup correctly rejects (4 → 5 → 6 in §8). One
hundred *distinct* successes yield 38 skills, because they are genuinely
different. The dedup check was doing its job on repeats and was never the
binding constraint on novel material.

And the result got worse: **24/100 is `skill/minimal`'s lowest score in any
condition** (32, 30, 30, 24), on its largest store and its largest injected block
(1380 tokens). §3 argued `skill`'s value was a handful of tactical recovery
procedures; 38 of them is not a bigger version of that.

### `full`'s deletion machinery went silent

`raw`'s store is **exactly 100 under both policies** — one entry per successful
episode, nothing pruned. In every other run `raw/full` is smaller than `raw/minimal`
because `full` deletes (17 at 50 attempts, 29 at 150, 27 across epochs, and the
§7 table records 4 entries deleted between legs). Here it deleted nothing.

This is the designed-in consequence of discarding failures, and it is worth
stating plainly because it limits what this section can be compared against:
`full`'s utility-based deletion demotes entries that get retrieved into episodes
that then **fail**. With no failed episode ever reaching `observe`, that signal
does not exist, and the mechanism has nothing to act on. `rule/full` still
shrank to 9 entries and `skill/full` to 48, because batch induction and
dedup-on-write are independent of it — but the deletion half of `full` was
effectively disabled for the whole of this section.

So this run answers "does memory built purely from successes help?" (no) but it
does **not** cleanly answer "is `full` better than `minimal` when failures are
discarded?", because discarding failures removes one of `full`'s mechanisms. A
variant that keeps the utility signal from failures while still refusing to write
from them would separate the two; it has not been run.

## Threats to validity

- **n = 100, single seed, and the effects are small.** Nothing here is
significant at 0.05 except the `raw/full` e100 drop, and §7 shows that one
reversed on the next leg. The strongest positive result (`raw/full` at e50,
+8.0) is p = 0.096 and would need roughly 300 tasks to resolve at this effect
size. Read the paired tests, not the deltas.
- **The paired noise floor is ≈ 4 points, which swallows every result at 150.**
No true replicate exists here — unlike SpreadsheetBench, no arm ever held its
memory fixed across two legs (`skill/full` came closest and still moved 249 →
310 → 538 injected tokens), so this is not a measured run-to-run σ. But the
discordant counts bound it: at 150 the eight arms disagree with the baseline on
15 to 18 tasks, so the SD of the paired difference is √(15…18) = **3.9 to 4.2
points**. Every delta in the 150 table is ≤ 4.0. A dedicated repeat of the
`none` baseline is still the missing measurement, and on this benchmark it is
cheap.
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

1. **A second no-memory baseline run.** Still the highest-value measurement, and
  the cheapest: one run, ~25 minutes. On AppWorld the equivalent replicate moved
   the reference by 5 points and nullified the entire table. Until it exists,
   every delta above is uncalibrated — and §7 has now shown this benchmark
   producing a 16-point swing on one arm with no mechanism behind it.
2. **The 25-step horizon control.** The one confound that touches every number
  above, and the third leg strengthened the case for it: §1's trade-off now
   holds for seven of eight arms under both policies, with `rule/full` and
   `skill/full` at opposite extremes of the same step budget. If the arms
   separate at 25 steps, the mechanism is confirmed and the content comparison
   becomes interpretable.
3. **Stop spending compute on more evolving episodes, in any composition.**
  Four conditions — 50 attempts, 150 attempts, 50 tasks × 3 epochs, and 100
   successes from ~350 tasks — rank 30.75, 28.25, 27.25, 26.88 on mean rate,
   against a 29.0% baseline. The cheapest is the best and the most expensive is
   the worst. The horizon control and the baseline replicate are where the
   remaining uncertainty actually lives.
4. **The failure-utility variant.** §9 discarded failed episodes entirely, which
  also removed the negative-utility signal `full`'s deletion depends on — its
   `raw` store came out at exactly 100, unpruned. Keeping that signal while
   still refusing to write from failures would separate "failures are bad
   training data" from "failures are needed to prune", and it is a one-flag
   change.
5. **≥ 3 seeds on** `skill/full`**.** It replaces `raw/full` as the candidate for
  a real effect: it is the only arm at or above baseline on both metrics at
   every leg, and it does it on a 2-entry store. `raw/full`, the previous
   candidate, went 37 → 21 → 31 and is now better read as a variance
   demonstration.
6. `equal_item_count` to separate rule's small injected block (372 tokens at
  150) from its content, the same control ALFWorld's §5 called for.


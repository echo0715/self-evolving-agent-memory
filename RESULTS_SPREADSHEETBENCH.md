# SpreadsheetBench — Memory Content results

Qwen3.5-9B, 50 / 100 / 150 evolving episodes plus two outcome budgets — 100
failures and 100 successes — 100 evaluation tasks, seed 42.
Adapter and protocol: [RUN_SPREADSHEETBENCH.md](RUN_SPREADSHEETBENCH.md).

---

## The noise floor, measured three times: σ ≈ 4.7 points

Unlike AppWorld's, this floor needed no dedicated baseline repeat — the
continuation runs produced a **true replicate**, and then produced it twice
more.

`skill / full` stopped writing after episode 50. Across all three legs its store
holds the same three entries — same ids, and the entry *text* is character-for-
character identical; only bookkeeping counters (`support`, `n_retrieved`,
`updated_at_step`) differ. The block the agent actually sees is the same in all
three: `mean_injected_tokens = 771.0` exactly, three times. So its three
evaluation passes are the same memory, the same scaffold and the same 100 tasks,
differing only in sampling:


|                            | eval success | vs previous, discordant (b/c) |
| -------------------------- | ------------ | ----------------------------- |
| `skill / full`, 50 evolve  | 27/100       | —                             |
| `skill / full`, 100 evolve | 27/100       | 11/11                         |
| `skill / full`, 150 evolve | 27/100       | 11/11                         |


And the third pairing, 50 vs 150, is also **11/11**. All three pairwise
comparisons of three independent runs give the identical discordance of 22.

**The aggregate landed on 27 all three times; 22 of 100 individual task outcomes
flipped in every pairing.** Under the null, the run-to-run difference in success
count has SD = √22 = **4.7 points**. A ±5 point swing between two identical
configurations is 1σ. 2σ is ±9.4 points.

That three independent replicates agree to the decimal is the strongest version
of this measurement the study has. It also matches AppWorld's independently
measured ±5 floor almost exactly.

### The aggregate is stable; the task set underneath it is not

Of the 100 evaluation tasks, across the three identical-memory replicates:


| solved by | tasks |
| --------- | ----- |
| all 3     | 11    |
| exactly 2 | 15    |
| exactly 1 | 18    |
| none      | 56    |


**Only 11 tasks are reliably solved. 33 are coin flips.** A configuration
scoring 27/100 is not solving a stable set of 27 tasks — it is solving 11, plus
roughly half of a pool of 33 it can sometimes reach. Every per-task claim in
this document, and any attempt to characterise *which* tasks memory helps with,
has to survive that.

### Every result in units of that floor


| arm        | policy  | evolve | success | delta | σ        |
| ---------- | ------- | ------ | ------- | ----- | -------- |
| raw        | full    | 150    | 27/100  | +14   | **+3.0** |
| skill      | full    | 150    | 27/100  | +14   | **+3.0** |
| skill      | full    | 100    | 27/100  | +14   | **+3.0** |
| skill      | full    | 50     | 27/100  | +14   | **+3.0** |
| reflection | minimal | 50     | 27/100  | +14   | **+3.0** |
| skill      | minimal | 150    | 26/100  | +13   | **+2.8** |
| rule       | full    | 50     | 26/100  | +13   | **+2.8** |
| reflection | full    | 150    | 25/100  | +12   | **+2.6** |
| reflection | minimal | 150    | 24/100  | +11   | **+2.3** |
| reflection | full    | 50     | 23/100  | +10   | **+2.1** |
| rule       | full    | 100    | 23/100  | +10   | **+2.1** |
| rule       | full    | 150    | 22/100  | +9    | +1.9     |
| raw        | full    | 100    | 21/100  | +8    | +1.7     |
| skill      | minimal | 50     | 20/100  | +7    | +1.5     |
| skill      | minimal | 100    | 20/100  | +7    | +1.5     |
| raw        | minimal | 50     | 19/100  | +6    | +1.3     |
| raw        | minimal | 100    | 19/100  | +6    | +1.3     |
| reflection | full    | 100    | 19/100  | +6    | +1.3     |
| raw        | full    | 50     | 18/100  | +5    | +1.1     |
| raw        | minimal | 150    | 18/100  | +5    | +1.1     |
| reflection | minimal | 100    | 18/100  | +5    | +1.1     |
| rule       | minimal | 100    | 18/100  | +5    | +1.1     |
| rule       | minimal | 150    | 18/100  | +5    | +1.1     |
| rule       | minimal | 50     | 17/100  | +4    | +0.9     |


**Eleven of twenty-four configurations clear 2σ** (delta ≥ +10). Everything
below is indistinguishable from a lucky draw against this baseline. The σ
figures in that column are all measured against a single baseline draw of
13/100 — which the failure-budget leg below has since shown to be the lowest of
three, so read the next subsection before quoting any of them.

What survives, stated conservatively: **memory helps on this benchmark, by
roughly 12–14 points at the top end. The ordering among memory types still does
not survive.** At 150 episodes the top four are raw, skill, skill and reflection
— spanning three of the four content types, separated by 2 points, in a study
whose noise floor is 4.7.

### The baseline, drawn three times — and 13/100 was the low draw

The failure-budget leg ("Budgeting by failures", below) produced two more
no-memory draws for free. `skill` writes nothing at all from failed episodes, so
in both of its `fail100` runs the store was empty at evaluation time and every
one of the 100 evaluation tasks was run with `injected_memory_tokens = 0`. The
agent's system prompt omits the memory section entirely when the block is empty
(`build_system_prompt`), so those two passes are **byte-identical in prompt to
the** `none` **arm** and differ from it only in sampling:


| run                                      | eval success |
| ---------------------------------------- | ------------ |
| `none` (the study's baseline)            | 13/100       |
| `skill / full`, fail100 (empty store)    | 16/100       |
| `skill / minimal`, fail100 (empty store) | 20/100       |


Mean 16.3, sample SD 3.5 — consistent with the σ ≈ 4.7 measured above, and aaieldf **7-point spread between two runs of the same memory-free configuration**.

This is the run item 1 of "What to run next" asked for, and it lands the way
that item warned it might: **13/100 was the low draw, and every delta in this
document is therefore ~3 points optimistic.** Re-centred on 16.3, the eleven
configurations that clear 2σ become roughly three, and the honest summary of the
study weakens to *the top end of the memory arms sits about 10 points above a
no-memory agent, and nothing separates the content types*.

Three draws is still a small sample for an SD; the σ ≈ 4.7 from the discordance
count is the better-founded number and the two agree, which is why nothing below
is re-scaled. The table above is left as it was measured — against 13 — with
this correction attached rather than folded in.

---



## What was run


|                  |                                                                         |
| ---------------- | ----------------------------------------------------------------------- |
| model            | `Qwen/Qwen3.5-9B` (non-thinking), vLLM                                  |
| data             | `all_data_912_v0.1`, 3 test cases per task                              |
| evolving (50)    | SkillOpt id split `train`, positions [0, 50) of the seed-42 permutation |
| evolving (100)   | continued: positions [50, 100), spilling into `val` (29 train + 21 val) |
| evolving (150)   | continued: positions [100, 150), spilling into `val` *and* `test`       |
| evaluation       | the same 100 tasks from `test` for every run                            |
| agent            | 30 turns max, fenced-block `bash` / `write_file` protocol               |
| injection budget | 2500 tokens                                                             |


A task counts as solved only if **all three test cases pass**. The agent sees
case 1; its `solution.py` is re-executed against cases 2 and 3.

Each leg *resumes* the previous leg's `store.jsonl` per arm and carries the
Evolver's step counter, so the legs extend rather than repeat, and `full`'s
every-25-episode batch induction fires where an uninterrupted 150-episode run
would have fired it.

Two caveats specific to the third leg:

- **It draws from** `test`**.** `train`+`val` holds only 117 selectable tasks once
the evaluation set's source workbooks are excluded, and 100 were already
spent, so positions [100, 150) take 33 of their 50 from the same split the
evaluation set is drawn from. The builder enforces that the evolving tasks
share no task id **and no source workbook** with the frozen eval set, and
refuses to write the manifest otherwise; this was re-verified before the run
(zero id overlap with eval, and zero with either earlier leg). The evaluation
set is untouched — but this leg's disjointness rests on the group filter,
where the first two legs got it from the split boundary.
- `reflection / full` **and** `rule / full` **ran on a 65536-token server**, the
other six arms on 32768. See "Batch induction outgrows the context window"
below. This cannot affect comparability: no call in the 50- or 100-episode
legs ever reached 32768 (checked across every log), so those legs would return
identical results on either server.



## Results



### `WritePolicy.minimal()` — append + merge only


| arm        | 50 evolve  |          | 100 evolve          |       | 150 evolve |       | store 50→100→150   |
| ---------- | ---------- | -------- | ------------------- | ----- | ---------- | ----- | ------------------ |
|            | success    | score    | success             | score | success    | score |                    |
| none       | 13/100     | 18.0     | *(baseline reused)* |       |            |       | —                  |
| raw        | 19/100     | 23.0     | 19/100              | 24.3  | 18/100     | 24.7  | 14 → 21 → 33       |
| reflection | **27/100** | **34.3** | 18/100              | 24.0  | 24/100     | 32.0  | 47 → 105 → **156** |
| rule       | 17/100     | 21.7     | 18/100              | 23.0  | 18/100     | 24.7  | 32 → 61 → 85       |
| skill      | 20/100     | 24.3     | 20/100              | 28.0  | **26/100** | 31.3  | 5 → 7 → 8          |




### `WritePolicy.full()` — every mechanism on


| arm        | 50 evolve  |          | 100 evolve          |          | 150 evolve |          | store 50→100→150 |
| ---------- | ---------- | -------- | ------------------- | -------- | ---------- | -------- | ---------------- |
|            | success    | score    | success             | score    | success    | score    |                  |
| none       | 13/100     | 18.0     | *(baseline reused)* |          |            |          | —                |
| raw        | 18/100     | 24.7     | 21/100              | 26.7     | **27/100** | **33.7** | 8 → 13 → 24      |
| reflection | 23/100     | 29.3     | 19/100              | 24.0     | 25/100     | 30.0     | 30 → 43 → 85     |
| rule       | 26/100     | 29.7     | 23/100              | 29.0     | 22/100     | 25.7     | 17 → 22 → 29     |
| skill      | **27/100** | **32.7** | **27/100**          | **32.7** | **27/100** | 31.7     | 3 → 3 → **3**    |


`score` = mean fraction of test cases passed, ×100.

---



## More episodes: still no trend that clears the floor


| arm / policy         | 50  | 100 | 150    | 50→150 | σ        |
| -------------------- | --- | --- | ------ | ------ | -------- |
| raw / minimal        | 19  | 19  | 18     | −1     | −0.2     |
| raw / full           | 18  | 21  | **27** | **+9** | **+1.9** |
| reflection / minimal | 27  | 18  | 24     | −3     | −0.6     |
| reflection / full    | 23  | 19  | 25     | +2     | +0.4     |
| rule / minimal       | 17  | 18  | 18     | +1     | +0.2     |
| rule / full          | 26  | 23  | 22     | −4     | −0.9     |
| skill / minimal      | 20  | 20  | **26** | **+6** | **+1.3** |
| skill / full         | 27  | 27  | 27     | 0      | 0.0      |


The 100-episode leg's headline — that fifty more episodes bought nothing — does
**not** simply extend. Two arms moved at 150:

- `raw / full` **climbed 18 → 21 → 27**, the only arm monotone across all three
legs, +9 over its own 50-episode run. At 1.9σ this is suggestive, not
established; but it is the single cleanest trend in the study, and it belongs
to the *least* structured content type. Its store grew 8 → 13 → 24.
- `skill / minimal` **jumped 20 → 20 → 26** (+6, 1.3σ) on a store that grew by
one entry, 7 → 8.

Set against that, `reflection / minimal` went 27 → 18 → 24 and `rule / full`
went 26 → 23 → 22. Neither excursion reaches 2σ in either direction. The honest
reading of all eight trajectories: **three legs of evidence still do not
establish that more evolving episodes help, but they no longer support the flat
claim that more episodes do nothing.** `raw / full` is the case to settle, and
a repeat of that arm alone would settle it.

Two mechanisms remain visible, pulling in opposite directions.

### Reflection accumulates faster than the budget can carry


|                      | store 50 | 100 | 150     | injected tokens    | success      |
| -------------------- | -------- | --- | ------- | ------------------ | ------------ |
| reflection / minimal | 47       | 105 | **156** | 1024 → 1020 → 1032 | 27 → 18 → 24 |
| reflection / full    | 30       | 43  | 85      | 997 → 1052 → 1066  | 23 → 19 → 25 |


The store more than tripled under `minimal` while the injected block stayed
within 12 tokens of where it started — it is capped at 2500 tokens and `pack()`
fills by score, so each new entry competes for a fixed slot count and the
marginal entry displaces an older one that may have been better.

At 100 episodes this looked like a clean story: store grew, score fell, both
policies. **The 150 leg breaks it.** Both reflection arms recovered — minimal
18 → 24, full 19 → 25 — while their stores kept growing, 105 → 156 and 43 → 85.
So the 100-episode dip was, in all likelihood, the ±5 noise floor appearing
twice rather than a crowding effect. The crowding mechanism is real and visible
in the token counts; the evidence that it *costs accuracy* did not survive a
third leg.

### Skill saturates and stops writing


|                                | write calls | landed              | store |
| ------------------------------ | ----------- | ------------------- | ----- |
| skill / full, episodes 0–50    | 12          | APPEND 12, DELETE 9 | 3     |
| skill / full, episodes 50–100  | 15          | APPEND 1, DELETE 1  | 3     |
| skill / full, episodes 100–150 | 12          | APPEND 2, DELETE 2  | 3     |


Across 150 episodes the writer proposed 39 skills; the grounding and dedup
checks let through a net of three. In the last two legs the store did not change
at all — the two entries appended in 100–150 were also the two deleted, leaving
the same three ids, with identical text, that it had at episode 50.

Those three procedures carried +14 **three times**. This is either the most
interesting result in the study — three general procedures are the whole of what
this benchmark rewards, and 100 further episodes of experience added nothing to
them — or an artefact of skill writing so rarely that it never had a chance to
overfit. Its irreducible constraint is visible in the counts: a procedural skill
needs one working path, so skill cannot write from an all-failure episode, and
35–38 of every 50 evolving episodes ended in all-failure.

### Batch induction outgrows the context window

`WritePolicy.full()` runs batch induction every 25 episodes. `writer.induce`
builds its prompt from the clustered episodes plus the existing entries, with no
bound on either. On this third leg two arms crossed the server's window and
died mid-run:


| arm                 | died at                 | induce prompt                      |
| ------------------- | ----------------------- | ---------------------------------- |
| `reflection / full` | evolve 24/50 (step 124) | ≥ 31745 tok, + 1024 output > 32768 |
| `rule / full`       | evolve 49/50 (step 149) | ≥ 31745 tok, + 1024 output > 32768 |


`raw` has no induction and `skill`'s store is three entries, so neither was
affected. Both failures landed on a 25-episode induction boundary.

This is the same accumulation described two sections above, in its terminal
form: reflection's store had reached 43 live entries and rule's 22, and the
prompt built from them no longer fit. **Re-running at 65536 is a reprieve, not a
fix** — the prompt grows with the store, so a fourth or fifth leg hits the same
wall. Bounding the `induce` prompt is the actual repair; it was deliberately not
done here, because truncating it would change writer semantics and make this leg
non-comparable with the two below it.

Recovered by resuming each arm's 100-episode store on a 65536-token server; both
then completed 150 episodes with no context error.

## Budgeting by failures: 100 failed episodes, successes discarded

Every leg above budgets by *tasks* — run 50, take whatever mix of outcomes falls
out. This one budgets by *outcome*. Each arm evolves until **100 episodes have
failed**, and successful episodes are discarded before the memory system
observes them: no extraction, no verify/refine/prune, no batch-induction
buffering, and no advance of the Evolver's step counter, so "100 failures" is
also the unit `full`'s every-25-episode induction cadence counts in. The store is
built out of failure evidence alone, and every utility signal that reaches the
deletion machinery is negative by construction.

The ALFWorld twin of this leg is RESULTS_ALFWORLD.md §10–§11; the flags
(`--evolve-until-failures 100 --failure-only-writes`) and their semantics are
identical.


|               |                                                                                                                                                                                |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| evolving pool | `spreadsheetbench_evolve_train+val+test_297_seed42.json` — the whole selectable pool. Positions [0,50), [50,100), [100,150), [150,200) are the four manifests above, id for id |
| store         | fresh, not a continuation                                                                                                                                                      |
| stopping rule | 100 failed episodes; the runner warns loudly rather than finishing short                                                                                                       |
| evaluation    | the same frozen 100 `test` tasks as every other run                                                                                                                            |
| server        | 131072-token window, so no arm hit the induction wall above                                                                                                                    |
| arms          | `none`, `reflection`, `rule`, `skill`. `raw` excluded: it keeps only `best_success()`, so under this filter it ends with an empty store and *is* the `none` baseline           |


**The mirror leg is not runnable on this pool.** At an evolve-time success rate
of 21–24%, 100 *successes* would cost 420–480 tasks and the pool holds 297. The
same arithmetic that makes failures cheap here makes them expensive on ALFWorld,
where the rate runs the other way. *(It was made runnable by enlarging the pool
to 798 — see "Budgeting by successes" below, which closes this comparison. The
cost estimate was right: it took 409–520.)*


| arm        | policy  | tasks spent | successes discarded | store | eval   | delta vs 13 | σ    | delta vs 16.3 | McNemar p |
| ---------- | ------- | ----------- | ------------------- | ----- | ------ | ----------- | ---- | ------------- | --------- |
| reflection | minimal | 132         | 32                  | 82    | 22/100 | +9          | +1.9 | +5.7          | 0.064     |
| reflection | full    | 130         | 30                  | 6     | 21/100 | +8          | +1.7 | +4.7          | 0.077     |
| rule       | full    | 129         | 29                  | 6     | 21/100 | +8          | +1.7 | +4.7          | 0.134     |
| rule       | minimal | 130         | 30                  | 46    | 21/100 | +8          | +1.7 | +4.7          | 0.096     |
| skill      | minimal | 126         | 26                  | **0** | 20/100 | +7          | +1.5 | +3.7          | 0.167     |
| skill      | full    | 128         | 28                  | **0** | 16/100 | +3          | +0.6 | −0.3          | 0.629     |
| none       | —       | 0           | —                   | —     | 13/100 | —           | —    | —             | —         |


Every arm hit exactly 100 failures and wrote exactly 100 episodes, spending
126–132 tasks. The budget did what it is for: the arms are matched on what the
memory learned from, not on what the agent was shown.

### `skill` writes nothing from failure

`skill / minimal` called the writer **zero times** in 126 tasks. `skill / full`
called it 13 times, appended one entry, and deleted it again. Both finished with
an empty store and injected 0 tokens on all 100 evaluation tasks.

SkillOpt-style extraction needs a trajectory that worked — a failed episode
offers no procedure to abstract — so the failure-only filter starves it
completely. This is the same shape as `raw`'s exclusion, arrived at
independently: two of the four content types have **nothing to say about a task
that went wrong**. It is also what turned this leg into two extra baseline draws
(see "The baseline, drawn three times").

### `full` deletes almost exactly what it appends

Writer ops over the 100 written episodes, split by where they came from:


| arm / policy         | online A / M / D / R | ops on induction steps | store peak → final |
| -------------------- | -------------------- | ---------------------- | ------------------ |
| reflection / full    | 89 / 18 / 89 / 1     | 9 A / 14 M / 4 D / 1 R | 11 → 6             |
| rule / full          | 46 / 5 / 44 / 1      | 12 A / 6 M / 8 D       | 8 → 6              |
| reflection / minimal | 82 / 20 / 0 / 0      | — (induction off)      | 82 → 82            |
| rule / minimal       | 62 / 8 / 16 / 1      | — (induction off)      | 47 → 46            |


A = APPEND, M = MERGE, D = DELETE, R = REVISE. `minimal` still shows DELETEs
because the online writer's op vocabulary includes them; what the policy turns
off is `verify`, `refine`, `delete_on_low_confidence`, `utility_deletion` and
`batch_induction`.

Under `full`, **verification refutes entries at the rate the writer produces
them** (89 appends against 89 deletes for reflection), and the store never
exceeds 11 live entries against `minimal`'s 82. The deletions are not the batch
inductions — only 4 of reflection's 93 fell on an induction step — they are
`verify` + `delete_on_low_confidence` firing at each online step. When every
episode is a failure, the only evidence available to *confirm* an entry is an
episode that went wrong, so almost nothing survives contact with the next
episode. Injected context follows: 567 tokens for reflection/full against 1054
for reflection/minimal.

It is the same direction as the fixed-count legs, where `full` also held rule's
store to 17→22→29 against `minimal`'s growth, but far more extreme: there the
filter was store size, here it is the sign of the evidence.

### Nothing here clears the floor

Against the study's original 13/100 baseline the writer arms land at +8/+9, or
1.7–1.9σ. Against the three-draw baseline mean of 16.3 they land at +4.7/+5.7,
about 1σ. No McNemar test clears 0.05, and the discordant counts are small
(b/c of 5/14, 4/12, 7/15, 5/13). Meanwhile the *memory-free* `skill` runs came
in at 16 and 20 on the same 100 tasks.

**So this leg does not establish that failure-only memory helps.** What it does
establish is sharper than that:

1. `skill` and `raw` cannot be built from failures at all — the content type
  decides whether failure evidence is usable, and for two of four the answer is
   that it is not.
2. `full`'s verify/delete machinery, given only negative evidence, holds the
  store at ~7% of `minimal`'s size without any gain in eval — the mechanism
   study's cost/benefit gets worse, not better, under this filter.
3. The eval set's noise floor swallows the rest, and re-measuring it (which this
  leg did by accident) moved the whole study's baseline up 3 points.

The comparison this leg was designed to make — failure-only against
success-only, at equal outcome budget — cannot be closed on this benchmark
without a larger evolving pool, per the arithmetic above.

## Budgeting by successes: 100 solved episodes, failures discarded

The mirror of the leg above, and the comparison it could not close. Each arm
evolves until **100 episodes have succeeded**, and failed episodes are discarded
before the memory system observes them: no extraction, no verify/refine/prune,
no batch-induction buffering, and no advance of the Evolver's step counter, so
"100 successes" is also the unit `full`'s every-25-episode induction cadence
counts in. The store is built out of successful evidence alone, and every
utility signal reaching the deletion machinery is positive by construction.

The flags are `--evolve-until-successes 100 --success-only-writes`. `success` is
`any_success`, the strict criterion — all of a task's test cases pass — so a
partially-correct episode is a failure and is thrown away.


|               |                                                                                                                                                                    |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| evolving pool | `spreadsheetbench_evolve_train+val+test+rest_798_seed42.json`. Its first 297 positions are the `fail100` manifest verbatim (verified id for id)                     |
| store         | fresh, not a continuation                                                                                                                                          |
| stopping rule | 100 successful episodes; the runner warns loudly rather than finishing short                                                                                       |
| evaluation    | the same frozen 100 `test` tasks as every other run                                                                                                                |
| server        | 131072-token window                                                                                                                                                |
| arms          | `none`, `reflection`, `rule`, `skill`. `raw` excluded for the opposite reason to `fail100`: it keeps only `best_success()` already, so the filter is a no-op for it |


**The pool had to spill outside SkillOpt's split.** `train`+`val`+`test` holds
297 selectable tasks and 100 successes costs 409–520, so the manifest appends
`rest` — the 512 tasks of the 912 release the 400-task Verified id split does
not cover. 114 were dropped for sharing a source workbook with the evaluation
set, leaving 798. The frozen eval set is disjoint by task id **and** by source
workbook (the builder refuses to write the manifest otherwise; re-verified after
the run: zero id overlap for all six arms). What is given up is that every arm
here drew 112–223 of its evolving tasks from outside the split the rest of the
SAGE ecosystem uses, which the `fail100` leg — spending only 126–132 tasks —
never touched. **That confound sits underneath every fail-vs-succ row below.**


| arm        | policy  | tasks spent | failures discarded | store | eval   | delta vs 13 | σ    | delta vs 16.3 | McNemar p |
| ---------- | ------- | ----------- | ------------------ | ----- | ------ | ----------- | ---- | ------------- | --------- |
| skill      | minimal | 463         | 363                | 17    | 27/100 | +14         | +3.0 | +10.7         | **0.003** |
| reflection | full    | 409         | 309                | 130   | 26/100 | +13         | +2.8 | +9.7          | **0.002** |
| rule       | full    | 501         | 401                | 52    | 23/100 | +10         | +2.1 | +6.7          | 0.031     |
| skill      | full    | 440         | 340                | 24    | 23/100 | +10         | +2.1 | +6.7          | 0.052     |
| reflection | minimal | 502         | 402                | 126   | 21/100 | +8          | +1.7 | +4.7          | 0.096     |
| rule       | minimal | 520         | 420                | 48    | 21/100 | +8          | +1.7 | +4.7          | 0.077     |
| none       | —       | 0           | —                  | —     | 13/100 | —           | —    | —             | —         |


Every arm hit exactly 100 successes and wrote exactly 100 episodes. The cost
ratio against the failure budget is **3.1–4.0×** the tasks for the same number
of written episodes, which is what an evolve-time success rate near 20% buys.

The McNemar p-values are computed against the *single* 13/100 baseline draw,
which "The baseline, drawn three times" has already shown to be the low draw of
three. Read them as ranking, not as significance: re-centred on 16.3 the two
strongest rows fall to +10.7 and +9.7, about 2σ.

### The two content types that cannot learn from failure can learn from success

This is what the leg was built to test, and it is the one result here that does
not depend on the noise floor:


| arm / policy    | store from 100 failures | store from 100 successes | eval fail100 → succ100 |
| --------------- | ----------------------- | ------------------------ | ---------------------- |
| skill / minimal | **0**                   | 17                       | 20 → **27**            |
| skill / full    | **0**                   | 24                       | 16 → **23**            |


`skill` called the writer zero times in 126 failure-only tasks and finished with
an empty store — its two `fail100` runs were byte-identical in prompt to `none`
and served as extra baseline draws. Under success-only it writes on essentially
every episode (19 APPENDs under `minimal`, 29 under `full`) and ends with a real
store. SkillOpt-style extraction needs a trajectory that worked; supply one and
the type is no longer starved.

Its injected block reaches **2359 tokens** against a 2500-token budget — the
only configuration in this study that fills it. In the fixed-count legs `skill`
saturated at three entries and injected 771; the constraint was never the
extractor, it was the supply of successful episodes to extract from.

### `full`'s deletion machinery does not fire on positive evidence

The `fail100` leg found `full` deleting at almost exactly the rate it appended
(89 A against 89 D for reflection) and holding the store to ~7% of `minimal`'s.
Flip the sign of the evidence and the mechanism disappears:


| arm / policy      | A / M / D / R, fail100 | A / M / D / R, succ100 | store fail100 → succ100 |
| ----------------- | ---------------------- | ---------------------- | ----------------------- |
| reflection / full | 98 / 32 / 93 / 2       | 130 / 53 / **0** / 0   | 6 → **130**             |
| rule / full       | 58 / 11 / 52 / 1       | 60 / 32 / 9 / 1        | 6 → 52                  |
| skill / full      | —                      | 29 / 4 / 5 / 0         | 0 → 24                  |


(fail100 columns are online + induction ops summed from the table two sections
above.) `reflection / full` issued **not one DELETE in 100 episodes**. When
every episode succeeded, `verify` finds confirming evidence for whatever was
injected and `delete_on_low_confidence` never trips, so `full` collapses to
`minimal` plus merging — and its store ends up *larger* than `minimal`'s (130
against 126), the reverse of every other leg in this study.

So the mechanism study's answer depends on the sign of the evidence, not on the
mechanism: **`full` is a pruner under failure and a no-op under success.** That
is a property of tying deletion to confidence signals that only failure can move.

### Failure-only against success-only, at equal outcome budget

The comparison the `fail100` leg could not make. Same 100 written episodes, same
eval set, same scaffold; only the sign of the filter and the pool tail differ.


| arm / policy         | fail100 | succ100 | delta  | discordant b/c | McNemar p |
| -------------------- | ------- | ------- | ------ | -------------- | --------- |
| skill / minimal      | 20      | **27**  | **+7** | 14/7           | 0.189     |
| skill / full         | 16      | **23**  | **+7** | 13/6           | 0.167     |
| reflection / full    | 21      | **26**  | +5     | 11/6           | 0.332     |
| rule / full          | 21      | 23      | +2     | 12/10          | 0.832     |
| rule / minimal       | 21      | 21      | 0      | 10/10          | 1.000     |
| reflection / minimal | 22      | 21      | −1     | 13/14          | 1.000     |


**Not one pairing clears 0.05, and the two largest are the two arms whose
failure-only store was empty** — i.e. they are success-only against *no memory*,
not against failure memory. Strip those and what remains is +5, +2, 0, −1 across
the four arms that could write from both: **no detectable difference between
learning from failure and learning from success on this benchmark**, at 4× the
cost for the success budget.

That null is worth stating precisely, because it is not the null the leg was
expected to produce. Failure evidence is roughly 4× cheaper to collect here and
buys the same eval score for the two content types that can use it at all. The
content type decides whether an outcome is usable; given a usable outcome, the
sign of it did not matter.

### Cost


| arm / policy         | tasks spent | agent time | writer prompt tok | writer completion tok |
| -------------------- | ----------- | ---------- | ----------------- | --------------------- |
| skill / full         | 440         | 5.2 h      | 1087k             | 119k                  |
| reflection / full    | 409         | 4.2 h      | 888k              | 103k                  |
| rule / full          | 501         | 5.0 h      | 744k              | 94k                   |
| skill / minimal      | 463         | 10.7 h     | 474k              | 85k                   |
| reflection / minimal | 502         | 10.5 h     | 361k              | 42k                   |
| rule / minimal       | 520         | 10.6 h     | 375k              | 45k                   |


Agent time is summed per-episode time, not wall clock: four arms shared two vLLM
servers and three of them were interrupted and resumed a day apart, so wall
clock is not a single measurement for them (`wall_seconds` is null in those
summaries; see "Reproducing"). Writer tokens are recomputed over both legs from
`evolve_log.jsonl` and are comparable.

## Failure modes

Counts over the 100 evaluation tasks, `none` arm:


| reason                           | n   | meaning                                |
| -------------------------------- | --- | -------------------------------------- |
| `eval-mismatch`                  | 77  | ran, produced a workbook, wrong values |
| `output-not-found`               | 7   | never wrote the output file            |
| `exec-error`                     | 2   | `solution.py` crashed on case 2 or 3   |
| `no-solution-py-for-other-cases` | 1   | edited case 1 by hand                  |


The benchmark is essentially all `eval-mismatch`: 97% of episodes produced a
`solution.py`, and the plumbing failures memory might plausibly fix account for
10 tasks in total. Memory arms cut those to 2–8, but the aggregate is dominated
by getting the *values* right, which is where the ceiling on any memory effect
comes from.

## Cost

Writer tokens **per 50-episode leg** (raw has no writer):


| arm / policy         | leg     | prompt tok | completion tok |
| -------------------- | ------- | ---------- | -------------- |
| skill / full         | 0–50    | 176k       | 18k            |
|                      | 100–150 | 252k       | 24k            |
| rule / full          | 0–50    | 353k       | 44k            |
|                      | 100–150 | 394k       | 48k            |
| reflection / full    | 0–50    | 405k       | 54k            |
|                      | 100–150 | 494k       | 67k            |
| skill / minimal      | 100–150 | 75k        | 10k            |
| rule / minimal       | 100–150 | 170k       | 21k            |
| reflection / minimal | 100–150 | 197k       | 23k            |


Wall time is not comparable across arms — four arms shared two vLLM servers.
The writer token columns are, and they are the real compute difference: `full`
costs reflection and rule 2.3–2.5× their `minimal` writer tokens for the
verify/refine/induce loop. **Per-leg cost also rises as the store grows** —
reflection/full spends 22% more in its third leg than its first — which is the
same unbounded-prompt growth that eventually killed it.

## What to run next, in order

1. ~~**A second**~~ `none` ~~**baseline.**~~ **Done, and it landed the way this item
  warned.** The two empty-store `skill / fail100` runs are memory-free
   replicates: 16/100 and 20/100 against the original 13/100. The baseline mean
   is 16.3, every delta above is ~3 points optimistic, and the count clearing 2σ
   drops from eleven to about three. A *fourth* draw is now the cheap way to
   pin the mean down — the SD of three points is not much to lean on.
2. **Repeat** `raw / full`**.** It is the only arm with a monotone trend across
  three legs (18 → 21 → 27) and it reached the study's joint-best result from
   the least structured content type. At 1.9σ it is exactly the kind of finding
   that a single repeat either establishes or kills.
3. **Bound the** `induce` **prompt**, then re-run `full` at 200 episodes. As it
  stands `full` cannot be run further for reflection or rule at any store size
   without hitting the same wall; 65536 only moves it.
4. **Raise the baseline before comparing content types.** At 13/100 with σ ≈ 4.7
  the whole study lives in a 14-point band ~3σ wide, and only 11 of 100 tasks
   are solved reliably at all. More agent turns or a stronger model would
   separate the content types better than more evolving episodes.
5. ~~**The success-budget mirror of the failure leg.**~~ **Done — "Budgeting by
  successes".** It closed the fail-vs-succ comparison (no detectable
   difference for the arms that can write from both, at 4× the task cost) and
   showed `skill` going from an empty store to a budget-filling one purely by
   changing the sign of the filter. Two follow-ups it opened:
   - **A fourth baseline draw is now cheaper than ever to want.** Two arms here
     clear McNemar p < 0.01 against 13/100, and that p is against the known low
     draw. Nothing in this document is safe at p-level precision until the
     baseline mean is pinned.
   - **`full`'s pruning is evidence-sign-dependent, not policy-dependent** —
     `reflection / full` issued zero DELETEs in 100 successful episodes and 93
     in 100 failed ones. The mechanism study's cost/benefit needs re-reading
     with that in mind: on a mixed stream, `full` prunes only in proportion to
     how much failure it sees.



## Reproducing

```bash
bash scripts/setup_spreadsheetbench.sh
python scripts/build_spreadsheetbench_manifests.py
python scripts/build_spreadsheetbench_manifests.py \
    --evolve-skip 50 --evolve-count 100 --evolve-overflow-split val \
    --exclude-groups-from manifests/spreadsheetbench_eval_test_100_seed42.json --no-eval
python scripts/build_spreadsheetbench_manifests.py \
    --evolve-skip 100 --evolve-count 150 \
    --evolve-overflow-split val --evolve-overflow-split test \
    --exclude-groups-from manifests/spreadsheetbench_eval_test_100_seed42.json --no-eval

bash scripts/serve_qwen.sh 0 8000 --background
bash scripts/serve_qwen.sh 1 8001 --background
bash scripts/run_spreadsheetbench_sweep.sh minimal
bash scripts/run_spreadsheetbench_sweep.sh full
bash scripts/run_spreadsheetbench_sweep.sh minimal100    # resumes each store
bash scripts/run_spreadsheetbench_sweep.sh full100
bash scripts/run_spreadsheetbench_sweep.sh minimal150
bash scripts/run_spreadsheetbench_sweep.sh full150

# reflection/full and rule/full need the wider window (see "Batch induction
# outgrows the context window"); the other arms are unaffected by it.
MEMSYS_MAX_MODEL_LEN=65536 bash scripts/serve_qwen.sh 0 8000 --background
MEMSYS_MAX_MODEL_LEN=65536 bash scripts/serve_qwen.sh 1 8001 --background
MEMSYS_ARMS="reflection rule" bash scripts/run_spreadsheetbench_sweep.sh full150

# The failure-budget leg. One manifest covering the whole selectable pool, and
# a 131072-token server so no arm meets the induction wall.
python scripts/build_spreadsheetbench_manifests.py \
    --evolve-count 297 --evolve-overflow-split val --evolve-overflow-split test \
    --exclude-groups-from manifests/spreadsheetbench_eval_test_100_seed42.json --no-eval
MEMSYS_EVAL_WORKERS=4 bash scripts/run_spreadsheetbench_sweep.sh full_fail100
MEMSYS_EVAL_WORKERS=4 bash scripts/run_spreadsheetbench_sweep.sh minimal_fail100

# The success-budget leg. A larger pool, because 100 successes costs 409-520
# tasks; its first 297 positions are the fail100 manifest verbatim.
python scripts/build_spreadsheetbench_manifests.py \
    --evolve-count 798 --evolve-overflow-split val --evolve-overflow-split test \
    --evolve-overflow-split rest \
    --exclude-groups-from manifests/spreadsheetbench_eval_test_100_seed42.json --no-eval
MEMSYS_EVAL_WORKERS=4 bash scripts/run_spreadsheetbench_sweep.sh full_succ100
MEMSYS_EVAL_WORKERS=4 bash scripts/run_spreadsheetbench_sweep.sh minimal_succ100
```

Raw outputs:
`$MEMSYS_RESULTS_ROOT/spreadsheetbench/{minimal,full,minimal_e100,full_e100,minimal_e150,full_e150,full_fail100,minimal_fail100,minimal_succ100,full_succ100_cont}/<arm>_<policy>/`
— `summary.json`, `eval.jsonl`, `evolve_episodes.jsonl`, `evolve_log.jsonl`,
`store.jsonl`, and per-task `solution.py` + predicted workbooks. The two crashed
attempts are kept alongside as `*_full.ooc-crash/`.

**The succ100 leg was interrupted mid-evolve and resumed**, which is why its
`full` arms live in `full_succ100_cont/` rather than `full_succ100/`. The runner
opens `evolve_episodes.jsonl` with `"w"`, so resuming in place would erase leg 1;
each `full` arm instead continued on a slice of the same 798-task pool starting
at the position it stopped (`_succ100_slices/`), resuming its own `store.jsonl`
with `--evolve-step-offset` set to the successes already written, so `full`'s
25-episode induction cadence fires where an uninterrupted run's would.
`minimal_succ100/reflection_minimal` completed its evolving before the
interruption and had only its evaluation re-run.

Both legs are stitched back onto one numbering by
`drivers/ssb_succ100_merge.py` — `task_index` and `step` are offset, leg 2's raw
files are kept as `*.leg2.jsonl`, and the evolve counters in `summary.json` are
recomputed over the whole run. `drivers/ssb_succ100_fix_usage.py` then rebuilds
`writer_usage` from the merged `evolve_log.jsonl` (the resumed process only knew
leg 2) and **nulls `wall_seconds`**, which is not a single measurement across two
days — use `evolve_agent_seconds`. What is *not* resumed, and is the one way
these arms differ from an uninterrupted run: the batch-induction buffer, which
restarts empty at the resume point. Same caveat as every earlier continuation
leg here.

The `fail100` and `succ100` rows in `evolve_episodes.jsonl` carry three extra fields:
`written` (did the memory system observe this episode), `task_index` (position
in the manifest) and a `step` that is `null` for discarded episodes, so tasks
attempted and episodes written stay separable. `summary.json` carries
`evolve_n` / `evolve_written` / `evolve_successes` / `evolve_failures` and an
`evolve_budget` block naming the stopping rule.

Manifest fingerprints (`task_ids_sha256`, first 16):
evolve [0,50) `654ef6ea3a1d748e`, evolve [50,100) `ba2969e56a40109e`,
evolve [100,150) `51f5a386a966ea84`, evolve [0,297) `65608bd67755f753`,
evolve [0,798) `38f2a0b3c0d3f222`, eval `5637cf201e1948d9`.
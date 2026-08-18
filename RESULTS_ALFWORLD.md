# ALFWorld Memory Content study — Qwen3.5-9B

Run 2026-08-09. 50 evolving tasks (`train`), 100 evaluation tasks
(`valid_unseen`, disjoint), 1 rollout per task, 50-step horizon, seed 42.
Model served locally by vLLM; the same model writes memory and acts.

Raw outputs: `/gpfs/radev/scratch/cohan/jw3278/memsys_results/{minimal,full}/`.
Reproduce with [RUN_ALFWORLD.md](RUN_ALFWORLD.md).

§1–§5 report the 50-evolving-task run. **§6 extends every arm to 100 evolving
tasks and overturns §1**, so read it before quoting the write-policy claim. §7
adds the orthogonal axis — the *same* 50 tasks three times over — where nothing
reaches significance but `raw`'s store is shown to close under repetition while
the writer arms' stores grow without bound. **§8 crosses the two axes**: 75
tasks twice is the same 150-episode budget as §9's 150 distinct tasks and §7's
50 × 3, so the three can be compared at fixed cost. None of the 16 paired tests
between them is significant — how the episodes are arranged does not measurably
matter, only which content type consumes them.

**§9 carries the stream to 150 and 200 evolving tasks and its job is to
falsify**: the third and fourth points retract §6.4 outright and dissolve the
apparent monotone recovery of `rule`. Read §9.2 before quoting any single delta
in this document — it puts a number on the churn that every other section is
swimming in. **§10 changes what an "episode" buys**: it budgets the stream by
100 *successes* rather than 100 attempts and drops failed episodes before the
memory system ever sees them. It contains the study's one large positive result
for an LLM-written memory — `rule/minimal` at 84.0%, statistically
indistinguishable from `raw` — and, in the same table, the study's largest
*harm* from the same intervention. **§11 flips that filter's sign**: 100
*failures* banked, successes discarded. No arm beats baseline there, and the two
cells where a failure-built store survives to evaluation land 26 and 38 points
below their §10 twins — but read §11.1 first, because one cell accidentally
re-ran the no-memory baseline and put a second number on the churn floor.

## Scaffold validation

The no-memory baseline scored **58.0%**, matching SAGE's independently measured
58% (MemRL) / 60% (ACE) for this model and scaffold. The prompt and six
demonstrations are byte-identical to theirs, deliberately: SAGE recorded that a
generic prompt with no demonstrations scored **18%** on the same tasks, so
scaffold drift dwarfs any memory effect. Treat the baseline as the canary — if
it moves off ~58%, distrust everything below.

## Results

Every arm evaluates the identical ordered task list, so comparisons against the
baseline are paired and McNemar's exact test applies. `b` = baseline solved it
and the arm did not; `c` = the reverse.


| arm        | policy  | rate      | Δ         | b/c   | McNemar p  | store | inj. tok | writer calls |
| ---------- | ------- | --------- | --------- | ----- | ---------- | ----- | -------- | ------------ |
| none       | —       | 58.0%     | —         | —     | —          | —     | 0        | —            |
| raw        | minimal | 75.0%     | +17.0     | 11/28 | **0.009**  | 38    | 1302     | —            |
| reflection | minimal | 65.0%     | +7.0      | 17/24 | 0.349      | 39    | 1022     | 73           |
| rule       | minimal | 53.0%     | −5.0      | 16/11 | 0.442      | 16    | 244      | 67           |
| skill      | minimal | **79.0%** | **+21.0** | 6/27  | **0.0002** | 4     | 938      | 36           |
| raw        | full    | **80.0%** | **+22.0** | 7/29  | **0.0002** | 36    | 1205     | —            |
| reflection | full    | 54.0%     | −4.0      | 18/14 | 0.597      | 43    | 942      | 145          |
| rule       | full    | 46.0%     | −12.0     | 21/9  | **0.043**  | 10    | 226      | 136          |
| skill      | full    | 64.0%     | +6.0      | 19/25 | 0.451      | 10    | 1392     | 103          |


Significant: `raw` (both policies), `skill/minimal`, and `rule/full` — the last
being significant **harm**. Reflection never reaches significance in either
direction.

## 1. More write mechanism made every LLM-written memory worse


| arm        | minimal | full  | Δ       | writer calls |
| ---------- | ------- | ----- | ------- | ------------ |
| raw        | 75.0%   | 80.0% | **+5**  | none → none  |
| reflection | 65.0%   | 54.0% | **−11** | 73 → 145     |
| rule       | 53.0%   | 46.0% | **−7**  | 67 → 136     |
| skill      | 79.0%   | 64.0% | **−15** | 36 → 103     |


The dissociation is clean and it is not about store size. `raw` is the only arm
with **no LLM writer**: the sole full-policy mechanism that touches it is
utility deletion, pure bookkeeping, and it gained 5 points. Every arm whose
extra mechanisms invoke the writer (verify → refine → batch induction) lost
ground while roughly doubling writer cost. Reflection's store even *grew*
(39 → 43) while its accuracy fell 11 points, so the damage is to entry quality,
not quantity.

For the write-mechanism ablation this is the headline: on ALFWorld at this
scale, `WritePolicy.minimal()` dominates `WritePolicy.full()` for every content
type that uses an LLM writer.

**This does not survive doubling the evolving stream.** At 100 evolving tasks
full's deficit halves for reflection and skill and reverses sign for rule — see
[§6](#6-doubling-the-evolving-stream-50--100-evolving-tasks). The scale
qualifier in the sentence above is doing all the work.

## 2. Abstraction invents procedures that verbatim replay cannot

`pick_heat_then_place_in_recep`, 21 eval tasks, baseline 9/21 (43%):


| arm        | minimal     | full            |
| ---------- | ----------- | --------------- |
| raw        | 13/21 (62%) | **14/21 (67%)** |
| reflection | 6/21 (29%)  | 6/21 (29%)      |
| rule       | 5/21 (24%)  | 4/21 (19%)      |
| skill      | 7/21 (33%)  | **0/21 (0%)**   |


**Every abstracting memory type lands below baseline on this family; only
verbatim trajectory replay improves it.** `skill/full` collapses to zero.

The cause is in the store. `skill/full` learned:

```
go to the microwave
open the microwave if it is closed
move <obj> to the microwave      <- the agent is no longer holding <obj>
close the microwave
heat <obj> with the microwave    <- cannot succeed
```

In ALFWorld `heat <obj> with microwave` is a single atomic action taken *while
holding* the object. This procedure puts the object inside the microwave first,
so the heat step can never fire. The agent follows it faithfully on all 21
tasks and fails all 21.

The same false belief appears in `reflection/minimal` as a lesson — *"Always
close the microwave before attempting to heat an object inside it"* — induced
from a single failed episode. Batch induction then consolidated it into one
authoritative, deterministically-executed procedure.

This is the sharpest result in the run: **procedural memory has higher leverage
in both directions.** A correct procedure generalises strongly (see §3); a
wrong one converts a 43% family into 0%. Raw replay is immune because it never
states a rule — it shows real action sequences, which necessarily contain the
true one.

Fifty more evolving episodes do not fix it and produce a second one of the same
shape — see [§6.3](#63-a-wrong-procedure-is-not-corrected-by-more-evidence).

## 3. Compression works, until the mechanism breaks it

`skill/minimal` is the best content type: **79.0%** from a store of **four**
skills and 36 writer calls — beating raw's 38 stored trajectories, with less
than a third the regressions (b=6 vs 11) and the smallest writer bill of any
LLM arm. It is also the only arm that never regresses the saturated
`pick_and_place_simple` family (18/18).

Caveat: those four skills name concrete objects (`find_two_soapbottles`) even
though `memsys/schemas.py` requires placeholders. They generalise anyway,
because what transfers is the step skeleton — *go to a likely location → check
→ take → go to target → place*. So the win is about procedural **structure**,
not the abstraction the schema tries to enforce. Enforcing placeholders is a
cheap untested ablation.

## 4. Content types help disjoint task families

Under `minimal`, per family (baseline in the first column):


| family                 | none  | raw       | reflection | rule  | skill     |
| ---------------------- | ----- | --------- | ---------- | ----- | --------- |
| pick_cool_then_place   | 5/15  | **14/15** | 7/15       | 7/15  | 12/15     |
| pick_clean_then_place  | 9/23  | 10/23     | 18/23      | 10/23 | **19/23** |
| pick_heat_then_place   | 9/21  | **13/21** | 6/21       | 5/21  | 7/21      |
| look_at_obj_in_light   | 10/13 | **13/13** | 8/13       | 7/13  | **13/13** |
| pick_two_obj_and_place | 7/10  | 8/10      | **10/10**  | 6/10  | **10/10** |
| pick_and_place_simple  | 18/18 | 17/18     | 16/18      | 18/18 | **18/18** |


Raw owns `cool` (+9), reflection owns `clean` (+9) while *hurting* `heat` (−3).
A single success-rate column erases this entirely. Note also that
`pick_and_place_simple` is saturated at 18/18, so 18 of the 100 eval tasks can
only contribute regressions — aggregate deltas are compressed by that ceiling.

## 5. Rule memory is the weakest content type here, and why

Rule never beats baseline (53.0% / 46.0%) and is significantly harmful under
`full`. Its store shows the writer producing instance-level reactions rather
than transferable principles:

- *"the agent is at a location and sees a newspaper → take the newspaper"*
- *"attempts to open a receptacle and receives 'N[othing happens]' → look"*

These are error-recovery patches keyed to specific objects, not the multi-step
procedures the baseline actually fails on.

**Fairness caveat.** Rule fills only ~240 of its 1500-token budget, against
~1000–1400 for the other types. The equal-budget control equalises the
*ceiling*, not the *usage*, so rule may be losing partly on quantity rather than
kind. `MemoryConfig(equal_item_count=...)` is the lever to separate these.

## 6. Doubling the evolving stream (50 → 100 evolving tasks)

Run 2026-08-11. Each arm resumed its 50-task store (`resumed_from` in every
`summary.json`) and evolved 50 further `train` tasks from
`manifests/evolve_train_50to100_seed42.json`, disjoint from the first 50. The
evaluation set, seed, horizon and scaffold are unchanged, so **every number
below is paired with every number above.** The `none` baseline has no store and
no writer, so it is reused verbatim (58.0%).

Raw outputs: `/gpfs/radev/scratch/cohan/jw3278/memsys_results/{minimal,full}_e100/`.


| arm        | policy  | rate      | Δ vs none | b/c   | McNemar p  | Δ vs 50-task | store | inj. tok | writer calls |
| ---------- | ------- | --------- | --------- | ----- | ---------- | ------------ | ----- | -------- | ------------ |
| none       | —       | 58.0%     | —         | —     | —          | —            | —     | 0        | —            |
| raw        | minimal | 75.0%     | +17.0     | 9/26  | **0.006**  | 0.0          | 75    | 1278     | —            |
| reflection | minimal | 67.0%     | +9.0      | 12/21 | 0.163      | +2.0         | 66    | 997      | 81           |
| rule       | minimal | 58.0%     | +0.0      | 21/21 | 1.000      | +5.0         | 25    | 260      | 83           |
| skill      | minimal | 70.0%     | +12.0     | 7/19  | **0.029**  | −9.0         | 6     | 1439     | 33           |
| raw        | full    | **81.0%** | **+23.0** | 6/29  | **0.0001** | +1.0         | 61    | 1244     | —            |
| reflection | full    | 62.0%     | +4.0      | 15/19 | 0.608      | +8.0         | 77    | 961      | 158          |
| rule       | full    | 61.0%     | +3.0      | 20/23 | 0.761      | **+15.0**    | 15    | 241      | 134          |
| skill      | full    | 59.0%     | +1.0      | 21/22 | 1.000      | −5.0         | 8     | 1325     | 117          |


Store sizes and writer calls are cumulative over all 100 evolving tasks.
Injected tokens are flat because the 1500-token budget binds, so raw's store
doubling costs nothing at inference.

### 6.1 Only `raw` is stable; everything else moves

`raw` is the one content type whose eval rate is unchanged by doubling the
data — 75.0% → 75.0% under minimal (13 tasks flipped each way, pure churn) and
80.0% → 81.0% under full. It is also the best arm in the whole study:
`raw/full` **at 81.0%, +23.0, p = 0.0001**, now perfect on
`pick_cool_then_place_in_recep` (15/15).

The paired 50 → 100 test on each arm's own two runs:


| arm        | policy  | 50 → 100 | b/c   | p         |
| ---------- | ------- | -------- | ----- | --------- |
| raw        | minimal | 75 → 75  | 13/13 | 1.000     |
| reflection | minimal | 65 → 67  | 16/18 | 0.864     |
| rule       | minimal | 53 → 58  | 13/18 | 0.473     |
| skill      | minimal | 79 → 70  | 16/7  | 0.093     |
| raw        | full    | 80 → 81  | 9/10  | 1.000     |
| reflection | full    | 54 → 62  | 13/21 | 0.229     |
| rule       | full    | 46 → 61  | 10/25 | **0.017** |
| skill      | full    | 64 → 59  | 14/9  | 0.405     |


Note how large the discordant counts are relative to the deltas: `raw/minimal`
changes 26 of 100 task outcomes to arrive at exactly the same score. Aggregate
stability is not per-task stability for any arm here.

### 6.2 The `full` penalty is a small-data artifact

This is the finding that costs us §1. Minimal-minus-full, at both stream
lengths:


| arm        | full − minimal @50 | full − minimal @100 |
| ---------- | ------------------ | ------------------- |
| raw        | +5                 | +6                  |
| reflection | −11                | −5                  |
| rule       | −7                 | **+3**              |
| skill      | −15                | −11                 |


Every LLM-writer arm's full-policy deficit shrinks, and rule's flips to a gain
of 15 points on the paired test (p = 0.017) — the largest single 50 → 100 move
in the run. `rule/full` recovers exactly where it was worst at 50 tasks:
`pick_clean_then_place_in_recep` 5/23 → 13/23, `pick_and_place_simple` 13/18 →
17/18, `look_at_obj_in_light` 7/13 → 11/13.

The reading that fits: the full-policy mechanisms are data-hungry. Verification,
refinement and batch induction all consolidate *across* episodes, and with 50
episodes they consolidate mostly noise — §5 already recorded `rule/full`'s store
collapsing to a single entry at episode 25. Given 100 episodes the same
machinery has enough evidence to be net-positive for rule and much less harmful
for reflection. So §1's claim should be narrowed to: **at 50 evolving episodes**
`full` is worse for every LLM-written type; the ordering is not stable in the
stream length, and the write-policy ablation cannot be run at one scale only.

`raw` is the control that makes this legible — it has no LLM writer, its only
full-policy mechanism is utility deletion, and its full-over-minimal gap is
+5/+6 at both lengths. The mechanism that does bookkeeping is scale-invariant;
the mechanisms that call the writer are not.

### 6.3 A wrong procedure is not corrected by more evidence

`skill/full`'s microwave procedure from §2 is **still in the store at episode
100**, now at version 5, still identical in the step that breaks it:

```
go to the microwave
open the microwave if it is closed
move <obj> to the microwave      <- the agent is no longer holding <obj>
close the microwave
heat <obj> with the microwave    <- cannot succeed
```

`pick_heat_then_place_in_recep` goes 0/21 → 1/21. In between, `skill.judge`
fired 50 more times and `skill.refine` 5 times and the entry was revised to
version 5 without the false step ever being touched.

Worse, the second 50 episodes manufacture a *new* error of the same shape.
`examine_obj_with_lamp` under `full` ended at 50 episodes as a four-step
procedure that takes the object and puts it down next to the lamp, never using
it. By episode 100 the writer has "fixed" it by appending the missing action:

```
go to the location containing <obj>
take <obj> from the location
go to the location containing <lamp>
move <obj> to the location containing <lamp>   <- puts the object down
use <lamp>                                     <- object is not held
```

`look_at_obj_in_light` falls 8/13 → **3/13**, and it is the only family where
`skill/full` is now far below baseline (10/13). The repair addressed the
symptom the writer could see (no `use` action) and preserved the premise that
caused the failure.

Contrast `skill/minimal`, whose lamp skill kept the object in hand from the
start — *"use the lamp to examine the held object"* — and scores 11/13. Same
model, same episodes, same content type; the arm with more repair machinery is
the one holding the false belief.

The generalisation of §2 is therefore stronger than §2 could state: procedural
memory errors are not self-correcting under more data. The entry is replayed
deterministically, so it suppresses the very trajectories that would falsify
it, and the writer then judges its own output against episodes it caused. Raw
replay has no such fixed point.

### 6.4 Skill is the only type that gets worse with more data — retracted

> **Retracted by [§9.1](#91-what-the-third-and-fourth-points-refute).** At 150
> evolving tasks `skill` returns to 0.72 (minimal) and 0.64 (full), and the
> first-to-last paired test is `10/10, p = 1.000` under `full`. The section
> below is left as written because the two-point reading it describes was
> reasonable on two points; it is exactly the shape §9.2 shows this benchmark
> manufactures for free.

`skill/minimal` — the best arm at 50 episodes, 79.0% from four skills — drops to
70.0% (b/c 16/7, p = 0.093; suggestive, not significant). Its store grew 4 → 6
and its per-family profile flattened: `pick_cool` 12/15 → 8/15, `look` 13/13 →
11/13, `pick_two_obj` 10/10 → 7/10, against a gain of 7/21 → 10/21 on `heat`.
It is still the second-best minimal arm and still the cheapest LLM arm (33
writer calls), but §3's "compression works" now carries a scale caveat of its
own: the four-skill store was the good one, and adding two more entries cost
9 points.

Combined with §6.3, skill is the only content type that is worse at 100
episodes than at 50 under **both** policies (−9 minimal, −5 full).

### 6.5 Evolve-time success does not predict eval

Every arm ran the identical second-50 evolving tasks, so their evolve-time
success rates are directly comparable — and they invert the eval ordering.
`raw/minimal` solved 80% of the second-half evolving tasks and evaluates at
75.0%; `raw/full` solved 52% and evaluates at 81.0%. Evolve-time success is
measured against a store that is still growing and differs per arm, so it is a
property of the run, not of the memory. Do not use it as a cheap proxy.

## 7. Three passes over the *same* 50 tasks

Run 2026-08-12. Each arm re-ran the identical frozen evolving manifest
(`manifests/evolve_train_50_seed42.json`, same order) twice more, each pass
resuming the previous pass's store with `--evolve-step-offset` 50 then 100.
Epoch 1 is the §1–§5 run. Evaluation, seed, horizon and scaffold are unchanged,
so every number below is paired with every number above.

This is a **repetition** axis, not §6's amount-of-experience axis, and the two
must not be read together. From epoch 2 on, the nearest neighbour of an evolving
task is usually the agent's own epoch-1 memory of *that same task* — for `raw`, a
near-verbatim replay of its own trajectory. Nothing leaks into the test set:
`valid_unseen` is held out and untouched.

Raw outputs: `memsys_results/{minimal,full}_x{2,3}/`. The two policies ran
concurrently on separate nodes (~10.3 h each).


| arms       | policy  | ep1  | ep2  | ep3  | ep3 vs none | p         | ep1 → ep3 | p     | store 1→2→3   |
| ---------- | ------- | ---- | ---- | ---- | ----------- | --------- | --------- | ----- | ------------- |
| raw        | minimal | 75.0 | 74.0 | 77.0 | +19.0       | **0.003** | +2.0      | 0.832 | 38 → 44 → 44  |
| reflection | minimal | 65.0 | 68.0 | 56.0 | −2.0        | 0.885     | −9.0      | 0.211 | 39 → 63 → 93  |
| rule       | minimal | 53.0 | 54.0 | 56.0 | −2.0        | 0.875     | +3.0      | 0.711 | 16 → 22 → 37  |
| skill      | minimal | 79.0 | 81.0 | 73.0 | +15.0       | **0.024** | −6.0      | 0.307 | 4 → 10 → 10   |
| raw        | full    | 80.0 | 75.0 | 79.0 | +21.0       | **0.001** | −1.0      | 1.000 | 36 → 45 → 45  |
| reflection | full    | 54.0 | 64.0 | 59.0 | +1.0        | 1.000     | +5.0      | 0.424 | 43 → 71 → 112 |
| rule       | full    | 46.0 | 50.0 | 58.0 | +0.0        | 1.000     | +12.0     | 0.096 | 10 → 11 → 23  |
| skill      | full    | 64.0 | 75.0 | 75.0 | +17.0       | **0.006** | +11.0     | 0.061 | 10 → 16 → 24  |


**No arm's epoch 1 → 3 change is significant**, and no arm improves
monotonically. Against the ±3–6 noise floor, six of the eight move by less than
the floor or non-monotonically. Repeating the same 50 tasks is not a way to buy
accuracy.

### 7.1 The arms that recover are exactly the ones `full` broke

The two largest 1 → 3 moves are `rule/full` (+12.0, p = 0.096) and `skill/full`
(+11.0, p = 0.061) — the two arms §1 identified as `full`'s worst casualties.
`rule/full` was the study's only case of **significant harm** (−12.0, p = 0.043);
after two more passes it sits exactly at baseline. `skill/full` goes from +6.0
(ns) to +17.0 (p = 0.006).

Neither is significant on its own, but the pattern is the same one §6 found by
adding new tasks: `full`'s deficit is a small-store artefact that more episodes
partly repair, and it repairs about as well from **re-processing old experience**
as from new experience. The arms that were already working (`raw` both policies,
`skill/minimal`) gain nothing — they are saturated at epoch 1.

### 7.2 Dedup closes `raw`'s store; the writer arms have no such stop

The mechanism behind the store column, from the evolve logs
(duplicate-rejections / total rejections, per epoch):


| arm        | policy  | ep1  | ep2   | ep3       |
| ---------- | ------- | ---- | ----- | --------- |
| raw        | minimal | 3/3  | 37/37 | **40/40** |
| raw        | full    | 2/2  | 32/32 | **36/36** |
| reflection | minimal | 0/17 | 0/25  | 0/10      |
| rule       | full    | 0/5  | 0/5   | 0/10      |
| skill      | full    | 0/7  | 0/10  | 0/11      |


By epoch 3 **every single one of** `raw`**'s write attempts is rejected as an exact
duplicate** — 40/40 and 36/36 — and its store stops growing outright (44 → 44,
45 → 45; `created_at_step` confirms 0 and 1 new entries respectively). Verbatim
storage is closed under repetition: once a task's successful trajectory is in the
store, re-solving it adds nothing.

**The LLM-writer arms register zero duplicate rejections in any epoch.** Dedup is
a content-hash check, and a writer's paraphrase of the same experience is never
byte-identical to its last paraphrase, so nothing stops it. Their stores grow
linearly over 50 unchanging tasks: `reflection/full` 43 → 71 → 112,
`reflection/minimal` 39 → 63 → 93. Reflection ends with 93–112 entries derived
from 50 experiences, and it is the worst-performing pair of runs in the epoch
sweep (−9.0 and +5.0, both ns). This is §1's "damage is to entry quality, not
quantity" with a mechanism attached: the writer cannot tell it has already
written this down.

That is the one clearly actionable finding here. Semantic dedup — or any
same-source check at write time — would cost the writer arms nothing and is the
obvious next change to the write path.

### 7.3 Injected tokens stay flat, so none of this is a context effect

Injected memory tokens move by <5% across all three epochs for every arm
(`reflection/full` 942 → 927 → 903, `rule/full` 226 → 241 → 251), because the
1500-token injection budget binds well before the store size does. A store that
triples costs nothing at inference and buys nothing at inference.

## 8. Same 150-episode budget, three ways (75 tasks × 2)

Run 2026-08-13. A **fresh** chain: 75 `train` tasks
(`manifests/evolve_train_75_seed42.json`, positions [0,75) of the same seeded
permutation, so it nests inside the 50/100/150 sets), evolved from an **empty**
store, then the identical manifest in the identical order a second time
(`--evolve-step-offset 75`). Evaluation, seed, horizon and scaffold unchanged,
so every number is paired with every number above.

Raw outputs: `memsys_results/{minimal,full}_e75{,_x2}/`. The two policies ran
concurrently on separate nodes.

The point of 75 × 2 is that it spends **150 evolving episodes**, the same budget
as §6's 150 distinct tasks and §7's 50 × 3. Holding episodes fixed and varying
only how many *distinct* tasks they cover is what makes diversity and repetition
separable; §6 and §7 each vary one while holding the other, and neither can be
subtracted from the other without this third point.


| arm        | policy  | ep1  | p vs none | ep2  | p vs none | ep1 → ep2 | b/c   | p     |
| ---------- | ------- | ---- | --------- | ---- | --------- | --------- | ----- | ----- |
| raw        | minimal | 74.0 | **0.017** | 74.0 | **0.009** | +0.0      | 13/13 | 1.000 |
| reflection | minimal | 53.0 | 0.487     | 56.0 | 0.878     | +3.0      | 14/17 | 0.720 |
| rule       | minimal | 66.0 | 0.280     | 63.0 | 0.500     | −3.0      | 18/15 | 0.728 |
| skill      | minimal | 79.0 | **0.001** | 68.0 | 0.132     | −11.0     | 20/9  | 0.061 |
| raw        | full    | 72.0 | **0.038** | 77.0 | **0.004** | +5.0      | 12/17 | 0.458 |
| reflection | full    | 65.0 | 0.324     | 63.0 | 0.511     | −2.0      | 16/14 | 0.856 |
| rule       | full    | 66.0 | 0.215     | 67.0 | 0.163     | +1.0      | 15/16 | 1.000 |
| skill      | full    | 77.0 | **0.001** | 68.0 | 0.110     | −9.0      | 19/10 | 0.136 |


`none` = 58.0% throughout (no store, no writer; reused verbatim).

Only `raw` is significant against the baseline in both epochs and under both
policies. `skill` is significant at epoch 1 and loses it at epoch 2. `rule`'s
+8.0 at epoch 1 under `minimal` looks like the §6 recovery continuing and is
not: **b/c 17/25, p = 0.280**, discordant almost evenly in both directions.

### 8.1 At fixed cost, how the episodes are arranged does not matter

Final-epoch rate for each arrangement of 150 evolving episodes:


| arm        | policy  | 150 distinct (§9) | 75 × 2 (§8) | 50 × 3 (§7) |
| ---------- | ------- | ----------------- | ----------- | ----------- |
| raw        | minimal | 83.0              | 74.0        | 77.0        |
| reflection | minimal | 54.0              | 56.0        | 56.0        |
| rule       | minimal | 67.0              | 63.0        | 56.0        |
| skill      | minimal | 72.0              | 68.0        | 73.0        |
| raw        | full    | 80.0              | 77.0        | 79.0        |
| reflection | full    | 62.0              | 63.0        | 59.0        |
| rule       | full    | 62.0              | 67.0        | 58.0        |
| skill      | full    | 64.0              | 68.0        | 75.0        |


Paired, task by task, against the 75 × 2 column:


| arm        | policy  | vs 150 distinct | p     | vs 50 × 3    | p     |
| ---------- | ------- | --------------- | ----- | ------------ | ----- |
| raw        | minimal | −9.0 (21/12)    | 0.163 | −3.0 (12/9)  | 0.664 |
| reflection | minimal | +2.0 (15/17)    | 0.860 | +0.0 (19/19) | 1.000 |
| rule       | minimal | −4.0 (20/16)    | 0.618 | +7.0 (15/22) | 0.324 |
| skill      | minimal | −4.0 (19/15)    | 0.608 | −5.0 (15/10) | 0.424 |
| raw        | full    | −3.0 (18/15)    | 0.728 | −2.0 (17/15) | 0.860 |
| reflection | full    | +1.0 (17/18)    | 1.000 | +4.0 (19/23) | 0.644 |
| rule       | full    | +5.0 (15/20)    | 0.500 | +9.0 (11/20) | 0.150 |
| skill      | full    | +4.0 (14/18)    | 0.597 | −7.0 (21/14) | 0.311 |


**Sixteen paired tests, none significant; the smallest p is 0.150.** The largest
single gap is `raw/minimal`'s −9.0 in favour of 150 distinct tasks (p = 0.163),
which is the direction one would predict and still does not clear the bar.

The spread *within* a row — `skill/full` spans 64.0 / 68.0 / 75.0, `rule/minimal`
spans 67.0 / 63.0 / 56.0 — is comparable to the spread *between* content types
at any fixed arrangement. At this scale the content type is doing the work and
the curriculum is not. That is a useful negative result for anyone planning to
spend compute on curriculum design before fixing the write path.

### 8.2 `raw`'s store is bounded by newly solved tasks, not by episodes

§7.2 showed dedup closing `raw`'s store under repetition. The mechanism is
visible more precisely here, from the evolve logs (APPENDs / rejections /
of which duplicates):


| arm        | policy  | ep1            | ep2              | store 1 → 2 |
| ---------- | ------- | -------------- | ---------------- | ----------- |
| raw        | minimal | 46 / 3 / **3** | 13 / 46 / **46** | 46 → 59     |
| raw        | full    | 53 / 3 / **3** | 12 / 53 / **53** | 53 → 65     |
| reflection | minimal | 56 / 31 / 0    | 44 / 32 / **0**  | 56 → 100    |
| reflection | full    | 80 / 38 / 1    | 70 / 32 / **0**  | 68 → 130    |
| rule       | minimal | 52 / 9 / 0     | 32 / 10 / **0**  | 52 → 84     |
| skill      | minimal | 9 / 29 / 0     | 3 / 8 / **0**    | 6 → 8       |


In epoch 2, `raw` is rejected as a duplicate **exactly as many times as it has
entries** — 46 of 46, 53 of 53. Every task it re-solves that it already recorded
is refused, and every entry it does add is a task it *failed* in epoch 1 and
solved this time: 13 of 13 APPENDs under `minimal`, 12 of 12 under `full`, with
no exceptions in either. So `raw`'s store grows with the number of distinct tasks
it has ever solved, never with the number of episodes spent. Repetition costs it
storage proportional to its own improvement and nothing else.

The writer arms register **zero duplicate rejections in either epoch**, at 75
tasks exactly as at 50. `reflection/full` writes 150 APPENDs across the two
epochs and ends with 130 live entries derived from 75 experiences, while its
accuracy moves −2.0. This is §7.2's finding at 1.5× the scale, and it is still
the one clearly actionable item: dedup is a content hash, a paraphrase is never
byte-identical, and nothing else stops the writer.

### 8.3 `skill` regresses under repetition, and not from store bloat

`skill` is the only arm whose second pass moves more than the noise floor, and
it moves down under both policies: −11.0 (p = 0.061) and −9.0 (p = 0.136).
Neither is significant alone, but they agree in sign, size and — decisively —
in *where* the loss lands:


| run               | look  | simple | clean | cool  | heat      | two_obj |
| ----------------- | ----- | ------ | ----- | ----- | --------- | ------- |
| skill/minimal ep1 | 13/13 | 18/18  | 17/23 | 13/15 | **12/21** | 6/10    |
| skill/minimal ep2 | 11/13 | 16/18  | 17/23 | 11/15 | **5/21**  | 8/10    |
| skill/full ep1    | 12/13 | 17/18  | 14/23 | 14/15 | **12/21** | 8/10    |
| skill/full ep2    | 11/13 | 18/18  | 15/23 | 9/15  | **7/21**  | 8/10    |


Both policies lose `pick_heat_then_place_in_recep` and little else — 12/21 → 5/21
and 12/21 → 7/21 — the family §2 identified as the one abstraction wins that
verbatim replay cannot.

The store rules out the obvious explanation. `skill` ends epoch 2 with **8 and 5
live entries**, having added 3 in each case; it is not being drowned in
accumulated text the way `reflection` is. The second pass is rewriting a small
set of procedures that already worked, and the rewrite is worse. That is the
same failure §6.3 documented for `rule` — a wrong procedure re-derived from more
evidence of the same kind — appearing in the arm that was the study's best.

### 8.4 Provenance: part of this run was re-executed

At 06:41:35 a co-tenant job finishing on the same node killed this chain's vLLM
servers with a host-wide `pkill -f`. The `full` arms in flight kept running
against a dead backend and **reported completed runs with** `rc=0` **and full**
`summary.json` **files** — the agent swallows connection errors and scores each
episode a failure. `reflection/full` reported 10.0% that way; re-run cleanly it
is 65.0%, a 55-point artefact that nothing in the exit status or the logs
flagged. `raw/full` epoch 2 reported 0.0%.

Everything written after that moment was discarded and re-run from the last
clean state (`memsys_results/_contaminated_20260813_0641/` retains it). The
numbers above come only from arms whose `summary.json` predates the kill or
which were re-run afterwards; the tell is eval episodes completing in under 6
seconds, and every arm in §8 was checked for it. The failure mode is now
described in RUN_ALFWORLD.md §5.

## 9. Carrying the stream to 150 and 200 evolving tasks

Run 2026-08-13. Two further legs of the same chain: positions [100,150) and
[150,200) of the one seeded permutation
(`manifests/evolve_train_{100to150,150to200}_seed42.json`), each resuming the
previous leg's store with `--evolve-step-offset` 100 then 150. Evaluation, seed,
horizon and scaffold unchanged, so every number is paired with every number
above. `none` is reused verbatim (58.0%).

The 200 leg was run for `reflection` and `rule` only — the two arms that still
looked like they were moving after 150. Raw outputs:
`memsys_results/{minimal,full}_e{150,200}/`.


| arm        | policy  | rate      | Δ vs none | b/c   | McNemar p  | vs previous leg | p         | store | inj. tok |
| ---------- | ------- | --------- | --------- | ----- | ---------- | --------------- | --------- | ----- | -------- |
| raw        | minimal | **83.0%** | **+25.0** | 10/35 | **0.0002** | +8.0            | 0.152     | 116   | 1261     |
| reflection | minimal | 54.0%     | −4.0      | 22/18 | 0.636      | −13.0           | **0.035** | 111   | 990      |
| rule       | minimal | 67.0%     | +9.0      | 13/22 | 0.176      | +9.0            | 0.176     | 36    | 259      |
| skill      | minimal | 72.0%     | +14.0     | 9/23  | **0.020**  | +2.0            | 0.860     | 13    | 1389     |
| raw        | full    | **80.0%** | **+22.0** | 10/32 | **0.0009** | −1.0            | 1.000     | 95    | 1244     |
| reflection | full    | 62.0%     | +4.0      | 12/16 | 0.572      | +0.0            | 1.000     | 124   | 958      |
| rule       | full    | 62.0%     | +4.0      | 17/21 | 0.627      | +1.0            | 1.000     | 34    | 294      |
| skill      | full    | 64.0%     | +6.0      | 15/21 | 0.405      | +5.0            | 0.442     | 6     | 1300     |


At 200 evolving tasks (`reflection` and `rule` only):


| arm        | policy  | rate  | Δ vs none | b/c   | McNemar p | 150 → 200 | p         | store |
| ---------- | ------- | ----- | --------- | ----- | --------- | --------- | --------- | ----- |
| reflection | minimal | 70.0% | +12.0     | 13/25 | 0.073     | +16.0     | **0.011** | 151   |
| rule       | minimal | 64.0% | +6.0      | 16/22 | 0.418     | −3.0      | 0.701     | 53    |
| reflection | full    | 74.0% | +16.0     | 11/27 | **0.014** | +12.0     | **0.036** | 162   |
| rule       | full    | 64.0% | +6.0      | 14/20 | 0.392     | +2.0      | 0.860     | 52    |


Holm-Bonferroni over the four 200-leg tests against the baseline: thresholds
0.0125 / 0.0167 / 0.025 / 0.05, smallest p = 0.014. **Nothing survives.**
`reflection/full` misses by 0.0014.

### 9.1 What the third and fourth points refute


| arm        | policy  | e50  | e100 | e150     | e200 | first → last  | p          |
| ---------- | ------- | ---- | ---- | -------- | ---- | ------------- | ---------- |
| raw        | minimal | 75.0 | 75.0 | **83.0** | —    | +8.0 (11/19)  | 0.201      |
| raw        | full    | 80.0 | 81.0 | 80.0     | —    | +0.0 (15/15)  | 1.000      |
| skill      | minimal | 79.0 | 70.0 | 72.0     | —    | −7.0 (18/11)  | 0.265      |
| skill      | full    | 64.0 | 59.0 | 64.0     | —    | +0.0 (10/10)  | 1.000      |
| reflection | minimal | 65.0 | 67.0 | 54.0     | 70.0 | +5.0 (15/20)  | 0.500      |
| reflection | full    | 54.0 | 62.0 | 62.0     | 74.0 | +20.0 (7/27)  | **0.0008** |
| rule       | minimal | 53.0 | 58.0 | 67.0     | 64.0 | +11.0 (11/22) | 0.080      |
| rule       | full    | 46.0 | 61.0 | 62.0     | 64.0 | +18.0 (10/28) | **0.005**  |


Two claims made earlier in this document do not survive the extra points.

**§6.4 is retracted.** "Skill is the only type that gets worse with more data"
rested on 79 → 70 and 64 → 59. The third point returns both to where they
started: `skill/full` is 64.0 → 59.0 → 64.0 with a first-to-last b/c of exactly
**10/10, p = 1.000**, and `skill/minimal` 79 → 70 → 72 at p = 0.265. There is no
decline to explain. What §6.4 actually described was one draw from the
distribution §9.2 quantifies.

`rule`**'s recovery is one step, not a trend.** `rule/full` climbs 0.46 → 0.61 →
0.62 → 0.64, which reads as monotone. Only the first step is real: e50 → e100 is
p = 0.017 (§6.2), and every step after it is p = 1.000 then p = 0.860. It rises
out of the significant *harm* §1 recorded and then stops at the baseline — from
e100 onward it is not distinguishable from having no memory at all (p = 0.761,
0.627, 0.392). Under `minimal` the fourth point breaks the shape outright:
0.53 → 0.58 → 0.67 → **0.64**. The honest statement is that `full`'s early
damage is repaired by more data and nothing further is bought.

`reflection/full`'s +20.0 first-to-last (p = 0.0008) is the one four-point trend
that survives its own paired test. It is still not a claim this document can
make — see §9.2.

### 9.2 The churn floor, measured

Every section above hedges against a "±3–6 point noise floor" inherited from
SAGE. The four-point sweep measures it directly, because adjacent legs differ
only in 50 extra evolving episodes and can be paired task by task.


| adjacent pair                  | Δ     | b   | c   | tasks flipped | p         |
| ------------------------------ | ----- | --- | --- | ------------- | --------- |
| reflection/full e100 → e150    | 0.0   | 15  | 15  | **30**        | 1.000     |
| reflection/minimal e100 → e150 | −13.0 | 23  | 10  | 33            | **0.035** |
| reflection/minimal e150 → e200 | +16.0 | 10  | 26  | 36            | **0.011** |
| reflection/minimal e100 → e200 | +3.0  | 16  | 19  | 35            | 0.736     |
| rule/full e150 → e200          | +2.0  | 15  | 17  | 32            | 0.860     |
| raw/full e100 → e150           | −1.0  | 11  | 10  | 21            | 1.000     |


Roughly **a third of the 100 eval tasks change outcome between adjacent legs of
the same arm**, whichever arm. `reflection/full` flips 30 tasks to arrive at
precisely the same score. And `reflection/minimal` produces two *nominally
significant* adjacent moves in **opposite directions** — −13.0 at p = 0.035 then
+16.0 at p = 0.011 — whose endpoints are identical (+3.0, p = 0.736).

That last row is the one to remember. A single significant adjacent-point jump
is not evidence of anything here: this benchmark manufactures them in both
directions at this sample size. It is why `reflection/full`'s 0.62 → 0.74 at
p = 0.036 is reported above and not believed — it has the same magnitude and the
same shape as a move we can watch reverse. Effects worth trusting on n = 100
need to look like `raw`'s: large, and stable across points rather than achieved
at one of them.

## 10. Budgeting by successes, and discarding failures

Run 2026-08-13. Two changes to the evolving loop, via new flags on
`scripts/run_alfworld.py` (both default off):

- `--success-only-writes` — a failed episode is dropped before
`system.observe()` is called, so it drives no extraction, and under `full` no
verification, refinement, confidence pruning, utility pruning or
batch-induction buffering either. Failures normally act as *counter*-evidence
against whatever was injected; here they act as nothing.
- `--evolve-until-successes 100` — the stream is budgeted by successes, not
attempts. The step counter advances only on written episodes, so `full`'s
every-25-episode batch induction also counts successes.

Tasks are drawn in permutation order from `manifests/evolve_train_300_seed42.json`,
a strict superset of the existing chain (positions [0,50), [50,100), [100,150)
and [150,200) match the earlier manifests exactly), from an **empty** store.

`raw` is not run: `RawTrajectorySystem.observe` already keeps only
`best_success()`, so the flag is a no-op for it. That is itself worth stating —
**the one arm that works is the one that was already doing this ablation**, and
§10 is the test of whether transplanting its input filter rescues the others.

Raw outputs: `memsys_results/{minimal,full}_succ100/`.


| arm        | policy      | tasks spent | rate      | Δ vs none | b/c   | McNemar p   | vs own e100 | p           | store | writer calls |
| ---------- | ----------- | ----------- | --------- | --------- | ----- | ----------- | ----------- | ----------- | ----- | ------------ |
| reflection | minimal     | 150         | 73.0%     | +15.0     | 7/22  | **0.008**   | +6.0        | 0.405       | 74    | 150          |
| **rule**   | **minimal** | 154         | **84.0%** | **+26.0** | 5/31  | **0.00001** | **+26.0**   | **<0.0001** | 26    | 172          |
| skill      | minimal     | 176         | 56.0%     | −2.0      | 17/15 | 0.860       | **−14.0**   | **0.024**   | 13    | 120          |
| reflection | full        | 135         | 70.0%     | +12.0     | 9/21  | 0.043       | +8.0        | 0.152       | 55    | 271          |
| rule       | full        | 147         | 70.0%     | +12.0     | 13/25 | 0.073       | +9.0        | 0.108       | 41    | 279          |
| skill      | full        | 136         | 68.0%     | +10.0     | 10/20 | 0.099       | +9.0        | 0.150       | 21    | 257          |


Holm-Bonferroni over the six baseline tests: `rule/minimal` (0.00001 vs 0.0083)
and `reflection/minimal` (0.0081 vs 0.0100) survive; `reflection/full` (0.043 vs
0.0125) does not.

"Tasks spent" is how many attempts it took to bank 100 successes — 135 to 176,
i.e. evolve-time success rates of 74% down to 57%. The arms are no longer
matched on attempts, by construction.

### 10.1 `rule/minimal` reaches `raw` parity

**84.0%, +26.0 over baseline, p = 0.00001** — the largest effect any LLM-written
memory produces anywhere in this study, from a store of **26 live entries**.
Against every other point `rule/minimal` has ever scored:


| comparison                           | Δ        | b/c       | p            |
| ------------------------------------ | -------- | --------- | ------------ |
| vs none (58.0)                       | +26.0    | 5/31      | **0.00001**  |
| vs own e50 (53.0)                    | +31.0    | 4/35      | **<0.00001** |
| vs own e100 (58.0)                   | +26.0    | 3/29      | **<0.00001** |
| vs own e150 (67.0)                   | +17.0    | 7/24      | **0.003**    |
| vs own e200 (64.0)                   | +20.0    | 2/22      | **0.00004**  |
| **vs** `raw`**/minimal e150 (83.0)** | **+1.0** | **13/14** | **1.000**    |


The last row is the finding. `rule` was §5's weakest content type — never
*significantly* beating baseline in eight previous runs (its best was 67.0 at
p = 0.176) and significantly **harmful** under `full` at 50 tasks — and with
failures withheld it becomes indistinguishable from verbatim trajectory replay. Unlike §9.2's suspect jumps, this is not one adjacent step: it clears
every one of its own prior points by 17–31 points, and it clears the ±10 band
those points scatter over.

Per family it is the first LLM arm to fix `pick_heat_then_place_in_recep`, §2's
graveyard for abstraction — 10/21 against a 9/21 baseline and 5/21 at e50 — while
holding `pick_clean` 21/23 and `pick_cool` 14/15.

### 10.2 The same intervention breaks `skill`

`skill/minimal` moves the other way, and significantly: **56.0%, below the
no-memory baseline**, down 14 points from its own e100 (b/c 24/10, p = 0.024).
It also needed the most attempts of any arm (176) because it was getting *worse
as it went*: on the 150 evolving tasks it shares with the fixed-count legs it
solved 0.57 against their 0.72 (b/c 36/14, **p = 0.0026**) — the only significant
evolve-time change in the whole table, and it is negative.

Where the loss lands is familiar:


| family                 | none  | skill/minimal e50 | skill/minimal succ100 |
| ---------------------- | ----- | ----------------- | --------------------- |
| pick_cool_then_place   | 5/15  | 12/15             | **3/15**              |
| pick_heat_then_place   | 9/21  | 7/21              | **2/21**              |
| pick_two_obj_and_place | 7/10  | 10/10             | **4/10**              |
| look_at_obj_in_light   | 10/13 | 13/13             | 13/13                 |
| pick_and_place_simple  | 18/18 | 18/18             | 18/18                 |


This is §2 and §8.3 again — a small procedural store rewritten into a worse one —
but with the sharpest possible framing: **it happened on a diet of nothing but
successful trajectories.** Whatever damages `skill`'s procedures, exposure to
failure is not it, and removing failure does not prevent it. A wrong procedure
induced from successes is still wrong, and with no counter-evidence admitted
there is now nothing in the loop that could ever contradict it.

### 10.3 No consistent evolve-time story

On the tasks each `succ100` run shares with the fixed-count legs, paired:


| arm        | policy  | fixed-count | succ100  | b/c   | p          |
| ---------- | ------- | ----------- | -------- | ----- | ---------- |
| reflection | full    | 0.61        | 0.74     | 11/29 | **0.006**  |
| rule       | full    | 0.53        | 0.68     | 13/35 | **0.002**  |
| skill      | full    | 0.71        | 0.74     | 15/18 | 0.728      |
| reflection | minimal | 0.60        | 0.67     | 19/29 | 0.193      |
| rule       | minimal | 0.55        | 0.65     | 23/39 | 0.056      |
| skill      | minimal | 0.72        | **0.57** | 36/14 | **0.0026** |


Four of six improve during evolving; the two `full` writer arms significantly so,
while their held-out eval does not move. That dissociation — better on the
distribution being trained on, no transfer — would be the interesting reading,
and `skill` refuses it in both directions: flat under `full`, significantly worse
under `minimal`. **The effect tracks the content type, not the policy and not the
intervention.** `rule` gains most, `reflection` middling, `skill` is harmed.

### 10.4 Two variables move at once

§10 changes the amount of successful experience (~60 → 100) *and* removes
failures from the write path. `rule/minimal`'s +26.0 cannot be attributed to
either alone from this table.

The comparison that separates them is a third condition — **100 successes
banked, failures still observed** — which is one flag away
(`--evolve-until-successes 100` without `--success-only-writes`) and needs only
`rule/minimal` and `reflection/minimal`. Until it is run, §10.1 establishes that
*some* property of this configuration produces raw-parity from `rule`, and not
which one. Given that it is the study's only large positive result for an
LLM-written memory, that experiment is the highest-value one outstanding.

## 11. The mirror: budgeting by failures, and discarding successes

Run 2026-08-14, the sign-flipped §10. Two more flags on `scripts/run_alfworld.py`:

- `--failure-only-writes` — a *successful* episode is dropped before
`system.observe()`. The store is built out of failure evidence alone.
- `--evolve-until-failures 100` — the stream is budgeted by failures, so the
step counter (and `full`'s every-25-episode induction) counts failures.

Same frozen `valid_unseen` 100, same `none` baseline (58.0%), same permutation —
tasks come from `manifests/evolve_train_600_seed42.json`, which nests the 300-task
§10 manifest and every earlier one exactly, from an **empty** store. Failures are
the scarcer outcome here, so 100 of them costs 184–257 tasks rather than §10's
135–176. `raw` is again not run, and this time for the opposite reason:
`RawTrajectorySystem` keeps only `best_success()`, so under this filter it writes
nothing at all and *is* the `none` arm.

Raw outputs: `memsys_results/{minimal,full}_fail100/`.


| arm            | policy      | tasks spent | rate      | Δ vs none | b/c   | McNemar p | vs own succ100 | p            | store | writer calls |
| -------------- | ----------- | ----------- | --------- | --------- | ----- | --------- | -------------- | ------------ | ----- | ------------ |
| reflection     | minimal     | 188         | 47.0%     | −11.0     | 29/18 | 0.144     | **−26.0**      | **0.0002**   | 65    | 146          |
| rule           | minimal     | 184         | 46.0%     | −12.0     | 22/10 | 0.050     | **−38.0**      | **<0.00001** | 37    | 168          |
| **skill**      | **minimal** | **235**     | **57.0%** | **−1.0**  | 18/17 | 1.000     | +1.0           | 1.000        | **0** | **0**        |
| reflection     | full        | 217         | 66.0%     | +8.0      | 14/22 | 0.243     | −4.0           | 0.557        | 2     | 282          |
| rule           | full        | 201         | 56.0%     | −2.0      | 18/16 | 0.864     | **−14.0**      | **0.024**    | 2     | 250          |
| skill          | full        | 257         | 69.0%     | +11.0     | 14/25 | 0.108     | +1.0           | 1.000        | 2     | 41           |


No arm beats the baseline. Under Holm over the six baseline tests nothing
survives — `rule/minimal`'s 0.050 needs 0.0083. The result is not "failures make
memory worse than successes make it better"; it is that **three different
mechanisms each independently prevent failure-only memory from existing at all**,
and only one of the six cells ends up carrying real content into evaluation.

### 11.1 `skill` cannot write from failure, by construction

`skill/minimal` made **zero writer calls across 235 tasks** and ended with an
empty store. `SkillWriter.propose` returns early when `episode.successes()` is
empty (`memsys/writers.py:571`) — a procedure needs at least one working path —
and with one rollout per task every observed episode is a pure failure. The LLM
was never called.

So that row is not a memory result. It is an **unintentional re-run of the
no-memory baseline**, on the same 100 eval tasks, with a different random seam
through vLLM — and it is the most useful number in the table:


| | rate | vs the other |
| --- | --- | --- |
| `none` baseline (2026-08-09) | 58.0% | — |
| `skill/minimal` fail100, empty store (2026-08-14) | 57.0% | b/c 18/17, p = 1.000 |


**Two runs of the identical no-memory configuration land 1 point apart while
disagreeing on 35 of 100 tasks.** That is §9.2's churn floor measured a second
time, and it is the yardstick every delta in this document should be held
against: a 35-task disagreement is what *no intervention whatsoever* produces.

### 11.2 Under `full`, the utility floor deletes everything, deterministically

The three `full` cells all end with **2 live entries** after 28–54 rows were
written. This is not the writer failing to produce; `reflection/full` proposed
104 APPENDs and 103 DELETEs, and **86 of those deletions are utility-floor
deletions** (45 of 66 for `rule/full`, 8 of 8 for `skill/full`).

The mechanism is exact and was predictable from the code. `_utility_prune`
deletes an entry once `n_retrieved >= 5` and its retrieval-success rate falls
below 0.20 (`memsys/config.py:113`), and `record_usage` is fed
`episode.any_success` — which under this filter is **always False**, because
successful episodes never reach `observe()`. Utility is `(hits+1)/(uses+2)`, so
every entry walks a fixed path to `1/7 = 0.14` and is deleted the fifth time it
is retrieved. Every DELETE reason string in the log reads `utility 0.14 below
floor`.

`full` therefore does not evaluate failure-derived memory — it evaluates an
almost-empty store, and lands where an almost-empty store should: +8.0, −2.0,
+11.0, none significant, all within the §11.1 churn floor. Worth stating plainly
because it looks like a result and is not one: **`skill/full`'s +11.0 comes from
two entries.**

### 11.3 Where failure-only memory does exist, it is harmful

That leaves `reflection/minimal` (65 entries, 981 injected tokens) and
`rule/minimal` (37 entries) as the only cells where a substantial failure-built
store actually reaches the agent. Both land **below the no-memory baseline**, and
both are far below their own §10 counterparts:


| cell               | fail100  | succ100 | e100 | e50  | vs succ100 p | vs e100 p  |
| ------------------ | -------- | ------- | ---- | ---- | ------------ | ---------- |
| reflection/minimal | **47.0** | 73.0    | 67.0 | 65.0 | **0.0002**   | **0.0029** |
| rule/minimal       | **46.0** | 84.0    | 58.0 | 53.0 | **<0.00001** | **0.043**  |


`rule/minimal` is the sharpest contrast in the study: **84.0% built from 100
successes, 46.0% built from 100 failures**, same arm, same policy, same
permutation, same evaluation — a 38-point swing produced entirely by which
episodes the writer was allowed to see. Per family the damage concentrates
exactly where §10.1's gain did: `pick_heat_then_place_in_recep` goes 10/21 →
4/21, and `reflection/minimal` takes it to **1/21** against a 9/21 baseline.

### 11.4 The evolve-time story, unlike §10.3, is consistent

Both chains start at position 0 of the same permutation, so their task prefixes
are identical in content *and* order — a properly paired evolve-time comparison,
which §10.3's pooled controls were not:


| cell               | shared tasks | succ100  | fail100  | b/c   | p          |
| ------------------ | ------------ | -------- | -------- | ----- | ---------- |
| reflection/minimal | 150          | 0.67     | **0.51** | 38/14 | **0.0012** |
| rule/minimal       | 154          | 0.65     | **0.47** | 45/17 | **0.0005** |
| skill/minimal      | 176          | 0.57     | 0.57     | 35/36 | 1.000      |
| reflection/full    | 135          | 0.74     | **0.56** | 36/12 | **0.0007** |
| rule/full          | 147          | 0.68     | **0.56** | 31/14 | **0.016**  |
| skill/full         | 136          | 0.74     | **0.60** | 31/13 | **0.0096** |


Five of six are significantly worse during evolving, and the sixth is
`skill/minimal` — the cell with no memory at all, which is flat to the
hundredth. The agent is being hurt *while it runs*, not only at evaluation, and
the one cell that cannot be hurt is the one carrying nothing.

### 11.5 What §11 does and does not settle

It does not isolate a single cause: §10.4's confound is mirrored here (the
amount of successful experience also drops, since successes are discarded), so
"failure content is bad" and "the absence of success content is bad" remain
entangled. `reflection/minimal` and `rule/minimal` at fail100 saw 88 and 84
successful episodes go by unused.

What it does settle is narrower and sturdier: **on this benchmark, with this
scaffold, none of the three write paths converts pure failure evidence into
memory that helps.** One cannot write from failure at all, one deletes what it
writes by a deterministic rule, and the one that accumulates ends up 26–38 points
below the same machinery fed successes. Together with §10 that makes the input
filter, not the content type, the largest single lever measured in this study —
`rule/minimal` spans 46.0% to 84.0% across the two.

## 12. Changing the *writer* model, holding the actor fixed

Every section above uses one model for both jobs: Qwen3.5-9B acts and writes the
memory. This one separates them. The actor stays Qwen3.5-9B on local vLLM; only
the memory writer moves, to `openai/gpt-5.6-terra` behind the Perplexity gateway
(`RUN_ALFWORLD.md` §3 has the wiring, which is not the usual OpenAI-compatible
chat API). Same `minimal` policy, same 50 evolving tasks, same frozen 100
`valid_unseen` evaluation, `MEMSYS_TAG_SUFFIX=_gpt56terra`.

`none` and `raw` are not re-measured — neither calls a writer LLM, so the writer
model cannot reach them — and their cells are copied from the `minimal` run
(marked in place by `REUSED_FROM_QWEN_RUN.txt`).

### 12.1 Two of three content types improve, one does not move

Paired McNemar over the same 100 ordered tasks, writer-vs-writer:

| arm | Qwen writer | gpt-5.6-terra writer | delta | b/c | p |
| --- | --- | --- | --- | --- | --- |
| reflection | 65.0% | **81.0%** | +16.0 | 8/24 | 0.007 |
| rule | 53.0% | **77.0%** | +24.0 | 9/33 | **0.000** |
| skill | 79.0% | 74.0% | −5.0 | 17/12 | 0.458 |

`b` = Qwen's writer produced memory that solved the task and gpt-5.6's did not;
`c` = the reverse.

**The writer model is a larger lever on `rule` than any write-mechanism change
measured in §1–§5.** `rule/minimal` was the weakest content type in this study
(§5, 53.0%, statistically indistinguishable from the 58.0% baseline); with a
frontier writer it becomes the second strongest. Nothing about the content
*type* changed — the schema, the prompt, the policy and the actor are identical.
What changed is that the trigger/directive pairs are now correct.

**`skill` does not move.** −5.0 at p = 0.458 with 17/12 discordant is churn in
both directions, and it is the one arm whose Qwen number was already high
(79.0%). Note the writer was called 34 times, not 50: `SkillWriter.propose`
returns early without a successful rollout, so the arm only sees the episodes
that went right, and at one rollout per task that ceiling binds regardless of
who writes. Whatever limits `skill` here, it is not writer quality.

### 12.2 The better writer writes less, and repeats itself less

| arm | live items Q → G | rows Q → G | merge calls Q → G | injected tokens Q → G |
| --- | --- | --- | --- | --- |
| reflection | 39 → 21 | 61 → 21 | 23 → **0** | 1022 → 790 |
| rule | 16 → 28 | 33 → 32 | 17 → **4** | 244 → 407 |
| skill | 4 → 9 | — | 0 → 0 | 1430 → 1430 |

The merge counts are the striking column. `reflection` under Qwen needed 23
merges across 50 writes — nearly half of everything it proposed collided with
something already stored — and produced 61 rows to keep 39 alive. Under
gpt-5.6 it proposed 21 items, kept all 21, and **never once proposed a
duplicate**. The same pattern holds for `rule` (17 → 4).

That is worth stating carefully, because it cuts against the natural reading of
§1. The merge machinery there looked like a mechanism that damages memory; part
of what it was actually doing was cleaning up after a writer that could not tell
it had already written something. A better writer does not need the cleanup, so
the mechanism has less to damage. These runs cannot separate those two stories —
that needs `full` under both writers, which was not run.

### 12.3 Cost

$1.14 for the whole sweep: 350k prompt + 37k completion tokens across the three
arms at terra's $2/$12 per 1M. Wall-clock went *down* (reflection 138 → 91 min),
because the writer no longer competes with the actor for the same GPU.

## Threats to validity

- **§12 is one run per cell, and `skill`'s −5.0 is inside the noise floor.**
§9.2's churn floor is ±3–6 points, so `reflection` (+16.0) and `rule` (+24.0)
clear it and `skill` (−5.0, p = 0.458) does not. Read §12.1 as "two arms improved,
one is unresolved" — *not* as evidence that a frontier writer hurts `skill`.
- **§12's `none` and `raw` cells were not measured in that session.** They are
copies of the `minimal` run, on the argument that neither arm calls a writer
LLM. The argument is sound for the memory path, but it means the §12 baseline
column was collected on different hardware at a different time from the three
arms compared against it. The writer-vs-writer tests in §12.1 are unaffected —
those are paired against the Qwen *arm*, not against `none`.
- **§12's evolving phases are not comparable episode-for-episode.** A better
writer improves the memory that is retrieved *during* evolving, so the actor's
evolve-time success rate moves too (`reflection` 0.50 → 0.82). The evolving
streams therefore diverge after the first write, and the two arms did not see
the same trajectories even though they saw the same task list. This is the
intended mechanism, not a confound, but it does mean §12 measures the whole
loop and cannot attribute the gain to write quality alone.
- **§12 adds 3 more arm-vs-arm tests** to a study already over 40. `rule`
(p = 0.000) survives any plausible correction; `reflection` (p = 0.007) survives
Holm within §12 but is not comfortable study-wide.
- **§11's `full` cells are uninformative about content.** With 2 live entries
they measure an empty store, and their spread (−2.0 to +11.0) sits inside the
§11.1 churn floor. Reading `skill/full`'s +11.0 as an effect of
failure-derived skills would be a mistake; a real test needs
`utility_deletion` disabled, or `record_usage` fed the true outcome even when
the writer is not.
- **§11 moves the same two variables as §10** (§11.5), in the same direction:
failures-only also means ~90 successful episodes discarded. The clean design
is the third condition named in §10.4 plus its mirror — outcome budget held
fixed, filter varied alone.
- **§11's `skill/minimal` row is a baseline, not an arm.** It is reported
because an accidental replication of the no-memory configuration is worth more
than the empty cell it came from, but it is not evidence about `skill`.
- **§7's three epochs are one seed and one order.** The manifest order is
identical in all three passes, so an epoch effect and an order effect are not
separable. Nothing in §7 reaches significance anyway.
- **n = 100, single seed.** SAGE measured a ±3–6 point noise floor across
identical ALFWorld runs, so read the paired tests, not the deltas. Effects
under ~10 points are not resolvable here; ≥3 seeds are needed.
- **One rollout per evolving task.** `Episode.outcome()` is therefore only ever
`all_success` or `all_failure`; Reflection's `from_contrast` mode never fired,
and skill could not write from failed episodes.
- **Writer model = actor model.** Qwen3.5-9B writes the memory it later
consumes, so content quality and consumption ability are confounded. That is
what the separate Memory Writing Model axis is for.
- **Batch induction is delete-heavy for rule** (20 DELETE vs 30 APPEND; the
store collapsed to 1 entry at episode 25 before rebuilding to 10). This is the
writer choosing to delete, *not* appends failing the grounding check —
rejections were only 5.
- **§8.1's three arrangements differ in provenance, not only in arrangement.**
75 × 2 is a fresh chain from an empty store; 150-distinct and 50 × 3 are both
resumptions of the 50-task chain. So "arrangement" and "started from scratch
vs continued" are confounded in that table. The confound cuts against the
negative result rather than manufacturing it — a fresh chain has every reason
to land somewhere else, and it did not — but a clean design would re-run all
three from empty stores.
- **§8.1's 16 tests are unadjusted for multiplicity.** With no correction and no
p below 0.150, correction could only weaken the comparisons further, so the
negative result stands; the same licence would not extend to a positive one.
- **§8's** `skill` **regression is two runs, not a replication.** `minimal` and
`full` agree on sign, size and family, which is why §8.3 states it, but they
share the manifest, the order and the seed. Three seeds would settle whether
the second pass reliably damages `pick_heat_then_place_in_recep` or whether
one bad rewrite is being counted twice.
- **§9's legs are resumptions too, so the four-point curves are not a curve.**
Every point conditions on the store the previous leg left behind, exactly as
§6 notes for 100. The four points bound a trend; they do not trace one.
- **§10 moves two variables at once** (§10.4), and its `succ100` runs start from
an empty store while the `*_e100` runs they are compared against are the first
leg of the chain — that part is matched, but the `e150`/`e200` comparisons in
§10.1 are against resumed stores. The one comparison free of this,
`succ100` vs `e100`, is also the one with the largest effect.
- **§10's evolve-time comparisons pool three separate runs as the control.**
The "fixed-count" column in §10.3 is assembled from the e50/e100/e150/e200
legs, which are four resumptions, against one continuous `succ100` run. At a
given task the two differ in more than the success filter — the amount and the
provenance of the store also differ — so those p-values overstate how cleanly
the intervention is isolated.
- **§9 and §10 add 10 more baseline tests to a document that never corrects
across sections.** Holm is applied within each new section (4 tests in §9, 6
in §10) and `rule/minimal` survives comfortably, but the study-wide count of
arm-vs-baseline tests is now over 40. Only `raw` (every leg) and
`rule/minimal` under `succ100` have effects large enough that no plausible
correction touches them.
- `rule/full`**'s eval at 150 was re-run alone.** Its evolving phase completed
and saved its store before a co-tenant's `pkill` took the servers down
(same incident as §8.4); only the evaluation was repeated, against the
preserved store, and `summary.json` carries `evolve_fields_reconstructed: true` because the evolve-side fields were rebuilt from the retained
`evolve_log.jsonl`. `store.jsonl` is byte-identical to its pre-incident backup.
- `minimal_e150` **ran on A100s; every other leg ran on H100s.** Same model,
same vLLM config, same sampling parameters. This should be far below the churn
floor §9.2 measures, but it is one more uncontrolled difference sitting under
a table of paired tests.
- **§6 is a resumption, not an independent 100-task run.** Each arm continued
from its own 50-task store, so its second half is conditioned on whatever that
store had become. A stream-length effect and a path-dependence effect are not
separable here; only a fresh run over 100 evolving tasks from an empty store
would separate them. The two lengths therefore bound the trend, they do not
establish a curve — and with two points and a ±3–6 noise floor, the 50 → 100
deltas under ~10 points (every arm except `rule/full`) are not resolvable
either.


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
the writer arms' stores grow without bound.

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

| arm | policy | rate | Δ | b/c | McNemar p | store | inj. tok | writer calls |
|---|---|---|---|---|---|---|---|---|
| none | — | 58.0% | — | — | — | — | 0 | — |
| raw | minimal | 75.0% | +17.0 | 11/28 | **0.009** | 38 | 1302 | — |
| reflection | minimal | 65.0% | +7.0 | 17/24 | 0.349 | 39 | 1022 | 73 |
| rule | minimal | 53.0% | −5.0 | 16/11 | 0.442 | 16 | 244 | 67 |
| skill | minimal | **79.0%** | **+21.0** | 6/27 | **0.0002** | 4 | 938 | 36 |
| raw | full | **80.0%** | **+22.0** | 7/29 | **0.0002** | 36 | 1205 | — |
| reflection | full | 54.0% | −4.0 | 18/14 | 0.597 | 43 | 942 | 145 |
| rule | full | 46.0% | −12.0 | 21/9 | **0.043** | 10 | 226 | 136 |
| skill | full | 64.0% | +6.0 | 19/25 | 0.451 | 10 | 1392 | 103 |

Significant: `raw` (both policies), `skill/minimal`, and `rule/full` — the last
being significant **harm**. Reflection never reaches significance in either
direction.

## 1. More write mechanism made every LLM-written memory worse

| arm | minimal | full | Δ | writer calls |
|---|---|---|---|---|
| raw | 75.0% | 80.0% | **+5** | none → none |
| reflection | 65.0% | 54.0% | **−11** | 73 → 145 |
| rule | 53.0% | 46.0% | **−7** | 67 → 136 |
| skill | 79.0% | 64.0% | **−15** | 36 → 103 |

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

| arm | minimal | full |
|---|---|---|
| raw | 13/21 (62%) | **14/21 (67%)** |
| reflection | 6/21 (29%) | 6/21 (29%) |
| rule | 5/21 (24%) | 4/21 (19%) |
| skill | 7/21 (33%) | **0/21 (0%)** |

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

| family | none | raw | reflection | rule | skill |
|---|---|---|---|---|---|
| pick_cool_then_place | 5/15 | **14/15** | 7/15 | 7/15 | 12/15 |
| pick_clean_then_place | 9/23 | 10/23 | 18/23 | 10/23 | **19/23** |
| pick_heat_then_place | 9/21 | **13/21** | 6/21 | 5/21 | 7/21 |
| look_at_obj_in_light | 10/13 | **13/13** | 8/13 | 7/13 | **13/13** |
| pick_two_obj_and_place | 7/10 | 8/10 | **10/10** | 6/10 | **10/10** |
| pick_and_place_simple | 18/18 | 17/18 | 16/18 | 18/18 | **18/18** |

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

| arm | policy | rate | Δ vs none | b/c | McNemar p | Δ vs 50-task | store | inj. tok | writer calls |
|---|---|---|---|---|---|---|---|---|---|
| none | — | 58.0% | — | — | — | — | — | 0 | — |
| raw | minimal | 75.0% | +17.0 | 9/26 | **0.006** | 0.0 | 75 | 1278 | — |
| reflection | minimal | 67.0% | +9.0 | 12/21 | 0.163 | +2.0 | 66 | 997 | 81 |
| rule | minimal | 58.0% | +0.0 | 21/21 | 1.000 | +5.0 | 25 | 260 | 83 |
| skill | minimal | 70.0% | +12.0 | 7/19 | **0.029** | −9.0 | 6 | 1439 | 33 |
| raw | full | **81.0%** | **+23.0** | 6/29 | **0.0001** | +1.0 | 61 | 1244 | — |
| reflection | full | 62.0% | +4.0 | 15/19 | 0.608 | +8.0 | 77 | 961 | 158 |
| rule | full | 61.0% | +3.0 | 20/23 | 0.761 | **+15.0** | 15 | 241 | 134 |
| skill | full | 59.0% | +1.0 | 21/22 | 1.000 | −5.0 | 8 | 1325 | 117 |

Store sizes and writer calls are cumulative over all 100 evolving tasks.
Injected tokens are flat because the 1500-token budget binds, so raw's store
doubling costs nothing at inference.

### 6.1 Only `raw` is stable; everything else moves

`raw` is the one content type whose eval rate is unchanged by doubling the
data — 75.0% → 75.0% under minimal (13 tasks flipped each way, pure churn) and
80.0% → 81.0% under full. It is also the best arm in the whole study:
**`raw/full` at 81.0%, +23.0, p = 0.0001**, now perfect on
`pick_cool_then_place_in_recep` (15/15).

The paired 50 → 100 test on each arm's own two runs:

| arm | policy | 50 → 100 | b/c | p |
|---|---|---|---|---|
| raw | minimal | 75 → 75 | 13/13 | 1.000 |
| reflection | minimal | 65 → 67 | 16/18 | 0.864 |
| rule | minimal | 53 → 58 | 13/18 | 0.473 |
| skill | minimal | 79 → 70 | 16/7 | 0.093 |
| raw | full | 80 → 81 | 9/10 | 1.000 |
| reflection | full | 54 → 62 | 13/21 | 0.229 |
| rule | full | 46 → 61 | 10/25 | **0.017** |
| skill | full | 64 → 59 | 14/9 | 0.405 |

Note how large the discordant counts are relative to the deltas: `raw/minimal`
changes 26 of 100 task outcomes to arrive at exactly the same score. Aggregate
stability is not per-task stability for any arm here.

### 6.2 The `full` penalty is a small-data artifact

This is the finding that costs us §1. Minimal-minus-full, at both stream
lengths:

| arm | full − minimal @50 | full − minimal @100 |
|---|---|---|
| raw | +5 | +6 |
| reflection | −11 | −5 |
| rule | −7 | **+3** |
| skill | −15 | −11 |

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

### 6.4 Skill is the only type that gets worse with more data

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

| arm | policy | ep1 | ep2 | ep3 | ep3 vs none | p | ep1 → ep3 | p | store 1→2→3 |
|---|---|---|---|---|---|---|---|---|---|
| raw | minimal | 75.0 | 74.0 | 77.0 | +19.0 | **0.003** | +2.0 | 0.832 | 38 → 44 → 44 |
| reflection | minimal | 65.0 | 68.0 | 56.0 | −2.0 | 0.885 | −9.0 | 0.211 | 39 → 63 → 93 |
| rule | minimal | 53.0 | 54.0 | 56.0 | −2.0 | 0.875 | +3.0 | 0.711 | 16 → 22 → 37 |
| skill | minimal | 79.0 | 81.0 | 73.0 | +15.0 | **0.024** | −6.0 | 0.307 | 4 → 10 → 10 |
| raw | full | 80.0 | 75.0 | 79.0 | +21.0 | **0.001** | −1.0 | 1.000 | 36 → 45 → 45 |
| reflection | full | 54.0 | 64.0 | 59.0 | +1.0 | 1.000 | +5.0 | 0.424 | 43 → 71 → 112 |
| rule | full | 46.0 | 50.0 | 58.0 | +0.0 | 1.000 | +12.0 | 0.096 | 10 → 11 → 23 |
| skill | full | 64.0 | 75.0 | 75.0 | +17.0 | **0.006** | +11.0 | 0.061 | 10 → 16 → 24 |

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

| arm | policy | ep1 | ep2 | ep3 |
|---|---|---|---|---|
| raw | minimal | 3/3 | 37/37 | **40/40** |
| raw | full | 2/2 | 32/32 | **36/36** |
| reflection | minimal | 0/17 | 0/25 | 0/10 |
| rule | full | 0/5 | 0/5 | 0/10 |
| skill | full | 0/7 | 0/10 | 0/11 |

By epoch 3 **every single one of `raw`'s write attempts is rejected as an exact
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

## Threats to validity

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
- **§6 is a resumption, not an independent 100-task run.** Each arm continued
  from its own 50-task store, so its second half is conditioned on whatever that
  store had become. A stream-length effect and a path-dependence effect are not
  separable here; only a fresh run over 100 evolving tasks from an empty store
  would separate them. The two lengths therefore bound the trend, they do not
  establish a curve — and with two points and a ±3–6 noise floor, the 50 → 100
  deltas under ~10 points (every arm except `rule/full`) are not resolvable
  either.

# ALFWorld Memory Content study — Qwen3.5-9B

Run 2026-08-09. 50 evolving tasks (`train`), 100 evaluation tasks
(`valid_unseen`, disjoint), 1 rollout per task, 50-step horizon, seed 42.
Model served locally by vLLM; the same model writes memory and acts.

Raw outputs: `/gpfs/radev/scratch/cohan/jw3278/memsys_results/{minimal,full}/`.
Reproduce with [RUN_ALFWORLD.md](RUN_ALFWORLD.md).

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

## Threats to validity

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

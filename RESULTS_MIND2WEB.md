# Mind2Web Memory Content study — Qwen3.5-9B

Run 2026-08-14 (§2–§3) and 2026-08-15 (§1.2, §5). Evolving annotations from
`train`, 100 evaluation annotations from `test_task` (**838 steps**, disjoint),
1 rollout per step, top-50 candidate pool, temperature 0.

Raw outputs: `/gpfs/radev/scratch/cohan/jw3278/memsys_results/mind2web/`.
Reproduce with [RUN_MIND2WEB.md](RUN_MIND2WEB.md).

The `none` baseline plus all four memory arms under `WritePolicy.minimal()` at
evolving budgets of 50 / 100 / 150 / 200 annotations, and two of the four under
`full()` at 50. **No arm beats the baseline at any budget, and more experience
changes nothing.** Three things carry the document:

- the benchmark has a **small but nonzero** run-to-run noise floor, ~1.4% of
  steps and ±0.5–1.1 points of step SR (§1.2 — this corrects an earlier claim
  that it was exactly zero);
- the losses are concentrated entirely in element selection (§3);
- the deficit is **flat** from 416 to 1559 evolving steps, and three of four arms
  stop accumulating memory entirely by the second leg (§5).

## 0. What one "task" is here, and why

One memsys episode is one Mind2Web **step**, not one annotation. This is the
adapter's single deliberate departure from the benchmark's framing and it
decides whether the experiment exists at all:

Whole-annotation success requires *every* step of a task to be exactly right.
Our baseline scores **3.0%** on that metric (§2). At annotation granularity,
then, ~97% of evolving episodes would be `all_failure` — `raw` and `skill` would
write nothing at all (both require a successful rollout by construction), and the
remaining two arms would be building memory from failures only, which is exactly
the configuration RESULTS_ALFWORLD.md §11 measured as harmful and degenerate.

At step granularity the success signal is the benchmark's own headline metric,
step success rate, which sits at 31.5% — the same band WebShop's episodes occupy.
Manifests are still drawn at *annotation* level and expanded in order, so "evolve
on 50" means 50 annotations = 416 episodes, and a step's memory includes what
earlier steps of its own task wrote.

## 1. Two properties of this benchmark that change how it is read

### 1.1 Twelve percent of the evaluation is unreachable

If the ground-truth element is not in the released ranker's top-50 candidates,
upstream scores the step 0 without calling the model, and this port does the
same. That is **102 of 838 eval steps (12.2%)**. No memory system can move them,
so they dilute every delta by ~12%. On the 736 scoreable steps the baseline is
35.9% rather than 31.5%.

### 1.2 The noise floor is small, but it is not zero

**Corrected 2026-08-15.** This section previously claimed that two identical runs
agree on every step, on the strength of a 24-step check. They do not. Replaying
the no-memory baseline over the full 838 eval steps, twice, on one GPU, one
server, one seed, at temperature 0:

| pair | disagreeing steps | step SR |
| --- | --- | --- |
| same-node replica 1 vs replica 2 | 12 / 838 (1.4%) | 31.9% vs 32.6% |
| replica 1 vs the 2026-08-14 run | 15 / 838 (1.8%) | — |
| replica 2 vs the 2026-08-14 run | 11 / 838 (1.3%) | 31.5% |

Two same-node replicas disagree at the same rate as a cross-date, cross-GPU pair,
so this is **not** a hardware effect: it is vLLM's continuous batching. Reduction
order depends on what else is in the batch, and temperature 0 does not pin that
down. A third, independent estimate falls in the same place — `raw`'s live entry
set stops changing after 100 annotations (§5), so its 100 / 150 / 200 cells are
the same configuration evaluated three times, and they disagree at 1.0–1.3%.

The original check was not wrong, only far too small: at a 1.4% per-step rate, 24
steps come up clean about 71% of the time.

What this costs, in order of importance:

- **The headline metric carries a ±0.5–1.1 point band.** Three no-memory runs
  scored 31.5%, 31.9% and 32.6%. Any delta inside that span is not evidence.
- **Single p-values in §2 and §5 are not trustworthy on their own.** Every arm is
  measured against *one* draw from that band; `rule` moves between p = 0.035 and
  p = 0.294 across evolving budgets while its store sits pinned at capacity.
  Deltas are therefore reported against both replicas from here on, with the
  least favourable p quoted.
- **The b/c columns survive.** Arms flip 100–130 of 838 steps against the
  baseline, five to nine times the churn rate. The memory block genuinely changes
  a large number of decisions; what the floor undermines is the interpretation of
  the small *net* result, not the claim that something is happening.

The comparison to ALFWorld still holds in direction — RESULTS_ALFWORLD.md §11.1
has two identical configurations disagreeing on 35 of 100 tasks — but the honest
statement is "a much smaller noise floor", not "no noise floor".

## 2. Results

`none` is the shared baseline (policy-independent). Rate = step success rate;
score = element accuracy; Op.F1 = operation F1; Task SR = every step of an
annotation correct.

| arm | policy | Step SR | Δ | b/c | McNemar p | Ele.Acc | Op.F1 | Task SR | store | mem tok |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **none** | — | **31.5%** | — | — | — | **35.0%** | **73.1** | **3.0%** | — | 0 |
| raw | minimal | 31.1% | −0.4 | 63/60 | 0.857 | 34.0% | 75.1 | 1.0% | 132 | 1417 |
| reflection | minimal | 29.6% | −1.9 | 73/57 | 0.188 | 33.4% | 72.2 | 2.0% | 200 | 939 |
| rule | minimal | 29.6% | −1.9 | 58/42 | 0.133 | 33.3% | 71.4 | 0.0% | 154 | 418 |
| skill | minimal | 29.8% | −1.7 | 62/48 | 0.215 | 32.7% | 74.8 | 1.0% | 10 | 1432 |
| raw | full | 30.0% | −1.5 | 60/47 | 0.246 | 32.9% | 73.9 | 2.0% | 81 | 1428 |
| reflection | full | 30.1% | −1.4 | 64/52 | 0.307 | 33.2% | 72.9 | 2.0% | 92 | 940 |

Six cells, six negative deltas, spread −0.4 to −1.9, none significant.

**Read that spread against §1.2's ±0.5–1.1 point band**, which was measured after
this table was written. Three of the six cells sit inside it. The sign
consistency is still worth noting — six draws landing on the same side of a
baseline is not what churn alone looks like — but the original argument for it
("with no noise floor, ... not six draws from a churn distribution") assumed a
property this benchmark does not have. §5 re-measures every arm against two
baseline replicas on one serving stack and reaches the same qualitative place
with honest error bars.

`rule/minimal`'s **0.0% task success rate** is the sharpest single number —
across 100 annotations it never got every step of any task right, against the
baseline's 3.

### 2.1 Memory did not help during evolving either

Evolve-time step SR over the same 416 steps in the same order:

| arm | policy | evolve Step SR | first 208 | last 208 |
| --- | --- | --- | --- | --- |
| raw | minimal | 0.353 | 0.370 | 0.337 |
| rule | minimal | 0.300 | 0.279 | 0.322 |
| skill | minimal | 0.303 | 0.288 | 0.317 |
| reflection | minimal | 0.274 | 0.279 | 0.269 |
| raw | full | 0.329 | 0.337 | 0.322 |
| reflection | full | 0.281 | 0.269 | 0.293 |

No arm shows a within-run trend that would suggest the memory is compounding:
the largest half-to-half move is +4.3 points (`rule/minimal`) and `raw` moves the
other way. These are *not* comparable to the baseline — the `none` arm has no
evolving phase, so there is no no-memory reference on the train split. Read the
column as an arm-vs-arm ordering only.

## 3. The loss is element selection, not operations

For each `minimal` arm, of the steps the baseline got right and the arm lost:

| arm | steps lost | lost by picking a different element | lost on the operation/value |
| --- | --- | --- | --- |
| raw | 63 | 60 | 3 |
| reflection | 73 | 65 | 8 |
| rule | 58 | 47 | 11 |
| skill | 62 | 58 | 4 |

**Around 90% of the damage is the model choosing a different element.** Operation
F1 barely moves anywhere in §2 (71.4–75.1 against a 73.1 baseline, and `raw` and
`skill` are *above* it). Whatever the memory is doing, it is not teaching the
agent to click when it should type; it is changing which of five candidate
elements looks best, and slightly for the worse.

That points at a mechanism the design predicts: the memory block is 418–1432
tokens injected ahead of a page serialisation that is typically 1–3k tokens, in a
task whose answer is one of five short strings. `rule`, with the *smallest*
block (418 tokens), loses the fewest steps by element and the most by operation —
the only arm whose profile differs, and it is the one whose entries are short
imperative directives rather than trajectory text.

## 4. What is not here

**`rule/full` and `skill/full` are missing.** Their node's allocation ended at
14:30 while they were 177/416 and 184/416 steps into evolving. Nothing is wrong
with the runs; they were killed mid-stream and the partial `evolve_episodes.jsonl`
files are on scratch. They need ~3 h each on a fresh allocation, and until they
land the `full` policy is represented by two arms out of four.

**~~No larger evolving budget.~~** Answered in §5: the chain now runs to 200
annotations (1559 evolving steps) under `minimal`. The small negative is stable —
it neither compounds nor washes out. `full` is still a 50-annotation result only.

**No `full` continuation.** §5 is `minimal` alone. Running the same four legs
under `full` would cost roughly another 4 h on two GPUs, and would answer a
different question: `full`'s deletion machinery is what decides *which* 200
entries survive at capacity, and §5.1 says capacity is reached by the 100 mark
for three of four arms.

**No larger `max_items`.** The more interesting variable than evolving budget, on
this evidence. Three arms are pinned at 200 live entries from 100 annotations
onward, so §5's 150 and 200 legs measure eviction, not accumulation. A run at
`--max-items 600` would separate the two for the first time.

## 5. Evolving budget: 50 → 200 annotations

Run 2026-08-15 on one node (2× H200), `minimal` policy. Each arm resumes its own
`store.jsonl` and evolves the next 50 annotations of the same seeded permutation,
so the four points are one chain, not four independent runs: 416 → 791 → 1181 →
1559 cumulative evolving steps. Evaluation is the same frozen 838 steps every
time.

Two things were done differently from §2, both because of §1.2:

- **The baseline is two replicas, on this stack.** Deltas are given as a range
  across both, and the p quoted is the least favourable of the two. An arm has to
  beat both draws to look significant.
- **The 50 point was re-scored here.** Same `store.jsonl` the 2026-08-14 sweep
  produced, no further evolving, evaluated on the same servers as the legs after
  it. Without that the first point of the curve would be measured against a
  baseline collected on other hardware on another day. The re-scoring reproduced
  the original within 0.6 points for all four arms — the memory-content ordering
  in §2 replicates.

| arm | budget | evolve steps | Step SR | Δ vs none | b/c | worst p | store | mem tok |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **none** (replica 1) | — | — | **31.9%** | — | — | — | — | 0 |
| **none** (replica 2) | — | — | **32.6%** | +0.7 | 3/9 | 0.146 | — | 0 |
| raw | 50 | 416 | 31.7% | −0.8..−0.1 | 62/61 | 1.000 | 132 | 1417 |
| raw | 100 | 791 | 31.3% | −1.3..−0.6 | 58/53 | 0.704 | 200 | 1413 |
| raw | 150 | 1181 | 31.4% | −1.2..−0.5 | 58/54 | 0.777 | 200 | 1413 |
| raw | 200 | 1559 | 31.6% | −1.0..−0.2 | 58/56 | 0.925 | 200 | 1413 |
| reflection | 50 | 416 | 30.1% | −2.5..−1.8 | 72/57 | 0.218 | 200 | 939 |
| reflection | 100 | 791 | 30.0% | −2.6..−1.9 | 68/52 | 0.171 | 200 | 945 |
| reflection | 150 | 1181 | 28.9% | −3.7..−3.0 | 75/50 | 0.031 | 200 | 944 |
| reflection | 200 | 1559 | 29.8% | −2.7..−2.0 | 68/51 | 0.142 | 200 | 953 |
| rule | 50 | 416 | 29.1% | −3.5..−2.7 | 66/43 | 0.035 | 200* | 418 |
| rule | 100 | 791 | 29.4% | −3.2..−2.5 | 63/42 | 0.050 | 200 | 418 |
| rule | 150 | 1181 | 30.4% | −2.1..−1.4 | 61/49 | 0.294 | 200 | 424 |
| rule | 200 | 1559 | 29.7% | −2.9..−2.1 | 64/46 | 0.105 | 200 | 426 |
| skill | 50 | 416 | 30.4% | −2.1..−1.4 | 60/48 | 0.290 | 10 | 1432 |
| skill | 100 | 791 | 30.3% | −2.3..−1.6 | 62/49 | 0.255 | 51 | 1408 |
| skill | 150 | 1181 | 31.3% | −1.3..−0.6 | 59/54 | 0.707 | 88 | 1401 |
| skill | 200 | 1559 | 30.4% | −2.1..−1.4 | 63/51 | 0.303 | 132 | 1399 |

\* 154 live at the end of the 50-annotation run; the table shows the store the
evaluation actually saw at each budget.

### 5.1 More experience does not help, and it does not hurt either

**No arm has a trend.** Across a 3.7× increase in evolving experience, the
largest excursion any arm makes is 1.5 points (`rule`, 29.1 → 30.4 → 29.7), and
it is not monotone in either direction. Every arm at 200 is within 0.7 points of
where it was at 50 — inside the baseline band. The §4 question, "whether the
small negative is stable or shrinks with more experience", answers **stable**:
the deficit neither compounds nor washes out.

**Three of the four arms stop learning long before 200.** `raw`, `reflection` and
`rule` all hit `max_items = 200` live entries by the 100 mark. Past that point
the experiment is no longer "more experience" but "the eviction policy under a
fixed budget", and only `skill` — which writes so rarely it reaches 132 entries
after 1559 steps — is still accumulating at 200.

`raw` makes the strongest version of the point. Its live entry set is *identical*
at 100, 150 and 200 (the same 200 ids; only utility counters differ), so the last
two legs never changed what the agent saw. Its 31.3 / 31.4 / 31.6 is three
evaluations of one configuration, and the 0.3-point spread is the §1.2 floor
measured a third way. **For `raw`, evolving past 100 annotations is a no-op** —
`RawTrajectorySystem` keeps `best_success()` trajectories and, once full, admits
nothing new.

### 5.2 What the significance column actually does

`rule` reads p = 0.035, 0.050, 0.294, 0.105 across the four budgets while its
store is pinned at capacity and its injected block never moves outside 418–426
tokens. `reflection` reads 0.218, 0.171, 0.031, 0.142 with an equally static
store. Neither pattern is a memory system doing something different at 150 than
at 100; both are ±1-point wobble against a baseline that is itself ±0.7,
resampled four times. The 16 cells here are the argument for not quoting any
single one of them — including `reflection @ 150`, the one that clears 0.05.

### 5.3 Evolve-time success rates

Step SR during evolving, per leg (not comparable to the eval column — different
split, and `none` has no evolving phase):

| arm | 50 | 100 | 150 | 200 |
| --- | --- | --- | --- | --- |
| raw | 0.353 | 0.331 | 0.274 | 0.344 |
| reflection | 0.274 | 0.275 | 0.264 | 0.302 |
| rule | 0.300 | 0.312 | 0.259 | 0.294 |
| skill | 0.303 | 0.285 | 0.269 | 0.328 |

All four arms dip on the third leg and recover on the fourth, together. That is a
property of which annotations sit at positions [100,150) of the permutation, not
of the memory — the legs are not equally hard, which is a further reason not to
read the eval column as a learning curve.

## Threats to validity

- **One seed, one order, one split.** `test_task` is the easiest of the three
  test splits (its websites appear in `train`, though nothing here is trained).
  `test_website` and `test_domain` would test whether the memory transfers
  across sites at all, which is the more interesting question for a *web* memory
  and is not answered here.
- **The absolute numbers are a port, and the port is unvalidated against a
  published table.** The scaffold is upstream's `llm_prompt.json` and the scoring
  is a line-by-line port of `metric.py`, so these numbers should be comparable to
  published Mind2Web LLM results in a way this study's WebShop and
  SpreadsheetBench numbers are not — but that comparison has not actually been
  run. Until it is, treat the `none` arm as the only reference, as elsewhere in
  this study.
- **Teacher forcing removes the thing memory might help most with.** Previous
  actions in the prompt are always the annotator's, so an agent cannot be
  rescued from its own earlier mistake — which is precisely where a "lesson from
  a past failure" would pay off on ALFWorld. Mind2Web-Live or any online variant
  would not have this property.
- **Six negative deltas are not six independent tests.** All six arms evaluate
  the same 838 steps against the same baseline and share the 12.2% unreachable
  set; their errors are correlated. The sign consistency is suggestive, not a
  meta-analysis. §5's 16 cells are worse in this respect, not better: four of
  them (`raw` at 100/150/200) are literally the same configuration re-evaluated.
- **The noise floor is measured from three replicas, not thirty.** §1.2's
  ±0.5–1.1 point band comes from two same-node runs plus one from 2026-08-14. It
  is enough to falsify "the floor is zero" and enough to say the §2 deltas are
  not clearly outside it; it is not a tight interval, and a proper one would want
  ~10 replicas. Every p-value here is quoted against a single baseline draw, which
  is why §5 reports the worse of two rather than the better.
- **`skill` wrote almost nothing** (10 live entries from 126 writer calls at 50
  annotations, still only 132 after 1559 evolving steps), so its row is closer to
  a lightly-perturbed baseline than to a test of procedural memory — the same
  pattern RESULTS_ALFWORLD.md §11.1 documents for a different reason.
- **§5's later legs measure eviction, not experience.** `raw`, `reflection` and
  `rule` are at the 200-entry cap by the 100 mark, and `raw`'s live set does not
  change at all after it. "Evolving on 200 annotations" is therefore not 4× the
  memory of "evolving on 50" for any arm but `skill`; it is the same-sized memory
  selected from a larger pool.
- **Step-level episodes make retrieval keys nearly duplicate.** Every step of an
  annotation shares the task text; the adapter appends "(step i of n)" to
  separate them, but a store built this way holds many near-identical keys, which
  is a plausible contributor to the retrieval quality problem in §3 and is not
  something the ALFWorld or WebShop runs had to contend with.

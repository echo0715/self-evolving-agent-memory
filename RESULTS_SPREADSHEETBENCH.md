# SpreadsheetBench — Memory Content results

Qwen3.5-9B, 50 / 100 / 150 evolving episodes, 100 evaluation tasks, seed 42.
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

| | eval success | vs previous, discordant (b/c) |
|---|---|---|
| `skill / full`, 50 evolve | 27/100 | — |
| `skill / full`, 100 evolve | 27/100 | 11/11 |
| `skill / full`, 150 evolve | 27/100 | 11/11 |

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
|---|---|
| all 3 | 11 |
| exactly 2 | 15 |
| exactly 1 | 18 |
| none | 56 |

**Only 11 tasks are reliably solved. 33 are coin flips.** A configuration
scoring 27/100 is not solving a stable set of 27 tasks — it is solving 11, plus
roughly half of a pool of 33 it can sometimes reach. Every per-task claim in
this document, and any attempt to characterise *which* tasks memory helps with,
has to survive that.

### Every result in units of that floor

| arm | policy | evolve | success | delta | σ |
|---|---|---|---|---|---|
| raw | full | 150 | 27/100 | +14 | **+3.0** |
| skill | full | 150 | 27/100 | +14 | **+3.0** |
| skill | full | 100 | 27/100 | +14 | **+3.0** |
| skill | full | 50 | 27/100 | +14 | **+3.0** |
| reflection | minimal | 50 | 27/100 | +14 | **+3.0** |
| skill | minimal | 150 | 26/100 | +13 | **+2.8** |
| rule | full | 50 | 26/100 | +13 | **+2.8** |
| reflection | full | 150 | 25/100 | +12 | **+2.6** |
| reflection | minimal | 150 | 24/100 | +11 | **+2.3** |
| reflection | full | 50 | 23/100 | +10 | **+2.1** |
| rule | full | 100 | 23/100 | +10 | **+2.1** |
| rule | full | 150 | 22/100 | +9 | +1.9 |
| raw | full | 100 | 21/100 | +8 | +1.7 |
| skill | minimal | 50 | 20/100 | +7 | +1.5 |
| skill | minimal | 100 | 20/100 | +7 | +1.5 |
| raw | minimal | 50 | 19/100 | +6 | +1.3 |
| raw | minimal | 100 | 19/100 | +6 | +1.3 |
| reflection | full | 100 | 19/100 | +6 | +1.3 |
| raw | full | 50 | 18/100 | +5 | +1.1 |
| raw | minimal | 150 | 18/100 | +5 | +1.1 |
| reflection | minimal | 100 | 18/100 | +5 | +1.1 |
| rule | minimal | 100 | 18/100 | +5 | +1.1 |
| rule | minimal | 150 | 18/100 | +5 | +1.1 |
| rule | minimal | 50 | 17/100 | +4 | +0.9 |

**Eleven of twenty-four configurations clear 2σ** (delta ≥ +10). Everything
below is indistinguishable from a lucky draw against this baseline. And the
baseline is still a *single* run (13/100), so these σ figures assume the
baseline itself was not an unlucky draw — a second one would still be worth 30
minutes, and is now the single highest-value half hour left in this study.

What survives, stated conservatively: **memory helps on this benchmark, by
roughly 12–14 points at the top end. The ordering among memory types still does
not survive.** At 150 episodes the top four are raw, skill, skill and reflection
— spanning three of the four content types, separated by 2 points, in a study
whose noise floor is 4.7.

---

## What was run

| | |
|---|---|
| model | `Qwen/Qwen3.5-9B` (non-thinking), vLLM |
| data | `all_data_912_v0.1`, 3 test cases per task |
| evolving (50) | SkillOpt id split `train`, positions [0, 50) of the seed-42 permutation |
| evolving (100) | continued: positions [50, 100), spilling into `val` (29 train + 21 val) |
| evolving (150) | continued: positions [100, 150), spilling into `val` *and* `test` |
| evaluation | the same 100 tasks from `test` for every run |
| agent | 30 turns max, fenced-block `bash` / `write_file` protocol |
| injection budget | 2500 tokens |

A task counts as solved only if **all three test cases pass**. The agent sees
case 1; its `solution.py` is re-executed against cases 2 and 3.

Each leg *resumes* the previous leg's `store.jsonl` per arm and carries the
Evolver's step counter, so the legs extend rather than repeat, and `full`'s
every-25-episode batch induction fires where an uninterrupted 150-episode run
would have fired it.

Two caveats specific to the third leg:

- **It draws from `test`.** `train`+`val` holds only 117 selectable tasks once
  the evaluation set's source workbooks are excluded, and 100 were already
  spent, so positions [100, 150) take 33 of their 50 from the same split the
  evaluation set is drawn from. The builder enforces that the evolving tasks
  share no task id **and no source workbook** with the frozen eval set, and
  refuses to write the manifest otherwise; this was re-verified before the run
  (zero id overlap with eval, and zero with either earlier leg). The evaluation
  set is untouched — but this leg's disjointness rests on the group filter,
  where the first two legs got it from the split boundary.
- **`reflection / full` and `rule / full` ran on a 65536-token server**, the
  other six arms on 32768. See "Batch induction outgrows the context window"
  below. This cannot affect comparability: no call in the 50- or 100-episode
  legs ever reached 32768 (checked across every log), so those legs would return
  identical results on either server.

## Results

### `WritePolicy.minimal()` — append + merge only

| arm | 50 evolve | | 100 evolve | | 150 evolve | | store 50→100→150 |
|---|---|---|---|---|---|---|---|
| | success | score | success | score | success | score | |
| none | 13/100 | 18.0 | *(baseline reused)* | | | | — |
| raw | 19/100 | 23.0 | 19/100 | 24.3 | 18/100 | 24.7 | 14 → 21 → 33 |
| reflection | **27/100** | **34.3** | 18/100 | 24.0 | 24/100 | 32.0 | 47 → 105 → **156** |
| rule | 17/100 | 21.7 | 18/100 | 23.0 | 18/100 | 24.7 | 32 → 61 → 85 |
| skill | 20/100 | 24.3 | 20/100 | 28.0 | **26/100** | 31.3 | 5 → 7 → 8 |

### `WritePolicy.full()` — every mechanism on

| arm | 50 evolve | | 100 evolve | | 150 evolve | | store 50→100→150 |
|---|---|---|---|---|---|---|---|
| | success | score | success | score | success | score | |
| none | 13/100 | 18.0 | *(baseline reused)* | | | | — |
| raw | 18/100 | 24.7 | 21/100 | 26.7 | **27/100** | **33.7** | 8 → 13 → 24 |
| reflection | 23/100 | 29.3 | 19/100 | 24.0 | 25/100 | 30.0 | 30 → 43 → 85 |
| rule | 26/100 | 29.7 | 23/100 | 29.0 | 22/100 | 25.7 | 17 → 22 → 29 |
| skill | **27/100** | **32.7** | **27/100** | **32.7** | **27/100** | 31.7 | 3 → 3 → **3** |

`score` = mean fraction of test cases passed, ×100.

---

## More episodes: still no trend that clears the floor

| arm / policy | 50 | 100 | 150 | 50→150 | σ |
|---|---|---|---|---|---|
| raw / minimal | 19 | 19 | 18 | −1 | −0.2 |
| raw / full | 18 | 21 | **27** | **+9** | **+1.9** |
| reflection / minimal | 27 | 18 | 24 | −3 | −0.6 |
| reflection / full | 23 | 19 | 25 | +2 | +0.4 |
| rule / minimal | 17 | 18 | 18 | +1 | +0.2 |
| rule / full | 26 | 23 | 22 | −4 | −0.9 |
| skill / minimal | 20 | 20 | **26** | **+6** | **+1.3** |
| skill / full | 27 | 27 | 27 | 0 | 0.0 |

The 100-episode leg's headline — that fifty more episodes bought nothing — does
**not** simply extend. Two arms moved at 150:

- **`raw / full` climbed 18 → 21 → 27**, the only arm monotone across all three
  legs, +9 over its own 50-episode run. At 1.9σ this is suggestive, not
  established; but it is the single cleanest trend in the study, and it belongs
  to the *least* structured content type. Its store grew 8 → 13 → 24.
- **`skill / minimal` jumped 20 → 20 → 26** (+6, 1.3σ) on a store that grew by
  one entry, 7 → 8.

Set against that, `reflection / minimal` went 27 → 18 → 24 and `rule / full`
went 26 → 23 → 22. Neither excursion reaches 2σ in either direction. The honest
reading of all eight trajectories: **three legs of evidence still do not
establish that more evolving episodes help, but they no longer support the flat
claim that more episodes do nothing.** `raw / full` is the case to settle, and
a repeat of that arm alone would settle it.

Two mechanisms remain visible, pulling in opposite directions.

### Reflection accumulates faster than the budget can carry

| | store 50 | 100 | 150 | injected tokens | success |
|---|---|---|---|---|---|
| reflection / minimal | 47 | 105 | **156** | 1024 → 1020 → 1032 | 27 → 18 → 24 |
| reflection / full | 30 | 43 | 85 | 997 → 1052 → 1066 | 23 → 19 → 25 |

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

| | write calls | landed | store |
|---|---|---|---|
| skill / full, episodes 0–50 | 12 | APPEND 12, DELETE 9 | 3 |
| skill / full, episodes 50–100 | 15 | APPEND 1, DELETE 1 | 3 |
| skill / full, episodes 100–150 | 12 | APPEND 2, DELETE 2 | 3 |

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

| arm | died at | induce prompt |
|---|---|---|
| `reflection / full` | evolve 24/50 (step 124) | ≥ 31745 tok, + 1024 output > 32768 |
| `rule / full` | evolve 49/50 (step 149) | ≥ 31745 tok, + 1024 output > 32768 |

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

## Failure modes

Counts over the 100 evaluation tasks, `none` arm:

| reason | n | meaning |
|---|---|---|
| `eval-mismatch` | 77 | ran, produced a workbook, wrong values |
| `output-not-found` | 7 | never wrote the output file |
| `exec-error` | 2 | `solution.py` crashed on case 2 or 3 |
| `no-solution-py-for-other-cases` | 1 | edited case 1 by hand |

The benchmark is essentially all `eval-mismatch`: 97% of episodes produced a
`solution.py`, and the plumbing failures memory might plausibly fix account for
10 tasks in total. Memory arms cut those to 2–8, but the aggregate is dominated
by getting the *values* right, which is where the ceiling on any memory effect
comes from.

## Cost

Writer tokens **per 50-episode leg** (raw has no writer):

| arm / policy | leg | prompt tok | completion tok |
|---|---|---|---|
| skill / full | 0–50 | 176k | 18k |
| | 100–150 | 252k | 24k |
| rule / full | 0–50 | 353k | 44k |
| | 100–150 | 394k | 48k |
| reflection / full | 0–50 | 405k | 54k |
| | 100–150 | 494k | 67k |
| skill / minimal | 100–150 | 75k | 10k |
| rule / minimal | 100–150 | 170k | 21k |
| reflection / minimal | 100–150 | 197k | 23k |

Wall time is not comparable across arms — four arms shared two vLLM servers.
The writer token columns are, and they are the real compute difference: `full`
costs reflection and rule 2.3–2.5× their `minimal` writer tokens for the
verify/refine/induce loop. **Per-leg cost also rises as the store grows** —
reflection/full spends 22% more in its third leg than its first — which is the
same unbounded-prompt growth that eventually killed it.

## What to run next, in order

1. **A second `none` baseline.** ~30 minutes, and now the highest-value run
   left. Every delta in this document is measured against a single draw of
   13/100. If that draw was 1σ low, every delta shrinks by 5 points and the
   count clearing 2σ drops from eleven to about three.
2. **Repeat `raw / full`.** It is the only arm with a monotone trend across
   three legs (18 → 21 → 27) and it reached the study's joint-best result from
   the least structured content type. At 1.9σ it is exactly the kind of finding
   that a single repeat either establishes or kills.
3. **Bound the `induce` prompt**, then re-run `full` at 200 episodes. As it
   stands `full` cannot be run further for reflection or rule at any store size
   without hitting the same wall; 65536 only moves it.
4. **Raise the baseline before comparing content types.** At 13/100 with σ ≈ 4.7
   the whole study lives in a 14-point band ~3σ wide, and only 11 of 100 tasks
   are solved reliably at all. More agent turns or a stronger model would
   separate the content types better than more evolving episodes.

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
```

Raw outputs:
`$MEMSYS_RESULTS_ROOT/spreadsheetbench/{minimal,full,minimal_e100,full_e100,minimal_e150,full_e150}/<arm>_<policy>/`
— `summary.json`, `eval.jsonl`, `evolve_episodes.jsonl`, `evolve_log.jsonl`,
`store.jsonl`, and per-task `solution.py` + predicted workbooks. The two crashed
attempts are kept alongside as `*_full.ooc-crash/`.

Manifest fingerprints (`task_ids_sha256`, first 16):
evolve [0,50) `654ef6ea3a1d748e`, evolve [50,100) `ba2969e56a40109e`,
evolve [100,150) `51f5a386a966ea84`, eval `5637cf201e1948d9`.

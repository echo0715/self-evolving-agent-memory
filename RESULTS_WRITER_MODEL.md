# The Memory Writing Model — what four writers evolve from the same experience

Run 2026-08-16 → 2026-08-19. The actor is held fixed at **Qwen3.5-9B** on local
vLLM for every cell in this document; the only thing that changes is the model
that *writes* memory. Same seeded manifests, same 50/100 evolving tasks, same
frozen 100-task evaluation set, `WritePolicy.minimal()` throughout. Any delta is
therefore attributable to what was written, not to who acted.

Four writers, three benchmarks:

| writer | served by | API | writer budget |
| --- | --- | --- | --- |
| `Qwen/Qwen3.5-9B` | local vLLM (the actor's own server) | chat, guided JSON | 1024 |
| `openai/gpt-5.6-terra` | Perplexity gateway | responses | 1024 |
| `google/gemini-3.7-flash` | Perplexity gateway | responses | 4096 |
| `perplexity/kimi-k3` | Perplexity gateway | responses | 4096 |

Raw outputs: `memsys_results[/<bench>]/minimal{,_e100}{,_gpt56terra,_gemini37f,_kimik3}/`.

**Two provenance facts to read before quoting anything.**

1. **The `none` and `raw` rows are copies, not re-measurements.** Neither calls a
   writer LLM — `none` has no memory system and `raw` stores trajectories
   verbatim — so the writer axis cannot move them. They are copied from the Qwen
   chain's matching leg so that McNemar has a baseline at all. Every such
   directory carries a `REUSED_FROM_QWEN_RUN.txt`.
2. **The writer budget is not constant across writers, and the reason matters.**
   gemini-3.7-flash and kimi-k3 run at 4096 output tokens rather than 1024. This
   was measured before launch on the real writer prompt, not guessed: at 1024,
   gemini's `skill` proposal is cut off mid-JSON at 1020 completion tokens and
   `parse_ops` returns zero ops — a silently empty store that reads exactly like
   the `none` arm. kimi spends 1048–2363 tokens on the same prompts. The
   confound is real but small in the other direction: gpt-5.6 and Qwen average
   268–435 completion tokens per call, so the 1024 cap was not routinely binding
   for them, and a *larger* budget cannot explain gemini's *smaller* stores.

**Pending cells.** ALFWorld × kimi-k3 at 100, and both SpreadsheetBench × kimi-k3
legs, were still running when this was written. The SpreadsheetBench × kimi-k3
e50 row is shown partial and must not be quoted: its `reflection` arm died at
evolve 24/50 to three consecutive 300-second gateway timeouts and is being
re-run at a 900-second timeout, and without `none` the cell has no delta column.

## Results

Every arm evaluates the identical ordered task list, so comparisons against the
baseline are paired and McNemar's exact test applies. Bold = p < 0.05.
Benchmark noise floors, measured from repeated no-memory draws: ALFWorld ±5,
AppWorld ±5, SpreadsheetBench ±4.7 points.

#### ALFWorld (eval `valid_unseen` 100, baseline 58)

| writer | evolve | reflection | rule | skill | raw |
| --- | --- | --- | --- | --- | --- |
| Qwen3.5-9B | 50 | 65 (+7) | 53 (−5) | **79 (+21, p=0.0002)** | **75 (+17)** |
| Qwen3.5-9B | 100 | 67 (+9) | 58 (+0) | **70 (+12, p=0.029)** | **75 (+17)** |
| gpt-5.6-terra | 50 | **81 (+23, p<0.001)** | **77 (+19, p=0.004)** | **74 (+16, p=0.026)** | **75 (+17)** |
| gpt-5.6-terra | 100 | 71 (+13) | **76 (+18, p=0.006)** | **83 (+25, p<0.001)** | **75 (+17)** |
| gemini-3.7-flash | 50 | **85 (+27, p<0.001)** | **77 (+19, p=0.001)** | **79 (+21, p<0.001)** | **75 (+17)** |
| gemini-3.7-flash | 100 | **91 (+33, p<0.001)** | **83 (+25, p<0.001)** | **80 (+22, p<0.001)** | **75 (+17)** |
| kimi-k3 | 50 | 64 (+6) | 67 (+9) | 57 (−1) | **75 (+17)** |

#### AppWorld (eval `test_normal` 100, baseline 20)

| writer | evolve | reflection | rule | skill | raw |
| --- | --- | --- | --- | --- | --- |
| Qwen3.5-9B | 50 | 22 (+2) | 25 (+5) | 25 (+5) | 26 (+6) |
| Qwen3.5-9B | 100 | 17 (−3) | **34 (+14, p=0.013)** | 23 (+3) | 23 (+3) |
| gpt-5.6-terra | 50 | 29 (+9) | **35 (+15, p=0.003)** | 29 (+9) | 26 (+6) |
| gpt-5.6-terra | 100 | 28 (+8) | 22 (+2) | 25 (+5) | 23 (+3) |
| gemini-3.7-flash | 50 | 26 (+6) | 29 (+9) | 23 (+3) | 26 (+6) |
| gemini-3.7-flash | 100 | 24 (+4) | 24 (+4) | 30 (+10) | 23 (+3) |
| kimi-k3 | 50 | 23 (+3) | 27 (+7) | 20 (+0) | 26 (+6) |
| kimi-k3 | 100 | 22 (+2) | 22 (+2) | 26 (+6) | 23 (+3) |

#### SpreadsheetBench (eval `test` 100, baseline 13)

| writer | evolve | reflection | rule | skill | raw |
| --- | --- | --- | --- | --- | --- |
| Qwen3.5-9B | 50 | **27 (+14, p=0.009)** | 17 (+4) | 20 (+7) | 19 (+6) |
| Qwen3.5-9B | 100 | 18 (+5) | 18 (+5) | 20 (+7) | 19 (+6) |
| gpt-5.6-terra | 50 | 17 (+4) | **24 (+11, p=0.019)** | **23 (+10, p=0.041)** | 19 (+6) |
| gpt-5.6-terra | 100 | 22 (+9) | 20 (+7) | **27 (+14, p=0.004)** | 19 (+6) |
| gemini-3.7-flash | 50 | 20 (+7) | **26 (+13, p=0.015)** | 20 (+7) | 19 (+6) |
| gemini-3.7-flash | 100 | 18 (+5) | 18 (+5) | **24 (+11, p=0.019)** | 19 (+6) |
| kimi-k3 | 50 | *in flight* | 11 | 23 | — |

## 1. The writer is worth 21 points on one benchmark and nothing on another

Hold the arm and the budget fixed and read down the writer column. ALFWorld
`reflection` at 50 episodes spans **64 (kimi) to 85 (gemini)** — a 21-point
spread from changing nothing but the model that writes the memory, against a
5-point noise floor. `rule` spans 53 to 77.

AppWorld does not reproduce this at all. Across four writers, two budgets and
three arms — 24 cells — exactly **two** clear the floor with significance, and
they are the same arm under different writers at different budgets: gpt-5.6's
`rule` at 50 (+15, p = 0.003) and Qwen's own `rule` at 100 (+14, p = 0.013).
Neither replicates in the other's chain — gpt-5.6's `rule` falls to +2 at 100
with a real paired drop (b/c 19/6, p = 0.015), and Qwen's is +5 at 50. The
remaining 22 cells sit inside ±10 with p > 0.05. **A better memory writer is not a general-purpose
lever; it is a lever on benchmarks where the missing knowledge is expressible as
memory.** ALFWorld's is (the simulator has a small set of hard mechanical rules,
§3); AppWorld's apparently is not.

SpreadsheetBench sits in between and is unstable in a specific way: whichever
arm leads at 50 is not the arm that leads at 100, for all three remote writers.

## 2. Store size varies six-fold and does not predict the delta

| ALFWorld `reflection`, 50 episodes | live entries | tok/entry | score |
| --- | --- | --- | --- |
| gemini-3.7-flash | **7** | 93 | **85** |
| gpt-5.6-terra | 21 | 91 | 81 |
| kimi-k3 | 37 | 114 | 64 |
| Qwen3.5-9B | 39 | 114 | 65 |

Across all 135 arm-cells in the study that have a `none` baseline and a store
(minimal policy, three benchmarks, all four writers), Spearman correlation
between live store size and delta is **−0.14** — ALFWorld −0.36, AppWorld −0.04,
SpreadsheetBench −0.14. Weakly negative, nowhere near a rule.

The cleanest single comparison is AppWorld `rule` at 50 episodes: gemini writes
**5** entries and scores 29; kimi writes **51** and scores 27. A ten-fold
difference in what was stored, the same number out.

Per writer, averaged over every cell:

| writer | mean live store | mean delta |
| --- | --- | --- |
| gemini-3.7-flash | **12.0** | **+12.8** |
| gpt-5.6-terra | 36.0 | +12.9 |
| Qwen3.5-9B | 44.8 | +7.2 |
| kimi-k3 | 32.8 | +3.8 |

gemini reaches gpt-5.6's effect with a third of the entries. Whatever the
writer contributes, it is not throughput.

## 3. Only one writer abstracts into slots; the others write episodes

The sharpest qualitative difference, and it is measurable. Counting entries
whose text contains a `<placeholder>` slot:

| ALFWorld, 50 episodes | reflection | rule |
| --- | --- | --- |
| gemini-3.7-flash | 71% | **100%** |
| kimi-k3 | 24% | 35% |
| gpt-5.6-terra | 0% | 0% |
| Qwen3.5-9B | 0% | 0% |

And, in the other direction, entries carrying a one-off instance name
(`cabinet 1`, `mug 2`) — which the writer prompt explicitly forbids:

| entries with an instance name | ALFWorld refl. | AppWorld refl. | SSB refl. |
| --- | --- | --- | --- |
| Qwen3.5-9B | 10% | 18% | 13% |
| gpt-5.6-terra | 0% | 4% | 11% |
| gemini-3.7-flash | 0% | 25% | 22% |
| kimi-k3 | 3% | 33% | 12% |

What this looks like on the page. gemini's ALFWorld rules read like a
specification of the *simulator's* contract:

```
when an action to open a receptacle results in 'Nothing happens.'
  -> execute 'go to <receptacle>' before attempting to open it          [retrieved 49x]

when the arrival observation states 'On the <receptacle>, you see'
  -> avoid executing 'open <receptacle>'                                [retrieved 47x]
```

kimi's, from the same 50 episodes, are about *task state*:

```
when in a pick_two task you have just picked up the first required object and
are still holding it
  -> place the held object into the destination receptacle before attempting
     to take the second required object                                 [retrieved 28x]
```

Both are true. The first holds for every receptacle interaction in the
environment; the second holds inside one task family. That is how eight gemini
entries cover all six ALFWorld task types while kimi needs 51 — and it is the
mechanism behind §2, not a separate finding.

Qwen writes at the opposite pole, recording what happened rather than what to
do next: *"When the 'move' action to a fixed fixture (like a toilet) fails, the
agent should consider that the object might need to be placed 'on' the fixture
instead of 'in' it."*

## 4. gemini's entries are all load-bearing; a tenth of the others' are dead weight

Retrieval counts over the 100 evaluation tasks, ALFWorld `reflection`, 50
episodes:

| writer | live entries | mean retrievals/entry | never retrieved |
| --- | --- | --- | --- |
| gemini-3.7-flash | 7 | **39.9** | **0%** |
| gpt-5.6-terra | 21 | 20.4 | 0% |
| kimi-k3 | 37 | 11.9 | 11% |
| Qwen3.5-9B | 39 | 11.1 | 8% |

gemini's store has no entry that is never read, on any of the three
benchmarks, in either arm. Qwen and kimi always carry some: 8% and 11% in the
table above, up to 23% for Qwen's SpreadsheetBench `reflection` store — entries
that cost a writer call and store space and that no evaluation task ever reads. This is the
retrieval-side statement of §3: a slot-shaped rule matches more situations, so
fewer of them are needed and none of them is idle.

## 5. The rejection profile is the writer's fingerprint

Schema validation drops proposals that break the content type's contract. Which
constraint a writer trips is highly characteristic. ALFWorld/AppWorld/SSB,
`reflection` + `rule`, 50 episodes:

| writer | rejected proposals | reasons (one entry can trip several) |
| --- | --- | --- |
| gemini-3.7-flash | **11** | content too long 7, evidence too long 4, hallucination guard 1 |
| gpt-5.6-terra | 33 | content too long 32, hallucination guard 1 |
| Qwen3.5-9B | 107 | content too long 56, evidence too long 19, **hallucination guard 15**, exceeds n_max 10 |
| kimi-k3 | **185** | **content too long 168**, multiple actions 10, hallucination guard 8, evidence too long 4 |

The count is a description of writing style, not a quality metric on its own:
gpt-5.6 is rejected three times as often as gemini and matches it on two of the
three benchmarks.

Two Qwen-specific failures are worth naming because no remote writer produced
them.

**It re-proposes knowledge it already wrote.** Duplicate retrieval keys in the
live store, summed over these six cells: Qwen **11**, gpt-5.6 1, gemini 0,
kimi 0. Two Qwen entries under the same key:

```
id b070d824860e  "Always close the refrigerator door after placing an object
                  inside and before attempting to cool it."          [retrieved 10x]
id bad1c7632306  "Always move the object into the refrigerator before attempting
                  to cool it, as the cool action requires the object to be
                  inside the appliance."                             [retrieved  2x]
```

It also has 12 (ALFWorld) and 21 (AppWorld) entries at `version > 1` while
issuing **zero** explicit REVISE ops — those versions come from the store
merging near-duplicates it kept re-proposing, not from the writer deciding to
correct itself.

**It quotes its own memory block back as trajectory evidence.** The grounding
guard rejects entries whose `evidence` cannot be found in the rollout. Qwen's
rejected proposal:

```
evidence: "Nothing happens.\n  7a4068a7a0e3: When the agent attempts to take an
           object from a location and receives 'Nothing happens..."
```

`7a4068a7a0e3` is a memory item id. The writer copied a fragment of the injected
memory block into the field that is supposed to hold a verbatim environment
observation — it could not tell its own prior output from the trajectory.

Per benchmark, guard rejections are ALFWorld / AppWorld / SpreadsheetBench =
Qwen 1 / 5 / 9, gpt-5.6 0 / 0 / 1, gemini 0 / 0 / 1, kimi 0 / 0 / 8. So it is
not purely a small-model failure — kimi trips it eight times on
SpreadsheetBench — but on the two environments whose trajectories are prose,
only Qwen produces it.

## 6. kimi's `skill` collapse is a length distribution, not a competence gap

ALFWorld `skill` under kimi scores **57/100 — one point below having no memory
at all**. The cause is exact and is not about the quality of the procedures: 34
writer calls produced **2 accepted entries against 26 rejections**, 23 of them
`content too long` against the type's 400-token cap. The arm evaluates a
two-entry store.

Same task family, same JSON structure, same `<obj>`/`<recep>` slots — field
lengths in characters:

| field | gemini (accepted) | kimi (rejected, 493 tok > 400) |
| --- | --- | --- |
| `steps` | 356 | **583** |
| `fallback` | 277 | **386** |
| `verification` | 170 | **291** |
| `preconditions` | 60 | **172** |
| `name` | 36 | 42 |

kimi is 60–190% longer in every prose field and identical in structure. It is
writing the same procedure with more words, and the cap deletes all of it.

This generalises: kimi averages 114–124 tokens per `reflection` entry against
gemini's 93–110 and gpt-5.6's 91–107, and it is the only writer for which
`content too long` is the dominant rejection on all three benchmarks (168 of
its 185 rejected proposals). On SpreadsheetBench `rule` it lost **51** entries
to the cap while keeping 27.

**A caveat on what this licenses.** The 400-token cap is a shared calibration
constant, not a kimi-specific handicap, and it was not adjusted for any writer.
The honest reading is that kimi-k3's natural output length is incompatible with
this schema, *not* that kimi-k3 writes bad procedures. Separating those two
claims requires a run with a raised cap, which would not be comparable to
anything in this document.

## 7. The best writer here is also the cheapest by a factor of sixteen

Writer-side token spend and gateway cost for one 50-episode leg, three arms
(reflection + rule + skill), at listed gateway prices:

| benchmark | writer | calls | completion tok/call | USD |
| --- | --- | --- | --- | --- |
| ALFWorld | gemini-3.7-flash | 140 | 491 | **0.27** |
| ALFWorld | gpt-5.6-terra | 138 | 268 | 1.14 |
| ALFWorld | kimi-k3 | 134 | **1809** | **4.96** |
| AppWorld | gemini-3.7-flash | 116 | 416 | **0.23** |
| AppWorld | gpt-5.6-terra | 111 | 351 | 1.20 |
| AppWorld | kimi-k3 | 113 | **2206** | **5.10** |
| SpreadsheetBench | gemini-3.7-flash | 112 | 674 | **0.29** |
| SpreadsheetBench | gpt-5.6-terra | 114 | 332 | 1.18 |

Totals for the three e50 legs: **gemini $0.78, gpt-5.6 $3.52, kimi $12.84**
(Qwen is free — it shares the actor's GPU). kimi is a reasoning model and its
completion counts include reasoning tokens, which is where the 5–8× per-call
gap comes from; it also cost the most wall-clock, 18–40 s per writer call
against gemini's 6–9 s, and evolving is sequential.

So on this study's three benchmarks, the cheapest remote writer produced the
largest single effect anywhere measured (ALFWorld `reflection` at 100 episodes,
91/100, +33, b/c 3/36, p = 3.6e-8, from a twelve-entry store), and the most expensive one
produced the only writer arm to finish *below* the no-memory baseline.

## 8. What this does and does not support

**Supported.**

- Changing only the memory-writing model moves ALFWorld by up to 21 points at a
  fixed actor, budget and policy. The writer is a first-class variable, not an
  implementation detail.
- How much a writer writes does not predict how much it helps (ρ = −0.14 over
  135 cells); a 5-entry store and a 51-entry store scored the same on AppWorld
  `rule`.
- Writers differ systematically and legibly in *what* they write: slot-shaped
  environment contracts (gemini) vs task-state episodes (kimi, gpt-5.6) vs
  descriptions of what happened (Qwen). The slot style produces fewer, more
  frequently retrieved, never-idle entries.
- Schema-validation failure modes are writer-specific and are a cheap diagnostic:
  `content too long` means the writer is verbose, `hallucination guard` means it
  cannot separate its own memory from the trajectory.

**Not supported.**

- That a better writer helps in general. AppWorld says otherwise in 22 of 24
  cells, and its two significant cells belong to gpt-5.6 and to *Qwen* — the
  local writer this axis was supposed to improve on. ScienceWorld (reported
  separately) says the same with gpt-5.6.
- That gemini-3.7-flash is "the best writer". It is the best on ALFWorld by a
  wide margin, indistinguishable from gpt-5.6 on AppWorld and SpreadsheetBench,
  and the 4096-token budget it shares only with kimi is an unresolved confound.
- Any ordering among arms within a single cell on SpreadsheetBench or AppWorld —
  the deltas there are inside the measured noise floors.
- Anything about kimi-k3's `skill` quality (§6), or any SpreadsheetBench ×
  kimi-k3 number (still running).

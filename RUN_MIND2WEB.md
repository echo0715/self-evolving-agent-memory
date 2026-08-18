# Running the memsys self-evolving pipeline on Mind2Web

How to reproduce the Memory Content study on Mind2Web with the locally served
Qwen3.5-9B, on the Yale `radev` cluster. `RUN_ALFWORLD.md` is the reference
document for the shared machinery (vLLM, manifests, sweeps, the failure modes
that cost real time); this file covers only what is different here.

---

## 0. Where everything lives

| Thing | Location |
| --- | --- |
| Mind2Web data (`train`, `test_*`, ranker scores) | `/gpfs/radev/scratch/cohan/jw3278/mind2web_data` |
| Extracted per-step cache | `/gpfs/radev/scratch/cohan/jw3278/mind2web_cache` |
| Upstream reference implementation | `/gpfs/radev/scratch/cohan/jw3278/mind2web_repo` |
| Run outputs | `/gpfs/radev/scratch/cohan/jw3278/memsys_results/mind2web` |
| Conda env | `/gpfs/radev/scratch/cohan/jw3278/envs/memsys-alfworld` (+ `lxml`) |

No new environment: the ALFWorld env runs this adapter after `pip install lxml`.
There is no browser, no server, and no simulator — Mind2Web is offline.

## 1. Get the data

```bash
S=/gpfs/radev/scratch/cohan/jw3278
python - <<'PY'
from huggingface_hub import hf_hub_download
files = ["test.zip", "scores_all_data.pkl"] + [f"data/train/train_{i}.json" for i in range(11)]
for f in files:
    hf_hub_download("osunlp/Mind2Web", f, repo_type="dataset",
                    local_dir="/gpfs/radev/scratch/cohan/jw3278/mind2web_data")
PY
cd $S/mind2web_data && unzip -P mind2web test.zip -d test
```

Three things are easy to get wrong here:

- **`test.zip` is password-protected** (`mind2web`, published in the upstream
  README) to keep the test split out of crawled training corpora. `unzip` without
  it fails with `unable to get password` *per file* and still exits non-zero
  after "skipping" everything — it looks like a corrupt archive, not a locked one.
  Do not redistribute the unzipped files.
- **`scores_all_data.pkl` is not optional.** It holds the released DeBERTa
  candidate generator's per-element `scores`/`ranks`, and `rank < top_k` is what
  *defines* the candidate pool the model chooses from. Building a pool any other
  way (say, ground truth plus random negatives) would make the numbers
  incomparable to every published Mind2Web result, in the favourable direction.
- **The payload is ~12 GB** and every split file carries `raw_html` plus hundreds
  of candidates per step. It goes on scratch, and nothing reads it at run time
  (see §2).

## 2. Build the manifests and the step cache

```bash
python scripts/build_mind2web_manifests.py \
  --data-root $S/mind2web_data --out manifests --cache-root $S/mind2web_cache \
  --evolve-split train --evolve-count 50 \
  --eval-split test_task --eval-count 100 --seed 42
```

Committed as `manifests/mind2web_evolve_train_50_seed42.json` (50 annotations →
**416 steps**) and `manifests/mind2web_eval_test_task_100_seed42.json` (100
annotations → **838 steps**). Both are prefixes of one seeded permutation of the
split's sorted `annotation_id`s, so "evolve on 50 / 100" compares amount of
experience, exactly as on ALFWorld.

The continuation legs are the same permutation, further along — `--evolve-skip`
selects positions `[skip, count)`:

```bash
for pair in "50 100" "100 150" "150 200"; do set -- $pair
  python scripts/build_mind2web_manifests.py \
    --data-root $S/mind2web_data --out manifests --cache-root $S/mind2web_cache \
    --evolve-split train --evolve-skip "$1" --evolve-count "$2" --seed 42 --evolve-only
done
```

| manifest | annotations | steps | cumulative steps |
| --- | --- | --- | --- |
| `mind2web_evolve_train_50_seed42` | 50 | 416 | 416 |
| `mind2web_evolve_train_50to100_seed42` | 50 | 375 | 791 |
| `mind2web_evolve_train_100to150_seed42` | 50 | 390 | 1181 |
| `mind2web_evolve_train_150to200_seed42` | 50 | 378 | 1559 |

The four annotation sets are disjoint and none of them touches the 100 `test_task`
annotations — worth re-checking after any rebuild, because a builder bug here
would leak the test set silently:

```bash
python - <<'PY'
import json
seen = set()
for n in ["50", "50to100", "100to150", "150to200"]:
    ids = {t["annotation_id"] for t in
           json.load(open(f"manifests/mind2web_evolve_train_{n}_seed42.json"))["tasks"]}
    print(n, len(ids), "overlap_prev=", len(seen & ids)); seen |= ids
ev = {t["annotation_id"] for t in
      json.load(open("manifests/mind2web_eval_test_task_100_seed42.json"))["tasks"]}
print("eval overlap:", len(ev & seen))
PY
```

The builder also writes one small JSON per annotation under `--cache-root`,
holding only what the prompt needs: `cleaned_html`, the operation, the
annotator's action strings, and the candidate pools already merged with the
ranker's ranks and truncated to `--cache-top-k` (default 50). This is not an
optimisation — an evaluation worker cannot re-parse a 0.6 GB shard per step, and
`--cache-top-k` below the `--top-k` used at run time would silently shrink the
pool, so the builder's default is the maximum the paper uses.

**The unit is a step, not an annotation.** One memsys episode is one Mind2Web
step. `RESULTS_MIND2WEB.md` §0 argues the case; the short version is that
annotation-level success ("every step exactly right") runs at 0–2% for GPT-3.5
and GPT-4 in the paper, which would make every episode a failure, leave `raw`
and `skill` unable to write anything at all, and reduce the whole study to the
degenerate regime RESULTS_ALFWORLD.md §11 already measured.

## 3. Run

```bash
bash scripts/run_mind2web_sweep.sh smoke     # 8 evolve / 24 eval steps, all arms
bash scripts/run_mind2web_sweep.sh minimal   # 416 evolve / 838 eval steps
bash scripts/run_mind2web_sweep.sh full
bash scripts/run_mind2web_sweep.sh minimal100  # the next 50 annotations, resuming each store
bash scripts/run_mind2web_sweep.sh minimal150  # ... and the 50 after that
bash scripts/run_mind2web_sweep.sh minimal200  # ... and the 50 after that
```

The `*100` / `*150` / `*200` legs are **resumptions, not independent runs**: each
arm reloads its own `store.jsonl` and evolves the next 50 annotations of the same
permutation, so 50 / 100 / 150 / 200 are four points on one amount-of-experience
curve. Two consequences worth stating:

- **The chain is serial and unforgiving.** Leg *n+1* reads the store leg *n*
  wrote, so a leg that dies half-way leaves a store that must not be resumed. The
  sweep's own exit status will not tell you — it backgrounds its arms and ends on
  `summarize.py || true` — so check each arm's `summary.json` before continuing.
- **`--evolve-step-offset` is in steps, and it is not a constant.** Leg 1 evolved
  416 steps, leg 2 375, leg 3 390. The sweep reads the prior leg's
  `evolve_total` per arm rather than hard-coding 50-annotation increments;
  getting it wrong would restart `batch_every` consolidation from zero on every
  leg. `none` is skipped in all of them (no memory to grow) and its stored
  baseline is copied in for the summary table.

Same servers as everything else (`bash scripts/serve_qwen.sh 0 8000 --background`
and `1 8001`), same two-arms-at-a-time dispatch, one arm per GPU. The two
policies are independent, so on two allocations they run in parallel — but only
one of them may run the `none` arm, or both nodes race to write the same shared
baseline directory:

```bash
MEMSYS_ARMS="raw reflection rule skill" bash scripts/run_mind2web_sweep.sh full
```

`MIND2WEB_TOP_K` (default 50) sets the candidate pool depth. It is the single
biggest cost lever: top-50 costs ~9 LLM calls per step, top-10 costs ~3. It also
moves the ceiling, so it is part of the result, not a tuning knob — see §5.

## 4. Read the results

`scripts/summarize.py` works unchanged (`--root .../mind2web/minimal`). Two
Mind2Web-specific things to know when reading its table:

- The **rate column is step success rate** — the benchmark's headline metric and
  the one the paired McNemar test runs on, with one Bernoulli per step.
- The **score column is element accuracy**. Step success additionally requires
  the operation *and* its typed/selected value to be exactly right, so the gap
  between the two columns is "found the right element, then did the wrong thing
  to it".

`summary.json` additionally carries `eval_action_f1` and `eval_task_success_rate`
(every step of an annotation correct) — the latter is the number the Mind2Web
paper's headline table reports, and it is near zero for a 9B model.

## 5. Things that will bite

- **12.2% of eval steps are automatic zeros.** If the ground-truth element is not
  in the ranker's top-50, upstream scores the step 0 without calling the model,
  and this port does the same. Step SR is therefore capped at ~87.8% at top-50,
  and lower at top-10. The builder prints the count; `summary.json` records it as
  `eval_skipped_no_pos_candidate`. A memory system cannot fix these steps, so
  they dilute every delta.
- **The tournament's quirks are load-bearing.** Candidates are shown five at a
  time; each round's winner is re-queued; `final_prediction` is whatever the
  *last* non-"A" round chose, not a global argmax; a round answering "A. None of
  the above" contributes nothing. All three are upstream's behaviour and all
  three are ported verbatim. "Fixing" any of them redefines the metric.
- **Seed the candidate shuffle with a string, and still expect ~1.4% churn.** The
  shuffle is `random.Random("42/task_id/r0")` (sha512-based) on purpose: an
  earlier version seeded it with `hash()`, which Python randomises per process,
  and two identical runs disagreed on 2 of 24 steps. Fixing that is necessary but
  **not** sufficient. Measured on 2026-08-15, two replays of the no-memory
  baseline on one GPU, one server, one seed, at temperature 0 still disagree on
  **12 of 838 steps (1.4%)**, worth ±0.5–1.1 points of step SR — vLLM's
  reductions depend on batch composition, which temperature does not control.
  This file previously claimed the floor was zero on the strength of a 24-step
  check; at 1.4%, 24 steps come up clean 71% of the time. See
  RESULTS_MIND2WEB.md §1.2, and do not read a sub-point delta as an effect.
- **Teacher forcing means errors do not compound.** The "Previous actions" in the
  prompt are always the annotator's. An agent that would have gone off the rails
  at step 2 is still asked a well-posed question at step 7. This makes Mind2Web a
  much *gentler* setting than ALFWorld and is the main reason its numbers cannot
  be compared across benchmarks.
- **Check that the hardware still reproduces the baseline before starting a
  chain.** §1.2 of `RESULTS_MIND2WEB.md` is the load-bearing claim on this
  benchmark: temperature 0 plus a string-seeded candidate shuffle means every
  step where an arm differs from `none` differs *because of the memory block*.
  That is a property of a fixed serving stack, not a theorem — a continuation leg
  run on a different GPU generation is comparing against a baseline collected
  elsewhere. Replaying ~200 baseline eval steps on the new node and diffing
  `step_acc` and `element_acc` per `task_id` against the stored `eval.jsonl`
  costs ~20 minutes and tells you whether the legs you are about to spend 12 GPU
  hours on are measuring memory or kernels.
- **Cap `--max-context-tokens` (12k here).** A handful of pruned pages serialise
  past 30k tokens, which the 32k-context server rejects outright — and the
  adapter swallows the error into `error` on the rollout, so the run completes
  with quiet zeros. Upstream does not truncate at all in its LLM path.

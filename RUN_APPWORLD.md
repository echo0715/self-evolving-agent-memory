# Running the memsys self-evolving pipeline on AppWorld

How to reproduce the Memory Content study (`plan.md`) for AppWorld with a locally
served Qwen3.5-9B, on the Yale `radev` cluster.

[RUN_ALFWORLD.md](RUN_ALFWORLD.md) and [RUN_WEBSHOP.md](RUN_WEBSHOP.md) are the
companions. This file covers what is different about AppWorld, and the silent
failures that cost real time here.

**Before trusting any number this produces, read
[RESULTS_APPWORLD.md](RESULTS_APPWORLD.md)'s opening section.** The measured
run-to-run noise floor is ±5 points of TGC, which is larger than almost every
effect in the study. Run the baseline twice before running anything else.

---

## 0. Where everything lives

| Thing | Location |
| --- | --- |
| AppWorld data + task DBs (`APPWORLD_ROOT`) | `/gpfs/radev/scratch/cohan/jw3278/appworld_root` |
| AppWorld conda env (python 3.12) | `.../envs/memsys-appworld` |
| memsys conda env (python 3.11) | `.../envs/memsys-alfworld` |
| vLLM serving env | `.../verl_qwen35_train` |
| Run outputs | `.../memsys_results/appworld` |

### Two interpreters, one upstream server

AppWorld needs python 3.12 and pins `pydantic 1.10`; memsys runs on 3.11. Unlike
WebShop, no bridge had to be written: AppWorld ships
`appworld serve environment`, a FastAPI server exposing `/initialize`,
`/execute`, `/evaluate`, `/task_completed`, `/tasks/{id}`. The adapter
(`memsys/adapters/appworld.py`) is a thin client for it.

## 1. Serve the model and the environments

```bash
salloc -p gpu --gres=gpu:2 -c 16 --mem=256G -t 24:00:00

bash scripts/serve_qwen.sh 0 8000 --background
bash scripts/serve_qwen.sh 1 8001 --background
bash scripts/serve_appworld.sh 16          # :9000 .. :9015
```

**One server per concurrent episode, not per machine.** `appworld serve
environment` keeps a single module-level `world`, so a process hosts exactly one
live task. Two episodes sharing a URL do not error — the second `/initialize`
replaces the first's world mid-episode and the run scores against a task it never
ran. `run_appworld.py` leases URLs exclusively and caps its thread pool at the
pool size.

Servers are cheap: 16 of them cost ~6.4 GiB total and start in seconds. Memory is
not the constraint here; the LLM is.

## 2. Build the manifests

Reads the split files and each task's `specs.json` off disk — no server, no
`appworld` interpreter needed:

```bash
python scripts/build_appworld_manifests.py --root "$APPWORLD_ROOT" \
  --out manifests --evolve-count 50 --eval-count 100 --seed 42
```

Committed as `manifests/appworld_evolve_train_50_seed42.json` (50 of 90 `train`
tasks) and `manifests/appworld_eval_test_normal_100_seed42.json` (100 of 168
`test_normal`).

To continue an existing 50-episode run to 100 rather than repeating it:

```bash
python scripts/build_appworld_manifests.py --root "$APPWORLD_ROOT" --out manifests \
  --evolve-count 100 --evolve-skip 50 --evolve-overflow-split dev --no-eval
```

Committed as `manifests/appworld_evolve_train+dev_50to100_seed42.json`: positions
[50, 100) of the same permutation, so it is disjoint from the first 50 by
construction and the two together are the 100-task prefix.

**`train` runs out, and the continuation spills into `dev`.** AppWorld's train
split holds 90 tasks; 50 are already spent, so positions [50, 100) are 40 train +
10 dev. `dev` is disjoint from `test_normal`, so the evaluation set is untouched
— but a 100-episode AppWorld store has seen two splits where ALFWorld's and
WebShop's saw one. The manifest records this as `"splits": ["dev", "train"]`.
`--evolve-overflow-split` refuses the eval split outright.

Three deliberate properties:

- **Whole scenarios, not individual tasks.** AppWorld ids are
  `<scenario>_<variant>` and the variants of a scenario are near-identical
  requests over the same world. Selection is over scenarios so variants never
  straddle a boundary, which also keeps the `task_type` scope key meaningful.
- **Nested prefixes**, so "evolve on 50 vs 100" compares amount of experience.
- **`specs.json` only, never `ground_truth/`.** That directory holds the
  solution, the expected answer and `required_apps.json`. `required_apps` would
  be the natural cluster key and is deliberately *not* used — putting it in a
  scope key hands every memory system part of the solution path.

## 3. Run

```bash
bash scripts/run_appworld_sweep.sh smoke      # 2 evolve / 4 eval, all arms
MEMSYS_ARM_CONCURRENCY=5 MEMSYS_SERVERS_PER_ARM=3 \
  bash scripts/run_appworld_sweep.sh minimal  # 50/100
MEMSYS_ARM_CONCURRENCY=5 MEMSYS_SERVERS_PER_ARM=3 \
  bash scripts/run_appworld_sweep.sh full
```

The 50 → 100 continuation, once `full`/`minimal` exist:

```bash
MEMSYS_ARM_CONCURRENCY=5 MEMSYS_SERVERS_PER_ARM=3 \
  bash scripts/run_appworld_sweep.sh full100     # -> appworld/full_e100/
MEMSYS_ARM_CONCURRENCY=5 MEMSYS_SERVERS_PER_ARM=3 \
  bash scripts/run_appworld_sweep.sh minimal100
```

Each arm resumes **its own** `store.jsonl` from the 50-episode run and evolves
the next 50 tasks, so the column measures amount of experience rather than a
second independent draw. Two things do not round-trip through `store.jsonl` — the
batch-induction buffer, and the Evolver's step counter — so the sweep passes
`--evolve-step-offset 50`; without it, `full`'s every-25-episode induction would
restart at 0 and fire at different episodes than an uninterrupted 100-episode
run. The `none` arm has no store to resume and reuses the cached baseline.

Each 100-mode run gets its own AppWorld experiment namespace
(`memsys_<arm>_<tag>`), so a continuation can never race a re-run of the 50-task
sweep over the same `experiments/outputs/` tree.

Then, and this is not optional, a second baseline:

```bash
python scripts/run_appworld.py --arm none \
  --eval-manifest manifests/appworld_eval_test_normal_100_seed42.json \
  --server http://localhost:9000 --server http://localhost:9001 --server http://localhost:9002 \
  --out $MEMSYS_RESULTS_ROOT/appworld/_baseline_none_rep2 \
  --experiment-name memsys_none_rep2
```

It runs on the slot-0 servers, which sit idle once the cached `none` arm is
reused, so it costs ~68 minutes of otherwise-wasted capacity and is the only
thing that makes the rest of the table interpretable.

**Servers are partitioned across concurrent arms.** Arm *i* gets ports
`[9000 + i*SERVERS_PER_ARM, ...)`. Each arm process runs its own lease pool, so
two arms handed the same URL would each believe they held it exclusively.

Timing: evolving is sequential within an arm and is the wall-clock floor (~50
tasks × ~2 min). Five arms at once puts a policy at ~2.5–3 h; both policies plus
the baseline replicate fit in ~7 h.

## 4. Read the results

```bash
python scripts/summarize.py --root .../memsys_results/appworld/full
```

Reports TGC (Task Goal Completion — every unit test passes) and **score** (the
graded fraction of tests passed), with paired McNemar tests. Compute every arm
against **both** baselines; see `RESULTS_APPWORLD.md` for what happens when you
do not.

## 5. Silent failures that cost time here

Every one of these produced output that looked correct.

- **The writer's JSON did not parse, and nothing said so.** Asked to quote a
  trajectory verbatim into `evidence`, Qwen3.5-9B emits JSON it cannot escape —
  AppWorld trajectories are Python code and API docs full of quotes and
  backslashes, and it drops the escaping mid-string
  (`\"api_name": "show_transactions"`). `parse_ops` returns `[]` for unparseable
  text, so the arm logs **writer calls, zero ops and zero rejections** and ends
  with an empty store, indistinguishable from `none` while still reporting a
  success rate. Not truncation — the response completes cleanly at 2048 tokens —
  and no repair pass fixes it, because where the escaping was lost is ambiguous.
  Fix: `OpenAIChatClient(json_mode=True)`, i.e. `response_format={"type":
  "json_object"}`. Off by default so the other benchmarks stay reproducible.
- **The evidence cap rejected almost every write.** `MAX_EVIDENCE_TOKENS = 80` is
  calibrated for ALFWorld/WebShop observation prose; AppWorld evidence is code,
  and the dominant rejection reason became "evidence too long". It bites hardest
  on the type that writes least often (`skill` writes only from successful
  episodes), so it reads as a content-type result. `--max-evidence-tokens 300`,
  applied uniformly to every arm and recorded in `config.json`. Safe because
  `render_evidence` is False — evidence is provenance, not injected context.
- **All arms shared one experiment namespace.** AppWorld writes each task's
  working DBs under `experiments/outputs/<experiment_name>/tasks/<task_id>/` and
  re-prepares that tree on every `/initialize`. Concurrent arms sharing a name
  race: one arm's setup removes what another is about to use, and it surfaces as
  `FileNotFoundError` from `os.makedirs(..., exist_ok=True)` naming a path that
  plainly should exist. `--experiment-name` defaults to `memsys_<arm>_<policy>`.
- **`GET /api_docs` is not the app list.** It returns the complete API reference
  for all nine apps — 519 KB, ~140k tokens — where the scaffold's
  `{{ app_descriptions }}` slot wants the ~1.4 KB summary from
  `apis.api_docs.show_app_descriptions()`. Substituting the wrong one turns a
  6.6k-token prompt into a 147k-token one. `fetch_app_descriptions()` resolves it
  once at startup, outside episode accounting, so it does not consume one of the
  agent's 30 interactions.
- **`/evaluate` defaults `suppress_errors=False`.** An unfinished task makes the
  evaluator raise on its first failed assertion, which reaches the client as an
  opaque HTTP 500. Since most episodes are incomplete, this would drop the
  majority of the data. The adapter always passes `suppress_errors=True`.
- **Do not launch the sweep from a command that then blocks.** A backgrounded
  sweep is still in the launching command's process group; when that command hits
  a tool timeout the whole group dies, taking the sweep with it and leaving
  nothing in any log. Use `setsid`.

## 6. Scaffold provenance

`memsys/adapters/appworld_react_prompt.txt` is ACE's published AppWorld ReAct
generator prompt, copied verbatim from
`SAGE/repos/ACE/ace-appworld/experiments/prompts/`. It already contains a
`{{ playbook }}` slot — ACE's own name for injected context — so the memory block
goes where that harness put its context rather than somewhere this study
invented. This is the ALFWorld situation (byte-identical to a published harness)
rather than the WebShop one (written here).

The slot is filled for **every** arm, including `none`, which receives an
explicit "no playbook entries yet" placeholder. Dropping the section for the
no-memory arm would make it a different scaffold rather than the same scaffold
with empty memory.

ACE evaluated frontier models; this is a 9B model at a 30-interaction horizon
against AppWorld's default 50. Absolute TGC is not comparable to ACE's numbers.

# Running the memsys self-evolving pipeline on SpreadsheetBench

How to reproduce the Memory Content study (`plan.md`) for SpreadsheetBench with a
locally served Qwen3.5-9B, on the Yale `radev` cluster.

[RUN_ALFWORLD.md](RUN_ALFWORLD.md), [RUN_WEBSHOP.md](RUN_WEBSHOP.md) and
[RUN_APPWORLD.md](RUN_APPWORLD.md) are the companions. This file covers what is
different here, and the silent failures that cost real time.

**Read [RESULTS_APPWORLD.md](RESULTS_APPWORLD.md)'s opening section before
trusting any number.** The measured run-to-run noise floor on AppWorld was ±5
points, larger than most effects in the study. SpreadsheetBench's floor has not
been measured; run the `none` baseline twice before anything else.

---

## 0. Where everything lives

| Thing | Location |
| --- | --- |
| Dataset (`SPREADSHEETBENCH_ROOT`) | `/gpfs/radev/scratch/cohan/jw3278/spreadsheetbench_root/all_data_912_v0.1` |
| Verified-400 archive (alternative) | `.../spreadsheetbench_root/spreadsheetbench_verified_400` |
| memsys conda env (python 3.11) | `.../envs/memsys-alfworld` |
| vLLM serving env | `.../verl_qwen35_train` |
| Run outputs | `.../memsys_results/spreadsheetbench` |

### One interpreter, no environment server

This is the simplest of the four to operate. WebShop needs a python-3.8 bridge
process and AppWorld needs a pool of upstream FastAPI servers; SpreadsheetBench
needs neither. The dataset is a directory of `.xlsx` files, upstream's evaluator
is ported into `memsys/adapters/spreadsheetbench.py`, and an episode is a temp
directory plus subprocesses. Setup is data plus two libraries.

It is also the only adapter that executes model-written shell commands **in the
memsys process's own environment**. `_run_bash` is `subprocess.run(shell=True)`
with a timeout and `cwd` set to a per-episode temp directory, which is what
upstream does, but it is not a sandbox. Run sweeps as an unprivileged user on
scratch.

## 1. Set up the data

```bash
bash scripts/setup_spreadsheetbench.sh
export SPREADSHEETBENCH_ROOT=/gpfs/radev/scratch/cohan/jw3278/spreadsheetbench_root/all_data_912_v0.1
```

Fetches both archives from `KAKA22/SpreadsheetBench` at revision `ab0b742`, and
installs `openpyxl` + `pandas` into the memsys interpreter.

**The libraries must go in the memsys interpreter, not somewhere else.** They are
needed three times over: by the evaluator, by the agent's `python solution.py`
inside the loop, and by `run_generated_code` when it re-applies that script to
cases 2 and 3 using `sys.executable`. The adapter puts `sys.executable`'s
directory first on the agent's `PATH` so all three are the same interpreter —
without that, a bare `python` on this cluster resolves to the system miniconda
and every episode dies with `ModuleNotFoundError: No module named 'openpyxl'`
against a prompt that promised openpyxl was available. It looks exactly like a
model that cannot write Python.

## 2. Build the manifests

```bash
python scripts/build_spreadsheetbench_manifests.py     # 50 evolve / 100 eval, seed 42
```

Writes `manifests/spreadsheetbench_evolve_train_50_seed42.json` and
`manifests/spreadsheetbench_eval_test_100_seed42.json`, both self-contained.

Task selection reuses **SkillOpt's published id split** for this benchmark
(`data/spreadsheetbench_id_split`: the 400-task Verified subset partitioned
train=80 / val=40 / test=280), so the task set matches the rest of the SAGE
ecosystem. The sibling SkillOpt checkout is a build-time dependency only.

## 3. Run

```bash
salloc -p gpu --gres=gpu:2 -c 32 --mem=256G -t 24:00:00
bash scripts/serve_qwen.sh 0 8000 --background
bash scripts/serve_qwen.sh 1 8001 --background

bash scripts/run_spreadsheetbench_sweep.sh smoke      # 2 evolve / 4 eval, all arms
bash scripts/run_spreadsheetbench_sweep.sh minimal
bash scripts/run_spreadsheetbench_sweep.sh full
python scripts/summarize.py --root .../memsys_results/spreadsheetbench/full
```

Concurrency is a CPU question, not a port question. There is no lease pool and
arms cannot corrupt each other's world, so the only ceiling is
`ARM_CONCURRENCY x EVAL_WORKERS` processes on this node — each one running the
agent's bash commands plus, at scoring time, up to two more `python solution.py`
subprocesses that load workbooks into memory. A few tasks have answer ranges of
~100k cells. Defaults are 4 x 6; drop `MEMSYS_EVAL_WORKERS` first on a small
node.

---

## The traps

### 1. Three test cases, and the whole benchmark depends on it

A task ships `1_<id>_input.xlsx` .. `3_<id>_input.xlsx` with matching `_answer`
files. **The agent only ever sees case 1.** Afterwards its `solution.py` is
re-executed against cases 2 and 3 with the paths swapped, and a task counts as
solved only if all three pass.

This is the benchmark's anti-hardcoding device and it is load-bearing. An agent
that reads the preview and writes literal answers passes case 1 and fails the
rest, scoring `reward = 1/3` and `success = False`. Both numbers are reported
because they separate exactly that behaviour from real generalisation:

- `eval_success_rate` — all cases pass. The strict number, upstream's metric.
- `eval_score` — mean `n_pass / n_cases`. The graded one.
- `eval_wrote_solution_rate` — did the agent produce a reusable script at all?
  The analogue of WebShop's purchase rate: it separates "solved case 1 by hand
  and could not generalise" from "never produced anything".
- `eval_cell_match_rate` — fraction of graded cells matching gold. **Diagnostic
  only, never a reported score.** Pass/fail cannot tell "one cell off by a
  rounding rule" from "wrote nothing", and that distinction is what makes a
  failed episode readable.

### 2. Use the 912 archive, and never mix it with Verified 400

Both archives cover every id in the split, but:

| | tasks | cases/task |
| --- | --- | --- |
| `all_data_912_v0.1` | 912 | 3 (905 tasks), 2 (4), 1 (3) |
| `spreadsheetbench_verified_400` | 400 | **1** |

With one case per task, `reward` collapses onto `success` and the
generalisation check above disappears entirely — an agent can hardcode and score
1.0. The manifests therefore point at the 912 archive. `--data-root` accepts
Verified 400 and the adapter reads its `_init`/`_golden` naming, but a run on it
measures something weaker, and the manifest records which root it was built
against.

**Never pair one archive's metadata with the other's files.** Verified 400
revised instructions *and* regenerated golden workbooks; grading a revised
instruction against the original answers scores the wrong task, and nothing about
the resulting number looks wrong.

### 3. An Excel formula scores zero with correct arithmetic

`=SUM(A1:A3)` written through openpyxl has no cached value, and upstream's
evaluator loads with `data_only=True`, so the graded cell reads back as `None`.
The agent has done the work and scores 0.

Two defences are in the scaffold, both carried over from SkillOpt, which is where
this was diagnosed: `CRITICAL_RULES` states it as rule 1, and `_auto_verify`
inspects the written workbook after every `python solution.py` and names the
offending coordinates. Check `mean_bash_errors` and the `eval_fail_reasons`
histogram in `summary.json` if a run scores far below its `cell_match_rate`.

`_auto_verify` is deliberately **gold-free** — it reports what the agent wrote,
never what the answer is. SkillOpt has a second verifier that diffs against the
golden workbook, but that one is only wired into its code-generation path.
Putting it here would leak the answer into the trajectory and from there into
every memory entry written from it. `test_auto_verify_never_reveals_the_expected_answer`
asserts this.

### 4. The evaluator quantizes; it does not tolerate

Numbers are compared after `round(float(v), 2)`. That is not a tolerance:
`1.234` and `1.2349` are equal (both round to `1.23`) while `1.234` and `1.236`
are not. `None` and `""` are equal; otherwise a type mismatch fails the cell
outright. Formatting is never compared, because upstream does not compare it.

`compare_workbooks` is a port of `evaluation/evaluation.py` from
RUCKBReasoning/SpreadsheetBench. Any "improvement" to it silently redefines the
benchmark. It is pinned by `CellComparatorTest`.

### 5. The scaffold was written here, like WebShop's

Upstream's and SkillOpt's ReAct agents both drive the model through OpenAI
**native tool calls** (`bash`, `write_file`). `scripts/serve_qwen.sh` does not
pass `--enable-auto-tool-choice`, and enabling it would change the server every
other benchmark in this study is running against. So the same two tools are
expressed as fenced blocks and parsed in the adapter, matching the other three:
a ```python block is saved as `solution.py`, a ```bash block is executed.

Consequence: the absolute number is not comparable to published SpreadsheetBench
results, and **the `none` arm is the only reference that means anything.** This
is the WebShop situation, not the ALFWorld one. `CRITICAL_RULES` and the protocol
are semantically SkillOpt's; the wording is not byte-identical to anything
published.

Every arm shares this scaffold exactly. The memory section is filled for *all* of
them — `none` gets `EMPTY_MEMORY`, an explicit "no entries yet" placeholder —
because dropping the section would make the no-memory arm a different scaffold
rather than the same scaffold with an empty memory.

### 6. Grounding evidence is code, so raise the cap

`--max-evidence-tokens` defaults to 300 here rather than the 80 used on ALFWorld
and WebShop, for the same reason as AppWorld: the writer is asked to quote the
trajectory verbatim into `evidence`, and the trajectory is Python and shell
output. At 80 the dominant rejection reason is "evidence too long", and it bites
hardest on the content types that write least often. Applied identically to every
arm, before any system is built.

`--writer-json-mode` is on by default for the matching reason — Qwen3.5-9B loses
its string escaping partway through quoted code and the response parses to
nothing, leaving an arm with an empty store that is silently identical to `none`.

### 7. The scope key has only two values

`task_type` is `cell_level` or `sheet_level`, because those are the only labels
SpreadsheetBench carries. That is coarser than ALFWorld's six procedure families
and closer to WebShop's product departments, so expect little from batch
induction clustering on it — and do not read that as a property of a memory type.

Task ids are `<workbook>-<question>` for sheet-level tasks, and questions sharing
the leading number are different asks over the *same* spreadsheet. The manifest
builder selects whole groups and drops any evaluation task whose workbook was
used for evolving; three ids in the published split (`493-5`, `408-5`, `82-38`)
share a group across train and test. `--allow-group-leak` reproduces a run built
without the guard.

### 8. The dataset must never be written through

The agent works on a *copy* of case 1's input in a temp directory. Writing
through to the dataset would corrupt the benchmark for every later episode, and
the only symptom would be unexplained score drift.
`test_the_dataset_workbook_is_never_written_through` asserts it.

---

## Reading `summary.json`

```
eval_success_rate       all test cases pass -- upstream's metric
eval_score              mean n_pass / n_cases -- the graded one
eval_wrote_solution_rate  produced a reusable solution.py
eval_cell_match_rate    diagnostic, never a reported score
eval_by_family          split by the scope key: cell_level / sheet_level
eval_fail_reasons       histogram over fail_reason kinds:
                          eval-mismatch   ran, produced a workbook, wrong values
                          exec-error      solution.py crashed on case 2 or 3
                          output-not-found  never wrote the output file
                          no-solution-py-for-other-cases  edited case 1 by hand
mean_bash_errors        tracebacks inside the agent loop
```

`exec-error` and `no-solution-py-for-other-cases` are the two that mean
"generalisation failed" rather than "the answer was wrong", and they are the
ones a procedural-skill memory should move first.

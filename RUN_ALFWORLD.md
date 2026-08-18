# Running the memsys self-evolving pipeline on ALFWorld

How to reproduce the Memory Content study (`plan.md`) for ALFWorld with a locally
served Qwen3.5-9B, on the Yale `radev` cluster.

`README.md` explains *what* the memory systems are. This file is *how to run
them*, plus the failures that cost real time so you do not rediscover them.

---

## 0. Where everything lives

Heavy artifacts go on **scratch**, never in the repo or `$HOME` (home is at its
inode quota; the shared `cohan` *project* group quota has been full before and
fails intermittently with `OSError [Errno 122]`).

| Thing | Location |
| --- | --- |
| ALFWorld data (`ALFWORLD_DATA`) | `/gpfs/radev/scratch/cohan/jw3278/alfworld_data` |
| memsys + ALFWorld conda env | `/gpfs/radev/scratch/cohan/jw3278/envs/memsys-alfworld` |
| vLLM serving env | `/gpfs/radev/scratch/cohan/jw3278/verl_qwen35_train` |
| Qwen3.5-9B weights (19 GB) | `/gpfs/radev/scratch/cohan/jw3278/hf_cache` |
| Run outputs | `/gpfs/radev/scratch/cohan/jw3278/memsys_results` |
| AppWorld env / data | `.../envs/memsys-appworld`, `.../appworld_root` |
| WebShop env / repo | `.../envs/memsys-webshop`, `.../webshop_repo` |

The env was cloned from SAGE's working `sage-alfworld` (alfworld 0.4.2 +
textworld 1.6.2) and extended with `openai`, `tiktoken`, `sentence-transformers`.
Rebuilding ALFWorld from scratch is avoidable; cloning is not laziness, it is
skipping a known-fiddly `fast_downward` build.

## 1. Get a GPU node and serve the model

```bash
salloc -p gpu --gres=gpu:2 -c 8 --mem=64G -t 24:00:00

bash scripts/serve_qwen.sh 0 8000 --background   # GPU 0
bash scripts/serve_qwen.sh 1 8001 --background   # GPU 1
until curl -sf -o /dev/null localhost:8000/health && \
      curl -sf -o /dev/null localhost:8001/health; do sleep 10; done
```

Two TP=1 servers, not one TP=2 server: the 9B needs ~19 GB of an 80 GB card, and
these A100s are PCIe-connected, so tensor parallelism would pay interconnect cost
for no capacity gain. Two independent servers give clean 2x throughput instead.

Three non-obvious requirements are baked into `serve_qwen.sh` — reproduce them if
you launch vLLM by hand:

1. **`ninja` and `nvcc` on `PATH`.** Qwen3.5's Gated-DeltaNet layer JIT-compiles
   kernels at load. Calling `$VLLM_ENV/bin/vllm` directly does not put the env's
   `bin` on `PATH`, and vLLM then reports only `Engine core initialization
   failed` while the real cause, `FileNotFoundError: 'ninja'`, is buried in the log.
2. **`--additional-config '{"gdn_prefill_backend":"triton"}'`.** Gated-DeltaNet's
   flashinfer prefill kernel is broken on this cluster.
3. **vLLM >= 0.24**, the first release whose registry has
   `Qwen3_5ForConditionalGeneration`. The driver is 570.x / CUDA 12.8, so the env
   uses `+cu129` wheels, not the PyPI-default CUDA 13 build.

Stop with `pkill -f 'verl_qwen35_train/bin/vll[m] serve'` (the character class
keeps the pattern from matching the shell you type it in).

## 2. Build the frozen manifests

```bash
export ALFWORLD_DATA=/gpfs/radev/scratch/cohan/jw3278/alfworld_data
python scripts/build_manifests.py --data-root "$ALFWORLD_DATA" --out manifests \
  --evolve-split train --evolve-count 50 \
  --eval-split valid_unseen --eval-count 100 --seed 42
```

Committed as `manifests/evolve_train_50_seed42.json` (50 `train` tasks) and
`manifests/eval_valid_unseen_100_seed42.json` (100 `valid_unseen` tasks). Once a
run starts, never regenerate under the same filename.

Two properties are deliberate:

- **Nested prefixes.** Selection is a prefix of one seeded permutation, not
  `random.sample(k)`. The 50-task evolve set is therefore a prefix of the
  100-task set, so the "evolve on 50 / 100 / 150" columns of `plan.md` compare
  *amount of experience*, not two unrelated task draws.
- **Resolved goal text.** Each entry carries its `"Your task is to: ..."` string.
  Retrieval must happen *before* the agent acts, but ALFWorld only reveals the
  goal on `reset()`. Without pre-resolving it, every memory system would have to
  retrieve on the gamefile path — a much weaker key, and one that would quietly
  handicap all arms equally while looking like it worked.

## 3. Run

```bash
bash scripts/run_sweep.sh smoke      # 2 evolve / 2 eval, all arms -- validates plumbing
bash scripts/run_sweep.sh minimal    # 50/100, WritePolicy.minimal()
bash scripts/run_sweep.sh full       # 50/100, WritePolicy.full()
```

Two continuation families, both resuming each arm's own `store.jsonl` and both
passing `--evolve-step-offset` so the Evolver's step counter (and with it
`full`'s every-25-episode batch induction) lines up with an uninterrupted run:

```bash
bash scripts/run_sweep.sh full100    # the NEXT 50 train tasks   -> full_e100/
bash scripts/run_sweep.sh full_x2    # the SAME 50 again         -> full_x2/
bash scripts/run_sweep.sh full_x3    # ... and a third time      -> full_x3/
```

They measure different things and their numbers are not interchangeable —
`*100` varies amount of experience, `_x2`/`_x3` vary repetition over fixed
experience (RESULTS_ALFWORLD.md §6 vs §7). `_x3` requires `_x2` to exist.
`*150` continues the first family to positions [100,150) and requires `*100`.

A third pair crosses the two axes at a fixed episode budget:

```bash
bash scripts/run_sweep.sh full75     # 75 tasks from an EMPTY store -> full_e75/
bash scripts/run_sweep.sh full75_x2  # the SAME 75 again            -> full_e75_x2/
```

75 × 2 spends the same 150 evolving episodes as `*150` (150 distinct tasks) and
as `_x3` (50 × 3), differing only in how many distinct tasks they cover, which
is what makes diversity and repetition separable (§8). Note that `full75` starts
from an empty store — it is a new chain, not a continuation — so the manifest
`evolve_train_75_seed42.json` is positions [0,75) and nests inside the 50/100/150
sets. Build it without re-resolving goal text, since it is exactly the 50-task
manifest plus the first 25 entries of the 50→100 one; verify with
`random.Random(42).shuffle` over the sorted gamefile list before trusting the
concatenation.

A fourth pair budgets by *outcome* instead of by task count (RESULTS §10, §11):

```bash
bash scripts/run_sweep.sh full_succ100   # evolve until 100 episodes SUCCEED, write only those
bash scripts/run_sweep.sh full_fail100   # evolve until 100 episodes FAIL,    write only those
```

Both start from an empty store over `evolve_train_600_seed42.json` (a nested
superset of every earlier manifest — verify with the prefix check below before
trusting it), and both stop on an outcome count, so the number of tasks spent
differs per arm: ~135–176 for 100 successes, ~184–257 for 100 failures. The
runner *warns* rather than finishing short if the manifest runs out first, so
grep the arm logs for `WARNING: manifest exhausted` before reading any number.

`raw` is excluded from both (the default arm list for these modes drops it):
`RawTrajectorySystem.observe` keeps only `best_success()`, so the success filter
is a no-op for it and the failure filter leaves it with an empty store. Two
further degeneracies are worth knowing before spending the GPU-hours, because
both produce a complete-looking summary:

- **`skill` + `--failure-only-writes` never calls the writer.**
  `SkillWriter.propose` returns early without a successful rollout, so at one
  rollout per task the arm makes zero LLM calls and evaluates an empty store.
- **`full` + `--failure-only-writes` deletes almost everything.**
  `record_usage` only ever sees `success=False`, so every entry hits the
  `utility < 0.20` floor on its fifth retrieval. Stores end at ~2 live entries.

```bash
python - <<'EOF'   # every earlier manifest must be a slice of the 600
import json
b=[t['task_id'] for t in json.load(open('manifests/evolve_train_600_seed42.json'))['tasks']]
for p,(i,j) in {'evolve_train_50_seed42':(0,50), 'evolve_train_50to100_seed42':(50,100),
                'evolve_train_100to150_seed42':(100,150), 'evolve_train_150to200_seed42':(150,200),
                'evolve_train_300_seed42':(0,300)}.items():
    o=[t['task_id'] for t in json.load(open(f'manifests/{p}.json'))['tasks']]
    print(p, 'nested' if o==b[i:j] else 'MISMATCH')
EOF
```

The two policies are independent chains, so on two allocations they run in
parallel — one policy per node, ~10 h each for both extra epochs:

```bash
srun --jobid=<other-job> --overlap -N1 -n1 bash /path/to/driver.sh &
```

The per-node driver must start its own vLLM pair; arm concurrency stays at 2
(see below).

Arms: `none`, `raw`, `reflection`, `rule`, `skill`. `none` is the no-memory
reference — without it no delta is interpretable, because ALFWorld's run-to-run
noise floor is several points (SAGE measured 54 / 58 / 60% across three
*identical* baseline runs).

One arm can also be run alone:

```bash
python scripts/run_alfworld.py --arm rule --policy full \
  --evolve-manifest manifests/evolve_train_50_seed42.json \
  --eval-manifest   manifests/eval_valid_unseen_100_seed42.json \
  --out $MEMSYS_RESULTS_ROOT/full/rule_full --agent-base-url http://localhost:8000/v1
```

### Varying the *writer* model (the Memory Writing Model axis)

Everything above uses one model for both jobs: Qwen3.5-9B acts *and* writes the
memory. `--writer-model` separates them, holding the actor fixed so any delta is
attributable to what was written rather than to who acted.

```bash
srun --jobid=<job> --overlap -N1 -n1 bash scripts/drive_gpt56_writer.sh
```

That driver serves the two local Qwen servers for the agent, smoke-tests, and
then runs `minimal` with the writer pointed at `openai/gpt-5.6-terra`. Its
knobs are plain environment variables on `run_sweep.sh`:

| Variable | Value used here |
| --- | --- |
| `MEMSYS_WRITER_MODEL` | `openai/gpt-5.6-terra` |
| `MEMSYS_WRITER_BASE_URL` | `https://api.perplexity.ai/v1` |
| `MEMSYS_WRITER_API` | `responses` |
| `MEMSYS_WRITER_API_KEY_ENV` | `perplexity_api_key` (read from `.env`) |
| `MEMSYS_TAG_SUFFIX` | `_gpt56terra` |

Four things about that endpoint are not guessable and each costs a run to find:

- **The gateway speaks only the Responses API.** `pplx-*` keys reach two
  different services on one host. `POST /chat/completions` is Perplexity's own
  Sonar API and answers every gateway model id with
  `invalid_model`; `POST /v1/responses` is the multi-provider gateway and serves
  `openai/gpt-5.6-{luna,terra,sol}`, `anthropic/*`, `google/*`, `xai/*`
  (`GET /v1/models` lists them with per-token prices). `/v1/chat/completions`
  is a 404. `OpenAIChatClient` therefore cannot be pointed at it at all — hence
  `OpenAIResponsesClient` and the `--writer-api` switch.
- **`text.format={"type":"json_object"}` is rejected** (400 `invalid request`),
  so there is no constrained-decoding path here. The writers' own JSON repair
  carries the parse rate; do not pass `--writer-json-mode`-style flags.
- **A reasoning model can spend its whole output budget thinking** and return
  `status="incomplete"` with empty text. That is silent — `parse_ops("")` is
  `[]`, so the arm logs writer calls, zero ops, no error, and finishes with an
  empty store that reads exactly like the `none` baseline. `OpenAIResponsesClient`
  retries with a doubled `max_output_tokens` instead of returning the blank.
- **`MEMSYS_TAG_SUFFIX` is not optional.** Without it a writer-model sweep
  writes into `memsys_results/minimal/`, on top of the Qwen-writer numbers it
  exists to be compared against.

`none` and `raw` never call a writer LLM (`RawTrajectorySystem` stores the
trajectory verbatim), so the writer model cannot move them and the driver does
not re-run them — compare against `minimal/{none,raw}_minimal`. Only
`reflection`, `rule` and `skill` are re-measured.

Cost is small enough not to plan around: the 50-task `minimal` sweep is ~385k
prompt + ~52k completion tokens across the three arms, about **$1.50** at
terra's $2/$12 per 1M. `summary.json` now carries `writer_model` alongside
`model` so the two runs are distinguishable after the fact.

### Parallelism rules (they are not the same in both phases)

- **Evolving is strictly sequential.** Memory written after episode *i* is
  retrieved by episode *i+1*. Parallelising it would not merely be unsafe, it
  would measure a different system.
- **Evaluation runs under `frozen()`** — every write raises — so tasks are
  independent and fan out. It uses **processes, not threads**: see §5.
- **Across arms**, two at a time, one per GPU.

## 4. Read the results

```bash
python scripts/summarize.py --root /gpfs/radev/scratch/cohan/jw3278/memsys_results/full
```

Writes `RESULTS.md` with success rate, delta vs the `none` baseline, **b/c
discordant counts and a McNemar exact p-value**, store size, injected tokens and
writer cost.

Report the paired test, not the raw delta. Every arm evaluates the identical
ordered task list, so the comparison is paired; SAGE reported a −10 point delta
on this benchmark that was **p = 0.383** — pure churn in both directions.

## 5. Failures that cost time here

- **`pkill -f` in a teardown script is a node-wide weapon, and a dead vLLM
  produces completed runs.** A 4-GPU node holds two 2-GPU jobs. On 2026-08-13 a
  finishing job ran `pkill -9 -f 'VLLM::EngineCore'` on its way out and killed
  the servers of an unrelated run on the same host. The victim then evaluated
  for two hours against nothing and **reported `rc=0` with complete
  `summary.json` files**: the agent swallows the connection error, every episode
  scores a failure, and `Connection error` appears zero times in the arm log.
  `reflection/full` came back 10.0% where a clean re-run gives 65.0%. The only
  tell is timing — eval episodes finishing in 4–5 s instead of 200–300 s — so
  check `[eval` line durations before believing any number, and compare
  `summary.json` mtimes against when the servers were last known healthy.
  Teardown must kill recorded PIDs (plus `pkill -P` for the renamed
  `VLLM::EngineCore` child), never a pattern; startup must refuse a busy port
  rather than clearing it. Note that a *process-group* kill is not the fix
  either: `serve_qwen.sh` backgrounds with `nohup` and not `setsid`, so the
  servers share the driver's pgid and `kill -- -PGID` takes the driver with them
  — one batch job finished everything, wrote its `RESULTS.md`, and still
  reported `FAILED` with `ExitCode 0:9`.
- **A driver must not export `HF_HOME` before calling `serve_qwen.sh`.** There
  are two caches and they are not interchangeable: `$SCRATCH/hf_cache` holds the
  Qwen3.5-9B snapshot, `$SCRATCH/.hf_home` holds sentence-transformers' encoder
  and is what the sweep scripts export for their own python. `serve_qwen.sh`
  takes `HF_HOME` from the environment (`${HF_HOME:-$SCRATCH/hf_cache}`), so a
  driver that sets the sweep's value up front points vLLM at a cache with no
  model in it. With `HF_HUB_OFFLINE=1` that is a `LocalEntryNotFoundError`
  *seconds after* `[serve] started (PID …)` prints — the driver reports a healthy
  PID, `nvidia-smi` shows 0 MiB, and the health-check loop then sits there for
  its full timeout. Set `HF_HOME` per-command on the memsys python invocation, or
  `unset` it before serving. Cost on 2026-08-15: one Mind2Web preflight.
- **A driver that starts vLLM must check the port first.** `serve_qwen.sh`
  backgrounds the server and returns; the health-check loop that follows then
  passes *immediately* if a previous allocation's servers are still listening on
  8000/8001, and the duplicates it just launched sit there trying to take 90% of
  a GPU that is already spoken for. On 2026-08-14 that happened on the second
  node of the `fail100` sweep: two extra `vllm serve` processes, no ports, no
  GPU memory, and a driver log that read `servers healthy` in the same second it
  started them. The tell is the timestamp — a real cold start is 4–5 minutes.
  Kill the duplicates by *recorded PID* (`pkill -P $pid` first for the renamed
  `VLLM::EngineCore` child), never by pattern, then confirm with
  `nvidia-smi --query-compute-apps` that the survivors are the ones holding the
  memory.
- **ALFWorld's PDDL backend is not thread-safe, in two separate places.**
  Loading a game parses the grammar with a *module-level* tatsu parser, and
  stepping goes through textworld's PDDL layer. Running evaluation 2-wide with
  threads produced `IndexError: pop from empty list` and
  `TypeError: 'NoneType' object is not subscriptable` deep inside tatsu — neither
  traceback mentions concurrency, and a sequential run never reproduces it. The
  fix is `ProcessPoolExecutor` (each worker gets its own copy of everything),
  plus a lock around registration/reset for any in-process use. Covered by
  `ConcurrentResetTest`, which skips without `ALFWORLD_DATA`.
- **`appworld download data` defaults `--root` to the CWD.** Run from the repo,
  it silently drops 162 MB of benchmark payload into the working tree. Always
  pass `--root`; `data/` is gitignored as a backstop.
- **Cap the agent's `max_tokens` (256 here).** SAGE lost a whole run to the
  opposite: at 2048 with no history management, Qwen filled the ceiling every
  turn, prompts reached 129k tokens, and **63.6% of calls died with HTTP 400** —
  silently, because the agent swallowed the error and kept going. The run looked
  healthy and still reported numbers.
- **Do not change the scaffold.** The system prompt and six demonstrations come
  verbatim from SAGE (`MemRL/memrl/agent/prompts.py`,
  `configs/alfworld/alfworld_examples.json`). Swapping in ACE's generic prompt
  with no demonstrations scored **18% vs 60%** on the same model and tasks. A
  40-point swing from prompt wording alone dwarfs any memory effect. If the
  `none` arm does not land near 58-60%, suspect the scaffold before believing
  any result.
- **Silence is not progress.** Check per-episode lines in the arm logs and CPU
  time growth; a hung worker looks exactly like a slow one.

## 6. Other benchmarks

```bash
bash scripts/setup_appworld.sh                 # 90 train / 57 dev / 168 test_normal / 417 test_challenge
bash scripts/setup_webshop.sh small            # 1,000 products indexed, 6,910 goals; `all` for the full corpus
```

Both are installed and verified (WebShop resets and returns a real instruction;
AppWorld loads all four splits). Neither has a memsys *adapter* yet — that is
the remaining work to extend this study beyond ALFWorld.

AppWorld is clean. WebShop needed six separate fixes, all of the same shape —
upstream pins a top-level package and lets its dependencies float, so five years
later the transitive resolution is broken. Each one fails with an error that
names something unrelated to the real cause:

| Symptom | Cause | Fix (in the script) |
| --- | --- | --- |
| `gdown`: "Cannot retrieve the public link" | corpus is on Google Drive, rate-limited | fetch from the HF mirror `YWZBrandon/webshop-data` first |
| `ImportError: libmkl_intel_lp64.so.1` | `faiss-cpu` from the `pytorch` channel links MKL | install `faiss-cpu` from conda-forge |
| `cannot import name 'url_quote' from 'werkzeug.urls'` | Flask 2.1.2 pinned, Werkzeug floated to >= 2.1 | pin `werkzeug<2.1`, `jinja2<3.1`, `itsdangerous<2.1` |
| `TypeError: issubclass() arg 1 must be a class` | spacy 3.3 + pydantic 1.x vs `typing_extensions>=4.6` | pin `typing_extensions<4.6` |
| `OSError [E050] Can't find model 'en_core_web_sm'` | setup.sh downloads only `_lg`; runtime loads `_sm` | download both |
| "Indexing Complete! **0 documents indexed**", exit 0 | conversion crashed; indexer does not care | assert `documents.jsonl` is non-empty before indexing |

That last one is the dangerous one: upstream's `run_indexing.sh` exits 0 after
indexing nothing, so a completely broken WebShop looks like a successful setup
until an agent silently retrieves no products.

The HF mirror is third-party, not Princeton's distribution. Filenames match and
the script checks record counts, but confirm provenance before publishing.

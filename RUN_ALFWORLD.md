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

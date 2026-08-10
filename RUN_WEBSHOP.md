# Running the memsys self-evolving pipeline on WebShop

How to reproduce the Memory Content study (`plan.md`) for WebShop with a locally
served Qwen3.5-9B, on the Yale `radev` cluster.

[RUN_ALFWORLD.md](RUN_ALFWORLD.md) is the companion for ALFWorld and most of the
model-serving section is shared. This file covers what is different about
WebShop, plus the failures that cost real time.

---

## 0. Where everything lives

| Thing | Location |
| --- | --- |
| WebShop repo (vendored, patched) | `/gpfs/radev/scratch/cohan/jw3278/webshop_repo` |
| Full corpus (5.5 GB) | `.../webshop_repo/data/items_shuffle.json` |
| Full BM25 index (1.18M docs) | `.../webshop_repo/search_engine/indexes_full` |
| WebShop conda env (python 3.8) | `.../envs/memsys-webshop` |
| memsys conda env (python 3.11) | `.../envs/memsys-alfworld` |
| vLLM serving env | `.../verl_qwen35_train` |
| Run outputs | `.../memsys_results/webshop` |

### Two interpreters, on purpose

WebShop is pinned to a 2022 dependency set — python 3.8.13, `torch 1.11`,
`transformers 4.19`, `flask 2.1.2`, `spacy 3.3`, `typing_extensions<4.6` — and
memsys needs a current `sentence-transformers` and `openai`. They cannot share an
interpreter, so the environment runs behind an HTTP boundary
(`scripts/webshop_server.py`) and the adapter is a client.

That split pays for itself twice over. The catalogue costs ~2 minutes and 14 GiB
to load; with an env per worker a ten-arm sweep would pay that dozens of times.
Here every agent in every arm shares one `SimServer`, and a per-agent session is
a `SimBrowser` — a few hundred bytes.

## 1. Serve the model and the store

```bash
salloc -p gpu --gres=gpu:2 -c 16 --mem=256G -t 24:00:00

bash scripts/serve_qwen.sh 0 8000 --background   # GPU 0
bash scripts/serve_qwen.sh 1 8001 --background   # GPU 1
bash scripts/serve_webshop.sh 2 full             # env servers on :7000, :7001
```

`--mem`: the corpus peaks at **17.6 GiB** while parsing (5.5 GB of JSON — the raw
bytes, the decoded text and the object graph are all live at once) and settles at
**14 GiB** per server. Two servers plus vLLM and the arm drivers fit in 128 GiB;
256 GiB is headroom. This is a fixed cost of the corpus and does not scale with
the number of tasks.

`-c 16`: WebShop is far more CPU-hungry per step than ALFWorld — every action
runs a Lucene query through pyjnius, renders a Jinja template and parses the
result with BeautifulSoup, where ALFWorld does a PDDL state transition.

Two env servers rather than one with more threads: each serialises env access
behind a single lock (see §5). Scaling out is the safe axis and costs only
14 GiB per process, since the catalogue is read-only.

## 2. Build the index and the manifests

The full corpus is a one-time build (~7 minutes total: 2 min convert, 4 min
index):

```bash
bash scripts/webshop_build_index.sh full     # -> resources_full/, indexes_full/
```

Then, against a running server:

```bash
python scripts/build_webshop_manifests.py --server http://localhost:7000 \
  --out manifests --evolve-count 50 --eval-count 100 --seed 42
```

Committed as `manifests/webshop_evolve_train_50_seed42.json` and
`manifests/webshop_eval_test_100_seed42.json`. Once a run starts, never
regenerate under the same filename.

Three properties are deliberate:

- **Split by index range.** WebShop has no named splits; its ecosystem partitions
  the shuffled goal list by position — `[0, 500)` test, `[500, 1500)` validation,
  `[1500, ...)` train. Evolving draws from train and evaluation from test, so the
  sets are disjoint by construction, as ALFWorld's `train` / `valid_unseen` are.
- **Nested prefixes.** Selection is a prefix of one seeded permutation of the
  split's range, not `random.sample(k)`, so "evolve on 50 / 100 / 150" compares
  *amount of experience* rather than unrelated task draws.
- **Resolved instructions.** Each entry carries the customer instruction.
  Retrieval must happen before the agent acts, but WebShop only reveals the
  instruction on `reset()`.

**A manifest is only valid for the server that produced it.** A WebShop task is
an integer index into a shuffled goal list, and what that integer *means* depends
on the corpus, on `human_goals`, and on the RNG seed. The manifest records the
server's `/health`, and `run_webshop.py` refuses to start against a server that
does not match. See §5 for why this is not paranoia.

## 3. Record the demonstrations

```bash
python scripts/webshop_make_demos.py --server http://localhost:7000 \
  --out memsys/adapters/webshop_examples.json
```

The actions and thoughts are hand-written; the observations are replayed out of
the live store and captured verbatim. A demonstration whose pages were written by
hand teaches a page format the store never emits, and that shows up as an agent
clicking buttons which do not exist. Both trajectories are asserted to end at
reward 1.0, which doubles as an end-to-end check that corpus, index and reward
function still agree.

**Scaffold provenance — read this before comparing to published numbers.** The
ALFWorld prompt is byte-identical to SAGE's, which pins that baseline to an
independently measured 58–60%. Nothing equivalent exists here: the WebShop system
prompt and demonstrations were written for this study. So the absolute number is
**not** comparable to published WebShop results, and the `none` arm is the only
reference that means anything. Every arm shares the scaffold exactly; only the
memory block differs.

## 4. Run

```bash
bash scripts/run_webshop_sweep.sh smoke      # 2 evolve / 4 eval, all arms
bash scripts/run_webshop_sweep.sh minimal    # 50/100, WritePolicy.minimal()
bash scripts/run_webshop_sweep.sh full       # 50/100, WritePolicy.full()
```

Arms: `none`, `raw`, `reflection`, `rule`, `skill`. `none` is the no-memory
reference — without it no delta is interpretable.

One arm alone:

```bash
python scripts/run_webshop.py --arm rule --policy full \
  --evolve-manifest manifests/webshop_evolve_train_50_seed42.json \
  --eval-manifest   manifests/webshop_eval_test_100_seed42.json \
  --server http://localhost:7000 --server http://localhost:7001 \
  --out $MEMSYS_RESULTS_ROOT/webshop/full/rule_full --agent-base-url http://localhost:8000/v1
```

### Parallelism

- **Evolving is strictly sequential.** Memory written after episode *i* is
  retrieved by episode *i+1*; parallelising it would measure a different system.
- **Evaluation runs under `frozen()`** — every write raises — so tasks are
  independent and fan out. Unlike ALFWorld this uses **threads, not processes**:
  the environment is not in this process at all, so a worker only makes HTTP
  calls to the env server and to vLLM, both IO-bound.
- **Across arms**, two at a time, one per vLLM server.

## 5. Failures that cost time here

- **WebShop's goals are not deterministic across processes, and nothing says so.**
  `SimServer` seeds `random` only just before shuffling the goal list
  (`random.seed(233)`) — *after* two draws that decide what the tasks are.
  `generate_product_prices` gives every product a `random.uniform` price, and
  `get_human_goals` samples each goal's "price lower than X dollars" clause from
  those prices. Unseeded, two server processes disagree about both the
  instruction text and the price component of the reward for the same goal index.
  Every arm still runs, still reports a success rate, and is quietly solving
  different tasks. `webshop_server.py --seed` seeds before construction; every
  server in a run must pass the same value, and the manifest records it.
- **`num_products=None` does not mean "the full corpus", it means "the directory
  called `indexes`".** Upstream derives the index path from `num_products` alone,
  so a checkout whose `indexes/` was built from `items_shuffle_1000.json` searches
  1,000 documents while the catalogue holds 1.18M — every search returning
  products that are not findable, with no error anywhere. This repo names corpora
  and indexes in pairs (`WEBSHOP_FILE_PATH` / `WEBSHOP_ATTR_PATH` /
  `WEBSHOP_INDEX_DIR`, selected together by `--scale`) and asserts the index
  directory exists.
- **`items_shuffle_1000.json` has only 13 human goals.** The 1,000-product subset
  keeps whichever products came first, and only 13 of them carry human-written
  instructions. Anything at that scale must fall back to synthetic templated
  goals, which is a different task distribution from every published WebShop
  result. The full corpus yields the canonical **12,087** human goals.
- **pyserini needs `JAVA_HOME`, and conda hides the JDK.** Importing WebShop pulls
  in pyserini, a JVM wrapper; pyjnius looks for `JAVA_HOME` and then for `javac`
  on `PATH`, and with neither it dies as `Exception: Unable to find javac` from
  inside an import chain that never mentions Java. The env has openjdk 11 but
  conda installs it under `$PREFIX/lib/jvm`, so deriving `JAVA_HOME` from
  `sys.prefix` also fails. `webshop_server.py` sets both `JAVA_HOME` and
  `JVM_PATH` from `sys.prefix/lib/jvm`.
- **Session ids are the task identity, and they collide.** `SimServer` keys
  `user_sessions` by session id and *reuses the goal already attached to one*.
  The obvious reading of upstream's `reset(session=int)` — use the goal index as
  the id — makes two concurrent workers collide the moment their indices match,
  silently giving them the same task. The server prefixes every session with a
  per-client token while still passing `session_int` to select the goal.
- **An invalid action is invisible.** `WebAgentTextEnv.step` falls through to
  `status = dict(reward=0, done=False)` for an unparseable action or an unknown
  clickable and returns the *unchanged page*, which reads to the agent exactly
  like an action that legitimately changed nothing. The server recomputes
  availability before stepping and reports `invalid`, and the adapter prefixes the
  observation with "Invalid action." `mean_invalid_actions` in `summary.json` is
  the health metric — if it climbs, suspect the action format before the memory.
- **`text` observation mode erases what is clickable.** It flattens the page to
  ` [SEP] `-joined strings, so a product title and a button look identical and
  option selection — which the reward grades directly — becomes guesswork. The
  adapter appends one `Available actions:` line per turn. It is the only thing
  the adapter adds to what WebShop emits, and it is added identically for every
  arm.
- **Do not use `pkill -f` on the server scripts.** The pattern matches the shell
  you type it in. Use `bash scripts/serve_webshop.sh stop`, which kills recorded
  PIDs.

## 6. Read the results

```bash
python scripts/summarize.py --root /gpfs/radev/scratch/cohan/jw3278/memsys_results/webshop/full
```

Writes `RESULTS.md` with success rate, **score**, delta vs the `none` baseline,
b/c discordant counts and a McNemar exact p-value, store size, injected tokens and
writer cost.

**WebShop reports two numbers and both are needed.** `get_reward` grades
attribute, option, type and price matches, so a rollout can score 0.67 by buying
a nearly-right product. *Score* is the mean graded reward; *success rate* is the
fraction scoring exactly 1.0. They can move in opposite directions — an arm that
buys plausible-but-wrong items faster raises score and lowers success rate — so
reporting only one hides that effect. The paired McNemar test is computed on the
strict outcome.

Report the paired test, not the raw delta. Every arm evaluates the identical
ordered task list, so the comparison is paired.

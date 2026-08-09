# memsys

Implementation of the three memory systems in `memory_strategy_design.md` —
**Reflection**, **Rule**, **Procedural Skill** — plus the **Raw Trajectory** baseline
and the `"all"` composite.

No benchmark integration yet: the systems consume `Episode` objects from anywhere.

```
pip install -e .        # optional; the core has zero dependencies
python -m examples.demo
python -m unittest discover -s tests
```

## Layout

| file | contents |
|---|---|
| `memsys/config.py` | `WritePolicy` + every knob; each experiment is a diff against this |
| `memsys/episode.py` | `Step` / `Rollout` / `Episode`, outcome bucketing, evidence guard |
| `memsys/schemas.py` | the four content schemas: validation, retrieval key, injection rendering |
| `memsys/item.py` | the shared `MemoryItem` envelope (provenance + usage stats) |
| `memsys/store.py` | scoped retrieval, token-budget packing, dedup lookup, eviction, persistence |
| `memsys/embedding.py` | `HashingEmbedder` (no deps) / `SentenceTransformerEmbedder` |
| `memsys/llm.py` | writer-model clients + token accounting |
| `memsys/writers.py` | prompt construction, JSON parsing, per-type validation |
| `memsys/systems.py` | the four systems + composite; write policy on top of the store |
| `memsys/runner.py` | `Evolver`, the design-doc §7 JSONL log, `frozen()` |
| `memsys/stub_llm.py` | offline stand-in writer for demos/CI — **never for paper numbers** |

## Usage

```python
from memsys import (build_system, Episode, Rollout, Step, Evolver, MemoryConfig,
                    WritePolicy, frozen)
from memsys.llm import OpenAIChatClient

llm    = OpenAIChatClient("Qwen3.5-9B", base_url="http://localhost:8000/v1")
config = MemoryConfig(injection_budget_tokens=1500, max_items=200,
                      policy=WritePolicy.full())        # same mechanisms for every type
system = build_system("rule", llm=llm, config=config)   # reflection | rule | skill | raw | all

# --- evolving phase ---
ev = Evolver(system, config=config)
for task in evolve_tasks:
    ret      = ev.retrieve(task.instruction, scope={"env": "webshop"})
    rollouts = agent.run(task, memory_block=ret.block)      # your agent
    ev.step_once(Episode(task.id, task.instruction, rollouts, scope={"env": "webshop"}), ret)

# --- evaluation phase: writes raise, retrieval still works ---
with frozen(system):
    for task in test_tasks:
        block = system.retrieve(task.instruction, {"env": "webshop"}).block
        agent.run(task, memory_block=block)

system.store.save("rule_store.jsonl")
```

The only thing a benchmark adapter must produce is `Episode` objects.
`ret.block` is the string to paste into the agent prompt.

## Content and mechanism are independent axes

The memory types differ **only** in their content schema and writer prompt:

- **Reflection** (§2) — `situation / lesson / rationale / evidence / outcome_tag`.
  The extraction mode follows the episode outcome: `from_success` / `from_failure` /
  `from_contrast`.
- **Rule** (§3) — `trigger / directive / polarity / exception / evidence`. One
  directive per rule; the trigger must be decidable from the current observation.
- **Skill** (§4) — `name / trigger / preconditions / steps / verification / fallback /
  evidence`. 3–12 steps, placeholders instead of instance names.
- **Raw** (§1) — append-only, successes only, no LLM. The baseline.

Every *mechanism* lives in `WritePolicy` and applies to all types alike:

| flag | what it does |
|---|---|
| `online_write` | per-episode extraction from the current rollouts |
| `merge_on_duplicate` | LLM merge on a near-duplicate, else drop it |
| `grounding_check` | `evidence` must occur in the real trajectory |
| `verify` | judge the injected entries each episode → `support` / `refute` |
| `refine` | rewrite an entry once counterevidence accumulates |
| `delete_on_low_confidence` | drop entries whose confidence collapses |
| `utility_deletion` | drop entries retrieved often but rarely alongside success |
| `batch_induction` | cross-task consolidation every `batch_every` episodes |
| `n_max` | entries the writer may append per episode |

```python
MemoryConfig(policy=WritePolicy.full())      # everything on, every type  <- content table
MemoryConfig(policy=WritePolicy.minimal())   # append/merge only, every type
MemoryConfig().native()                      # the old per-type asymmetry <- ablation only
```

This matters: the first version of the design attached a *different* mechanism to each
type (rule alone verified, skill alone induced across tasks, reflection did neither),
so the three differed in two ways at once and a skill win could not be attributed to
procedural content rather than to its writer seeing 25 episodes at a time. Hold the
policy fixed for the Memory Content table; vary it for a separate write-mechanism
ablation. `python -m examples.demo` prints the mechanism × type matrix for all three
presets.

Two asymmetries are irreducible and should be stated in the paper rather than
engineered away: skill cannot write anything from an all-failure episode (a procedure
needs one working path), and the raw baseline has no LLM writer, so it cannot verify
or induce at all.

## Fairness controls worth keeping an eye on

- **Equal write mechanism.** `WritePolicy` above. `MechanismFairnessTest` asserts that
  under a uniform policy all three types fire the same set of mechanisms and get the
  same `n_max` and `batch_every`, and that all three writers see identical rollouts.
- **Equal token budget.** `pack()` fills to `injection_budget_tokens` by score and
  drops what doesn't fit — it never truncates an item. `BudgetFairnessTest` asserts
  every system, including raw, obeys the same `B`. Set `equal_item_count=3` for the
  equal-#items ablation in the appendix.
- **Writer cost.** Rule's verification loop and the composite's 3× writes are real
  compute differences. `llm.usage` and `CompositeSystem.writer_usage()` expose them;
  the per-step JSONL log records them so the cost table needs no re-run.
- **Frozen evaluation.** `frozen()` makes `observe()` raise rather than relying on
  discipline.
- **Order dependence.** Retrieval is on during evolving (§5.3), so results depend on
  task order. Fix the order and run ≥3 seeds.

## Swapping in real backends

```python
from memsys import MemoryStore, SentenceTransformerEmbedder, MemoryConfig
store = MemoryStore(embedder=SentenceTransformerEmbedder("BAAI/bge-m3"), config=config)
system = build_system("skill", llm=llm, config=config, store=store)
```

`HashingEmbedder` is the default so the package runs anywhere; it is fine for dedup
thresholds and tests but should not be used for the reported experiments.

## Not implemented yet

Benchmark adapters (ALFWorld / WebShop / AppWorld), the agent loop itself, the
difficulty-filtering and repeat-evolving dataset builders from §6, and failing-step
attribution for skills (currently read from `episode.meta`, not inferred).

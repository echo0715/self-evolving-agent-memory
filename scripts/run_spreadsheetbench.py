#!/usr/bin/env python
"""Run one memsys arm end to end on SpreadsheetBench: evolve, then frozen evaluation.

    python scripts/run_spreadsheetbench.py --arm rule --policy full \
        --evolve-manifest manifests/spreadsheetbench_evolve_train_50_seed42.json \
        --eval-manifest   manifests/spreadsheetbench_eval_test_100_seed42.json \
        --data-root $SPREADSHEETBENCH_ROOT/all_data_912_v0.1 \
        --out $MEMSYS_RESULTS_ROOT/spreadsheetbench/full/rule_full

Same two-phase shape as the other three runners: evolving is strictly sequential
because memory written after episode *i* is retrieved by episode *i+1*, and
evaluation runs under `frozen()` and fans out.

The structural difference is that there is **no environment server**. An episode
is a temp directory plus subprocesses, so evaluation concurrency is bounded by
this machine's CPUs and by how many concurrent requests the vLLM server behind
`--agent-base-url` will take, not by a lease pool. `--eval-workers` is therefore
a free knob, and the default of 8 is chosen against the shared server rather
than against anything in the benchmark.

Two costs to keep in view when setting it. Each worker runs the agent's bash
commands and, at scoring time, up to two more `python solution.py` subprocesses,
all of which load workbooks into memory -- a few tasks have answer ranges of
~100k cells. And `run_generated_code` executes model-written code with
`sys.executable`, so `--eval-workers` is also the number of untrusted
subprocesses that may be running at once.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memsys.schemas import set_max_evidence_tokens  # noqa: E402
from memsys import (  # noqa: E402
    Episode,
    Evolver,
    MemoryConfig,
    MemoryStore,
    OpenAIChatClient,
    OpenAIResponsesClient,
    RunLogger,
    WritePolicy,
    build_system,
    frozen,
)
from memsys.adapters.spreadsheetbench import (  # noqa: E402
    AgentConfig,
    SpreadsheetAgent,
    default_data_root,
    load_manifest,
    run_task,
)

ARMS = ("none", "raw", "reflection", "rule", "skill", "all")


def build_config(args) -> MemoryConfig:
    policy = WritePolicy.full() if args.policy == "full" else WritePolicy.minimal()
    policy.n_max = args.n_max
    policy.batch_every = args.batch_every
    return MemoryConfig(
        policy=policy,
        # 2500, as on AppWorld rather than ALFWorld/WebShop's 1500: this scaffold
        # is also a code-writing one, its trajectories are Python and shell
        # output, and a 1500-token cap would make the block a rounding error
        # against the rest of the prompt.
        injection_budget_tokens=args.injection_budget,
        max_items=args.max_items,
        seed=args.seed,
    )


def build_store(args, config: MemoryConfig) -> MemoryStore:
    if args.embedder == "hashing":
        from memsys import HashingEmbedder

        return MemoryStore(embedder=HashingEmbedder(), config=config)
    from memsys import SentenceTransformerEmbedder

    # device="cpu": both GPUs are held by vLLM at 0.90 memory utilisation.
    return MemoryStore(
        embedder=SentenceTransformerEmbedder(args.embedding_model, device=args.embedding_device),
        config=config,
    )


def _dotenv(name: str) -> str | None:
    """Read one key from the repo's .env, which is gitignored and holds the
    gateway credential. Real environment variables win over the file."""
    if os.environ.get(name):
        return os.environ[name]
    path = Path(__file__).resolve().parent.parent / ".env"
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        if k.strip() == name:
            return v.strip().strip("'\"")
    return None


def build_writer_llm(args, writer_url: str):
    """The memory writer's client, deliberately independent of the agent's:
    `--writer-model` is the independent variable of the Memory Writing Model
    study, while the actor stays on `--model` (the local Qwen) so any delta is
    attributable to what was written rather than to who acted. Same contract as
    `run_alfworld.py:build_writer_llm`, with one benchmark-specific difference —
    `--writer-json-mode` is passed on the chat path, because Qwen3.5-9B loses its
    string escaping partway through the quoted Python and shell output this
    benchmark's evidence is made of. The Responses path has no such option: the
    gateway rejects `text.format={"type":"json_object"}`, so the writers' own
    JSON repair is what keeps parse rates up there."""
    model = args.writer_model or args.model
    key = "EMPTY"
    if args.writer_api_key_env:
        key = _dotenv(args.writer_api_key_env)
        if not key:
            # Failing here is the point: an unauthenticated writer 401s on every
            # call, and `parse_ops` turns that into an empty store rather than an
            # error, so the arm would finish and report a plausible number.
            raise SystemExit(
                f"--writer-api-key-env {args.writer_api_key_env!r} is set but no such key "
                f"is in the environment or .env")
    if args.writer_api == "responses":
        return OpenAIResponsesClient(
            model, base_url=writer_url, api_key=key,
            temperature=0.0, max_tokens=args.writer_max_tokens, timeout=args.timeout,
            reasoning_effort=args.writer_reasoning_effort,
        )
    return OpenAIChatClient(
        model, base_url=writer_url, api_key=key,
        temperature=0.0, max_tokens=args.writer_max_tokens, timeout=args.timeout,
        json_mode=args.writer_json_mode,
    )


def make_agent(args) -> SpreadsheetAgent:
    from openai import OpenAI

    client = OpenAI(base_url=args.agent_base_url, api_key="EMPTY",
                    timeout=args.timeout, max_retries=3)
    return SpreadsheetAgent(
        client=client,
        model=args.model,
        config=AgentConfig(
            max_turns=args.max_turns,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            bash_timeout=args.bash_timeout,
        ),
    )


def episode_row(ep: Episode, seconds: float) -> dict:
    return {
        "task_id": ep.task_id,
        "instruction": ep.instruction,
        "task_family": ep.scope.get("task_type", ""),
        "outcome": ep.outcome(),
        "success_rate": ep.success_rate,
        # Strict: every one of the task's test cases passes.
        "any_success": ep.any_success,
        # Graded: fraction of test cases passed.
        "score": max((r.reward for r in ep.rollouts), default=0.0),
        "rollouts": [
            {
                "rollout_id": r.rollout_id,
                "success": r.success,
                "reward": r.reward,
                "n_steps": len(r.steps),
                "error": r.error,
                **{k: r.meta.get(k) for k in
                   ("n_cases", "n_pass", "n_exec_ok", "cell_match_rate", "fail_reason",
                    "wrote_solution", "parse_failures", "bash_errors", "prompt_tokens",
                    "completion_tokens", "injected_memory_tokens")},
            }
            for r in ep.rollouts
        ],
        "seconds": round(seconds, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--policy", choices=("minimal", "full"), default="full")
    ap.add_argument("--evolve-manifest")
    ap.add_argument("--eval-manifest", required=True)
    ap.add_argument("--data-root", default=default_data_root())
    ap.add_argument("--out", required=True)

    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--agent-base-url", default="http://localhost:8000/v1")
    ap.add_argument("--writer-base-url", default=None,
                    help="defaults to --agent-base-url; point elsewhere to split GPUs "
                         "or to reach a remote writer gateway")
    ap.add_argument("--writer-model", default=None,
                    help="memory-writing model; defaults to --model (the agent's). Set this "
                         "to vary the writer independently of the actor, which is the point "
                         "of the Memory Writing Model study")
    ap.add_argument("--writer-api", choices=("chat", "responses"), default="chat",
                    help="wire protocol for the writer endpoint. vLLM speaks 'chat'; the "
                         "Perplexity gateway serving openai/gpt-5.6-* speaks only 'responses'")
    ap.add_argument("--writer-api-key-env", default=None,
                    help="environment variable holding the writer API key (also read from "
                         "the repo .env). Local vLLM needs none")
    ap.add_argument("--writer-reasoning-effort", default=None,
                    help="reasoning effort for a 'responses' writer, e.g. low/medium/high")
    ap.add_argument("--timeout", type=float, default=300.0)

    ap.add_argument("--n-rollouts", type=int, default=1)
    ap.add_argument("--max-turns", type=int, default=30)
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--bash-timeout", type=float, default=120.0)
    ap.add_argument("--exec-timeout", type=float, default=180.0,
                    help="per-test-case timeout when re-running solution.py at scoring time")
    ap.add_argument("--eval-workers", type=int, default=8)
    ap.add_argument("--writer-max-tokens", type=int, default=1024)
    ap.add_argument("--writer-json-mode", action=argparse.BooleanOptionalAction, default=True,
                    help="constrain writer output to valid JSON via the server's guided decoding")
    ap.add_argument("--max-evidence-tokens", type=int, default=300,
                    help="grounding-evidence cap; 80 (the default elsewhere) rejects most "
                         "writes here because the evidence is Python and shell output, not prose")

    ap.add_argument("--injection-budget", type=int, default=2500)
    ap.add_argument("--max-items", type=int, default=300)
    ap.add_argument("--n-max", type=int, default=2)
    ap.add_argument("--batch-every", type=int, default=25)
    ap.add_argument("--embedder", choices=("hashing", "st"), default="st")
    ap.add_argument("--embedding-model", default="BAAI/bge-large-en-v1.5")
    ap.add_argument("--embedding-device", default="cpu")

    ap.add_argument("--evolve-limit", type=int, default=0)
    ap.add_argument("--eval-limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    # The outcome-budget axis, identical in flag names and semantics to
    # scripts/run_alfworld.py so the two benchmarks' fail100/succ100 legs are the
    # same experiment. Failures are the cheap outcome here: SpreadsheetBench's
    # evolve-time success rate is 12-32%, so 100 failures costs ~125-145 tasks
    # (against ALFWorld's ~190, where the rate runs the other way).
    ap.add_argument("--success-only-writes", action="store_true",
                    help="discard failed episodes entirely: the memory system never "
                         "observes them, so they drive no extraction, verification, "
                         "refinement, pruning or batch induction")
    ap.add_argument("--evolve-until-successes", type=int, default=0,
                    help="stop evolving once this many episodes have succeeded, "
                         "instead of after a fixed number of tasks (0 = off)")
    ap.add_argument("--failure-only-writes", action="store_true",
                    help="the mirror of --success-only-writes: discard SUCCESSFUL episodes, "
                         "so the memory is built from nothing but what went wrong. Note that "
                         "`raw` then writes nothing at all (it keeps best_success()) and ends "
                         "up identical to the `none` arm")
    ap.add_argument("--evolve-until-failures", type=int, default=0,
                    help="stop evolving once this many episodes have FAILED (0 = off)")
    ap.add_argument("--resume-store", default=None,
                    help="load this store.jsonl before evolving, to continue a previous "
                         "run instead of starting from an empty memory")
    ap.add_argument("--evolve-step-offset", type=int, default=0,
                    help="step index the evolving loop starts from; set to the number of "
                         "episodes the resumed store already saw")
    args = ap.parse_args()

    # The two filters and the two budgets are mirrors of each other, so asking
    # for both is always a mistake -- and a silent one: "success only" wins by
    # ordering in the loop below and the run would look like a success-budget
    # run under a failure-budget name.
    if args.success_only_writes and args.failure_only_writes:
        raise SystemExit("--success-only-writes and --failure-only-writes are mutually exclusive")
    if args.evolve_until_successes and args.evolve_until_failures:
        raise SystemExit("--evolve-until-successes and --evolve-until-failures are mutually exclusive")
    if args.failure_only_writes and args.arm == "raw":
        # Not fatal: an empty-store `raw` run is a legitimate (if uninformative)
        # data point, and refusing would hide that the filter is a no-op here.
        print("[warn] arm=raw with --failure-only-writes: RawTrajectorySystem keeps only "
              "best_success(), so nothing will ever be written and this run is the `none` "
              "baseline with extra steps", flush=True)

    data_root = Path(args.data_root).expanduser().resolve()
    if not (data_root / "dataset.json").is_file():
        raise SystemExit(f"no dataset.json under --data-root {data_root}; "
                         f"run scripts/setup_spreadsheetbench.sh")
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        raise SystemExit("openpyxl is not installed in this interpreter -- the evaluator and "
                         "the agent's own solution.py both need it; see "
                         "scripts/setup_spreadsheetbench.sh")

    # Applied before any system is built, and identically for every arm: a
    # benchmark-calibration constant, not an arm-level knob. Same reason as
    # AppWorld -- the writer quotes code into `evidence` and the 80-token default
    # rejects it, hardest on the types that write least often.
    set_max_evidence_tokens(args.max_evidence_tokens)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    config = build_config(args)
    writer_url = args.writer_base_url or args.agent_base_url

    system = None
    if args.arm != "none":
        # `--writer-json-mode` is applied inside build_writer_llm on the chat
        # path. As on AppWorld it is not a nicety: the writer is asked to quote
        # the trajectory verbatim into `evidence`, the trajectory here is Python
        # and shell output full of quotes and backslashes, and Qwen3.5-9B loses
        # the escaping partway through the string. The response then parses to
        # nothing and the arm ends with an empty store -- silently identical to
        # `none`. See memsys/llm.py.
        llm = build_writer_llm(args, writer_url)
        system = build_system(args.arm, llm=llm, config=config, store=build_store(args, config))
        if args.resume_store:
            # Continue a previous run's memory rather than starting empty. Item
            # stats (n_retrieved, support/refute, created_at_step) round-trip
            # through store.jsonl; embeddings are not serialised but `retrieve`
            # re-encodes lazily, so this is exact as long as the same embedder is
            # used. Two things do NOT resume: the batch-induction buffer, and the
            # Evolver's step counter -- hence --evolve-step-offset, without which
            # `full`'s every-25-episode induction cadence would restart at 0 and
            # fire at a different point than an uninterrupted 100-episode run.
            store = getattr(system, "store", None)
            if store is None:
                raise SystemExit(f"--resume-store given but arm {args.arm!r} has no store")
            store.load(args.resume_store)
            print(f"[resume] loaded {len(store)} live entries from {args.resume_store}", flush=True)

    agent = make_agent(args)

    (out / "config.json").write_text(json.dumps({
        "args": vars(args),
        "data_root": str(data_root),
        "max_evidence_tokens": args.max_evidence_tokens,
        "policy": asdict(config.policy),
        "memory_config": {k: v for k, v in config.to_dict().items() if k != "policy_by_type"},
    }, indent=2, default=str) + "\n", encoding="utf-8")

    # ------------------------------------------------------------- evolving
    evolve_rows = []
    if system is not None and args.evolve_manifest:
        tasks = load_manifest(args.evolve_manifest)
        if args.evolve_limit:
            tasks = tasks[: args.evolve_limit]
        logger = RunLogger(str(out / "evolve_log.jsonl"))
        ev = Evolver(system, config=config, logger=logger)
        ev.step = args.evolve_step_offset
        n_success = 0
        n_fail = 0
        with (out / "evolve_episodes.jsonl").open("w", encoding="utf-8") as fh:
            for i, task in enumerate(tasks):
                t0 = time.time()
                ret = ev.retrieve(task.instruction, scope=task.scope())
                ep = run_task(agent, task, data_root, out / "evolve_preds",
                              memory_block=ret.block, n_rollouts=args.n_rollouts,
                              exec_timeout=args.exec_timeout)
                # --failure-only-writes drops the successful episode entirely
                # rather than letting the writer see it: no extraction, and also
                # no verify/refine/prune and no batch-induction buffering, all of
                # which otherwise take a success as evidence *for* whatever was
                # injected. The store is then built out of failure evidence
                # alone, and under `full` every utility signal reaching the
                # deletion machinery is negative. --success-only-writes is the
                # same filter with the sign flipped. `raw` writes only on success
                # by construction (RawTrajectorySystem.observe keeps
                # best_success()), so failure-only leaves it with an empty store.
                #
                # Note this is `any_success`, the strict criterion -- all of the
                # task's test cases pass -- so a partially-correct episode counts
                # as a failure and is written. That is the same threshold
                # `evolve_outcomes` and the headline eval number use.
                if args.success_only_writes:
                    written = bool(ep.any_success)
                elif args.failure_only_writes:
                    written = not ep.any_success
                else:
                    written = True
                if written:
                    # step_once advances ev.step, so with a filter on, the step
                    # counter -- and with it `full`'s every-25-episode batch
                    # induction cadence -- counts *written* episodes, not tasks
                    # attempted. That is what makes "100 failures" the unit.
                    row_step = ev.step
                    ev.step_once(ep, ret)
                else:
                    row_step = None
                row = episode_row(ep, time.time() - t0)
                row["step"] = row_step
                row["written"] = written
                row["task_index"] = i
                evolve_rows.append(row)
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                n_success += int(bool(ep.any_success))
                n_fail += int(not ep.any_success)
                store = getattr(system, "store", None)
                # Checkpoint the store every episode. Without this a crash --
                # a dead vLLM, a reclaimed allocation -- loses every episode of
                # the leg, because the only save used to be after the loop. The
                # write is a few hundred KB against a step that costs tens of
                # seconds, and the post-loop save below still has the last word,
                # so this only ever adds a recoverable artefact.
                if store is not None:
                    store.save(str(out / "store.jsonl"))
                print(f"[evolve {i+1}/{len(tasks)}] {row['outcome']:<12} "
                      f"score={row['score']:.2f} "
                      f"{'W' if written else '.'} succ={n_success} fail={n_fail} "
                      f"store={len(store) if store is not None else '-'} "
                      f"{row['seconds']:.0f}s {task.task_id}", flush=True)
                # Budget the run by outcome rather than by attempts: arms differ
                # by up to 20 points in evolve-time success rate, so a fixed task
                # count hands them different amounts of the evidence they are
                # allowed to write from.
                if args.evolve_until_successes and n_success >= args.evolve_until_successes:
                    print(f"[evolve] reached {n_success} successes after {i+1} tasks", flush=True)
                    break
                if args.evolve_until_failures and n_fail >= args.evolve_until_failures:
                    print(f"[evolve] reached {n_fail} failures after {i+1} tasks", flush=True)
                    break
            else:
                # Running out of manifest short of the target is a failed run,
                # not a shorter one -- it would be compared against arms that did
                # reach it -- so say so rather than writing a summary that looks
                # complete.
                if args.evolve_until_successes:
                    print(f"[evolve] WARNING: manifest exhausted at {n_success} successes "
                          f"(<{args.evolve_until_successes}) after {len(tasks)} tasks", flush=True)
                if args.evolve_until_failures:
                    print(f"[evolve] WARNING: manifest exhausted at {n_fail} failures "
                          f"(<{args.evolve_until_failures}) after {len(tasks)} tasks", flush=True)
        ev.flush()
        if getattr(system, "store", None) is not None:
            system.store.save(str(out / "store.jsonl"))

    # ----------------------------------------------------------- evaluation
    eval_tasks = load_manifest(args.eval_manifest)
    if args.eval_limit:
        eval_tasks = eval_tasks[: args.eval_limit]

    # Retrieve every block up front, inside `frozen()`, then fan out: retrieval
    # touches the shared store and the evaluation workers must not.
    ctx = frozen(system) if system is not None else _nullcontext()
    with ctx:
        payloads = []
        for task in eval_tasks:
            block = ""
            if system is not None:
                block = system.retrieve(task.instruction, scope=task.scope()).block
            payloads.append((task, block))

    def _run_one(payload):
        task, block = payload
        t0 = time.time()
        ep = run_task(agent, task, data_root, out / "eval_preds", memory_block=block,
                      n_rollouts=1, exec_timeout=args.exec_timeout)
        return episode_row(ep, time.time() - t0)

    fh = (out / "eval.jsonl").open("w", encoding="utf-8")
    eval_rows = []
    with ThreadPoolExecutor(max_workers=max(1, args.eval_workers)) as tp:
        futures = [tp.submit(_run_one, p) for p in payloads]
        for fut in as_completed(futures):
            row = fut.result()
            eval_rows.append(row)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"[eval {len(eval_rows)}/{len(eval_tasks)}] "
                  f"{'OK  ' if row['any_success'] else 'FAIL'} score={row['score']:.2f} "
                  f"{row['seconds']:.0f}s {row['task_id']}", flush=True)
    fh.close()

    # -------------------------------------------------------------- summary
    n_ok = sum(1 for r in eval_rows if r["any_success"])
    summary = {
        "benchmark": "SpreadsheetBench",
        "arm": args.arm,
        "policy": args.policy,
        "model": args.model,
        # The actor and the writer are separate variables; recording only one of
        # them makes two runs that differ in the writer indistinguishable in the
        # summary. `none`/`raw` have no writer LLM at all, hence the empty string.
        "writer_model": getattr(
            getattr(getattr(system, "writer", None), "llm", None), "model", ""),
        "data_root": str(data_root),
        "eval_split": eval_tasks[0].split if eval_tasks else "",
        "eval_n": len(eval_rows),
        "eval_success": n_ok,
        "eval_success_rate": n_ok / len(eval_rows) if eval_rows else 0.0,
        "eval_score": _mean(r["score"] for r in eval_rows),
        # Did the agent produce a reusable solution.py at all? The SpreadsheetBench
        # analogue of WebShop's purchase rate: it separates "solved case 1 by hand
        # and could not generalise" from "never produced anything".
        "eval_wrote_solution_rate": _mean(
            1.0 if r["rollouts"][0].get("wrote_solution") else 0.0 for r in eval_rows
        ),
        # Diagnostic only, never a reported score -- see the adapter.
        "eval_cell_match_rate": _mean(
            r["rollouts"][0].get("cell_match_rate") or 0.0 for r in eval_rows
        ),
        "eval_by_family": _by_family(eval_rows),
        "eval_fail_reasons": _counts(
            _reason_bucket(r["rollouts"][0].get("fail_reason")) for r in eval_rows
            if not r["any_success"]
        ),
        # Tasks attempted. Under an outcome budget this is larger than the number
        # the memory actually saw; the two are separated on purpose so a "100
        # failures" run cannot be mistaken for a "100 episodes" one.
        "evolve_n": len(evolve_rows),
        "evolve_written": sum(1 for r in evolve_rows if r.get("written", True)),
        "evolve_successes": sum(1 for r in evolve_rows if r.get("any_success")),
        "evolve_failures": sum(1 for r in evolve_rows if not r.get("any_success")),
        "success_only_writes": args.success_only_writes,
        "failure_only_writes": args.failure_only_writes,
        "evolve_budget": (
            {"kind": "successes", "target": args.evolve_until_successes}
            if args.evolve_until_successes else
            {"kind": "failures", "target": args.evolve_until_failures}
            if args.evolve_until_failures else {"kind": "tasks", "target": len(evolve_rows)}
        ),
        # Episodes this store has now seen in total, across the resumed run and
        # this one. `evolve_n` alone would read as 50 on a 100-episode store.
        # Counts *written* episodes, so it stays the memory's own experience
        # count when the other outcome is discarded.
        "evolve_total": args.evolve_step_offset + sum(
            1 for r in evolve_rows if r.get("written", True)
        ),
        "resumed_from": args.resume_store,
        "evolve_success_rate": (
            sum(r["success_rate"] for r in evolve_rows) / len(evolve_rows) if evolve_rows else None
        ),
        "evolve_score": _mean(r["score"] for r in evolve_rows) if evolve_rows else None,
        "evolve_outcomes": _counts(r["outcome"] for r in evolve_rows),
        "store": system.store.summary() if getattr(system, "store", None) is not None else None,
        "writer_usage": (
            system.writer.llm.usage.to_dict()
            if getattr(system, "writer", None) is not None else None
        ),
        "mean_injected_tokens": _mean(
            r["rollouts"][0].get("injected_memory_tokens") or 0 for r in eval_rows
        ),
        "mean_steps": _mean(r["rollouts"][0].get("n_steps") or 0 for r in eval_rows),
        "mean_bash_errors": _mean(r["rollouts"][0].get("bash_errors") or 0 for r in eval_rows),
        "mean_prompt_tokens": _mean(r["rollouts"][0].get("prompt_tokens") or 0 for r in eval_rows),
        "wall_seconds": round(time.time() - t_start, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n",
                                      encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))


def _counts(values):
    out: dict[str, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def _mean(values):
    vals = list(values)
    return round(sum(vals) / len(vals), 3) if vals else 0.0


def _reason_bucket(reason: str | None) -> str:
    """Collapse a `fail_reason` to its kind; the detail is per-task noise."""
    text = str(reason or "unknown")
    return text.split(":", 1)[0].strip() or "unknown"


def _by_family(rows):
    """Grouped by the scope key -- only `cell_level` / `sheet_level` exist.

    Unlike AppWorld this is coarse enough to read directly, and it is the one
    grouping the memory systems actually cluster on, so it is worth reporting as
    the scope key rather than as something else.
    """
    agg: dict[str, list[tuple[int, float]]] = {}
    for r in rows:
        agg.setdefault(r.get("task_family") or "unknown", []).append(
            (1 if r["any_success"] else 0, r["score"])
        )
    return {
        k: {
            "n": len(v),
            "success": sum(s for s, _ in v),
            "rate": round(sum(s for s, _ in v) / len(v), 3),
            "score": round(sum(sc for _, sc in v) / len(v), 3),
        }
        for k, v in sorted(agg.items())
    }


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *_):
        return False


if __name__ == "__main__":
    main()

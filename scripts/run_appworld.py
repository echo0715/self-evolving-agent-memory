#!/usr/bin/env python
"""Run one memsys arm end to end on AppWorld: evolve, then frozen evaluation.

    python scripts/run_appworld.py --arm rule --policy full \
        --evolve-manifest manifests/appworld_evolve_train_50_seed42.json \
        --eval-manifest   manifests/appworld_eval_test_normal_100_seed42.json \
        --server http://localhost:9000 --server http://localhost:9001 \
        --out $MEMSYS_RESULTS_ROOT/appworld/full/rule_full

Same two-phase shape as `run_alfworld.py` and `run_webshop.py`: evolving is
strictly sequential because memory written after episode *i* is retrieved by
episode *i+1*; evaluation runs under `frozen()` and fans out.

The one structural difference is the server pool. `appworld serve environment`
keeps a **single module-level `world`**, so a server process hosts exactly one
live task -- two concurrent episodes on the same URL would silently execute each
other's code against the wrong world state. Workers therefore *lease* a URL for a
whole episode and return it afterwards, and evaluation concurrency is capped at
the number of servers rather than set freely.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
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
from memsys.adapters.appworld import (  # noqa: E402
    AgentConfig,
    AppWorldAgent,
    AppWorldEnvironment,
    AppWorldError,
    TaskSpec,
    fetch_app_descriptions,
    load_manifest,
    run_task,
)

ARMS = ("none", "raw", "reflection", "rule", "skill", "all")


class ServerPool:
    """Hands out AppWorld server URLs under an exclusive lease.

    A plain round-robin -- which is what the WebShop runner uses -- would be
    wrong here: WebShop's server multiplexes sessions, AppWorld's holds one
    world. Handing the same URL to two workers makes the second `/initialize`
    replace the first worker's world underneath it, and the symptom is not an
    error but an episode that scores against a task it never ran.
    """

    def __init__(self, urls: list[str]):
        if not urls:
            raise ValueError("need at least one AppWorld server URL")
        self.urls = list(urls)
        self._free: queue.Queue[str] = queue.Queue()
        for url in self.urls:
            self._free.put(url)

    @contextmanager
    def lease(self):
        url = self._free.get()
        try:
            yield url
        finally:
            self._free.put(url)


def check_servers(urls: list[str]) -> list[dict]:
    healths = []
    for url in urls:
        try:
            healths.append({"url": url, **AppWorldEnvironment(url, timeout=30).health()})
        except AppWorldError as exc:
            raise SystemExit(f"no AppWorld server at {url}: {exc}")
    return healths


def build_config(args) -> MemoryConfig:
    policy = WritePolicy.full() if args.policy == "full" else WritePolicy.minimal()
    policy.n_max = args.n_max
    policy.batch_every = args.batch_every
    return MemoryConfig(
        policy=policy,
        # 2500 rather than ALFWorld/WebShop's 1500: the design doc sets a larger
        # budget for AppWorld, and the scaffold is already ~6.6k tokens, so a
        # 1500-token cap would make the memory block a rounding error against it.
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


def make_agent(args, base_url: str, app_descriptions: str) -> AppWorldAgent:
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key="EMPTY", timeout=args.timeout, max_retries=3)
    return AppWorldAgent(
        client=client,
        model=args.model,
        app_descriptions=app_descriptions,
        config=AgentConfig(
            max_interactions=args.max_interactions,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        ),
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
    study, while the actor stays fixed at `--model` so any delta is attributable
    to what was written and not to who acted.

    `json_mode` only reaches the chat path. It is vLLM guided decoding, and the
    gateway that serves the frontier writers rejects `text.format` outright (see
    OpenAIResponsesClient); those models are asked for JSON in the prompt and
    lean on the writers' own repair pass instead.
    """
    model = args.writer_model or args.model
    key = "EMPTY"
    if args.writer_api_key_env:
        key = _dotenv(args.writer_api_key_env)
        if not key:
            # Failing here is the point: an unauthenticated writer would 401 on
            # every call, and `parse_ops` turns that into an empty store rather
            # than an error, so the arm would finish and report a number.
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
        # On AppWorld this is not a nicety. The writer is asked to quote the
        # trajectory verbatim into `evidence`, and AppWorld trajectories are
        # Python code and API docs full of quotes and backslashes; Qwen3.5-9B
        # loses the escaping partway through the string and the whole response
        # parses to nothing. Measured on the smoke run: the skill arm made writer
        # calls on every successful episode and produced zero entries, ending
        # with an empty store -- an arm silently identical to `none`. See
        # memsys/llm.py for why no repair pass fixes it.
        json_mode=args.writer_json_mode,
    )


def episode_row(ep: Episode, seconds: float) -> dict:
    return {
        "task_id": ep.task_id,
        "instruction": ep.instruction,
        "task_family": ep.scope.get("task_type", ""),
        "outcome": ep.outcome(),
        "success_rate": ep.success_rate,
        # Task Goal Completion: every unit test for the task passes.
        "any_success": ep.any_success,
        # The graded number: fraction of the task's unit tests that pass.
        "score": max((r.reward for r in ep.rollouts), default=0.0),
        "difficulty": ep.rollouts[0].meta.get("difficulty") if ep.rollouts else None,
        "rollouts": [
            {
                "rollout_id": r.rollout_id,
                "success": r.success,
                "reward": r.reward,
                "n_steps": len(r.steps),
                "error": r.error,
                **{k: r.meta.get(k) for k in
                   ("parse_failures", "exec_errors", "completed_task", "num_tests",
                    "n_passed", "difficulty", "prompt_tokens", "completion_tokens",
                    "injected_memory_tokens")},
            }
            for r in ep.rollouts
        ],
        "seconds": round(seconds, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--policy", choices=("minimal", "full"), default="full")
    ap.add_argument("--evolve-manifest")
    ap.add_argument("--eval-manifest", required=True)
    ap.add_argument("--server", action="append", default=None,
                    help="AppWorld env server URL; repeat once per concurrent episode")
    ap.add_argument("--out", required=True)

    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--agent-base-url", default="http://localhost:8000/v1")
    ap.add_argument("--writer-base-url", default=None,
                    help="defaults to --agent-base-url; point elsewhere to split GPUs "
                         "or at a remote gateway")
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
    ap.add_argument("--env-timeout", type=float, default=900.0)

    ap.add_argument("--n-rollouts", type=int, default=1)
    ap.add_argument("--max-interactions", type=int, default=30)
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--writer-max-tokens", type=int, default=1024)
    ap.add_argument("--writer-json-mode", action=argparse.BooleanOptionalAction, default=True,
                    help="constrain writer output to valid JSON via the server's guided decoding")
    ap.add_argument("--max-evidence-tokens", type=int, default=300,
                    help="grounding-evidence cap; 80 (the default elsewhere) rejects most "
                         "AppWorld writes because its evidence is code, not prose")

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
    ap.add_argument("--resume-store", default=None,
                    help="load this store.jsonl before evolving, to continue a previous "
                         "run instead of starting from an empty memory")
    ap.add_argument("--evolve-step-offset", type=int, default=0,
                    help="step index the evolving loop starts from; set to the number of "
                         "episodes the resumed store already saw")
    ap.add_argument("--experiment-name", default=None,
                    help="AppWorld experiment output namespace; must be unique per "
                         "concurrent arm (defaults to memsys_<arm>_<policy>)")
    args = ap.parse_args()
    experiment_name = args.experiment_name or f"memsys_{args.arm}_{args.policy}"

    servers = args.server or [os.environ.get("APPWORLD_SERVER_URL", "http://localhost:9000")]
    healths = check_servers(servers)
    pool = ServerPool(servers)

    # Applied before any system is built, and identically for every arm: this is
    # a benchmark-calibration constant, not an arm-level knob. At the default 80
    # the measured rejection reason on AppWorld is overwhelmingly
    # "evidence too long", because the writer quotes Python code -- and it bites
    # hardest on the content types that write least often.
    set_max_evidence_tokens(args.max_evidence_tokens)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    config = build_config(args)
    writer_url = args.writer_base_url or args.agent_base_url

    system = None
    if args.arm != "none":
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

    # Resolve the scaffold's app list once, while the pool is idle. Doing it per
    # episode would spend one of the agent's interactions on it.
    eval_tasks_peek = load_manifest(args.eval_manifest)
    app_descriptions = fetch_app_descriptions(servers[0], eval_tasks_peek[0].task_id,
                                              experiment_name=experiment_name)
    agent = make_agent(args, args.agent_base_url, app_descriptions)

    (out / "config.json").write_text(json.dumps({
        "args": vars(args),
        "servers": healths,
        "app_descriptions_chars": len(app_descriptions),
        "max_evidence_tokens": args.max_evidence_tokens,
        "experiment_name": experiment_name,
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
        with (out / "evolve_episodes.jsonl").open("w", encoding="utf-8") as fh:
            for i, task in enumerate(tasks):
                t0 = time.time()
                ret = ev.retrieve(task.instruction, scope=task.scope())
                with pool.lease() as url:
                    ep = run_task(agent, url, task, memory_block=ret.block,
                                  n_rollouts=args.n_rollouts, timeout=args.env_timeout,
                                  experiment_name=experiment_name)
                ev.step_once(ep, ret)
                row = episode_row(ep, time.time() - t0)
                row["step"] = args.evolve_step_offset + i
                evolve_rows.append(row)
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                store = getattr(system, "store", None)
                print(f"[evolve {i+1}/{len(tasks)}] {row['outcome']:<12} "
                      f"score={row['score']:.2f} "
                      f"store={len(store) if store is not None else '-'} "
                      f"{row['seconds']:.0f}s {task.task_id}", flush=True)
        ev.flush()
        if getattr(system, "store", None) is not None:
            system.store.save(str(out / "store.jsonl"))

    # ----------------------------------------------------------- evaluation
    eval_tasks = eval_tasks_peek
    if args.eval_limit:
        eval_tasks = eval_tasks[: args.eval_limit]

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
        with pool.lease() as url:
            ep = run_task(agent, url, task, memory_block=block,
                          n_rollouts=1, timeout=args.env_timeout,
                          experiment_name=experiment_name)
        return episode_row(ep, time.time() - t0)

    fh = (out / "eval.jsonl").open("w", encoding="utf-8")
    eval_rows = []
    # One worker per server, never more: a worker without a free server just
    # blocks on the lease, and oversubscribing only lengthens the queue.
    with ThreadPoolExecutor(max_workers=len(servers)) as tp:
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
        "benchmark": "AppWorld",
        "arm": args.arm,
        "policy": args.policy,
        "model": args.model,
        "eval_split": eval_tasks[0].split if eval_tasks else "",
        "eval_n": len(eval_rows),
        "eval_success": n_ok,
        "eval_success_rate": n_ok / len(eval_rows) if eval_rows else 0.0,
        "eval_score": _mean(r["score"] for r in eval_rows),
        # The AppWorld analogue of WebShop's purchase rate: did the agent declare
        # itself finished, or simply run out of interactions?
        "eval_completed_rate": _mean(
            1.0 if r["rollouts"][0].get("completed_task") else 0.0 for r in eval_rows
        ),
        "eval_by_family": _by_difficulty(eval_rows),
        "evolve_n": len(evolve_rows),
        # Episodes this memory has seen in total, including any resumed run.
        "evolve_total": args.evolve_step_offset + len(evolve_rows),
        "resumed_from": args.resume_store,
        "evolve_success_rate": (
            sum(r["success_rate"] for r in evolve_rows) / len(evolve_rows) if evolve_rows else None
        ),
        "evolve_score": _mean(r["score"] for r in evolve_rows) if evolve_rows else None,
        "evolve_outcomes": _counts(r["outcome"] for r in evolve_rows),
        "store": system.store.summary() if getattr(system, "store", None) is not None else None,
        # The actor and the writer are separate variables; recording only one of
        # them makes two runs that differ in the writer indistinguishable in the
        # summary. `none`/`raw` have no writer LLM at all, hence the empty string.
        "writer_model": getattr(
            getattr(getattr(system, "writer", None), "llm", None), "model", ""),
        "writer_usage": (
            system.writer.llm.usage.to_dict()
            if getattr(system, "writer", None) is not None else None
        ),
        "mean_injected_tokens": _mean(
            r["rollouts"][0].get("injected_memory_tokens") or 0 for r in eval_rows
        ),
        "mean_steps": _mean(r["rollouts"][0].get("n_steps") or 0 for r in eval_rows),
        "mean_exec_errors": _mean(r["rollouts"][0].get("exec_errors") or 0 for r in eval_rows),
        "mean_prompt_tokens": _mean(r["rollouts"][0].get("prompt_tokens") or 0 for r in eval_rows),
        "wall_seconds": round(time.time() - t_start, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))


def _counts(values):
    out: dict[str, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return out


def _mean(values):
    vals = list(values)
    return round(sum(vals) / len(vals), 3) if vals else 0.0


def _by_difficulty(rows):
    """Grouped by the evaluator's difficulty, not by the scope key.

    `task_type` is the scenario, and there are ~34 of them across 100 evaluation
    tasks -- far too fine to read. Difficulty comes back from `/evaluate`, so it
    costs nothing and actually separates the tasks.
    """
    agg: dict[str, list[tuple[int, float]]] = {}
    for r in rows:
        key = f"difficulty_{r.get('difficulty')}" if r.get("difficulty") is not None else "unknown"
        agg.setdefault(key, []).append((1 if r["any_success"] else 0, r["score"]))
    return {
        k: {
            "n": len(v),
            "success": sum(s for s, _ in v),
            "rate": sum(s for s, _ in v) / len(v),
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

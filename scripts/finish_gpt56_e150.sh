#!/usr/bin/env bash
# Finish the interrupted third leg of the gpt-5.6-terra writer chain on ALFWorld.
#
# `drive_gpt56_writer_100_150.sh` completed `minimal100` and was killed partway
# into `minimal150` (STEP 2276408.3 CANCELLED at 2026-08-16T12:27). What survived:
#
#   reflection  evolve DONE (50 episodes, store 74 rows)   eval 72/100  -> killed
#   rule        evolve DONE (50 episodes, store 95 rows)   eval 32/100  -> killed
#   skill       never started
#   none / raw  never seeded from the Qwen chain
#
# The evolving phase is the expensive, sequential, non-reproducible part -- it
# spends gpt-5.6 gateway calls and its result is already on disk in a *complete*
# store.jsonl (run_alfworld.py writes the store only after the evolve loop
# finishes, so a store.jsonl at all means that arm's 50 episodes all ran).
# Evaluation is the cheap, frozen, embarrassingly parallel part and writes
# eval.jsonl with mode "w", i.e. it cannot resume mid-file.
#
# So the two half-finished arms are re-run for EVALUATION ONLY, resuming their
# own finished store: omitting --evolve-manifest skips the evolve loop entirely,
# which also leaves store.jsonl / evolve_episodes.jsonl / evolve_log.jsonl
# untouched. `skill` has nothing to salvage and runs the normal leg via
# run_sweep.sh.
#
# The one thing an eval-only run cannot reproduce is the evolve half of
# summary.json (evolve_n, writer_usage, ...), because those are accumulated
# in-process. Every field of it is recoverable from evolve_episodes.jsonl and
# evolve_log.jsonl, which the killed run flushed per episode -- verified against
# the completed e100 leg, where the per-episode writer_calls /
# writer_{prompt,completion}_tokens sum exactly to summary.writer_usage for all
# three arms. Step 4 below patches them back in; without it the summary would
# claim zero writer calls for arms that made ~50 each.
#
#   srun --jobid=<job> --overlap -N1 -n1 bash scripts/finish_gpt56_e150.sh
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
S=/gpfs/radev/scratch/cohan/jw3278
R="$S/memsys_results"
TAG=minimal_e150_gpt56terra
OUT="$R/$TAG"
PY="${MEMSYS_PY:-$S/envs/memsys-alfworld/bin/python}"

# See drive_gpt56_writer.sh: serve_qwen.sh needs $S/hf_cache (the Qwen snapshot)
# and the runner needs $S/.hf_home (sentence-transformers). Exporting either one
# up front points vLLM at a cache with no model in it, and with HF_HUB_OFFLINE=1
# that fails seconds *after* "[serve] started (PID ...)" prints.
unset HF_HOME

export MEMSYS_ARMS="skill"
export MEMSYS_WRITER_MODEL="openai/gpt-5.6-terra"
export MEMSYS_WRITER_BASE_URL="https://api.perplexity.ai/v1"
export MEMSYS_WRITER_API="responses"
export MEMSYS_WRITER_API_KEY_ENV="perplexity_api_key"
export MEMSYS_TAG_SUFFIX="_gpt56terra"

EVAL_MANIFEST="$REPO/manifests/eval_valid_unseen_100_seed42.json"
ALFWORLD_DATA="${ALFWORLD_DATA:-$S/alfworld_data}"

# --- 0. preconditions -----------------------------------------------------
# `skill` resumes the e100 store; reflection/rule resume their own e150 store.
# A missing file here means a different failure than the one this script fixes,
# so stop rather than silently re-running an arm from empty.
[[ -f "$R/minimal_e100_gpt56terra/skill_minimal/store.jsonl" ]] || {
  echo "[finish] FATAL: no e100 skill store to resume" >&2; exit 1; }
for arm in reflection rule; do
  d="$OUT/${arm}_minimal"
  [[ -f "$d/store.jsonl" ]] || { echo "[finish] FATAL: no $d/store.jsonl" >&2; exit 1; }
  n=$(wc -l < "$d/evolve_episodes.jsonl")
  (( n == 50 )) || { echo "[finish] FATAL: $arm evolved $n/50 episodes -- its store is not final; "\
"re-run the whole leg for this arm instead of eval-only" >&2; exit 1; }
  echo "[finish] $arm: evolve complete ($n episodes, $(wc -l < "$d/store.jsonl") store rows)"
done

# --- 1. servers -----------------------------------------------------------
for p in 8000 8001; do
  if curl -sf -m 5 -o /dev/null "localhost:$p/health"; then
    echo "[finish] FATAL: port $p already healthy -- another allocation owns it." >&2
    exit 1
  fi
done
echo "[finish] starting vLLM on $(hostname -s)"
bash "$REPO/scripts/serve_qwen.sh" 0 8000 --background
bash "$REPO/scripts/serve_qwen.sh" 1 8001 --background
deadline=$(( SECONDS + 1800 ))
until curl -sf -m 5 -o /dev/null localhost:8000/health && \
      curl -sf -m 5 -o /dev/null localhost:8001/health; do
  if (( SECONDS > deadline )); then
    echo "[finish] FATAL: servers not healthy after 30 min; see $S/memsys_qwen9b_$(hostname -s)_gpu*.log" >&2
    exit 1
  fi
  sleep 15
done
echo "[finish] servers healthy after ${SECONDS}s"

# --- 2. the work ----------------------------------------------------------
# GPU 0 / :8000 runs `skill` (evolve 50 + eval 100, the long pole ~2h).
# GPU 1 / :8001 runs the two eval-only arms back to back (~45 min each).
# One arm per server: Qwen3.5 is a hybrid model, so vLLM disables prefix caching
# and over-subscribing a server costs 3-5x per episode.

eval_only() {  # $1 = arm, $2 = port
  local arm="$1" port="$2" d="$OUT/${1}_minimal"
  # Preserve what the killed run produced before overwriting it. The partial
  # eval is kept, not deleted: it is the evidence of where the job died, and
  # `[eval N/100]` counts in the .log are the only other record.
  cp -n "$d/store.jsonl"  "$d/store.evolve.jsonl"   2>/dev/null
  cp -n "$d/config.json"  "$d/config.evolve.json"   2>/dev/null
  [[ -f "$d/eval.jsonl" ]] && mv -f "$d/eval.jsonl" "$d/eval.interrupted.jsonl"
  # --resume-store points at the *copy*: an eval-only run does not save the
  # store (run_alfworld.py saves it inside the evolve branch only), but reading
  # the file it would write to is a footgun waiting for that to change.
  #
  # No --evolve-manifest => no evolving phase. --evolve-step-offset 150 so
  # summary.evolve_total is the memory's true experience count (100 resumed +
  # the 50 already on disk) rather than 0.
  HF_HOME="$S/.hf_home" TOKENIZERS_PARALLELISM=false \
  "$PY" "$REPO/scripts/run_alfworld.py" \
    --arm "$arm" --policy minimal \
    --eval-manifest "$EVAL_MANIFEST" --eval-limit 0 \
    --data-root "$ALFWORLD_DATA" --out "$d" \
    --model "${MEMSYS_MODEL:-Qwen/Qwen3.5-9B}" \
    --agent-base-url "http://localhost:$port/v1" \
    --writer-model "$MEMSYS_WRITER_MODEL" \
    --writer-base-url "$MEMSYS_WRITER_BASE_URL" \
    --writer-api "$MEMSYS_WRITER_API" \
    --writer-api-key-env "$MEMSYS_WRITER_API_KEY_ENV" \
    --embedder st --eval-workers "${MEMSYS_WORKERS:-4}" \
    --resume-store "$d/store.evolve.jsonl" \
    --evolve-step-offset 150 \
    > "$OUT/${arm}_minimal.evalonly.log" 2>&1
  echo "[finish] DONE eval-only $arm rc=$? -> $d"
}

echo "[finish] === launching skill (:8000) and reflection/rule eval-only (:8001) ==="
( MEMSYS_URL_A=http://localhost:8000/v1 MEMSYS_URL_B=http://localhost:8000/v1 \
  bash "$REPO/scripts/run_sweep.sh" minimal150 ) &
skill_pid=$!
( eval_only reflection 8001; eval_only rule 8001 ) &
evals_pid=$!
wait $skill_pid; skill_rc=$?
wait $evals_pid
echo "[finish] skill leg rc=$skill_rc"

# --- 3. baselines ---------------------------------------------------------
# Neither `none` (no memory system) nor `raw` (stores trajectories verbatim,
# no writer LLM) can be moved by the writer model, so both are copied from the
# Qwen chain's matching leg rather than re-measured. summarize.py needs `none`
# present to compute any delta at all.
for arm in none raw; do
  src="$R/minimal_e150/${arm}_minimal" dst="$OUT/${arm}_minimal"
  [[ -d "$src" ]] || { echo "[finish] warn: no $src to copy"; continue; }
  mkdir -p "$dst" && cp -f "$src"/* "$dst"/
  cat > "$dst/REUSED_FROM_QWEN_RUN.txt" <<TXT
Copied from $src. Neither arm calls a writer LLM (none has no memory system;
raw stores trajectories verbatim), so the writer model cannot move them and
they were not re-run for the gpt-5.6-terra chain.
TXT
done

# --- 4. restore the evolve half of the eval-only summaries ----------------
"$PY" - "$OUT" reflection rule <<'PY'
import json, sys
from pathlib import Path

out = Path(sys.argv[1])
for arm in sys.argv[2:]:
    d = out / f"{arm}_minimal"
    sp = d / "summary.json"
    if not sp.is_file():
        print(f"[patch] {arm}: no summary.json (eval-only run failed?) -- skipped")
        continue
    s = json.loads(sp.read_text())
    if s.get("evolve_n"):
        print(f"[patch] {arm}: evolve_n={s['evolve_n']} already set -- skipped")
        continue
    eps = [json.loads(l) for l in (d / "evolve_episodes.jsonl").read_text().splitlines() if l.strip()]
    log = [json.loads(l) for l in (d / "evolve_log.jsonl").read_text().splitlines() if l.strip()]

    outcomes = {}
    for e in eps:
        outcomes[e["outcome"]] = outcomes.get(e["outcome"], 0) + 1
    written = sum(1 for e in eps if e.get("written", True))
    # by_tag is not in the per-episode log (it lives in the LLM client's usage
    # counter), so it is dropped rather than guessed. The three scalars that the
    # results table actually reads are exact.
    s.update({
        "evolve_n": len(eps),
        "evolve_written": written,
        "evolve_successes": sum(1 for e in eps if e.get("any_success")),
        "evolve_failures": sum(1 for e in eps if not e.get("any_success")),
        "evolve_budget": {"kind": "tasks", "target": len(eps)},
        "evolve_total": 100 + written,
        "evolve_success_rate": sum(e["success_rate"] for e in eps) / len(eps) if eps else None,
        "evolve_outcomes": outcomes,
        "resumed_from": str(
            Path(str(d).replace("_e150_", "_e100_")) / "store.jsonl"),
        "writer_usage": {
            "n_calls": sum(r["writer_calls"] for r in log),
            "prompt_tokens": sum(r["writer_prompt_tokens"] for r in log),
            "completion_tokens": sum(r["writer_completion_tokens"] for r in log),
            "by_tag": None,
        },
        "eval_only_rerun": (
            "evaluation re-run after the original leg was killed mid-eval; the "
            "evolve fields above are restored from evolve_episodes.jsonl and "
            "evolve_log.jsonl, and writer_usage.by_tag is unrecoverable"),
    })
    sp.write_text(json.dumps(s, indent=2, default=str) + "\n")
    print(f"[patch] {arm}: evolve_n={len(eps)} writer_calls={s['writer_usage']['n_calls']} "
          f"eval {s['eval_success']}/{s['eval_n']}")
PY

# --- 5. the table ---------------------------------------------------------
"$PY" "$REPO/scripts/summarize.py" --root "$OUT" || true
echo "[finish] done"

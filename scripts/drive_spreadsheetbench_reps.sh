#!/usr/bin/env bash
# Drive the SpreadsheetBench *repetition vs diversity* legs: 75 x 2 epochs and
# 50 x 3 epochs, both landing on 150 evolving episodes.
#
#   bash scripts/drive_spreadsheetbench_reps.sh minimal75 minimal75_x2 minimal_x2 minimal_x3
#   MEMSYS_MAX_MODEL_LEN=131072 \
#     bash scripts/drive_spreadsheetbench_reps.sh full75 full75_x2 full_x2 full_x3
#
# Modes are run in the order given and are STRICTLY ORDERED within a chain: an
# `_x2` leg resumes the store its epoch-1 leg wrote, so it cannot start until
# that leg has finished. The two chains (75-task and 50-task) are independent of
# each other; only the legs inside one are sequenced.
#
# Actor and memory writer are both the local Qwen3.5-9B -- no --writer-model, so
# `run_spreadsheetbench_sweep.sh` leaves the writer on the agent's own server.
# That is the setting every SpreadsheetBench section except the writer-model
# study uses, and it is what makes these legs comparable to `*_e150`.
#
# Two operational facts this script exists to encode:
#
# 1. **Idempotent by leg.** A leg whose every arm already has a summary.json is
#    skipped, so relaunching after a kill resumes at the leg that died rather
#    than redoing finished ones. It does NOT resume a half-finished leg -- the
#    sweep has no mid-leg resume, and re-running one restarts its evolving phase
#    from the store it was told to resume, which is still correct.
# 2. **`full` needs a 131072-token server.** Batch induction's prompt grows with
#    the store; reflection/full and rule/full already died against 32768 at
#    [100,150) and 65536 only moved the wall (RESULTS_SPREADSHEETBENCH.md,
#    "Batch induction outgrows the context window"). These chains reach 150
#    episodes, so serve at 131072 for any `full*` mode. `minimal` has no
#    induction and is unaffected. The script refuses the combination rather than
#    letting an arm die 6 hours in.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
S=/gpfs/radev/scratch/cohan/jw3278
OUT_ROOT="$S/memsys_results/spreadsheetbench"
MODES=("$@")
(( ${#MODES[@]} )) || { echo "usage: $0 <mode> [<mode> ...]" >&2; exit 2; }

# See drive_gpt56_writer.sh: serve_qwen.sh needs $S/hf_cache (the Qwen weights),
# the sweep needs $S/.hf_home (sentence-transformers), and exporting either one
# up front points vLLM at a cache with no model in it.
unset HF_HOME

MAX_LEN="${MEMSYS_MAX_MODEL_LEN:-32768}"
# 4 arms x 4 eval workers = 16 concurrent episodes, each running the agent's
# bash plus up to two `python solution.py` subprocesses. The sweep's default of
# 6 is sized for a >= 32-core node; these allocations have 16 cores.
export MEMSYS_EVAL_WORKERS="${MEMSYS_EVAL_WORKERS:-4}"
ARMS=(${MEMSYS_ARMS:-none raw reflection rule skill})

tag_of() {  # mode -> the sweep's output directory name
  case "$1" in
    minimal|full)         echo "$1" ;;
    minimal75|full75)     echo "${1%75}_e75" ;;
    minimal75_x2|full75_x2) echo "${1%75_x2}_e75_x2" ;;
    minimal_x2|full_x2|minimal_x3|full_x3) echo "$1" ;;
    *) echo "" ;;
  esac
}
policy_of() { case "$1" in full*) echo full ;; *) echo minimal ;; esac; }

# --- 0. refuse the known-fatal combination before spending anything --------
for m in "${MODES[@]}"; do
  [[ -n "$(tag_of "$m")" ]] || { echo "[reps] FATAL: unknown mode $m" >&2; exit 2; }
  if [[ "$m" == full* && "$MAX_LEN" -lt 131072 ]]; then
    echo "[reps] FATAL: $m is a \`full\` leg reaching 150 episodes but MEMSYS_MAX_MODEL_LEN=$MAX_LEN." >&2
    echo "[reps] Batch induction will outgrow that window. Re-run with MEMSYS_MAX_MODEL_LEN=131072." >&2
    exit 1
  fi
done

# --- 1. servers -----------------------------------------------------------
# Reuse a healthy pair rather than refusing: unlike the ALFWorld drivers this
# script is meant to run against servers an earlier sweep left behind. What it
# will NOT do is reuse a pair serving the wrong window, which is invisible until
# an induction prompt overflows -- so check the window it actually advertises.
serve_needed=0
for p in 8000 8001; do
  curl -sf -m 5 -o /dev/null "localhost:$p/health" || serve_needed=1
done
if (( serve_needed )); then
  echo "[reps] starting vLLM on $(hostname -s) at max_model_len=$MAX_LEN"
  MEMSYS_MAX_MODEL_LEN="$MAX_LEN" bash "$REPO/scripts/serve_qwen.sh" 0 8000 --background
  MEMSYS_MAX_MODEL_LEN="$MAX_LEN" bash "$REPO/scripts/serve_qwen.sh" 1 8001 --background
  deadline=$(( SECONDS + 1800 ))
  until curl -sf -m 5 -o /dev/null localhost:8000/health && \
        curl -sf -m 5 -o /dev/null localhost:8001/health; do
    if (( SECONDS > deadline )); then
      echo "[reps] FATAL: servers not healthy after 30 min; see $S/memsys_qwen9b_$(hostname -s)_gpu*.log" >&2
      exit 1
    fi
    sleep 15
  done
  echo "[reps] servers healthy after ${SECONDS}s"
else
  echo "[reps] reusing the healthy servers already on $(hostname -s)"
fi
for p in 8000 8001; do
  have=$(curl -sf -m 10 "localhost:$p/v1/models" \
         | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["max_model_len"])' 2>/dev/null)
  echo "[reps] :$p max_model_len=$have"
  if [[ -n "$have" && "$have" -lt "$MAX_LEN" ]]; then
    echo "[reps] FATAL: :$p serves $have but this run needs $MAX_LEN. Stop it with" >&2
    echo "[reps]   pkill -f 'verl_qwen35_train/bin/vll[m] serve'" >&2
    echo "[reps] and re-run so the pair comes up at the right window." >&2
    exit 1
  fi
done

# --- 2. the legs, in order ------------------------------------------------
for m in "${MODES[@]}"; do
  tag="$(tag_of "$m")"; policy="$(policy_of "$m")"
  done_all=1
  for arm in "${ARMS[@]}"; do
    [[ -f "$OUT_ROOT/$tag/${arm}_${policy}/summary.json" ]] || done_all=0
  done
  if (( done_all )); then
    echo "[reps] SKIP $m -- every arm in $tag already has a summary.json"
    continue
  fi
  echo "[reps] $(date +%F\ %T) === START $m -> $tag ==="
  bash "$REPO/scripts/run_spreadsheetbench_sweep.sh" "$m"
  rc=$?
  echo "[reps] $(date +%F\ %T) === DONE  $m rc=$rc ==="
  # A failed leg poisons every leg after it in the same chain -- `_x2` would
  # resume a store that was never written, or worse, a stale one from an earlier
  # attempt. Stop rather than build on it.
  (( rc == 0 )) || { echo "[reps] FATAL: $m failed; later legs in this chain would resume a bad store" >&2; exit "$rc"; }
done

echo "[reps] all modes complete: ${MODES[*]}"

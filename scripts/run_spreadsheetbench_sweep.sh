#!/usr/bin/env bash
# The SpreadsheetBench Memory Content sweep: every arm x every write policy.
#
#   bash scripts/run_spreadsheetbench_sweep.sh smoke      # 2 evolve / 4 eval, all arms
#   bash scripts/run_spreadsheetbench_sweep.sh minimal    # 50 evolve / 100 eval, WritePolicy.minimal()
#   bash scripts/run_spreadsheetbench_sweep.sh full       # 50 evolve / 100 eval, WritePolicy.full()
#   bash scripts/run_spreadsheetbench_sweep.sh minimal100 # continue each store to 100 evolve episodes
#   bash scripts/run_spreadsheetbench_sweep.sh full100
#   bash scripts/run_spreadsheetbench_sweep.sh minimal150 # ... and on to 150
#   bash scripts/run_spreadsheetbench_sweep.sh full150
#
# Prerequisites:
#   bash scripts/setup_spreadsheetbench.sh
#   python scripts/build_spreadsheetbench_manifests.py
#   bash scripts/serve_qwen.sh 0 8000 --background
#   bash scripts/serve_qwen.sh 1 8001 --background
#
# **No environment servers, and that is the whole difference from the AppWorld
# sweep.** An episode is a temp directory plus subprocesses, so arms need no port
# partitioning and cannot corrupt each other's world. What they do share is this
# machine: every concurrent episode runs the agent's bash commands plus up to two
# `python solution.py` subprocesses at scoring time, so the real ceiling is
# ARM_CONCURRENCY x EVAL_WORKERS processes and the memory they load workbooks
# into. The defaults (4 x 6 = 24) are sized for a node with >= 32 cores; drop
# EVAL_WORKERS first on anything smaller.
#
# Sizing: evolving is sequential *within* an arm, so it is the wall-clock floor.
# Running four arms at once is what shortens the sweep; EVAL_WORKERS only speeds
# the evaluation phase.
set -uo pipefail

MODE="${1:-smoke}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
S=/gpfs/radev/scratch/cohan/jw3278

PY="${MEMSYS_PY:-$S/envs/memsys-alfworld/bin/python}"
export HF_HOME="${HF_HOME:-$S/.hf_home}"
export TOKENIZERS_PARALLELISM=false
OUT_ROOT="${MEMSYS_RESULTS_ROOT:-$S/memsys_results}/spreadsheetbench"
MODEL="${MEMSYS_MODEL:-Qwen/Qwen3.5-9B}"
URL_A="${MEMSYS_URL_A:-http://localhost:8000/v1}"
URL_B="${MEMSYS_URL_B:-http://localhost:8001/v1}"
DATA_ROOT="${SPREADSHEETBENCH_ROOT:-$S/spreadsheetbench_root/all_data_912_v0.1}"

ARM_CONCURRENCY="${MEMSYS_ARM_CONCURRENCY:-4}"
EVAL_WORKERS="${MEMSYS_EVAL_WORKERS:-6}"

EVOLVE_MANIFEST="$REPO/manifests/spreadsheetbench_evolve_train_50_seed42.json"
EVAL_MANIFEST="$REPO/manifests/spreadsheetbench_eval_test_100_seed42.json"

# `minimal100` / `full100` continue an existing 50-episode run rather than
# repeating it: each arm resumes its own store.jsonl and evolves the *next* 50
# tasks of the same seeded permutation, so the "evolve on 50 vs 100" column is a
# comparison of amount of experience. Running 100 from scratch would instead
# repeat 50 episodes of compute and throw away the memory they produced.
#
# The continuation manifest spills into `val`: train holds 80 tasks and 50 are
# already spent, so positions [50, 100) are 29 train + 21 val. `val` is disjoint
# from `test`, and the manifest was built excluding any source workbook used by
# the frozen eval set, so the evaluation set is untouched.
#
# The eval manifest is the SAME 100 tasks as the 50-episode runs, which is what
# makes the two directly comparable -- and what lets the `none` baseline be
# reused rather than re-run.
RESUME_ROOT=""; EVOLVE_OFFSET=0
case "$MODE" in
  smoke)
    POLICIES=(full); EVOLVE_LIMIT=2; EVAL_LIMIT=4
    EMBEDDER="${MEMSYS_EMBEDDER:-hashing}"; TAG=smoke ;;
  minimal|full)
    POLICIES=("$MODE"); EVOLVE_LIMIT=0; EVAL_LIMIT=0
    EMBEDDER="${MEMSYS_EMBEDDER:-st}"; TAG="$MODE" ;;
  minimal100|full100)
    POLICIES=("${MODE%100}"); EVOLVE_LIMIT=0; EVAL_LIMIT=0
    EMBEDDER="${MEMSYS_EMBEDDER:-st}"; TAG="${MODE%100}_e100"
    EVOLVE_MANIFEST="$REPO/manifests/spreadsheetbench_evolve_train+val_50to100_seed42.json"
    RESUME_ROOT="$OUT_ROOT/${MODE%100}"; EVOLVE_OFFSET=50 ;;
  # The third leg, [100, 150). train+val is spent -- it holds 117 selectable
  # tasks once the eval set's source workbooks are excluded -- so this leg
  # overflows into `test` as well. That is safe but worth stating plainly: the
  # extra 33 tasks come from the same split the evaluation set is drawn from,
  # and they are disjoint from it only because the group filter makes them so.
  # The builder enforces both halves of that (no shared task id, no shared
  # source workbook) and refuses to write the manifest otherwise. Positions
  # [0, 100) are bit-identical to the two manifests above, so the three legs
  # remain one nested sequence.
  minimal150|full150)
    POLICIES=("${MODE%150}"); EVOLVE_LIMIT=0; EVAL_LIMIT=0
    EMBEDDER="${MEMSYS_EMBEDDER:-st}"; TAG="${MODE%150}_e150"
    EVOLVE_MANIFEST="$REPO/manifests/spreadsheetbench_evolve_train+val+test_100to150_seed42.json"
    RESUME_ROOT="$OUT_ROOT/${MODE%150}_e100"; EVOLVE_OFFSET=100 ;;
  *) echo "usage: $0 {smoke|minimal|full|minimal100|full100|minimal150|full150}" >&2; exit 2 ;;
esac

ARMS=(${MEMSYS_ARMS:-none raw reflection rule skill})

for url in "$URL_A" "$URL_B"; do
  curl -sf -m 5 -o /dev/null "${url%/v1}/health" \
    || { echo "[sweep] FAILED: no vLLM at $url (run scripts/serve_qwen.sh)"; exit 1; }
done
[[ -f "$DATA_ROOT/dataset.json" ]] \
  || { echo "[sweep] FAILED: no dataset.json under $DATA_ROOT (run scripts/setup_spreadsheetbench.sh)"; exit 1; }
for m in "$EVOLVE_MANIFEST" "$EVAL_MANIFEST"; do
  [[ -f "$m" ]] || { echo "[sweep] FAILED: missing $m (run scripts/build_spreadsheetbench_manifests.py)"; exit 1; }
done
"$PY" -c "import openpyxl, pandas" \
  || { echo "[sweep] FAILED: openpyxl/pandas missing from $PY (run scripts/setup_spreadsheetbench.sh)"; exit 1; }
mkdir -p "$OUT_ROOT"

run_arm() {  # $1=arm $2=policy $3=vllm_url
  local arm="$1" policy="$2" url="$3"
  local out="$OUT_ROOT/$TAG/${arm}_${policy}"
  local log="$out.log"
  mkdir -p "$(dirname "$log")"

  # The "none" arm builds no memory system, so WritePolicy cannot reach it: run
  # it once and reuse it as the shared baseline. The eval size is in the
  # directory name so a 4-task smoke baseline can never be mistaken for the
  # 100-task reference.
  local shared="$OUT_ROOT/_baseline_none_e${EVAL_LIMIT}"
  if [[ "$arm" == "none" ]]; then
    if [[ -f "$shared/summary.json" ]]; then
      echo "[sweep] REUSE none baseline from $shared"
      mkdir -p "$out" && cp -rf "$shared"/* "$out"/ && return 0
    fi
    out="$shared"; log="$shared.log"
  fi
  local evolve_args=(--evolve-manifest "$EVOLVE_MANIFEST" --evolve-limit "$EVOLVE_LIMIT")
  [[ "$arm" == "none" ]] && evolve_args=()
  # Continuation modes: resume this arm's own 50-episode store and carry the
  # Evolver's step counter, so `full`'s every-25-episode batch induction fires
  # where an uninterrupted 100-episode run would have fired it.
  if [[ -n "$RESUME_ROOT" && "$arm" != "none" ]]; then
    local prior="$RESUME_ROOT/${arm}_${policy}/store.jsonl"
    [[ -f "$prior" ]] || { echo "[sweep] FAILED: no store to resume at $prior"; return 1; }
    evolve_args+=(--resume-store "$prior" --evolve-step-offset "$EVOLVE_OFFSET")
  fi
  echo "[sweep] START $arm/$policy -> $url  (log: $log)"
  "$PY" "$REPO/scripts/run_spreadsheetbench.py" \
    --arm "$arm" --policy "$policy" \
    "${evolve_args[@]}" \
    --eval-manifest "$EVAL_MANIFEST" --eval-limit "$EVAL_LIMIT" \
    --data-root "$DATA_ROOT" --out "$out" \
    --model "$MODEL" --agent-base-url "$url" \
    --eval-workers "$EVAL_WORKERS" \
    --embedder "$EMBEDDER" \
    > "$log" 2>&1
  local rc=$?
  echo "[sweep] DONE  $arm/$policy rc=$rc -> $out"
  if [[ "$arm" == "none" && $rc -eq 0 ]]; then
    mkdir -p "$OUT_ROOT/$TAG/none_${policy}"
    cp -rf "$out"/* "$OUT_ROOT/$TAG/none_${policy}"/
  fi
  return $rc
}

URLS=("$URL_A" "$URL_B")

for policy in "${POLICIES[@]}"; do
  i=0
  for arm in "${ARMS[@]}"; do
    run_arm "$arm" "$policy" "${URLS[$(( i % ${#URLS[@]} ))]}" &
    i=$((i+1))
    if (( i % ARM_CONCURRENCY == 0 )); then wait; fi
  done
  wait
done

echo "[sweep] all arms complete for: ${POLICIES[*]}"
"$PY" "$REPO/scripts/summarize.py" --root "$OUT_ROOT/$TAG" || true

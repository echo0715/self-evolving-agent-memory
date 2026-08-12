#!/usr/bin/env bash
# The ALFWorld Memory Content sweep: every arm x every write policy.
#
#   bash scripts/run_sweep.sh smoke     # 2 evolve / 2 eval, all arms, validates plumbing
#   bash scripts/run_sweep.sh minimal   # 50 evolve / 100 eval, WritePolicy.minimal()
#   bash scripts/run_sweep.sh full      # 50 evolve / 100 eval, WritePolicy.full()
#
# Arms are dispatched two at a time, one per vLLM server (GPU 0 -> :8000,
# GPU 1 -> :8001). Within an arm the evolving phase is sequential by
# construction; only the frozen evaluation fans out across threads.
set -uo pipefail

MODE="${1:-smoke}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
S=/gpfs/radev/scratch/cohan/jw3278

PY="${MEMSYS_PY:-$S/envs/memsys-alfworld/bin/python}"
export ALFWORLD_DATA="${ALFWORLD_DATA:-$S/alfworld_data}"
# Keep the sentence-transformers download/cache off HOME (at its inode quota).
# This cache already holds BAAI/bge-large-en-v1.5.
export HF_HOME="${HF_HOME:-$S/.hf_home}"
export TOKENIZERS_PARALLELISM=false   # silences a fork warning under ProcessPool
# Outputs go to scratch: the shared cohan *project* group quota has been full
# before (SAGE hit intermittent OSError 122 writing results there).
OUT_ROOT="${MEMSYS_RESULTS_ROOT:-$S/memsys_results}"
MODEL="${MEMSYS_MODEL:-Qwen/Qwen3.5-9B}"
URL_A="${MEMSYS_URL_A:-http://localhost:8000/v1}"
URL_B="${MEMSYS_URL_B:-http://localhost:8001/v1}"

EVOLVE_MANIFEST="$REPO/manifests/evolve_train_50_seed42.json"
EVAL_MANIFEST="$REPO/manifests/eval_valid_unseen_100_seed42.json"

# `minimal100` / `full100` continue an existing 50-episode run: each arm resumes
# its own store.jsonl and evolves the *next* 50 tasks of the same seeded
# permutation, making the "evolve on 50 vs 100" column a comparison of amount of
# experience rather than two independent runs.
RESUME_ROOT=""; EVOLVE_OFFSET=0
case "$MODE" in
  smoke)
    POLICIES=(full); EVOLVE_LIMIT=2; EVAL_LIMIT=2; WORKERS=2
    EMBEDDER="${MEMSYS_EMBEDDER:-hashing}"; TAG=smoke ;;
  minimal|full)
    POLICIES=("$MODE"); EVOLVE_LIMIT=0; EVAL_LIMIT=0; WORKERS="${MEMSYS_WORKERS:-4}"
    EMBEDDER="${MEMSYS_EMBEDDER:-st}"; TAG="$MODE" ;;
  minimal100|full100)
    POLICIES=("${MODE%100}"); EVOLVE_LIMIT=0; EVAL_LIMIT=0; WORKERS="${MEMSYS_WORKERS:-4}"
    EMBEDDER="${MEMSYS_EMBEDDER:-st}"; TAG="${MODE%100}_e100"
    EVOLVE_MANIFEST="$REPO/manifests/evolve_train_50to100_seed42.json"
    RESUME_ROOT="$OUT_ROOT/${MODE%100}"; EVOLVE_OFFSET=50 ;;
  *) echo "usage: $0 {smoke|minimal|full|minimal100|full100}" >&2; exit 2 ;;
esac

ARMS=(${MEMSYS_ARMS:-none raw reflection rule skill})
mkdir -p "$OUT_ROOT"

run_arm() {  # $1=arm $2=policy $3=base_url
  local arm="$1" policy="$2" url="$3"
  local out="$OUT_ROOT/$TAG/${arm}_${policy}"
  local log="$out.log"
  mkdir -p "$(dirname "$log")"

  # The "none" arm builds no memory system, so WritePolicy cannot reach it: a
  # minimal and a full run would be the same 100 episodes twice. Run it once and
  # reuse it as the shared baseline for every policy.
  #
  # The eval size is in the directory name on purpose. Without it the 2-task
  # smoke baseline would be silently reused as the baseline for the 100-task
  # run -- every delta would then be computed against the wrong reference, and
  # nothing in the output would look wrong.
  local shared="$OUT_ROOT/_baseline_none_e${EVAL_LIMIT}"
  if [[ "$arm" == "none" ]]; then
    if [[ -f "$shared/summary.json" ]]; then
      echo "[sweep] REUSE none baseline from $shared"
      mkdir -p "$out" && cp -f "$shared"/* "$out"/ && return 0
    fi
    out="$shared"; log="$shared.log"
  fi
  # The "none" arm has nothing to evolve; passing a manifest would be a no-op
  # but the flag is omitted so the intent is visible in config.json.
  local evolve_args=(--evolve-manifest "$EVOLVE_MANIFEST" --evolve-limit "$EVOLVE_LIMIT")
  if [[ -n "$RESUME_ROOT" ]]; then
    local prior="$RESUME_ROOT/${arm}_${policy}/store.jsonl"
    [[ -f "$prior" ]] || { echo "[sweep] FAILED: no store to resume at $prior"; return 1; }
    evolve_args+=(--resume-store "$prior" --evolve-step-offset "$EVOLVE_OFFSET")
  fi
  # `none` has no memory: its evaluation is identical at 50 and 100 evolving
  # episodes, so it is reused rather than re-run.
  [[ "$arm" == "none" ]] && evolve_args=()
  echo "[sweep] START $arm/$policy -> $url  (log: $log)"
  "$PY" "$REPO/scripts/run_alfworld.py" \
    --arm "$arm" --policy "$policy" \
    "${evolve_args[@]}" \
    --eval-manifest "$EVAL_MANIFEST" --eval-limit "$EVAL_LIMIT" \
    --data-root "$ALFWORLD_DATA" --out "$out" \
    --model "$MODEL" --agent-base-url "$url" \
    --embedder "$EMBEDDER" --eval-workers "$WORKERS" \
    > "$log" 2>&1
  local rc=$?
  echo "[sweep] DONE  $arm/$policy rc=$rc -> $out"
  # Seed the reusable copy so the second policy does not re-run the baseline.
  if [[ "$arm" == "none" && $rc -eq 0 ]]; then
    mkdir -p "$OUT_ROOT/$TAG/none_${policy}"
    cp -f "$out"/* "$OUT_ROOT/$TAG/none_${policy}"/
  fi
  return $rc
}

# How many arms run at once, round-robined across the two servers. Default 2
# (one per GPU) is the conservative setting; raise it only on measured evidence
# from scripts/bench_concurrency.py. Qwen3.5 is a hybrid model, so vLLM disables
# prefix caching and every ReAct step re-prefills the whole conversation --
# SAGE measured a 3-5x per-episode slowdown from over-subscribing this.
ARM_CONCURRENCY="${MEMSYS_ARM_CONCURRENCY:-2}"
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

#!/usr/bin/env bash
# Run the ScienceWorld amount-of-experience chain: evolve 25 -> 50 -> 100,
# each leg evaluated on the same frozen 100 `test` tasks.
#
#   bash scripts/drive_scienceworld_chain.sh minimal25 minimal50 minimal100
#   MEMSYS_MAX_MODEL_LEN=131072 bash scripts/drive_scienceworld_chain.sh full25 full50 full100
#
# The legs are STRICTLY ORDERED: 50 resumes the store 25 wrote and 100 resumes
# 50, so a leg cannot start until its predecessor finished. Idempotent by leg --
# a leg whose every arm already has a summary.json is skipped, so relaunching
# after a kill resumes where it died instead of redoing finished work.
#
# `full` legs reach 100 episodes with batch induction on, whose prompt grows
# with the store; SpreadsheetBench hit a 32768 context wall doing exactly this
# (RESULTS_SPREADSHEETBENCH.md, "Batch induction outgrows the context window").
# ScienceWorld stores are smaller, but the servers this study runs are already
# at 131072, so the check is cheap insurance rather than a prediction.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
S=/gpfs/radev/scratch/cohan/jw3278
OUT_ROOT="$S/memsys_results/scienceworld"
MODES=("$@")
(( ${#MODES[@]} )) || { echo "usage: $0 <mode> [<mode> ...]" >&2; exit 2; }

ARMS=(${MEMSYS_ARMS:-none raw reflection rule skill})
export MEMSYS_EVAL_WORKERS="${MEMSYS_EVAL_WORKERS:-4}"

tag_of() {
  case "$1" in
    minimal25|full25)   echo "${1%25}_e25" ;;
    minimal50|full50)   echo "${1%50}_e50" ;;
    minimal100|full100) echo "${1%100}_e100" ;;
    *) echo "" ;;
  esac
}
policy_of() { case "$1" in full*) echo full ;; *) echo minimal ;; esac; }

for m in "${MODES[@]}"; do
  [[ -n "$(tag_of "$m")" ]] || { echo "[sw] FATAL: unknown mode $m" >&2; exit 2; }
done

for p in 8000 8001; do
  curl -sf -m 5 -o /dev/null "localhost:$p/health" \
    || { echo "[sw] FATAL: no vLLM on :$p (run scripts/serve_qwen.sh)" >&2; exit 1; }
done

for m in "${MODES[@]}"; do
  tag="$(tag_of "$m")"; policy="$(policy_of "$m")"
  done_all=1
  for arm in "${ARMS[@]}"; do
    [[ -f "$OUT_ROOT/$tag/${arm}_${policy}/summary.json" ]] || done_all=0
  done
  if (( done_all )); then
    echo "[sw] SKIP $m -- every arm in $tag already has a summary.json"
    continue
  fi
  echo "[sw] $(date +%F\ %T) === START $m -> $tag ==="
  bash "$REPO/scripts/run_scienceworld_sweep.sh" "$m"
  rc=$?
  echo "[sw] $(date +%F\ %T) === DONE  $m rc=$rc ==="
  # A failed leg poisons every leg after it: the next one would resume a store
  # that was never written, or a stale one from an earlier attempt.
  (( rc == 0 )) || { echo "[sw] FATAL: $m failed; later legs would resume a bad store" >&2; exit "$rc"; }
done

echo "[sw] all modes complete: ${MODES[*]}"

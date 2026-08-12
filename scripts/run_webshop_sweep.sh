#!/usr/bin/env bash
# The WebShop Memory Content sweep: every arm x every write policy.
#
#   bash scripts/run_webshop_sweep.sh smoke     # 2 evolve / 4 eval, all arms -- validates plumbing
#   bash scripts/run_webshop_sweep.sh minimal   # 50 evolve / 100 eval, WritePolicy.minimal()
#   bash scripts/run_webshop_sweep.sh full      # 50 evolve / 100 eval, WritePolicy.full()
#
# Prerequisites, both of which this script checks rather than assumes:
#   bash scripts/serve_qwen.sh 0 8000 --background   # vLLM, GPU 0
#   bash scripts/serve_qwen.sh 1 8001 --background   # vLLM, GPU 1
#   bash scripts/serve_webshop.sh 2 full             # env servers on :7000, :7001
#
# Arms are dispatched two at a time, one per vLLM server. Within an arm the
# evolving phase is sequential by construction; only the frozen evaluation fans
# out, and it does so across threads because the environment is out of process.
set -uo pipefail

MODE="${1:-smoke}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
S=/gpfs/radev/scratch/cohan/jw3278

PY="${MEMSYS_PY:-$S/envs/memsys-alfworld/bin/python}"
# Keep the sentence-transformers cache off HOME (at its inode quota).
export HF_HOME="${HF_HOME:-$S/.hf_home}"
export TOKENIZERS_PARALLELISM=false
OUT_ROOT="${MEMSYS_RESULTS_ROOT:-$S/memsys_results}/webshop"
MODEL="${MEMSYS_MODEL:-Qwen/Qwen3.5-9B}"
URL_A="${MEMSYS_URL_A:-http://localhost:8000/v1}"
URL_B="${MEMSYS_URL_B:-http://localhost:8001/v1}"
WS_A="${MEMSYS_WS_A:-http://localhost:7000}"
WS_B="${MEMSYS_WS_B:-http://localhost:7001}"

EVOLVE_MANIFEST="$REPO/manifests/webshop_evolve_train_50_seed42.json"
EVAL_MANIFEST="$REPO/manifests/webshop_eval_test_100_seed42.json"

# `minimal100` / `full100` continue an existing 50-episode run rather than
# repeating it: each arm resumes its own store.jsonl and evolves the *next* 50
# tasks of the same seeded permutation. That is what makes the "evolve on 50 vs
# 100" column a comparison of amount of experience -- the alternative, running
# 100 from scratch, would repeat 50 episodes of compute per arm and would also
# not reuse the memory those episodes produced.
RESUME_ROOT=""; EVOLVE_OFFSET=0
case "$MODE" in
  smoke)
    POLICIES=(full); EVOLVE_LIMIT=2; EVAL_LIMIT=4; WORKERS=2
    EMBEDDER="${MEMSYS_EMBEDDER:-hashing}"; TAG=smoke ;;
  minimal|full)
    POLICIES=("$MODE"); EVOLVE_LIMIT=0; EVAL_LIMIT=0; WORKERS="${MEMSYS_WORKERS:-8}"
    EMBEDDER="${MEMSYS_EMBEDDER:-st}"; TAG="$MODE" ;;
  minimal100|full100)
    POLICIES=("${MODE%100}"); EVOLVE_LIMIT=0; EVAL_LIMIT=0; WORKERS="${MEMSYS_WORKERS:-8}"
    EMBEDDER="${MEMSYS_EMBEDDER:-st}"; TAG="${MODE%100}_e100"
    EVOLVE_MANIFEST="$REPO/manifests/webshop_evolve_train_50to100_seed42.json"
    RESUME_ROOT="$OUT_ROOT/${MODE%100}"; EVOLVE_OFFSET=50 ;;
  *) echo "usage: $0 {smoke|minimal|full|minimal100|full100}" >&2; exit 2 ;;
esac

ARMS=(${MEMSYS_ARMS:-none raw reflection rule skill})

for url in "$URL_A" "$URL_B"; do
  curl -sf -m 5 -o /dev/null "${url%/v1}/health" \
    || { echo "[sweep] FAILED: no vLLM at $url (run scripts/serve_qwen.sh)"; exit 1; }
done
for url in "$WS_A" "$WS_B"; do
  curl -sf -m 5 -o /dev/null "$url/health" \
    || { echo "[sweep] FAILED: no WebShop server at $url (run scripts/serve_webshop.sh)"; exit 1; }
done
mkdir -p "$OUT_ROOT"

run_arm() {  # $1=arm $2=policy $3=vllm_url
  local arm="$1" policy="$2" url="$3"
  local out="$OUT_ROOT/$TAG/${arm}_${policy}"
  local log="$out.log"
  mkdir -p "$(dirname "$log")"

  # The "none" arm builds no memory system, so WritePolicy cannot reach it: a
  # minimal and a full run would be the same 100 episodes twice. Run it once and
  # reuse it as the shared baseline for every policy.
  #
  # The eval size is in the directory name on purpose. Without it the 4-task
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
  local evolve_args=(--evolve-manifest "$EVOLVE_MANIFEST" --evolve-limit "$EVOLVE_LIMIT")
  if [[ -n "$RESUME_ROOT" ]]; then
    local prior="$RESUME_ROOT/${arm}_${policy}/store.jsonl"
    [[ -f "$prior" ]] || { echo "[sweep] FAILED: no store to resume at $prior"; return 1; }
    evolve_args+=(--resume-store "$prior" --evolve-step-offset "$EVOLVE_OFFSET")
  fi
  # `none` has no memory, so its evaluation is identical at 50 and 100 evolving
  # episodes; it is reused, never re-run.
  [[ "$arm" == "none" ]] && evolve_args=()
  echo "[sweep] START $arm/$policy -> $url  (log: $log)"
  "$PY" "$REPO/scripts/run_webshop.py" \
    --arm "$arm" --policy "$policy" \
    "${evolve_args[@]}" \
    --eval-manifest "$EVAL_MANIFEST" --eval-limit "$EVAL_LIMIT" \
    --server "$WS_A" --server "$WS_B" --out "$out" \
    --model "$MODEL" --agent-base-url "$url" \
    --embedder "$EMBEDDER" --eval-workers "$WORKERS" \
    > "$log" 2>&1
  local rc=$?
  echo "[sweep] DONE  $arm/$policy rc=$rc -> $out"
  if [[ "$arm" == "none" && $rc -eq 0 ]]; then
    mkdir -p "$OUT_ROOT/$TAG/none_${policy}"
    cp -f "$out"/* "$OUT_ROOT/$TAG/none_${policy}"/
  fi
  return $rc
}

# How many arms run at once, round-robined across the two vLLM servers. Qwen3.5
# is a hybrid model, so vLLM disables prefix caching and every ReAct step
# re-prefills the whole conversation -- SAGE measured a 3-5x per-episode
# slowdown from over-subscribing this.
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

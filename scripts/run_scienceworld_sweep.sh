#!/usr/bin/env bash
# The ScienceWorld Memory Content sweep: every arm x every write policy.
#
#   bash scripts/run_scienceworld_sweep.sh smoke      # 2 evolve / 2 eval, all arms
#   bash scripts/run_scienceworld_sweep.sh minimal25  # 25 evolve / 100 eval, fresh store
#   bash scripts/run_scienceworld_sweep.sh minimal50  # continue each store to 50
#   bash scripts/run_scienceworld_sweep.sh minimal100 # ... and on to 100
#   bash scripts/run_scienceworld_sweep.sh full25 | full50 | full100
#
# Prerequisites:
#   bash scripts/setup_scienceworld.sh
#   $SW_PY scripts/build_scienceworld_manifests.py --out manifests \
#       --evolve-count 100 --eval-count 100 --seed 42
#   $SW_PY scripts/build_scienceworld_manifests.py --evolve-skip 25 --evolve-count 50 --no-eval
#   $SW_PY scripts/build_scienceworld_manifests.py --evolve-skip 50 --evolve-count 100 --no-eval
#   bash scripts/serve_qwen.sh 0 8000 --background
#   bash scripts/serve_qwen.sh 1 8001 --background
#
# **The 25 / 50 / 100 points are one chain, not three independent runs.** 50
# resumes the store 25 left behind and evolves positions [25,50) of the same
# seeded round-robin; 100 resumes 50 and evolves [50,100). So the three points
# cost 100 evolving episodes per arm rather than 175, and they compare *amount
# of experience* rather than three unrelated task draws. `--evolve-step-offset`
# carries the Evolver's step counter so `full`'s every-25-episode batch
# induction fires where an uninterrupted 100-episode run would have fired it.
#
# **A separate conda env.** ScienceWorld is a Scala simulator behind py4j and
# needs its own env (with a JDK in it); it is NOT installed in the ALFWorld env
# that the other sweeps use. MEMSYS_PY therefore defaults differently here.
#
# Concurrency is a memory question as much as a CPU one: every eval worker owns
# a JVM, so the node carries ARM_CONCURRENCY x EVAL_WORKERS simulators at once.
# The defaults (4 x 4 = 16) are sized for the 16-core allocations this study
# uses; drop MEMSYS_EVAL_WORKERS first on anything smaller.
set -uo pipefail

MODE="${1:-smoke}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
S=/gpfs/radev/scratch/cohan/jw3278

PY="${MEMSYS_PY:-$S/envs/memsys-scienceworld/bin/python}"
export HF_HOME="${HF_HOME:-$S/.hf_home}"
export TOKENIZERS_PARALLELISM=false
OUT_ROOT="${MEMSYS_RESULTS_ROOT:-$S/memsys_results}/scienceworld"
MODEL="${MEMSYS_MODEL:-Qwen/Qwen3.5-9B}"
URL_A="${MEMSYS_URL_A:-http://localhost:8000/v1}"
URL_B="${MEMSYS_URL_B:-http://localhost:8001/v1}"

ARM_CONCURRENCY="${MEMSYS_ARM_CONCURRENCY:-4}"
EVAL_WORKERS="${MEMSYS_EVAL_WORKERS:-4}"

# --- writer model (the Memory Writing Model axis) -------------------------
# Unset, the writer is the agent's own local Qwen on the same vLLM server.
WRITER_ARGS=()
[[ -n "${MEMSYS_WRITER_MODEL:-}"       ]] && WRITER_ARGS+=(--writer-model "$MEMSYS_WRITER_MODEL")
[[ -n "${MEMSYS_WRITER_BASE_URL:-}"    ]] && WRITER_ARGS+=(--writer-base-url "$MEMSYS_WRITER_BASE_URL")
[[ -n "${MEMSYS_WRITER_API:-}"         ]] && WRITER_ARGS+=(--writer-api "$MEMSYS_WRITER_API")
[[ -n "${MEMSYS_WRITER_API_KEY_ENV:-}" ]] && WRITER_ARGS+=(--writer-api-key-env "$MEMSYS_WRITER_API_KEY_ENV")
[[ -n "${MEMSYS_WRITER_MAX_TOKENS:-}"  ]] && WRITER_ARGS+=(--writer-max-tokens "$MEMSYS_WRITER_MAX_TOKENS")

EVAL_MANIFEST="$REPO/manifests/scienceworld_eval_test_100_seed42.json"
EVOLVE_MANIFEST="$REPO/manifests/scienceworld_evolve_train_100_seed42.json"

RESUME_ROOT=""; EVOLVE_OFFSET=0
case "$MODE" in
  smoke)
    POLICIES=(full); EVOLVE_LIMIT=2; EVAL_LIMIT=2
    EMBEDDER="${MEMSYS_EMBEDDER:-hashing}"; TAG=smoke ;;
  # `--evolve-limit 25` truncates the 100-task manifest, and the runner slices it
  # as a prefix, so these 25 tasks are literally the first 25 episodes of the
  # 100-task order. No extra manifest, and 25 / 50 / 100 stay one nested chain.
  minimal25|full25)
    POLICIES=("${MODE%25}"); EVOLVE_LIMIT=25; EVAL_LIMIT=0
    EMBEDDER="${MEMSYS_EMBEDDER:-st}"; TAG="${MODE%25}_e25" ;;
  # The lowest point of the same nested chain -- the first 10 tasks of the
  # 100-task order, by the same prefix truncation.
  minimal10|full10)
    POLICIES=("${MODE%10}"); EVOLVE_LIMIT=10; EVAL_LIMIT=0
    EMBEDDER="${MEMSYS_EMBEDDER:-st}"; TAG="${MODE%10}_e10" ;;
  minimal50|full50)
    POLICIES=("${MODE%50}"); EVOLVE_LIMIT=0; EVAL_LIMIT=0
    EMBEDDER="${MEMSYS_EMBEDDER:-st}"; TAG="${MODE%50}_e50"
    EVOLVE_MANIFEST="$REPO/manifests/scienceworld_evolve_train_25to50_seed42.json"
    RESUME_ROOT="$OUT_ROOT/${MODE%50}_e25"; EVOLVE_OFFSET=25 ;;
  minimal100|full100)
    POLICIES=("${MODE%100}"); EVOLVE_LIMIT=0; EVAL_LIMIT=0
    EMBEDDER="${MEMSYS_EMBEDDER:-st}"; TAG="${MODE%100}_e100"
    EVOLVE_MANIFEST="$REPO/manifests/scienceworld_evolve_train_50to100_seed42.json"
    RESUME_ROOT="$OUT_ROOT/${MODE%100}_e50"; EVOLVE_OFFSET=50 ;;
  *) echo "usage: $0 {smoke|{minimal,full}{10,25,50,100}}" >&2; exit 2 ;;
esac

ARMS=(${MEMSYS_ARMS:-none raw reflection rule skill})
TAG="${TAG}${MEMSYS_TAG_SUFFIX:-}"
[[ -n "$RESUME_ROOT" ]] && RESUME_ROOT="${RESUME_ROOT}${MEMSYS_TAG_SUFFIX:-}"
[[ -n "$RESUME_ROOT" ]] && echo "[sweep] resuming stores from $RESUME_ROOT (offset $EVOLVE_OFFSET)"

for url in "$URL_A" "$URL_B"; do
  curl -sf -m 5 -o /dev/null "${url%/v1}/health" \
    || { echo "[sweep] FAILED: no vLLM at $url (run scripts/serve_qwen.sh)"; exit 1; }
done
for m in "$EVOLVE_MANIFEST" "$EVAL_MANIFEST"; do
  [[ -f "$m" ]] || { echo "[sweep] FAILED: missing $m (run scripts/build_scienceworld_manifests.py)"; exit 1; }
done
# A green `import scienceworld` is not enough: the simulator only fails when it
# tries to launch a JVM, and that failure (`FileNotFoundError: 'java'`) surfaces
# from inside py4j once per arm, several minutes in.
#
# PYTHONPATH, because `memsys` is not installed into the env -- the runners get
# it from their own `sys.path.insert(repo_root)`, which a `-c` snippet has no
# equivalent of. Without this the check imports nothing and reports a broken JVM
# from any cwd but the repo root, which is exactly what a batch job has (a job's
# cwd is wherever `sbatch` ran). Stderr goes to the log for the same reason:
# swallowing it turned a one-line ModuleNotFoundError into a wrong diagnosis.
PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}" "$PY" -c "
from memsys.adapters.scienceworld import ScienceWorldEnvironment
e = ScienceWorldEnvironment(max_steps=5); e._ensure_env(); e.close()
" \
  || { echo "[sweep] FAILED: cannot boot a ScienceWorld JVM with $PY (run scripts/setup_scienceworld.sh)"; exit 1; }
mkdir -p "$OUT_ROOT"

run_arm() {  # $1=arm $2=policy $3=vllm_url
  local arm="$1" policy="$2" url="$3"
  local out="$OUT_ROOT/$TAG/${arm}_${policy}"
  local log="$out.log"
  mkdir -p "$(dirname "$log")"

  # The "none" arm builds no memory system, so WritePolicy cannot reach it: run
  # it once and reuse it as the shared baseline for every policy and every leg.
  # The eval size is in the directory name so a 2-task smoke baseline can never
  # be mistaken for the 100-task reference.
  local shared="$OUT_ROOT/_baseline_none_e${EVAL_LIMIT}"
  if [[ "$arm" == "none" ]]; then
    if [[ -f "$shared/summary.json" ]]; then
      echo "[sweep] REUSE none baseline from $shared"
      mkdir -p "$out" && cp -f "$shared"/* "$out"/ && return 0
    fi
    out="$shared"; log="$shared.log"
  fi
  local evolve_args=(--evolve-manifest "$EVOLVE_MANIFEST" --evolve-limit "$EVOLVE_LIMIT")
  if [[ -n "$RESUME_ROOT" && "$arm" != "none" ]]; then
    local prior="$RESUME_ROOT/${arm}_${policy}/store.jsonl"
    [[ -f "$prior" ]] || { echo "[sweep] FAILED: no store to resume at $prior"; return 1; }
    evolve_args+=(--resume-store "$prior" --evolve-step-offset "$EVOLVE_OFFSET")
  fi
  # `none` has nothing to evolve; the flag is dropped so the intent is visible
  # in config.json rather than being a silent no-op.
  [[ "$arm" == "none" ]] && evolve_args=()
  echo "[sweep] START $arm/$policy -> $url  (log: $log)"
  "$PY" "$REPO/scripts/run_scienceworld.py" \
    --arm "$arm" --policy "$policy" \
    "${evolve_args[@]}" \
    --eval-manifest "$EVAL_MANIFEST" --eval-limit "$EVAL_LIMIT" \
    --out "$out" \
    --model "$MODEL" --agent-base-url "$url" \
    ${WRITER_ARGS+"${WRITER_ARGS[@]}"} \
    --embedder "$EMBEDDER" --eval-workers "$EVAL_WORKERS" \
    > "$log" 2>&1
  local rc=$?
  echo "[sweep] DONE  $arm/$policy rc=$rc -> $out"
  if [[ "$arm" == "none" && $rc -eq 0 ]]; then
    mkdir -p "$OUT_ROOT/$TAG/none_${policy}"
    cp -f "$out"/* "$OUT_ROOT/$TAG/none_${policy}"/
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

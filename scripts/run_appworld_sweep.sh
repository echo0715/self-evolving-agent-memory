#!/usr/bin/env bash
# The AppWorld Memory Content sweep: every arm x every write policy.
#
#   bash scripts/run_appworld_sweep.sh smoke      # 2 evolve / 4 eval, all arms
#   bash scripts/run_appworld_sweep.sh minimal    # 50 evolve / 100 eval, WritePolicy.minimal()
#   bash scripts/run_appworld_sweep.sh full       # 50 evolve / 100 eval, WritePolicy.full()
#   bash scripts/run_appworld_sweep.sh minimal100 # continue each store to 100 evolve episodes
#   bash scripts/run_appworld_sweep.sh full100
#
# Prerequisites:
#   bash scripts/serve_qwen.sh 0 8000 --background
#   bash scripts/serve_qwen.sh 1 8001 --background
#   bash scripts/serve_appworld.sh 16
#
# **Servers are partitioned across concurrent arms, not shared.** This is the one
# thing that differs structurally from the WebShop sweep, and getting it wrong is
# silent. An AppWorld server hosts a single module-level `world`, and each arm
# process runs its own lease pool -- two arm processes handed the same URL would
# each believe they held it exclusively, and the second `/initialize` would
# replace the first's world mid-episode. The result is not a crash: it is an
# episode evaluated against a task it never ran. So arm i gets ports
# [BASE + i*SERVERS_PER_ARM, BASE + (i+1)*SERVERS_PER_ARM).
#
# Sizing: evolving is sequential *within* an arm, so it is the wall-clock floor
# (~50 tasks x ~3 min). Running four arms at once is what shortens the sweep;
# more servers per arm only speeds the evaluation phase.
set -uo pipefail

MODE="${1:-smoke}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
S=/gpfs/radev/scratch/cohan/jw3278

PY="${MEMSYS_PY:-$S/envs/memsys-alfworld/bin/python}"
export HF_HOME="${HF_HOME:-$S/.hf_home}"
export TOKENIZERS_PARALLELISM=false
OUT_ROOT="${MEMSYS_RESULTS_ROOT:-$S/memsys_results}/appworld"
MODEL="${MEMSYS_MODEL:-Qwen/Qwen3.5-9B}"
URL_A="${MEMSYS_URL_A:-http://localhost:8000/v1}"
URL_B="${MEMSYS_URL_B:-http://localhost:8001/v1}"

# --- writer model (the Memory Writing Model axis) -------------------------
# Unset, the writer is the agent's own local Qwen on the same vLLM server. Set
# MEMSYS_WRITER_MODEL to vary the writer alone; the actor stays on $MODEL so any
# delta is attributable to what was written and not to who acted. The Perplexity
# gateway that serves openai/gpt-5.6-* speaks only the Responses API, so
# MEMSYS_WRITER_API=responses is not optional there.
WRITER_ARGS=()
[[ -n "${MEMSYS_WRITER_MODEL:-}"       ]] && WRITER_ARGS+=(--writer-model "$MEMSYS_WRITER_MODEL")
[[ -n "${MEMSYS_WRITER_BASE_URL:-}"    ]] && WRITER_ARGS+=(--writer-base-url "$MEMSYS_WRITER_BASE_URL")
[[ -n "${MEMSYS_WRITER_API:-}"         ]] && WRITER_ARGS+=(--writer-api "$MEMSYS_WRITER_API")
[[ -n "${MEMSYS_WRITER_API_KEY_ENV:-}" ]] && WRITER_ARGS+=(--writer-api-key-env "$MEMSYS_WRITER_API_KEY_ENV")
[[ -n "${MEMSYS_WRITER_REASONING:-}"   ]] && WRITER_ARGS+=(--writer-reasoning-effort "$MEMSYS_WRITER_REASONING")
[[ -n "${MEMSYS_WRITER_MAX_TOKENS:-}"  ]] && WRITER_ARGS+=(--writer-max-tokens "$MEMSYS_WRITER_MAX_TOKENS")

APPWORLD_BASE_PORT="${APPWORLD_BASE_PORT:-9000}"
SERVERS_PER_ARM="${MEMSYS_SERVERS_PER_ARM:-4}"
ARM_CONCURRENCY="${MEMSYS_ARM_CONCURRENCY:-4}"

EVOLVE_MANIFEST="$REPO/manifests/appworld_evolve_train_50_seed42.json"
EVAL_MANIFEST="$REPO/manifests/appworld_eval_test_normal_100_seed42.json"

# `minimal100` / `full100` continue an existing 50-episode run rather than
# repeating it: each arm resumes its own store.jsonl and evolves the *next* 50
# tasks of the same seeded permutation, so the "evolve on 50 vs 100" column is a
# comparison of amount of experience. Running 100 from scratch would instead
# repeat 50 episodes of compute and throw away the memory they produced.
#
# The continuation manifest spills into `dev`: train holds 90 tasks and 50 are
# already spent, so positions [50, 100) are 40 train + 10 dev. `dev` is disjoint
# from `test_normal`, so the evaluation set is untouched.
RESUME_ROOT=""; EVOLVE_OFFSET=0
case "$MODE" in
  smoke)
    POLICIES=(full); EVOLVE_LIMIT=2; EVAL_LIMIT=4
    EMBEDDER="${MEMSYS_EMBEDDER:-hashing}"; TAG=smoke ;;
  minimal|full)
    POLICIES=("$MODE"); EVOLVE_LIMIT=0; EVAL_LIMIT=0
    EMBEDDER="${MEMSYS_EMBEDDER:-st}"; TAG="$MODE" ;;
  # A *smaller* evolving budget, and the cheapest possible one to state exactly:
  # `--evolve-limit 25` truncates the same 50-task manifest, and run_appworld.py
  # slices it as a prefix, so these 25 tasks are literally the first 25 episodes
  # of the 50-task run in the same order. No new manifest, and 25 / 50 / 100 stay
  # one nested sequence rather than three independent draws.
  minimal25|full25)
    POLICIES=("${MODE%25}"); EVOLVE_LIMIT=25; EVAL_LIMIT=0
    EMBEDDER="${MEMSYS_EMBEDDER:-st}"; TAG="${MODE%25}_e25" ;;
  # 20 distinct tasks x K epochs -- the repetition axis, which this benchmark did
  # not have at all. `--evolve-limit 20` truncates the same 50-task manifest, so
  # these are the first 20 tasks of the one seeded permutation and epoch 1 nests
  # inside the 25 / 50 / 100 sequence rather than being an independent draw.
  #
  # 20 x 5 = 100 evolving episodes, the same budget as the `*100` chain (100
  # distinct tasks) -- so the pair separates task diversity from repetition at a
  # fixed episode count. They are comparable because both evaluate the same
  # frozen `test_normal` 100, which nothing here writes.
  #
  # Worth stating for this benchmark in particular: at 25 episodes the `skill`
  # arm made only 3 writer calls, because it writes from successful episodes and
  # AppWorld's evolve-time success rate is low. Repetition is the axis most
  # likely to change that -- the agent gets five attempts at each task -- which
  # is also why a flat epoch curve here would be an informative result rather
  # than a null one.
  #
  # Epoch 1 is `minimal20`; epoch K > 1 is `minimal20_xK`, resuming epoch K-1's
  # store and carrying the step offset so `full`'s batch induction fires where an
  # uninterrupted 100-episode run would have fired it. Epoch 1 starts from an
  # EMPTY store -- a new chain, not a continuation of the 50-task one.
  minimal20|full20)
    POLICIES=("${MODE%20}"); EVOLVE_LIMIT=20; EVAL_LIMIT="${MEMSYS_EVAL_LIMIT:-0}"
    EMBEDDER="${MEMSYS_EMBEDDER:-st}"; TAG="${MODE%20}_e20" ;;
  minimal20_x[2-9]|full20_x[2-9])
    EPOCH="${MODE##*_x}"; EPOL="${MODE%%20_x*}"
    POLICIES=("$EPOL"); EVOLVE_LIMIT=20; EVAL_LIMIT="${MEMSYS_EVAL_LIMIT:-0}"
    EMBEDDER="${MEMSYS_EMBEDDER:-st}"; TAG="${EPOL}_e20_x${EPOCH}"
    PREV_TAG="${EPOL}_e20"
    if (( EPOCH > 2 )); then PREV_TAG="${EPOL}_e20_x$((EPOCH-1))"; fi
    RESUME_ROOT="$OUT_ROOT/$PREV_TAG"; EVOLVE_OFFSET=$(( 20 * (EPOCH - 1) )) ;;
  minimal100|full100)
    POLICIES=("${MODE%100}"); EVOLVE_LIMIT=0; EVAL_LIMIT=0
    EMBEDDER="${MEMSYS_EMBEDDER:-st}"; TAG="${MODE%100}_e100"
    EVOLVE_MANIFEST="$REPO/manifests/appworld_evolve_train+dev_50to100_seed42.json"
    RESUME_ROOT="$OUT_ROOT/${MODE%100}"; EVOLVE_OFFSET=50 ;;
  *) echo "usage: $0 {smoke|minimal|full|{minimal,full}{25,100}|{minimal,full}20[_x{2..9}]}" >&2; exit 2 ;;
esac

ARMS=(${MEMSYS_ARMS:-none raw reflection rule skill})

# A run that differs only in the writer model must not land in the directory the
# Qwen-writer run owns -- same mode, same manifests, different independent
# variable. MEMSYS_TAG_SUFFIX keeps it in its own tree, and the resume root
# carries it too so a `*100` leg continues the stores its OWN writer wrote
# rather than silently inheriting the Qwen chain's.
TAG="${TAG}${MEMSYS_TAG_SUFFIX:-}"
[[ -n "$RESUME_ROOT" ]] && RESUME_ROOT="${RESUME_ROOT}${MEMSYS_TAG_SUFFIX:-}"
echo "[sweep] mode=$MODE tag=$TAG writer=${MEMSYS_WRITER_MODEL:-<agent model>}${RESUME_ROOT:+ resume=$RESUME_ROOT offset=$EVOLVE_OFFSET}"

for url in "$URL_A" "$URL_B"; do
  curl -sf -m 5 -o /dev/null "${url%/v1}/health" \
    || { echo "[sweep] FAILED: no vLLM at $url (run scripts/serve_qwen.sh)"; exit 1; }
done
NEEDED=$(( ARM_CONCURRENCY * SERVERS_PER_ARM ))
for ((i = 0; i < NEEDED; i++)); do
  port=$((APPWORLD_BASE_PORT + i))
  curl -sf -m 5 -o /dev/null "http://localhost:$port/" \
    || { echo "[sweep] FAILED: no AppWorld server at :$port (need $NEEDED; run scripts/serve_appworld.sh $NEEDED)"; exit 1; }
done
mkdir -p "$OUT_ROOT"

run_arm() {  # $1=arm $2=policy $3=vllm_url $4=slot index
  local arm="$1" policy="$2" url="$3" slot="$4"
  local out="$OUT_ROOT/$TAG/${arm}_${policy}"
  local log="$out.log"
  mkdir -p "$(dirname "$log")"

  local server_args=()
  for ((k = 0; k < SERVERS_PER_ARM; k++)); do
    server_args+=(--server "http://localhost:$((APPWORLD_BASE_PORT + slot * SERVERS_PER_ARM + k))")
  done

  # The "none" arm builds no memory system, so WritePolicy cannot reach it: run
  # it once and reuse it as the shared baseline. The eval size is in the
  # directory name so a 4-task smoke baseline can never be mistaken for the
  # 100-task reference.
  local shared="$OUT_ROOT/_baseline_none_e${EVAL_LIMIT}"
  if [[ "$arm" == "none" ]]; then
    if [[ -f "$shared/summary.json" ]]; then
      echo "[sweep] REUSE none baseline from $shared"
      mkdir -p "$out" && cp -f "$shared"/* "$out"/ && return 0
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
  echo "[sweep] START $arm/$policy -> $url  servers=slot$slot  (log: $log)"
  "$PY" "$REPO/scripts/run_appworld.py" \
    --arm "$arm" --policy "$policy" \
    "${evolve_args[@]}" \
    --eval-manifest "$EVAL_MANIFEST" --eval-limit "$EVAL_LIMIT" \
    "${server_args[@]}" --out "$out" \
    --model "$MODEL" --agent-base-url "$url" \
    ${WRITER_ARGS+"${WRITER_ARGS[@]}"} \
    --embedder "$EMBEDDER" \
    --experiment-name "memsys_${arm}_${TAG}" \
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
    slot=$(( i % ARM_CONCURRENCY ))
    run_arm "$arm" "$policy" "${URLS[$(( i % ${#URLS[@]} ))]}" "$slot" &
    i=$((i+1))
    if (( i % ARM_CONCURRENCY == 0 )); then wait; fi
  done
  wait
done

echo "[sweep] all arms complete for: ${POLICIES[*]}"
"$PY" "$REPO/scripts/summarize.py" --root "$OUT_ROOT/$TAG" || true

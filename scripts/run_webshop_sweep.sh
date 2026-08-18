#!/usr/bin/env bash
# The WebShop Memory Content sweep: every arm x every write policy.
#
#   bash scripts/run_webshop_sweep.sh smoke     # 2 evolve / 4 eval, all arms -- validates plumbing
#   bash scripts/run_webshop_sweep.sh minimal   # 50 evolve / 100 eval, WritePolicy.minimal()
#   bash scripts/run_webshop_sweep.sh full      # 50 evolve / 100 eval, WritePolicy.full()
#   bash scripts/run_webshop_sweep.sh minimal25 # the first 25 of those same tasks
#   bash scripts/run_webshop_sweep.sh full25
#   bash scripts/run_webshop_sweep.sh minimal100 # continue each store to 100 evolve episodes
#   bash scripts/run_webshop_sweep.sh full100
#   bash scripts/run_webshop_sweep.sh minimal150 # ... and on to 150
#   bash scripts/run_webshop_sweep.sh full150
#   bash scripts/run_webshop_sweep.sh full_x2    # the *same* 50 tasks a second time
#   bash scripts/run_webshop_sweep.sh full_x3    # ... and a third
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

# --- writer model (the Memory Writing Model axis) -------------------------
# Unset, the writer is the agent's own local Qwen on the same vLLM server, which
# is what every WebShop result before 2026-08-16 used. Set MEMSYS_WRITER_MODEL to
# vary the writer alone; the actor stays on $MODEL so any delta is attributable
# to what got written. A remote writer needs its own base URL, wire protocol and
# key -- the Perplexity gateway serving openai/gpt-5.6-* speaks only the
# Responses API, so MEMSYS_WRITER_API=responses is not optional there.
#
# MEMSYS_TAG_SUFFIX keeps such a run in its own directory. Without it a gpt-5.6
# sweep would overwrite the Qwen-writer results it exists to be compared against.
WRITER_ARGS=()
[[ -n "${MEMSYS_WRITER_MODEL:-}"       ]] && WRITER_ARGS+=(--writer-model "$MEMSYS_WRITER_MODEL")
[[ -n "${MEMSYS_WRITER_BASE_URL:-}"    ]] && WRITER_ARGS+=(--writer-base-url "$MEMSYS_WRITER_BASE_URL")
[[ -n "${MEMSYS_WRITER_API:-}"         ]] && WRITER_ARGS+=(--writer-api "$MEMSYS_WRITER_API")
[[ -n "${MEMSYS_WRITER_API_KEY_ENV:-}" ]] && WRITER_ARGS+=(--writer-api-key-env "$MEMSYS_WRITER_API_KEY_ENV")
[[ -n "${MEMSYS_WRITER_REASONING:-}"   ]] && WRITER_ARGS+=(--writer-reasoning-effort "$MEMSYS_WRITER_REASONING")
[[ -n "${MEMSYS_WRITER_MAX_TOKENS:-}"  ]] && WRITER_ARGS+=(--writer-max-tokens "$MEMSYS_WRITER_MAX_TOKENS")

EVOLVE_MANIFEST="$REPO/manifests/webshop_evolve_train_50_seed42.json"
EVAL_MANIFEST="$REPO/manifests/webshop_eval_test_100_seed42.json"

# `minimal100` / `full100` continue an existing 50-episode run rather than
# repeating it: each arm resumes its own store.jsonl and evolves the *next* 50
# tasks of the same seeded permutation. That is what makes the "evolve on 50 vs
# 100" column a comparison of amount of experience -- the alternative, running
# 100 from scratch, would repeat 50 episodes of compute per arm and would also
# not reuse the memory those episodes produced.
RESUME_ROOT=""; EVOLVE_OFFSET=0; EVOLVE_EXTRA=()
case "$MODE" in
  smoke)
    POLICIES=(full); EVOLVE_LIMIT=2; EVAL_LIMIT=4; WORKERS=2
    EMBEDDER="${MEMSYS_EMBEDDER:-hashing}"; TAG=smoke ;;
  minimal|full)
    POLICIES=("$MODE"); EVOLVE_LIMIT=0; EVAL_LIMIT=0; WORKERS="${MEMSYS_WORKERS:-8}"
    EMBEDDER="${MEMSYS_EMBEDDER:-st}"; TAG="$MODE" ;;
  # A *smaller* evolving budget, and the cheapest possible one to state exactly:
  # `--evolve-limit 25` truncates the same 50-task manifest, and run_webshop.py
  # slices it as a prefix, so these 25 tasks are literally the first 25 episodes
  # of the 50-task run in the same order. No new manifest, and 25 / 50 / 100 /
  # 150 stay one nested sequence rather than four independent draws. Matches the
  # `*25` modes in run_sweep.sh, run_appworld_sweep.sh and
  # run_spreadsheetbench_sweep.sh.
  minimal25|full25)
    POLICIES=("${MODE%25}"); EVOLVE_LIMIT=25; EVAL_LIMIT=0; WORKERS="${MEMSYS_WORKERS:-8}"
    EMBEDDER="${MEMSYS_EMBEDDER:-st}"; TAG="${MODE%25}_e25" ;;
  minimal100|full100)
    POLICIES=("${MODE%100}"); EVOLVE_LIMIT=0; EVAL_LIMIT=0; WORKERS="${MEMSYS_WORKERS:-8}"
    EMBEDDER="${MEMSYS_EMBEDDER:-st}"; TAG="${MODE%100}_e100"
    EVOLVE_MANIFEST="$REPO/manifests/webshop_evolve_train_50to100_seed42.json"
    RESUME_ROOT="$OUT_ROOT/${MODE%100}"; EVOLVE_OFFSET=50 ;;
  # The third leg, [100, 150). Unlike SpreadsheetBench's, this one needs no
  # overflow split and no group filter: `train` holds 10,587 selectable goals
  # against the 150 spent here, and the evaluation set is drawn from `test`, so
  # the two are disjoint by split rather than by construction. The builder still
  # verifies it -- it exits non-zero if any goal index appears in both.
  minimal150|full150)
    POLICIES=("${MODE%150}"); EVOLVE_LIMIT=0; EVAL_LIMIT=0; WORKERS="${MEMSYS_WORKERS:-8}"
    EMBEDDER="${MEMSYS_EMBEDDER:-st}"; TAG="${MODE%150}_e150"
    EVOLVE_MANIFEST="$REPO/manifests/webshop_evolve_train_100to150_seed42.json"
    RESUME_ROOT="$OUT_ROOT/${MODE%150}_e100"; EVOLVE_OFFSET=100 ;;
  # `_x2` / `_x3` are the *repetition* axis, not the amount-of-experience one:
  # the SAME 50 tasks in the SAME frozen order (the default manifest above is
  # deliberately left in place), run again over the store the previous pass left
  # behind. Everything that differs between epoch k and k+1 is memory state.
  # Note what that implies for retrieval -- on epoch 2 the nearest neighbour of
  # a task is usually the agent's own epoch-1 memory of *that same task*, which
  # for `raw` is a near-verbatim replay of its own trajectory. That is the
  # phenomenon under test, and it is why these runs are not comparable to
  # `*100` / `*150`, where the later tasks are new. Evaluation is unchanged (the
  # frozen `test` 100), so nothing leaks into the test set.
  #
  # `_x3` therefore reaches 150 evolving episodes over 50 distinct tasks, which
  # is the same episode budget as the `*150` chain over 150 distinct tasks --
  # holding episodes fixed and varying only task diversity is what makes
  # repetition and diversity separable on this benchmark.
  minimal_x2|full_x2)
    POLICIES=("${MODE%_x2}"); EVOLVE_LIMIT=0; EVAL_LIMIT=0; WORKERS="${MEMSYS_WORKERS:-8}"
    EMBEDDER="${MEMSYS_EMBEDDER:-st}"; TAG="$MODE"
    RESUME_ROOT="$OUT_ROOT/${MODE%_x2}"; EVOLVE_OFFSET=50 ;;
  minimal_x3|full_x3)
    POLICIES=("${MODE%_x3}"); EVOLVE_LIMIT=0; EVAL_LIMIT=0; WORKERS="${MEMSYS_WORKERS:-8}"
    EMBEDDER="${MEMSYS_EMBEDDER:-st}"; TAG="$MODE"
    RESUME_ROOT="$OUT_ROOT/${MODE%_x3}_x2"; EVOLVE_OFFSET=100 ;;
  # The *success-budget* axis: keep evolving until 100 episodes have succeeded,
  # and let only those 100 reach the memory system. Failed episodes are run --
  # the agent still attempts them, and they still cost wall-clock -- but they are
  # discarded before `observe`, so they produce no writer call, no utility
  # signal, and do not advance the induction cadence.
  #
  # Every other mode holds the number of *attempts* fixed and lets the amount of
  # successful experience float. This one inverts that. WebShop's evolving
  # success rate is roughly 20-35%, so the task cost is ~300-500 and differs per
  # arm; the 600-task manifest is sized to cover the worst case, and the runner
  # warns loudly rather than silently finishing short if it does not.
  #
  # This is a fresh chain from an empty store, not a continuation. The 600-task
  # pool is a superset of all three 50-task legs, but the builder sorts within
  # each count, so the order is its own -- these runs are comparable to the
  # others through the shared frozen evaluation, not through a shared prefix.
  minimal_ok100|full_ok100)
    POLICIES=("${MODE%_ok100}"); EVOLVE_LIMIT=0; EVAL_LIMIT=0; WORKERS="${MEMSYS_WORKERS:-8}"
    EMBEDDER="${MEMSYS_EMBEDDER:-st}"; TAG="$MODE"
    EVOLVE_MANIFEST="$REPO/manifests/webshop_evolve_train_600_seed42.json"
    EVOLVE_EXTRA=(--evolve-until-successes 100 --write-only-on-success) ;;
  *) echo "usage: $0 {smoke|minimal|full|{minimal,full}{25,100,150}|{minimal,full}_x{2,3}|{minimal,full}_ok100}" >&2; exit 2 ;;
esac

ARMS=(${MEMSYS_ARMS:-none raw reflection rule skill})
# Applied after the mode set TAG, so a writer-model variant lands beside the
# baseline run rather than on top of it. RESUME_ROOT carries it too: a gpt-5.6
# continuation leg must reload the gpt-5.6 store, not the Qwen-writer one it
# sits beside.
TAG="${TAG}${MEMSYS_TAG_SUFFIX:-}"
[[ -n "$RESUME_ROOT" ]] && RESUME_ROOT="${RESUME_ROOT}${MEMSYS_TAG_SUFFIX:-}"

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
  local evolve_args=(--evolve-manifest "$EVOLVE_MANIFEST" --evolve-limit "$EVOLVE_LIMIT"
                     ${EVOLVE_EXTRA+"${EVOLVE_EXTRA[@]}"})
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
    ${WRITER_ARGS+"${WRITER_ARGS[@]}"} \
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

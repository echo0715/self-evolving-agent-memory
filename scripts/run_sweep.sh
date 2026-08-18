#!/usr/bin/env bash
# The ALFWorld Memory Content sweep: every arm x every write policy.
#
#   bash scripts/run_sweep.sh smoke     # 2 evolve / 2 eval, all arms, validates plumbing
#   bash scripts/run_sweep.sh minimal   # 50 evolve / 100 eval, WritePolicy.minimal()
#   bash scripts/run_sweep.sh full      # 50 evolve / 100 eval, WritePolicy.full()
#   bash scripts/run_sweep.sh full100   # the *next* 50 tasks, resuming each store
#   bash scripts/run_sweep.sh full150   # ... and the 50 after that (-> 150 total)
#   bash scripts/run_sweep.sh full_x2   # the *same* 50 tasks a second time
#   bash scripts/run_sweep.sh full_x3   # ... and a third
#   bash scripts/run_sweep.sh minimal30 # 30 evolve / 100 eval, fresh store
#   bash scripts/run_sweep.sh minimal30_x2 # the *same* 30 again, ... up to _x9
#                                          # (30 x 5 = 150 episodes over 30 tasks)
#   bash scripts/run_sweep.sh full75    # 75 evolve / 100 eval, fresh store
#   bash scripts/run_sweep.sh full75_x2 # the *same* 75 again (-> 150 episodes)
#   bash scripts/run_sweep.sh full_succ100 # evolve until 100 episodes SUCCEED,
#                                          # writing only those
#   bash scripts/run_sweep.sh full_fail100 # evolve until 100 episodes FAIL,
#                                          # writing only those
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

# --- writer model (the Memory Writing Model axis) -------------------------
# Unset, the writer is the agent's own local Qwen on the same vLLM server, which
# is what every result before 2026-08-16 used. Set MEMSYS_WRITER_MODEL to vary
# the writer alone; the actor stays on $MODEL so any delta is attributable to
# what got written. A remote writer needs its own base URL, wire protocol and
# key -- the Perplexity gateway serving openai/gpt-5.6-* speaks only the
# Responses API, so MEMSYS_WRITER_API=responses is not optional there.
#
# MEMSYS_TAG_SUFFIX keeps such a run in its own directory. Without it a
# gpt-5.6 sweep would overwrite the Qwen-writer results it exists to be
# compared against.
WRITER_ARGS=()
[[ -n "${MEMSYS_WRITER_MODEL:-}"       ]] && WRITER_ARGS+=(--writer-model "$MEMSYS_WRITER_MODEL")
[[ -n "${MEMSYS_WRITER_BASE_URL:-}"    ]] && WRITER_ARGS+=(--writer-base-url "$MEMSYS_WRITER_BASE_URL")
[[ -n "${MEMSYS_WRITER_API:-}"         ]] && WRITER_ARGS+=(--writer-api "$MEMSYS_WRITER_API")
[[ -n "${MEMSYS_WRITER_API_KEY_ENV:-}" ]] && WRITER_ARGS+=(--writer-api-key-env "$MEMSYS_WRITER_API_KEY_ENV")
[[ -n "${MEMSYS_WRITER_REASONING:-}"   ]] && WRITER_ARGS+=(--writer-reasoning-effort "$MEMSYS_WRITER_REASONING")
[[ -n "${MEMSYS_WRITER_MAX_TOKENS:-}"  ]] && WRITER_ARGS+=(--writer-max-tokens "$MEMSYS_WRITER_MAX_TOKENS")

EVOLVE_MANIFEST="$REPO/manifests/evolve_train_50_seed42.json"
EVAL_MANIFEST="$REPO/manifests/eval_valid_unseen_100_seed42.json"

# `minimal100` / `full100` continue an existing 50-episode run: each arm resumes
# its own store.jsonl and evolves the *next* 50 tasks of the same seeded
# permutation, making the "evolve on 50 vs 100" column a comparison of amount of
# experience rather than two independent runs.
RESUME_ROOT=""; EVOLVE_OFFSET=0; EVOLVE_EXTRA=()
case "$MODE" in
  smoke)
    POLICIES=(full); EVOLVE_LIMIT=2; EVAL_LIMIT=2; WORKERS=2
    EMBEDDER="${MEMSYS_EMBEDDER:-hashing}"; TAG=smoke ;;
  minimal|full)
    POLICIES=("$MODE"); EVOLVE_LIMIT=0; EVAL_LIMIT=0; WORKERS="${MEMSYS_WORKERS:-4}"
    EMBEDDER="${MEMSYS_EMBEDDER:-st}"; TAG="$MODE" ;;
  # A *smaller* evolving budget, and the cheapest possible one to state exactly:
  # `--evolve-limit 25` truncates the same 50-task manifest, and every runner
  # slices it as a prefix, so these 25 tasks are literally the first 25 episodes
  # of the 50-task run in the same order. No new manifest, and 25 / 50 / 100 /
  # 150 stay one nested sequence rather than four independent draws.
  minimal25|full25)
    POLICIES=("${MODE%25}"); EVOLVE_LIMIT=25; EVAL_LIMIT=0; WORKERS="${MEMSYS_WORKERS:-4}"
    EMBEDDER="${MEMSYS_EMBEDDER:-st}"; TAG="${MODE%25}_e25" ;;
  # The lowest point of the same nested chain: the first 10 tasks of the one
  # seeded permutation, by the same prefix truncation as `*25`. At 10 episodes
  # `full`'s batch induction (every 25) never fires, so `full10` differs from
  # `minimal10` only in the utility/deletion machinery.
  minimal10|full10)
    POLICIES=("${MODE%10}"); EVOLVE_LIMIT=10; EVAL_LIMIT=0; WORKERS="${MEMSYS_WORKERS:-4}"
    EMBEDDER="${MEMSYS_EMBEDDER:-st}"; TAG="${MODE%10}_e10" ;;
  # 30 distinct tasks x K epochs -- the repetition axis at a task count the
  # hardcoded 50x3 and 75x2 chains do not cover. `--evolve-limit 30` truncates
  # the same 50-task manifest, so these are the first 30 tasks of the one seeded
  # permutation and epoch 1 nests inside the 25 / 50 / 100 / 150 sequence rather
  # than being an independent draw.
  #
  # 30 x 5 = 150 evolving episodes: the same budget as `*150` (150 distinct
  # tasks), `_x3` (50 x 3) and `75_x2` (75 x 2), and a fourth point on the
  # diversity-vs-repetition curve. The four are comparable only because all of
  # them evaluate the same frozen `valid_unseen` 100, which nothing here writes.
  #
  # Epoch 1 is `minimal30`; epoch K > 1 is `minimal30_xK`, which resumes epoch
  # K-1's store and carries the step offset so `full`'s batch induction fires
  # where an uninterrupted 150-episode run would have fired it. Epoch 1 starts
  # from an EMPTY store -- this is a new chain, not a continuation of the
  # 50-task one.
  minimal30|full30)
    POLICIES=("${MODE%30}"); EVOLVE_LIMIT=30; EVAL_LIMIT="${MEMSYS_EVAL_LIMIT:-0}"; WORKERS="${MEMSYS_WORKERS:-4}"
    EMBEDDER="${MEMSYS_EMBEDDER:-st}"; TAG="${MODE%30}_e30" ;;
  minimal30_x[2-9]|full30_x[2-9])
    EPOCH="${MODE##*_x}"; EPOL="${MODE%%30_x*}"
    POLICIES=("$EPOL"); EVOLVE_LIMIT=30; EVAL_LIMIT="${MEMSYS_EVAL_LIMIT:-0}"; WORKERS="${MEMSYS_WORKERS:-4}"
    EMBEDDER="${MEMSYS_EMBEDDER:-st}"; TAG="${EPOL}_e30_x${EPOCH}"
    PREV_TAG="${EPOL}_e30"
    if (( EPOCH > 2 )); then PREV_TAG="${EPOL}_e30_x$((EPOCH-1))"; fi
    RESUME_ROOT="$OUT_ROOT/$PREV_TAG"; EVOLVE_OFFSET=$(( 30 * (EPOCH - 1) )) ;;
  minimal100|full100)
    POLICIES=("${MODE%100}"); EVOLVE_LIMIT=0; EVAL_LIMIT=0; WORKERS="${MEMSYS_WORKERS:-4}"
    EMBEDDER="${MEMSYS_EMBEDDER:-st}"; TAG="${MODE%100}_e100"
    EVOLVE_MANIFEST="$REPO/manifests/evolve_train_50to100_seed42.json"
    RESUME_ROOT="$OUT_ROOT/${MODE%100}"; EVOLVE_OFFSET=50 ;;
  # The third leg of the same chain: positions [100,150) of the one seeded
  # permutation, resuming the `*_e100` stores. Same axis as `*100` (amount of
  # experience), so 50 / 100 / 150 are three points on one curve -- and, like
  # `*100`, a resumption rather than an independent 150-task run.
  minimal150|full150)
    POLICIES=("${MODE%150}"); EVOLVE_LIMIT=0; EVAL_LIMIT=0; WORKERS="${MEMSYS_WORKERS:-4}"
    EMBEDDER="${MEMSYS_EMBEDDER:-st}"; TAG="${MODE%150}_e150"
    EVOLVE_MANIFEST="$REPO/manifests/evolve_train_100to150_seed42.json"
    RESUME_ROOT="$OUT_ROOT/${MODE%150}_e100"; EVOLVE_OFFSET=100 ;;
  # `_x2` / `_x3` are the *repetition* axis, not the amount-of-experience one:
  # the SAME 50 tasks in the SAME frozen order, run again over the store the
  # previous pass left behind. Everything that differs between epoch k and k+1
  # is memory state. Note what that implies for retrieval -- on epoch 2 the
  # nearest neighbour of a task is usually the agent's own epoch-1 memory of
  # *that same task*, which for `raw` is a near-verbatim replay of its own
  # trajectory. That is the phenomenon under test, and it is why these runs are
  # not comparable to `*100`, where the second 50 tasks are new. Evaluation is
  # unchanged (held-out `valid_unseen`), so nothing leaks into the test set.
  minimal_x2|full_x2)
    POLICIES=("${MODE%_x2}"); EVOLVE_LIMIT=0; EVAL_LIMIT=0; WORKERS="${MEMSYS_WORKERS:-4}"
    EMBEDDER="${MEMSYS_EMBEDDER:-st}"; TAG="$MODE"
    RESUME_ROOT="$OUT_ROOT/${MODE%_x2}"; EVOLVE_OFFSET=50 ;;
  minimal_x3|full_x3)
    POLICIES=("${MODE%_x3}"); EVOLVE_LIMIT=0; EVAL_LIMIT=0; WORKERS="${MEMSYS_WORKERS:-4}"
    EMBEDDER="${MEMSYS_EMBEDDER:-st}"; TAG="$MODE"
    RESUME_ROOT="$OUT_ROOT/${MODE%_x3}_x2"; EVOLVE_OFFSET=100 ;;
  # 75 x 2 epochs = 150 evolving episodes, the same budget as the `*150` chain
  # (150 distinct tasks) and as `_x3` (50 tasks x 3). Holding episodes fixed and
  # varying only how many *distinct* tasks they cover is what makes diversity vs
  # repetition separable; the three points are only comparable because all three
  # evaluate the same frozen `valid_unseen` 100. Epoch 1 starts from an empty
  # store -- it is a new chain, not a continuation of the 50-task one.
  minimal75|full75)
    POLICIES=("${MODE%75}"); EVOLVE_LIMIT=0; EVAL_LIMIT=0; WORKERS="${MEMSYS_WORKERS:-4}"
    EMBEDDER="${MEMSYS_EMBEDDER:-st}"; TAG="${MODE%75}_e75"
    EVOLVE_MANIFEST="$REPO/manifests/evolve_train_75_seed42.json" ;;
  minimal75_x2|full75_x2)
    POLICIES=("${MODE%75_x2}"); EVOLVE_LIMIT=0; EVAL_LIMIT=0; WORKERS="${MEMSYS_WORKERS:-4}"
    EMBEDDER="${MEMSYS_EMBEDDER:-st}"; TAG="${MODE%75_x2}_e75_x2"
    EVOLVE_MANIFEST="$REPO/manifests/evolve_train_75_seed42.json"
    RESUME_ROOT="$OUT_ROOT/${MODE%75_x2}_e75"; EVOLVE_OFFSET=75 ;;
  # The *outcome-budget* axis (RESULTS_ALFWORLD.md §10, §11). Both legs run from
  # an EMPTY store over `evolve_train_600_seed42.json` -- the same seeded
  # permutation as every other manifest here, so positions [0,50), [50,100),
  # [100,150), [150,200) and [0,300) are the earlier files verbatim -- and both
  # stop on an outcome count rather than a task count. The number of tasks spent
  # therefore differs per arm, which is the point: the arms are matched on what
  # the memory *learned from*, not on what the agent was shown.
  #
  #   succ100 -- 100 successful episodes, failures discarded before observe()
  #   fail100 -- 100 FAILED episodes, successes discarded before observe()
  #
  # `fail100` is the mirror image: the store is built out of failure evidence
  # alone, so under `full` every utility signal reaching the deletion machinery
  # is negative and nothing can ever be confirmed by a task going right.
  # Failures cost more tasks to collect than successes on ALFWorld (evolve-time
  # success rate is 55-75%), which is why the manifest is 600 rather than 300 --
  # the runner warns loudly rather than finishing short if even that runs out.
  #
  # `raw` is excluded from both by default (MEMSYS_ARMS): RawTrajectorySystem
  # keeps only best_success(), so success-only is a no-op for it and
  # failure-only leaves it with an empty store, i.e. the `none` baseline.
  minimal_succ100|full_succ100|minimal_fail100|full_fail100)
    POLICIES=("${MODE%_*}"); EVOLVE_LIMIT=0; EVAL_LIMIT=0; WORKERS="${MEMSYS_WORKERS:-4}"
    EMBEDDER="${MEMSYS_EMBEDDER:-st}"; TAG="$MODE"
    ARMS_DEFAULT="none reflection rule skill"
    case "$MODE" in
      *_succ100)
        EVOLVE_MANIFEST="$REPO/manifests/evolve_train_300_seed42.json"
        EVOLVE_EXTRA=(--evolve-until-successes 100 --success-only-writes) ;;
      *_fail100)
        EVOLVE_MANIFEST="$REPO/manifests/evolve_train_600_seed42.json"
        EVOLVE_EXTRA=(--evolve-until-failures 100 --failure-only-writes) ;;
    esac ;;
  *) echo "usage: $0 {smoke|minimal|full|{minimal,full}{10,25,100,150}|{minimal,full}_x{2,3}|{minimal,full}30[_x{2..9}]|{minimal,full}75[_x2]|{minimal,full}_{succ,fail}100}" >&2; exit 2 ;;
esac

ARMS=(${MEMSYS_ARMS:-${ARMS_DEFAULT:-none raw reflection rule skill}})
# Applied after the mode set TAG, so a writer-model variant lands beside the
# baseline run rather than on top of it.
#
# RESUME_ROOT takes the same suffix, and that is not cosmetic. Every
# continuation mode names the chain it resumes by reconstructing the previous
# leg's directory from $MODE, which is the *unsuffixed* name -- so without this
# a gpt-5.6 `minimal100` would load the QWEN run's store.jsonl and evolve on top
# of it. The result is a store whose first 50 entries were written by one model
# and whose next 50 were written by another, with nothing in the summary to say
# so: `writer_model` records only the model of the run that just finished.
TAG="${TAG}${MEMSYS_TAG_SUFFIX:-}"
[[ -n "$RESUME_ROOT" ]] && RESUME_ROOT="${RESUME_ROOT}${MEMSYS_TAG_SUFFIX:-}"
# Print the *resolved* chain. A wrong RESUME_ROOT does not fail -- it finds a
# real store belonging to a different chain and evolves on top of it -- so the
# path belongs in the log, not just in the code.
[[ -n "$RESUME_ROOT" ]] && echo "[sweep] resuming stores from $RESUME_ROOT (offset $EVOLVE_OFFSET)"
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
  local evolve_args=(--evolve-manifest "$EVOLVE_MANIFEST" --evolve-limit "$EVOLVE_LIMIT"
                     ${EVOLVE_EXTRA+"${EVOLVE_EXTRA[@]}"})
  if [[ -n "$RESUME_ROOT" && "$arm" != "none" ]]; then
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
    ${WRITER_ARGS+"${WRITER_ARGS[@]}"} \
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

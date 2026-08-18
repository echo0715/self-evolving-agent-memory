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
#   bash scripts/run_spreadsheetbench_sweep.sh minimal200 # ... and on to 200
#   bash scripts/run_spreadsheetbench_sweep.sh full200    # needs MEMSYS_MAX_MODEL_LEN=131072
#   bash scripts/run_spreadsheetbench_sweep.sh minimal_x2  # the *same* 50 tasks a second time
#   bash scripts/run_spreadsheetbench_sweep.sh minimal_x3  # ... and a third
#   bash scripts/run_spreadsheetbench_sweep.sh minimal75   # 75 evolve / 100 eval, fresh store
#   bash scripts/run_spreadsheetbench_sweep.sh minimal75_x2 # the *same* 75 again (-> 150 episodes)
#   bash scripts/run_spreadsheetbench_sweep.sh minimal_fail100 # evolve until 100 episodes
#   bash scripts/run_spreadsheetbench_sweep.sh full_fail100    # FAIL, writing only those
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

# --- writer model (the Memory Writing Model axis) -------------------------
# Unset, the writer is the agent's own local Qwen on the same vLLM server, which
# is what every SpreadsheetBench result before 2026-08-17 used. Set
# MEMSYS_WRITER_MODEL to vary the writer alone; the actor stays on $MODEL so any
# delta is attributable to what got written. A remote writer needs its own base
# URL, wire protocol and key -- the Perplexity gateway serving openai/gpt-5.6-*
# speaks only the Responses API, so MEMSYS_WRITER_API=responses is not optional
# there.
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
RESUME_ROOT=""; EVOLVE_OFFSET=0; EVOLVE_EXTRA=()
case "$MODE" in
  smoke)
    POLICIES=(full); EVOLVE_LIMIT=2; EVAL_LIMIT=4
    EMBEDDER="${MEMSYS_EMBEDDER:-hashing}"; TAG=smoke ;;
  minimal|full)
    POLICIES=("$MODE"); EVOLVE_LIMIT=0; EVAL_LIMIT=0
    EMBEDDER="${MEMSYS_EMBEDDER:-st}"; TAG="$MODE" ;;
  # A *smaller* evolving budget, and the cheapest possible one to state exactly:
  # `--evolve-limit 25` truncates the same 50-task manifest, and
  # run_spreadsheetbench.py slices it as a prefix, so these 25 tasks are
  # literally the first 25 episodes of the 50-task run in the same order. No new
  # manifest, and 25 / 50 / 100 / 150 / 200 stay one nested sequence rather than
  # five independent draws.
  minimal25|full25)
    POLICIES=("${MODE%25}"); EVOLVE_LIMIT=25; EVAL_LIMIT=0
    EMBEDDER="${MEMSYS_EMBEDDER:-st}"; TAG="${MODE%25}_e25" ;;
  # The lowest point of that same nested sequence: the first 10 tasks of the
  # train-only 50-task manifest, so it touches no eval-set source workbook.
  minimal10|full10)
    POLICIES=("${MODE%10}"); EVOLVE_LIMIT=10; EVAL_LIMIT=0
    EMBEDDER="${MEMSYS_EMBEDDER:-st}"; TAG="${MODE%10}_e10" ;;
  # 30 distinct tasks x K epochs -- the repetition axis at a task count the
  # hardcoded 50x3 and 75x2 chains do not cover. `--evolve-limit 30` truncates
  # the same 50-task manifest, so these are the first 30 tasks of the one seeded
  # permutation and epoch 1 nests inside the 25 / 50 / 100 / 150 / 200 sequence
  # rather than being an independent draw. Because the manifest is the train-only
  # 50-task one, no leg of this chain touches the eval set's source workbooks.
  #
  # 30 x 5 = 150 evolving episodes: the same budget as `*150` (150 distinct
  # tasks), `_x3` (50 x 3) and `75_x2` (75 x 2), and a fourth point on the
  # diversity-vs-repetition curve. The four are comparable only because all of
  # them evaluate the same frozen `test` 100, which nothing here writes.
  #
  # Epoch 1 is `minimal30`; epoch K > 1 is `minimal30_xK`, which resumes epoch
  # K-1's store and carries the step offset so `full`'s batch induction fires
  # where an uninterrupted 150-episode run would have fired it. Epoch 1 starts
  # from an EMPTY store -- a new chain, not a continuation of the 50-task one.
  minimal30|full30)
    POLICIES=("${MODE%30}"); EVOLVE_LIMIT=30; EVAL_LIMIT="${MEMSYS_EVAL_LIMIT:-0}"
    EMBEDDER="${MEMSYS_EMBEDDER:-st}"; TAG="${MODE%30}_e30" ;;
  minimal30_x[2-9]|full30_x[2-9])
    EPOCH="${MODE##*_x}"; EPOL="${MODE%%30_x*}"
    POLICIES=("$EPOL"); EVOLVE_LIMIT=30; EVAL_LIMIT="${MEMSYS_EVAL_LIMIT:-0}"
    EMBEDDER="${MEMSYS_EMBEDDER:-st}"; TAG="${EPOL}_e30_x${EPOCH}"
    PREV_TAG="${EPOL}_e30"
    if (( EPOCH > 2 )); then PREV_TAG="${EPOL}_e30_x$((EPOCH-1))"; fi
    RESUME_ROOT="$OUT_ROOT/$PREV_TAG"; EVOLVE_OFFSET=$(( 30 * (EPOCH - 1) )) ;;
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
  # The fourth leg, [150, 200). Every one of its 50 tasks comes from `test` --
  # train+val was exhausted at position 117 -- again disjoint from the frozen
  # eval set by task id *and* by source workbook, enforced by the builder. One
  # workbook (416) recurs across evolving legs (416-15 at [50,100), 416-27
  # here); that is evolve-to-evolve and does not touch the eval set.
  #
  # **Serve at 131072 for this leg.** `full`'s batch induction prompt grows with
  # the store, and it already needed 65536 at [100, 150) -- reflection/full and
  # rule/full both died against 32768 there. Starting from stores of 85 and 29
  # entries, the two inductions in this leg (steps 174 and 199) are larger
  # again. See RESULTS_SPREADSHEETBENCH.md, "Batch induction outgrows the
  # context window". `minimal` has no induction and is unaffected either way.
  minimal200|full200)
    POLICIES=("${MODE%200}"); EVOLVE_LIMIT=0; EVAL_LIMIT=0
    EMBEDDER="${MEMSYS_EMBEDDER:-st}"; TAG="${MODE%200}_e200"
    EVOLVE_MANIFEST="$REPO/manifests/spreadsheetbench_evolve_train+val+test_150to200_seed42.json"
    RESUME_ROOT="$OUT_ROOT/${MODE%200}_e150"; EVOLVE_OFFSET=150 ;;
  # The *repetition* axis, not the amount-of-experience one: the SAME 50 tasks
  # in the SAME frozen order, run again over the store the previous pass left
  # behind. Everything that differs between epoch k and k+1 is memory state.
  # Note what that implies for retrieval -- on epoch 2 the nearest neighbour of
  # a task is usually the agent's own epoch-1 memory of *that same task*, which
  # for `raw` is a near-verbatim replay of its own solution. That is the
  # phenomenon under test, and it is why these are not comparable to `*100`,
  # where the second 50 tasks are new. Evaluation is unchanged (the frozen 100
  # `test` tasks), so nothing leaks into the evaluation set.
  minimal_x2|full_x2)
    POLICIES=("${MODE%_x2}"); EVOLVE_LIMIT=0; EVAL_LIMIT=0
    EMBEDDER="${MEMSYS_EMBEDDER:-st}"; TAG="$MODE"
    RESUME_ROOT="$OUT_ROOT/${MODE%_x2}"; EVOLVE_OFFSET=50 ;;
  minimal_x3|full_x3)
    POLICIES=("${MODE%_x3}"); EVOLVE_LIMIT=0; EVAL_LIMIT=0
    EMBEDDER="${MEMSYS_EMBEDDER:-st}"; TAG="$MODE"
    RESUME_ROOT="$OUT_ROOT/${MODE%_x3}_x2"; EVOLVE_OFFSET=100 ;;
  # 75 x 2 epochs = 150 evolving episodes, the same budget as the `*150` chain
  # (150 distinct tasks) and as `_x3` (50 tasks x 3). Holding episodes fixed and
  # varying only how many *distinct* tasks they cover is what makes diversity vs
  # repetition separable; the three points are only comparable because all three
  # evaluate the same frozen `test` 100. Epoch 1 starts from an EMPTY store --
  # it is a new chain, not a continuation of the 50-task one.
  #
  # The 75-task manifest is train-only (train holds 79 selectable once the eval
  # set's source workbooks are excluded) and its positions [0,50) are the
  # 50-task manifest verbatim, [50,75) the first 25 of the [50,100)
  # continuation -- so this leg sits inside the same seeded sequence as the
  # e50/e100/e150 chain rather than beside it.
  minimal75|full75)
    POLICIES=("${MODE%75}"); EVOLVE_LIMIT=0; EVAL_LIMIT=0
    EMBEDDER="${MEMSYS_EMBEDDER:-st}"; TAG="${MODE%75}_e75"
    EVOLVE_MANIFEST="$REPO/manifests/spreadsheetbench_evolve_train_75_seed42.json" ;;
  minimal75_x2|full75_x2)
    POLICIES=("${MODE%75_x2}"); EVOLVE_LIMIT=0; EVAL_LIMIT=0
    EMBEDDER="${MEMSYS_EMBEDDER:-st}"; TAG="${MODE%75_x2}_e75_x2"
    EVOLVE_MANIFEST="$REPO/manifests/spreadsheetbench_evolve_train_75_seed42.json"
    RESUME_ROOT="$OUT_ROOT/${MODE%75_x2}_e75"; EVOLVE_OFFSET=75 ;;
  # The *outcome-budget* leg, the SpreadsheetBench twin of run_sweep.sh's
  # `{minimal,full}_fail100` (RESULTS_ALFWORLD.md §10, §11). It runs from an
  # EMPTY store over `spreadsheetbench_evolve_train+val+test_297_seed42.json` --
  # the whole selectable evolve pool, whose positions [0,50), [50,100),
  # [100,150) and [150,200) are the four manifests above verbatim -- and stops
  # on a *failure* count rather than a task count. Successful episodes are
  # discarded before the memory system observes them, so the store is built out
  # of failure evidence alone and the number of tasks spent differs per arm.
  #
  # Sizing: evolve-time success here runs 12-32%, so 100 failures costs roughly
  # 125-145 tasks and the 297-task manifest has ample headroom. The mirror leg
  # (`succ100`) is NOT offered for this benchmark for the same reason read the
  # other way: 100 successes would cost 310-830 tasks and the pool holds 297.
  #
  # `raw` is excluded by default (ARMS_DEFAULT): RawTrajectorySystem keeps only
  # best_success(), so under failure-only writes it ends with an empty store,
  # i.e. the `none` baseline with extra steps. The runner warns if it is added
  # back via MEMSYS_ARMS.
  minimal_fail100|full_fail100)
    POLICIES=("${MODE%_fail100}"); EVOLVE_LIMIT=0; EVAL_LIMIT=0
    EMBEDDER="${MEMSYS_EMBEDDER:-st}"; TAG="$MODE"
    ARMS_DEFAULT="none reflection rule skill"
    EVOLVE_MANIFEST="$REPO/manifests/spreadsheetbench_evolve_train+val+test_297_seed42.json"
    EVOLVE_EXTRA=(--evolve-until-failures 100 --failure-only-writes) ;;
  # The mirror: evolve until 100 episodes SUCCEED, discarding every failure
  # before the memory system observes it. Two things differ from `fail100`
  # beyond the sign, and both follow from success being the *rare* outcome here.
  #
  # 1. **A bigger pool, which spills outside SkillOpt's split.** At 21-24%
  #    evolve-time success, 100 successes costs 420-480 tasks; train+val+test
  #    holds 297 selectable. The manifest therefore appends `rest` -- the 512
  #    tasks of the 912 release that the 400-task Verified id split does not
  #    cover, 501 after the eval set's source workbooks are excluded. Its first
  #    297 positions are the fail100 manifest verbatim, and the frozen eval set
  #    is still disjoint by task id AND by source workbook (the builder refuses
  #    to write the manifest otherwise). What is given up is that this leg's
  #    evolving tasks are no longer all from the split the rest of the SAGE
  #    ecosystem uses; the evaluation set is untouched.
  # 2. **~3.5x the wall clock of fail100** for the same 100 written episodes,
  #    because ~78% of the tasks it walks through are thrown away.
  #
  # `raw` is excluded by default here for the opposite reason to fail100: it
  # keeps only best_success() already, so success-only writes is a no-op for it
  # and the run would duplicate the plain fixed-count leg.
  minimal_succ100|full_succ100)
    POLICIES=("${MODE%_succ100}"); EVOLVE_LIMIT=0; EVAL_LIMIT=0
    EMBEDDER="${MEMSYS_EMBEDDER:-st}"; TAG="$MODE"
    ARMS_DEFAULT="none reflection rule skill"
    EVOLVE_MANIFEST="$REPO/manifests/spreadsheetbench_evolve_train+val+test+rest_798_seed42.json"
    EVOLVE_EXTRA=(--evolve-until-successes 100 --success-only-writes) ;;
  *) echo "usage: $0 {smoke|minimal|full|minimal{10,25,100,150,200}|full{10,25,100,150,200}|"\
"{minimal,full}_x{2,3}|{minimal,full}30[_x{2..9}]|{minimal,full}75[_x2]|"\
"{minimal,full}_{fail,succ}100}" >&2; exit 2 ;;
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
echo "[sweep] mode=$MODE tag=$TAG writer=${MEMSYS_WRITER_MODEL:-<agent model>}${RESUME_ROOT:+ resume=$RESUME_ROOT offset=$EVOLVE_OFFSET}"

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
  local evolve_args=(--evolve-manifest "$EVOLVE_MANIFEST" --evolve-limit "$EVOLVE_LIMIT"
                     ${EVOLVE_EXTRA+"${EVOLVE_EXTRA[@]}"})
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
    ${WRITER_ARGS+"${WRITER_ARGS[@]}"} \
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

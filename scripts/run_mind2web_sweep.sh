#!/usr/bin/env bash
# The Mind2Web Memory Content sweep: every arm x one write policy.
#
#   bash scripts/run_mind2web_sweep.sh smoke     # 8 evolve / 24 eval steps, validates plumbing
#   bash scripts/run_mind2web_sweep.sh minimal   # 50 annotations (416 steps) / 100 annotations (838 steps)
#   bash scripts/run_mind2web_sweep.sh full      # ... same, WritePolicy.full()
#   bash scripts/run_mind2web_sweep.sh minimal100 # the *next* 50 annotations, resuming each store
#   bash scripts/run_mind2web_sweep.sh minimal150 # ... and the 50 after that (-> 150 total)
#   bash scripts/run_mind2web_sweep.sh minimal200 # ... and the 50 after that (-> 200 total)
#
# Unit note, because every count here is ambiguous otherwise: manifests are drawn
# at *annotation* level and expanded into steps, and one step is one memsys
# episode. "50" means 50 annotations = 416 evolving episodes.
#
# Arms are dispatched two at a time, one per vLLM server (GPU 0 -> :8000,
# GPU 1 -> :8001). Evolving is sequential within an arm by construction; only the
# frozen evaluation fans out, over threads (there is no environment to corrupt).
set -uo pipefail

MODE="${1:-smoke}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
S=/gpfs/radev/scratch/cohan/jw3278

PY="${MEMSYS_PY:-$S/envs/memsys-alfworld/bin/python}"
export HF_HOME="${HF_HOME:-$S/.hf_home}"
export TOKENIZERS_PARALLELISM=false
CACHE_ROOT="${MIND2WEB_CACHE:-$S/mind2web_cache}"
OUT_ROOT="${MEMSYS_RESULTS_ROOT:-$S/memsys_results}/mind2web"
MODEL="${MEMSYS_MODEL:-Qwen/Qwen3.5-9B}"
URL_A="${MEMSYS_URL_A:-http://localhost:8000/v1}"
URL_B="${MEMSYS_URL_B:-http://localhost:8001/v1}"

# --- writer model (the Memory Writing Model axis) -------------------------
# Unset, the writer is the agent's own local Qwen on the same vLLM server, which
# is what every Mind2Web result before 2026-08-16 used. Set MEMSYS_WRITER_MODEL
# to vary the writer alone; the actor stays on $MODEL so any delta is
# attributable to what got written. A remote writer needs its own base URL, wire
# protocol and key -- the Perplexity gateway serving openai/gpt-5.6-* speaks only
# the Responses API, so MEMSYS_WRITER_API=responses is not optional there.
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

EVOLVE_MANIFEST="$REPO/manifests/mind2web_evolve_train_50_seed42.json"
EVAL_MANIFEST="$REPO/manifests/mind2web_eval_test_task_100_seed42.json"
# Candidate pool depth. 50 is the paper's setting for everything except GPT-4,
# and it costs ~9 LLM calls per step here; dropping to 10 is ~3 but also lowers
# the ceiling, since a step whose ground-truth element is outside the pool is an
# automatic zero (12.2% of eval steps at top-50).
TOP_K="${MIND2WEB_TOP_K:-50}"

# `*100` / `*150` / `*200` continue an existing chain: each arm reloads its own
# store.jsonl and evolves the *next* 50 annotations of the same seeded
# permutation, so 50 / 100 / 150 / 200 are four points on one amount-of-
# experience curve rather than four independent runs. Evaluation is the same
# frozen 100 `test_task` annotations every time.
#
# The offset handed to --evolve-step-offset is in *steps*, not annotations,
# because that is the memsys episode unit here -- and it is not a constant: leg 1
# is 416 steps, later legs are however many the previous leg actually ran. It is
# therefore read from the prior run's summary.json per arm (see run_arm), never
# hard-coded.
RESUME_ROOT=""
case "$MODE" in
  smoke)
    POLICIES=(minimal); EVOLVE_LIMIT=8; EVAL_LIMIT=24; WORKERS=8
    EMBEDDER="${MEMSYS_EMBEDDER:-hashing}"; TAG=smoke ;;
  minimal|full)
    POLICIES=("$MODE"); EVOLVE_LIMIT=0; EVAL_LIMIT=0; WORKERS="${MEMSYS_WORKERS:-8}"
    EMBEDDER="${MEMSYS_EMBEDDER:-st}"; TAG="$MODE" ;;
  # The lowest point of the same chain: the first 10 annotations of the
  # 50-annotation leg, in the same order. `--evolve-limit` counts STEPS, not
  # annotations (see the unit note at the top), so the limit is the number of
  # steps those 10 annotations expand to -- 79 for the frozen seed-42 manifest.
  # It is read from the manifest rather than hard-coded: a wrong constant would
  # not fail, it would silently evolve on 9 or 11 annotations and still be
  # labelled `_e10`. Steps are laid out annotation by annotation, so the prefix
  # is exactly those 10 annotations and nothing else.
  minimal10|full10)
    POLICIES=("${MODE%10}"); EVAL_LIMIT=0; WORKERS="${MEMSYS_WORKERS:-8}"
    EMBEDDER="${MEMSYS_EMBEDDER:-st}"; TAG="${MODE%10}_e10"
    EVOLVE_LIMIT="$("$PY" - "$EVOLVE_MANIFEST" <<'EOF'
import json, sys
tasks = json.load(open(sys.argv[1]))["tasks"]
seen, n = [], 0
for s in tasks:
    a = s["annotation_id"]
    if not seen or seen[-1] != a:
        if len(seen) == 10: break
        seen.append(a)
    n += 1
assert len(seen) == 10, f"manifest holds only {len(seen)} annotations"
print(n)
EOF
)" || { echo "[sweep] FAILED: could not size the 10-annotation prefix of $EVOLVE_MANIFEST" >&2; exit 1; }
    echo "[sweep] e10 = first 10 annotations = $EVOLVE_LIMIT steps" ;;
  minimal100|full100|minimal150|full150|minimal200|full200)
    LEG="${MODE: -3}"; POL="${MODE%$LEG}"
    POLICIES=("$POL"); EVOLVE_LIMIT=0; EVAL_LIMIT=0; WORKERS="${MEMSYS_WORKERS:-8}"
    EMBEDDER="${MEMSYS_EMBEDDER:-st}"; TAG="${POL}_e${LEG}"
    EVOLVE_MANIFEST="$REPO/manifests/mind2web_evolve_train_$((LEG - 50))to${LEG}_seed42.json"
    # The suffix belongs on the resume root too: a gpt-5.6 continuation leg must
    # reload the gpt-5.6 store, not the Qwen-writer one it sits beside.
    RESUME_ROOT="$OUT_ROOT/${POL}${MEMSYS_TAG_SUFFIX:-}"
    [[ "$LEG" != "100" ]] && RESUME_ROOT="$OUT_ROOT/${POL}_e$((LEG - 50))${MEMSYS_TAG_SUFFIX:-}"
    [[ -f "$EVOLVE_MANIFEST" ]] || {
      echo "[sweep] FAILED: no manifest at $EVOLVE_MANIFEST -- build it with" >&2
      echo "  scripts/build_mind2web_manifests.py --evolve-skip $((LEG - 50)) --evolve-count $LEG --evolve-only" >&2
      exit 1; } ;;
  *) echo "usage: $0 {smoke|minimal|full|{minimal,full}{10,100,150,200}}" >&2; exit 2 ;;
esac

ARMS=(${MEMSYS_ARMS:-none raw reflection rule skill})
# Applied after the mode set TAG, so a writer-model variant lands beside the
# baseline run rather than on top of it.
TAG="${TAG}${MEMSYS_TAG_SUFFIX:-}"
mkdir -p "$OUT_ROOT"

[[ -d "$CACHE_ROOT" ]] || {
  echo "[sweep] FAILED: no step cache at $CACHE_ROOT -- run scripts/build_mind2web_manifests.py" >&2
  exit 1; }

run_arm() {  # $1=arm $2=policy $3=base_url
  local arm="$1" policy="$2" url="$3"
  local out="$OUT_ROOT/$TAG/${arm}_${policy}"
  local log="$out.log"
  mkdir -p "$(dirname "$log")"

  # The "none" arm has no memory system, so WritePolicy cannot reach it: a
  # minimal and a full run would be the same evaluation twice. Run it once and
  # reuse it as the shared baseline. The eval size and top-k are both in the
  # directory name -- reusing a top-10 baseline for a top-50 run would compute
  # every delta against the wrong reference with nothing looking wrong.
  local shared="$OUT_ROOT/_baseline_none_e${EVAL_LIMIT}_k${TOP_K}"
  if [[ "$arm" == "none" ]]; then
    if [[ -f "$shared/summary.json" ]]; then
      echo "[sweep] REUSE none baseline from $shared"
      mkdir -p "$out" && cp -f "$shared"/* "$out"/ && return 0
    fi
    out="$shared"; log="$shared.log"
  fi
  local evolve_args=(--evolve-manifest "$EVOLVE_MANIFEST" --evolve-limit "$EVOLVE_LIMIT")
  if [[ -n "$RESUME_ROOT" && "$arm" != "none" ]]; then
    local prior="$RESUME_ROOT/${arm}_${policy}"
    [[ -f "$prior/store.jsonl" ]] || {
      echo "[sweep] FAILED: no store to resume at $prior/store.jsonl"; return 1; }
    # Steps already evolved on this chain, so `batch_every` consolidation keeps
    # counting across legs instead of restarting at 0 every 50 annotations.
    local offset
    offset=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['evolve_total'])" \
             "$prior/summary.json") || {
      echo "[sweep] FAILED: cannot read evolve_total from $prior/summary.json"; return 1; }
    evolve_args+=(--resume-store "$prior/store.jsonl" --evolve-step-offset "$offset")
  fi
  # `none` has no memory: its evaluation is identical at every evolving budget,
  # so it is reused rather than re-run.
  [[ "$arm" == "none" ]] && evolve_args=()
  echo "[sweep] START $arm/$policy -> $url  (log: $log)"
  "$PY" "$REPO/scripts/run_mind2web.py" \
    --arm "$arm" --policy "$policy" \
    "${evolve_args[@]}" \
    --eval-manifest "$EVAL_MANIFEST" --eval-limit "$EVAL_LIMIT" \
    --cache-root "$CACHE_ROOT" --out "$out" \
    --model "$MODEL" --agent-base-url "$url" --top-k "$TOP_K" \
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

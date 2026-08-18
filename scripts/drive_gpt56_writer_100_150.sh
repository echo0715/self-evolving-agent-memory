#!/usr/bin/env bash
# Continue the gpt-5.6-terra writer chain to 100 and then 150 evolving tasks.
#
# Both legs are *resumptions*: each arm loads the store its previous leg left
# behind and evolves the next 50 tasks of the same seeded permutation. They are
# therefore strictly ordered -- minimal150 cannot start until minimal100 has
# written its stores -- and this script runs them in sequence, not in parallel.
#
#   minimal100 -> minimal_e100_gpt56terra   (tasks [50,100),  --evolve-step-offset 50)
#   minimal150 -> minimal_e150_gpt56terra   (tasks [100,150), --evolve-step-offset 100)
#
#   srun --jobid=<job> --overlap -N1 -n1 bash scripts/drive_gpt56_writer_100_150.sh
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
S=/gpfs/radev/scratch/cohan/jw3278
R="$S/memsys_results"

# See drive_gpt56_writer.sh: serve_qwen.sh needs $S/hf_cache, the sweep needs
# $S/.hf_home, and exporting either here points vLLM at the wrong one.
unset HF_HOME

export MEMSYS_ARMS="reflection rule skill"
export MEMSYS_WRITER_MODEL="openai/gpt-5.6-terra"
export MEMSYS_WRITER_BASE_URL="https://api.perplexity.ai/v1"
export MEMSYS_WRITER_API="responses"
export MEMSYS_WRITER_API_KEY_ENV="perplexity_api_key"
export MEMSYS_TAG_SUFFIX="_gpt56terra"

# --- 0. the stores this chain must resume from ----------------------------
# run_sweep.sh reconstructs the previous leg's directory from $MODE and appends
# MEMSYS_TAG_SUFFIX. If the suffix were ever dropped it would silently resume
# the *Qwen* chain instead, producing a store written half by one model and half
# by another with nothing in the summary to say so. Check before spending GPU.
for arm in $MEMSYS_ARMS; do
  prior="$R/minimal_gpt56terra/${arm}_minimal/store.jsonl"
  [[ -f "$prior" ]] || { echo "[drive] FATAL: no store to resume at $prior" >&2; exit 1; }
  echo "[drive] resume source $arm: $(wc -l < "$prior") rows"
done

# --- 1. servers ------------------------------------------------------------
for p in 8000 8001; do
  if curl -sf -m 5 -o /dev/null "localhost:$p/health"; then
    echo "[drive] FATAL: port $p already healthy -- another allocation owns it." >&2
    exit 1
  fi
done
echo "[drive] starting vLLM on $(hostname -s)"
bash "$REPO/scripts/serve_qwen.sh" 0 8000 --background
bash "$REPO/scripts/serve_qwen.sh" 1 8001 --background
deadline=$(( SECONDS + 1800 ))
until curl -sf -m 5 -o /dev/null localhost:8000/health && \
      curl -sf -m 5 -o /dev/null localhost:8001/health; do
  if (( SECONDS > deadline )); then
    echo "[drive] FATAL: servers not healthy after 30 min; see $S/memsys_qwen9b_gpu*.log" >&2
    exit 1
  fi
  sleep 15
done
echo "[drive] servers healthy after ${SECONDS}s"

# --- 2. the two legs, in order --------------------------------------------
# `none` and `raw` call no writer LLM, so they are copied from the Qwen chain's
# matching leg rather than re-measured; summarize.py needs `none` present to
# compute any delta at all.
seed_baselines() {  # $1 = qwen tag, $2 = gpt tag
  for arm in none raw; do
    local src="$R/$1/${arm}_minimal" dst="$R/$2/${arm}_minimal"
    [[ -d "$src" ]] || { echo "[drive] warn: no $src to copy"; continue; }
    mkdir -p "$dst" && cp -f "$src"/* "$dst"/
    cat > "$dst/REUSED_FROM_QWEN_RUN.txt" <<TXT
Copied from $src. Neither arm calls a writer LLM (none has no memory system;
raw stores trajectories verbatim), so the writer model cannot move them and
they were not re-run for the gpt-5.6-terra chain.
TXT
  done
}

for leg in minimal100 minimal150; do
  echo "[drive] === $leg ==="
  bash "$REPO/scripts/run_sweep.sh" "$leg" || { echo "[drive] FATAL: $leg failed" >&2; exit 1; }
  case "$leg" in
    minimal100) seed_baselines minimal_e100 minimal_e100_gpt56terra
                tag=minimal_e100_gpt56terra ;;
    minimal150) seed_baselines minimal_e150 minimal_e150_gpt56terra
                tag=minimal_e150_gpt56terra ;;
  esac
  # Re-summarize now that `none` is present (run_sweep.sh already ran it once,
  # before the baseline was copied in, so that table has no delta column).
  "${MEMSYS_PY:-$S/envs/memsys-alfworld/bin/python}" "$REPO/scripts/summarize.py" --root "$R/$tag" || true
done

echo "[drive] both legs complete"

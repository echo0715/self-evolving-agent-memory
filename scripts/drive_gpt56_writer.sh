#!/usr/bin/env bash
# Drive the ALFWorld `minimal` sweep with a *remote* memory writer.
#
# The actor stays Qwen3.5-9B on local vLLM; only the memory-writing model moves
# to openai/gpt-5.6-terra behind the Perplexity gateway. That gateway serves
# ONLY the Responses API (`/v1/responses`) -- its `/chat/completions` is the
# native Sonar endpoint and 400s every gateway model id -- hence
# MEMSYS_WRITER_API=responses.
#
# Runs on the compute node, e.g.
#   srun --jobid=<job> --overlap -N1 -n1 bash scripts/drive_gpt56_writer.sh
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
S=/gpfs/radev/scratch/cohan/jw3278

# Deliberately NOT exporting HF_HOME here: serve_qwen.sh reads it from the
# environment and needs $S/hf_cache (the Qwen snapshot), while the sweep needs
# $S/.hf_home (sentence-transformers). Setting it up front points vLLM at a
# cache with no model in it, which with HF_HUB_OFFLINE=1 fails seconds *after*
# "[serve] started (PID ...)" prints. run_sweep.sh exports its own value.
unset HF_HOME

# --- 1. refuse a busy port rather than launching duplicates ---------------
for p in 8000 8001; do
  if curl -sf -m 5 -o /dev/null "localhost:$p/health"; then
    echo "[drive] FATAL: port $p already healthy -- another allocation owns it." >&2
    echo "[drive] Reuse it (skip this script's serve step) or kill by recorded PID." >&2
    exit 1
  fi
done

echo "[drive] starting vLLM on $(hostname -s)"
bash "$REPO/scripts/serve_qwen.sh" 0 8000 --background
bash "$REPO/scripts/serve_qwen.sh" 1 8001 --background

# A real cold start is 4-5 minutes (19 GB load + CUDA graph capture). A pass in
# under a minute means something else is listening.
deadline=$(( SECONDS + 1800 ))
until curl -sf -m 5 -o /dev/null localhost:8000/health && \
      curl -sf -m 5 -o /dev/null localhost:8001/health; do
  if (( SECONDS > deadline )); then
    echo "[drive] FATAL: servers not healthy after 30 min; see $S/memsys_qwen9b_*gpu*.log" >&2
    exit 1
  fi
  sleep 15
done
echo "[drive] servers healthy after ${SECONDS}s"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

# --- 2. the sweep ---------------------------------------------------------
# `none` and `raw` never call a writer LLM, so their numbers are identical to
# the existing Qwen-writer `minimal` run and are not re-measured; compare
# against $S/memsys_results/minimal/{none,raw}_minimal.
export MEMSYS_ARMS="reflection rule skill"
export MEMSYS_WRITER_MODEL="openai/gpt-5.6-terra"
export MEMSYS_WRITER_BASE_URL="https://api.perplexity.ai/v1"
export MEMSYS_WRITER_API="responses"
export MEMSYS_WRITER_API_KEY_ENV="perplexity_api_key"
export MEMSYS_TAG_SUFFIX="_gpt56terra"

# A 2x2 smoke first. A dead server or an unauthenticated writer does not raise
# here -- the agent swallows connection errors and `parse_ops` turns a 401 into
# zero ops -- so both failure modes produce a complete-looking summary.json.
# The gate is therefore on the *content*: every writer arm must have made real
# writer calls and stored something.
echo "[drive] smoke test"
MEMSYS_TAG_SUFFIX="_gpt56terra_smoke" bash "$REPO/scripts/run_sweep.sh" smoke
python3 - "$S/memsys_results/smoke_gpt56terra_smoke" <<'PY' || exit 1
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
bad = []
for arm in ("reflection", "rule", "skill"):
    p = root / f"{arm}_full" / "summary.json"
    if not p.exists():
        bad.append(f"{arm}: no summary.json"); continue
    s = json.loads(p.read_text())
    u = s.get("writer_usage") or {}
    live = (s.get("store") or {}).get("n_live", 0)
    if not u.get("n_calls"):
        bad.append(f"{arm}: zero writer calls (writer unreachable or unauthenticated?)")
    elif not live:
        bad.append(f"{arm}: {u['n_calls']} writer calls but empty store (parse failure?)")
    elif s.get("writer_model") != "openai/gpt-5.6-terra":
        bad.append(f"{arm}: writer_model is {s.get('writer_model')!r}, not the gateway model")
    else:
        print(f"[smoke] {arm}: {u['n_calls']} writer calls, {live} live items, "
              f"eval {s.get('eval_success')}/{s.get('eval_n')}")
if bad:
    print("[smoke] FAILED:\n  " + "\n  ".join(bad)); sys.exit(1)
print("[smoke] OK")
PY

echo "[drive] full minimal sweep: 50 evolve / 100 eval"
bash "$REPO/scripts/run_sweep.sh" minimal
rc=$?
echo "[drive] sweep rc=$rc"
exit $rc

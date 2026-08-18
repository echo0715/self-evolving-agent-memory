#!/usr/bin/env bash
# Drive the WebShop `minimal` sweep with a *remote* memory writer.
#
# The actor stays Qwen3.5-9B on local vLLM; only the memory-writing model moves
# to openai/gpt-5.6-terra behind the Perplexity gateway, which serves ONLY the
# Responses API -- hence MEMSYS_WRITER_API=responses. WebShop sibling of
# scripts/drive_gpt56_writer.sh and scripts/drive_mind2web_gpt56_writer.sh.
#
# Unlike those two this script *reuses* healthy servers rather than refusing to
# start: on a node that has just run another benchmark the vLLM pair is already
# up, and re-serving would cost 5 minutes and a duplicate 19 GB load.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
S=/gpfs/radev/scratch/cohan/jw3278

# serve_qwen.sh needs HF_HOME=$S/hf_cache (the Qwen snapshot) while the sweep
# needs $S/.hf_home (sentence-transformers); each sets its own, so leave it unset.
unset HF_HOME

# --- 1. servers -----------------------------------------------------------
for spec in "0 8000" "1 8001"; do
  set -- $spec
  if curl -sf -m 5 -o /dev/null "localhost:$2/health"; then
    echo "[drive] vLLM :$2 already healthy, reusing"
  else
    bash "$REPO/scripts/serve_qwen.sh" "$1" "$2" --background
  fi
done
# ~2 min per env server: 5.5 GB of JSON into 1.18M product dicts, ~14 GiB each.
# Idempotent -- it skips a port that already answers /health.
bash "$REPO/scripts/serve_webshop.sh" 2 full

deadline=$(( SECONDS + 1800 ))
until curl -sf -m 5 -o /dev/null localhost:8000/health && \
      curl -sf -m 5 -o /dev/null localhost:8001/health && \
      curl -sf -m 5 -o /dev/null localhost:7000/health && \
      curl -sf -m 5 -o /dev/null localhost:7001/health; do
  if (( SECONDS > deadline )); then
    echo "[drive] FATAL: servers not healthy after 30 min; see" >&2
    echo "  $S/memsys_qwen9b_*gpu*.log and $S/memsys_webshop_srv_*.log" >&2
    exit 1
  fi
  sleep 15
done
echo "[drive] all four servers healthy after ${SECONDS}s"

# --- 2. the sweep ---------------------------------------------------------
# `none` and `raw` never call a writer LLM, so their numbers are identical to
# the existing Qwen-writer `minimal` run and are not re-measured; compare
# against $S/memsys_results/webshop/minimal/{none,raw}_minimal.
export MEMSYS_ARMS="${MEMSYS_ARMS:-reflection rule skill}"
export MEMSYS_WRITER_MODEL="${MEMSYS_WRITER_MODEL:-openai/gpt-5.6-terra}"
export MEMSYS_WRITER_BASE_URL="${MEMSYS_WRITER_BASE_URL:-https://api.perplexity.ai/v1}"
export MEMSYS_WRITER_API="${MEMSYS_WRITER_API:-responses}"
export MEMSYS_WRITER_API_KEY_ENV="${MEMSYS_WRITER_API_KEY_ENV:-perplexity_api_key}"
export MEMSYS_TAG_SUFFIX="${MEMSYS_TAG_SUFFIX:-_gpt56terra}"

# A smoke first (2 evolve / 4 eval, policy `full`). A dead server or an
# unauthenticated writer does not raise here -- the agent swallows connection
# errors and `parse_ops` turns a 401 into zero ops -- so both failure modes
# produce a complete-looking summary.json. The gate is therefore on the
# *content*. An arm that saw no successful evolving episode is exempt from both
# checks: SkillLibrary writes only from a successful rollout, and with 2
# evolving episodes at a ~25-35% success rate zero writer calls is that arm
# working, not the gateway failing.
echo "[drive] smoke test"
MEMSYS_TAG_SUFFIX="${MEMSYS_TAG_SUFFIX}_smoke" \
  bash "$REPO/scripts/run_webshop_sweep.sh" smoke
"$S/envs/memsys-alfworld/bin/python" - \
  "$S/memsys_results/webshop/smoke${MEMSYS_TAG_SUFFIX}_smoke" \
  "$MEMSYS_WRITER_MODEL" "$MEMSYS_ARMS" <<'PY' || exit 1
import json, sys
from pathlib import Path
root, want_model, arms = Path(sys.argv[1]), sys.argv[2], sys.argv[3].split()
bad = []
for arm in arms:
    p = root / f"{arm}_full" / "summary.json"   # smoke runs WritePolicy.full()
    if not p.exists():
        bad.append(f"{arm}: no summary.json"); continue
    s = json.loads(p.read_text())
    u = s.get("writer_usage") or {}
    live = (s.get("store") or {}).get("n_live", 0)
    outcomes = s.get("evolve_outcomes") or {}
    had_success = any(v for k, v in outcomes.items() if "success" in k or k == "mixed")
    if s.get("writer_model") != want_model:
        bad.append(f"{arm}: writer_model is {s.get('writer_model')!r}, not the gateway model")
    elif not u.get("n_calls"):
        if had_success:
            bad.append(f"{arm}: zero writer calls (writer unreachable or unauthenticated?)")
        else:
            print(f"[smoke] {arm}: no successful evolving episode ({outcomes}); "
                  f"zero writer calls is correct for this arm")
    elif not live:
        bad.append(f"{arm}: {u['n_calls']} writer calls but empty store (parse failure?)")
    else:
        print(f"[smoke] {arm}: {u['n_calls']} writer calls, {live} live items, "
              f"eval {s.get('eval_success')}/{s.get('eval_n')}")
if bad:
    print("[smoke] FAILED:\n  " + "\n  ".join(bad)); sys.exit(1)
print("[smoke] OK")
PY

echo "[drive] full minimal sweep: 50 evolve / 100 eval"
bash "$REPO/scripts/run_webshop_sweep.sh" minimal
rc=$?
echo "[drive] sweep rc=$rc"
exit $rc

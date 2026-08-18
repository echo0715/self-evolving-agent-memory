#!/usr/bin/env bash
# Start (or check) a pool of AppWorld environment servers.
#
#   bash scripts/serve_appworld.sh 8      # :9000 .. :9007
#   bash scripts/serve_appworld.sh stop
#
# One server per *concurrent episode*, not per machine. `appworld serve
# environment` keeps a single module-level `world`, so a process hosts exactly
# one live task; two episodes sharing a URL would execute against each other's
# world state and score against the wrong task without raising anything. The
# runner leases URLs exclusively and caps its thread pool at the pool size, so
# this number is the evaluation concurrency.
set -uo pipefail

N="${1:-8}"
S=/gpfs/radev/scratch/cohan/jw3278
PY_BIN="${APPWORLD_ENV:-$S/envs/memsys-appworld}/bin/appworld"
export APPWORLD_ROOT="${APPWORLD_ROOT:-$S/appworld_root}"
BASE_PORT="${APPWORLD_BASE_PORT:-9000}"
# Per-host pid and log names. $S is shared across nodes and these paths were
# keyed only on port, so two allocations each serving :9000 overwrote each
# other's pidfile -- after which `stop` on one node kills whatever now holds
# that pid on *this* node, which need not be an AppWorld server at all. Same fix
# serve_webshop.sh and serve_qwen.sh already carry.
H=$(hostname -s)

if [[ "$N" == "stop" ]]; then
  for pidfile in "$S"/appworld_srv_"$H"_*.pid; do
    [[ -f "$pidfile" ]] || continue
    pid=$(cat "$pidfile")
    if kill -0 "$pid" 2>/dev/null; then kill "$pid" && echo "[appworld] stopped $pid ($pidfile)"; fi
    rm -f "$pidfile"
  done
  exit 0
fi

for ((i = 0; i < N; i++)); do
  port=$((BASE_PORT + i))
  if curl -sf -m 3 -o /dev/null "http://localhost:$port/"; then
    echo "[appworld] :$port already healthy"
    continue
  fi
  log="$S/memsys_appworld_srv_${H}_$port.log"
  # --root is required: the server resolves data relative to the CWD otherwise,
  # and the dataset lives on scratch.
  nohup "$PY_BIN" serve environment --port "$port" --root "$APPWORLD_ROOT" --no-show-usage \
    > "$log" 2>&1 &
  echo $! > "$S/appworld_srv_${H}_$port.pid"
  echo "[appworld] starting :$port (pid $!, log $log)"
done

echo "[appworld] waiting for health ..."
deadline=$((SECONDS + 300))
for ((i = 0; i < N; i++)); do
  port=$((BASE_PORT + i))
  until curl -sf -m 3 -o /dev/null "http://localhost:$port/"; do
    if (( SECONDS > deadline )); then
      echo "[appworld] FAILED: :$port never became healthy; tail of its log:"
      tail -20 "$S/memsys_appworld_srv_${H}_$port.log"
      exit 1
    fi
    sleep 2
  done
done
echo "[appworld] $N servers healthy on ports $BASE_PORT..$((BASE_PORT + N - 1))"

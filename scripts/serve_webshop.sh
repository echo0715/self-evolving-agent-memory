#!/usr/bin/env bash
# Start (or check) the WebShop environment servers.
#
#   bash scripts/serve_webshop.sh 2 full     # two servers on :7000 and :7001
#   bash scripts/serve_webshop.sh stop
#
# Why several processes rather than one with more threads: each server
# serialises env access behind a single lock, because WebShop's step path runs a
# Lucene query through pyjnius, renders a Jinja template and parses it with
# BeautifulSoup over structures upstream never made concurrent. Scaling out is
# the safe axis, and it is affordable -- a server is ~14 GiB and the catalogue is
# read-only, so N of them cost N x 14 GiB and nothing else.
#
# Startup is ~2 minutes: 5.5 GB of JSON parsed into 1.18M product dicts. Leave
# them up for the whole sweep; every arm shares them.
set -uo pipefail

N="${1:-2}"
SCALE="${2:-full}"
S=/gpfs/radev/scratch/cohan/jw3278
PY="${WEBSHOP_PY:-$S/envs/memsys-webshop/bin/python}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_PORT="${WEBSHOP_BASE_PORT:-7000}"
SEED="${WEBSHOP_SEED:-42}"
# Per-host pid and log names. $S is shared across nodes and these paths were
# keyed only on port, so two allocations each serving :7000 overwrote each
# other's pidfile -- after which `stop` on one node kills whatever now holds
# that pid on *this* node, which need not be a WebShop server at all. The logs
# had the matching problem: two hosts interleaving into one file.
H=$(hostname -s)

if [[ "$N" == "stop" ]]; then
  for pidfile in "$S"/webshop_srv_"$H"_*.pid; do
    [[ -f "$pidfile" ]] || continue
    pid=$(cat "$pidfile")
    if kill -0 "$pid" 2>/dev/null; then kill "$pid" && echo "[webshop] stopped $pid ($pidfile)"; fi
    rm -f "$pidfile"
  done
  exit 0
fi

for ((i = 0; i < N; i++)); do
  port=$((BASE_PORT + i))
  if curl -sf -m 3 -o /dev/null "http://localhost:$port/health"; then
    echo "[webshop] :$port already healthy"
    continue
  fi
  log="$S/memsys_webshop_srv_${H}_$port.log"
  # --seed is not cosmetic. WebShop draws every product price from an unseeded
  # global RNG at load and samples each goal's "price lower than X dollars"
  # clause from those prices, so two servers with different seeds disagree about
  # what goal N *is*. Every server in a run must share this value, and it is
  # recorded in the manifest so run_webshop.py can refuse a mismatch.
  nohup "$PY" "$REPO/scripts/webshop_server.py" \
    --port "$port" --scale "$SCALE" --human-goals 1 --seed "$SEED" > "$log" 2>&1 &
  echo $! > "$S/webshop_srv_${H}_$port.pid"
  echo "[webshop] starting :$port (pid $!, log $log)"
done

echo "[webshop] waiting for health (corpus load is ~2 min) ..."
deadline=$((SECONDS + 900))
for ((i = 0; i < N; i++)); do
  port=$((BASE_PORT + i))
  until curl -sf -m 3 -o /dev/null "http://localhost:$port/health"; do
    if (( SECONDS > deadline )); then
      echo "[webshop] FAILED: :$port never became healthy; tail of its log:"
      tail -20 "$S/memsys_webshop_srv_${H}_$port.log"
      exit 1
    fi
    sleep 5
  done
  echo "[webshop] :$port  $(curl -s "http://localhost:$port/health")"
done

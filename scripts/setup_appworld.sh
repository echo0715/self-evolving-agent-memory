#!/usr/bin/env bash
# AppWorld environment setup (TODO item 1).
#
#   bash scripts/setup_appworld.sh
#
# AppWorld ships its own installer and dataset downloader, so this is the
# cleanest of the three benchmarks. Everything lands on scratch: home is at its
# inode quota and the shared `cohan` project group quota has been full before.
set -euo pipefail

S=/gpfs/radev/scratch/cohan/jw3278
ENV_PREFIX="${APPWORLD_ENV:-$S/envs/memsys-appworld}"
# AppWorld keeps data, per-task DBs and experiment outputs under one root.
export APPWORLD_ROOT="${APPWORLD_ROOT:-$S/appworld_root}"

export PIP_CACHE_DIR="$S/.pip-cache"
export TMPDIR="$S/tmp"
mkdir -p "$APPWORLD_ROOT" "$PIP_CACHE_DIR" "$TMPDIR"

source /gpfs/radev/apps/avx512/software/miniconda/24.3.0-miniforge/etc/profile.d/conda.sh

if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
  # appworth requires >=3.11,<4.0
  conda create -y -p "$ENV_PREFIX" python=3.12 pip
fi

"$ENV_PREFIX/bin/pip" install -q --upgrade pip
"$ENV_PREFIX/bin/pip" install -q appworld
# Installs the app source tree AppWorld's runtime executes against.
"$ENV_PREFIX/bin/appworld" install
# `--root` is REQUIRED here: it defaults to `.`, and APPWORLD_ROOT is not
# consulted by the downloader, so without it the 162 MB payload lands in
# whatever directory you happened to run from (for us: inside the git repo).
"$ENV_PREFIX/bin/appworld" download data --root "$APPWORLD_ROOT"

echo
echo "[appworld] env  : $ENV_PREFIX"
echo "[appworld] root : $APPWORLD_ROOT"
"$ENV_PREFIX/bin/python" - <<'PY'
import os
from appworld import AppWorld, load_task_ids
for split in ("train", "dev", "test_normal", "test_challenge"):
    try:
        print(f"  {split:<15} {len(load_task_ids(split))} tasks")
    except Exception as exc:
        print(f"  {split:<15} unavailable: {type(exc).__name__}: {exc}")
PY
echo "[appworld] OK"

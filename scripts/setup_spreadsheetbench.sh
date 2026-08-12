#!/usr/bin/env bash
# One-time SpreadsheetBench setup: fetch the dataset archives and add the two
# libraries the agent and the evaluator both need.
#
#   bash scripts/setup_spreadsheetbench.sh
#
# Unlike WebShop and AppWorld there is nothing to serve and no second python
# environment: SpreadsheetBench is a directory of .xlsx files plus an evaluator,
# and the evaluator is ported into memsys/adapters/spreadsheetbench.py. Setup is
# therefore just data plus `openpyxl` and `pandas`.
#
# `sys.executable` is what re-runs the agent's solution.py at scoring time, so
# the libraries must be in the *memsys* interpreter, not in some other env. A
# model told it may use pandas and handed an interpreter without it will write
# correct code that fails on import, and the episode reads as an agent error.
set -euo pipefail

S=/gpfs/radev/scratch/cohan/jw3278
PY="${MEMSYS_PY:-$S/envs/memsys-alfworld/bin/python}"
ROOT="${SPREADSHEETBENCH_ROOT_DIR:-$S/spreadsheetbench_root}"
REVISION="${SPREADSHEETBENCH_REVISION:-ab0b742b0fc95b946f212d80ac7771b5531272e4}"

export HF_HOME="${HF_HOME:-$S/hf_cache}"
export HF_HUB_DISABLE_XET=1
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$S/pip-cache}"
# scripts/serve_qwen.sh exports HF_HUB_OFFLINE=1; this script must download.
unset HF_HUB_OFFLINE

echo "[setup] interpreter: $PY"
"$PY" -c "import openpyxl, pandas" 2>/dev/null \
  || "$PY" -m pip install --quiet openpyxl pandas
"$PY" -c "import openpyxl, pandas; print(f'[setup] openpyxl {openpyxl.__version__}, pandas {pandas.__version__}')"

mkdir -p "$ROOT"
# Both archives are fetched. `all_data_912_v0.1` is what the manifests point at
# -- three test cases per task, upstream's evaluation protocol. Verified 400 is
# fetched too because SkillOpt's published id split names it as its source and
# a run may want to reproduce that setting, but it ships ONE test case per task,
# which collapses the graded score onto the strict one. Do not mix the two:
# Verified revised both instructions and golden workbooks, so an instruction
# from one archive graded against the other's answers scores the wrong task.
for FILE in spreadsheetbench_912_v0.1 spreadsheetbench_verified_400; do
  echo "[setup] fetching $FILE.tar.gz"
  TGZ=$("$PY" - "$FILE" "$REVISION" <<'EOF'
import sys
from huggingface_hub import hf_hub_download
print(hf_hub_download("KAKA22/SpreadsheetBench", f"{sys.argv[1]}.tar.gz",
                      repo_type="dataset", revision=sys.argv[2]))
EOF
)
  tar -xzf "$TGZ" -C "$ROOT"
done

echo
for d in "$ROOT"/*/; do
  [[ -f "$d/dataset.json" ]] || continue
  "$PY" - "$d" <<'EOF'
import collections, glob, json, os, sys
root = sys.argv[1]
rows = json.load(open(os.path.join(root, "dataset.json")))
cases = collections.Counter(
    len(glob.glob(os.path.join(root, r["spreadsheet_path"], "*_input.xlsx")))
    or len(glob.glob(os.path.join(root, r["spreadsheet_path"], "*_init.xlsx")))
    or len(glob.glob(os.path.join(root, r["spreadsheet_path"], "initial.xlsx")))
    for r in rows
)
types = collections.Counter(r["instruction_type"] for r in rows)
print(f"[setup] {root}\n"
      f"          {len(rows)} tasks  cases/task={dict(sorted(cases.items()))}  {dict(types)}")
EOF
done

cat <<EOF

[setup] done. Next:
  export SPREADSHEETBENCH_ROOT=$ROOT/all_data_912_v0.1
  $PY scripts/build_spreadsheetbench_manifests.py
  bash scripts/run_spreadsheetbench_sweep.sh smoke
EOF

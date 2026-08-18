#!/usr/bin/env bash
# ScienceWorld environment setup.
#
#   bash scripts/setup_scienceworld.sh
#
# ScienceWorld is a Scala simulator driven from Python over py4j: `pip install
# scienceworld` ships the JAR, and every `ScienceWorldEnv()` starts a JVM and
# talks to it on a socket. Two consequences shape everything downstream:
#
#   * **A JDK must be on PATH**, and this cluster has none system-wide. The
#     module works interactively but not inside a detached sweep, so the JDK is
#     installed *into the env* -- the env is then self-contained and a runner
#     needs no `module load`.
#   * **Each concurrent episode is a JVM**, not a thread. Sizing evaluation
#     workers is a memory question, not just a CPU one.
#
# A fresh env rather than a clone of memsys-alfworld: that env carries the
# whole alfworld + textworld + fast_downward stack, none of which ScienceWorld
# needs, and cloning it would also make this setup depend on an env that sweeps
# are actively using.
set -uo pipefail

S=/gpfs/radev/scratch/cohan/jw3278
ENV_PREFIX="${SCIENCEWORLD_ENV:-$S/envs/memsys-scienceworld}"

export PIP_CACHE_DIR="$S/.pip-cache"
export TMPDIR="$S/tmp"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"

source /gpfs/radev/apps/avx512/software/miniconda/24.3.0-miniforge/etc/profile.d/conda.sh

if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
  # 3.11 matches memsys-alfworld, so anything that works there works here.
  conda create -y -p "$ENV_PREFIX" python=3.11 pip || exit 1
fi
conda activate "$ENV_PREFIX" || exit 1

# openjdk in the env, not `module load Java`: a sweep launched with setsid does
# not inherit an interactively loaded module, and the failure surfaces as py4j
# refusing a connection rather than as "no java".
conda install -y -c conda-forge openjdk=17 || exit 1

pip install -q scienceworld || { echo "[sw] FAILED: pip install scienceworld"; exit 1; }
# What memsys itself needs at run time. Core is dependency-free by design
# (pyproject.toml); these are the three optional extras a real run uses.
pip install -q "openai>=1.30" "tiktoken>=0.7" "sentence-transformers>=3.0" \
  || { echo "[sw] FAILED: pip install memsys extras"; exit 1; }

echo "[sw] versions:"
python - <<'PY'
import subprocess, sys
import scienceworld
print("  python           ", sys.version.split()[0])
print("  scienceworld     ", getattr(scienceworld, "__version__", "?"))
print("  java             ", subprocess.run(["java", "-version"], capture_output=True,
                                            text=True).stderr.splitlines()[0])
PY

# The JAR is fetched/started on first construction, so a green pip install is
# not evidence the simulator runs. Boot one env and step it once.
echo "[sw] smoke test: boot the simulator and take one step"
python - <<'PY' || { echo "[sw] FAILED: simulator smoke test"; exit 1; }
from scienceworld import ScienceWorldEnv
env = ScienceWorldEnv("", envStepLimit=10)
names = sorted(env.get_task_names())
print(f"  {len(names)} task names, e.g. {names[:3]}")
env.load(names[0], 0, "")
obs, info = env.reset()
print(f"  reset ok: {len(obs)} chars, score={info.get('score')}")
obs, reward, done, info = env.step("look around")
print(f"  step ok: score={info.get('score')} moves={info.get('moves')}")
env.close()
PY
echo "[sw] OK -> $ENV_PREFIX"

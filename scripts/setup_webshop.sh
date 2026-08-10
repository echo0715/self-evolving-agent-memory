#!/usr/bin/env bash
# WebShop environment setup (TODO item 1).
#
#   bash scripts/setup_webshop.sh small     # 1,000 products -- fast, good for wiring
#   bash scripts/setup_webshop.sh all       # full 1.18M-product corpus (~1.5 GB download)
#
# This is the most fragile of the three benchmarks, for reasons worth knowing
# before you run it:
#
#   * Python is pinned to 3.8.13 by upstream (torch 1.11 / transformers 4.19 /
#     pyserini 0.17 will not resolve on 3.11+).
#   * The BM25 index is built by pyserini, which is a JVM wrapper -- Java 11 must
#     be on PATH. The cluster provides it as a module, not system-wide.
#   * The product corpus is hosted on **Google Drive** and fetched with `gdown`.
#     Large-file Drive links fail with a quota/confirmation error whenever the
#     file is popular that day. That failure is external and no amount of
#     retrying inside this script fixes it, so the script reports it loudly
#     rather than continuing to the index build and dying confusingly.
set -uo pipefail

DATA_SCALE="${1:-small}"
S=/gpfs/radev/scratch/cohan/jw3278
ENV_PREFIX="${WEBSHOP_ENV:-$S/envs/memsys-webshop}"
REPO="${WEBSHOP_REPO:-$S/webshop_repo}"

export PIP_CACHE_DIR="$S/.pip-cache"
export TMPDIR="$S/tmp"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"

source /gpfs/radev/apps/avx512/software/miniconda/24.3.0-miniforge/etc/profile.d/conda.sh
module load Java/11.0.16 2>/dev/null || echo "[webshop] WARN: could not module-load Java; pyserini indexing will fail"

if [[ ! -d "$REPO" ]]; then
  git clone --depth 1 https://github.com/princeton-nlp/WebShop.git "$REPO"
fi

if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
  conda create -y -p "$ENV_PREFIX" python=3.8.13 pip
fi
conda activate "$ENV_PREFIX"
# faiss + a JDK inside the env, so the env is self-contained even without the module.
# faiss from the `pytorch` channel links against MKL, which is not installed here
# (ImportError: libmkl_intel_lp64.so.1). The conda-forge build bundles its own BLAS.
conda install -y -c conda-forge faiss-cpu
conda install -y -c conda-forge openjdk=11

cd "$REPO"
pip install -q -r requirements.txt || { echo "[webshop] FAILED: pip install"; exit 1; }
# requirements.txt pins Flask==2.1.2 but leaves its dependencies open, so pip
# installs a modern Werkzeug/Jinja2 and every `import web_agent_site` dies with
# `ImportError: cannot import name 'url_quote' from 'werkzeug.urls'` (removed in
# Werkzeug 2.1). That failure is what silently produced a 0-document index.
pip install -q "werkzeug<2.1" "jinja2<3.1" "itsdangerous<2.1" \
  || { echo "[webshop] FAILED: pinning flask deps"; exit 1; }
# spacy 3.3 builds its config models with pydantic 1.x, which chokes on
# typing_extensions >= 4.6 with the opaque `TypeError: issubclass() arg 1 must
# be a class` while importing spacy.schemas. Same shape of problem as Werkzeug:
# upstream pins the top-level package and lets its dependencies float.
pip install -q "typing_extensions<4.6" \
  || { echo "[webshop] FAILED: pinning typing_extensions"; exit 1; }
python -c "from web_agent_site.utils import DEFAULT_FILE_PATH; print('[webshop] import OK')" \
  || { echo "[webshop] FAILED: web_agent_site does not import"; exit 1; }
# setup.sh only fetches _lg, but web_agent_site actually loads `en_core_web_sm`
# at runtime (OSError E050 at env construction). Fetch both.
python -m spacy download en_core_web_lg
python -m spacy download en_core_web_sm

mkdir -p data && cd data
unset HF_HUB_OFFLINE

# Upstream fetches the product corpus from Google Drive with `gdown`, and that
# link reliably fails here with "Cannot retrieve the public link of the file...
# or have had many accesses". So try a Hugging Face mirror FIRST and keep gdown
# as the fallback.
#
# NOTE: `YWZBrandon/webshop-data` is a third-party mirror, not Princeton's own
# distribution. It carries byte-identical filenames to setup.sh's targets, and
# the script sanity-checks record counts below, but if you are publishing
# numbers you should confirm provenance yourself.
HF_MIRROR="${WEBSHOP_HF_MIRROR:-YWZBrandon/webshop-data}"

if [[ "$DATA_SCALE" == "small" ]]; then
  WANT=(items_shuffle_1000.json items_ins_v2_1000.json items_human_ins.json)
else
  WANT=(items_shuffle.json items_ins_v2.json items_human_ins.json)
fi

pip install -q "huggingface_hub" || true
python - "$HF_MIRROR" "${WANT[@]}" <<'PY'
import shutil, sys
from pathlib import Path
repo, names = sys.argv[1], sys.argv[2:]
try:
    from huggingface_hub import hf_hub_download
except ImportError:
    sys.exit("huggingface_hub unavailable")
for name in names:
    if Path(name).is_file():
        print(f"[webshop] {name} already present")
        continue
    print(f"[webshop] fetching {name} from {repo} ...", flush=True)
    path = hf_hub_download(repo_id=repo, filename=name, repo_type="dataset")
    shutil.copyfile(path, name)   # copy: the cache lives elsewhere on scratch
    print(f"[webshop]   -> {Path(name).stat().st_size/1e6:.1f} MB")
PY

gdown_fetch() {  # $1=drive_id  $2=name -- fallback only
  [[ -f "$2" ]] && return 0
  echo "[webshop] HF mirror missed $2; trying Google Drive ..."
  gdown "https://drive.google.com/uc?id=$1" || {
    echo "[webshop] FAILED to obtain $2 from either source."
    echo "[webshop] Download it by hand into $REPO/data and re-run."
    return 1
  }
}
if [[ "$DATA_SCALE" == "small" ]]; then
  gdown_fetch 1EgHdxQ_YxqIQlvvq5iKlCrkEKR6-j0Ib items_shuffle_1000.json || exit 1
  gdown_fetch 1IduG0xl544V_A_jv3tHXC0kyFi7PnyBu items_ins_v2_1000.json  || exit 1
else
  gdown_fetch 1A2whVgOO0euk5O13n2iYDM0bQRkkRduB items_shuffle.json || exit 1
  gdown_fetch 1s2j6NgHljiZzQNL3veZaAiyW_qDEgBNi items_ins_v2.json   || exit 1
fi
gdown_fetch 14Kb5SPBk_jfdLZ_CDBNitW98QLDlKR5O items_human_ins.json || exit 1

python - "${WANT[@]}" <<'PY'
import json, sys
for name in sys.argv[1:]:
    with open(name) as fh:
        data = json.load(fh)
    print(f"[webshop] {name}: {len(data)} records ({type(data).__name__})")
    assert len(data) > 0, f"{name} is empty"
PY
cd ..

cd search_engine
mkdir -p resources resources_100 resources_1k resources_100k indexes
python convert_product_file_format.py \
  || { echo "[webshop] FAILED: convert_product_file_format.py"; exit 1; }
# Upstream's run_indexing.sh happily reports "Indexing Complete! 0 documents
# indexed" and exits 0 when the conversion produced nothing, so a broken setup
# looks like a successful one. Check the corpus before indexing it.
for d in resources resources_1k; do
  n=$(wc -l < "$d/documents.jsonl" 2>/dev/null || echo 0)
  [[ "$n" -gt 0 ]] || { echo "[webshop] FAILED: $d/documents.jsonl is empty"; exit 1; }
  echo "[webshop] $d: $n documents"
done
./run_indexing.sh || { echo "[webshop] FAILED: pyserini indexing (is Java 11 on PATH?)"; exit 1; }
cd ..

echo
echo "[webshop] env  : $ENV_PREFIX"
echo "[webshop] repo : $REPO  (scale=$DATA_SCALE)"
python -c "
from web_agent_site.envs import WebAgentTextEnv
env = WebAgentTextEnv(observation_mode='text', num_products=None)
env.reset()
print('[webshop] reset OK; first observation:')
print(env.observation[:300])
" || echo "[webshop] WARN: import smoke failed; check the log above"

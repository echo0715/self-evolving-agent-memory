#!/usr/bin/env bash
# Build the WebShop BM25 index for a chosen corpus scale.
#
#   bash scripts/webshop_build_index.sh full    # 1.18M products -> resources_full/, indexes_full/
#   bash scripts/webshop_build_index.sh small   # 1,000 products  -> resources_1k/,   indexes_1k/
#
# `scripts/setup_webshop.sh` already built the small index; this script exists for
# the full corpus, which upstream cannot build without editing a vendored source
# file (see the WEBSHOP_FILE_PATH patch in web_agent_site/utils.py).
#
# Why a separate index directory name instead of upstream's `indexes`: upstream
# derives the index path from `num_products`, so `num_products=None` means
# "`indexes`" regardless of which corpus produced it. This checkout already has a
# 1,000-document `indexes/` built from items_shuffle_1000.json. Writing the full
# index there would leave the two indistinguishable, and a corpus paired with the
# wrong index does not raise -- searches simply return products that are not in
# the catalogue. Naming them apart makes the pairing an explicit choice.
set -uo pipefail

SCALE="${1:-full}"
S=/gpfs/radev/scratch/cohan/jw3278
ENV_PREFIX="${WEBSHOP_ENV:-$S/envs/memsys-webshop}"
REPO="${WEBSHOP_REPO:-$S/webshop_repo}"
PY="$ENV_PREFIX/bin/python"

case "$SCALE" in
  full)  FILE=items_shuffle.json;      ATTR=items_ins_v2.json;      RES=resources_full; IDX=indexes_full ;;
  small) FILE=items_shuffle_1000.json; ATTR=items_ins_v2_1000.json; RES=resources_1k;   IDX=indexes_1k ;;
  *) echo "usage: $0 {full|small}" >&2; exit 2 ;;
esac

export WEBSHOP_FILE_PATH="$REPO/data/$FILE"
export WEBSHOP_ATTR_PATH="$REPO/data/$ATTR"
export TMPDIR="${TMPDIR:-$S/tmp}"
# pyserini is a JVM wrapper (pyjnius), so it needs a JDK *and* libjvm.so, which
# it locates from JAVA_HOME. The conda env carries its own openjdk 11, but it
# installs the JDK under $PREFIX/lib/jvm, not at $PREFIX -- pointing JAVA_HOME at
# the env prefix gets as far as `java` on PATH and then dies in pyjnius with a
# "set JAVA_HOME / JVM_PATH" message that names neither.
export JAVA_HOME="$ENV_PREFIX/lib/jvm"
export JVM_PATH="$JAVA_HOME/lib/server/libjvm.so"
export PATH="$ENV_PREFIX/bin:$JAVA_HOME/bin:$PATH"
[[ -f "$JVM_PATH" ]] || { echo "[index] FAILED: no libjvm.so at $JVM_PATH"; exit 1; }
mkdir -p "$TMPDIR"

for f in "$WEBSHOP_FILE_PATH" "$WEBSHOP_ATTR_PATH"; do
  [[ -f "$f" ]] || { echo "[index] FAILED: missing corpus file $f"; exit 1; }
done

cd "$REPO/search_engine" || exit 1
mkdir -p "$RES"

if [[ -s "$RES/documents.jsonl" ]]; then
  echo "[index] $RES/documents.jsonl already present ($(wc -l < "$RES/documents.jsonl") docs); skipping conversion"
else
  echo "[index] converting $FILE -> $RES/documents.jsonl (this loads the whole corpus into RAM)"
  "$PY" - "$RES" <<'PY' || exit 1
import json, resource, sys, time
sys.path.insert(0, '../')
from web_agent_site.utils import DEFAULT_FILE_PATH
from web_agent_site.engine.engine import load_products

res_dir = sys.argv[1]
t0 = time.time()
all_products, *_ = load_products(filepath=DEFAULT_FILE_PATH)
# ru_maxrss is KiB on Linux. Reported because the peak here is what decides the
# SLURM --mem for every later run, and it is several times the file size: the
# raw bytes, the decoded text and the object graph are all live during parsing.
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
print(f'[index] loaded {len(all_products)} products in {time.time()-t0:.0f}s, peak RSS {peak:.1f} GiB', flush=True)

n = 0
with open(f'{res_dir}/documents.jsonl', 'w') as fh:
    for p in all_products:
        options = p.get('options', {})
        option_text = ', and '.join(
            f'{name}: {", ".join(contents)}' for name, contents in options.items()
        )
        # Field selection is upstream's, verbatim: the BM25 index has to contain
        # exactly what upstream's does or search behaviour stops being comparable.
        fh.write(json.dumps({
            'id': p['asin'],
            'contents': ' '.join([
                p['Title'], p['Description'], p['BulletPoints'][0], option_text,
            ]).lower(),
            'product': p,
        }) + '\n')
        n += 1
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
print(f'[index] wrote {n} documents, peak RSS {peak:.1f} GiB, {time.time()-t0:.0f}s total', flush=True)
PY
fi

# Upstream's run_indexing.sh exits 0 after "Indexing Complete! 0 documents
# indexed" when the conversion produced nothing, so a broken setup is
# indistinguishable from a working one. Check before, and check again after.
n_docs=$(wc -l < "$RES/documents.jsonl")
[[ "$n_docs" -gt 0 ]] || { echo "[index] FAILED: $RES/documents.jsonl is empty"; exit 1; }
echo "[index] indexing $n_docs documents -> $IDX"

rm -rf "$IDX"
"$PY" -m pyserini.index.lucene \
  --collection JsonCollection \
  --input "$RES" \
  --index "$IDX" \
  --generator DefaultLuceneDocumentGenerator \
  --threads "${WEBSHOP_INDEX_THREADS:-8}" \
  --storePositions --storeDocvectors --storeRaw || {
    echo "[index] FAILED: pyserini indexing"; exit 1; }

"$PY" - "$IDX" "$n_docs" <<'PY' || exit 1
import sys
from pyserini.search.lucene import LuceneSearcher
idx, expected = sys.argv[1], int(sys.argv[2])
s = LuceneSearcher(idx)
got = s.num_docs
print(f'[index] {idx}: {got} documents indexed (corpus had {expected})')
assert got > 0, 'index is empty'
# Duplicate asins in the corpus are dropped by load_products, so `got` can be
# below `expected` only if pyserini itself skipped documents.
assert got >= expected * 0.99, f'indexed {got} of {expected} documents'
hits = s.search('running shoes for men')
print(f'[index] smoke query returned {len(hits)} hits; top id = {hits[0].docid if hits else None}')
assert hits, 'index returns no hits for a generic query'
PY

echo "[index] OK  corpus=$FILE  resources=$RES  index=$IDX"
echo "[index] to use it:  WEBSHOP_FILE_PATH=$WEBSHOP_FILE_PATH WEBSHOP_ATTR_PATH=$WEBSHOP_ATTR_PATH WEBSHOP_INDEX_DIR=$IDX"

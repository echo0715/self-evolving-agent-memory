#!/usr/bin/env bash
# Serve Qwen3.5-9B as an OpenAI-compatible endpoint, one server per GPU.
#
# Adapted from SAGE/harness/scripts/serve_qwen.sh. The difference is that this
# one is parameterised by GPU index and port, so several independent TP=1
# servers can run side by side -- a 9B model is 19 GB and an A100 is 80 GB, so
# two servers on two GPUs beat one TP=2 server on PCIe-connected cards.
#
#   bash scripts/serve_qwen.sh 0 8000 --background
#   bash scripts/serve_qwen.sh 1 8001 --background
#
# Readiness (~2-4 min: 19 GB load + CUDA graph capture):
#   curl -s -o /dev/null -w '%{http_code}\n' http://localhost:8000/health   # -> 200
set -euo pipefail

GPU="${1:-0}"
PORT="${2:-8000}"
BACKGROUND="${3:-}"

SCRATCH=/gpfs/radev/scratch/cohan/jw3278

# Pre-built serving env: python 3.12, torch 2.11.0+cu129, vllm 0.24.0,
# transformers 5.13.0. vllm 0.24 is the first release whose registry contains
# Qwen3_5ForConditionalGeneration (model_type "qwen3_5"). The cluster driver is
# 570.x / CUDA 12.8, hence +cu129 wheels and not the PyPI-default CUDA 13 build.
VLLM_ENV="${MEMSYS_VLLM_ENV:-$SCRATCH/verl_qwen35_train}"
MODEL="${MEMSYS_MODEL:-Qwen/Qwen3.5-9B}"
# Qwen3.5-9B supports 262144 natively. We cap agent completions hard (see
# scripts/run_alfworld.py --max-tokens), so contexts stay in the low thousands;
# 32768 is ample here and leaves more KV cache for concurrent requests.
MAX_LEN="${MEMSYS_MAX_MODEL_LEN:-32768}"
GPU_UTIL="${MEMSYS_GPU_UTIL:-0.90}"
LOG="${MEMSYS_SERVE_LOG:-$SCRATCH/memsys_qwen9b_gpu${GPU}_${PORT}.log}"

# Keep HF + compile caches off HOME (home is at its inode quota).
export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
export HF_HUB_DISABLE_XET=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
# Per-port cache roots: two servers compiling into one dir race each other.
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$SCRATCH/env-caches/memsys-vllm-$PORT}"
export OUTLINES_CACHE_DIR="$VLLM_CACHE_ROOT/outlines"
export TRITON_CACHE_DIR="$VLLM_CACHE_ROOT/triton"
export TMPDIR="${TMPDIR:-$SCRATCH/tmp}"
mkdir -p "$VLLM_CACHE_ROOT" "$OUTLINES_CACHE_DIR" "$TRITON_CACHE_DIR" "$TMPDIR"

export CUDA_VISIBLE_DEVICES="$GPU"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Cluster CUDA 12.8 toolkit for runtime JIT (flashinfer/triton kernels).
export CUDA_HOME="${CUDA_HOME:-/gpfs/radev/apps/avx512/software/CUDA/12.8.0}"
# Qwen3.5's Gated-DeltaNet layer JIT-compiles kernels at load time and shells out
# to `ninja` and `nvcc`. Calling $VLLM_ENV/bin/vllm directly does NOT put the
# env's bin on PATH, and the failure is misleading: vLLM reports only
# "Engine core initialization failed" while the real cause buried in the log is
# FileNotFoundError: 'ninja'. Put both on PATH explicitly.
export PATH="$VLLM_ENV/bin:$CUDA_HOME/bin:$PATH"

CMD=("$VLLM_ENV/bin/vllm" serve "$MODEL"
  --served-model-name "$MODEL"
  --host 127.0.0.1 --port "$PORT"
  --tensor-parallel-size 1
  --max-model-len "$MAX_LEN"
  --gpu-memory-utilization "$GPU_UTIL"
  # Gated-DeltaNet's flashinfer *prefill* kernel is broken on this cluster;
  # force the Triton path.
  --additional-config '{"gdn_prefill_backend":"triton"}'
  # Non-thinking mode, forced server-side: Qwen3.5's chat template gates its
  # reasoning block on `enable_thinking`, and vLLM merges these kwargs into
  # every request, so no client needs per-call plumbing.
  --default-chat-template-kwargs '{"enable_thinking": false}')

echo "[serve] gpu=$GPU port=$PORT model=$MODEL max_model_len=$MAX_LEN log=$LOG"
nvidia-smi -L || { echo "ERROR: no GPU visible; run this on a GPU node" >&2; exit 1; }

if [[ "$BACKGROUND" == "--background" ]]; then
  nohup "${CMD[@]}" > "$LOG" 2>&1 &
  echo "[serve] started (PID $!). log: $LOG"
else
  exec "${CMD[@]}"
fi

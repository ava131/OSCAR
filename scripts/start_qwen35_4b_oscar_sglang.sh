#!/usr/bin/env bash
set -euo pipefail

# Run from the OSCAR repo root:
#   bash start_qwen35_4b_oscar_sglang.sh
export TORCH_EXTENSIONS_DIR="/tmp/torch_extensions"
export TMPDIR="/tmp"
export RAY_TMPDIR="/tmp"

export CUDA_LAUNCH_BLOCKING=1

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export MODEL_PATH="${MODEL_PATH:-$HF_HOME/hub/models--Qwen--Qwen3.5-4B/snapshots}"
export MODEL_PATH="/home/z50058184/qwen3.5-4b"
export ROTATION_ROOT="${ROTATION_ROOT:-$PWD/rotzoo/Qwen3.5-4B}"
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-30000}"
export TP_SIZE="${TP_SIZE:-2}"


# Qwen3.5 support is on the OSCAR/SGLang hybrid branch and needs Lloyd-Max enabled.
export SGLANG_LLOYD_MAX=1
export SGLANG_ENABLE_MIXED_KV_WINDOWS=1
export SGLANG_OSCAR_K_CLIP_RATIO="${SGLANG_OSCAR_K_CLIP_RATIO:-0.96}"
export SGLANG_OSCAR_V_CLIP_RATIO="${SGLANG_OSCAR_V_CLIP_RATIO:-0.92}"
export SGLANG_OSCAR_ABSORB_V_ROTATION=1
export SGLANG_MIXED_KV_PREFIX_TOKENS="${SGLANG_MIXED_KV_PREFIX_TOKENS:-64}"
export SGLANG_MIXED_KV_RECENT_TOKENS="${SGLANG_MIXED_KV_RECENT_TOKENS:-256}"
export SGLANG_MIXED_KV_HP_MAX_SPLITS="${SGLANG_MIXED_KV_HP_MAX_SPLITS:-8}"
export SGLANG_MIXED_KV_HP_DTYPE="${SGLANG_MIXED_KV_HP_DTYPE:-bfloat16}"
export SGLANG_MIXED_KV_SCALE_DTYPE="${SGLANG_MIXED_KV_SCALE_DTYPE:-float32}"

export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

if [[ -d "$MODEL_PATH" ]]; then
  MODEL_PATH="$(find "$MODEL_PATH" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
fi

export MODEL_PATH="/home/z50058184/qwen3.5-4b"

ROT_DIR="${ROT_DIR:-$(find "$ROTATION_ROOT" -type d -name rotations 2>/dev/null | sort | tail -n 1)}"

if [[ -z "${ROT_DIR:-}" ]]; then
  ROT_DIR="$(find "$ROTATION_ROOT" -type f -name 'k_rotation_qqt_r_h_pbr.pt' -exec dirname {} \; | sort | tail -n 1)"
fi

echo $ROT_DIR

export SGLANG_OSCAR_K_ROTATION_PATH="$ROT_DIR/k_rotation_qqt_r_h_pbr.pt"
export SGLANG_OSCAR_V_ROTATION_PATH="$ROT_DIR/v_rotation_sst_r_h_pbr.pt"

echo "MODEL_PATH=$MODEL_PATH"
echo "ROT_DIR=$ROT_DIR"
echo "PORT=$PORT TP_SIZE=$TP_SIZE CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

CUDA_LAUNCH_BLOCKING=1 \
python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --host "$HOST" \
  --port "$PORT" \
  --tensor-parallel-size "$TP_SIZE" \
  --dtype float16 \
  --prefill-attention-backend triton \
  --decode-attention-backend triton \
  --disable-cuda-graph \
  --disable-radix-cache \
  --sampling-backend pytorch \
  --tokenizer-worker-num 1 \
  --disable-overlap-schedule \
  --trust-remote-code

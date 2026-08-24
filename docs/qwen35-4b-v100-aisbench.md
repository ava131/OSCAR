# OSCAR + SGLang + Qwen3.5-4B on 2x V100

This note records a minimal setup path for running OSCAR KV-cache quantization experiments with Qwen3.5-4B through SGLang, then evaluating with AISBench.

## 0. Environment

Recommended base environment:

```bash
conda create -n oscar python=3.10 -y
conda activate oscar

pip install -U pip setuptools wheel ninja cmake packaging
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

For V100, avoid FP8/H100-only paths. Prefer Triton decode backends when FlashAttention-3 or flashinfer kernels are unavailable.

## 1. Download Qwen3.5-4B

```bash
pip install -U huggingface_hub

huggingface-cli download Qwen/Qwen3.5-4B \
  --local-dir /data/models/Qwen3.5-4B \
  --local-dir-use-symlinks False
```

If using the instruction-tuned checkpoint, replace the model id and directory:

```bash
huggingface-cli download Qwen/Qwen3.5-4B-Instruct \
  --local-dir /data/models/Qwen3.5-4B-Instruct \
  --local-dir-use-symlinks False
```

## 2. Clone OSCAR and Rotation Zoo

```bash
git clone https://github.com/FutureMLS-Lab/OSCAR.git
cd OSCAR

git lfs install
git clone https://huggingface.co/Zhongzhu/OSCAR-RotationZoo rotzoo
```

## 3. Install OSCAR

From the OSCAR repository root:

```bash
pip install -r requirements.txt
pip install -e .
```

## 4. Install OSCAR SGLang Fork

OSCAR uses a customized SGLang branch for hybrid KV-cache quantization support.

```bash
cd ..
git clone https://github.com/FutureMLS-Lab/sglang.git
cd sglang
git checkout hybrid-model

pip install -U pip setuptools wheel ninja cmake packaging
pip install -e "python[all]"
```

Check the installed server entrypoint:

```bash
python -m sglang.launch_server --help | grep -i kv
```

## 5. Start SGLang Server

Save this as `start_qwen35_4b_oscar_sglang.sh` and run it from the OSCAR repository root.

```bash
#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export MODEL_PATH="${MODEL_PATH:-/data/models/Qwen3.5-4B}"
export ROTATION_ROOT="${ROTATION_ROOT:-$PWD/rotzoo/Qwen3.5-4B}"
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-30000}"
export TP_SIZE="${TP_SIZE:-2}"

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

ROT_DIR="${ROT_DIR:-$(find "$ROTATION_ROOT" -type d -name rotations 2>/dev/null | sort | tail -n 1)}"
if [[ -z "${ROT_DIR:-}" ]]; then
  ROT_DIR="$(find "$ROTATION_ROOT" -type f -name 'k_rotation_qqt_r_h_pbr.pt' -exec dirname {} \; | sort | tail -n 1)"
fi

export SGLANG_OSCAR_K_ROTATION_PATH="$ROT_DIR/k_rotation_qqt_r_h_pbr.pt"
export SGLANG_OSCAR_V_ROTATION_PATH="$ROT_DIR/v_rotation_sst_r_h_pbr.pt"

echo "MODEL_PATH=$MODEL_PATH"
echo "ROT_DIR=$ROT_DIR"
echo "PORT=$PORT TP_SIZE=$TP_SIZE CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --host "$HOST" \
  --port "$PORT" \
  --tensor-parallel-size "$TP_SIZE" \
  --kv-cache-dtype int2 \
  --kv-cache-quant-group-size 128 \
  --decode-attention-backend triton \
  --trust-remote-code
```

Run:

```bash
chmod +x start_qwen35_4b_oscar_sglang.sh

MODEL_PATH=/data/models/Qwen3.5-4B \
ROTATION_ROOT=/path/to/OSCAR/rotzoo/Qwen3.5-4B \
CUDA_VISIBLE_DEVICES=0,1 \
TP_SIZE=2 \
PORT=30000 \
./start_qwen35_4b_oscar_sglang.sh
```

## 6. AISBench

Point AISBench at the OpenAI-compatible SGLang endpoint:

```text
http://127.0.0.1:30000/v1
```

Use the same model name/path that SGLang exposes for the server.

## Notes

- The original paper link `https://arxiv.org/html/2410.18963v1` appears to refer to a different OSCAR project. The Qwen3.5-4B + SGLang + KV-cache setup corresponds to `FutureMLS-Lab/OSCAR`.
- If `--kv-cache-dtype int2` or `--kv-cache-quant-group-size` is rejected, verify the exact CLI flags in the checked-out SGLang branch with `python -m sglang.launch_server --help`.
- V100 does not support FP8 Tensor Cores, so FP8 KV-cache settings should not be used for this setup.

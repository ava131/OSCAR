# OSCAR + Qwen3.5-4B + SGLang in Docker with uv

This guide assumes you are already inside a Linux Docker container on the evaluation server.

Goal:

- avoid Conda entirely;
- avoid modifying the container's `base` Python environment;
- use `uv` to create a project-local `.venv`;
- install OSCAR's vendored SGLang from local source;
- run Qwen3.5-4B on 2x V100 for AISBench evaluation.

## 0. Check the Container

Run these inside the Docker container:

```bash
nvidia-smi
python --version
which python
```

You should see both V100 GPUs in `nvidia-smi`.

## 1. Install uv

If the container has network access:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version
```

If `curl` is unavailable but `pip` exists, this is also acceptable:

```bash
python -m pip install --user uv
export PATH="$HOME/.local/bin:$PATH"
uv --version
```

This installs only the `uv` tool under the user directory, not into Conda base packages.

## 2. Clone OSCAR

Use the fork with this guide:

```bash
git clone --recursive https://github.com/ava131/OSCAR.git
cd OSCAR
```

If submodules were not fetched:

```bash
git submodule update --init --recursive
```

## 3. Create a Local uv Environment

Create `.venv` inside the OSCAR repo:

```bash
uv venv .venv --python 3.12
source .venv/bin/activate
```

Confirm you are using the local environment:

```bash
which python
python --version
```

Expected path shape:

```text
/path/to/OSCAR/.venv/bin/python
```

## 4. Install PyTorch

OSCAR's README targets recent SGLang with CUDA 12.8/12.9-era packages. In many Docker images, PyTorch may already be installed. First check:

```bash
python - <<'PY'
try:
    import torch
    print("torch:", torch.__version__)
    print("cuda:", torch.version.cuda)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu0:", torch.cuda.get_device_name(0))
except Exception as e:
    print("torch import failed:", repr(e))
PY
```

If PyTorch is missing, install a CUDA wheel into `.venv`. For CUDA 12.1:

```bash
uv pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu121
```

For CUDA 12.8/12.9 images, prefer the wheel index matching the image/toolkit. If your container already has a working PyTorch, skip this step.

## 5. Install Vendored SGLang Locally

OSCAR already vendors SGLang under:

```text
sglang-research/
sglang-dump-qkv/
```

Do not clone another SGLang repo.

Install the eval-side SGLang from local source:

```bash
uv pip install -e sglang-research/python
```

If you want to avoid dependency resolution because the container already has the correct packages, use:

```bash
uv pip install --no-deps -e sglang-research/python
```

The `--no-deps` version is the least disruptive. The non-`--no-deps` version is easier when starting from a clean Docker image.

## 6. Verify Imports

```bash
python - <<'PY'
import torch
import transformers
import sglang

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("transformers:", transformers.__version__)
print("sglang:", sglang.__file__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))
PY
```

## 7. Download Qwen3.5-4B

Install Hugging Face CLI only if it is missing:

```bash
uv pip install huggingface_hub
```

Download the model:

```bash
mkdir -p /data/models

huggingface-cli download Qwen/Qwen3.5-4B \
  --local-dir /data/models/Qwen3.5-4B \
  --local-dir-use-symlinks False
```

If you use the instruction model, replace the repo id:

```bash
huggingface-cli download Qwen/Qwen3.5-4B-Instruct \
  --local-dir /data/models/Qwen3.5-4B-Instruct \
  --local-dir-use-symlinks False
```

## 8. Download OSCAR Rotation Zoo

```bash
git lfs install
git clone https://huggingface.co/Zhongzhu/OSCAR-RotationZoo rotzoo
```

If the container does not have Git LFS:

```bash
apt-get update
apt-get install -y git-lfs
git lfs install
```

## 9. Start Server

Use the provided launcher:

```bash
chmod +x scripts/start_qwen35_4b_oscar_sglang.sh

MODEL_PATH=/data/models/Qwen3.5-4B \
ROTATION_ROOT=$PWD/rotzoo/Qwen3.5-4B \
CUDA_VISIBLE_DEVICES=0,1 \
TP_SIZE=2 \
PORT=30000 \
scripts/start_qwen35_4b_oscar_sglang.sh
```

AISBench can target:

```text
http://127.0.0.1:30000/v1
```

## 10. If V100 Rejects a Backend

V100 does not support FP8 Tensor Cores and may not support some newer FlashAttention-3 paths.

If server launch fails around attention backend flags, inspect available flags:

```bash
python -m sglang.launch_server --help | grep -i attention
python -m sglang.launch_server --help | grep -i kv
```

Prefer Triton decode backend on V100:

```bash
--decode-attention-backend triton
```

Avoid FP8 KV settings on V100.

## Minimal Command Block

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

git clone --recursive https://github.com/ava131/OSCAR.git
cd OSCAR

uv venv .venv --python 3.12
source .venv/bin/activate

# If torch is not already available in the Docker image:
uv pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu121

uv pip install -e sglang-research/python
uv pip install huggingface_hub

mkdir -p /data/models
huggingface-cli download Qwen/Qwen3.5-4B \
  --local-dir /data/models/Qwen3.5-4B \
  --local-dir-use-symlinks False

git lfs install
git clone https://huggingface.co/Zhongzhu/OSCAR-RotationZoo rotzoo

chmod +x scripts/start_qwen35_4b_oscar_sglang.sh
MODEL_PATH=/data/models/Qwen3.5-4B \
ROTATION_ROOT=$PWD/rotzoo/Qwen3.5-4B \
CUDA_VISIBLE_DEVICES=0,1 \
TP_SIZE=2 \
PORT=30000 \
scripts/start_qwen35_4b_oscar_sglang.sh
```

# OSCAR + Qwen3.5-4B + 2x V100：uv 完整部署指南

本文假设你已经进入 Linux Docker，能看到 NVIDIA GPU，且服务器上已有 Python，但不使用 Conda/Miniconda。所有 Python 包安装到项目自己的 `.venv`，不修改系统 Python。

## 0. 版本和兼容性原则

OSCAR 当前 vendored SGLang 的依赖声明包含：

```text
Python >= 3.10
torch == 2.9.1
torchaudio == 2.9.1
transformers == 5.3.0
cuda-python == 12.9
flashinfer_python == 0.6.7.post3
flashinfer_cubin == 0.6.7.post3
flash-attn-4 >= 4.0.0b4
sglang-kernel == 0.4.1
xgrammar == 0.1.32
```

这些版本偏新，而 V100 是 Volta，通常为计算能力 `7.0`。因此不要一开始无条件安装全部依赖。按这个顺序验收：

```text
GPU/驱动 -> Python -> uv/.venv -> PyTorch/CUDA -> 本地 SGLang
-> CUDA 扩展/backend -> 模型 -> Rotation Zoo -> 服务
```

当前 launcher 默认使用 `fa3` prefill 和 `triton` decode。V100 首要风险是 FA3、FlashInfer、新 CUDA kernel 和 FP8 backend。

## 1. 检查 Docker、驱动和 GPU

```bash
nvidia-smi
nvidia-smi -L
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
uname -m
cat /etc/os-release
```

应能看到两张 V100。Docker 通常使用宿主机 NVIDIA driver；不要因为 `nvidia-smi` 显示 CUDA 12.x 就误以为容器里一定有 `nvcc`。

## 2. 检查 Python

```bash
python3 --version
which python3
python3 -c 'import sys; print(sys.executable); print(sys.version)'
```

推荐 Python 3.12；3.10/3.11 通常也可以。低于 3.10 不建议继续。

## 3. 安装 uv

既然服务器可以正常使用 pip，直接执行：

```bash
python3 -m pip install --user -U uv
```

如果出现 externally-managed-environment：

```bash
python3 -m pip install --user -U --break-system-packages uv
```

恢复 PATH：

```bash
export PATH="$(python3 -m site --user-base)/bin:$HOME/.local/bin:$PATH"
```

验证：

```bash
command -v uv
uv --version
uv --help | sed -n '1,40p'
```

如果找不到：

```bash
python3 -m pip show uv
python3 -m site --user-base
find "$HOME" -maxdepth 4 -type f -name uv -perm -111 2>/dev/null
```

## 4. 准备 Python 3.12

查看 uv 可用 Python：

```bash
uv python list
```

如果系统没有 3.12，让 uv 下载：

```bash
uv python install 3.12
uv python find 3.12
```

如果服务器无法下载 Python，直接使用现有的 3.10-3.12：

```bash
python3 --version
```

## 5. 下载 OSCAR

```bash
mkdir -p /data
cd /data
git clone --recursive https://github.com/ava131/OSCAR.git
cd OSCAR
git submodule update --init --recursive
```

检查源码：

```bash
test -d sglang-research/python && echo 'vendored SGLang: OK'
test -d sglang-dump-qkv && echo 'dump-qkv: OK'
test -f scripts/start_qwen35_4b_oscar_sglang.sh && echo 'launcher: OK'
```

不需要另行 clone 外部 SGLang。

## 6. 创建隔离环境

```bash
cd /data/OSCAR
uv venv .venv --python 3.12
source .venv/bin/activate
```

确认没有使用系统 Python：

```bash
which python
which pip
python --version
python -c 'import sys; print(sys.prefix); print(sys.base_prefix)'
```

预期路径：

```text
/data/OSCAR/.venv/bin/python
/data/OSCAR/.venv/bin/pip
sys.prefix != sys.base_prefix
```

## 7. 设置缓存和下载参数

```bash
export UV_CACHE_DIR=/data/.cache/uv
export HF_HOME=/data/.cache/huggingface
mkdir -p "$UV_CACHE_DIR" "$HF_HOME"
```

Hugging Face 网络不稳定时：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DOWNLOAD_TIMEOUT=600
export HF_HUB_ETAG_TIMEOUT=600
```

## 8. 安装 PyTorch

先检查：

```bash
python - <<'PY'
try:
    import torch
    print('torch:', torch.__version__)
    print('torch cuda:', torch.version.cuda)
except Exception as e:
    print('torch unavailable:', repr(e))
PY
```

OSCAR 声明目标是 torch 2.9.1，先尝试对应 CUDA wheel：

```bash
uv pip install \
  'torch==2.9.1' \
  'torchaudio==2.9.1' \
  torchvision \
  --index-url https://download.pytorch.org/whl/cu129
```

安装后必须马上验证，不要直接继续：

```bash
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('torch cuda:', torch.version.cuda)
print('cuda available:', torch.cuda.is_available())
print('device count:', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
    print('capability:', torch.cuda.get_device_capability(i))
PY
```

V100 通常应显示 `capability: (7, 0)`，并且 `device count: 2`。

如果 cu129 wheel 不可用，或 PyTorch 无法在 V100 上运行，记录错误后尝试一组较保守的 wheel：

```bash
uv pip uninstall -y torch torchaudio torchvision
uv pip install \
  'torch==2.7.1' \
  'torchaudio==2.7.1' \
  torchvision \
  --index-url https://download.pytorch.org/whl/cu126
```

具体可用组合应以 PyTorch 官方 Previous Versions 页面为准。宿主机 driver、容器 CUDA runtime、PyTorch wheel 内置 CUDA 是三个不同概念。

如果 `nvidia-smi` 能看到 GPU，但 PyTorch 看不到：

```bash
echo "$CUDA_VISIBLE_DEVICES"
ls -l /dev/nvidia*
nvidia-smi -L
unset CUDA_VISIBLE_DEVICES
```

## 9. 安装本地 SGLang

先不要执行普通 editable 安装，因为它会自动解析整套新依赖：

```text
不要直接执行：uv pip install -e sglang-research/python
```

最小安装：

```bash
cd /data/OSCAR
source .venv/bin/activate
uv pip install --no-deps -e sglang-research/python
```

验证：

```bash
python - <<'PY'
import torch
import sglang
print('torch:', torch.__version__)
print('cuda:', torch.cuda.is_available())
print('sglang:', sglang.__file__)
PY
```

如果 traceback 缺少普通 Python 包，逐步补装：

```bash
uv pip install aiohttp fastapi uvicorn uvloop
uv pip install transformers sentencepiece tiktoken
uv pip install numpy scipy einops
uv pip install requests pydantic orjson
```

先不要主动安装这些高风险 CUDA 包：

```text
flash-attn-4
flashinfer_python
flashinfer_cubin
sglang-kernel
quack-kernels
torchao
```

只有 traceback 明确要求、且确认支持 V100 时，才单独安装。

检查：

```bash
uv pip check
uv pip list | sort
uv pip tree | sed -n '1,240p'
```

如果必须对齐仓库声明的 Transformers：

```bash
uv pip install 'transformers==5.3.0'
```

## 10. 安装新版 hf CLI

```bash
uv pip install -U huggingface_hub
export PATH="/data/OSCAR/.venv/bin:$PATH"
hf --help
```

不要使用已废弃的 `huggingface-cli`。

需要权限时：

```bash
hf auth login
hf auth whoami
```

## 11. 下载模型和 Rotation Zoo

```bash
mkdir -p /data/models
hf download Qwen/Qwen3.5-4B \
  --local-dir /data/models/Qwen3.5-4B \
  --resume-download
```

检查：

```bash
test -f /data/models/Qwen3.5-4B/config.json && echo 'model config: OK'
find /data/models/Qwen3.5-4B -maxdepth 2 -type f | sort | sed -n '1,80p'
```

下载 Rotation Zoo：

```bash
cd /data/OSCAR
hf download Zhongzhu/OSCAR-RotationZoo \
  --repo-type model \
  --include 'Qwen3.5-4B/**' \
  --local-dir rotzoo \
  --resume-download
```

检查：

```bash
find /data/OSCAR/rotzoo -type f | sort
find /data/OSCAR/rotzoo -type f -name 'k_rotation_qqt_r_h_pbr.pt'
find /data/OSCAR/rotzoo -type f -name 'v_rotation_sst_r_h_pbr.pt'
```

如果出现 `LocalEntryNotFound`、`error 110` 或 timeout：

```bash
curl -I --max-time 20 \
  "$HF_ENDPOINT/Qwen/Qwen3.5-4B/resolve/main/config.json"
```

这代表远端连接问题，不代表模型名错误。必要时在其他机器下载后，通过共享盘、scp 或 Docker volume 传到服务器。

## 12. CUDA 编译和 backend 检查

```bash
which gcc || true
which g++ || true
which nvcc || true
nvcc --version || true
python -c 'import torch; print(torch.utils.cpp_extension.CUDA_HOME)' 2>/dev/null || true
```

如果没有 `nvcc`，PyTorch 仍可能能推理，但编译 CUDA 扩展可能失败；这时需要带 CUDA development tools 的 Docker 镜像或管理员提供 Toolkit。

查看 backend：

```bash
python -m sglang.launch_server --help | grep -Ei 'attention|kv|flash|triton'
```

V100 建议：

- decode 优先使用 Triton；
- 不使用 FP8 KV；
- 如果 FA3 编译或运行失败，先把 prefill backend 改为兼容 backend 或默认值；
- 不要假设 H100/A100 的 FlashAttention kernel 能在 V100 工作。

## 13. 启动服务

仓库 launcher 默认配置了 OSCAR 的环境变量、int2 KV cache、FA3 prefill 和 Triton decode：

```bash
cd /data/OSCAR
source .venv/bin/activate
chmod +x scripts/start_qwen35_4b_oscar_sglang.sh

MODEL_PATH=/data/models/Qwen3.5-4B \
ROTATION_ROOT=/data/OSCAR/rotzoo/Qwen3.5-4B \
CUDA_VISIBLE_DEVICES=0,1 \
TP_SIZE=2 \
PORT=30000 \
bash scripts/start_qwen35_4b_oscar_sglang.sh
```

启动成功后：

```bash
curl http://127.0.0.1:30000/health
curl http://127.0.0.1:30000/v1/models
```

AISBench endpoint：

```text
http://127.0.0.1:30000/v1
```

如果 AISBench 在另一个容器或机器中，不能使用 `127.0.0.1`，应使用宿主机 IP、服务容器 IP 或端口映射地址。

## 14. 出错时的最小诊断包

出现问题时，先收集：

```bash
nvidia-smi
python --version
which python
uv --version
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())'
python -c 'import sglang; print(sglang.__file__)'
uv pip list | sort
git rev-parse HEAD
git submodule status
```

最重要的停止点：

1. PyTorch 必须能识别两张 V100；
2. `import sglang` 必须成功；
3. rotation 文件必须存在；
4. 最后才处理 attention backend 和服务启动。

## 15. 时间预算

网络正常时：

| 阶段 | 预计时间 |
|---|---:|
| 安装 uv/Python | 1-10 分钟 |
| Clone OSCAR 和子模块 | 5-20 分钟 |
| 创建 .venv | 10 秒以内 |
| 下载 PyTorch | 5-30 分钟 |
| 普通 Python 依赖 | 5-30 分钟 |
| CUDA/C++ 扩展编译 | 10-60 分钟 |
| 下载 Qwen 模型 | 10 分钟到数小时 |
| 下载 Rotation Zoo | 1-30 分钟 |
| 首次启动排错 | 20-120 分钟 |

现实预算约为：

- 依赖兼容且网络顺利：1-2 小时；
- 需要 CUDA 扩展排错：2-4 小时；
- FA3/FlashInfer 不支持 V100，需要改 backend 或回退版本：4-8 小时。


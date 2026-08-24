# OSCAR + Qwen3.5-4B + SGLang on 2x V100 (Docker + Conda base)

这份指南假设你已经进入服务器上的 Linux Docker 容器，并且只能使用 Miniconda 的 `base` 环境。为了减少污染，不重新安装已有依赖；项目本身的 SGLang 使用本地 editable 安装，并关闭依赖自动解析。

## 1. 检查容器和 base

```bash
nvidia-smi
conda activate base
python --version
which python
```

后续命令都应在 `base` 中执行。先确认 PyTorch 和 GPU：

```bash
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('torch cuda:', torch.version.cuda)
print('cuda available:', torch.cuda.is_available())
print('gpu count:', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY
```

如果这里已经能看到 2 张 V100，跳过 PyTorch 安装。只有 `import torch` 失败时，才按容器 CUDA 版本安装对应 PyTorch；不要盲目覆盖现有版本。

## 2. 下载 OSCAR

```bash
git clone --recursive https://github.com/ava131/OSCAR.git
cd OSCAR
git submodule update --init --recursive
```

OSCAR 已经包含实验所需的 SGLang 源码，不需要另行 clone 外部 SGLang：

```text
sglang-research/
sglang-dump-qkv/
```

## 3. 最小安装本地 SGLang

如果需要使用仓库里的修改版 SGLang，执行：

```bash
python -m pip install --no-deps -e sglang-research/python
```

`--no-deps` 会阻止 pip 重新解析和下载一整套依赖，避免进一步改动 `base`。如果导入时报缺包，再只补装具体缺失的包；不要直接执行无参数的 `pip install -e`。

验证：

```bash
python - <<'PY'
import torch, transformers, sglang
print('torch:', torch.__version__)
print('transformers:', transformers.__version__)
print('sglang:', sglang.__file__)
print('cuda:', torch.cuda.is_available())
PY
```

## 4. 下载模型

`huggingface-cli` 已废弃，使用新版 `hf`。如果环境中没有 `hf`，只安装 CLI 工具本身：

```bash
python -m pip install --user -U huggingface_hub
export PATH="$HOME/.local/bin:$PATH"
hf --help
```

需要权限时登录：

```bash
hf auth login
```

网络不稳定时可先设置镜像和较长超时：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DOWNLOAD_TIMEOUT=600
export HF_HUB_ETAG_TIMEOUT=600
```

下载 Qwen3.5-4B，并支持断点续传：

```bash
mkdir -p /data/models
hf download Qwen/Qwen3.5-4B --local-dir /data/models/Qwen3.5-4B --resume-download
```

下载 OSCAR 的 Rotation Zoo：

```bash
hf download Zhongzhu/OSCAR-RotationZoo --repo-type model --include 'Qwen3.5-4B/**' --local-dir rotzoo --resume-download
```

如果仍然报 `LocalEntryNotFound` 或 `error 110`，这是远端连接超时，不是模型文件名错误。先测试：

```bash
curl -I --max-time 20 "$HF_ENDPOINT/Qwen/Qwen3.5-4B/resolve/main/config.json"
```

若容器不能访问镜像，需要在可联网机器下载后，将模型目录和 `rotzoo` 目录通过 `scp`、共享盘或 Docker volume 放到服务器。

## 5. 启动 SGLang

```bash
chmod +x scripts/start_qwen35_4b_oscar_sglang.sh
MODEL_PATH=/data/models/Qwen3.5-4B ROTATION_ROOT=$PWD/rotzoo/Qwen3.5-4B CUDA_VISIBLE_DEVICES=0,1 TP_SIZE=2 PORT=30000 scripts/start_qwen35_4b_oscar_sglang.sh
```

AISBench 服务地址：`http://127.0.0.1:30000/v1`

V100 不支持 FP8 Tensor Core。若 attention backend 报错，查看参数并必要时使用 Triton decode backend，避免 FP8 KV：

```bash
python -m sglang.launch_server --help | grep -Ei 'attention|kv'
```

```text
--decode-attention-backend triton
```

## 最小命令清单

```bash
conda activate base
git clone --recursive https://github.com/ava131/OSCAR.git
cd OSCAR
python -m pip install --no-deps -e sglang-research/python
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DOWNLOAD_TIMEOUT=600
export HF_HUB_ETAG_TIMEOUT=600
hf download Qwen/Qwen3.5-4B --local-dir /data/models/Qwen3.5-4B --resume-download
hf download Zhongzhu/OSCAR-RotationZoo --repo-type model --include 'Qwen3.5-4B/**' --local-dir rotzoo --resume-download
MODEL_PATH=/data/models/Qwen3.5-4B ROTATION_ROOT=$PWD/rotzoo/Qwen3.5-4B CUDA_VISIBLE_DEVICES=0,1 TP_SIZE=2 PORT=30000 scripts/start_qwen35_4b_oscar_sglang.sh
```

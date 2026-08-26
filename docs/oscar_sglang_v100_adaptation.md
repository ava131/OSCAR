# OSCAR / SGLang 适配 V100：安装与运行踩坑记录

本文根据分支 fix/setup_and_run 的 qas.txt、启动脚本和代码 diff 整理。

## 当前现状与可行性评估

### 当前实际状态

目前需要明确区分两种情况。

当 INT2 关闭时，启动脚本删除了：

~~~text
--kv-cache-dtype int2
--kv-cache-quant-group-size 128
~~~

此时服务可以启动，Prefill、Decode 和 HTTP 接口可以执行；但是输出出现感叹号、乱码或多语言混杂，说明模型数值、attention、position 或 KV 路径仍然异常。因此“整体流程跑通”不等于“模型精度正确”。

当 INT2 开启时，恢复：

~~~text
--kv-cache-dtype int2
--kv-cache-quant-group-size 128
~~~

服务在启动阶段就失败，尚未进入正常推理。因此当前不能表述为“INT2 精度崩溃”，更准确的说法是：

~~~text
V100 + OSCAR INT2 路径无法完成 kernel 初始化、编译或加载。
~~~

Q&A 中出现过的相关现象包括：

~~~text
Failed to load JIT KV-Cache kernel
no kernel image is available for execution on the device
SGLang only supports sm75 and above.
~~~

所以当前有两个独立问题：

1. INT2 关闭时，FP16 服务能启动，但输出精度异常；
2. INT2 开启时，服务在 INT2 kernel 初始化/编译/加载阶段就无法启动。

### V100 能不能实现 INT2

需要区分三件事：

1. V100 可以存储 INT2。INT2 可以打包到 uint8 或 uint32 中，这不要求 Tensor Core 支持 INT2。
2. V100 可以通过软件 CUDA kernel 解包 INT2，反量化到 FP16/FP32，再使用普通 CUDA core 计算。
3. V100 没有可直接利用的原生 INT2 Tensor Core 矩阵乘路径。

NVIDIA Volta/Turing Tensor Core 资料列出了 FP16、INT8、INT4 和二值 INT1 等精度，没有 INT2 Tensor Core 指令。CUTLASS 文档也将窄整数类型概括为 4-bit 和 8-bit，并没有把 INT2 列为通用 Tensor Core 数据类型。因此，V100 上的 INT2 只能依赖软件解包、反量化和普通 CUDA/Triton/PyTorch 计算，而不是原生 INT2 Tensor Core 加速。

这意味着：

~~~text
V100 可以实现 INT2 软件路径；
V100 没有原生 INT2 Tensor Core；
当前 OSCAR INT2 路径不能直接在 V100 上运行。
~~~

### 是否值得继续实现

理论上可以重写一个 V100 兼容的 reference INT2 backend：

~~~text
packed INT2 KV
    -> 自定义 sm70 CUDA kernel 解包
    -> FP16/FP32 反量化
    -> Triton 或 PyTorch attention
~~~

但需要重新确认或实现：

- INT2 packing 和 unpacking；
- scale、zero point 和 group size 128；
- Hadamard rotation 的顺序和 layout；
- K/V cache 写入与读取；
- attention 中的 dequant；
- sm70 编译目标；
- Qwen3.5 hybrid attention/Mamba 的接口。

当前最现实的目标是“让 INT2 软件路径启动并保持可接受精度”，不是获得原生 INT2 加速。建议验收顺序是：

1. INT2 kernel 能初始化；
2. INT2 backend 能输出合理文本；
3. INT2 与 FP16 baseline 的 logits/困惑度/任务准确率接近；
4. 最后再测速度和显存收益。

可行性判断：

| 目标 | 判断 |
|---|---|
| 当前 OSCAR INT2 参数在 V100 上直接启动 | 低概率 |
| 修复当前 INT2 kernel 使其支持 sm70 | 有可能，但需要较大改动 |
| 自己写软件解包/反量化 INT2 路径 | 可行 |
| 保留 INT2 的显存压缩效果 | 可行 |
| 获得原生 INT2 加速 | 不可行 |
| 得到稳定且高精度的 OSCAR INT2 | 未知，需要独立实验 |

估计工作量：定位 kernel 失败位置约半天到 1 天；写出可运行的 sm70 reference path 约 1 到 3 天；接入 SGLang KV cache 约 3 到 7 天；精度和性能验证约 1 到 2 周。如果 Qwen3.5 的 hybrid attention、rotation 或 position 仍有语义问题，时间会更长。

因此，当前应先把 INT2 关闭，解决 FP16 baseline 的乱码问题；之后再把 INT2 作为独立实验分支排查。否则 FP16 数值问题和 INT2 kernel 问题会相互干扰。

约定：
- “实际采用”表示能从该分支最终代码或运行脚本确认。
- “备选方案”表示 Q&A 讨论过，但最终代码不能确认。
- 目标环境是 2x V100、Qwen3.5-4B、OSCAR vendored SGLang。

## 1. 总流程

实际排障顺序：

1. 创建 Python 3.11 的项目虚拟环境。
2. 使用 uv/pip 安装项目依赖。
3. 处理 Docker、网络、证书和包管理器问题。
4. 处理 CUDA 编译工具和 nvcc。
5. 处理 V100 的 sm70 与新 SGLang kernel 不兼容。
6. 关闭 V100 上不稳定或不支持的 backend。
7. 修改 JIT、RMSNorm、采样、KV cache 和模型代码。
8. 让服务完成 Prefill/Decode。
9. 用 curl 验证 OpenAI 兼容接口。

## 2. 环境配置问题

### 2.1 uv、pip 和虚拟环境路径混乱

关键报错：

~~~text
bash: uv: command not found
bash: pip: command not found
~~~

原因：

- uv 实际安装在另一个 Docker 容器；
- 切换用户或进入新容器后，原来的 /root/miniconda3/bin/uv 不存在；
- uv 创建的 .venv 默认不一定包含 pip 命令；
- 用户级安装目录没有加入当前 shell 的 PATH。

排查和恢复：

~~~bash
whoami
echo "$HOME"
echo "$PATH"
find / -name uv -type f 2>/dev/null
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
python -m ensurepip --upgrade
~~~

分支日志确认最终使用 Python 3.11：

~~~text
/home/z50058184/OSCAR-V100/OSCAR/.venv/lib/python3.11/
~~~

确认环境：

~~~bash
which python
python --version
python -c 'import sys; print(sys.executable); print(sys.prefix)'
~~~

实际采用：项目 .venv；uv 和 pip 根据所在容器/用户分别恢复。

### 2.2 Conda SSL / 网关问题

关键报错：

~~~text
CondaSSLError: Encountered an SSL error.
HTTPSConnectionPool(host='repo.anaconda.com', port=443)
SSLError: WRONG_VERSION_NUMBER
~~~

原因通常是错误代理、网关 TLS 拦截或证书链问题。

讨论过的处理：

~~~bash
unset http_proxy https_proxy all_proxy
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY
~~~

也讨论过换 Anaconda 镜像，但最终路线没有继续依赖 Conda，而是使用项目 .venv、uv 和 pip。

实际结论：只要项目虚拟环境能创建，后续 Python 包不需要 Conda。

### 2.3 PyPI / CUDA 包镜像

Q&A 中使用过 uv 配置：

~~~toml
allow-insecure-host = ["mirrors.aliyun.com", "download.pytorch.org"]

[pip]
index-url = "http://mirrors.aliyun.com/pypi/simple/"
~~~

也使用过 pip 的镜像参数：

~~~bash
python -m pip install PACKAGE -i https://pypi.tuna.tsinghua.edu.cn/simple
~~~

这是网络 workaround。HTTP 镜像会降低安全性，长期应优先使用 HTTPS 镜像或修复 CA 证书。

### 2.4 CUDA Toolkit / nvcc 不存在

关键现象：

~~~text
which nvcc
# 没有输出

FileNotFoundError: ... /usr/local/cuda/bin/nvcc
~~~

尝试过：

~~~bash
apt-get update
apt-get install -y cuda-toolkit-12-*
apt-get install -y cuda-nvcc-12-1 cuda-compiler-12-1
~~~

但 APT 源没有 NVIDIA CUDA 仓库时：

~~~text
Unable to locate package cuda-nvcc-12-1
~~~

也尝试：

~~~bash
uv pip install nvidia-cuda-nvcc-cu12
~~~

但该 PyPI 拆分包没有可靠提供完整 nvcc，只能看到 ptxas 等部分工具。

实际结论：

- nvidia-cuda-nvcc-cu12 不能可靠替代完整 CUDA Toolkit；
- 需要编译 CUDA 扩展时，应使用带 CUDA development tools 的 Docker 镜像，或安装完整 Toolkit；
- 分支最终通过修改 JIT 路径绕过部分编译需求。

检查：

~~~bash
which nvcc || true
nvcc --version || true
ls -l /usr/local/cuda/bin/nvcc 2>/dev/null || true
echo "$CUDA_HOME"
~~~

### 2.5 nvcc 版本和参数不匹配

关键报错：

~~~text
nvcc: Unknown option '--generate-dependencies-with-compile'
~~~

以及 nvcc 编译失败、架构不匹配等。

讨论过的处理：

- 确认真正使用的 nvcc 路径；
- 设置 CUDA_HOME 和 PATH；
- 编译目标使用 compute_70 / sm_70；
- 部分 SGLang JIT 编译参数从 C++20/C++17 调整为 C++14。

分支最终确认的修改：

~~~text
sglang-research/python/sglang/jit_kernel/utils.py
DEFAULT_CFLAGS: -std=c++20 -> -std=c++14
CUDA target flags: -std=c++20 -> -std=c++14
~~~

### 2.6 C++14 与 C++20 头文件冲突

之后遇到：

~~~text
fatal error: version: No such file or directory
#include <version>
~~~

说明统一降到 C++14 可能会与代码使用的 C++ 标准库特性冲突。应区分 CUDA kernel 的编译标准、SGLang include 使用的标准库特性，以及 GCC/nvcc 版本。

分支保留了 C++14 修改，但 Q&A 没有证明它是一个完全通用的最终解法，部署时应记录 GCC、nvcc 和 CUDA 版本。

## 3. 代码与硬件不兼容问题

### 3.1 SGLang 拒绝 sm75 以下 GPU

关键报错：

~~~text
RuntimeError: SGLang only supports sm75 and above.
~~~

V100 是 sm70。分支修改：

~~~text
sglang-research/python/sglang/srt/model_executor/model_runner.py
~~~

实际改为跳过显式 RuntimeError。

这只是允许程序继续启动，并不会让所有底层 kernel 自动支持 sm70。

### 3.2 bfloat16 / fused kernel 的 no kernel image

关键报错：

~~~text
torch.AcceleratorError:
CUDA error: no kernel image is available for execution on the device
~~~

最初定位在：

~~~python
x = x + residual
~~~

即使打印：

~~~text
x.dtype: torch.float16
residual.dtype: torch.float16
~~~

仍可能出错，因为 CUDA 异步执行，真正失败的 kernel 可能在前面；也可能是为 sm80/sm90 编译的 attention、RMSNorm、FlashInfer 或 Triton kernel。

Q&A 讨论过 CPU 转换：

~~~python
x = x.cpu().to(torch.float16).to(x.device)
residual = residual.cpu().to(torch.float16).to(residual.device)
~~~

分支最终在 layernorm.py 采用更激进的 fallback：

- residual 加法前 contiguous；
- 异常时 CPU FP32 加法，再转 GPU FP16；
- bfloat16 权重先 CPU 转 FP16，再回 GPU；
- 用纯 PyTorch RMSNorm 替代原实现。

代价是性能下降和可能的数值行为变化。

### 3.3 clamp_position CUDA JIT 不兼容

分支修改：

~~~text
sglang-research/python/sglang/jit_kernel/clamp_position.py
~~~

原实现依赖 JIT CUDA module，最终改为纯 PyTorch：

~~~python
return torch.cat([
    torch.arange(int(s), device=seq_lens.device)
    for s in seq_lens
])
~~~

实际采用：绕过该位置的 JIT 编译和 nvcc。

注意：需要回归检查 decode position 是否仍然是绝对位置，不能因为每次重新构造 arange 而导致位置错位。

### 3.4 JIT KV-cache kernel 编译失败

关键日志：

~~~text
Failed to load JIT KV-Cache kernel with row_bytes=2048: ninja
~~~

相关原因包括 nvcc 缺失、nvcc 版本过旧、C++ 标准不兼容、生成参数不支持和缺少 sm70 代码。

分支没有完整修复全部 CUDA 编译链，而是配合启动参数和代码 fallback 让服务继续运行。

### 3.5 INT2 KV cache 与 Hadamard rotation

原启动参数：

~~~text
--kv-cache-dtype int2
--kv-cache-quant-group-size 128
~~~

Q&A 观察到：Hadamard rotation kernel 未成功编译时，INT2 KV cache 误差巨大，最终输出乱码。

可选方案：

A. 修复 Hadamard kernel 和 rotation 路径，保留 INT2。
B. 删除 INT2 参数，先使用默认 FP16 KV cache 验证模型。

从最终启动脚本确认，实际采用 B：

~~~text
删除 --kv-cache-dtype int2
删除 --kv-cache-quant-group-size 128
~~~

### 3.6 FA3 不适合 V100

原参数：

~~~text
--prefill-attention-backend fa3
~~~

分支最终改为：

~~~text
--prefill-attention-backend triton
--decode-attention-backend triton
~~~

所以实际采用 Triton attention。

Q&A 还讨论过 torch_native 或 pytorch backend，但最终脚本确认使用 Triton；PyTorch backend 属于备选 fallback。

### 3.7 关闭 CUDA Graph、Radix Cache 和 overlap schedule

最终新增：

~~~text
--disable-cuda-graph
--disable-radix-cache
--disable-overlap-schedule
--sampling-backend pytorch
--tokenizer-worker-num 1
~~~

并设置：

~~~bash
export CUDA_LAUNCH_BLOCKING=1
~~~

实际含义：

- 关闭 CUDA Graph；
- 关闭 Radix Cache；
- 关闭 overlap schedule；
- 使用 PyTorch sampling；
- tokenizer worker 设为 1；
- 尽量把异步错误定位到实际位置。

### 3.8 NCCL P2P / IB 不稳定

分支新增：

~~~bash
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_NET=Socket
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export NCCL_SHM_DISABLE=1
~~~

实际采用 Socket/loopback 通信，关闭 IB、P2P 和共享内存路径。

注意：run.sh 中是：

~~~text
CUDA_VISIBLE_DEVICES=0,1
TP_SIZE=1
~~~

这并不等于明确使用两路 tensor parallel。若目标是 2-way TP，应单独验证 TP_SIZE=2。

### 3.9 HybridLinearKVPool 参数接口不一致

关键报错：

~~~text
TypeError: HybridLinearKVPool.set_kv_buffer()
got an unexpected keyword argument 'is_decode'
~~~

原因是调用方传入 is_decode，而目标方法签名不接收。

分支实际修改方法签名，加入：

~~~python
*args, **kwargs
~~~

备选是删除调用方的 is_decode，或显式添加 is_decode=False。根据最终 diff，实际采用兼容额外参数的方案。

### 3.10 LayerNorm / RMSNorm 改为纯 PyTorch

GemmaRMSNorm.forward_cuda 被改为：

- contiguous 后做 residual 加法；
- variance 使用 x.pow(2).mean；
- rsqrt；
- 普通乘法；
- 必要时 CPU fallback。

目标是绕开 V100 不支持的 fused RMSNorm、bfloat16 或 Triton kernel。

代价是性能和数值行为可能变化。

### 3.11 sampler 的 device-side assert

关键报错：

~~~text
torch.AcceleratorError:
CUDA error: device-side assert triggered
sampled_index = torch.multinomial(probs, num_samples=1)
~~~

Q&A 判断可能是 NaN/Inf、概率全零或概率和非法。

分支实际在 sampler.py 加入：

~~~python
probs = torch.nan_to_num(
    probs, nan=0.0, posinf=0.0, neginf=0.0
)
prob_sum = probs.sum(dim=-1, keepdim=True)
zero_mask = prob_sum <= 0
if zero_mask.any():
    probs = torch.where(
        zero_mask, torch.ones_like(probs), probs
    )
    prob_sum = probs.sum(dim=-1, keepdim=True)
probs = probs / prob_sum
sampled_index = torch.multinomial(probs, num_samples=1)
~~~

备选是请求侧 temperature=0，让服务走 greedy/argmax。最终分支保留了概率清洗，因此实际采用 sampler 兜底。

### 3.12 输出为感叹号或多语言乱码

关键输出：

~~~text
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
~~~

以及：

~~~text
sao Crushardador...想了解...lecimana...
~~~

这表示 HTTP、tokenizer、Prefill/Decode 已经跑通，但模型数值或 KV cache 语义不正确。

Q&A 提出过：

- FP16 logits 溢出；
- Triton attention 缩放或 sm70 兼容性；
- Qwen3.5 混合注意力/Mamba position 错位；
- INT2 没有正确使用 Hadamard rotation；
- RoPE/position_ids 错位。

最终代码确认的处理：

- 删除 INT2 KV cache 参数；
- 使用 FP16；
- Triton prefill/decode；
- 关闭 CUDA Graph、Radix Cache、overlap schedule；
- 修改 clamp_position；
- 修改 layernorm、sampler 和部分 dtype 路径。

Q&A 建议把 lm_head logits 转 FP32，但最终 diff 不能确认该修改已采用，因此标为备选方案。

### 3.13 Mamba causal conv dtype

分支修改：

~~~text
sglang-research/python/sglang/srt/layers/attention/mamba/causal_conv1d_triton.py
~~~

在 tl.load 后显式 .to(tl.float16)，说明 Mamba causal convolution 也遇到了 dtype 或 kernel 推断兼容问题。

### 3.14 Qwen3 VL position embedding CPU fallback

分支修改 qwen3_vl.py：

- position embedding 权重搬到 CPU FP32；
- 坐标插值在 CPU；
- embedding lookup 和加权求和在 CPU；
- 结果转 GPU FP16；
- 部分视觉/语言权重由 bfloat16 转 FP16。

纯文本 Qwen3.5-4B 是否会走该路径，需要根据模型配置确认。

## 4. 分支最终运行配置

scripts/start_qwen35_4b_oscar_sglang.sh 确认包含：

~~~bash
export TORCH_EXTENSIONS_DIR=/tmp/torch_extensions
export TMPDIR=/tmp
export RAY_TMPDIR=/tmp
export CUDA_LAUNCH_BLOCKING=1
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
~~~

启动参数确认包含：

~~~text
--dtype float16
--prefill-attention-backend triton
--decode-attention-backend triton
--disable-cuda-graph
--disable-radix-cache
--sampling-backend pytorch
--tokenizer-worker-num 1
--disable-overlap-schedule
~~~

确认删除：

~~~text
--kv-cache-dtype int2
--kv-cache-quant-group-size 128
--prefill-attention-backend fa3
~~~

## 5. 仍需核对

### 5.1 TP_SIZE

run.sh 使用 TP_SIZE=1，尽管可见 GPU 是 0,1。要做 2-way tensor parallel，需要确认改为 TP_SIZE=2 是否稳定。

### 5.2 Rotation 文件

不要求一定存在 rotations 目录，但必须找到：

~~~text
k_rotation_qqt_r_h_pbr.pt
v_rotation_sst_r_h_pbr.pt
~~~

检查：

~~~bash
find rotzoo -type f \(   -name 'k_rotation_qqt_r_h_pbr.pt' -o   -name 'v_rotation_sst_r_h_pbr.pt' \) -print
~~~

### 5.3 输出质量

服务返回 HTTP 200 不代表精度正确。AISBench 前至少验证：

- temperature=0；
- temperature>0；
- 中英文 prompt；
- 单轮/多轮；
- 不同长度；
- TP=1/TP=2；
- Triton/PyTorch attention backend。

## 6. 推荐复现命令

~~~bash
cd /home/z50058184/OSCAR-V100/OSCAR
source .venv/bin/activate
nvidia-smi

python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.device_count())'

find rotzoo -type f -name '*rotation*.pt'

bash scripts/start_qwen35_4b_oscar_sglang.sh
~~~

英文 greedy 请求：

~~~bash
curl http://127.0.0.1:6068/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-4b",
    "messages": [
      {"role": "user", "content": "Hello, please introduce yourself in one short sentence."}
    ],
    "max_tokens": 50,
    "temperature": 0.0
  }'
~~~

## 7. 结论

这次实际采用的是稳定性优先路线：

- 项目 .venv，uv/pip；
- FP16，放弃 INT2；
- FA3 改 Triton；
- 关闭 CUDA Graph、Radix Cache、overlap schedule；
- 绕过部分 JIT；
- C++ 编译标准改 C++14；
- 放宽 sm75 检查；
- LayerNorm/RMSNorm 纯 PyTorch fallback；
- sampler 概率清洗；
- HybridLinearKVPool 接收额外参数；
- Mamba load 后转 FP16；
- NCCL 禁用 IB/P2P，使用 Socket。

这些修改让服务更可能在 V100 上跑通，但不等于已经恢复原始 OSCAR 的 INT2 精度、吞吐和数值行为。上传前应先确认 TP 配置和输出质量。

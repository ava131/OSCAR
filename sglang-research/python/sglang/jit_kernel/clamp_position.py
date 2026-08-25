from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_clamp_position_module(dtype: torch.dtype) -> Module:
    """Compile and cache the JIT clamp_position module for a given dtype."""
    args = make_cpp_args(dtype)
    return load_jit(
        "clamp_position",
        *args,
        cuda_files=["elementwise/clamp_position.cuh"],
        cuda_wrappers=[
            ("clamp_position", f"ClampPosition<{args}>::run"),
        ],
    )


def clamp_position_cuda(seq_lens: torch.Tensor) -> torch.Tensor:
    # 纯 PyTorch 实现，彻底绕过 JIT 编译和 nvcc
    return torch.cat([torch.arange(int(s), device=seq_lens.device) for s in seq_lens])

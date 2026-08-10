import os
from typing import Any

import torch

from sglang.srt.environ import envs
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.utils import get_bool_env_var, is_hcu, is_hip

_is_hip = is_hip()
_is_hcu = is_hcu()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

if _use_aiter:
    from aiter.tuned_gemm import tgemm

_linear_bf16_fp32_algo = envs.SGLANG_OPT_BF16_FP32_GEMM_ALGO.get()

_jit_sgl_gemm_module_cache = None


def _jit_sgl_gemm_module() -> Any:
    """JIT-compile the sgl bf16->fp32 GEMM (n16 K-split + n128 N-split) from
    csrc/deepseek_v4/gemm.cuh. ROCm/HIP only (AMD MFMA). For the router/wkv_gate
    decode path."""
    global _jit_sgl_gemm_module_cache
    if _jit_sgl_gemm_module_cache is not None:
        return _jit_sgl_gemm_module_cache
    import torch.utils.cpp_extension

    csrc_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "csrc")
    with open(os.path.join(csrc_dir, "deepseek_v4", "gemm.cuh")) as f:
        cuh = f.read()
    source = (
        "#include <torch/extension.h>\n"
        + cuh
        + "\nPYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {"
        ' m.def("gemm_opt_fp32", &gemm_opt_fp32, "bf16->fp32 GEMM (sgl n16/n128)"); }\n'
    )
    module = torch.utils.cpp_extension.load_inline(
        name="sgl_bf16_fp32_gemm",
        cpp_sources="",
        cuda_sources=source,
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )
    _jit_sgl_gemm_module_cache = module
    return module


def linear_bf16_fp32(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if _is_hcu and _linear_bf16_fp32_algo == "deep_gemm":
        z = torch.empty(x.size(0), y.size(0), dtype=torch.float32, device=x.device)
        deep_gemm_wrapper.gemm_nt_bf16bf16f32(x, y, z)
        return z
    elif _use_aiter and _is_hcu:
        return tgemm.mm(x, y, otype=torch.float32)
    elif _use_aiter:
        return tgemm.mm(x, y, otype=x.dtype).float()
    elif _linear_bf16_fp32_algo == "deep_gemm":
        z = torch.empty(x.size(0), y.size(0), dtype=torch.float32, device=x.device)
        deep_gemm_wrapper.gemm_nt_bf16bf16f32(x, y, z)
        return z
    elif _linear_bf16_fp32_algo == "sgl":
        # sgl bf16->fp32 GEMM: bf16 [M,K] x bf16 [N,K]^T -> fp32 [M,N].
        # Shape-directed: only router/wkv_gate decode shapes (M<=64, N in
        # {256,1024,2048}) benefit from the sgl MFMA kernel. Large M (prefill)
        # and large N (logits/vocab) are bandwidth-limited in the sgl kernel
        # -> fall back to the cublas path. JIT-compiled from
        # csrc/deepseek_v4/gemm.cuh (AMD MFMA, ROCm only).
        _M, _K = x.shape
        _N = y.shape[0] if y.stride(1) == 1 else y.shape[1]
        if _M <= 64 and _N in (256, 1024, 2048):
            return _jit_sgl_gemm_module().gemm_opt_fp32(x, y)
        return torch.mm(x, y.t(), out_dtype=torch.float32)
    else:
        return torch.mm(x, y.t(), out_dtype=torch.float32)

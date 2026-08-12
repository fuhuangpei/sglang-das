import os
from typing import Any

import torch

from sglang.srt.environ import envs
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.utils import get_bool_env_var, is_hcu, is_hip

_is_hip = is_hip()
_is_hcu = is_hcu()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
_use_linear_bf16_fp32_use_blaslt = get_bool_env_var(
    "SGLANG_USE_LINEAR_BF16_FP32_USE_BLASLT"
)

if _use_aiter:
    from aiter.tuned_gemm import tgemm

_linear_bf16_fp32_algo = envs.SGLANG_OPT_BF16_FP32_GEMM_ALGO.get()

_cublas_bf16_fp32_module_cache = {}


def _jit_torch_cublas_bf16_fp32(use_blaslt=None) -> Any:
    import torch.utils.cpp_extension

    # use_blaslt=None -> follow the env var (backward compatible);
    # pass an explicit bool to force a specific variant (used by "auto" dispatch,
    # which needs both the cublasGemmEx and hipblasLt modules available at once).
    if use_blaslt is None:
        use_blaslt = _use_linear_bf16_fp32_use_blaslt
    want_blaslt = bool(_is_hcu and use_blaslt)
    cached = _cublas_bf16_fp32_module_cache.get(want_blaslt)
    if cached is not None:
        return cached

    if want_blaslt:
        source = """
        #include <torch/extension.h>
        #include <ATen/cuda/CUDAContext.h>
        #include <hipblaslt/hipblaslt.h>

        torch::Tensor linear_bf16_fp32(
            torch::Tensor X,
            torch::Tensor W)
        {
            int batch = X.size(0);
            int in_features = X.size(1);
            int out_features = W.size(0);

            auto Y = torch::empty(
                {batch, out_features},
                torch::dtype(torch::kFloat32).device(X.device()));

            static thread_local hipblasLtHandle_t handle = nullptr;
            if (handle == nullptr) {
                hipblasLtCreate(&handle);
            }

            hipblasLtMatmulDesc_t matmul_desc;
            hipblasLtMatmulDescCreate(&matmul_desc, HIPBLAS_COMPUTE_32F, HIP_R_32F);

            int transA = HIPBLAS_OP_T;
            int transB = HIPBLAS_OP_N;
            hipblasLtMatmulDescSetAttribute(matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA));
            hipblasLtMatmulDescSetAttribute(matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB));

            hipblasLtMatrixLayout_t layoutA, layoutB, layoutC;
            hipblasLtMatrixLayoutCreate(&layoutA, HIP_R_16BF, in_features, out_features, in_features);
            hipblasLtMatrixLayoutCreate(&layoutB, HIP_R_16BF, in_features, batch, in_features);
            hipblasLtMatrixLayoutCreate(&layoutC, HIP_R_32F, out_features, batch, out_features);

            float alpha = 1.0f;
            float beta = 0.0f;

            hipStream_t stream = at::cuda::getCurrentCUDAStream().stream();

            hipblasLtMatmul(
                handle,
                matmul_desc,
                &alpha,
                W.data_ptr(), layoutA,
                X.data_ptr(), layoutB,
                &beta,
                Y.data_ptr(), layoutC,
                Y.data_ptr(), layoutC,
                nullptr,
                nullptr,
                0,
                stream
            );

            hipblasLtMatmulDescDestroy(matmul_desc);
            hipblasLtMatrixLayoutDestroy(layoutA);
            hipblasLtMatrixLayoutDestroy(layoutB);
            hipblasLtMatrixLayoutDestroy(layoutC);

            return Y;
        }

        PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
            m.def("linear_bf16_fp32", &linear_bf16_fp32, "BF16xBF16 -> FP32 linear using hipblasLt");
        }
        """
    else:
        source = """
    #include <torch/extension.h>
    #include <ATen/cuda/CUDAContext.h>
    #include <cublas_v2.h>

    torch::Tensor linear_bf16_fp32(
        torch::Tensor X,
        torch::Tensor W)
    {
        int batch = X.size(0);
        int in_features = X.size(1);
        int out_features = W.size(0);

        auto Y = torch::empty(
            {batch, out_features},
            torch::dtype(torch::kFloat32).device(X.device()));

        cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();

        float alpha = 1.0f;
        float beta = 0.0f;

        cublasGemmEx(
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            out_features,
            batch,
            in_features,
            &alpha,
            W.data_ptr(), CUDA_R_16BF, in_features,
            X.data_ptr(), CUDA_R_16BF, in_features,
            &beta,
            Y.data_ptr(), CUDA_R_32F, out_features,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        );

        return Y;
    }

    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("linear_bf16_fp32", &linear_bf16_fp32, "BF16xBF16 -> FP32 linear (no bias)");
    }
    """

    module = torch.utils.cpp_extension.load_inline(
        name="linear_bf16_fp32_hipblaslt" if want_blaslt else "linear_bf16_fp32_cublas",
        cpp_sources="",
        cuda_sources=source,
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )
    _cublas_bf16_fp32_module_cache[want_blaslt] = module
    return module


_jit_sgl_gemm_module_cache = None


def _jit_sgl_gemm_module() -> Any:
    """JIT-compile the sgl bf16->fp32 GEMM (n16 K-split + splitK N64) from
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
        ' m.def("gemm_opt_fp32", &gemm_opt_fp32, "bf16->fp32 GEMM (sgl n16/splitK)"); }\n'
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


_linear_bf16_fp32_trace_seen = set()
_LINEAR_BF16_FP32_TRACE_PATH = os.environ.get("SGLANG_LINEAR_BF16_FP32_TRACE", "")


def _trace_linear_bf16_fp32(x: torch.Tensor, y: torch.Tensor) -> None:
    # Lightweight shape/dtype tracer for the linear_bf16_fp32 interface.
    # Enabled by setting SGLANG_LINEAR_BF16_FP32_TRACE=<output_jsonl_path>.
    # Records one line per unique (caller, shape, dtype) tuple so we can
    # collect the real prefill call cases for offline benchmarking.
    if not _LINEAR_BF16_FP32_TRACE_PATH:
        return
    import sys

    frame = sys._getframe(2)
    caller = f"{frame.f_code.co_filename}:{frame.f_lineno}"
    m, k = int(x.shape[0]), int(x.shape[1])
    n = int(y.shape[0]) if y.stride(1) == 1 else int(y.shape[1])
    key = (caller, m, k, n, str(x.dtype), str(y.dtype))
    if key in _linear_bf16_fp32_trace_seen:
        return
    _linear_bf16_fp32_trace_seen.add(key)
    import json

    rec = {
        "caller": caller,
        "M": m,
        "K": k,
        "N": n,
        "x_dtype": str(x.dtype),
        "y_dtype": str(y.dtype),
        "y_contig": bool(y.is_contiguous()),
        "y_shape": list(y.shape),
        "y_stride": list(y.stride()),
        "pid": os.getpid(),
    }
    # one file per worker process to avoid interleaved appends across TP ranks
    with open(f"{_LINEAR_BF16_FP32_TRACE_PATH}.{os.getpid()}", "a") as f:
        f.write(json.dumps(rec) + "\n")


def _cublas_gemmex(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return _jit_torch_cublas_bf16_fp32(use_blaslt=False).linear_bf16_fp32(x, y)


def _hipblaslt(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return _jit_torch_cublas_bf16_fp32(use_blaslt=True).linear_bf16_fp32(x, y)


def _torch_fp32(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.linear(x.float(), y.float())


def _auto_dispatch_bf16_fp32(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Shape-directed dispatch to the fastest bf16->fp32 GEMM backend per call site.

    On DCU, no single backend is uniformly fastest for the DeepSeek-V4 prefill
    bf16->fp32 GEMMs: hipblasLt is great for the compressor wkv_gate (N=1024/2048)
    but catastrophically slow for the router (N=256) at prefill M (up to ~24x),
    while cublasGemmEx is the opposite. The call site is uniquely identified by N
    (256 = MoEGate router; 512 = indexer compressor wkv_gate; 1024/2048 = attention
    compressor wkv_gate). This table encodes the measured-optimal backend per (N, M)
    from mission_linear_bf16_fp32/results/bench_backends_confirm.csv.

    Non-DCU builds fall back to cublasGemmEx (no hipblasLt available).
    """
    M, K = x.shape
    N = int(y.shape[0]) if y.stride(1) == 1 else int(y.shape[1])

    # decode / small-M: the sgl MFMA kernel dominates for its supported shapes
    if M <= 64 and N in (256, 1024, 2048):
        return _jit_sgl_gemm_module().gemm_opt_fp32(x, y)

    if not _is_hcu:
        return _cublas_gemmex(x, y)

    # prefill (M > 64), routed by call site (N):
    if N == 256:  # MoEGate router: hipblasLt cliffs badly at large M; cublas is flat/robust
        return _torch_fp32(x, y) if M <= 768 else _cublas_gemmex(x, y)
    if N == 512:  # indexer compressor wkv_gate: hipblasLt cliffs at mid M; cublas great at large M
        return _torch_fp32(x, y) if M <= 1024 else _cublas_gemmex(x, y)
    if N == 1024:  # attn compressor wkv_gate (compress_ratio=128)
        return _torch_fp32(x, y) if M <= 768 else _hipblaslt(x, y)
    if N == 2048:  # attn compressor wkv_gate (compress_ratio=4)
        return _hipblaslt(x, y)

    # unknown shape: cublasGemmEx is the most robust generic choice on DCU
    return _cublas_gemmex(x, y)


def linear_bf16_fp32(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    _trace_linear_bf16_fp32(x, y)
    if _linear_bf16_fp32_algo == "auto":
        return _auto_dispatch_bf16_fp32(x, y)
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

#pragma once
// bf16 -> fp32 GEMM kernels for DeepSeek V4 router(N=256)/wkv_gate(N=1024,2048) decode.
// AMD MFMA (gfx936/928/938).
// n16 = K-split across warps + smem tree reduce + direct store (deterministic, 1 node).
// n128 = N-split across warps (large N, A reuse).
// ROCm-only; compiled via torch.utils.cpp_extension (hipify converts <<<>>> / at::cuda).
#if defined(__HIP_PLATFORM_HCC__) || defined(__HIP_PLATFORM_AMD__)

#include <ATen/ATen.h>
#include <c10/macros/Macros.h>
#include <c10/hip/HIPStream.h>

using half2_t = __attribute__((__vector_size__(2 * sizeof(_Float16)))) _Float16;
using half4_t = __attribute__((__vector_size__(4 * sizeof(_Float16)))) _Float16;
using v4bh = __attribute__((__vector_size__(4 * sizeof(short)))) short;
using float4_t = __attribute__((__vector_size__(4 * sizeof(float)))) float;
struct half4x2 { half4_t data[2]; };

template<bool is_half = true>
inline __device__ void builtin_amdgcn_mmac(const half4_t& reg_a, const half4_t& reg_b, float4_t& reg_c) {
    #if defined(__gfx936__) || defined(__gfx928__)
        if constexpr (is_half) reg_c = __builtin_amdgcn_mmac_f32_16x16x16f16(reg_a, reg_b, reg_c);
        else reg_c = __builtin_amdgcn_mmac_f32_16x16x16bf16(*(v4bh*)&reg_a, *(v4bh*)&reg_b, reg_c);
    #elif defined(__gfx938__)
        if constexpr (is_half) reg_c = __builtin_hcu_mmac_f32_16x16x16_f16_lit_lts(reg_a, reg_b, reg_c, false, false);
        else reg_c = __builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts(*(v4bh*)&reg_a, *(v4bh*)&reg_b, reg_c, false, false);
    #endif
}

// 16x16 tile, 4 warps split K (each K/4) + smem tree reduce + direct fp32 store.
template <typename scalar_t, int NUM_WARPS = 4, int NPerBlock = 16>
__global__ void gemm_nt_fp16_fp32out(scalar_t *a, scalar_t *b, float *d, int m, int n, int k) {
    const int bid_x = blockIdx.x;
    const int bid_y = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp_idx = __builtin_amdgcn_readfirstlane(tid / C10_WARP_SIZE);
    const int lane = tid % C10_WARP_SIZE;
    const int rowid = lane % 16;
    const int rows = lane / 16;
    constexpr bool is_half = std::is_same<scalar_t, at::Half>::value;
    const int row = bid_y * 16 + rowid;
    const int col = bid_x * 16 + rowid;
    int k_off = warp_idx * 16 * 2 + rows * 4 * 2;
    int offset_a = row * k + k_off;
    int offset_b = col * k + k_off;
    half4x2 a_vec, b_vec;
    float4_t d_vec = {0, 0, 0, 0};
    a_vec.data[0] = {0, 0, 0, 0};
    a_vec.data[1] = {0, 0, 0, 0};
    a += offset_a;
    b += offset_b;
    d += NPerBlock * bid_x + row * n + rows;
    for (int i = 0; i + 16 * 2 * warp_idx < k; i += 16 * 2 * NUM_WARPS) {
        if (row < m) a_vec = *(half4x2 *)(a + i);
        b_vec = *(half4x2 *)(b + i);
        builtin_amdgcn_mmac<is_half>(a_vec.data[0], b_vec.data[0], d_vec);
        builtin_amdgcn_mmac<is_half>(a_vec.data[1], b_vec.data[1], d_vec);
    }
    if constexpr (NUM_WARPS > 1) {
        extern __shared__ float4_t out_smem[];
        if (row < m) {
            #pragma unroll
            for (int i = NUM_WARPS; i > 1; i /= 2) {
                int mid = i / 2;
                if (warp_idx >= mid && warp_idx < i) {
                    out_smem[(warp_idx - mid) * 4 * 16 + rowid * 4 + rows] = d_vec;
                }
                __syncthreads();
                if (warp_idx < mid) {
                    float4_t tmp = out_smem[warp_idx * 4 * 16 + rowid * 4 + rows];
                    #pragma unroll
                    for (int i = 0; i < 4; i++) d_vec[i] += tmp[i];
                }
                __syncthreads();
            }
        }
    }
    if (row < m && warp_idx == 0) {
        for (int i = 0; i < 4; i++) d[i * 4] = d_vec[i];  // fp32 direct store
    }
}

// 16x128 tile, 4 warps split N (each 16x32 = 2 MFMA), A reused across both MFMA.
template <typename scalar_t, int NUM_WARPS = 4, int NPerBlock = 128>
__global__ void gemm_nt_fp16_fp32out_n128(scalar_t *a, scalar_t *b, float *d, int m, int n, int k) {
    constexpr int N_PER_WARP = NPerBlock / NUM_WARPS;
    const int bid_x = blockIdx.x;
    const int bid_y = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp_idx = __builtin_amdgcn_readfirstlane(tid / C10_WARP_SIZE);
    const int lane = tid % C10_WARP_SIZE;
    const int rowid = lane % 16;
    const int rows = lane / 16;
    constexpr bool is_half = std::is_same<scalar_t, at::Half>::value;
    const int row = bid_y * 16 + rowid;
    const int col_warp = bid_x * NPerBlock + warp_idx * N_PER_WARP;
    const int col0 = col_warp + rowid;
    const int col1 = col_warp + 16 + rowid;
    int k_off = rows * 4 * 2;
    scalar_t *a_ptr = a + row * k + k_off;
    scalar_t *b_ptr0 = b + col0 * k + k_off;
    scalar_t *b_ptr1 = b + col1 * k + k_off;
    float *d_ptr0 = d + row * n + col_warp + rows;
    float *d_ptr1 = d + row * n + col_warp + 16 + rows;
    float4_t d0 = {0, 0, 0, 0}, d1 = {0, 0, 0, 0};
    half4x2 a_vec, b0, b1;
    a_vec.data[0] = {0, 0, 0, 0};
    a_vec.data[1] = {0, 0, 0, 0};
    for (int i = 0; i + 32 <= k; i += 32) {
        if (row < m) a_vec = *(half4x2 *)(a_ptr + i);
        b0 = *(half4x2 *)(b_ptr0 + i);
        b1 = *(half4x2 *)(b_ptr1 + i);
        builtin_amdgcn_mmac<is_half>(a_vec.data[0], b0.data[0], d0);
        builtin_amdgcn_mmac<is_half>(a_vec.data[1], b0.data[1], d0);
        builtin_amdgcn_mmac<is_half>(a_vec.data[0], b1.data[0], d1);
        builtin_amdgcn_mmac<is_half>(a_vec.data[1], b1.data[1], d1);
    }
    if (row < m) {
        for (int i = 0; i < 4; i++) {
            d_ptr0[i * 4] = d0[i];
            d_ptr1[i * 4] = d1[i];
        }
    }
}

// bf16 [M,K] x bf16 [N,K]^T -> fp32 [M,N]. Shape-directed: n128 for large N+M, n16 otherwise.
at::Tensor gemm_opt_fp32(const at::Tensor &x, const at::Tensor &weight) {
    int m = x.sizes()[0];
    int k = x.sizes()[1];
    int n;
    if (weight.strides()[1] == 1) n = weight.sizes()[0];
    else n = weight.sizes()[1];
    std::vector<long> size(2);
    size[0] = m;
    size[1] = n;
    at::Tensor ret = at::empty(size, x.options().dtype(torch::kFloat32));
    auto stream = c10::hip::getCurrentHIPStream().stream();
    AT_DISPATCH_REDUCED_FLOATING_TYPES(x.scalar_type(), "gemm_opt_fp32", [&] {
        scalar_t *a = x.data_ptr<scalar_t>();
        scalar_t *b = weight.data_ptr<scalar_t>();
        float *d = ret.data_ptr<float>();
        constexpr int NumWarps = 4;
        if (n >= 2048 && (n % 128) == 0 && m > 48) {
            constexpr int NPerBlock = 128;
            int blocks_x = (n - 1) / NPerBlock + 1;
            int blocks_y = (m - 1) / 16 + 1;
            dim3 grid(blocks_x, blocks_y);
            gemm_nt_fp16_fp32out_n128<scalar_t, NumWarps, NPerBlock>
                <<<grid, NumWarps * C10_WARP_SIZE, 0, stream>>>(a, b, d, m, n, k);
        } else {
            constexpr int NPerBlock = 16;
            int blocks_x = (n - 1) / NPerBlock + 1;
            int blocks_y = (m - 1) / 16 + 1;
            dim3 grid(blocks_x, blocks_y);
            int lds_size = NumWarps / 2 * NPerBlock * 16 * sizeof(float);
            gemm_nt_fp16_fp32out<scalar_t, NumWarps, NPerBlock>
                <<<grid, NumWarps * C10_WARP_SIZE, lds_size, stream>>>(a, b, d, m, n, k);
        }
    });
    return ret;
}

#endif

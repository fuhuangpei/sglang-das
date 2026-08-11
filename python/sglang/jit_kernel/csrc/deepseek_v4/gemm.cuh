#pragma once
// bf16 -> fp32 GEMM kernels for DeepSeek V4 router(N=256)/wkv_gate(N=1024,2048) decode.
// AMD MFMA (gfx936/928/938).
// n16 = K-split across warps + smem tree reduce + direct store (deterministic, 1 node).
// n64_splitk = async global->LDS load + double buffer + block K-split (b tile < L2) + atomicAdd reduce.
// ROCm-only; compiled via torch.utils.cpp_extension (hipify converts <<<>>> / at::cuda).
#if defined(__HIP_PLATFORM_HCC__) || defined(__HIP_PLATFORM_AMD__)

#include <ATen/ATen.h>
#include <c10/macros/Macros.h>
#include <c10/hip/HIPStream.h>
#include <cstdint>

using half4_t = __attribute__((__vector_size__(4 * sizeof(_Float16)))) _Float16;
using v4bh = __attribute__((__vector_size__(4 * sizeof(short)))) short;
using float4_t = __attribute__((__vector_size__(4 * sizeof(float)))) float;
struct half4x2 { half4_t data[2]; };
using uint32x4_t = uint32_t __attribute__((ext_vector_type(4)));

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

// ---- async global->LDS load helpers (gfx936/938, cp.async equivalent) ----
__device__ __forceinline__ uint32x4_t make_buffer_resource(const uint32_t *ptr) {
    uint32x4_t res = {};
    const uint64_t address = reinterpret_cast<uint64_t>(ptr);
    res[0] = __builtin_amdgcn_readfirstlane(uint32_t(address));
    res[1] = __builtin_amdgcn_readfirstlane(uint32_t(address >> 32));
    res[2] = 0x80000000u;
    res[3] = 0x00020000u;
    return res;
}

// async load 8 bf16 (16B) from gmem[res + gmem_off] to lds[lds_base + lds_off]
template<typename scalar_t>
__device__ __forceinline__ void async_load8(scalar_t* lds_base, int lds_off, uint32x4_t res, int gmem_off) {
    auto *p = (__attribute__((address_space(3))) int*)(lds_base + lds_off);
    __builtin_hcu_raw_buffer_load_lds(res, p, 16, gmem_off * 2, 0, 0, 0);
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

// n64_splitk: async global->LDS + double buffer + N_TILE=64 + block K-split + atomicAdd.
// For large M / large N where b (N*K) > L2 (8MB): splitK makes b tile (N_TILE * K/K_SPLIT) < L2.
template <typename scalar_t, int NUM_WARPS = 4, int NPerBlock = 64, int K_STAGE = 64, int K_SPLIT = 4>
__global__ void gemm_nt_fp16_fp32out_n64_splitk(scalar_t *a, scalar_t *b, float *d, int m, int n, int k) {
    constexpr int N_PER_WARP = NPerBlock / NUM_WARPS;  // 16
    const int bid_x = blockIdx.x, bid_y = blockIdx.y, bid_k = blockIdx.z, tid = threadIdx.x;
    const int warp_idx = __builtin_amdgcn_readfirstlane(tid / C10_WARP_SIZE);
    const int lane = tid % C10_WARP_SIZE;
    const int rowid = lane % 16, rows = lane / 16;
    constexpr bool is_half = std::is_same<scalar_t, at::Half>::value;
    const int row = bid_y * 16 + rowid;
    const int col_warp = bid_x * NPerBlock + warp_idx * N_PER_WARP;
    const int k_off = rows * 8;
    const int k_per_split = k / K_SPLIT;
    const int k_start = bid_k * k_per_split;
    const int nstage = k_per_split / K_STAGE;

    __shared__ scalar_t a_sm[2][16][K_STAGE];
    __shared__ scalar_t b_sm[2][NPerBlock][K_STAGE];
    uint32x4_t res_a = make_buffer_resource((const uint32_t *)a);
    uint32x4_t res_b = make_buffer_resource((const uint32_t *)b);
    int br_row_base = bid_x * NPerBlock;
    float4_t d0 = {0, 0, 0, 0};

    // A[16][K_STAGE]=128 dwordx4: tid 0-127, r=tid/8, c8=tid%8. B[NPerBlock][K_STAGE]=512: tid 0-255 x2.
    auto do_load = [&](int kk, int buf) {
        if (tid < 128) {
            int r = tid >> 3, c8 = tid & 7;
            int ar_row = bid_y * 16 + r;
            if (ar_row < m) async_load8<scalar_t>(&a_sm[buf][0][0], r * 64 + c8 * 8, res_a, ar_row * k + kk + c8 * 8);
        }
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            int idx = tid + j * 256;
            int r = idx >> 3, c8 = idx & 7;  // r 0..63
            async_load8<scalar_t>(&b_sm[buf][0][0], r * 64 + c8 * 8, res_b, (br_row_base + r) * k + kk + c8 * 8);
        }
    };

    do_load(k_start, 0);
    __builtin_amdgcn_s_waitcnt(0xF70);
    __builtin_amdgcn_s_barrier();

    for (int s = 1; s < nstage; s++) {
        int nxt = s & 1, cur = nxt ^ 1, kk = k_start + s * K_STAGE;
        do_load(kk, nxt);
        #pragma unroll
        for (int i = 0; i < K_STAGE; i += 32) {
            half4x2 av = *(half4x2 *)(&a_sm[cur][rowid][k_off + i]);
            half4x2 bv = *(half4x2 *)(&b_sm[cur][warp_idx * N_PER_WARP + rowid][k_off + i]);
            builtin_amdgcn_mmac<is_half>(av.data[0], bv.data[0], d0);
            builtin_amdgcn_mmac<is_half>(av.data[1], bv.data[1], d0);
        }
        __builtin_amdgcn_s_waitcnt(0xF70);
        __builtin_amdgcn_s_barrier();
    }
    {
        int cur = (nstage - 1) & 1;
        #pragma unroll
        for (int i = 0; i < K_STAGE; i += 32) {
            half4x2 av = *(half4x2 *)(&a_sm[cur][rowid][k_off + i]);
            half4x2 bv = *(half4x2 *)(&b_sm[cur][warp_idx * N_PER_WARP + rowid][k_off + i]);
            builtin_amdgcn_mmac<is_half>(av.data[0], bv.data[0], d0);
            builtin_amdgcn_mmac<is_half>(av.data[1], bv.data[1], d0);
        }
    }
    if (row < m) {
        float *dp = d + row * n + col_warp + rows;
        for (int i = 0; i < 4; i++) atomicAdd(&dp[i * 4], d0[i]);
    }
}

// bf16 [M,K] x bf16 [N,K]^T -> fp32 [M,N]. Shape-directed dispatch.
at::Tensor gemm_opt_fp32(const at::Tensor &x, const at::Tensor &weight) {
    int m = x.sizes()[0];
    int k = x.sizes()[1];
    int n;
    if (weight.strides()[1] == 1) n = weight.sizes()[0];
    else n = weight.sizes()[1];
    std::vector<long> size(2);
    size[0] = m;
    size[1] = n;
    // n64_splitk: b (N*K) > L2 8MB -> splitK makes b tile < L2.
    // Simple dispatch: N=1024 (b=8MB) any M -> splitK ks8; N=2048 (b=16MB) M>=8 -> splitK ks4;
    //   N=2048 M<8 -> n16 (b 16MB bw-bound, splitK no gain); N=256 -> n16 (b 2MB < L2).
    // ks chosen per-N (not per-M) for maintainability; small perf cost at N=1024 M=32 / N=2048 M=32.
    bool use_splitk = (n % 64 == 0) && (k % 512 == 0) && (n == 1024 || (n == 2048 && m >= 8));
    int ks = (n == 1024) ? 8 : 4;
    at::Tensor ret = use_splitk ? at::zeros(size, x.options().dtype(torch::kFloat32))
                                : at::empty(size, x.options().dtype(torch::kFloat32));
    auto stream = c10::hip::getCurrentHIPStream().stream();
    AT_DISPATCH_REDUCED_FLOATING_TYPES(x.scalar_type(), "gemm_opt_fp32", [&] {
        scalar_t *a = x.data_ptr<scalar_t>();
        scalar_t *b = weight.data_ptr<scalar_t>();
        float *d = ret.data_ptr<float>();
        constexpr int NumWarps = 4;
        if (use_splitk) {
            constexpr int NPerBlock = 64, K_STAGE = 64;
            int blocks_x = (n - 1) / NPerBlock + 1;
            int blocks_y = (m - 1) / 16 + 1;
            if (ks == 8) {
                dim3 grid(blocks_x, blocks_y, 8);
                gemm_nt_fp16_fp32out_n64_splitk<scalar_t, NumWarps, NPerBlock, K_STAGE, 8>
                    <<<grid, NumWarps * C10_WARP_SIZE, 0, stream>>>(a, b, d, m, n, k);
            } else {
                dim3 grid(blocks_x, blocks_y, 4);
                gemm_nt_fp16_fp32out_n64_splitk<scalar_t, NumWarps, NPerBlock, K_STAGE, 4>
                    <<<grid, NumWarps * C10_WARP_SIZE, 0, stream>>>(a, b, d, m, n, k);
            }
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

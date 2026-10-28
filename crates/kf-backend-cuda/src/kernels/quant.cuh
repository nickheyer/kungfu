#pragma once
#include "common.cuh"

#define KF_TM 32
#define KF_TN 64
#define KF_TK 64

// Eight elements of group gi in a row, zero padded past n for single element types
template <typename T>
__device__ __forceinline__ int kf_load8(const u8* __restrict__ row, int gi, int n, float* v) {
    if constexpr (T::QK == 1) {
        int base = gi * 8, cnt = min(8, n - base);
#pragma unroll
        for (int i = 0; i < 8; i++) v[i] = i < cnt ? T::at(row, base + i) : 0.0f;
        return cnt;
    } else {
        constexpr int G = T::QK / 8;
        T::dq8(row + (i64)(gi / G) * T::BYTES, gi % G, v);
        return 8;
    }
}

template <typename T>
__device__ __forceinline__ float kf_dot8(const float* v, const float* __restrict__ x, int cnt) {
    if constexpr (T::QK == 1) {
        float s = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; i++) s += i < cnt ? v[i] * x[i] : 0.0f;
        return s;
    } else {
        float4 a = reinterpret_cast<const float4*>(x)[0];
        float4 b = reinterpret_cast<const float4*>(x)[1];
        return v[0] * a.x + v[1] * a.y + v[2] * a.z + v[3] * a.w + v[4] * b.x + v[5] * b.y + v[6] * b.z + v[7] * b.w;
    }
}

template <typename T>
__device__ void kf_dequant(const u8* __restrict__ src, float* __restrict__ dst, int n, i64 rows, i64 row_bytes) {
    i64 ng = (n + 7) / 8, total = ng * rows;
    for (i64 t = blockIdx.x * (i64)blockDim.x + threadIdx.x; t < total; t += (i64)gridDim.x * blockDim.x) {
        i64 r = t / ng;
        int gi = (int)(t - r * ng);
        float v[8];
        int cnt = kf_load8<T>(src + r * row_bytes, gi, n, v);
        float* o = dst + r * n + gi * 8;
        for (int i = 0; i < cnt; i++) o[i] = v[i];
    }
}

#define KF_MV_ROWS 8

// One warp per output row, each weight group read once and dotted with every input row, at most eight rows
template <typename T>
__device__ void kf_matvec(const u8* __restrict__ w, const float* __restrict__ x, const float* __restrict__ bias, float* __restrict__ y, int rows, int n, int n_out, i64 row_bytes) {
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int o = blockIdx.x * (blockDim.x >> 5) + warp;
    if (o >= n_out) return;
    const u8* row = w + (i64)o * row_bytes;
    int ng = (n + 7) >> 3;
    float acc[KF_MV_ROWS];
#pragma unroll
    for (int r = 0; r < KF_MV_ROWS; r++) acc[r] = 0.0f;
    for (int gi = lane; gi < ng; gi += 32) {
        float v[8];
        int cnt = kf_load8<T>(row, gi, n, v);
#pragma unroll
        for (int r = 0; r < KF_MV_ROWS; r++) {
            if (r < rows) acc[r] += kf_dot8<T>(v, x + (i64)r * n + gi * 8, cnt);
        }
    }
#pragma unroll
    for (int r = 0; r < KF_MV_ROWS; r++) {
        if (r >= rows) break;
        float s = kf_warp_sum(acc[r]);
        if (lane == 0) y[(i64)r * n_out + o] = s + (bias ? bias[o] : 0.0f);
    }
}

// Shared tile product over rows named by xrow and yrow, minus one marks padding
template <typename T>
__device__ void kf_mm_tile(const u8* __restrict__ w, i64 row_bytes, const float* __restrict__ x, const float* __restrict__ bias, float* __restrict__ y, int n, int n_out, const int* xrow, const int* yrow) {
    __shared__ float xs[KF_TM][KF_TK + 1];
    __shared__ float ws[KF_TN][KF_TK + 1];
    int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    int o0 = blockIdx.x * KF_TN;
    float acc[2][4] = {{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}};
    for (int k0 = 0; k0 < n; k0 += KF_TK) {
        for (int i = tid; i < KF_TM * KF_TK; i += blockDim.x) {
            int rr = i / KF_TK, kk = i % KF_TK, k = k0 + kk, xr = xrow[rr];
            xs[rr][kk] = (xr >= 0 && k < n) ? x[(i64)xr * n + k] : 0.0f;
        }
        for (int i = tid; i < KF_TN * (KF_TK / 8); i += blockDim.x) {
            int oo = i / (KF_TK / 8), gk = i % (KF_TK / 8), o = o0 + oo, gi = (k0 >> 3) + gk;
            float v[8];
            int cnt = 0;
            if (o < n_out && gi * 8 < n) cnt = kf_load8<T>(w + (i64)o * row_bytes, gi, n, v);
#pragma unroll
            for (int j = 0; j < 8; j++) ws[oo][gk * 8 + j] = j < cnt ? v[j] : 0.0f;
        }
        __syncthreads();
#pragma unroll 8
        for (int kk = 0; kk < KF_TK; kk++) {
            float a0 = xs[ty * 2][kk], a1 = xs[ty * 2 + 1][kk];
#pragma unroll
            for (int c = 0; c < 4; c++) {
                float b = ws[tx + 16 * c][kk];
                acc[0][c] += a0 * b;
                acc[1][c] += a1 * b;
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 2; i++) {
        int yr = yrow[ty * 2 + i];
        if (yr < 0) continue;
#pragma unroll
        for (int c = 0; c < 4; c++) {
            int o = o0 + tx + 16 * c;
            if (o < n_out) y[(i64)yr * n_out + o] = acc[i][c] + (bias ? bias[o] : 0.0f);
        }
    }
}

template <typename T>
__device__ void kf_matmul(const u8* __restrict__ w, const float* __restrict__ x, const float* __restrict__ bias, float* __restrict__ y, int rows, int n, int n_out, i64 row_bytes) {
    __shared__ int rows_s[KF_TM];
    int r0 = blockIdx.y * KF_TM;
    if (threadIdx.x < KF_TM) rows_s[threadIdx.x] = r0 + (int)threadIdx.x < rows ? r0 + (int)threadIdx.x : -1;
    __syncthreads();
    kf_mm_tile<T>(w, row_bytes, x, bias, y, n, n_out, rows_s, rows_s);
}

// Int8 tensor core tile. Weights give sixteen integers per group with w = d * q - m, activations arrive as int8 with a scale and
// an integer sum per sixteen columns, so each sixteen wide step is one exact integer product scaled once:
// sum w x = d dx sum(q qx) - m dx sum(qx)
#define KF_I8_BM 64
#define KF_I8_BN 64
#define KF_I8_BK 64
#define KF_I8_LD (KF_I8_BK + 16)
#define KF_I8_G (KF_I8_BK / 16)

// One 16 by 8 by 16 int8 product into four int32 lanes, zero accumulator, the Ampere fragment layout
__device__ __forceinline__ void kf_mma_i8(int* c, u32 a0, u32 a1, u32 b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%7,%7,%7};"
                 : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
                 : "r"(a0), "r"(a1), "r"(b), "r"(0));
#else
    c[0] = c[1] = c[2] = c[3] = 0;
    __trap();
#endif
}

template <typename T>
__device__ void kf_mm_tile_i8(const u8* __restrict__ w, i64 row_bytes, const i8* __restrict__ xq, const float* __restrict__ xs, const int* __restrict__ xsum, const float* __restrict__ bias, float* __restrict__ y, int n, int n_out, const int* xrow, const int* yrow) {
    __shared__ __align__(16) i8 Xs[KF_I8_BM][KF_I8_LD];
    __shared__ __align__(16) i8 Ws[KF_I8_BN][KF_I8_LD];
    __shared__ float dxs[KF_I8_BM][KF_I8_G], sxs[KF_I8_BM][KF_I8_G];
    __shared__ float dws[KF_I8_BN][KF_I8_G], mws[KF_I8_BN][KF_I8_G];
    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
    int wm = warp >> 1, wn = warp & 1;
    int o0 = blockIdx.x * KF_I8_BN;
    int n16 = n >> 4;
    float acc[2][4][4];
#pragma unroll
    for (int mi = 0; mi < 2; mi++)
#pragma unroll
        for (int ni = 0; ni < 4; ni++)
#pragma unroll
            for (int e = 0; e < 4; e++) acc[mi][ni][e] = 0.0f;
    for (int k0 = 0; k0 < n; k0 += KF_I8_BK) {
#pragma unroll
        for (int it = 0; it < 2; it++) {
            int c = tid + 128 * it, r = c >> 2, kc = (c & 3) * 16, xr = xrow[r];
            int4 v = make_int4(0, 0, 0, 0);
            if (xr >= 0 && k0 + kc < n) v = *reinterpret_cast<const int4*>(xq + (i64)xr * n + k0 + kc);
            *reinterpret_cast<int4*>(&Xs[r][kc]) = v;
            int gi = c & 3, k16 = (k0 >> 4) + gi;
            float d = 0.0f, s = 0.0f;
            if (xr >= 0 && k16 < n16) {
                i64 at = (i64)xr * n16 + k16;
                d = xs[at];
                s = d * (float)xsum[at];
            }
            dxs[r][gi] = d;
            sxs[r][gi] = s;
            int oo = c >> 2, o = o0 + oo, k = k0 + 16 * gi;
            __align__(16) i8 q[16];
            float dw = 0.0f, mw = 0.0f;
            if (o < n_out && k < n) {
                T::q16(w + (i64)o * row_bytes + (i64)(k / T::QK) * T::BYTES, (k % T::QK) >> 4, q, dw, mw);
            } else {
#pragma unroll
                for (int j = 0; j < 16; j++) q[j] = 0;
            }
            *reinterpret_cast<int4*>(&Ws[oo][16 * gi]) = *reinterpret_cast<const int4*>(q);
            dws[oo][gi] = dw;
            mws[oo][gi] = mw;
        }
        __syncthreads();
#pragma unroll
        for (int ks = 0; ks < KF_I8_G; ks++) {
            u32 a[2][2], b[4];
#pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                int r = wm * 32 + mi * 16 + g;
                a[mi][0] = *reinterpret_cast<const u32*>(&Xs[r][ks * 16 + 4 * t]);
                a[mi][1] = *reinterpret_cast<const u32*>(&Xs[r + 8][ks * 16 + 4 * t]);
            }
#pragma unroll
            for (int ni = 0; ni < 4; ni++) b[ni] = *reinterpret_cast<const u32*>(&Ws[wn * 32 + ni * 8 + g][ks * 16 + 4 * t]);
            float dxr[2][2], sxr[2][2];
#pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                int r = wm * 32 + mi * 16 + g;
                dxr[mi][0] = dxs[r][ks];
                dxr[mi][1] = dxs[r + 8][ks];
                sxr[mi][0] = sxs[r][ks];
                sxr[mi][1] = sxs[r + 8][ks];
            }
#pragma unroll
            for (int ni = 0; ni < 4; ni++) {
                int c0 = wn * 32 + ni * 8 + 2 * t;
                float dw0 = dws[c0][ks], dw1 = dws[c0 + 1][ks], mw0 = mws[c0][ks], mw1 = mws[c0 + 1][ks];
#pragma unroll
                for (int mi = 0; mi < 2; mi++) {
                    int c[4];
                    kf_mma_i8(c, a[mi][0], a[mi][1], b[ni]);
                    acc[mi][ni][0] += (float)c[0] * (dxr[mi][0] * dw0) - mw0 * sxr[mi][0];
                    acc[mi][ni][1] += (float)c[1] * (dxr[mi][0] * dw1) - mw1 * sxr[mi][0];
                    acc[mi][ni][2] += (float)c[2] * (dxr[mi][1] * dw0) - mw0 * sxr[mi][1];
                    acc[mi][ni][3] += (float)c[3] * (dxr[mi][1] * dw1) - mw1 * sxr[mi][1];
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int mi = 0; mi < 2; mi++) {
#pragma unroll
        for (int h = 0; h < 2; h++) {
            int yr = yrow[wm * 32 + mi * 16 + g + 8 * h];
            if (yr < 0) continue;
#pragma unroll
            for (int ni = 0; ni < 4; ni++) {
#pragma unroll
                for (int j = 0; j < 2; j++) {
                    int o = o0 + wn * 32 + ni * 8 + 2 * t + j;
                    if (o < n_out) y[(i64)yr * n_out + o] = acc[mi][ni][2 * h + j] + (bias ? bias[o] : 0.0f);
                }
            }
        }
    }
}

template <typename T>
__device__ void kf_matmul_i8(const u8* __restrict__ w, const i8* __restrict__ xq, const float* __restrict__ xs, const int* __restrict__ xsum, const float* __restrict__ bias, float* __restrict__ y, int rows, int n, int n_out, i64 row_bytes) {
    __shared__ int rows_s[KF_I8_BM];
    int r0 = blockIdx.y * KF_I8_BM;
    if (threadIdx.x < KF_I8_BM) rows_s[threadIdx.x] = r0 + (int)threadIdx.x < rows ? r0 + (int)threadIdx.x : -1;
    __syncthreads();
    kf_mm_tile_i8<T>(w, row_bytes, xq, xs, xsum, bias, y, n, n_out, rows_s, rows_s);
}

template <typename T>
__device__ void kf_embed(const u8* __restrict__ w, const int* __restrict__ tokens, float* __restrict__ out, int n, int n_tok, i64 row_bytes) {
    int ng = (n + 7) / 8;
    i64 total = (i64)ng * n_tok;
    for (i64 t = blockIdx.x * (i64)blockDim.x + threadIdx.x; t < total; t += (i64)gridDim.x * blockDim.x) {
        int r = (int)(t / ng), gi = (int)(t - (i64)r * ng);
        float v[8];
        int cnt = kf_load8<T>(w + (i64)tokens[r] * row_bytes, gi, n, v);
        float* o = out + (i64)r * n + gi * 8;
        for (int i = 0; i < cnt; i++) o[i] = v[i];
    }
}

// Expert rows for one routed pair per grid column, slots below zero produce zeros
template <typename T>
__device__ void kf_moe_matvec(const u8* __restrict__ base, i64 expert_bytes, i64 row_bytes, const int* __restrict__ slots, const float* __restrict__ x, int x_per_slot, float* __restrict__ y, int n, int n_out, int n_used) {
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int o = blockIdx.x * (blockDim.x >> 5) + warp;
    if (o >= n_out) return;
    int rj = blockIdx.z * n_used + blockIdx.y;
    int slot = slots[rj];
    float* yo = y + (i64)rj * n_out + o;
    if (slot < 0) {
        if (lane == 0) *yo = 0.0f;
        return;
    }
    const u8* row = base + (i64)slot * expert_bytes + (i64)o * row_bytes;
    const float* xr = x + (i64)(x_per_slot ? rj : (int)blockIdx.z) * n;
    int ng = (n + 7) >> 3;
    float acc = 0.0f;
    for (int gi = lane; gi < ng; gi += 32) {
        float v[8];
        int cnt = kf_load8<T>(row, gi, n, v);
        acc += kf_dot8<T>(v, xr + gi * 8, cnt);
    }
    acc = kf_warp_sum(acc);
    if (lane == 0) *yo = acc;
}

// Tiles of the routed pairs grouped by slot, the grid depth walks the slots
template <typename T>
__device__ void kf_moe_matmul(const u8* __restrict__ base, i64 expert_bytes, i64 row_bytes, const int* __restrict__ order, const int* __restrict__ offsets, const float* __restrict__ x, int x_per_slot, float* __restrict__ y, int n, int n_out, int n_used) {
    __shared__ int xr_s[KF_TM], yr_s[KF_TM];
    int e = blockIdx.z, begin = offsets[e], cnt = offsets[e + 1] - begin, r0 = blockIdx.y * KF_TM;
    if (r0 >= cnt) return;
    if (threadIdx.x < KF_TM) {
        int i = r0 + (int)threadIdx.x;
        int rj = i < cnt ? order[begin + i] : -1;
        xr_s[threadIdx.x] = rj < 0 ? -1 : (x_per_slot ? rj : rj / n_used);
        yr_s[threadIdx.x] = rj;
    }
    __syncthreads();
    kf_mm_tile<T>(base + (i64)e * expert_bytes, row_bytes, x, nullptr, y, n, n_out, xr_s, yr_s);
}

#define KF_QUANT_KERNELS(T)                                                                                                                                  \
    extern "C" __global__ void dequant(const u8* src, float* dst, int n, i64 rows, i64 row_bytes) { kf_dequant<T>(src, dst, n, rows, row_bytes); }           \
    extern "C" __global__ void matvec(const u8* w, const float* x, const float* bias, float* y, int rows, int n, int n_out, i64 row_bytes) {                \
        kf_matvec<T>(w, x, bias, y, rows, n, n_out, row_bytes);                                                                                              \
    }                                                                                                                                                        \
    extern "C" __global__ void matmul(const u8* w, const float* x, const float* bias, float* y, int rows, int n, int n_out, i64 row_bytes) {                \
        kf_matmul<T>(w, x, bias, y, rows, n, n_out, row_bytes);                                                                                              \
    }                                                                                                                                                        \
    extern "C" __global__ void embed(const u8* w, const int* tokens, float* out, int n, int n_tok, i64 row_bytes) { kf_embed<T>(w, tokens, out, n, n_tok, row_bytes); } \
    extern "C" __global__ void moe_matvec(const u8* base, i64 expert_bytes, i64 row_bytes, const int* slots, const float* x, int x_per_slot, float* y, int n, int n_out, int n_used) { \
        kf_moe_matvec<T>(base, expert_bytes, row_bytes, slots, x, x_per_slot, y, n, n_out, n_used);                                                          \
    }                                                                                                                                                        \
    extern "C" __global__ void moe_matmul(const u8* base, i64 expert_bytes, i64 row_bytes, const int* order, const int* offsets, const float* x, int x_per_slot, float* y, int n, int n_out, int n_used) { \
        kf_moe_matmul<T>(base, expert_bytes, row_bytes, order, offsets, x, x_per_slot, y, n, n_out, n_used);                                                 \
    }

// The int8 tensor core product, for types whose struct gives q16, listed on the host in INT8_TYPES
#define KF_QUANT_KERNELS_I8(T)                                                                                                                                 \
    KF_QUANT_KERNELS(T)                                                                                                                                        \
    extern "C" __global__ void __launch_bounds__(128) matmul_i8(const u8* w, const i8* xq, const float* xs, const int* xsum, const float* bias, float* y, int rows, int n, int n_out, i64 row_bytes) { \
        kf_matmul_i8<T>(w, xq, xs, xsum, bias, y, rows, n, n_out, row_bytes);                                                                                  \
    }

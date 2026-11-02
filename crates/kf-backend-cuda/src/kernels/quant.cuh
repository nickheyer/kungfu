#pragma once
#include "common.cuh"

#define KF_TM 32
#define KF_TN 64
#define KF_TK 64

// Eight elements of group gi in a row, zero padded past n for single element types, a whole group of a float type
// read as one or two sixteen byte words
template <typename T>
__device__ __forceinline__ int kf_load8(const u8* __restrict__ row, int gi, int n, float* v) {
    if constexpr (T::QK == 1) {
        int base = gi * 8, cnt = min(8, n - base);
        if (cnt == 8 && ((reinterpret_cast<size_t>(row) + (size_t)base * T::BYTES) & 15) == 0) {
            if constexpr (T::BYTES == 4) {
                const float4 a = reinterpret_cast<const float4*>(row + (i64)base * 4)[0];
                const float4 b = reinterpret_cast<const float4*>(row + (i64)base * 4)[1];
                v[0] = a.x;
                v[1] = a.y;
                v[2] = a.z;
                v[3] = a.w;
                v[4] = b.x;
                v[5] = b.y;
                v[6] = b.z;
                v[7] = b.w;
                return 8;
            } else if constexpr (T::BYTES == 2) {
                const uint4 u = *reinterpret_cast<const uint4*>(row + (i64)base * 2);
#pragma unroll
                for (int i = 0; i < 8; i++) v[i] = T::at(reinterpret_cast<const u8*>(&u), i);
                return 8;
            }
        }
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
        if (cnt < 8 || (reinterpret_cast<size_t>(x) & 15) != 0) {
            float s = 0.0f;
#pragma unroll
            for (int i = 0; i < 8; i++) s += i < cnt ? v[i] * x[i] : 0.0f;
            return s;
        }
        float4 a = reinterpret_cast<const float4*>(x)[0];
        float4 b = reinterpret_cast<const float4*>(x)[1];
        return v[0] * a.x + v[1] * a.y + v[2] * a.z + v[3] * a.w + v[4] * b.x + v[5] * b.y + v[6] * b.z + v[7] * b.w;
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

// One warp per output row, each weight group read once and dotted with every input row, at most eight rows,
// the residual rows added when given
template <typename T>
__device__ void kf_matvec(const u8* __restrict__ w, const float* __restrict__ x, const float* __restrict__ bias, const float* __restrict__ res, float* __restrict__ y, int rows, int n, int n_out, i64 row_bytes) {
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
        if (lane == 0) y[(i64)r * n_out + o] = s + (bias ? bias[o] : 0.0f) + (res ? res[(i64)r * n_out + o] : 0.0f);
    }
}

// Shared tile product over rows named by xrow and yrow, minus one marks padding
template <typename T>
__device__ void kf_mm_tile(const u8* __restrict__ w, i64 row_bytes, const float* __restrict__ x, const float* __restrict__ bias, const float* __restrict__ res, float* __restrict__ y, int n, int n_out, const int* xrow, const int* yrow) {
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
            if (o < n_out) y[(i64)yr * n_out + o] = acc[i][c] + (bias ? bias[o] : 0.0f) + (res ? res[(i64)yr * n_out + o] : 0.0f);
        }
    }
}

template <typename T>
__device__ void kf_matmul(const u8* __restrict__ w, const float* __restrict__ x, const float* __restrict__ bias, const float* __restrict__ res, float* __restrict__ y, int rows, int n, int n_out, i64 row_bytes) {
    __shared__ int rows_s[KF_TM];
    int r0 = blockIdx.y * KF_TM;
    if (threadIdx.x < KF_TM) rows_s[threadIdx.x] = r0 + (int)threadIdx.x < rows ? r0 + (int)threadIdx.x : -1;
    __syncthreads();
    kf_mm_tile<T>(w, row_bytes, x, bias, res, y, n, n_out, rows_s, rows_s);
}

// Token rows expanded to f32 by threads t0, t0 + stride, and so on, each taking eight elements at a time
template <typename T>
__device__ __forceinline__ void kf_embed_part(const u8* __restrict__ w, const int* __restrict__ tokens, float* __restrict__ out, int n, int n_tok, i64 row_bytes, i64 t0, i64 stride) {
    int ng = (n + 7) / 8;
    i64 total = (i64)ng * n_tok;
    for (i64 t = t0; t < total; t += stride) {
        int r = (int)(t / ng), gi = (int)(t - (i64)r * ng);
        float v[8];
        int cnt = kf_load8<T>(w + (i64)tokens[r] * row_bytes, gi, n, v);
        float* o = out + (i64)r * n + gi * 8;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            if (i < cnt) o[i] = v[i];
        }
    }
}

template <typename T>
__device__ void kf_embed(const u8* __restrict__ w, const int* __restrict__ tokens, float* __restrict__ out, int n, int n_tok, i64 row_bytes) {
    kf_embed_part<T>(w, tokens, out, n, n_tok, row_bytes, blockIdx.x * (i64)blockDim.x + threadIdx.x, (i64)gridDim.x * blockDim.x);
}

// Expert rows for one routed pair per grid column, pairs with slots below zero left alone
template <typename T>
__device__ void kf_moe_matvec(const u8* __restrict__ base, i64 expert_bytes, i64 row_bytes, const int* __restrict__ slots, const float* __restrict__ x, int x_per_slot, float* __restrict__ y, int n, int n_out, int n_used) {
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int o = blockIdx.x * (blockDim.x >> 5) + warp;
    if (o >= n_out) return;
    int rj = blockIdx.z * n_used + blockIdx.y;
    int slot = slots[rj];
    if (slot < 0) return;
    float* yo = y + (i64)rj * n_out + o;
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
    kf_mm_tile<T>(base + (i64)e * expert_bytes, row_bytes, x, nullptr, nullptr, y, n, n_out, xr_s, yr_s);
}

#define KF_QUANT_KERNELS(T)                                                                                                                                  \
    extern "C" __global__ void dequant(const u8* src, float* dst, int n, i64 rows, i64 row_bytes) { kf_dequant<T>(src, dst, n, rows, row_bytes); }           \
    extern "C" __global__ void matvec(const u8* w, const float* x, const float* bias, const float* res, float* y, int rows, int n, int n_out, i64 row_bytes) { \
        kf_matvec<T>(w, x, bias, res, y, rows, n, n_out, row_bytes);                                                                                         \
    }                                                                                                                                                        \
    extern "C" __global__ void matmul(const u8* w, const float* x, const float* bias, const float* res, float* y, int rows, int n, int n_out, i64 row_bytes) { \
        kf_matmul<T>(w, x, bias, res, y, rows, n, n_out, row_bytes);                                                                                         \
    }                                                                                                                                                        \
    extern "C" __global__ void embed(const u8* w, const int* tokens, float* out, int n, int n_tok, i64 row_bytes) { kf_embed<T>(w, tokens, out, n, n_tok, row_bytes); } \
    extern "C" __global__ void moe_matvec(const u8* base, i64 expert_bytes, i64 row_bytes, const int* slots, const float* x, int x_per_slot, float* y, int n, int n_out, int n_used) { \
        kf_moe_matvec<T>(base, expert_bytes, row_bytes, slots, x, x_per_slot, y, n, n_out, n_used);                                                          \
    }                                                                                                                                                        \
    extern "C" __global__ void moe_matmul(const u8* base, i64 expert_bytes, i64 row_bytes, const int* order, const int* offsets, const float* x, int x_per_slot, float* y, int n, int n_out, int n_used) { \
        kf_moe_matmul<T>(base, expert_bytes, row_bytes, order, offsets, x, x_per_slot, y, n, n_out, n_used);                                                 \
    }

// Sixteen bytes from any two byte aligned address as four words, through the aligned words around them
__device__ __forceinline__ int4 kf_ld16(const u8* __restrict__ p) {
    u64 a = (u64)p;
    const u32* w = reinterpret_cast<const u32*>(a & ~(u64)3);
    u32 sh = (u32)(a & 3) * 8;
    u32 w0 = w[0], w1 = w[1], w2 = w[2], w3 = w[3], w4 = w[4];
    int4 r;
    r.x = (int)__funnelshift_r(w0, w1, sh);
    r.y = (int)__funnelshift_r(w1, w2, sh);
    r.z = (int)__funnelshift_r(w2, w3, sh);
    r.w = (int)__funnelshift_r(w3, w4, sh);
    return r;
}

// Each byte negated as a signed eight bit integer, no carry between bytes
__device__ __forceinline__ u32 kf_neg4(u32 x) {
    u32 t = ~x;
    return ((t & 0x7F7F7F7Fu) + 0x01010101u) ^ (t & 0x80808080u);
}

// Bytes of w negated where the matching low four bits of signs are set
__device__ __forceinline__ u32 kf_sign4(u32 w, u32 signs) {
    u32 m = (((signs & 0xFu) * 0x00204081u) & 0x01010101u) * 0xFFu;
    return (w & ~m) | (kf_neg4(w) & m);
}

// Eight bytes of a grid entry with eight sign bits applied, as two words
__device__ __forceinline__ void kf_grid8(u64 grid, u32 signs, u32& lo, u32& hi) {
    lo = kf_sign4((u32)grid, signs);
    hi = kf_sign4((u32)(grid >> 32), signs >> 4);
}

// The four nibbles at the low half of each byte mapped through the iq4 nonlinear table, as four signed bytes
__device__ __forceinline__ u32 kf_iq4nl4(u32 nib) {
    u32 sel = (nib & 0xFu) | ((nib >> 4) & 0xF0u) | ((nib >> 8) & 0xF00u) | ((nib >> 12) & 0xF000u);
    u32 lo = __byte_perm(0xBFAD9881u, 0xF6EADDCFu, sel & 0x7777u);
    u32 hi = __byte_perm(0x26190D01u, 0x71594535u, sel & 0x7777u);
    u32 m = ((nib >> 3) & 0x01010101u) * 0xFFu;
    return (lo & ~m) | (hi & m);
}

// Byte idx of a run of words
__device__ __forceinline__ int kf_wbyte(const u32* __restrict__ w, int idx) {
    return (int)((w[idx >> 2] >> (8 * (idx & 3))) & 0xffu);
}

// Sixteen integers of group g of a block as four words with the scale and offset. The default goes through the
// type's q16, the common types specialize it with word loads.
template <typename T>
struct KfQ16 {
    static __device__ __forceinline__ void load(const u8* __restrict__ b, int g, int4& wv, float& d, float& m) {
        __align__(16) i8 q[16];
        T::q16(b, g, q, d, m);
        wv = *reinterpret_cast<const int4*>(q);
    }
};

// Steps of words of the int8 matvec a warp keeps in flight ahead of the one it multiplies
#ifndef KF_MV_PF
#define KF_MV_PF 1
#endif

// Columns one warp step of the int8 matvec covers, sixteen per lane
#define KF_MV_STEP 512

// Segment table words a stacked product leaves per member, offsets from the launch pointers so the loads stay global
#define KF_SEG_WORDS 6
#define KF_SEG_NONE (-0x7fffffffffffffffLL - 1)

// Columns one lane of the int8 matvec takes from a block, as chunks of sixteen CSTRIDE columns apart starting at
// base(gg) for lane gg of the block, in NS runs sharing one weight scale. Sixteen for every type, the common types
// widen it so one header decode and one request serve more columns.
template <typename T>
struct KfLane {
    static constexpr int COLS = 16, NS = 1, CSTRIDE = 16;
    static __device__ __forceinline__ int base(int gg) {
        return gg * 16;
    }
};

// The words of lane gg's columns of one block with the scale and offset of each run, sixteen columns through the
// type's q16 form and wider lanes through their own loaders
template <typename T, int COLS>
struct KfQw {
    static __device__ __forceinline__ void load(const u8* __restrict__ b, int gg, int4* wv, float* d, float* m) {
        KfQ16<T>::load(b, gg, wv[0], d[0], m[0]);
    }
};

// One lane's weight words and scales for its columns of one block, each run of RUN columns under one weight scale
// and so scaled against the activation scale and sum of the RUN wide group it lies in
template <typename T>
struct KfLaneWords {
    static constexpr int COLS = KfLane<T>::COLS, NS = KfLane<T>::NS, NW = COLS / 16, WPS = NW / NS, CSTRIDE = KfLane<T>::CSTRIDE;
    static constexpr int RUN = COLS / NS, GSH = RUN == 32 ? 5 : 4;
    int4 wv[NW];
    float d[NS], m[NS];

    __device__ __forceinline__ void zero() {
#pragma unroll
        for (int i = 0; i < NW; i++) wv[i] = make_int4(0, 0, 0, 0);
#pragma unroll
        for (int s = 0; s < NS; s++) d[s] = m[s] = 0.0f;
    }

    __device__ __forceinline__ void load(const u8* __restrict__ b, int gg) {
        KfQw<T, COLS>::load(b, gg, wv, d, m);
    }

    // Adds the products of the lane's columns from column k of one activation row, xdr and xsr the row's scales and
    // integer sums per RUN columns as the quantizer wrote them
    __device__ __forceinline__ void dot(const i8* __restrict__ xrow, const float* __restrict__ xdr, const float* __restrict__ xsr, int k, float& acc) const {
#pragma unroll
        for (int s = 0; s < NS; s++) {
            int dt = 0;
#pragma unroll
            for (int wi = 0; wi < WPS; wi++) {
                const int4 xv = *reinterpret_cast<const int4*>(xrow + k + (s * WPS + wi) * CSTRIDE);
                const int4 ww = wv[s * WPS + wi];
                dt = __dp4a(xv.x, ww.x, dt);
                dt = __dp4a(xv.y, ww.y, dt);
                dt = __dp4a(xv.z, ww.z, dt);
                dt = __dp4a(xv.w, ww.w, dt);
            }
            const int gi = (k + s * WPS * CSTRIDE) >> GSH;
            acc = fmaf(xdr[gi], fmaf(d[s], (float)dt, -m[s] * xsr[gi]), acc);
        }
    }
};

// The words of KF_MV_PF steps of one warp over one weight row, the QK / COLS lanes of a block adjacent so a warp
// covers 32 / (QK / COLS) blocks per step and reads them as one run. Warp split of wpr takes steps split,
// split + wpr, and so on.
template <typename T, int PF_ = KF_MV_PF>
struct KfMvRing {
    static constexpr int LPB = T::QK / KfLane<T>::COLS, BPW = 32 / LPB, PF = PF_;
    KfLaneWords<T> ring[PF];
    const u8* row;
    int nblk, nsteps, bl, gg, split, wpr;

    __device__ __forceinline__ void fill(int i, int s) {
        const int b = s * BPW + bl;
        if (b < nblk) {
            ring[i].load(row + (i64)b * T::BYTES, gg);
        } else {
            ring[i].zero();
        }
    }

    // Starts the loads of the warp's first PF steps
    __device__ __forceinline__ void prime(const u8* __restrict__ row_, int n, int lane, int split_, int wpr_) {
        row = row_;
        nblk = n / T::QK;
        nsteps = (nblk + BPW - 1) / BPW;
        bl = lane / LPB;
        gg = lane - bl * LPB;
        split = split_;
        wpr = wpr_;
#pragma unroll
        for (int i = 0; i < PF; i++) fill(i, split + i * wpr);
    }

    // Multiplies every step of the warp against up to NR activation rows n bytes apart with scales ng floats apart
    template <int NR>
    __device__ __forceinline__ void run(const i8* __restrict__ xq, const float* __restrict__ xd, const float* __restrict__ xs, int rows, int n, float* acc) {
        const int ng = n >> KfLaneWords<T>::GSH;
        const int kb = KfLane<T>::base(gg);
        for (int s0 = split; s0 < nsteps; s0 += PF * wpr) {
#pragma unroll
            for (int i = 0; i < PF; i++) {
                const int s = s0 + i * wpr;
                if (s >= nsteps) break;
                const int b = s * BPW + bl;
                const KfLaneWords<T> cur = ring[i];
                fill(i, s + PF * wpr);
                if (b < nblk) {
                    const int k = b * T::QK + kb;
#pragma unroll
                    for (int r = 0; r < NR; r++) {
                        if (r >= rows) break;
                        cur.dot(xq + (i64)r * n, xd + (i64)r * ng, xs + (i64)r * ng, k, acc[r]);
                    }
                }
            }
        }
    }
};

#include "mm_i8.cuh"
#include "mm_frag.cuh"

// Product of up to NR rows of int8 activations against native weight rows, eight warps a block, wpr of them per output
// row over interleaved steps of its columns and joined through shared memory. The activations come with scales and
// sums per 16 and per 32 columns and the type's lane run picks the pair. One row's activations are staged in shared
// memory first so the weights alone stream through the caches. The residual rows are added when given. With segments
// the rows are nseg products stacked, each row inside the one it falls in, the member's weights, output, bias, and
// residual given as byte offsets from the launch pointers or KF_SEG_NONE for a bias or residual it lacks.
template <typename T, int NR>
__device__ void kf_matvec_i8(const u8* __restrict__ w, const i8* __restrict__ xq, const float* __restrict__ xd16, const float* __restrict__ xs16, const float* __restrict__ xd32, const float* __restrict__ xs32, const float* __restrict__ bias, const float* __restrict__ res, float* __restrict__ y, int rows, int n, int n_out, i64 row_bytes, const long long* __restrict__ segs, int nseg, int wpr) {
    __shared__ float part[8][NR];
    extern __shared__ __align__(16) u8 kf_mv_smem[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    constexpr int GSH = KfLaneWords<T>::GSH;
    const float* xd = GSH == 5 ? xd32 : xd16;
    const float* xs = GSH == 5 ? xs32 : xs16;
    const int wsh = wpr == 8 ? 3 : (wpr == 4 ? 2 : (wpr == 2 ? 1 : 0));
    const int rpb = 8 >> wsh, rw = warp >> wsh, split = warp - (rw << wsh);
    int o = blockIdx.x * rpb + rw;
    bool live = true;
    if (segs) {
        const long long* e = segs;
        int i = 0;
        for (; i < nseg; i++, e += KF_SEG_WORDS) {
            if (o >= (int)e[4] && o < (int)(e[4] + e[5])) break;
        }
        if (i == nseg) {
            live = false;
            e = segs;
        }
        u8* yb = reinterpret_cast<u8*>(y);
        w += e[0];
        y = reinterpret_cast<float*>(yb + e[1]);
        bias = e[2] == KF_SEG_NONE ? nullptr : reinterpret_cast<const float*>(yb + e[2]);
        res = e[3] == KF_SEG_NONE ? nullptr : reinterpret_cast<const float*>(yb + e[3]);
        o -= (int)e[4];
        n_out = (int)e[5];
    }
    live = live && o < n_out;
    const u8* row = w + (i64)(live ? o : 0) * row_bytes;
    float acc[NR];
#pragma unroll
    for (int r = 0; r < NR; r++) acc[r] = 0.0f;
    KfMvRing<T> ring;
    ring.prime(row, n, lane, split, wpr);
    if constexpr (NR == 1) {
        // The activation row and its scales land in shared memory while the first weights are in flight
        const int ng = n >> GSH;
        int4* sq = reinterpret_cast<int4*>(kf_mv_smem);
        float* sd = reinterpret_cast<float*>(kf_mv_smem + n);
        float* ss = sd + ng;
        for (int i = tid; i < (n >> 4); i += blockDim.x) sq[i] = __ldg(reinterpret_cast<const int4*>(xq) + i);
        // Scales and sums go one float at a time, a row of under four groups having no whole sixteen bytes of them
        for (int i = tid; i < ng; i += blockDim.x) {
            sd[i] = __ldg(xd + i);
            ss[i] = __ldg(xs + i);
        }
        __syncthreads();
        if (live) ring.template run<1>(reinterpret_cast<const i8*>(kf_mv_smem), reinterpret_cast<const float*>(kf_mv_smem + n), reinterpret_cast<const float*>(kf_mv_smem + n) + ng, 1, n, acc);
    } else if (live) {
        ring.template run<NR>(xq, xd, xs, rows, n, acc);
    }
#pragma unroll
    for (int r = 0; r < NR; r++) {
        if (r >= rows) break;
        acc[r] = kf_warp_sum(acc[r]);
    }
    if (wpr > 1) {
        if (split > 0 && lane == 0) {
#pragma unroll
            for (int r = 0; r < NR; r++) part[warp][r] = acc[r];
        }
        __syncthreads();
        if (split > 0) return;
        for (int s = 1; s < wpr; s++) {
#pragma unroll
            for (int r = 0; r < NR; r++) acc[r] += part[warp + s][r];
        }
    }
    if (!live || lane != 0) return;
#pragma unroll
    for (int r = 0; r < NR; r++) {
        if (r >= rows) break;
        y[(i64)r * n_out + o] = acc[r] + (bias ? bias[o] : 0.0f) + (res ? res[(i64)r * n_out + o] : 0.0f);
    }
}

// Steps of words the expert kernels keep in flight per row, their rows being short
#define KF_MOE_PF 2

// Expert rows for one routed pair per grid column against that pair's int8 activation row, eight warps a block each
// on one output row in the lane layout of kf_matvec_i8, pairs with slots below zero left alone
template <typename T>
__device__ void kf_moe_matvec_i8(const u8* __restrict__ base, i64 expert_bytes, i64 row_bytes, const int* __restrict__ slots, const i8* __restrict__ xq, const float* __restrict__ xd16, const float* __restrict__ xs16, const float* __restrict__ xd32, const float* __restrict__ xs32, int x_per_slot, float* __restrict__ y, int n, int n_out, int n_used) {
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int o = blockIdx.x * 8 + warp;
    const int rj = blockIdx.y;
    const int slot = slots[rj];
    if (slot < 0 || o >= n_out) return;
    constexpr int GSH = KfLaneWords<T>::GSH;
    float* yo = y + (i64)rj * n_out + o;
    const u8* row = base + (i64)slot * expert_bytes + (i64)o * row_bytes;
    const int xr = x_per_slot ? rj : rj / n_used;
    const i8* xrow = xq + (i64)xr * n;
    const float* xdr = (GSH == 5 ? xd32 : xd16) + (i64)xr * (n >> GSH);
    const float* xsr = (GSH == 5 ? xs32 : xs16) + (i64)xr * (n >> GSH);
    float acc[1] = {0.0f};
    KfMvRing<T, KF_MOE_PF> ring;
    ring.prime(row, n, lane, 0, 1);
    ring.template run<1>(xrow, xdr, xsr, 1, n, acc);
    const float v = kf_warp_sum(acc[0]);
    if (lane == 0) *yo = v;
}

// Gate and up rows of one routed pair per grid column against the pair's int8 activation row, eight warps a block
// each on one output, both rows' words in flight together, the gated activation written straight into the hidden
// row. With ubase null the up rows follow the gate rows inside the gate tensor.
template <typename T>
__device__ void kf_moe_gate_up_i8(const u8* __restrict__ gbase, const u8* __restrict__ ubase, i64 expert_bytes_g, i64 expert_bytes_u, i64 row_bytes, const int* __restrict__ slots, const i8* __restrict__ xq, const float* __restrict__ xd16, const float* __restrict__ xs16, const float* __restrict__ xd32, const float* __restrict__ xs32, float* __restrict__ h, int n, int n_ff, int n_used, int act) {
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int o = blockIdx.x * 8 + warp;
    const int rj = blockIdx.y;
    const int slot = slots[rj];
    if (slot < 0 || o >= n_ff) return;
    constexpr int GSH = KfLaneWords<T>::GSH;
    const u8* grow = gbase + (i64)slot * expert_bytes_g + (i64)o * row_bytes;
    const u8* urow = ubase ? ubase + (i64)slot * expert_bytes_u + (i64)o * row_bytes : gbase + (i64)slot * expert_bytes_g + (i64)(n_ff + o) * row_bytes;
    const int xr = rj / n_used;
    const i8* xrow = xq + (i64)xr * n;
    const float* xdr = (GSH == 5 ? xd32 : xd16) + (i64)xr * (n >> GSH);
    const float* xsr = (GSH == 5 ? xs32 : xs16) + (i64)xr * (n >> GSH);
    KfMvRing<T, KF_MOE_PF> rg, ru;
    rg.prime(grow, n, lane, 0, 1);
    ru.prime(urow, n, lane, 0, 1);
    const int kb = KfLane<T>::base(rg.gg);
    float ag = 0.0f, au = 0.0f;
    for (int s0 = 0; s0 < rg.nsteps; s0 += KF_MOE_PF) {
#pragma unroll
        for (int i = 0; i < KF_MOE_PF; i++) {
            const int st = s0 + i;
            if (st >= rg.nsteps) break;
            const int b = st * KfMvRing<T, KF_MOE_PF>::BPW + rg.bl;
            const KfLaneWords<T> cg = rg.ring[i], cu = ru.ring[i];
            rg.fill(i, st + KF_MOE_PF);
            ru.fill(i, st + KF_MOE_PF);
            if (b < rg.nblk) {
                const int k = b * T::QK + kb;
                cg.dot(xrow, xdr, xsr, k, ag);
                cu.dot(xrow, xdr, xsr, k, au);
            }
        }
    }
    const float g = kf_warp_sum(ag), u = kf_warp_sum(au);
    if (lane == 0) h[(i64)rj * n_ff + o] = kf_act(g, act) * u;
}

// Rows of native bytes into the KF8 form, int8 integers with one f32 scale and offset per g columns, see docs/kf8-format.md.
// One thread per sixteen, the scale written by the first sixteen of each group, the type guaranteeing it is constant over the group.
template <typename T>
__device__ void kf_unpack_i8(const u8* __restrict__ w, i8* __restrict__ q, float* __restrict__ d, float* __restrict__ m, int n, i64 rows, i64 row_bytes, int g) {
    int n16 = n >> 4, per = g >> 4, ng = n / g;
    i64 total = rows * n16;
    for (i64 t = blockIdx.x * (i64)blockDim.x + threadIdx.x; t < total; t += (i64)gridDim.x * blockDim.x) {
        i64 r = t / n16;
        int gi = (int)(t - r * n16), k = gi * 16;
        int4 v;
        float dd, mm;
        KfQ16<T>::load(w + r * row_bytes + (i64)(k / T::QK) * T::BYTES, (k % T::QK) >> 4, v, dd, mm);
        *reinterpret_cast<int4*>(q + r * n + k) = v;
        if (gi % per == 0) {
            d[r * ng + gi / per] = dd;
            if (m) m[r * ng + gi / per] = mm;
        }
    }
}

// Types whose struct gives q16 and so the KF8 unpack, listed on the host in INT8_TYPES
#define KF_QUANT_KERNELS_I8(T)                                                                                                                                 \
    KF_QUANT_KERNELS(T)                                                                                                                                        \
    extern "C" __global__ void unpack_i8(const u8* w, i8* q, float* d, float* m, int n, i64 rows, i64 row_bytes, int g) { kf_unpack_i8<T>(w, q, d, m, n, rows, row_bytes, g); } \
    extern "C" __global__ void __launch_bounds__(256) matvec_i8(const u8* w, const i8* xq, const float* xd16, const float* xs16, const float* xd32, const float* xs32, const float* bias, const float* res, float* y, int rows, int n, int n_out, i64 row_bytes, const long long* segs, int nseg, int wpr) { \
        kf_matvec_i8<T, 8>(w, xq, xd16, xs16, xd32, xs32, bias, res, y, rows, n, n_out, row_bytes, segs, nseg, wpr);                                           \
    }                                                                                                                                                          \
    extern "C" __global__ void __launch_bounds__(256) matvec_i8_1(const u8* w, const i8* xq, const float* xd16, const float* xs16, const float* xd32, const float* xs32, const float* bias, const float* res, float* y, int rows, int n, int n_out, i64 row_bytes, const long long* segs, int nseg, int wpr) { \
        kf_matvec_i8<T, 1>(w, xq, xd16, xs16, xd32, xs32, bias, res, y, rows, n, n_out, row_bytes, segs, nseg, wpr);                                           \
    }                                                                                                                                                          \
    extern "C" __global__ void __launch_bounds__(256, 4) matvec_i8_1s(const u8* w, const i8* xq, const float* xd16, const float* xs16, const float* xd32, const float* xs32, const float* bias, const float* res, float* y, int rows, int n, int n_out, i64 row_bytes, const long long* segs, int nseg, int wpr) { \
        kf_matvec_i8<T, 1>(w, xq, xd16, xs16, xd32, xs32, bias, res, y, rows, n, n_out, row_bytes, segs, nseg, wpr);                                           \
    }                                                                                                                                                          \
    extern "C" __global__ void __launch_bounds__(256, 2) mm_i8_8(const u8* w, const i8* xq, const float* xd16, const float* xs16, const float* xd32, const float* xs32, const float* bias, const float* res, float* y, int rows, int n, int n_out, i64 row_bytes, const long long* segs, int nseg, int wpr, int wsteps) { \
        if (wsteps > 0) kf_mm_i8<T, 8, true>(w, xq, xd16, xs16, xd32, xs32, bias, res, y, rows, n, n_out, row_bytes, segs, nseg, wpr, wsteps);                     \
        else kf_mm_i8<T, 8, false>(w, xq, xd16, xs16, xd32, xs32, bias, res, y, rows, n, n_out, row_bytes, segs, nseg, wpr, 0);                                        \
    }                                                                                                                                                          \
    extern "C" __global__ void __launch_bounds__(256, 2) mm_i8_16(const u8* w, const i8* xq, const float* xd16, const float* xs16, const float* xd32, const float* xs32, const float* bias, const float* res, float* y, int rows, int n, int n_out, i64 row_bytes, const long long* segs, int nseg, int wpr, int wsteps) { \
        if (wsteps > 0) kf_mm_i8<T, 16, true>(w, xq, xd16, xs16, xd32, xs32, bias, res, y, rows, n, n_out, row_bytes, segs, nseg, wpr, wsteps);                     \
        else kf_mm_i8<T, 16, false>(w, xq, xd16, xs16, xd32, xs32, bias, res, y, rows, n, n_out, row_bytes, segs, nseg, wpr, 0);                                       \
    }                                                                                                                                                          \
    extern "C" __global__ void __launch_bounds__(256, 2) mm_i8_32(const u8* w, const i8* xq, const float* xd16, const float* xs16, const float* xd32, const float* xs32, const float* bias, const float* res, float* y, int rows, int n, int n_out, i64 row_bytes, const long long* segs, int nseg, int wpr, int wsteps) { \
        if (wsteps > 0) kf_mm_i8<T, 32, true>(w, xq, xd16, xs16, xd32, xs32, bias, res, y, rows, n, n_out, row_bytes, segs, nseg, wpr, wsteps);                     \
        else kf_mm_i8<T, 32, false>(w, xq, xd16, xs16, xd32, xs32, bias, res, y, rows, n, n_out, row_bytes, segs, nseg, wpr, 0);                                       \
    }                                                                                                                                                          \
    extern "C" __global__ void __launch_bounds__(256) moe_matvec_i8(const u8* base, i64 expert_bytes, i64 row_bytes, const int* slots, const i8* xq, const float* xd16, const float* xs16, const float* xd32, const float* xs32, int x_per_slot, float* y, int n, int n_out, int n_used) { \
        kf_moe_matvec_i8<T>(base, expert_bytes, row_bytes, slots, xq, xd16, xs16, xd32, xs32, x_per_slot, y, n, n_out, n_used);                                \
    }                                                                                                                                                          \
    extern "C" __global__ void __launch_bounds__(256) moe_gate_up_i8(const u8* gbase, const u8* ubase, i64 expert_bytes_g, i64 expert_bytes_u, i64 row_bytes, const int* slots, const i8* xq, const float* xd16, const float* xs16, const float* xd32, const float* xs32, float* h, int n, int n_ff, int n_used, int act) { \
        kf_moe_gate_up_i8<T>(gbase, ubase, expert_bytes_g, expert_bytes_u, row_bytes, slots, xq, xd16, xs16, xd32, xs32, h, n, n_ff, n_used, act);              \
    }

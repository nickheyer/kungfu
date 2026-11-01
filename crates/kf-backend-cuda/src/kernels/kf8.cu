// The int8 tensor core product on the KF8 weight form, see docs/kf8-format.md.
//
// Weights arrive as int8 rows with one f32 scale and optional f32 offset per G columns, activations from quant_act
// as int8 rows with the scale of their W wide block and the integer sum of each G columns. A block tile is BM tokens
// by BN outputs by 64 columns, eight warps as two by four, a three stage cp.async pipeline, ldmatrix fragments out
// of XOR swizzled shared memory. The mid tile of 128 by 128 runs one block per SM, the small tile of 64 by 128 two.
//
// Each group of G columns is one mma per tile, k32 for G of 32 and k16 for G of 16, whose C operand is 0x4B400000
// so the int32 sum lands in the low mantissa bits of a float. One fused multiply add with the precomputed
// -magic * scale recovers sum * scale exactly in a single rounding, a second applies the activation scale, once per
// group when W equals G and once per W block otherwise. There is no int to float conversion.
#include "common.cuh"

#ifndef KF8_BK
#define KF8_BK 64
#endif
#ifndef KF8_STAGES
#define KF8_STAGES 3
#endif
#ifndef KF8_MID_BLOCKS
#define KF8_MID_BLOCKS 1
#endif
#ifndef KF8_SMALL_BLOCKS
#define KF8_SMALL_BLOCKS 2
#endif
// Threads of the mid tile, 256 as two by four warps or 512 as four by four
#ifndef KF8_MID_THREADS
#define KF8_MID_THREADS 256
#endif
#define KF8_THREADS 256
#define KF8_MAGIC 0x4B400000u
#define KF8_MAGIC_F 12582912.0f
// Sixteen byte chunks per tile row and tile rows per 128 byte shared memory line
#define KF8_CPR (KF8_BK / 16)
#define KF8_RPL (128 / KF8_BK)

// c = a * b + magic over k16, the whole of a group of sixteen
__device__ __forceinline__ void kf8_mma16(int* c, u32 a0, u32 a1, u32 b) {
#ifdef KF8_EXPERIMENT_NO_MMA
    c[0] = a0 + b;
    c[1] = a1 ^ b;
    c[2] = a0 - b;
    c[3] = a1 | b;
    return;
#endif
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%7,%7,%7};"
                 : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
                 : "r"(a0), "r"(a1), "r"(b), "r"(KF8_MAGIC));
#endif
}

// c = a * b + magic over k32, the whole of a group of thirty two
__device__ __forceinline__ void kf8_mma32(int* c, u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
#ifdef KF8_EXPERIMENT_NO_MMA
    c[0] = a0 + b0;
    c[1] = a1 ^ b1;
    c[2] = a2 - b0;
    c[3] = a3 | b1;
    return;
#endif
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};"
                 : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(KF8_MAGIC));
#endif
}

// Eight byte asynchronous copy into shared memory, zero filled when fill is false
__device__ __forceinline__ void kf8_cp8(u32 dst, const void* src, bool fill) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    int n = fill ? 8 : 0;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;" ::"r"(dst), "l"(src), "r"(n));
#endif
}

// Byte offset of a 16 byte chunk inside a tile row, swizzled so ldmatrix reads eight rows without bank conflicts
__device__ __forceinline__ int kf8_swz(int row, int chunk) {
    return row * KF8_BK + ((chunk ^ ((row / KF8_RPL) & (KF8_CPR - 1))) << 4);
}

// One stage of shared memory: the two int8 tiles then the scales of each side laid out [row][group]
template <int BM, int BN, int G, bool HAS_M>
struct Kf8Stage {
    static constexpr int NG = KF8_BK / G;
    static constexpr int A = 0;
    static constexpr int B = A + BM * KF8_BK;
    static constexpr int DX = B + BN * KF8_BK;
    static constexpr int SX = DX + BM * NG * 4;
    static constexpr int DW = SX + (HAS_M ? BM * NG * 4 : 0);
    static constexpr int MW = DW + BN * NG * 4;
    static constexpr int BYTES = MW + (HAS_M ? BN * NG * 4 : 0);
    static constexpr int SMEM = KF8_STAGES * BYTES;
};

// The per row scale words of one stage, NG floats per row, as one or two copies per row
template <int NG>
__device__ __forceinline__ void kf8_cp_scales(u32 dst, const float* src, bool fill) {
    if constexpr (NG == 8) {
        kf_cp_async16(dst, src, fill);
        kf_cp_async16(dst + 16, src + 4, fill);
    } else if constexpr (NG == 4) {
        kf_cp_async16(dst, src, fill);
    } else {
        kf8_cp8(dst, src, fill);
    }
}

// y[yrow[r]][o] = res[yrow[r]][o] + sum_k x[xrow[r]][k] * w[o][k] + bias[o], rows beyond the mapping or outputs
// beyond n_out skipped, res the residual rows with the stride of y or null.
// G is the weight group, W the activation block, both dividing n. W == G takes the direct path with one f32
// accumulator, W > G keeps a second accumulator per W block. WIDE marks scales of more than 22 significant bits,
// whose -magic * scale is not exact, so the sum is recovered by a subtraction before the scale is applied.
// ocol is the first y column of this block's tile and n_out_y the columns y holds, which differ from the weight
// row base and count when the weights are a group of products sharing one output tile grid.
template <int BM, int BN, int G, int W, bool HAS_M, bool WIDE, int THREADS, int WN>
__device__ __forceinline__ void kf8_gemm(const i8* __restrict__ wq, const float* __restrict__ wd, const float* __restrict__ wm, const i8* __restrict__ xq, const float* __restrict__ xd, const float* __restrict__ xs, const float* __restrict__ bias, const float* __restrict__ res, float* __restrict__ y, int n, int n_out, int ocol, int n_out_y, int ldy, const int* xrow, const int* yrow, u8* smem) {
    typedef Kf8Stage<BM, BN, G, HAS_M> S;
    constexpr int NG = S::NG;
    // Warps stand WM by WN over the tile, each owning BM / WM rows by BN / WN columns
    constexpr int WM = THREADS / (32 * WN);
    constexpr int MI = BM / (16 * WM), NI = BN / (8 * WN);
    // n tiles one B ldmatrix feeds, two at k32 and four at k16, folded together
    constexpr int NB = G == 32 ? 2 : 4;
#ifdef KF8_DIRECT
    constexpr bool WINDOWED = false;
#else
    constexpr bool WINDOWED = W > G;
#endif
    // Groups per W block when W fits in a tile, else tiles per W block
    constexpr int GPW = W <= KF8_BK ? W / G : NG;
    constexpr int TPW = W <= KF8_BK ? 1 : W / KF8_BK;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
    const int wm_ = warp / WN, wn = warp % WN;
    const int o0 = blockIdx.y * BN;
    const int ng = n / G;
    const u32 sbase = kf_smem_addr(smem);

    float acc[MI][NI][4];
    float accw[WINDOWED ? MI : 1][WINDOWED ? NI : 1][4];
#pragma unroll
    for (int mi = 0; mi < MI; mi++)
#pragma unroll
        for (int ni = 0; ni < NI; ni++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                acc[mi][ni][e] = 0.0f;
                if (WINDOWED) accw[mi][ni][e] = 0.0f;
            }

    // The threads cover RPP tile rows per pass, one chunk each, so a thread moves BM / RPP chunks of A and BN / RPP of B
    // per stage, then one scale word group per row
    constexpr int RPP = THREADS / KF8_CPR;
    constexpr int ACH = BM >= RPP ? BM / RPP : 1, BCH = BN / RPP;
    const int lr = tid / KF8_CPR, lc0 = tid % KF8_CPR;
    const bool a_thread = BM >= RPP || lr < BM;
    int xr[ACH];
    const i8* xa[ACH];
#pragma unroll
    for (int i = 0; i < ACH; i++) {
        xr[i] = a_thread ? xrow[lr + RPP * i] : -1;
        xa[i] = xq + (i64)(xr[i] < 0 ? 0 : xr[i]) * n;
    }
    int wo[BCH];
    const i8* wa[BCH];
#pragma unroll
    for (int i = 0; i < BCH; i++) {
        wo[i] = o0 + lr + RPP * i;
        wa[i] = wq + (i64)(wo[i] < n_out ? wo[i] : 0) * n;
    }
    const int sxr = tid < BM ? xrow[tid] : -1;
    const float* xd_row = xd + (i64)(sxr < 0 ? 0 : sxr) * ng;
    const float* xs_row = xs + (i64)(sxr < 0 ? 0 : sxr) * ng;

    auto load_stage = [&](int stage, int k0) {
        u32 sb = sbase + stage * S::BYTES;
        int k = k0 + lc0 * 16;
        bool kin = k < n;
#pragma unroll
        for (int i = 0; i < ACH; i++) {
            if (a_thread) kf_cp_async16(sb + S::A + kf8_swz(lr + RPP * i, lc0), kin && xr[i] >= 0 ? xa[i] + k : xq, kin && xr[i] >= 0);
        }
#pragma unroll
        for (int i = 0; i < BCH; i++) kf_cp_async16(sb + S::B + kf8_swz(lr + RPP * i, lc0), kin && wo[i] < n_out ? wa[i] + k : wq, kin && wo[i] < n_out);
        int kg = k0 / G;
        if (tid < BM) {
            bool in = sxr >= 0;
            kf8_cp_scales<NG>(sb + S::DX + tid * NG * 4, in ? xd_row + kg : xd, in);
            if (HAS_M) kf8_cp_scales<NG>(sb + S::SX + tid * NG * 4, in ? xs_row + kg : xs, in);
        }
        if (tid < BN) {
            bool in = o0 + tid < n_out;
            kf8_cp_scales<NG>(sb + S::DW + tid * NG * 4, in ? wd + (i64)(o0 + tid) * ng + kg : wd, in);
            if (HAS_M) kf8_cp_scales<NG>(sb + S::MW + tid * NG * 4, in ? wm + (i64)(o0 + tid) * ng + kg : wm, in);
        }
    };

    const int ktiles = (n + KF8_BK - 1) / KF8_BK;
#pragma unroll
    for (int s = 0; s < KF8_STAGES - 1; s++) {
        if (s < ktiles) load_stage(s, s * KF8_BK);
        kf_cp_commit();
    }

    // ldmatrix lane addressing: row inside a sixteen row span and which half of the span's chunks
    const int lm_row = (lane & 7) + ((lane >> 3) & 1) * 8;
    const int lm_hi = lane >> 4;
    // Rows and columns of this lane's accumulators inside the block tile
    const int r0 = wm_ * (BM / WM) + g, c0 = wn * (BN / WN) + 2 * t;

    for (int kt = 0; kt < ktiles; kt++) {
        kf_cp_wait<KF8_STAGES - 2>();
        __syncthreads();
        {
            int ns = kt + KF8_STAGES - 1;
            if (ns < ktiles) load_stage(ns % KF8_STAGES, ns * KF8_BK);
            kf_cp_commit();
        }
        const int stage = kt % KF8_STAGES;
        const u8* sp = smem + stage * S::BYTES;
        const u32 soff = stage * S::BYTES;
        const float* dxs = reinterpret_cast<const float*>(sp + S::DX);
        const float* sxs = reinterpret_cast<const float*>(sp + S::SX);
        const float* dws = reinterpret_cast<const float*>(sp + S::DW);
        const float* mws = reinterpret_cast<const float*>(sp + S::MW);
        const bool last_tile_of_w = TPW == 1 || (kt % TPW) == TPW - 1;

#pragma unroll
        for (int gi = 0; gi < NG; gi++) {
            // A fragments of every m tile for this group
            u32 a[MI][G == 32 ? 4 : 2];
            if constexpr (G == 32) {
#pragma unroll
                for (int mi = 0; mi < MI; mi++) {
                    int row = wm_ * (BM / WM) + mi * 16 + lm_row;
                    kf_ldmatrix_x4(a[mi][0], a[mi][1], a[mi][2], a[mi][3], sbase + soff + S::A + kf8_swz(row, 2 * gi + lm_hi));
                }
            } else if constexpr (MI == 1) {
                // One m tile: the two halves of the x4 cover its sixteen rows twice, the second pair unused
                int row = wm_ * (BM / WM) + lm_row;
                u32 u0, u1;
                kf_ldmatrix_x4(a[0][0], a[0][1], u0, u1, sbase + soff + S::A + kf8_swz(row, gi));
                (void)u0;
                (void)u1;
            } else {
#pragma unroll
                for (int mi = 0; mi < MI; mi += 2) {
                    int row = wm_ * (BM / WM) + mi * 16 + lm_row + lm_hi * 16;
                    kf_ldmatrix_x4(a[mi][0], a[mi][1], a[mi + 1][0], a[mi + 1][1], sbase + soff + S::A + kf8_swz(row, gi));
                }
            }
            float sx[MI][2], dx[MI][2];
            if (HAS_M) {
#pragma unroll
                for (int mi = 0; mi < MI; mi++) {
                    sx[mi][0] = sxs[(r0 + mi * 16) * NG + gi];
                    sx[mi][1] = sxs[(r0 + mi * 16 + 8) * NG + gi];
                }
            }
            if (!WINDOWED) {
#pragma unroll
                for (int mi = 0; mi < MI; mi++) {
                    dx[mi][0] = dxs[(r0 + mi * 16) * NG + gi];
                    dx[mi][1] = dxs[(r0 + mi * 16 + 8) * NG + gi];
                }
            }

#pragma unroll
            for (int nb = 0; nb < NI; nb += NB) {
                int c[MI][NB][4];
                if constexpr (G == 32) {
                    // Two n tiles, each with both k halves of the group
                    u32 b[2][2];
                    int row = wn * (BN / WN) + nb * 8 + lm_hi * 8 + (lane & 7);
                    kf_ldmatrix_x4(b[0][0], b[0][1], b[1][0], b[1][1], sbase + soff + S::B + kf8_swz(row, 2 * gi + ((lane >> 3) & 1)));
#pragma unroll
                    for (int mi = 0; mi < MI; mi++)
#pragma unroll
                        for (int j = 0; j < 2; j++) kf8_mma32(c[mi][j], a[mi][0], a[mi][1], a[mi][2], a[mi][3], b[j][0], b[j][1]);
                } else {
                    // Four n tiles at one k16 chunk
                    u32 b[4];
                    int row = wn * (BN / WN) + nb * 8 + (lane >> 3) * 8 + (lane & 7);
                    kf_ldmatrix_x4(b[0], b[1], b[2], b[3], sbase + soff + S::B + kf8_swz(row, gi));
#pragma unroll
                    for (int mi = 0; mi < MI; mi++)
#pragma unroll
                        for (int j = 0; j < 4; j++) kf8_mma16(c[mi][j], a[mi][0], a[mi][1], b[j]);
                }

                // Fold these tiles into the accumulators: sum * dw in one fused op, then the activation scale
#pragma unroll
                for (int j = 0; j < NB; j++) {
                    int ni = nb + j;
                    int col = c0 + ni * 8;
                    float dw0 = dws[col * NG + gi], dw1 = dws[(col + 1) * NG + gi];
                    float ndw0 = -KF8_MAGIC_F * dw0, ndw1 = -KF8_MAGIC_F * dw1;
                    float mw0 = HAS_M ? mws[col * NG + gi] : 0.0f, mw1 = HAS_M ? mws[(col + 1) * NG + gi] : 0.0f;
#pragma unroll
                    for (int mi = 0; mi < MI; mi++) {
#pragma unroll
                        for (int e = 0; e < 4; e++) {
                            float cf = __int_as_float(c[mi][j][e]);
#ifdef KF8_EXPERIMENT_NO_FOLD
                            acc[mi][ni][e] += cf;
                            continue;
#endif
                            float v = WIDE ? (cf - KF8_MAGIC_F) * (e & 1 ? dw1 : dw0) : fmaf(cf, e & 1 ? dw1 : dw0, e & 1 ? ndw1 : ndw0);
#ifndef KF8_EXPERIMENT_NO_M
                            if (HAS_M) v = fmaf(-(e & 1 ? mw1 : mw0), sx[mi][e >> 1], v);
#endif
                            if (WINDOWED) {
                                accw[mi][ni][e] += v;
                            } else {
                                acc[mi][ni][e] = fmaf(v, dx[mi][e >> 1], acc[mi][ni][e]);
                            }
                        }
                    }
                }
            }

            // End of a W block: apply its activation scale and clear the block accumulator
            if (WINDOWED && (gi % GPW) == GPW - 1 && last_tile_of_w) {
#pragma unroll
                for (int mi = 0; mi < MI; mi++) {
                    float d0 = dxs[(r0 + mi * 16) * NG + gi], d1 = dxs[(r0 + mi * 16 + 8) * NG + gi];
#pragma unroll
                    for (int ni = 0; ni < NI; ni++) {
                        acc[mi][ni][0] = fmaf(accw[mi][ni][0], d0, acc[mi][ni][0]);
                        acc[mi][ni][1] = fmaf(accw[mi][ni][1], d0, acc[mi][ni][1]);
                        acc[mi][ni][2] = fmaf(accw[mi][ni][2], d1, acc[mi][ni][2]);
                        acc[mi][ni][3] = fmaf(accw[mi][ni][3], d1, acc[mi][ni][3]);
                        accw[mi][ni][0] = accw[mi][ni][1] = accw[mi][ni][2] = accw[mi][ni][3] = 0.0f;
                    }
                }
            }
        }
    }
    kf_cp_wait<0>();

#pragma unroll
    for (int mi = 0; mi < MI; mi++) {
#pragma unroll
        for (int h = 0; h < 2; h++) {
            int yr = yrow[r0 + mi * 16 + 8 * h];
            if (yr < 0) continue;
            float* yp = y + (i64)yr * ldy;
            const float* rp = res ? res + (i64)yr * ldy : nullptr;
#pragma unroll
            for (int ni = 0; ni < NI; ni++) {
                int o = ocol + c0 + ni * 8;
                float v0 = acc[mi][ni][2 * h], v1 = acc[mi][ni][2 * h + 1];
                if (bias) {
                    v0 += o < n_out_y ? bias[o] : 0.0f;
                    v1 += o + 1 < n_out_y ? bias[o + 1] : 0.0f;
                }
                if (rp) {
                    v0 += o < n_out_y ? rp[o] : 0.0f;
                    v1 += o + 1 < n_out_y ? rp[o + 1] : 0.0f;
                }
                if (o + 1 < n_out_y) {
                    *reinterpret_cast<float2*>(yp + o) = make_float2(v0, v1);
                } else if (o < n_out_y) {
                    yp[o] = v0;
                }
            }
        }
    }
}

// Dense rows: row r of x is row r of y, both counted by rows. Row tiles run along grid x so the blocks in flight
// together share one weight tile and the weights stream from memory once. With segments, the weights are nseg
// products stacked along their rows and each column tile writes the product whose entry of y, bias, res, first
// row, and row count holds it.
template <int BM, int BN, int G, int W, bool HAS_M, bool WIDE, int THREADS, int WN>
__device__ __forceinline__ void kf8_gemm_dense(const i8* wq, const float* wd, const float* wm, const i8* xq, const float* xd, const float* xs, const float* bias, const float* res, float* y, int rows, int n, int n_out, const long long* segs, int nseg) {
    extern __shared__ __align__(128) u8 kf8_smem[];
    __shared__ int rows_s[256];
    int r0 = blockIdx.x * BM;
    if (threadIdx.x < BM) rows_s[threadIdx.x] = r0 + (int)threadIdx.x < rows ? r0 + (int)threadIdx.x : -1;
    __syncthreads();
    int ocol = blockIdx.y * BN, n_out_y = n_out;
    if (segs) {
        const long long* e = segs;
        for (int i = 0; i < nseg; i++, e += 5) {
            if (ocol >= (int)e[3] && ocol < (int)(e[3] + e[4])) break;
        }
        y = reinterpret_cast<float*>(e[0]);
        bias = reinterpret_cast<const float*>(e[1]);
        res = reinterpret_cast<const float*>(e[2]);
        ocol -= (int)e[3];
        n_out_y = (int)e[4];
    }
    kf8_gemm<BM, BN, G, W, HAS_M, WIDE, THREADS, WN>(wq, wd, wm, xq, xd, xs, bias, res, y, n, n_out, ocol, n_out_y, n_out_y, rows_s, rows_s, kf8_smem);
}

// Routed pairs grouped by slot: grid z names the slot, grid x its row tiles, order and offsets the pairs of each
// slot, x_per_slot whether a pair reads its own activation row or its token's. The weights are the KF8 form of the
// slots from slot0 on, stacked along their rows.
template <int BM, int BN, int G, int W, bool HAS_M, bool WIDE>
__device__ __forceinline__ void kf8_gemm_moe(const i8* wq, const float* wd, const float* wm, const i8* xq, const float* xd, const float* xs, const int* order, const int* offsets, int x_per_slot, float* y, int n, int n_out, int n_used, int slot0) {
    extern __shared__ __align__(128) u8 kf8_smem[];
    __shared__ int xr_s[128], yr_s[128];
    const int e = slot0 + blockIdx.z, begin = offsets[e], cnt = offsets[e + 1] - begin, r0 = blockIdx.x * BM;
    if (r0 >= cnt) return;
    if (threadIdx.x < BM) {
        int i = r0 + (int)threadIdx.x;
        int rj = i < cnt ? order[begin + i] : -1;
        xr_s[threadIdx.x] = rj < 0 ? -1 : (x_per_slot ? rj : rj / n_used);
        yr_s[threadIdx.x] = rj;
    }
    __syncthreads();
    const i64 rows_before = (i64)blockIdx.z * n_out;
    const int ng = n / G;
    kf8_gemm<BM, BN, G, W, HAS_M, WIDE, KF8_THREADS, 4>(wq + rows_before * n, wd + rows_before * ng, HAS_M ? wm + rows_before * ng : wm, xq, xd, xs, nullptr, nullptr, y, n, n_out, blockIdx.y * BN, n_out, n_out, xr_s, yr_s, kf8_smem);
}

#define KF8_ENTRY_W(NAME, BM, BN, BLOCKS, THREADS, WN, G, W, HAS_M, WIDE)                                                                                                     \
    extern "C" __global__ void __launch_bounds__(KF8_THREADS, BLOCKS) NAME##_moe(const i8* wq, const float* wd, const float* wm, const i8* xq, const float* xd, const float* xs, \
                                                                                 const int* order, const int* offsets, int x_per_slot, float* y, int n, int n_out, int n_used, int slot0) { \
        kf8_gemm_moe<BM, BN, G, W, HAS_M, WIDE>(wq, wd, wm, xq, xd, xs, order, offsets, x_per_slot, y, n, n_out, n_used, slot0);                                          \
    }                                                                                                                                                                      \
    extern "C" __global__ void __launch_bounds__(THREADS, BLOCKS) NAME(const i8* wq, const float* wd, const float* wm, const i8* xq, const float* xd, const float* xs, \
                                                                       const float* bias, const float* res, float* y, int rows, int n, int n_out, const long long* segs, int nseg) { \
        kf8_gemm_dense<BM, BN, G, W, HAS_M, WIDE, THREADS, WN>(wq, wd, wm, xq, xd, xs, bias, res, y, rows, n, n_out, segs, nseg);                                       \
    }
#define KF8_ENTRY(NAME, BM, BN, BLOCKS, THREADS, WN, G, W, HAS_M) KF8_ENTRY_W(NAME, BM, BN, BLOCKS, THREADS, WN, G, W, HAS_M, false)

// The tiny tile of 32 rows only serves the routed experts, whose pairs per expert are few
#define KF8_ENTRY_MOE_W(NAME, BM, BN, BLOCKS, G, W, HAS_M, WIDE)                                                                                                             \
    extern "C" __global__ void __launch_bounds__(KF8_THREADS, BLOCKS) NAME##_moe(const i8* wq, const float* wd, const float* wm, const i8* xq, const float* xd, const float* xs, \
                                                                                 const int* order, const int* offsets, int x_per_slot, float* y, int n, int n_out, int n_used, int slot0) { \
        kf8_gemm_moe<BM, BN, G, W, HAS_M, WIDE>(wq, wd, wm, xq, xd, xs, order, offsets, x_per_slot, y, n, n_out, n_used, slot0);                                          \
    }
#define KF8_ENTRY_MOE(NAME, BM, BN, BLOCKS, G, W, HAS_M) KF8_ENTRY_MOE_W(NAME, BM, BN, BLOCKS, G, W, HAS_M, false)

KF8_ENTRY(gemm_mid_g16_w32, 128, 128, KF8_MID_BLOCKS, KF8_MID_THREADS, 4, 16, 32, false)
KF8_ENTRY(gemm_mid_g16_w32_m, 128, 128, KF8_MID_BLOCKS, KF8_MID_THREADS, 4, 16, 32, true)
KF8_ENTRY(gemm_mid_g32_w32, 128, 128, KF8_MID_BLOCKS, KF8_MID_THREADS, 4, 32, 32, false)
KF8_ENTRY(gemm_mid_g32_w32_m, 128, 128, KF8_MID_BLOCKS, KF8_MID_THREADS, 4, 32, 32, true)
KF8_ENTRY(gemm_mid_g16_w256, 128, 128, KF8_MID_BLOCKS, KF8_MID_THREADS, 4, 16, 256, false)
KF8_ENTRY(gemm_mid_g16_w256_m, 128, 128, KF8_MID_BLOCKS, KF8_MID_THREADS, 4, 16, 256, true)
KF8_ENTRY(gemm_mid_g32_w256, 128, 128, KF8_MID_BLOCKS, KF8_MID_THREADS, 4, 32, 256, false)
KF8_ENTRY(gemm_mid_g32_w256_m, 128, 128, KF8_MID_BLOCKS, KF8_MID_THREADS, 4, 32, 256, true)
KF8_ENTRY_W(gemm_mid_g32_w256_f32, 128, 128, KF8_MID_BLOCKS, KF8_MID_THREADS, 4, 32, 256, false, true)
KF8_ENTRY(gemm_small_g16_w32, 64, 128, KF8_SMALL_BLOCKS, KF8_THREADS, 4, 16, 32, false)
KF8_ENTRY(gemm_small_g16_w32_m, 64, 128, KF8_SMALL_BLOCKS, KF8_THREADS, 4, 16, 32, true)
KF8_ENTRY(gemm_small_g32_w32, 64, 128, KF8_SMALL_BLOCKS, KF8_THREADS, 4, 32, 32, false)
KF8_ENTRY(gemm_small_g32_w32_m, 64, 128, KF8_SMALL_BLOCKS, KF8_THREADS, 4, 32, 32, true)
KF8_ENTRY(gemm_small_g16_w256, 64, 128, KF8_SMALL_BLOCKS, KF8_THREADS, 4, 16, 256, false)
KF8_ENTRY(gemm_small_g16_w256_m, 64, 128, KF8_SMALL_BLOCKS, KF8_THREADS, 4, 16, 256, true)
KF8_ENTRY(gemm_small_g32_w256, 64, 128, KF8_SMALL_BLOCKS, KF8_THREADS, 4, 32, 256, false)
KF8_ENTRY(gemm_small_g32_w256_m, 64, 128, KF8_SMALL_BLOCKS, KF8_THREADS, 4, 32, 256, true)
KF8_ENTRY_W(gemm_small_g32_w256_f32, 64, 128, KF8_SMALL_BLOCKS, KF8_THREADS, 4, 32, 256, false, true)
KF8_ENTRY_MOE(gemm_tiny_g16_w32, 32, 128, 2, 16, 32, false)
KF8_ENTRY_MOE(gemm_tiny_g16_w32_m, 32, 128, 2, 16, 32, true)
KF8_ENTRY_MOE(gemm_tiny_g32_w32, 32, 128, 2, 32, 32, false)
KF8_ENTRY_MOE(gemm_tiny_g32_w32_m, 32, 128, 2, 32, 32, true)
KF8_ENTRY_MOE(gemm_tiny_g16_w256, 32, 128, 2, 16, 256, false)
KF8_ENTRY_MOE(gemm_tiny_g16_w256_m, 32, 128, 2, 16, 256, true)
KF8_ENTRY_MOE(gemm_tiny_g32_w256, 32, 128, 2, 32, 256, false)
KF8_ENTRY_MOE(gemm_tiny_g32_w256_m, 32, 128, 2, 32, 256, true)
KF8_ENTRY_MOE_W(gemm_tiny_g32_w256_f32, 32, 128, 2, 32, 256, false, true)

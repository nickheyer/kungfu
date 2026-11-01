#pragma once
// Product of up to R int8 activation rows against native weight rows on the tensor cores. A warp owns eight weight
// rows, four lanes to a row, and walks their columns four lane widths at a time, so the lanes of a row read its bytes
// contiguously as the matvec lanes do and every weight byte is decoded once. The decoded int8 words land in a warp
// private shared memory tile and ldmatrix turns the tile into the A fragments of eight row mma tiles, which multiply
// every activation row. With WIN the activation rows of the block's current window of steps sit in shared memory too,
// with their scales and sums per run, and ldmatrix turns them into the B fragments the same way, so a step touches no
// activation through the caches, the block refilling the window together every wsteps steps. Without it, when the
// warps of a row group split a long row and stand on steps apart, the B fragments and sums of each quarter of a step
// come straight from the quantized rows through the caches. No fragment lives past the quarter that uses it. The chunks of a run
// under one weight scale chain through one accumulator seeded with the magic constant, so the int32 sum lands in the
// mantissa and one fused multiply add against the scale and the exact negated magic times scale recovers the scaled
// sum. The scaled sums of the runs inside one activation block add up and the block's activation scale applies once,
// with the offsets times the activation sums taken off first for the types that have them. The warps per row group
// split the steps between them and join through shared memory. Grid y walks tiles of R activation rows over the same
// weights.

#define KF_MM_MAGIC 0x4B400000u
#define KF_MM_MAGIC_F 12582912.0f
#define KF_MM_THREADS 256

// Types whose group scale carries more than 22 significant bits, so the magic fold is inexact and the sum is
// recovered by a subtraction first, the f32 scale of q8_K alone
template <typename T>
struct KfWide {
    static constexpr bool V = false;
};

// Types whose groups carry an offset below the scaled integers, q4_1 q5_1 q2_K q4_K and q5_K, which specialize this
template <typename T>
struct KfOffset {
    static constexpr bool V = false;
};

// c = a b + c over an eight row tile and sixteen columns
__device__ __forceinline__ void kf_mm_mma8(int* c, u32 a, u32 b, int c0, int c1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%4,%5};" : "=r"(c[0]), "=r"(c[1]) : "r"(a), "r"(b), "r"(c0), "r"(c1));
#endif
}

// The step of a warp and its eight row tile, whose rows run sixteen bytes past the step so ldmatrix meets distinct
// banks, and the activation window of the block behind the eight tiles, see mm_smem on the host
template <typename T>
struct KfMmShape {
    static constexpr int COLS = KfLane<T>::COLS, SW = 4 * COLS, PITCH = SW + 16, TILE = 8 * PITCH;
};

template <typename T, int R, bool WIN>
__device__ void kf_mm_i8(const u8* __restrict__ w, const i8* __restrict__ xq, const float* __restrict__ xd16, const float* __restrict__ xs16, const float* __restrict__ xd32, const float* __restrict__ xs32, const float* __restrict__ bias, const float* __restrict__ res, float* __restrict__ y, int rows, int n, int n_out, i64 row_bytes, const long long* __restrict__ segs, int nseg, int wpr, int wsteps) {
    extern __shared__ __align__(128) u8 kf_mm_smem[];
    typedef KfLaneWords<T> LW;
    typedef KfMmShape<T> S;
    constexpr int COLS = LW::COLS, NS = LW::NS, NW = LW::NW, WPS = LW::WPS, CSTRIDE = LW::CSTRIDE, GSH = LW::GSH;
    constexpr int KC = LW::RUN / 16, LPB = T::QK / COLS, SW = S::SW, PITCH = S::PITCH, TILE = S::TILE, NT = R / 8;
    constexpr bool WIDE = KfWide<T>::V, HAS_M = KfOffset<T>::V;
    // Columns of one activation scale, the whole super block of the 256 wide types and 32 otherwise
    constexpr int W = T::QK == 256 ? 256 : 32;
    // Sixteen column chunks of a step, runs of a step, and runs of a quarter of a step
    constexpr int NCH = SW / 16, RPS = SW >> GSH, RPQ = 64 >> GSH;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, gid = lane >> 2, tig = lane & 3;
    const int wsh = wpr == 8 ? 3 : (wpr == 4 ? 2 : (wpr == 2 ? 1 : 0));
    const int rg = warp >> wsh, split = warp - (rg << wsh);
    const int act0 = blockIdx.y * R;
    // The member of the segment table the tile falls in, every member starting on a multiple of eight rows
    int o0 = blockIdx.x * (64 >> wsh) + rg * 8;
    bool live = true;
    if (segs) {
        const long long* e = segs;
        int i = 0;
        for (; i < nseg; i++, e += KF_SEG_WORDS) {
            if (o0 >= (int)e[4] && o0 < (int)(e[4] + e[5])) break;
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
        o0 -= (int)e[4];
        n_out = (int)e[5];
    }
    const int o = o0 + gid;
    const u8* row = w + (i64)(live && o < n_out ? o : 0) * row_bytes;
    // Tile column of the lane's first word, the matvec lane layout of four lanes repeated per step
    const int tcol = (tig / LPB) * T::QK + KfLane<T>::base(tig % LPB);
    u8* tile = kf_mm_smem + warp * TILE;
    const u32 ts = kf_smem_addr(tile);
    // The window: R activation rows of wsteps steps, then their sums and scales per run
    const int PB = wsteps * SW + 16, NRW = wsteps * RPS;
    u8* bwin = kf_mm_smem + 8 * TILE;
    float* xswin = reinterpret_cast<float*>(bwin + R * PB);
    float* xdwin = xswin + R * NRW;
    const u32 bs = kf_smem_addr(bwin);
    const int nsteps = n / SW;
    const float* xd = GSH == 5 ? xd32 : xd16;
    const float* xs = GSH == 5 ? xs32 : xs16;
    const int ng = n >> GSH;
    const int last = rows - 1;
    float acc[NT][2], accb[NT][2], macc[NT][2];
#pragma unroll
    for (int nt = 0; nt < NT; nt++) acc[nt][0] = acc[nt][1] = accb[nt][0] = accb[nt][1] = macc[nt][0] = macc[nt][1] = 0.0f;
    // The row of this lane's two C columns per n tile, in the window and clamped into the rows present in memory
    int wr[NT][2], xo[NT], so[NT][2];
#pragma unroll
    for (int nt = 0; nt < NT; nt++) {
        const int ar = act0 + nt * 8;
        xo[nt] = min(ar + gid, last) * n + tig * 4;
#pragma unroll
        for (int e = 0; e < 2; e++) {
            wr[nt][e] = nt * 8 + 2 * tig + e;
            so[nt][e] = min(ar + 2 * tig + e, last) * ng;
        }
    }
    // The block's scaled sums less its offsets times its sums, scaled by the activation scale of run gi, counted from
    // the window's first run or the row's, into acc
    auto fold = [&](int gi) {
#pragma unroll
        for (int nt = 0; nt < NT; nt++) {
#pragma unroll
            for (int e = 0; e < 2; e++) {
                const float xda = WIN ? xdwin[wr[nt][e] * NRW + gi] : __ldg(xd + so[nt][e] + gi);
                acc[nt][e] = fmaf(xda, accb[nt][e] - macc[nt][e], acc[nt][e]);
                accb[nt][e] = macc[nt][e] = 0.0f;
            }
        }
    };
    LW cur, nxt;
    int t = split;
    if (t < nsteps) {
        const int lc = 4 * t + tig;
        cur.load(row + (i64)(lc / LPB) * T::BYTES, lc % LPB);
    }
    const int wspan = WIN ? wsteps : nsteps;
    for (int w0 = 0; w0 < nsteps; w0 += wspan) {
        if constexpr (WIN) {
            // Every warp is done with the window before the block refills it with the rows of the coming steps
            __syncthreads();
            const int kw0 = w0 * SW, pieces = (wsteps * SW) >> 4;
            for (int idx = tid; idx < R * pieces; idx += KF_MM_THREADS) {
                const int r = idx / pieces, c = (idx - r * pieces) << 4;
                if (kw0 + c < n) *reinterpret_cast<int4*>(bwin + r * PB + c) = __ldg(reinterpret_cast<const int4*>(xq + (i64)min(act0 + r, last) * n + kw0 + c));
            }
            const int g0 = kw0 >> GSH;
            for (int idx = tid; idx < R * NRW; idx += KF_MM_THREADS) {
                const int r = idx / NRW, gi = idx - r * NRW;
                if (g0 + gi < ng) {
                    const i64 at = (i64)min(act0 + r, last) * ng + g0 + gi;
                    xswin[idx] = __ldg(xs + at);
                    xdwin[idx] = __ldg(xd + at);
                }
            }
            __syncthreads();
        }
        const int wend = min(w0 + wspan, nsteps);
        for (; t < wend; t += wpr) {
            const int tn = t + wpr;
            if (tn < nsteps) {
                const int lc = 4 * tn + tig;
                nxt.load(row + (i64)(lc / LPB) * T::BYTES, lc % LPB);
            }
            // The step's first run, counted from the window's first run or the row's
            const int tl = t - w0, gs = WIN ? tl * RPS : t * RPS, k0 = t * SW;
            __syncwarp();
#pragma unroll
            for (int i = 0; i < NW; i++) *reinterpret_cast<int4*>(tile + gid * PITCH + tcol + i * CSTRIDE) = cur.wv[i];
            __syncwarp();
#pragma unroll
            for (int q = 0; q < NCH / 4; q++) {
                u32 a[4], b[NT][4];
                kf_ldmatrix_x4(a[0], a[1], a[2], a[3], ts + (lane & 7) * PITCH + (4 * q + (lane >> 3)) * 16);
                // The activation sums of the quarter's runs for the two C columns of every n tile
                float xsq[NT][2][RPQ];
#pragma unroll
                for (int nt = 0; nt < NT; nt++) {
                    if constexpr (WIN) {
                        kf_ldmatrix_x4(b[nt][0], b[nt][1], b[nt][2], b[nt][3], bs + (nt * 8 + (lane & 7)) * PB + tl * SW + (4 * q + (lane >> 3)) * 16);
                    } else {
#pragma unroll
                        for (int c = 0; c < 4; c++) b[nt][c] = __ldg(reinterpret_cast<const u32*>(xq + xo[nt] + k0 + (4 * q + c) * 16));
                    }
                    if constexpr (HAS_M) {
#pragma unroll
                        for (int e = 0; e < 2; e++) {
                            const float* p = WIN ? xswin + wr[nt][e] * NRW + gs + q * RPQ : xs + so[nt][e] + gs + q * RPQ;
                            if constexpr (RPQ == 4) {
                                const float4 v = WIN ? *reinterpret_cast<const float4*>(p) : __ldg(reinterpret_cast<const float4*>(p));
                                xsq[nt][e][0] = v.x;
                                xsq[nt][e][1] = v.y;
                                xsq[nt][e][2] = v.z;
                                xsq[nt][e][3] = v.w;
                            } else {
                                const float2 v = WIN ? *reinterpret_cast<const float2*>(p) : __ldg(reinterpret_cast<const float2*>(p));
                                xsq[nt][e][0] = v.x;
                                xsq[nt][e][1] = v.y;
                            }
                        }
                    }
                }
                // Run s of the lane src of each row, when its chunks lie in this quarter of the step
#pragma unroll
                for (int src = 0; src < 4; src++) {
#pragma unroll
                    for (int s = 0; s < NS; s++) {
                        const int rc = ((src / LPB) * T::QK + KfLane<T>::base(src % LPB) + s * WPS * CSTRIDE) >> 4;
                        if (rc / 4 != q) continue;
                        const int ca = rc - 4 * q, ri = rc >> (GSH - 4), rq = ri - q * RPQ;
                        const float d = __shfl_sync(0xffffffffu, cur.d[s], (lane & ~3) | src);
                        const float m = HAS_M ? __shfl_sync(0xffffffffu, cur.m[s], (lane & ~3) | src) : 0.0f;
                        const float nmd = -KF_MM_MAGIC_F * d;
#pragma unroll
                        for (int nt = 0; nt < NT; nt++) {
                            int cc[2];
                            kf_mm_mma8(cc, a[ca], b[nt][ca], (int)KF_MM_MAGIC, (int)KF_MM_MAGIC);
                            if constexpr (KC == 2) kf_mm_mma8(cc, a[ca + 1], b[nt][ca + 1], cc[0], cc[1]);
#pragma unroll
                            for (int e = 0; e < 2; e++) {
                                accb[nt][e] += WIDE ? (__int_as_float(cc[e]) - KF_MM_MAGIC_F) * d : fmaf(__int_as_float(cc[e]), d, nmd);
                                if constexpr (HAS_M) macc[nt][e] = fmaf(m, xsq[nt][e][rq], macc[nt][e]);
                            }
                        }
                        // An activation block narrower than a step ends inside it, the runs coming in column order
                        if constexpr (W < SW) {
                            if ((16 * (rc + KC)) % W == 0) fold(gs + ri);
                        }
                    }
                }
            }
            // A step inside one activation block folds at its end, the scale being the same for the whole block
            if constexpr (W >= SW) fold(gs);
            cur = nxt;
        }
    }
    if (wpr > 1) {
        // The splits of a row group join in the tile space, lane by lane
        float* join = reinterpret_cast<float*>(kf_mm_smem);
        __syncthreads();
        if (split > 0) {
#pragma unroll
            for (int nt = 0; nt < NT; nt++) {
                join[((warp * 32 + lane) * NT + nt) * 2] = acc[nt][0];
                join[((warp * 32 + lane) * NT + nt) * 2 + 1] = acc[nt][1];
            }
        }
        __syncthreads();
        if (split > 0) return;
        for (int s = 1; s < wpr; s++) {
#pragma unroll
            for (int nt = 0; nt < NT; nt++) {
                acc[nt][0] += join[(((warp + s) * 32 + lane) * NT + nt) * 2];
                acc[nt][1] += join[(((warp + s) * 32 + lane) * NT + nt) * 2 + 1];
            }
        }
    }
    if (!live || o >= n_out) return;
    const float b = bias ? bias[o] : 0.0f;
#pragma unroll
    for (int nt = 0; nt < NT; nt++) {
        const int a0 = act0 + nt * 8 + 2 * tig, a1 = a0 + 1;
        if (a0 < rows) y[(i64)a0 * n_out + o] = acc[nt][0] + b + (res ? res[(i64)a0 * n_out + o] : 0.0f);
        if (a1 < rows) y[(i64)a1 * n_out + o] = acc[nt][1] + b + (res ? res[(i64)a1 * n_out + o] : 0.0f);
    }
}

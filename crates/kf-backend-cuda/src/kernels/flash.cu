// Tensor core attention over the half precision ring cache for passes of many rows.
//
// One block takes 64 query rows of one head and one sequence slot, four warps of sixteen rows each, and walks the keys
// in tiles of BK through a two stage cp.async pipeline. Scores come from m16n8k16 half products with single precision
// accumulation, the softmax runs online in single precision per row, probabilities round to half for the value
// product, and the output accumulates in single precision. Q is rounded to half once, as the per row kernel does.
#include "common.cuh"

#define FA_BQ 64
#define FA_WARPS 4
#define FA_THREADS 128
#define FA_LOG2E 1.4426950408889634f

template <int HD>
struct FaCfg {
    static constexpr int BK = HD <= 128 ? 64 : 32;
    // Chunks of sixteen bytes per row, padded to whole groups of eight so the xor swizzle stays inside the row
    static constexpr int NCH = ((HD / 8 + 7) / 8) * 8;
    static constexpr int ROWB = NCH * 16;
    static constexpr int Q_BYTES = FA_BQ * ROWB;
    static constexpr int KV_BYTES = BK * ROWB;
    static constexpr int SMEM = Q_BYTES + 4 * KV_BYTES;
    static constexpr int NT = BK / 8;
    static constexpr int KS = HD / 16;
    static constexpr int OT = HD / 8;
};

__device__ __forceinline__ int fa_swz(int row, int chunk) {
    return (chunk ^ (row & 7)) << 4;
}

template <int HD>
__device__ __forceinline__ void fa_load_kv(u32 ks, u32 vs, const u16* __restrict__ kb, const u16* __restrict__ vb, i64 kw, i64 vw, int cap, int j0, int hi_max) {
    typedef FaCfg<HD> C;
    constexpr int CH = HD / 8;
    constexpr int PER = C::BK * CH / FA_THREADS;
#pragma unroll
    for (int i = 0; i < PER; i++) {
        int idx = threadIdx.x + i * FA_THREADS;
        int row = idx / CH, ch = idx - row * CH;
        int j = j0 + row;
        bool ok = j <= hi_max;
        int ring = ok ? j % cap : 0;
        const u16* ksrc = kb + (i64)ring * kw + ch * 8;
        const u16* vsrc = vb + (i64)ring * vw + ch * 8;
        u32 off = row * C::ROWB + fa_swz(row, ch);
        kf_cp_async16(ks + off, ksrc, ok);
        kf_cp_async16(vs + off, vsrc, ok);
    }
}

template <int HD>
__device__ __forceinline__ void fa_kernel(const float* __restrict__ q, const u16* __restrict__ kc, const u16* __restrict__ vc, const int* __restrict__ pos, const int* __restrict__ seq, const int* __restrict__ attend, const int* __restrict__ tiles, const float* __restrict__ sinks, float* __restrict__ out, int n_head, int n_head_kv, float scale, int window, float softcap, int cap) {
    typedef FaCfg<HD> C;
    extern __shared__ __align__(128) u8 fa_smem[];
    __shared__ int lo_s[FA_BQ], hi_s[FA_BQ];
    __shared__ int range_s[2];
    const int tile = blockIdx.x;
    if (tile >= tiles[0]) return;
    const int r0 = tiles[4 + 4 * tile], nrows = tiles[5 + 4 * tile], sq = tiles[6 + 4 * tile];
    const int h = blockIdx.y;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
    const int hk = h / (n_head / max(n_head_kv, 1));
    const i64 kw = (i64)n_head_kv * HD, vw = (i64)n_head_kv * HD;
    const u16* kb = kc + (i64)sq * cap * kw + (i64)hk * HD;
    const u16* vb = vc + (i64)sq * cap * vw + (i64)hk * HD;
    const i64 qw = (i64)n_head * HD;

    // Key range of every row and of the block
    if (tid < FA_BQ) {
        int lo = 0, hi = -1;
        if (tid < nrows) {
            int r = r0 + tid;
            int p = pos[r];
            hi = max(p, attend[r]);
            lo = window > 0 ? max(0, p - (window - 1)) : 0;
        }
        lo_s[tid] = lo;
        hi_s[tid] = hi;
    }
    __syncthreads();
    if (tid == 0) {
        int lo = 0x7fffffff, hi = -1;
        for (int i = 0; i < nrows; i++) {
            lo = min(lo, lo_s[i]);
            hi = max(hi, hi_s[i]);
        }
        range_s[0] = lo;
        range_s[1] = hi;
    }

    // Q rows of this head as halves in shared memory, rows past nrows zero
    u8* qs = fa_smem;
    {
        constexpr int F4 = HD / 4;
        for (int idx = tid; idx < FA_BQ * F4; idx += FA_THREADS) {
            int row = idx / F4, c4 = idx - row * F4;
            float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (row < nrows) v = *reinterpret_cast<const float4*>(q + (i64)(r0 + row) * qw + (i64)h * HD + c4 * 4);
            uint2 pk = make_uint2(kf_pack_half2(v.x, v.y), kf_pack_half2(v.z, v.w));
            int d = c4 * 4, ch = d >> 3, within = (d & 7) * 2;
            *reinterpret_cast<uint2*>(qs + row * C::ROWB + fa_swz(row, ch) + within) = pk;
        }
    }
    __syncthreads();
    const int lo_min = range_s[0], hi_max = range_s[1];
    if (hi_max < 0) return;
    const int j_start = (lo_min / C::BK) * C::BK;
    const int ntiles = (hi_max - j_start) / C::BK + 1;

    const u32 sbase = kf_smem_addr(fa_smem);
    const u32 qbase = sbase;
    const u32 kbase = sbase + C::Q_BYTES;
    const u32 vbase = kbase + 2 * C::KV_BYTES;

    // Per thread rows inside the warp's sixteen and their key bounds
    const int row_a = warp * 16 + g, row_b = row_a + 8;
    const int lo_a = lo_s[row_a], hi_a = hi_s[row_a], lo_b = lo_s[row_b], hi_b = hi_s[row_b];

    float o[C::OT][4];
#pragma unroll
    for (int n = 0; n < C::OT; n++) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.0f;
    float m_a = KF_NEG_INF, m_b = KF_NEG_INF, l_a = 0.0f, l_b = 0.0f;

    // ldmatrix lane addressing inside the warp's Q rows, the K rows, and the V rows
    const int lm_row = (lane & 7) + ((lane >> 3) & 1) * 8;
    const int lm_hi = lane >> 4;
    const int q_row = warp * 16 + lm_row;
    const u32 q_lane = qbase + q_row * C::ROWB;
    const int q_key = q_row & 7;

    fa_load_kv<HD>(kbase, vbase, kb, vb, kw, vw, cap, j_start, hi_max);
    kf_cp_commit();

    for (int it = 0; it < ntiles; it++) {
        const int stage = it & 1;
        if (it + 1 < ntiles) fa_load_kv<HD>(kbase + (stage ^ 1) * C::KV_BYTES, vbase + (stage ^ 1) * C::KV_BYTES, kb, vb, kw, vw, cap, j_start + (it + 1) * C::BK, hi_max);
        kf_cp_commit();
        kf_cp_wait<1>();
        __syncthreads();
        const u32 ks = kbase + stage * C::KV_BYTES, vs = vbase + stage * C::KV_BYTES;
        const int j0 = j_start + it * C::BK;

        // S = Q K^T over this tile, sixteen rows by BK keys per warp
        float s[C::NT][4];
#pragma unroll
        for (int n = 0; n < C::NT; n++) s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.0f;
#pragma unroll
        for (int kk = 0; kk < C::KS; kk++) {
            u32 a0, a1, a2, a3;
            kf_ldmatrix_x4(a0, a1, a2, a3, q_lane + (((2 * kk + lm_hi) ^ q_key) << 4));
#pragma unroll
            for (int n = 0; n < C::NT; n += 2) {
                int krow = 8 * (n + lm_hi) + (lane & 7);
                u32 b0, b1, b2, b3;
                kf_ldmatrix_x4(b0, b1, b2, b3, ks + krow * C::ROWB + (((2 * kk + ((lane >> 3) & 1)) ^ (krow & 7)) << 4));
                kf_mma_f16(s[n], a0, a1, a2, a3, b0, b1);
                kf_mma_f16(s[n + 1], a0, a1, a2, a3, b2, b3);
            }
        }

        // Scale, cap, and mask, then the online softmax step of both rows
        float mx_a = KF_NEG_INF, mx_b = KF_NEG_INF;
#pragma unroll
        for (int n = 0; n < C::NT; n++) {
#pragma unroll
            for (int e = 0; e < 4; e++) {
                int j = j0 + 8 * n + 2 * t + (e & 1);
                float v = s[n][e] * scale;
                if (softcap != 0.0f) v = softcap * tanhf(v / softcap);
                bool ok = (e < 2) ? (j >= lo_a && j <= hi_a) : (j >= lo_b && j <= hi_b);
                v = ok ? v : KF_NEG_INF;
                s[n][e] = v;
                if (e < 2) mx_a = fmaxf(mx_a, v);
                else mx_b = fmaxf(mx_b, v);
            }
        }
        mx_a = fmaxf(mx_a, __shfl_xor_sync(0xffffffffu, mx_a, 1));
        mx_a = fmaxf(mx_a, __shfl_xor_sync(0xffffffffu, mx_a, 2));
        mx_b = fmaxf(mx_b, __shfl_xor_sync(0xffffffffu, mx_b, 1));
        mx_b = fmaxf(mx_b, __shfl_xor_sync(0xffffffffu, mx_b, 2));
        float mn_a = fmaxf(m_a, mx_a), mn_b = fmaxf(m_b, mx_b);
        float c_a = mn_a == KF_NEG_INF ? 1.0f : exp2f((m_a - mn_a) * FA_LOG2E);
        float c_b = mn_b == KF_NEG_INF ? 1.0f : exp2f((m_b - mn_b) * FA_LOG2E);
        float sum_a = 0.0f, sum_b = 0.0f;
        u32 pa[C::NT / 2][4];
#pragma unroll
        for (int n = 0; n < C::NT; n++) {
            float p0 = s[n][0] == KF_NEG_INF ? 0.0f : exp2f((s[n][0] - mn_a) * FA_LOG2E);
            float p1 = s[n][1] == KF_NEG_INF ? 0.0f : exp2f((s[n][1] - mn_a) * FA_LOG2E);
            float p2 = s[n][2] == KF_NEG_INF ? 0.0f : exp2f((s[n][2] - mn_b) * FA_LOG2E);
            float p3 = s[n][3] == KF_NEG_INF ? 0.0f : exp2f((s[n][3] - mn_b) * FA_LOG2E);
            u32 h01 = kf_pack_half2(p0, p1), h23 = kf_pack_half2(p2, p3);
            // The rounded probabilities are what the value product sees, so the row sums use them too
            sum_a += kf_h2f(h01 & 0xffffu) + kf_h2f(h01 >> 16);
            sum_b += kf_h2f(h23 & 0xffffu) + kf_h2f(h23 >> 16);
            pa[n >> 1][(n & 1) * 2] = h01;
            pa[n >> 1][(n & 1) * 2 + 1] = h23;
        }
        sum_a += __shfl_xor_sync(0xffffffffu, sum_a, 1);
        sum_a += __shfl_xor_sync(0xffffffffu, sum_a, 2);
        sum_b += __shfl_xor_sync(0xffffffffu, sum_b, 1);
        sum_b += __shfl_xor_sync(0xffffffffu, sum_b, 2);
        l_a = l_a * c_a + sum_a;
        l_b = l_b * c_b + sum_b;
        m_a = mn_a;
        m_b = mn_b;
#pragma unroll
        for (int n = 0; n < C::OT; n++) {
            o[n][0] *= c_a;
            o[n][1] *= c_a;
            o[n][2] *= c_b;
            o[n][3] *= c_b;
        }

        // O += P V, keys in steps of sixteen, values transposed out of shared memory
#pragma unroll
        for (int kk = 0; kk < C::BK / 16; kk++) {
#pragma unroll
            for (int n = 0; n < C::OT; n += 2) {
                int vrow = 16 * kk + lm_row;
                u32 b0, b1, b2, b3;
                kf_ldmatrix_x4_trans(b0, b1, b2, b3, vs + vrow * C::ROWB + (((n + lm_hi) ^ (vrow & 7)) << 4));
                kf_mma_f16(o[n], pa[kk][0], pa[kk][1], pa[kk][2], pa[kk][3], b0, b1);
                kf_mma_f16(o[n + 1], pa[kk][0], pa[kk][1], pa[kk][2], pa[kk][3], b2, b3);
            }
        }
        __syncthreads();
    }
    kf_cp_wait<0>();

    // Normalize with the sinks and write the rows that exist
    float sink = sinks ? sinks[h] : KF_NEG_INF;
    float tot_a = l_a + (sinks ? exp2f((sink - m_a) * FA_LOG2E) : 0.0f);
    float tot_b = l_b + (sinks ? exp2f((sink - m_b) * FA_LOG2E) : 0.0f);
    float inv_a = tot_a > 0.0f ? 1.0f / tot_a : 0.0f, inv_b = tot_b > 0.0f ? 1.0f / tot_b : 0.0f;
    const i64 ow = (i64)n_head * HD;
    if (row_a < nrows) {
        float* orow = out + (i64)(r0 + row_a) * ow + (i64)h * HD;
#pragma unroll
        for (int n = 0; n < C::OT; n++) *reinterpret_cast<float2*>(orow + 8 * n + 2 * t) = make_float2(o[n][0] * inv_a, o[n][1] * inv_a);
    }
    if (row_b < nrows) {
        float* orow = out + (i64)(r0 + row_b) * ow + (i64)h * HD;
#pragma unroll
        for (int n = 0; n < C::OT; n++) *reinterpret_cast<float2*>(orow + 8 * n + 2 * t) = make_float2(o[n][2] * inv_b, o[n][3] * inv_b);
    }
}

#define FA_ENTRY(HD)                                                                                                                                                      \
    extern "C" __global__ void __launch_bounds__(FA_THREADS, 1) flash_prefill_hd##HD(const float* q, const u16* kc, const u16* vc, const int* pos, const int* seq, const int* attend, \
                                                                                     const int* tiles, const float* sinks, float* out, int n_head, int n_head_kv, float scale, int window, \
                                                                                     float softcap, int cap) {                                                                           \
        fa_kernel<HD>(q, kc, vc, pos, seq, attend, tiles, sinks, out, n_head, n_head_kv, scale, window, softcap, cap);                                                                   \
    }

FA_ENTRY(64)
FA_ENTRY(80)
FA_ENTRY(96)
FA_ENTRY(112)
FA_ENTRY(128)
FA_ENTRY(256)

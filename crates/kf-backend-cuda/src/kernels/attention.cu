#include "common.cuh"
#include "attention_decode.cuh"

#define KF_AT_WARPS 4
#define KF_AT_V 16
#define KF_AT_CHUNK (KF_AT_V * 32)
#define KF_AT_PART 128

// Copies the new key and value rows into the half precision ring cache of each row's slot at the row's position
extern "C" __global__ void kv_append(const float* k, const float* v, u16* kc, u16* vc, const int* pos, const int* seq, int rows, int kw, int vw, int cap) {
    i64 nk = (i64)rows * kw, nv = (i64)rows * vw;
    for (i64 t = blockIdx.x * (i64)blockDim.x + threadIdx.x; t < nk + nv; t += (i64)gridDim.x * blockDim.x) {
        if (t < nk) {
            i64 r = t / kw, c = t - r * kw;
            i64 slot = (i64)seq[r] * cap + (pos[r] % cap);
            kc[slot * kw + c] = kf_f2h(k[t]);
        } else {
            i64 u = t - nk, r = u / vw, c = u - r * vw;
            i64 slot = (i64)seq[r] * cap + (pos[r] % cap);
            vc[slot * vw + c] = kf_f2h(v[u]);
        }
    }
}

// Moves the cache rows base plus idx[i] of one slot to base plus i, one block reading every kept row before any is
// written, so a row a later index keeps is never overwritten first
extern "C" __global__ void kv_compact(u16* kc, u16* vc, int cap, int kw, int vw, int seq, int base, const int* idx, int n) {
    extern __shared__ __align__(16) u16 held[];
    const int per = kw + vw;
    for (int t = threadIdx.x; t < n * per; t += blockDim.x) {
        const int i = t / per, c = t - i * per;
        const i64 slot = (i64)seq * cap + ((base + idx[i]) % cap);
        held[t] = c < kw ? kc[slot * kw + c] : vc[slot * vw + (c - kw)];
    }
    __syncthreads();
    for (int t = threadIdx.x; t < n * per; t += blockDim.x) {
        const int i = t / per, c = t - i * per;
        const i64 slot = (i64)seq * cap + ((base + i) % cap);
        if (c < kw) kc[slot * kw + c] = held[t];
        else vc[slot * vw + (c - kw)] = held[t];
    }
}

// Dot of the query in shared memory with one half precision key row, eight elements per vector load when the width allows
__device__ __forceinline__ float kf_at_score(const float* qs, const u16* kj, int hd, bool vec) {
    float d = 0.0f;
    if (vec) {
        const uint4* k4 = (const uint4*)kj;
        const float4* q4 = (const float4*)qs;
        for (int i = 0; i < hd; i += 8) {
            uint4 u = k4[i >> 3];
            float4 a = q4[i >> 2], b = q4[(i >> 2) + 1];
            d += a.x * kf_h2f(u.x & 0xffffu) + a.y * kf_h2f(u.x >> 16) + a.z * kf_h2f(u.y & 0xffffu) + a.w * kf_h2f(u.y >> 16);
            d += b.x * kf_h2f(u.z & 0xffffu) + b.y * kf_h2f(u.z >> 16) + b.z * kf_h2f(u.w & 0xffffu) + b.w * kf_h2f(u.w >> 16);
        }
    } else {
        for (int i = 0; i < hd; i++) d += qs[i] * kf_h2f(kj[i]);
    }
    return d;
}

// The three words of a row of a tree pass as the decode kernel reads them, see kf_attention_decode, every key seen
// when the pass is no tree
__device__ __forceinline__ unsigned long long kf_at_tree_mask(const int* tree, int r) {
    return tree ? ((unsigned long long)(u32)tree[3 * r] | ((unsigned long long)(u32)tree[3 * r + 1] << 32)) : ~0ull;
}

// Online softmax of one warp over tiles of 32 keys, one key per lane scored at once, values accumulated in key order
// with half precision rounding, the keys within 64 of the row's last key hi_row seen as the tree mask says
template <int NV>
__device__ __forceinline__ void kf_at_tiles(const float* qs, const u16* kb, const u16* vb, i64 kw, i64 vw, int hd, int hdv, int cap, int base, int lo, int hi, int hi_row, unsigned long long tmask, float scale, float softcap, float& m, float& l, float* acc) {
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    bool vec = (hd & 7) == 0;
    for (int j0 = lo + warp * 32; j0 <= hi; j0 += KF_AT_WARPS * 32) {
        int j = j0 + lane;
        bool ok = j <= hi && (hi_row - j >= 64 || ((tmask >> (hi_row - j)) & 1ull));
        float s = KF_NEG_INF;
        if (ok) {
            i64 idx = cap > 0 ? (i64)(j % cap) : (i64)(j - base);
            s = kf_at_score(qs, kb + idx * kw, hd, vec) * scale;
            if (softcap != 0.0f) s = softcap * tanhf(s / softcap);
        }
        float mn = fmaxf(m, kf_warp_max(s));
        float c = expf(m - mn);
        float p = ok ? expf(s - mn) : 0.0f;
        l = l * c + kf_warp_sum(p);
#pragma unroll
        for (int t = 0; t < NV; t++) acc[t] = kf_round_f16(acc[t] * c);
        m = mn;
        int n = min(32, hi - j0 + 1);
        for (int jj = 0; jj < n; jj++) {
            float pj = __shfl_sync(0xffffffffu, p, jj);
            int jk = j0 + jj;
            i64 idx = cap > 0 ? (i64)(jk % cap) : (i64)(jk - base);
            const u16* vj = vb + idx * vw;
#pragma unroll
            for (int t = 0; t < NV; t++) {
                int i = lane + 32 * t;
                if (i < hdv) acc[t] = kf_round_f16(acc[t] + pj * kf_h2f(vj[i]));
            }
        }
    }
}

// Joins the warps of a block into the output row with the sinks, or into a partial of maximum, sum, and accumulator when prow is set
template <int NV>
__device__ __forceinline__ void kf_at_finish(float m, float l, const float* acc, float* ms, float* ls, float* accs, int hdv, const float* sinks, int h, float* orow, float* prow) {
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (lane == 0) {
        ms[warp] = m;
        ls[warp] = l;
    }
#pragma unroll
    for (int t = 0; t < NV; t++) accs[warp * KF_AT_CHUNK + lane + 32 * t] = acc[t];
    __syncthreads();
    float mx = KF_NEG_INF;
    for (int w = 0; w < KF_AT_WARPS; w++) if (ls[w] > 0.0f) mx = fmaxf(mx, ms[w]);
    float sink = (sinks && !prow) ? sinks[h] : 0.0f;
    if (sinks && !prow) mx = fmaxf(mx, sink);
    float tot = (sinks && !prow) ? expf(sink - mx) : 0.0f;
    for (int w = 0; w < KF_AT_WARPS; w++) tot += ls[w] > 0.0f ? ls[w] * expf(ms[w] - mx) : 0.0f;
    if (prow && threadIdx.x == 0) {
        prow[0] = mx;
        prow[1] = tot;
    }
    for (int i = threadIdx.x; i < hdv; i += blockDim.x) {
        float o = 0.0f;
        for (int w = 0; w < KF_AT_WARPS; w++) o += ls[w] > 0.0f ? accs[w * KF_AT_CHUNK + i] * expf(ms[w] - mx) : 0.0f;
        if (prow) prow[2 + i] = o;
        else orow[i] = tot > 0.0f ? o / tot : 0.0f;
    }
}

#define KF_AT_RUN(NV)                                                                                                       \
    {                                                                                                                       \
        kf_at_tiles<NV>(qs, kb, vb, kw, vw, hd, hdv, cap, base, lo, hi, hi_row, tmask, scale, softcap, m, l, acc);          \
        kf_at_finish<NV>(m, l, acc, ms, ls, accs, hdv, sinks, h, orow, prow);                                               \
    }

#define KF_AT_DISPATCH(hdv)                                                                                                 \
    if (hdv <= 32) KF_AT_RUN(1) else if (hdv <= 64) KF_AT_RUN(2) else if (hdv <= 128) KF_AT_RUN(4) else if (hdv <= 256) KF_AT_RUN(8) else KF_AT_RUN(16)

// Causal attention of one head and row over the half precision ring cache, the keys of grid depth z's part, written to the output when there is one part and as a partial otherwise
extern "C" __global__ void attention(const float* q, const u16* kc, const u16* vc, const int* pos_in, const int* seq, const int* attend, const float* sinks, float* out, float* parts, int part_keys, int n_head, int n_head_kv, int hd, int hdv, float scale, int window, float softcap, int cap, const int* tree) {
    extern __shared__ float smem[];
    float* qs = smem;
    float* ms = qs + hd;
    float* ls = ms + KF_AT_WARPS;
    float* accs = ls + KF_AT_WARPS;
    int r = blockIdx.x, h = blockIdx.y, part = blockIdx.z;
    int pos = pos_in[r];
    int hi = max(pos, attend[r]);
    const int hi_row = hi;
    const unsigned long long tmask = kf_at_tree_mask(tree, r);
    const int wpos = tree ? tree[3 * r + 2] : pos;
    int lo = window > 0 ? max(0, wpos - (window - 1)) : 0;
    // Parts split the row's own key range, which starts past zero and can run past the ring size under a window
    if (gridDim.z > 1) {
        int base = lo;
        lo = base + part * part_keys;
        hi = min(hi, base + part * part_keys + part_keys - 1);
    }
    int group = n_head / max(n_head_kv, 1);
    int hk = h / group;
    i64 kw = (i64)n_head_kv * hd, vw = (i64)n_head_kv * hdv;
    int base = 0;
    const u16* kb = kc + (i64)seq[r] * cap * kw + (i64)hk * hd;
    const u16* vb = vc + (i64)seq[r] * cap * vw + (i64)hk * hdv;
    const float* qr = q + ((i64)r * n_head + h) * hd;
    for (int i = threadIdx.x; i < hd; i += blockDim.x) qs[i] = kf_round_f16(qr[i]);
    __syncthreads();
    float m = KF_NEG_INF, l = 0.0f, acc[KF_AT_V];
#pragma unroll
    for (int t = 0; t < KF_AT_V; t++) acc[t] = 0.0f;
    float* orow = out + ((i64)r * n_head + h) * hdv;
    float* prow = gridDim.z > 1 ? parts + (((i64)part * gridDim.x + r) * n_head + h) * (2 + hdv) : nullptr;
    KF_AT_DISPATCH(hdv)
}

// Sets every partial softmax to empty, a maximum of minus infinity with a zero sum and accumulator
extern "C" __global__ void attention_clear(float* parts, i64 n, int width) {
    for (i64 t = blockIdx.x * (i64)blockDim.x + threadIdx.x; t < n; t += (i64)gridDim.x * blockDim.x) {
        parts[t] = (t % width) == 0 ? KF_NEG_INF : 0.0f;
    }
}

// Partial attention of one head and row over key positions jlo through jhi held contiguously from base or in a ring of cap rows per slot, written as maximum, sum, and unnormalized accumulator
extern "C" __global__ void attention_part(const float* q, const u16* kb_in, const u16* vb_in, const int* pos_in, const int* seq, const int* attend, int row0, int base, int cap, int jlo, int jhi, float* parts, int part, int rows_total, int n_head, int n_head_kv, int hd, int hdv, float scale, int window, float softcap, const int* tree) {
    extern __shared__ float smem[];
    float* qs = smem;
    float* ms = qs + hd;
    float* ls = ms + KF_AT_WARPS;
    float* accs = ls + KF_AT_WARPS;
    int r = row0 + blockIdx.x, h = blockIdx.y;
    int pos = pos_in[r];
    const int hi_row = max(pos, attend[r]);
    const unsigned long long tmask = kf_at_tree_mask(tree, r);
    const int wpos = tree ? tree[3 * r + 2] : pos;
    int start = window > 0 ? max(0, wpos - (window - 1)) : 0;
    int lo = max(jlo, start), hi = min(jhi, hi_row);
    int group = n_head / max(n_head_kv, 1);
    int hk = h / group;
    i64 kw = (i64)n_head_kv * hd, vw = (i64)n_head_kv * hdv;
    const u16* kb = (cap > 0 ? kb_in + (i64)seq[r] * cap * kw : kb_in) + (i64)hk * hd;
    const u16* vb = (cap > 0 ? vb_in + (i64)seq[r] * cap * vw : vb_in) + (i64)hk * hdv;
    const float* qr = q + ((i64)r * n_head + h) * hd;
    for (int i = threadIdx.x; i < hd; i += blockDim.x) qs[i] = kf_round_f16(qr[i]);
    __syncthreads();
    float m = KF_NEG_INF, l = 0.0f, acc[KF_AT_V];
#pragma unroll
    for (int t = 0; t < KF_AT_V; t++) acc[t] = 0.0f;
    const float* sinks = nullptr;
    float* orow = nullptr;
    float* prow = parts + (((i64)part * rows_total + r) * n_head + h) * (2 + hdv);
    KF_AT_DISPATCH(hdv)
}

// Joins the partial softmaxes of every part with the sinks into the attention output of one row and head
extern "C" __global__ void attention_merge(const float* parts, int n_parts, int rows_total, const float* sinks, float* out, int n_head, int hdv) {
    int r = blockIdx.x, h = blockIdx.y;
    int width = 2 + hdv;
    const float* p0 = parts + ((i64)r * n_head + h) * width;
    i64 stride = (i64)rows_total * n_head * width;
    float mx = KF_NEG_INF;
    for (int p = 0; p < n_parts; p++) {
        const float* pp = p0 + p * stride;
        if (pp[1] > 0.0f) mx = fmaxf(mx, pp[0]);
    }
    float sink = sinks ? sinks[h] : 0.0f;
    if (sinks) mx = fmaxf(mx, sink);
    float tot = sinks ? expf(sink - mx) : 0.0f;
    for (int p = 0; p < n_parts; p++) {
        const float* pp = p0 + p * stride;
        if (pp[1] > 0.0f) tot += pp[1] * expf(pp[0] - mx);
    }
    float* orow = out + ((i64)r * n_head + h) * hdv;
    for (int i = threadIdx.x; i < hdv; i += blockDim.x) {
        float o = 0.0f;
        for (int p = 0; p < n_parts; p++) {
            const float* pp = p0 + p * stride;
            if (pp[1] > 0.0f) o += pp[2 + i] * expf(pp[0] - mx);
        }
        orow[i] = tot > 0.0f ? o / tot : 0.0f;
    }
}

// Attention of one head and row over every row of the pass, keys and values single precision and uncached
extern "C" __global__ void attention_full(const float* q, const float* k, const float* v, float* out, int rows, int n_head, int hd, float scale) {
    extern __shared__ float smem[];
    float* qs = smem;
    float* ms = qs + hd;
    float* ls = ms + KF_AT_WARPS;
    float* accs = ls + KF_AT_WARPS;
    int r = blockIdx.x, h = blockIdx.y, v0 = blockIdx.z * KF_AT_CHUNK;
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    i64 width = (i64)n_head * hd;
    const float* qr = q + (i64)r * width + (i64)h * hd;
    for (int i = threadIdx.x; i < hd; i += blockDim.x) qs[i] = qr[i];
    __syncthreads();
    float m = KF_NEG_INF, l = 0.0f, acc[KF_AT_V];
#pragma unroll
    for (int t = 0; t < KF_AT_V; t++) acc[t] = 0.0f;
    for (int j = warp; j < rows; j += KF_AT_WARPS) {
        const float* kj = k + (i64)j * width + (i64)h * hd;
        float d = 0.0f;
        for (int i = lane; i < hd; i += 32) d += qs[i] * kj[i];
        d = kf_warp_sum(d);
        float s = d * scale;
        float mn = fmaxf(m, s);
        float c = expf(m - mn), p = expf(s - mn);
        l = l * c + p;
        const float* vj = v + (i64)j * width + (i64)h * hd + v0;
#pragma unroll
        for (int t = 0; t < KF_AT_V; t++) {
            int i = lane + 32 * t;
            if (v0 + i < hd) acc[t] = acc[t] * c + p * vj[i];
        }
        m = mn;
    }
    if (lane == 0) {
        ms[warp] = m;
        ls[warp] = l;
    }
#pragma unroll
    for (int t = 0; t < KF_AT_V; t++) accs[warp * KF_AT_CHUNK + lane + 32 * t] = acc[t];
    __syncthreads();
    float mx = KF_NEG_INF;
    for (int w = 0; w < KF_AT_WARPS; w++) mx = fmaxf(mx, ms[w]);
    float tot = 0.0f;
    for (int w = 0; w < KF_AT_WARPS; w++) tot += ls[w] > 0.0f ? ls[w] * expf(ms[w] - mx) : 0.0f;
    float* orow = out + (i64)r * width + (i64)h * hd + v0;
    for (int i = threadIdx.x; i < KF_AT_CHUNK && v0 + i < hd; i += blockDim.x) {
        float o = 0.0f;
        for (int w = 0; w < KF_AT_WARPS; w++) o += ls[w] > 0.0f ? accs[w * KF_AT_CHUNK + i] * expf(ms[w] - mx) : 0.0f;
        orow[i] = o / tot;
    }
}

#define KF_AT_DECODE(HD)                                                                                                                                                 \
    extern "C" __global__ void __launch_bounds__(KF_DEC_THREADS) attention_decode_hd##HD(const float* q, u16* kc, u16* vc, const int* pos, const int* seq, const int* attend, \
                                                                              const float* sinks, float* out, float* parts, int n_head, int n_head_kv, float scale, int window, \
                                                                              float softcap, int cap, int rows, int* counters, int kind, i8* qo, float* s16o, float* m16o, float* s32o, float* m32o, \
                                                                              const float* k_in, const float* v_in, const int* rope4, const float2* cst, int half, int neox, float mscale, int mode, int s0, int s1, int s2, int sect, \
                                                                              int norm_q, const float* wq, const float* bq, float eps_q, int norm_k, const float* wk, const float* bk, float eps_k, const int* tree) { \
        kf_attention_decode<HD, 4, 8, 4>(blockIdx.x, blockIdx.y, blockIdx.z, gridDim.y, gridDim.z, q, kc, vc, pos, seq, attend, sinks, out, parts, n_head, n_head_kv, scale, window, softcap, cap, rows, counters, kind, qo, s16o, m16o, s32o, m32o, \
                                k_in, v_in, rope4, cst, half, neox, mscale, mode, s0, s1, s2, sect, norm_q, wq, bq, eps_q, norm_k, wk, bk, eps_k, tree);                        \
    }

KF_AT_DECODE(64)
KF_AT_DECODE(128)
KF_AT_DECODE(256)

#include "common.cuh"

#define KF_AT_WARPS 4
#define KF_AT_V 16
#define KF_AT_CHUNK (KF_AT_V * 32)

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

// Causal attention of one head and row over the half precision cache with online softmax, sinks, window and softcap
extern "C" __global__ void attention(const float* q, const u16* kc, const u16* vc, const int* pos_in, const int* seq, const int* attend, const float* sinks, float* out, int n_head, int n_head_kv, int hd, int hdv, float scale, int window, float softcap, int cap) {
    extern __shared__ float smem[];
    float* qs = smem;
    float* ms = qs + hd;
    float* ls = ms + KF_AT_WARPS;
    float* accs = ls + KF_AT_WARPS;
    int r = blockIdx.x, h = blockIdx.y, v0 = blockIdx.z * KF_AT_CHUNK;
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int pos = pos_in[r];
    int hi = max(pos, attend[r]);
    int start = window > 0 ? max(0, pos - (window - 1)) : 0;
    i64 ring = (i64)seq[r] * cap;
    int group = n_head / max(n_head_kv, 1);
    int hk = h / group;
    i64 kw = (i64)n_head_kv * hd, vw = (i64)n_head_kv * hdv;
    const float* qr = q + ((i64)r * n_head + h) * hd;
    for (int i = threadIdx.x; i < hd; i += blockDim.x) qs[i] = kf_round_f16(qr[i]);
    __syncthreads();
    float m = KF_NEG_INF, l = 0.0f, acc[KF_AT_V];
#pragma unroll
    for (int t = 0; t < KF_AT_V; t++) acc[t] = 0.0f;
    for (int j = start + warp; j <= hi; j += KF_AT_WARPS) {
        const u16* kj = kc + (ring + j % cap) * kw + (i64)hk * hd;
        float d = 0.0f;
        for (int i = lane; i < hd; i += 32) d += qs[i] * kf_h2f(kj[i]);
        d = kf_warp_sum(d);
        float s = d * scale;
        if (softcap != 0.0f) s = softcap * tanhf(s / softcap);
        float mn = fmaxf(m, s);
        float c = expf(m - mn), p = expf(s - mn);
        l = l * c + p;
        const u16* vj = vc + (ring + j % cap) * vw + (i64)hk * hdv + v0;
#pragma unroll
        for (int t = 0; t < KF_AT_V; t++) {
            int i = lane + 32 * t;
            if (v0 + i < hdv) acc[t] = kf_round_f16(kf_round_f16(acc[t] * c) + p * kf_h2f(vj[i]));
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
    float sink = sinks ? sinks[h] : 0.0f;
    if (sinks) mx = fmaxf(mx, sink);
    float tot = sinks ? expf(sink - mx) : 0.0f;
    for (int w = 0; w < KF_AT_WARPS; w++) tot += ls[w] * expf(ms[w] - mx);
    float* orow = out + ((i64)r * n_head + h) * hdv + v0;
    for (int i = threadIdx.x; i < KF_AT_CHUNK && v0 + i < hdv; i += blockDim.x) {
        float o = 0.0f;
        for (int w = 0; w < KF_AT_WARPS; w++) o += accs[w * KF_AT_CHUNK + i] * expf(ms[w] - mx);
        orow[i] = o / tot;
    }
}

// Sets every partial softmax to empty, a maximum of minus infinity with a zero sum and accumulator
extern "C" __global__ void attention_clear(float* parts, i64 n, int width) {
    for (i64 t = blockIdx.x * (i64)blockDim.x + threadIdx.x; t < n; t += (i64)gridDim.x * blockDim.x) {
        parts[t] = (t % width) == 0 ? KF_NEG_INF : 0.0f;
    }
}

// Partial attention of one head and row over key positions jlo through jhi held contiguously from base or in a ring of cap rows per slot, written as maximum, sum, and unnormalized accumulator
extern "C" __global__ void attention_part(const float* q, const u16* kb, const u16* vb, const int* pos_in, const int* seq, const int* attend, int row0, int base, int cap, int jlo, int jhi, float* parts, int part, int rows_total, int n_head, int n_head_kv, int hd, int hdv, float scale, int window, float softcap) {
    extern __shared__ float smem[];
    float* qs = smem;
    float* ms = qs + hd;
    float* ls = ms + KF_AT_WARPS;
    float* accs = ls + KF_AT_WARPS;
    int r = row0 + blockIdx.x, h = blockIdx.y, v0 = blockIdx.z * KF_AT_CHUNK;
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int pos = pos_in[r];
    int start = window > 0 ? max(0, pos - (window - 1)) : 0;
    int lo = max(jlo, start), hi = min(jhi, max(pos, attend[r]));
    int group = n_head / max(n_head_kv, 1);
    int hk = h / group;
    i64 kw = (i64)n_head_kv * hd, vw = (i64)n_head_kv * hdv;
    const u16* kbase = cap > 0 ? kb + (i64)seq[r] * cap * kw : kb;
    const u16* vbase = cap > 0 ? vb + (i64)seq[r] * cap * vw : vb;
    const float* qr = q + ((i64)r * n_head + h) * hd;
    for (int i = threadIdx.x; i < hd; i += blockDim.x) qs[i] = kf_round_f16(qr[i]);
    __syncthreads();
    float m = KF_NEG_INF, l = 0.0f, acc[KF_AT_V];
#pragma unroll
    for (int t = 0; t < KF_AT_V; t++) acc[t] = 0.0f;
    for (int j = lo + warp; j <= hi; j += KF_AT_WARPS) {
        i64 idx = cap > 0 ? (i64)(j % cap) : (i64)(j - base);
        const u16* kj = kbase + idx * kw + (i64)hk * hd;
        float d = 0.0f;
        for (int i = lane; i < hd; i += 32) d += qs[i] * kf_h2f(kj[i]);
        d = kf_warp_sum(d);
        float s = d * scale;
        if (softcap != 0.0f) s = softcap * tanhf(s / softcap);
        float mn = fmaxf(m, s);
        float c = expf(m - mn), p = expf(s - mn);
        l = l * c + p;
        const u16* vj = vbase + idx * vw + (i64)hk * hdv + v0;
#pragma unroll
        for (int t = 0; t < KF_AT_V; t++) {
            int i = lane + 32 * t;
            if (v0 + i < hdv) acc[t] = kf_round_f16(kf_round_f16(acc[t] * c) + p * kf_h2f(vj[i]));
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
    float* prow = parts + (((i64)part * rows_total + r) * n_head + h) * (2 + hdv);
    if (blockIdx.z == 0 && threadIdx.x == 0) {
        prow[0] = mx;
        prow[1] = tot;
    }
    for (int i = threadIdx.x; i < KF_AT_CHUNK && v0 + i < hdv; i += blockDim.x) {
        float o = 0.0f;
        for (int w = 0; w < KF_AT_WARPS; w++) o += ls[w] > 0.0f ? accs[w * KF_AT_CHUNK + i] * expf(ms[w] - mx) : 0.0f;
        prow[2 + v0 + i] = o;
    }
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

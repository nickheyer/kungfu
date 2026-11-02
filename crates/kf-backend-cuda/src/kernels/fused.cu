// Fused producers of the int8 activations the products consume, and the per head norm folded into rope.
//
// Every kernel here quantizes through kf_quant_span, the warp wide form of the reference's activation rounding:
// kind 3 scales each 32 columns by its absolute maximum as Q8_0 does, kind 4 scales each 256 columns by its signed
// maximum as Q8_K does. A lane holds eight consecutive columns, a warp a span of 256, a block of eight warps a span
// of KF_QUANT_SPAN named by grid y, and the scales and group sums come out per 16 and per 32 columns.
#include "common.cuh"

#define KF_FUSED_THREADS 256

// Rows of n activations quantized as the reference rounds them before a product against the weight type
extern "C" __global__ void quant_act(const float* x, i8* q, float* s16, float* m16, float* s32, float* m32, int n, int kind) {
    const int row = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int span = blockIdx.y * KF_QUANT_SPAN + warp * 256;
    if (span >= n) return;
    const KfActRow o = kf_act_row(q, s16, m16, s32, m32, row, n, gridDim.x);
    const int c0 = span + lane * 8;
    float v[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    if (c0 < n) kf_load8(x + (i64)row * n + c0, v);
    kf_quant_span(v, c0, n, kind, o);
}

// Root mean square norm of each row with its weight and bias, then quantization, every block of a row summing the row
extern "C" __global__ void rms_norm_q8(const float* x, const float* w, const float* b, int n, float eps, int kind, i8* q, float* s16, float* m16, float* s32, float* m32) {
    __shared__ float scratch[32];
    const int row = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int span = blockIdx.y * KF_QUANT_SPAN + warp * 256;
    const float* xr = x + (i64)row * n;
    float s = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += xr[i] * xr[i];
    s = kf_block_sum(s, scratch);
    if (span >= n) return;
    const float inv = 1.0f / sqrtf(s / (float)n + eps);
    const KfActRow o = kf_act_row(q, s16, m16, s32, m32, row, n, gridDim.x);
    const int c0 = span + lane * 8;
    float v[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    if (c0 < n) {
        kf_load8(xr + c0, v);
        float ww[8] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
        float bb[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        if (w) kf_load8(w + c0, ww);
        if (b) kf_load8(b + c0, bb);
#pragma unroll
        for (int j = 0; j < 8; j++) v[j] = v[j] * (inv * ww[j]) + bb[j];
    }
    kf_quant_span(v, c0, n, kind, o);
}

// Activation of the gate rows times the up rows, then quantization, the two held apart or side by side in one row
extern "C" __global__ void gated_act_q8(const float* gate, const float* up, int n, int stride, int act, int kind, i8* q, float* s16, float* m16, float* s32, float* m32) {
    const int row = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int span = blockIdx.y * KF_QUANT_SPAN + warp * 256;
    if (span >= n) return;
    const KfActRow o = kf_act_row(q, s16, m16, s32, m32, row, n, gridDim.x);
    const int c0 = span + lane * 8;
    float v[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    if (c0 < n) {
        float gg[8], uu[8];
        kf_load8(gate + (i64)row * stride + c0, gg);
        kf_load8(up + (i64)row * stride + c0, uu);
#pragma unroll
        for (int j = 0; j < 8; j++) v[j] = kf_act(gg[j], act) * uu[j];
    }
    kf_quant_span(v, c0, n, kind, o);
}

// Element idx of a register array through a select chain, so the array never spills to local memory
template <int K>
__device__ __forceinline__ float kf_sel(const float* v, int idx) {
    float r = v[0];
#pragma unroll
    for (int k = 1; k < K; k++) r = idx == k ? v[k] : r;
    return r;
}

// One head of one row normed and rotated in registers: the lane holds elements lane + 32 k of the head, rp the
// row's four position streams
template <int K>
__device__ __forceinline__ void kf_norm_rope_head(const float* __restrict__ head, float* v, const int* __restrict__ rp, const float* __restrict__ freq, int head_dim, int half, int neox, float mscale, int norm, const float* __restrict__ w, const float* __restrict__ b, float eps, int mode, int s0, int s1, int s2, int sect) {
    const int lane = threadIdx.x & 31;
    float s = 0.0f;
#pragma unroll
    for (int k = 0; k < K; k++) {
        v[k] = head[lane + 32 * k];
        s += v[k] * v[k];
    }
    if (norm) {
        s = kf_warp_sum(s);
        float inv = 1.0f / sqrtf(s / (float)head_dim + eps);
#pragma unroll
        for (int k = 0; k < K; k++) {
            int c = lane + 32 * k;
            v[k] = v[k] * (inv * (w ? w[c] : 1.0f)) + (b ? b[c] : 0.0f);
        }
    }
    float r[K];
    if (neox) {
        // Pairs c and c + half, both in this lane since half is a multiple of 32
        const int kh = half / 32;
#pragma unroll
        for (int k = 0; k < K; k++) {
            int c = lane + 32 * k;
            if (c >= 2 * half) {
                r[k] = v[k];
                continue;
            }
            bool first = k < kh;
            int i = first ? c : c - half;
            float x0 = first ? v[k] : kf_sel<K>(v, k - kh);
            float x1 = first ? kf_sel<K>(v, k + kh) : v[k];
            float sn, co;
            sincosf(freq[i] * (float)(u32)rp[kf_rope_stream(i, mode, s0, s1, s2, sect)], &sn, &co);
            r[k] = first ? (x0 * co - x1 * sn) * mscale : (x0 * sn + x1 * co) * mscale;
        }
    } else {
        // Pairs of neighbours, the partner held by the neighbouring lane
#pragma unroll
        for (int k = 0; k < K; k++) {
            int c = lane + 32 * k;
            float other = __shfl_xor_sync(0xffffffffu, v[k], 1);
            if (c >= 2 * half) {
                r[k] = v[k];
                continue;
            }
            bool first = (c & 1) == 0;
            int i = c >> 1;
            float x0 = first ? v[k] : other, x1 = first ? other : v[k];
            float sn, co;
            sincosf(freq[i] * (float)(u32)rp[kf_rope_stream(i, mode, s0, s1, s2, sect)], &sn, &co);
            r[k] = first ? (x0 * co - x1 * sn) * mscale : (x0 * sn + x1 * co) * mscale;
        }
    }
#pragma unroll
    for (int k = 0; k < K; k++) v[k] = r[k];
}

// Per head norm then rope over rows of n_head heads, warp wid of the grid taking one row and head, written as f32
template <int K>
__device__ __forceinline__ void kf_norm_rope(int wid, const float* x, float* out, const int* rope4, const float* freq, int rows, int n_head, int head_dim, int half, int neox, float mscale, int norm, const float* w, const float* b, float eps, int mode, int s0, int s1, int s2, int sect) {
    if (wid >= rows * n_head) return;
    const int r = wid / n_head, h = wid - r * n_head, lane = threadIdx.x & 31;
    const i64 width = (i64)n_head * head_dim;
    float v[K];
    kf_norm_rope_head<K>(x + r * width + (i64)h * head_dim, v, rope4 + 4 * r, freq, head_dim, half, neox, mscale, norm, w, b, eps, mode, s0, s1, s2, sect);
    float* o = out + r * width + (i64)h * head_dim;
#pragma unroll
    for (int k = 0; k < K; k++) o[lane + 32 * k] = v[k];
}

// Per head norm then rope of the key rows appended to the half precision ring cache with the value rows, warp wid taking one row and kv head
template <int K>
__device__ __forceinline__ void kf_norm_rope_append(int wid, const float* k_in, const float* v_in, u16* kc, u16* vc, const int* rope4, const int* pos, const int* seq, int cap, const float* freq, int rows, int n_head_kv, int head_dim, int head_dim_v, int half, int neox, float mscale, int norm, const float* w, const float* b, float eps, int mode, int s0, int s1, int s2, int sect) {
    if (wid >= rows * n_head_kv) return;
    const int r = wid / n_head_kv, h = wid - r * n_head_kv, lane = threadIdx.x & 31;
    const i64 kw = (i64)n_head_kv * head_dim, vw = (i64)n_head_kv * head_dim_v;
    const i64 slot = (i64)seq[r] * cap + (pos[r] % cap);
    float v[K];
    kf_norm_rope_head<K>(k_in + r * kw + (i64)h * head_dim, v, rope4 + 4 * r, freq, head_dim, half, neox, mscale, norm, w, b, eps, mode, s0, s1, s2, sect);
    u16* ko = kc + slot * kw + (i64)h * head_dim;
#pragma unroll
    for (int k = 0; k < K; k++) ko[lane + 32 * k] = kf_f2h(v[k]);
    const float* vr = v_in + r * vw + (i64)h * head_dim_v;
    u16* vo = vc + slot * vw + (i64)h * head_dim_v;
    for (int c = lane; c < head_dim_v; c += 32) vo[c] = kf_f2h(vr[c]);
}

// The query rope and the key and value append of one layer in one launch, the query warps first
template <int K>
__device__ __forceinline__ void kf_norm_rope_pair(const float* q_in, float* q_out, int norm_q, const float* wq, const float* bq, float eps_q, const float* k_in, const float* v_in, u16* kc, u16* vc, int cap, int norm_k, const float* wk, const float* bk, float eps_k, const int* rope4, const int* pos, const int* seq, const float* freq, int rows, int n_head, int n_head_kv, int head_dim, int head_dim_v, int half, int neox, float mscale, int mode, int s0, int s1, int s2, int sect) {
    const int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int nq = rows * n_head;
    if (wid < nq) {
        kf_norm_rope<K>(wid, q_in, q_out, rope4, freq, rows, n_head, head_dim, half, neox, mscale, norm_q, wq, bq, eps_q, mode, s0, s1, s2, sect);
    } else {
        kf_norm_rope_append<K>(wid - nq, k_in, v_in, kc, vc, rope4, pos, seq, cap, freq, rows, n_head_kv, head_dim, head_dim_v, half, neox, mscale, norm_k, wk, bk, eps_k, mode, s0, s1, s2, sect);
    }
}

#define KF_ROPE_ENTRIES(K, HD)                                                                                                                                                 \
    extern "C" __global__ void norm_rope_hd##HD(const float* x, float* out, const int* rope4, const float* freq, int rows, int n_head, int half, int neox, float mscale, int norm, const float* w, const float* b, float eps, int mode, int s0, int s1, int s2, int sect) { \
        kf_norm_rope<K>((blockIdx.x * blockDim.x + threadIdx.x) >> 5, x, out, rope4, freq, rows, n_head, HD, half, neox, mscale, norm, w, b, eps, mode, s0, s1, s2, sect);    \
    }                                                                                                                                                                          \
    extern "C" __global__ void norm_rope_append_hd##HD(const float* k_in, const float* v_in, u16* kc, u16* vc, const int* rope4, const int* pos, const int* seq, int cap, const float* freq, int rows, int n_head_kv, int head_dim_v, int half, int neox, float mscale, int norm, const float* w, const float* b, float eps, int mode, int s0, int s1, int s2, int sect) { \
        kf_norm_rope_append<K>((blockIdx.x * blockDim.x + threadIdx.x) >> 5, k_in, v_in, kc, vc, rope4, pos, seq, cap, freq, rows, n_head_kv, HD, head_dim_v, half, neox, mscale, norm, w, b, eps, mode, s0, s1, s2, sect); \
    }                                                                                                                                                                          \
    extern "C" __global__ void norm_rope_pair_hd##HD(const float* q_in, float* q_out, int norm_q, const float* wq, const float* bq, float eps_q, const float* k_in, const float* v_in, u16* kc, u16* vc, int cap, int norm_k, const float* wk, const float* bk, float eps_k, const int* rope4, const int* pos, const int* seq, const float* freq, int rows, int n_head, int n_head_kv, int head_dim_v, int half, int neox, float mscale, int mode, int s0, int s1, int s2, int sect) { \
        kf_norm_rope_pair<K>(q_in, q_out, norm_q, wq, bq, eps_q, k_in, v_in, kc, vc, cap, norm_k, wk, bk, eps_k, rope4, pos, seq, freq, rows, n_head, n_head_kv, HD, head_dim_v, half, neox, mscale, mode, s0, s1, s2, sect); \
    }

KF_ROPE_ENTRIES(1, 32)
KF_ROPE_ENTRIES(2, 64)
KF_ROPE_ENTRIES(3, 96)
KF_ROPE_ENTRIES(4, 128)
KF_ROPE_ENTRIES(6, 192)
KF_ROPE_ENTRIES(8, 256)

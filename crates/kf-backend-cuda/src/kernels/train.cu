#include "common.cuh"

// The kernels a draft head learns with, a single precision product with either operand transposed, the gated
// activation and the rms norm with their backward passes, AdamW, and the smooth L1 loss with its gradient. The
// product is a 64 by 64 tile of 256 threads, four by four outputs each, over sixteen deep steps staged through shared
// memory, the transposes resolved while staging.
#define KF_TR_BM 64
#define KF_TR_BN 64
#define KF_TR_BK 16
#define KF_TR_THREADS 256

// C = alpha op(A) op(B) + beta C, A m by k and B k by n after the transposes, row major with the given leading dimensions
extern "C" __global__ void __launch_bounds__(KF_TR_THREADS) gemm_f32(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K, int lda, int ldb, int ldc, int ta, int tb, float alpha, float beta) {
    __shared__ float as[KF_TR_BK][KF_TR_BM + 4];
    __shared__ float bs[KF_TR_BK][KF_TR_BN + 4];
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    const int m0 = blockIdx.y * KF_TR_BM, n0 = blockIdx.x * KF_TR_BN;
    float acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = 0.0f;
    for (int k0 = 0; k0 < K; k0 += KF_TR_BK) {
        // 64 by 16 of A and 16 by 64 of B, four elements per thread each
#pragma unroll
        for (int e = 0; e < 4; e++) {
            const int idx = tid + e * KF_TR_THREADS;
            const int am = idx / KF_TR_BK, ak = idx - am * KF_TR_BK;
            const int gm = m0 + am, gk = k0 + ak;
            float v = 0.0f;
            if (gm < M && gk < K) v = ta ? A[(i64)gk * lda + gm] : A[(i64)gm * lda + gk];
            as[ak][am] = v;
            const int bk = idx / KF_TR_BN, bn = idx - bk * KF_TR_BN;
            const int gk2 = k0 + bk, gn = n0 + bn;
            float w = 0.0f;
            if (gk2 < K && gn < N) w = tb ? B[(i64)gn * ldb + gk2] : B[(i64)gk2 * ldb + gn];
            bs[bk][bn] = w;
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < KF_TR_BK; kk++) {
            float a[4], b[4];
#pragma unroll
            for (int i = 0; i < 4; i++) a[i] = as[kk][ty * 4 + i];
#pragma unroll
            for (int j = 0; j < 4; j++) b[j] = bs[kk][tx * 4 + j];
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 4; j++) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int m = m0 + ty * 4 + i;
        if (m >= M) continue;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int n = n0 + tx * 4 + j;
            if (n >= N) continue;
            float* c = C + (i64)m * ldc + n;
            *c = beta == 0.0f ? alpha * acc[i][j] : fmaf(alpha, acc[i][j], beta * *c);
        }
    }
}

__device__ __forceinline__ float kf_tr_silu(float x) {
    return x / (1.0f + __expf(-x));
}

// h = silu(g) u
extern "C" __global__ void silu_gate_fwd(const float* g, const float* u, float* h, i64 n) {
    for (i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x; i < n; i += (i64)gridDim.x * blockDim.x) h[i] = kf_tr_silu(g[i]) * u[i];
}

// dg = dh u silu'(g), du = dh silu(g)
extern "C" __global__ void silu_gate_bwd(const float* g, const float* u, const float* dh, float* dg, float* du, i64 n) {
    for (i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x; i < n; i += (i64)gridDim.x * blockDim.x) {
        const float x = g[i], s = 1.0f / (1.0f + __expf(-x)), si = x * s;
        dg[i] = dh[i] * u[i] * (s + si * (1.0f - s));
        du[i] = dh[i] * si;
    }
}

// y = x w / rms(x) per row, the inverse rms of every row kept for the backward pass
extern "C" __global__ void rms_fwd(const float* x, const float* w, float* y, float* inv, int cols, float eps) {
    __shared__ float red[32];
    const float* xr = x + (i64)blockIdx.x * cols;
    float s = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) s += xr[c] * xr[c];
    s = kf_block_sum(s, red);
    const float r = rsqrtf(s / (float)cols + eps);
    if (threadIdx.x == 0) inv[blockIdx.x] = r;
    float* yr = y + (i64)blockIdx.x * cols;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) yr[c] = xr[c] * r * w[c];
}

// dx = inv (w dy - x inv^2 mean(w dy x)) per row
extern "C" __global__ void rms_bwd(const float* x, const float* w, const float* inv, const float* dy, float* dx, int cols) {
    __shared__ float red[32];
    const i64 off = (i64)blockIdx.x * cols;
    const float r = inv[blockIdx.x];
    float s = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) s += w[c] * dy[off + c] * x[off + c];
    s = kf_block_sum(s, red);
    const float k = r * r * s / (float)cols;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) dx[off + c] = r * (w[c] * dy[off + c] - x[off + c] * k);
}

// dw[c] += sum over rows of dy x inv
extern "C" __global__ void rms_bwd_w(const float* x, const float* inv, const float* dy, float* dw, int rows, int cols) {
    for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < cols; c += gridDim.x * blockDim.x) {
        float s = 0.0f;
        for (int r = 0; r < rows; r++) s += dy[(i64)r * cols + c] * x[(i64)r * cols + c] * inv[r];
        dw[c] += s;
    }
}

// One AdamW step over n parameters with bias correction at step t, the gradient left untouched
extern "C" __global__ void adamw(float* p, const float* g, float* m, float* v, i64 n, float lr, float b1, float b2, float eps, float wd, int t) {
    const float c1 = 1.0f - powf(b1, (float)t), c2 = 1.0f - powf(b2, (float)t);
    for (i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x; i < n; i += (i64)gridDim.x * blockDim.x) {
        const float gi = g[i];
        const float mi = m[i] = b1 * m[i] + (1.0f - b1) * gi;
        const float vi = v[i] = b2 * v[i] + (1.0f - b2) * gi * gi;
        p[i] -= lr * ((mi / c1) / (sqrtf(vi / c2) + eps) + wd * p[i]);
    }
}

// y += alpha x
extern "C" __global__ void axpy(const float* x, float* y, i64 n, float alpha) {
    for (i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x; i < n; i += (i64)gridDim.x * blockDim.x) y[i] = fmaf(alpha, x[i], y[i]);
}

// Rows of a row major matrix gathered by index into another, each row cols wide
extern "C" __global__ void gather_rows(const float* src, const int* idx, float* dst, int rows, int cols) {
    for (i64 t = blockIdx.x * (i64)blockDim.x + threadIdx.x; t < (i64)rows * cols; t += (i64)gridDim.x * blockDim.x) {
        const int r = (int)(t / cols), c = (int)(t - (i64)r * cols);
        dst[t] = src[(i64)idx[r] * cols + c];
    }
}

// Half precision rows gathered by index into single precision rows
extern "C" __global__ void gather_rows_h(const u16* src, const int* idx, float* dst, int rows, int cols) {
    for (i64 t = blockIdx.x * (i64)blockDim.x + threadIdx.x; t < (i64)rows * cols; t += (i64)gridDim.x * blockDim.x) {
        const int r = (int)(t / cols), c = (int)(t - (i64)r * cols);
        dst[t] = kf_h2f(src[(i64)idx[r] * cols + c]);
    }
}

// Rows of bytes gathered by index into another array of rows
extern "C" __global__ void gather_bytes(const u8* src, const int* idx, u8* dst, int rows, i64 row_bytes) {
    for (i64 t = blockIdx.x * (i64)blockDim.x + threadIdx.x; t < (i64)rows * row_bytes; t += (i64)gridDim.x * blockDim.x) {
        const int r = (int)(t / row_bytes);
        const i64 c = t - (i64)r * row_bytes;
        dst[t] = src[(i64)idx[r] * row_bytes + c];
    }
}

// Product of up to KF_SK_ROWS rows against a row major weight, every block a slab of KF_SK_THREADS columns over one
// split of k with the rows of a staged KF_SK_K columns at a time, the partial sums of split s at parts[s][m][n]
#define KF_SK_ROWS 16
#define KF_SK_THREADS 128
#define KF_SK_K 64
extern "C" __global__ void __launch_bounds__(KF_SK_THREADS) gemm_skinny(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ parts, int M, int N, int K, int lda, int ldb, int kper) {
    __shared__ float as[KF_SK_ROWS][KF_SK_K];
    const int n = blockIdx.x * KF_SK_THREADS + threadIdx.x;
    const int k0 = blockIdx.y * kper, k1 = min(K, k0 + kper);
    float acc[KF_SK_ROWS];
#pragma unroll
    for (int r = 0; r < KF_SK_ROWS; r++) acc[r] = 0.0f;
    for (int kb = k0; kb < k1; kb += KF_SK_K) {
        const int kk = min(KF_SK_K, k1 - kb);
        for (int i = threadIdx.x; i < KF_SK_ROWS * KF_SK_K; i += KF_SK_THREADS) {
            const int r = i / KF_SK_K, c = i - r * KF_SK_K;
            as[r][c] = (r < M && c < kk) ? A[(i64)r * lda + kb + c] : 0.0f;
        }
        __syncthreads();
        if (n < N) {
            for (int c = 0; c < kk; c += 4) {
                float bv[4];
#pragma unroll
                for (int j = 0; j < 4; j++) bv[j] = c + j < kk ? B[(i64)(kb + c + j) * ldb + n] : 0.0f;
#pragma unroll
                for (int j = 0; j < 4; j++)
#pragma unroll
                    for (int r = 0; r < KF_SK_ROWS; r++) acc[r] = fmaf(as[r][c + j], bv[j], acc[r]);
            }
        }
        __syncthreads();
    }
    if (n < N) {
        for (int r = 0; r < M; r++) parts[((i64)blockIdx.y * M + r) * N + n] = acc[r];
    }
}

// C = alpha (sum of the splits' parts) + beta C, the splits added in order so the sum is the same every run
extern "C" __global__ void gemm_skinny_reduce(const float* __restrict__ parts, float* __restrict__ C, int M, int N, int ldc, int splits, float alpha, float beta) {
    for (i64 t = blockIdx.x * (i64)blockDim.x + threadIdx.x; t < (i64)M * N; t += (i64)gridDim.x * blockDim.x) {
        const int r = (int)(t / N), n = (int)(t - (i64)r * N);
        float s = 0.0f;
        for (int p = 0; p < splits; p++) s += parts[((i64)p * M + r) * N + n];
        float* c = C + (i64)r * ldc + n;
        *c = beta == 0.0f ? alpha * s : fmaf(alpha, s, beta * *c);
    }
}

#define KF_TA_T 64
#define KF_TA_THREADS 256
#define KF_DA_ANC 80
#define KF_SOFT_K 64
#define KF_BV_DEPTH 8

// Attention of m draft rows, one block per row and head, each row sees the rows of its slot's history ring whose
// positions lie at or before its own and within the window, the rows its ancestor list names among the node rows,
// and itself, which stands in the node rows at its own index. The ring holds cap rows by position modulo cap, hpos
// naming the position of each ring row and a negative position an empty one.
#define KF_DA_ANC 80
#define KF_SOFT_K 64
extern "C" __global__ void __launch_bounds__(256) draft_attn(const float* __restrict__ q, const float* __restrict__ kring, const float* __restrict__ vring, const int* __restrict__ hpos, int cap, const float* __restrict__ knode, const float* __restrict__ vnode, const int* __restrict__ anc, const int* __restrict__ pos, int window, float* __restrict__ out, int H, int hd, float scale) {
    extern __shared__ float sm[];
    float* qs = sm;
    float* sc = qs + hd;
    const int r = blockIdx.x / H, h = blockIdx.x - r * H, tid = threadIdx.x, D = H * hd;
    for (int d = tid; d < hd; d += blockDim.x) qs[d] = q[(i64)r * D + h * hd + d];
    __syncthreads();
    int nanc = 0;
    while (nanc < KF_DA_ANC && anc[r * KF_DA_ANC + nanc] >= 0) nanc++;
    const int p = pos[r];
    const int nk = cap + nanc;
    for (int j = tid; j < nk; j += blockDim.x) {
        float s = KF_NEG_INF;
        const float* kj = nullptr;
        if (j < cap) {
            const int hp = hpos[j];
            if (hp >= 0 && hp <= p && hp + window > p) kj = kring + (i64)j * D + h * hd;
        } else {
            kj = knode + (i64)anc[r * KF_DA_ANC + (j - cap)] * D + h * hd;
        }
        if (kj) {
            s = 0.0f;
            for (int d = 0; d < hd; d++) s = fmaf(qs[d], kj[d], s);
            s *= scale;
        }
        sc[j] = s;
    }
    __syncthreads();
    __shared__ float red[32];
    float m = KF_NEG_INF;
    for (int j = tid; j < nk; j += blockDim.x) m = fmaxf(m, sc[j]);
    m = kf_block_max(m, red);
    float l = 0.0f;
    for (int j = tid; j < nk; j += blockDim.x) {
        const float e = sc[j] == KF_NEG_INF ? 0.0f : __expf(sc[j] - m);
        sc[j] = e;
        l += e;
    }
    l = kf_block_sum(l, red);
    const float inv = l > 0.0f ? 1.0f / l : 0.0f;
    for (int d = tid; d < hd; d += blockDim.x) {
        float o = 0.0f;
        for (int j = 0; j < nk; j++) {
            if (sc[j] == 0.0f) continue;
            const float* vj = j < cap ? vring + (i64)j * D + h * hd : vnode + (i64)anc[r * KF_DA_ANC + (j - cap)] * D + h * hd;
            o = fmaf(sc[j], vj[d], o);
        }
        out[(i64)r * D + h * hd + d] = o * inv;
    }
}

// Attention of one query group of the block's training at drafted step `step`, one block per chunk and head
// query row i sees the window keys at or before it within the window, strictly before it when self_key is zero,
// and its own row at every chain step first to step, the last of those itself. The chain keys and values of step m
// lie at m times stepstride floats into k and v, the window keys in kwin and vwin as [B T D]. Probabilities go out
// as [B, H, T, T + chain] rows for the backward pass, chain being step minus first plus one.
extern "C" __global__ void __launch_bounds__(KF_TA_THREADS) chain_attn_fwd(const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v, i64 stepstride, int step, const float* __restrict__ kwin, const float* __restrict__ vwin, int first, int self_key, float* __restrict__ out, float* __restrict__ probs, int T, int H, int hd, int window, float scale) {
    extern __shared__ float sm[];
    float* ks = sm;
    float* vs = ks + KF_TA_T * hd;
    float* ps = vs + KF_TA_T * hd;
    const int b = blockIdx.x / H, h = blockIdx.x - b * H;
    const int tid = threadIdx.x, D = H * hd, chain = step - first + 1, nk = T + chain;
    const i64 base = (i64)b * T * D + (i64)h * hd;
    for (int i = tid; i < T * hd; i += KF_TA_THREADS) {
        const int t = i / hd, d = i - t * hd;
        ks[t * hd + d] = kwin[base + (i64)t * D + d];
        vs[t * hd + d] = vwin[base + (i64)t * D + d];
    }
    __syncthreads();
    const float* qk = q + (i64)step * stepstride;
    for (int e = tid; e < T * nk; e += KF_TA_THREADS) {
        const int i = e / nk, j = e - i * nk;
        float s = KF_NEG_INF;
        const float* qi = qk + base + (i64)i * D;
        if (j < T) {
            const bool seen = self_key ? (j <= i) : (j < i);
            if (seen && j > i - window) {
                s = 0.0f;
                for (int d = 0; d < hd; d++) s = fmaf(qi[d], ks[j * hd + d], s);
                s *= scale;
            }
        } else {
            const float* kj = k + (i64)(first + j - T) * stepstride + base + (i64)i * D;
            s = 0.0f;
            for (int d = 0; d < hd; d++) s = fmaf(qi[d], kj[d], s);
            s *= scale;
        }
        ps[e] = s;
    }
    __syncthreads();
    for (int i = tid / 4; i < T; i += KF_TA_THREADS / 4) {
        const int part = tid & 3;
        float m = KF_NEG_INF;
        for (int j = part; j < nk; j += 4) m = fmaxf(m, ps[i * nk + j]);
#pragma unroll
        for (int o = 1; o < 4; o <<= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        float l = 0.0f;
        for (int j = part; j < nk; j += 4) {
            const float p = ps[i * nk + j] == KF_NEG_INF ? 0.0f : __expf(ps[i * nk + j] - m);
            ps[i * nk + j] = p;
            l += p;
        }
#pragma unroll
        for (int o = 1; o < 4; o <<= 1) l += __shfl_xor_sync(0xffffffffu, l, o);
        const float inv = 1.0f / l;
        for (int j = part; j < nk; j += 4) ps[i * nk + j] *= inv;
    }
    __syncthreads();
    float* pout = probs + ((i64)b * H + h) * T * nk;
    for (int e = tid; e < T * nk; e += KF_TA_THREADS) pout[e] = ps[e];
    for (int e = tid; e < T * hd; e += KF_TA_THREADS) {
        const int i = e / hd, d = e - i * hd;
        float o = 0.0f;
        for (int j = 0; j <= i && j < T; j++) o = fmaf(ps[i * nk + j], vs[j * hd + d], o);
        for (int c = 0; c < chain; c++) o = fmaf(ps[i * nk + T + c], v[(i64)(first + c) * stepstride + base + (i64)i * D + d], o);
        out[(i64)step * stepstride + base + (i64)i * D + d] = o;
    }
}

// Gradients of chain_attn_fwd given dout and the saved probabilities. dq lands at step `step` of dq, the window
// keys' gradients add into dkwin and dvwin, the chain rows' into dk and dv at their steps, so every buffer
// accumulates the contributions of the later steps' queries.
extern "C" __global__ void __launch_bounds__(KF_TA_THREADS) chain_attn_bwd(const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v, i64 stepstride, int step, const float* __restrict__ kwin, const float* __restrict__ vwin, int first, const float* __restrict__ probs, const float* __restrict__ dout, float* __restrict__ dq, float* __restrict__ dk, float* __restrict__ dv, float* __restrict__ dkwin, float* __restrict__ dvwin, int T, int H, int hd, float scale) {
    extern __shared__ float sm[];
    float* as = sm;
    float* bs = as + KF_TA_T * hd;
    float* ps = bs + KF_TA_T * hd;
    const int b = blockIdx.x / H, h = blockIdx.x - b * H;
    const int tid = threadIdx.x, D = H * hd, chain = step - first + 1, nk = T + chain;
    const i64 base = (i64)b * T * D + (i64)h * hd;
    const i64 at = (i64)step * stepstride;
    const float* pin = probs + ((i64)b * H + h) * T * nk;
    // Phase one, window v and this step's dout staged, dv for the window keys, dp = do v for every key, ds kept in ps
    for (int i = tid; i < T * hd; i += KF_TA_THREADS) {
        const int t = i / hd, d = i - t * hd;
        as[t * hd + d] = vwin[base + (i64)t * D + d];
        bs[t * hd + d] = dout[at + base + (i64)t * D + d];
    }
    for (int e = tid; e < T * nk; e += KF_TA_THREADS) ps[e] = pin[e];
    __syncthreads();
    for (int e = tid; e < T * hd; e += KF_TA_THREADS) {
        const int j = e / hd, d = e - j * hd;
        float s = 0.0f;
        for (int i = j; i < T; i++) s = fmaf(ps[i * nk + j], bs[i * hd + d], s);
        dvwin[base + (i64)j * D + d] += s;
    }
    for (int e = tid; e < T * hd; e += KF_TA_THREADS) {
        const int i = e / hd, d = e - i * hd;
        for (int c = 0; c < chain; c++) dv[(i64)(first + c) * stepstride + base + (i64)i * D + d] += ps[i * nk + T + c] * bs[i * hd + d];
    }
    __syncthreads();
    for (int i = tid / 4; i < T; i += KF_TA_THREADS / 4) {
        const int part = tid & 3;
        float dot = 0.0f;
        for (int j = part; j < nk; j += 4) {
            const float p = ps[i * nk + j];
            if (p == 0.0f) continue;
            float dp = 0.0f;
            if (j < T) {
                for (int d = 0; d < hd; d++) dp = fmaf(bs[i * hd + d], as[j * hd + d], dp);
            } else {
                const float* vj = v + (i64)(first + j - T) * stepstride + base + (i64)i * D;
                for (int d = 0; d < hd; d++) dp = fmaf(bs[i * hd + d], vj[d], dp);
            }
            dot += p * dp;
            ps[i * nk + j] = p * dp;
        }
#pragma unroll
        for (int o = 1; o < 4; o <<= 1) dot += __shfl_xor_sync(0xffffffffu, dot, o);
        for (int j = part; j < nk; j += 4) {
            const float p = pin[i * nk + j];
            ps[i * nk + j] = p == 0.0f ? 0.0f : ps[i * nk + j] - p * dot;
        }
    }
    __syncthreads();
    // Phase two, window k and this step's q staged, dq for this step's rows, dk for the window and chain rows
    for (int i = tid; i < T * hd; i += KF_TA_THREADS) {
        const int t = i / hd, d = i - t * hd;
        as[t * hd + d] = kwin[base + (i64)t * D + d];
        bs[t * hd + d] = q[at + base + (i64)t * D + d];
    }
    __syncthreads();
    for (int e = tid; e < T * hd; e += KF_TA_THREADS) {
        const int i = e / hd, d = e - i * hd;
        float s = 0.0f;
        for (int j = 0; j <= i; j++) s = fmaf(ps[i * nk + j], as[j * hd + d], s);
        for (int c = 0; c < chain; c++) s = fmaf(ps[i * nk + T + c], k[(i64)(first + c) * stepstride + base + (i64)i * D + d], s);
        dq[at + base + (i64)i * D + d] = s * scale;
        float t = 0.0f;
        for (int ii = i; ii < T; ii++) t = fmaf(ps[ii * nk + i], bs[ii * hd + d], t);
        dkwin[base + (i64)i * D + d] += t * scale;
        for (int c = 0; c < chain; c++) dk[(i64)(first + c) * stepstride + base + (i64)i * D + d] += ps[i * nk + T + c] * bs[i * hd + d] * scale;
    }
}

// The soft target of one row read from packed candidates, k ids as floats then k logits then the log sum exp,
// every id mapped to a draft column by lut or dropped below zero, the kept mass renormalized into ps
__device__ __forceinline__ float kf_soft_target(const float* c, int k, const int* lut, int* cols, float* ps, float* red) {
    const float lse_t = c[2 * k];
    float mass = 0.0f;
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        const int col = lut[(int)c[i]];
        cols[i] = col;
        const float p = col >= 0 ? __expf(c[k + i] - lse_t) : 0.0f;
        ps[i] = p;
        mass += p;
    }
    return kf_block_sum(mass, red);
}

// The log sum exp of one row of V logits over the block
__device__ __forceinline__ float kf_row_lse(const float* zr, int V, float* red) {
    float m = KF_NEG_INF;
    for (int j = threadIdx.x; j < V; j += blockDim.x) m = fmaxf(m, zr[j]);
    m = kf_block_max(m, red);
    float l = 0.0f;
    for (int j = threadIdx.x; j < V; j += blockDim.x) l += __expf(zr[j] - m);
    l = kf_block_sum(l, red);
    return m + __logf(l);
}

// Cross entropy of rows of logits over a draft vocabulary against soft targets from packed candidate rows, weighted
// per row, the row's own loss into losses, minus one for a row without a target, its weighted loss into loss, the
// rows with a target into counted, and dz the weight times softmax minus target for rows with a target and zero
// elsewhere.
extern "C" __global__ void __launch_bounds__(256) soft_ce(const float* __restrict__ z, int V, const float* __restrict__ cands, int k, const int* __restrict__ lut, const float* __restrict__ w, float* __restrict__ dz, float* __restrict__ losses, float* __restrict__ loss, int* __restrict__ counted) {
    __shared__ float red[32];
    __shared__ int cols[KF_SOFT_K];
    __shared__ float ps[KF_SOFT_K];
    __shared__ float mass_s;
    const int r = blockIdx.x, tid = threadIdx.x;
    const float* zr = z + (i64)r * V;
    float* dr = dz + (i64)r * V;
    const float mass = kf_soft_target(cands + (i64)r * (2 * k + 1), k, lut, cols, ps, red);
    if (tid == 0) mass_s = mass;
    __syncthreads();
    if (mass_s <= 0.0f) {
        for (int j = tid; j < V; j += blockDim.x) dr[j] = 0.0f;
        if (tid == 0) losses[r] = -1.0f;
        return;
    }
    const float lse = kf_row_lse(zr, V, red);
    const float wr = w[r];
    for (int j = tid; j < V; j += blockDim.x) dr[j] = wr * __expf(zr[j] - lse);
    __syncthreads();
    float part = 0.0f;
    for (int i = tid; i < k; i += blockDim.x) {
        if (cols[i] < 0) continue;
        const float p = ps[i] / mass_s;
        part += p * (lse - zr[cols[i]]);
        atomicAdd(&dr[cols[i]], -wr * p);
    }
    part = kf_block_sum(part, red);
    if (tid == 0) {
        losses[r] = part;
        atomicAdd(loss, wr * part);
        atomicAdd(counted, 1);
    }
}

// Acceptance and rejection cost of every row's drafted position against its target, the mass the draft and the
// target share over the candidates, and the target's mass the draft misses off the end column over the target's
// mass off it, zero and zero for a row without a target
extern "C" __global__ void __launch_bounds__(256) edr_terms(const float* __restrict__ z, int V, const float* __restrict__ cands, int k, const int* __restrict__ lut, int eos, float* __restrict__ a_out, float* __restrict__ c_out) {
    __shared__ float red[32];
    __shared__ int cols[KF_SOFT_K];
    __shared__ float ps[KF_SOFT_K];
    __shared__ float mass_s;
    const int r = blockIdx.x, tid = threadIdx.x;
    const float* zr = z + (i64)r * V;
    const float mass = kf_soft_target(cands + (i64)r * (2 * k + 1), k, lut, cols, ps, red);
    if (tid == 0) mass_s = mass;
    __syncthreads();
    if (mass_s <= 0.0f) {
        if (tid == 0) {
            a_out[r] = 0.0f;
            c_out[r] = 0.0f;
        }
        return;
    }
    const float lse = kf_row_lse(zr, V, red);
    float a = 0.0f, miss = 0.0f, p_eos = 0.0f;
    for (int i = tid; i < k; i += blockDim.x) {
        if (cols[i] < 0) continue;
        const float p = ps[i] / mass_s, q = __expf(zr[cols[i]] - lse);
        a += fminf(p, q);
        if (cols[i] == eos) p_eos += p; else miss += fmaxf(p - q, 0.0f);
    }
    a = kf_block_sum(a, red);
    miss = kf_block_sum(miss, red);
    p_eos = kf_block_sum(p_eos, red);
    if (tid == 0) {
        a_out[r] = a;
        c_out[r] = miss / fmaxf(1.0f - p_eos, 1e-6f);
    }
}

// Gradient of every row's share of the expected rounds into its logits, scaled by w, the occupancy times the
// rejection cost's slope plus the rounds saved by accepting times the acceptance's slope, through the softmax
extern "C" __global__ void __launch_bounds__(256) edr_grad(const float* __restrict__ z, int V, const float* __restrict__ cands, int k, const int* __restrict__ lut, int eos, const float* __restrict__ omega, const float* __restrict__ dv, const float* __restrict__ w, float* __restrict__ dz) {
    __shared__ float red[32];
    __shared__ int cols[KF_SOFT_K];
    __shared__ float ps[KF_SOFT_K];
    __shared__ float gs[KF_SOFT_K];
    __shared__ float mass_s, dot_s;
    const int r = blockIdx.x, tid = threadIdx.x;
    const float* zr = z + (i64)r * V;
    float* dr = dz + (i64)r * V;
    const float mass = kf_soft_target(cands + (i64)r * (2 * k + 1), k, lut, cols, ps, red);
    if (tid == 0) mass_s = mass;
    __syncthreads();
    if (mass_s <= 0.0f || w[r] == 0.0f) return;
    const float lse = kf_row_lse(zr, V, red);
    float p_eos = 0.0f;
    for (int i = tid; i < k; i += blockDim.x) if (cols[i] >= 0 && cols[i] == eos) p_eos += ps[i] / mass_s;
    p_eos = kf_block_sum(p_eos, red);
    const float norm = 1.0f / fmaxf(1.0f - p_eos, 1e-6f), om = omega[r], dvr = dv[r];
    float dot = 0.0f;
    for (int i = tid; i < k; i += blockDim.x) {
        float g = 0.0f;
        if (cols[i] >= 0) {
            const float p = ps[i] / mass_s, q = __expf(zr[cols[i]] - lse);
            if (q < p) {
                if (cols[i] != eos) g -= om * norm;
                g += om * dvr;
            }
            dot += q * g;
        }
        gs[i] = g;
    }
    dot = kf_block_sum(dot, red);
    if (tid == 0) dot_s = dot;
    __syncthreads();
    const float wr = w[r];
    for (int j = tid; j < V; j += blockDim.x) dr[j] = fmaf(-wr * __expf(zr[j] - lse), dot_s, dr[j]);
    __syncthreads();
    for (int i = tid; i < k; i += blockDim.x) {
        if (cols[i] < 0 || gs[i] == 0.0f) continue;
        atomicAdd(&dr[cols[i]], wr * __expf(zr[cols[i]] - lse) * gs[i]);
    }
}

// The vocabulary integrated block verification loss of one row's drafted positions along the model's own path,
// one block per row, the positions' logits lie step_stride floats apart from z plus the row, their candidates
// cand_stride floats apart, truth holds every position's path column or minus one at the same stride in rows, and
// the gradient times w adds into dz. The loss, log depth minus log the expected accepted length, lands in losses.
extern "C" __global__ void __launch_bounds__(256) bv_row(const float* __restrict__ z, int V, i64 step_stride, const float* __restrict__ cands, int k, i64 cand_stride, const int* __restrict__ truth, int truth_stride, const int* __restrict__ lut, int depth, const float* __restrict__ w, float* __restrict__ dz, float* __restrict__ losses) {
    __shared__ float red[32];
    __shared__ int cols[KF_SOFT_K];
    __shared__ float ps[KF_SOFT_K];
    __shared__ float lses[KF_BV_DEPTH];
    __shared__ float masses[KF_BV_DEPTH];
    __shared__ float rr[KF_BV_DEPTH + 1], alpha[KF_BV_DEPTH + 1], sq[KF_BV_DEPTH + 1], sp[KF_BV_DEPTH + 1], bar[KF_BV_DEPTH + 1];
    __shared__ int argmin[KF_BV_DEPTH + 1], ended;
    __shared__ float e_s, dot_s;
    const int r = blockIdx.x, tid = threadIdx.x;
    if (tid == 0) {
        rr[0] = 1.0f;
        alpha[0] = 1.0f;
        argmin[0] = 0;
        sq[0] = 0.0f;
        sp[0] = 0.0f;
        bar[0] = 0.0f;
        ended = depth + 1;
    }
    __syncthreads();
    // Forward over the positions, the score, its sums, and the path ratio and retention after it
    for (int i = 1; i <= depth; i++) {
        if (i >= ended) break;
        const float* zr = z + (i64)(i - 1) * step_stride + (i64)r * V;
        const float* c = cands + (i64)(i - 1) * cand_stride + (i64)r * (2 * k + 1);
        const float mass = kf_soft_target(c, k, lut, cols, ps, red);
        if (tid == 0) masses[i - 1] = mass;
        __syncthreads();
        if (mass <= 0.0f) {
            if (tid == 0) ended = i;
            __syncthreads();
            break;
        }
        const float lse = kf_row_lse(zr, V, red);
        if (tid == 0) lses[i - 1] = lse;
        float b = 0.0f, s_q = 0.0f, s_p = 0.0f;
        for (int j = tid; j < k; j += blockDim.x) {
            if (cols[j] < 0) continue;
            const float p = ps[j] / mass, q = __expf(zr[cols[j]] - lse);
            const float qa = alpha[i - 1] * p, qr = rr[i - 1] * q;
            if (qr < qa) { b += qr; s_q += q; } else { b += qa; s_p += p; }
        }
        b = kf_block_sum(b, red);
        s_q = kf_block_sum(s_q, red);
        s_p = kf_block_sum(s_p, red);
        if (tid == 0) {
            bar[i] = b;
            sq[i] = s_q;
            sp[i] = s_p;
            const int y = truth[(i64)(i - 1) * truth_stride + r];
            if (y < 0) {
                rr[i] = 0.0f;
                alpha[i] = 0.0f;
                argmin[i] = i;
            } else {
                float py = 0.0f;
                for (int j = 0; j < k; j++) if (cols[j] == y) py += ps[j] / mass;
                py = fmaxf(py, 1e-6f);
                const float qy = fmaxf(__expf(zr[y] - lse), 1e-12f);
                rr[i] = rr[i - 1] * qy / py;
                if (rr[i] < alpha[i - 1]) { alpha[i] = rr[i]; argmin[i] = i; } else { alpha[i] = alpha[i - 1]; argmin[i] = argmin[i - 1]; }
            }
        }
        __syncthreads();
    }
    const int last = ended - 1;
    float e = 0.0f;
    for (int i = 1; i <= last; i++) e += bar[i];
    e = fmaxf(e, 1e-12f);
    if (tid == 0) {
        e_s = e;
        losses[r] = __logf((float)depth) - __logf(e);
    }
    __syncthreads();
    const float scale = -w[r] / e_s;
    if (w[r] == 0.0f) return;
    // Backward over the positions, h over the candidates and the path column, then the softmax's pull
    for (int i = 1; i <= last; i++) {
        const float* zr = z + (i64)(i - 1) * step_stride + (i64)r * V;
        float* dr = dz + (i64)(i - 1) * step_stride + (i64)r * V;
        const float* c = cands + (i64)(i - 1) * cand_stride + (i64)r * (2 * k + 1);
        kf_soft_target(c, k, lut, cols, ps, red);
        __syncthreads();
        const float mass = masses[i - 1], lse = lses[i - 1];
        const int y = truth[(i64)(i - 1) * truth_stride + r];
        float gain = 0.0f;
        if (y >= 0) {
            for (int kk = i + 1; kk <= last; kk++) {
                gain += sq[kk] * rr[kk - 1];
                if (alpha[kk - 1] < 1.0f && i <= argmin[kk - 1]) gain += sp[kk] * alpha[kk - 1];
            }
        }
        float dot = 0.0f;
        for (int j = tid; j < k; j += blockDim.x) {
            if (cols[j] < 0) continue;
            const float p = ps[j] / mass, q = __expf(zr[cols[j]] - lse);
            if (rr[i - 1] * q < alpha[i - 1] * p) dot += rr[i - 1] * q;
        }
        dot = kf_block_sum(dot, red);
        if (tid == 0) dot_s = dot + gain;
        __syncthreads();
        for (int u = tid; u < V; u += blockDim.x) dr[u] = fmaf(-scale * __expf(zr[u] - lse), dot_s, dr[u]);
        __syncthreads();
        for (int j = tid; j < k; j += blockDim.x) {
            if (cols[j] < 0) continue;
            const float p = ps[j] / mass, q = __expf(zr[cols[j]] - lse);
            if (rr[i - 1] * q < alpha[i - 1] * p) atomicAdd(&dr[cols[j]], scale * rr[i - 1] * q);
        }
        if (tid == 0 && y >= 0) atomicAdd(&dr[y], scale * gain);
        __syncthreads();
    }
}

// One optimistic AdamW step at step t, p holds the played point, a step of the last direction u ahead of the
// trained point, so the update recovers the trained point, moves it, and plays the new direction ahead
extern "C" __global__ void adamw_opt(float* p, const float* g, float* m, float* v, float* u, i64 n, float lr, float b1, float b2, float eps, float wd, int t) {
    const float c1 = 1.0f - powf(b1, (float)t), c2 = 1.0f - powf(b2, (float)t);
    for (i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x; i < n; i += (i64)gridDim.x * blockDim.x) {
        const float gi = g[i];
        const float mi = m[i] = b1 * m[i] + (1.0f - b1) * gi;
        const float vi = v[i] = b2 * v[i] + (1.0f - b2) * gi * gi;
        const float un = (mi / c1) / (sqrtf(vi / c2) + eps);
        const float w = p[i] + lr * u[i];
        p[i] = w - lr * (un + wd * w) - lr * un;
        u[i] = un;
    }
}

// out = w0 a + w1 b + w2 c + w3 d, the weighted mixture of up to four parameter buffers, a null pointer weighing nothing
extern "C" __global__ void mix4(float* out, const float* a, const float* b, const float* c, const float* d, float w0, float w1, float w2, float w3, i64 n) {
    for (i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x; i < n; i += (i64)gridDim.x * blockDim.x) {
        float s = w0 * a[i];
        if (b) s = fmaf(w1, b[i], s);
        if (c) s = fmaf(w2, c[i], s);
        if (d) s = fmaf(w3, d[i], s);
        out[i] = s;
    }
}

// y = a y + (1 - a) x, the moving average the reference keeps
extern "C" __global__ void ema(float* y, const float* x, i64 n, float a) {
    for (i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x; i < n; i += (i64)gridDim.x * blockDim.x) y[i] = fmaf(a, y[i], (1.0f - a) * x[i]);
}

// The columns of a row major k by n matrix as n rows of q8_0 blocks, thread per column and block of 32, the block's
// scale the largest magnitude over 127 as f16, its integers the values over that scale rounded to nearest even
extern "C" __global__ void quant_q8_0_t(const float* __restrict__ src, u8* __restrict__ dst, int k, int n) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x, b = blockIdx.y;
    if (r >= n) return;
    float vals[32];
    float amax = 0.0f;
#pragma unroll
    for (int j = 0; j < 32; j++) {
        vals[j] = src[(i64)(b * 32 + j) * n + r];
        amax = fmaxf(amax, fabsf(vals[j]));
    }
    const float d = amax / 127.0f, id = d != 0.0f ? 1.0f / d : 0.0f;
    u8* out = dst + (i64)r * (k / 32) * 34 + (i64)b * 34;
    const u16 dh = kf_f2h(d);
    out[0] = (u8)(dh & 0xff);
    out[1] = (u8)(dh >> 8);
#pragma unroll
    for (int j = 0; j < 32; j++) {
        int q = __float2int_rn(vals[j] * id);
        q = max(-127, min(127, q));
        out[2 + j] = (u8)(i8)q;
    }
}

// Predictive entropy of every row of V logits, the log sum exp less the mean logit under the softmax
extern "C" __global__ void __launch_bounds__(256) row_entropy(const float* __restrict__ z, int V, float* __restrict__ out) {
    __shared__ float red[32];
    const float* zr = z + (i64)blockIdx.x * V;
    float m = KF_NEG_INF;
    for (int j = threadIdx.x; j < V; j += blockDim.x) m = fmaxf(m, zr[j]);
    m = kf_block_max(m, red);
    float l = 0.0f, lz = 0.0f;
    for (int j = threadIdx.x; j < V; j += blockDim.x) {
        const float e = __expf(zr[j] - m);
        l += e;
        lz = fmaf(e, zr[j], lz);
    }
    l = kf_block_sum(l, red);
    lz = kf_block_sum(lz, red);
    if (threadIdx.x == 0) out[blockIdx.x] = m + __logf(l) - lz / l;
}

// Adds n_sel rows of src, those idx names in order, into the block rows of z starting at slot off modulo block
extern "C" __global__ void cal_accum(const float* __restrict__ src, const int* __restrict__ idx, int n_sel, int off, int block, float* __restrict__ z, int d) {
    for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < d; c += gridDim.x * blockDim.x) {
        for (int i = 0; i < n_sel; i++) {
            const int slot = (off + i) % block;
            z[(i64)slot * d + c] += src[(i64)idx[i] * d + c];
        }
    }
}

// out[c] += the sum over rows of x[r][c]
extern "C" __global__ void col_sum(const float* __restrict__ x, int rows, int cols, float* __restrict__ out) {
    for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < cols; c += gridDim.x * blockDim.x) {
        float s = 0.0f;
        for (int r = 0; r < rows; r++) s += x[(i64)r * cols + c];
        out[c] += s;
    }
}

// dst[r][off + c] = src[r][c] for rows of cols, dst rows stride floats apart
extern "C" __global__ void copy_cols(const float* __restrict__ src, float* __restrict__ dst, int rows, int cols, i64 stride, int off) {
    for (i64 t = blockIdx.x * (i64)blockDim.x + threadIdx.x; t < (i64)rows * cols; t += (i64)gridDim.x * blockDim.x) {
        const int r = (int)(t / cols), c = (int)(t - (i64)r * cols);
        dst[(i64)r * stride + off + c] = src[t];
    }
}

// Rows of the step zero keys and values of drafted rows copied into a slot's position ring at their positions
extern "C" __global__ void ring_write(const float* __restrict__ k, const float* __restrict__ v, const int* __restrict__ rows, const int* __restrict__ pos, int n, int cap, int D, float* __restrict__ kring, float* __restrict__ vring, int* __restrict__ hpos) {
    for (i64 t = blockIdx.x * (i64)blockDim.x + threadIdx.x; t < (i64)n * D; t += (i64)gridDim.x * blockDim.x) {
        const int i = (int)(t / D), d = (int)(t - (i64)i * D);
        const int slot = pos[i] % cap;
        kring[(i64)slot * D + d] = k[(i64)rows[i] * D + d];
        vring[(i64)slot * D + d] = v[(i64)rows[i] * D + d];
        if (d == 0) hpos[slot] = pos[i];
    }
}

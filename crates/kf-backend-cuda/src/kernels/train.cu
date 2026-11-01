#include "common.cuh"

// The kernels a draft head learns with: a single precision product with either operand transposed, the gated
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

// Smooth L1 between pred and target over n values, the mean loss added into loss and the gradient of the mean into grad
extern "C" __global__ void smooth_l1(const float* pred, const float* target, float* grad, float* loss, i64 n, float beta) {
    __shared__ float red[32];
    float s = 0.0f;
    const float inv_n = 1.0f / (float)n;
    for (i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x; i < n; i += (i64)gridDim.x * blockDim.x) {
        const float d = pred[i] - target[i];
        const float a = fabsf(d);
        s += a < beta ? 0.5f * d * d / beta : a - 0.5f * beta;
        grad[i] = (a < beta ? d / beta : (d > 0.0f ? 1.0f : -1.0f)) * inv_n;
    }
    s = kf_block_sum(s, red);
    if (threadIdx.x == 0) atomicAdd(loss, s * inv_n);
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

// Attention of a draft head over chunks of T positions, one block per sequence and head, every query seeing the keys
// at or before it within the window. Keys and values of the chunk sit in shared memory, the scores and probabilities
// of the chunk in registers and shared memory, the probabilities written out for the backward pass.
#define KF_TA_T 64
#define KF_TA_THREADS 256

// q k v [B, T, H hd] row major with the heads side by side, out the same, probs [B, H, T, T]
extern "C" __global__ void __launch_bounds__(KF_TA_THREADS) head_attn_fwd(const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v, float* __restrict__ out, float* __restrict__ probs, int T, int H, int hd, int window, float scale) {
    extern __shared__ float sm[];
    float* ks = sm;
    float* vs = ks + KF_TA_T * hd;
    float* ps = vs + KF_TA_T * hd;
    const int b = blockIdx.x / H, h = blockIdx.x - b * H;
    const int tid = threadIdx.x, D = H * hd;
    const i64 base = (i64)b * T * D + (i64)h * hd;
    for (int i = tid; i < T * hd; i += KF_TA_THREADS) {
        const int t = i / hd, d = i - t * hd;
        ks[t * hd + d] = k[base + (i64)t * D + d];
        vs[t * hd + d] = v[base + (i64)t * D + d];
    }
    __syncthreads();
    // Scores, each thread a run of (i, j) pairs
    for (int e = tid; e < T * T; e += KF_TA_THREADS) {
        const int i = e / T, j = e - i * T;
        float s = KF_NEG_INF;
        if (j <= i && j > i - window) {
            s = 0.0f;
            const float* qi = q + base + (i64)i * D;
            for (int d = 0; d < hd; d++) s = fmaf(qi[d], ks[j * hd + d], s);
            s *= scale;
        }
        ps[e] = s;
    }
    __syncthreads();
    // Softmax of every row, four threads a row
    for (int i = tid / 4; i < T; i += KF_TA_THREADS / 4) {
        const int part = tid & 3;
        float m = KF_NEG_INF;
        for (int j = part; j < T; j += 4) m = fmaxf(m, ps[i * T + j]);
#pragma unroll
        for (int o = 1; o < 4; o <<= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        float l = 0.0f;
        for (int j = part; j < T; j += 4) {
            const float p = ps[i * T + j] == KF_NEG_INF ? 0.0f : __expf(ps[i * T + j] - m);
            ps[i * T + j] = p;
            l += p;
        }
#pragma unroll
        for (int o = 1; o < 4; o <<= 1) l += __shfl_xor_sync(0xffffffffu, l, o);
        const float inv = 1.0f / l;
        for (int j = part; j < T; j += 4) ps[i * T + j] *= inv;
    }
    __syncthreads();
    float* pout = probs + ((i64)b * H + h) * T * T;
    for (int e = tid; e < T * T; e += KF_TA_THREADS) pout[e] = ps[e];
    // Output rows
    for (int e = tid; e < T * hd; e += KF_TA_THREADS) {
        const int i = e / hd, d = e - i * hd;
        float o = 0.0f;
        for (int j = 0; j <= i; j++) o = fmaf(ps[i * T + j], vs[j * hd + d], o);
        out[base + (i64)i * D + d] = o;
    }
}

// Gradients of head_attn_fwd given dout and the saved probabilities: dv = P^T do, ds = P (dp - rowsum(dp P)) with
// dp = do v^T, dq = ds k scale, dk = ds^T q scale
extern "C" __global__ void __launch_bounds__(KF_TA_THREADS) head_attn_bwd(const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v, const float* __restrict__ probs, const float* __restrict__ dout, float* __restrict__ dq, float* __restrict__ dk, float* __restrict__ dv, int T, int H, int hd, float scale) {
    extern __shared__ float sm[];
    float* as = sm;
    float* bs = as + KF_TA_T * hd;
    float* ps = bs + KF_TA_T * hd;
    const int b = blockIdx.x / H, h = blockIdx.x - b * H;
    const int tid = threadIdx.x, D = H * hd;
    const i64 base = (i64)b * T * D + (i64)h * hd;
    const float* pin = probs + ((i64)b * H + h) * T * T;
    // Phase one, v and do in shared memory: dv = P^T do and dp = do v^T, dp overwriting ps as ds
    for (int i = tid; i < T * hd; i += KF_TA_THREADS) {
        const int t = i / hd, d = i - t * hd;
        as[t * hd + d] = v[base + (i64)t * D + d];
        bs[t * hd + d] = dout[base + (i64)t * D + d];
    }
    for (int e = tid; e < T * T; e += KF_TA_THREADS) ps[e] = pin[e];
    __syncthreads();
    for (int e = tid; e < T * hd; e += KF_TA_THREADS) {
        const int j = e / hd, d = e - j * hd;
        float s = 0.0f;
        for (int i = j; i < T; i++) s = fmaf(ps[i * T + j], bs[i * hd + d], s);
        dv[base + (i64)j * D + d] = s;
    }
    __syncthreads();
    // ds rows, four threads a row: dp(i, j) = do_i . v_j, then ds = P (dp - sum_j P dp)
    for (int i = tid / 4; i < T; i += KF_TA_THREADS / 4) {
        const int part = tid & 3;
        float dot = 0.0f;
        for (int j = part; j <= i; j += 4) {
            float dp = 0.0f;
            for (int d = 0; d < hd; d++) dp = fmaf(bs[i * hd + d], as[j * hd + d], dp);
            dot += ps[i * T + j] * dp;
            // The products of this row are needed again below, kept in the upper half of the padded row space
            ps[i * T + j] = ps[i * T + j] * dp;
        }
#pragma unroll
        for (int o = 1; o < 4; o <<= 1) dot += __shfl_xor_sync(0xffffffffu, dot, o);
        for (int j = part; j < T; j += 4) {
            if (j <= i) {
                const float p = pin[i * T + j];
                ps[i * T + j] = ps[i * T + j] - p * dot;
            } else {
                ps[i * T + j] = 0.0f;
            }
        }
    }
    __syncthreads();
    // Phase two, k and q in shared memory: dq = ds k scale, dk = ds^T q scale
    for (int i = tid; i < T * hd; i += KF_TA_THREADS) {
        const int t = i / hd, d = i - t * hd;
        as[t * hd + d] = k[base + (i64)t * D + d];
        bs[t * hd + d] = q[base + (i64)t * D + d];
    }
    __syncthreads();
    for (int e = tid; e < T * hd; e += KF_TA_THREADS) {
        const int i = e / hd, d = e - i * hd;
        float s = 0.0f;
        for (int j = 0; j <= i; j++) s = fmaf(ps[i * T + j], as[j * hd + d], s);
        dq[base + (i64)i * D + d] = s * scale;
        float t = 0.0f;
        for (int ii = i; ii < T; ii++) t = fmaf(ps[ii * T + i], bs[ii * hd + d], t);
        dk[base + (i64)i * D + d] = t * scale;
    }
}

// Attention of m draft rows, one block per row and head, each row seeing its slot's window of nwin cached keys and
// values, the keys and values of its ancestors among the draft rows, listed per row as up to KF_TD_ANC indices into
// kanc and vanc with minus one ending the list, and itself, whose key and value stand in kanc and vanc at its own index
#define KF_TD_ANC 8
extern "C" __global__ void __launch_bounds__(256) head_attn_draft(const float* __restrict__ q, const float* __restrict__ kwin, const float* __restrict__ vwin, int nwin, const float* __restrict__ kanc, const float* __restrict__ vanc, const int* __restrict__ anc, float* __restrict__ out, int H, int hd, float scale) {
    extern __shared__ float sm[];
    float* qs = sm;
    float* sc = qs + hd;
    const int r = blockIdx.x / H, h = blockIdx.x - r * H, tid = threadIdx.x, D = H * hd;
    for (int d = tid; d < hd; d += blockDim.x) qs[d] = q[(i64)r * D + h * hd + d];
    __syncthreads();
    // Keys: the window first, then the ancestors and the row itself
    int nanc = 0;
    while (nanc < KF_TD_ANC && anc[r * KF_TD_ANC + nanc] >= 0) nanc++;
    const int nk = nwin + nanc;
    for (int j = tid; j < nk; j += blockDim.x) {
        const float* kj = j < nwin ? kwin + (i64)j * D + h * hd : kanc + (i64)anc[r * KF_TD_ANC + (j - nwin)] * D + h * hd;
        float s = 0.0f;
        for (int d = 0; d < hd; d++) s = fmaf(qs[d], kj[d], s);
        sc[j] = s * scale;
    }
    __syncthreads();
    __shared__ float red[32];
    float m = KF_NEG_INF;
    for (int j = tid; j < nk; j += blockDim.x) m = fmaxf(m, sc[j]);
    m = kf_block_max(m, red);
    float l = 0.0f;
    for (int j = tid; j < nk; j += blockDim.x) {
        const float p = __expf(sc[j] - m);
        sc[j] = p;
        l += p;
    }
    l = kf_block_sum(l, red);
    const float inv = 1.0f / l;
    for (int d = tid; d < hd; d += blockDim.x) {
        float o = 0.0f;
        for (int j = 0; j < nk; j++) {
            const float* vj = j < nwin ? vwin + (i64)j * D + h * hd : vanc + (i64)anc[r * KF_TD_ANC + (j - nwin)] * D + h * hd;
            o = fmaf(sc[j], vj[d], o);
        }
        out[(i64)r * D + h * hd + d] = o * inv;
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

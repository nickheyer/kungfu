#include "common.cuh"

#define KF_GRID_STRIDE(i, n) for (i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x; i < (n); i += (i64)gridDim.x * blockDim.x)

// Token or position ids as floats, the value an input node holds
extern "C" __global__ void input_f32(const int* src, float* out, int n) {
    KF_GRID_STRIDE(i, n) out[i] = (float)(u32)src[i];
}

extern "C" __global__ void add(const float* a, const float* b, float* out, int n) {
    KF_GRID_STRIDE(i, n) out[i] = a[i] + b[i];
}

extern "C" __global__ void mul(const float* a, const float* b, float* out, int n) {
    KF_GRID_STRIDE(i, n) out[i] = a[i] * b[i];
}

extern "C" __global__ void scale(const float* a, float s, float* out, int n) {
    KF_GRID_STRIDE(i, n) out[i] = a[i] * s;
}

extern "C" __global__ void silu(const float* a, float* out, int n) {
    KF_GRID_STRIDE(i, n) out[i] = kf_act(a[i], 0);
}

extern "C" __global__ void gelu(const float* a, float* out, int n) {
    KF_GRID_STRIDE(i, n) out[i] = kf_act(a[i], 1);
}

extern "C" __global__ void relu(const float* a, float* out, int n) {
    KF_GRID_STRIDE(i, n) out[i] = kf_act(a[i], 2);
}

extern "C" __global__ void sigmoid(const float* a, float* out, int n) {
    KF_GRID_STRIDE(i, n) out[i] = 1.0f / (1.0f + expf(-a[i]));
}

extern "C" __global__ void softcap(const float* a, float c, float* out, int n) {
    KF_GRID_STRIDE(i, n) out[i] = c * tanhf(a[i] / c);
}

extern "C" __global__ void gated_act(const float* gate, const float* up, float* out, int n, int act) {
    KF_GRID_STRIDE(i, n) out[i] = kf_act(gate[i], act) * up[i];
}

// Activation of the first half of each row times its second half
extern "C" __global__ void gated_act_fused(const float* x, float* out, int rows, int cols, int half, int act) {
    KF_GRID_STRIDE(i, (i64)rows * half) {
        i64 r = i / half, c = i - r * half;
        const float* xr = x + r * cols;
        out[i] = kf_act(xr[c], act) * xr[half + c];
    }
}

// Row vector multiply add or divide, a single element weight broadcast to every column
extern "C" __global__ void row_op(const float* x, const float* w, float* out, int rows, int cols, int wn, int op) {
    KF_GRID_STRIDE(i, (i64)rows * cols) {
        float v = w[wn == 1 ? 0 : i % cols];
        out[i] = op == 0 ? x[i] * v : (op == 1 ? x[i] + v : x[i] / v);
    }
}

// Binary op with the second operand broadcast over columns, rows, or as one scalar
extern "C" __global__ void binary(const float* a, const float* b, float* out, int rows, int cols, int b_rows, int b_cols, int op) {
    KF_GRID_STRIDE(i, (i64)rows * cols) {
        i64 r = i / cols, c = i - r * cols;
        i64 j = (b_rows == rows && b_cols == cols) ? i : ((b_rows * b_cols == 1) ? 0 : (b_cols == 1 ? r : c));
        float x = a[i], y = b[j];
        out[i] = op == 0 ? x + y : (op == 1 ? x - y : (op == 2 ? x * y : x / y));
    }
}

// Unary op by code with two parameters, tanh exp sqrt sqr abs sign log round softplus clamp scale_bias gelu_erf relu_sqr
extern "C" __global__ void unary(const float* a, float* out, int n, int op, float p0, float p1) {
    KF_GRID_STRIDE(i, n) {
        float x = a[i], y;
        switch (op) {
            case 0: y = tanhf(x); break;
            case 1: y = expf(x); break;
            case 2: y = sqrtf(x); break;
            case 3: y = x * x; break;
            case 4: y = fabsf(x); break;
            case 5: y = x > 0.0f ? 1.0f : (x < 0.0f ? -1.0f : 0.0f); break;
            case 6: y = logf(x); break;
            case 7: y = rintf(x); break;
            case 8: y = p0 * x > 20.0f ? x : log1pf(expf(p0 * x)) / p0; break;
            case 9: y = fminf(fmaxf(x, p0), p1); break;
            case 10: y = x * p0 + p1; break;
            case 11: y = kf_act(x, 3); break;
            case 12: y = kf_act(x, 4); break;
            default: y = x / (1.0f + expf(-1.702f * x)); break;
        }
        out[i] = y;
    }
}

extern "C" __global__ void concat(const float* a, const float* b, float* out, int rows, int ca, int cb) {
    int cols = ca + cb;
    KF_GRID_STRIDE(i, (i64)rows * cols) {
        i64 r = i / cols, c = i - r * cols;
        out[i] = c < ca ? a[r * ca + c] : b[r * cb + (c - ca)];
    }
}

extern "C" __global__ void tile(const float* x, float* out, int rows, int cols, int n) {
    KF_GRID_STRIDE(i, (i64)rows * cols * n) {
        i64 r = i / ((i64)cols * n), c = i - r * cols * n;
        out[i] = x[r * cols + c % cols];
    }
}

extern "C" __global__ void pad_cols(const float* x, float* out, int rows, int cols_in, int cols_out) {
    KF_GRID_STRIDE(i, (i64)rows * cols_out) {
        i64 r = i / cols_out, c = i - r * cols_out;
        out[i] = c < cols_in ? x[r * cols_in + c] : 0.0f;
    }
}

extern "C" __global__ void mask_cols(const float* x, float* out, int rows, int cols, int start, int len, float value) {
    KF_GRID_STRIDE(i, (i64)rows * cols) {
        i64 c = i % cols;
        out[i] = (c >= start && c < start + len) ? value : x[i];
    }
}

extern "C" __global__ void cols(const float* x, float* out, int rows, int cols_in, int start, int len) {
    KF_GRID_STRIDE(i, (i64)rows * len) {
        i64 r = i / len, c = i - r * len;
        out[i] = x[r * cols_in + start + c];
    }
}

// Activations rounded the way the reference quantizes them before a product against the weight type
extern "C" __global__ void round_act(const float* x, float* out, int n, int kind) {
    __shared__ float sa[256];
    __shared__ int si[256];
    int tid = threadIdx.x, c0 = blockIdx.x * 256, c = c0 + tid;
    const float* xr = x + (i64)blockIdx.y * n;
    float* orow = out + (i64)blockIdx.y * n;
    float v = c < n ? xr[c] : 0.0f;
    if (kind == 1) {
        if (c < n) orow[c] = kf_round_f16(v);
        return;
    }
    if (kind == 2) {
        if (c < n) orow[c] = kf_round_bf16(v);
        return;
    }
    if (kind == 3) {
        float amax = kf_warp_max(fabsf(v));
        float d = amax / 127.0f;
        float id = d != 0.0f ? 1.0f / d : 0.0f;
        float dh = kf_round_f16(d);
        if (c < n) orow[c] = dh * roundf(v * id);
        return;
    }
    sa[tid] = c < n ? fabsf(v) : -1.0f;
    si[tid] = tid;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            float oa = sa[tid + s];
            int oi = si[tid + s];
            if (oa > sa[tid] || (oa == sa[tid] && oi < si[tid])) {
                sa[tid] = oa;
                si[tid] = oi;
            }
        }
        __syncthreads();
    }
    if (c >= n) return;
    if (sa[0] <= 0.0f) {
        orow[c] = 0.0f;
        return;
    }
    float mx = xr[c0 + si[0]];
    float iscale = -127.0f / mx;
    float d = 1.0f / iscale;
    orow[c] = d * fminf(rintf(iscale * v), 127.0f);
}

// Mean of every kernel by kernel block of a side by side grid of rows, raster order in and out
extern "C" __global__ void pool_grid(const float* x, float* out, int side, int kernel, int cols) {
    int o = side / kernel;
    KF_GRID_STRIDE(i, (i64)o * o * cols) {
        i64 orow = i / cols, c = i - orow * cols;
        int by = (int)(orow / o), bx = (int)(orow % o);
        float acc = 0.0f;
        for (int dy = 0; dy < kernel; dy++) {
            for (int dx = 0; dx < kernel; dx++) {
                acc += x[((i64)(by * kernel + dy) * side + bx * kernel + dx) * cols + c];
            }
        }
        out[i] = acc / (float)(kernel * kernel);
    }
}

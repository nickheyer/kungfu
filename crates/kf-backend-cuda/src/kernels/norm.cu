#include "common.cuh"

// Root mean square normalization per row with an optional elementwise weight
extern "C" __global__ void rms_norm(const float* x, const float* w, const float* b, float* out, int cols, float eps) {
    __shared__ float scratch[32];
    const float* xr = x + (i64)blockIdx.x * cols;
    float* orow = out + (i64)blockIdx.x * cols;
    float s = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) s += xr[i] * xr[i];
    s = kf_block_sum(s, scratch);
    float inv = 1.0f / sqrtf(s / (float)cols + eps);
    for (int i = threadIdx.x; i < cols; i += blockDim.x) orow[i] = xr[i] * (inv * (w ? w[i] : 1.0f)) + (b ? b[i] : 0.0f);
}

// Each row over the square root of its squared sum plus eps
extern "C" __global__ void l2_norm(const float* x, float* out, int cols, float eps) {
    __shared__ float scratch[32];
    const float* xr = x + (i64)blockIdx.x * cols;
    float* orow = out + (i64)blockIdx.x * cols;
    float s = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) s += xr[i] * xr[i];
    s = kf_block_sum(s, scratch);
    float inv = 1.0f / sqrtf(s + eps);
    for (int i = threadIdx.x; i < cols; i += blockDim.x) orow[i] = xr[i] * inv;
}

extern "C" __global__ void sum_rows(const float* x, float* out, int cols) {
    __shared__ float scratch[32];
    const float* xr = x + (i64)blockIdx.x * cols;
    float s = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) s += xr[i];
    s = kf_block_sum(s, scratch);
    if (threadIdx.x == 0) out[blockIdx.x] = s;
}

extern "C" __global__ void layer_norm(const float* x, const float* w, const float* b, float* out, int cols, float eps) {
    __shared__ float scratch[32];
    const float* xr = x + (i64)blockIdx.x * cols;
    float* orow = out + (i64)blockIdx.x * cols;
    float s = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) s += xr[i];
    float mean = kf_block_sum(s, scratch) / (float)cols;
    float v = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) v += (xr[i] - mean) * (xr[i] - mean);
    float inv = 1.0f / sqrtf(kf_block_sum(v, scratch) / (float)cols + eps);
    for (int i = threadIdx.x; i < cols; i += blockDim.x) orow[i] = (xr[i] - mean) * inv * (w ? w[i] : 1.0f) + (b ? b[i] : 0.0f);
}

extern "C" __global__ void softmax(const float* x, float* out, int cols) {
    __shared__ float scratch[32];
    const float* xr = x + (i64)blockIdx.x * cols;
    float* orow = out + (i64)blockIdx.x * cols;
    float m = KF_NEG_INF;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) m = fmaxf(m, xr[i]);
    m = kf_block_max(m, scratch);
    float s = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float e = expf(xr[i] - m);
        orow[i] = e;
        s += e;
    }
    s = kf_block_sum(s, scratch);
    for (int i = threadIdx.x; i < cols; i += blockDim.x) orow[i] = orow[i] / s;
}

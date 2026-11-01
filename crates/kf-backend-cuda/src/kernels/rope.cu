#include "common.cuh"

// The cosine and sine of every pair angle at every position, row p holding the half pairs of position p
extern "C" __global__ void rope_angles(const float* freq, float2* cst, int half, int n_pos) {
    i64 n = (i64)n_pos * half;
    for (i64 t = blockIdx.x * (i64)blockDim.x + threadIdx.x; t < n; t += (i64)gridDim.x * blockDim.x) {
        int i = (int)(t % half), p = (int)(t / half);
        float s, c;
        sincosf(freq[i] * (float)(u32)p, &s, &c);
        cst[t] = make_float2(c, s);
    }
}

// Rotates each pair by its frequency times the position of its stream, neox pairs halves and normal pairs neighbours,
// the pairs starting at offset inside the head and the sine carrying sign, minus one for the inverse rotation
extern "C" __global__ void rope(const float* x, float* out, const int* rope4, const float* freq, int rows, int n_head, int head_dim, int half, int neox, float mscale, int mode, int s0, int s1, int s2, int sect, int offset, float sign) {
    i64 n = (i64)rows * n_head * head_dim;
    for (i64 t = blockIdx.x * (i64)blockDim.x + threadIdx.x; t < n; t += (i64)gridDim.x * blockDim.x) {
        i64 rh = t / head_dim;
        int c = (int)(t - rh * head_dim) - offset;
        int r = (int)(rh / n_head);
        const float* head = x + rh * head_dim + offset;
        if (c < 0 || c >= 2 * half) {
            out[t] = head[c];
            continue;
        }
        int i = neox ? (c < half ? c : c - half) : c / 2;
        bool first = neox ? c < half : (c & 1) == 0;
        int a = neox ? i : 2 * i, b = neox ? i + half : 2 * i + 1;
        float theta = freq[i] * (float)(u32)rope4[4 * r + kf_rope_stream(i, mode, s0, s1, s2, sect)];
        float s, co;
        sincosf(theta, &s, &co);
        s *= sign;
        float x0 = head[a], x1 = head[b];
        out[t] = first ? (x0 * co - x1 * s) * mscale : (x0 * s + x1 * co) * mscale;
    }
}

#include "common.cuh"

// Rotates each pair by its frequency times the token position, neox pairs halves and normal pairs neighbours
extern "C" __global__ void rope(const float* x, float* out, const int* pos, const float* freq, int rows, int n_head, int head_dim, int half, int neox, float mscale) {
    i64 n = (i64)rows * n_head * head_dim;
    for (i64 t = blockIdx.x * (i64)blockDim.x + threadIdx.x; t < n; t += (i64)gridDim.x * blockDim.x) {
        i64 rh = t / head_dim;
        int c = (int)(t - rh * head_dim);
        int r = (int)(rh / n_head);
        const float* head = x + rh * head_dim;
        if (c >= 2 * half) {
            out[t] = head[c];
            continue;
        }
        int i = neox ? (c < half ? c : c - half) : c / 2;
        bool first = neox ? c < half : (c & 1) == 0;
        int a = neox ? i : 2 * i, b = neox ? i + half : 2 * i + 1;
        float theta = freq[i] * (float)(u32)pos[r];
        float s, co;
        sincosf(theta, &s, &co);
        float x0 = head[a], x1 = head[b];
        out[t] = first ? (x0 * co - x1 * s) * mscale : (x0 * s + x1 * co) * mscale;
    }
}

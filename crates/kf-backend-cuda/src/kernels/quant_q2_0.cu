#include "quant.cuh"

struct Q2_0 {
    static constexpr int QK = 64, BYTES = 18;
    static __device__ __forceinline__ void dq8(const u8* b, int g, float* v) {
        float d = kf_f16(b);
#pragma unroll
        for (int i = 0; i < 8; i++) {
            int j = g * 8 + i;
            v[i] = (float)((int)((b[2 + j / 4] >> ((j % 4) * 2)) & 3) - 1) * d;
        }
    }

    static __device__ __forceinline__ void q16(const u8* b, int g, i8* q, float& d, float& m) {
        d = kf_f16(b);
        m = 0.0f;
        const u8* p = b + 2 + 4 * g;
#pragma unroll
        for (int j = 0; j < 16; j++) q[j] = (i8)((int)((p[j >> 2] >> ((j & 3) * 2)) & 3) - 1);
    }
};

KF_QUANT_KERNELS_I8(Q2_0)

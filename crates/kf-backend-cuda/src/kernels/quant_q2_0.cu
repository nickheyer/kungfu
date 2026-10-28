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
};

KF_QUANT_KERNELS(Q2_0)

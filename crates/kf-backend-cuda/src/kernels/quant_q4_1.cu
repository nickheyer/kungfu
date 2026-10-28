#include "quant.cuh"

struct Q4_1 {
    static constexpr int QK = 32, BYTES = 20;
    static __device__ __forceinline__ void dq8(const u8* b, int g, float* v) {
        float d = kf_f16(b), m = kf_f16(b + 2);
        int j0 = (g & 1) * 8, sh = (g >> 1) * 4;
#pragma unroll
        for (int i = 0; i < 8; i++) v[i] = (float)((b[4 + j0 + i] >> sh) & 0xF) * d + m;
    }
};

KF_QUANT_KERNELS(Q4_1)

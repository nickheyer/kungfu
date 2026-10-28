#include "quant.cuh"

struct Tq2_0 {
    static constexpr int QK = 256, BYTES = 66;
    static __device__ __forceinline__ void dq8(const u8* b, int g, float* v) {
        float d = kf_f16(b + 64);
        int e0 = g * 8, c = e0 >> 7, l = (e0 & 127) >> 5, m0 = e0 & 31;
        const u8* q = b + 32 * c + m0;
#pragma unroll
        for (int i = 0; i < 8; i++) v[i] = (float)((int)((q[i] >> (2 * l)) & 3) - 1) * d;
    }
};

KF_QUANT_KERNELS(Tq2_0)

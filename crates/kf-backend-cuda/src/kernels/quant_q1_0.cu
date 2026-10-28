#include "quant.cuh"

struct Q1_0 {
    static constexpr int QK = 128, BYTES = 18;
    static __device__ __forceinline__ void dq8(const u8* b, int g, float* v) {
        float d = kf_f16(b);
        u8 bits = b[2 + g];
#pragma unroll
        for (int i = 0; i < 8; i++) v[i] = ((bits >> i) & 1) ? d : -d;
    }
};

KF_QUANT_KERNELS(Q1_0)

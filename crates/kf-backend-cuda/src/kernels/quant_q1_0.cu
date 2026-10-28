#include "quant.cuh"

struct Q1_0 {
    static constexpr int QK = 128, BYTES = 18;
    static __device__ __forceinline__ void dq8(const u8* b, int g, float* v) {
        float d = kf_f16(b);
        u8 bits = b[2 + g];
#pragma unroll
        for (int i = 0; i < 8; i++) v[i] = ((bits >> i) & 1) ? d : -d;
    }

    static __device__ __forceinline__ void q16(const u8* b, int g, i8* q, float& d, float& m) {
        d = kf_f16(b);
        m = 0.0f;
        u16 bits = kf_u16(b + 2 + 2 * g);
#pragma unroll
        for (int j = 0; j < 16; j++) q[j] = ((bits >> j) & 1) ? 1 : -1;
    }
};

KF_QUANT_KERNELS_I8(Q1_0)

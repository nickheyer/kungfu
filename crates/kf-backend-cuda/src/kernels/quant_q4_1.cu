#include "quant.cuh"

struct Q4_1 {
    static constexpr int QK = 32, BYTES = 20;
    static __device__ __forceinline__ void dq8(const u8* b, int g, float* v) {
        float d = kf_f16(b), m = kf_f16(b + 2);
        int j0 = (g & 1) * 8, sh = (g >> 1) * 4;
#pragma unroll
        for (int i = 0; i < 8; i++) v[i] = (float)((b[4 + j0 + i] >> sh) & 0xF) * d + m;
    }

    // Value is d times nibble plus the block minimum, so the offset is minus that minimum
    static __device__ __forceinline__ void q16(const u8* b, int g, i8* q, float& d, float& m) {
        d = kf_f16(b);
        m = -kf_f16(b + 2);
        int sh = g * 4;
#pragma unroll
        for (int j = 0; j < 16; j++) q[j] = (i8)((b[4 + j] >> sh) & 0xF);
    }
};

KF_QUANT_KERNELS_I8(Q4_1)

#include "quant.cuh"

struct Q8_0 {
    static constexpr int QK = 32, BYTES = 34;
    static __device__ __forceinline__ void dq8(const u8* b, int g, float* v) {
        float d = kf_f16(b);
        const u16* q = reinterpret_cast<const u16*>(b + 2 + 8 * g);
#pragma unroll
        for (int i = 0; i < 4; i++) {
            u16 p = q[i];
            v[2 * i] = (float)(i8)(p & 0xFF) * d;
            v[2 * i + 1] = (float)(i8)(p >> 8) * d;
        }
    }
};

KF_QUANT_KERNELS(Q8_0)

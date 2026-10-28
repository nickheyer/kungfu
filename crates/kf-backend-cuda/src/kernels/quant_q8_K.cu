#include "quant.cuh"

struct Q8K {
    static constexpr int QK = 256, BYTES = 292;
    static __device__ __forceinline__ void dq8(const u8* b, int g, float* v) {
        float d = __int_as_float((int)kf_u32(b));
        const u32* q = reinterpret_cast<const u32*>(b + 4 + 8 * g);
#pragma unroll
        for (int i = 0; i < 2; i++) {
            u32 p = q[i];
#pragma unroll
            for (int k = 0; k < 4; k++) v[4 * i + k] = d * (float)(i8)((p >> (8 * k)) & 0xFF);
        }
    }
};

KF_QUANT_KERNELS(Q8K)

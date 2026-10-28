#include "quant.cuh"

struct Q8_1 {
    static constexpr int QK = 32, BYTES = 36;
    static __device__ __forceinline__ void dq8(const u8* b, int g, float* v) {
        float d = kf_f16(b);
        const u32* q = reinterpret_cast<const u32*>(b + 4 + 8 * g);
#pragma unroll
        for (int i = 0; i < 2; i++) {
            u32 p = q[i];
#pragma unroll
            for (int k = 0; k < 4; k++) v[4 * i + k] = (float)(i8)((p >> (8 * k)) & 0xFF) * d;
        }
    }

    static __device__ __forceinline__ void q16(const u8* b, int g, i8* q, float& d, float& m) {
        d = kf_f16(b);
        m = 0.0f;
        const u8* p = b + 4 + 16 * g;
#pragma unroll
        for (int j = 0; j < 16; j++) q[j] = (i8)p[j];
    }
};

KF_QUANT_KERNELS_I8(Q8_1)

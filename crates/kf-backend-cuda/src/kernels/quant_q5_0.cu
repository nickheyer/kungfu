#include "quant.cuh"

struct Q5_0 {
    static constexpr int QK = 32, BYTES = 22;
    static __device__ __forceinline__ void dq8(const u8* b, int g, float* v) {
        float d = kf_f16(b);
        u32 qh = kf_u32(b + 2);
        int j0 = (g & 1) * 8;
        bool hi = g >= 2;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            int j = j0 + i;
            u32 q = hi ? ((u32)(b[6 + j] >> 4) | ((qh >> (j + 12)) & 0x10)) : ((u32)(b[6 + j] & 0xF) | (((qh >> j) << 4) & 0x10));
            v[i] = (float)((int)q - 16) * d;
        }
    }
};

KF_QUANT_KERNELS(Q5_0)

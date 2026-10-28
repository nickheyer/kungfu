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

    static __device__ __forceinline__ void q16(const u8* b, int g, i8* q, float& d, float& m) {
        d = kf_f16(b);
        m = 0.0f;
        u32 qh = kf_u32(b + 2);
        int sh = g * 4;
#pragma unroll
        for (int j = 0; j < 16; j++) {
            u32 v = ((u32)(b[6 + j] >> sh) & 0xF) | (g ? ((qh >> (j + 12)) & 0x10) : (((qh >> j) << 4) & 0x10));
            q[j] = (i8)((int)v - 16);
        }
    }
};

KF_QUANT_KERNELS_I8(Q5_0)

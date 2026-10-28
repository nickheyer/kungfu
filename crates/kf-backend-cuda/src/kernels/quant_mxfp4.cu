#include "quant.cuh"
#include "grids.cuh"

struct Mxfp4 {
    static constexpr int QK = 32, BYTES = 17;
    // Half of the power of two an e8m0 byte encodes, denormal below two
    static __device__ __forceinline__ float e8m0_half(u8 x) {
        u32 bits = x < 2 ? (0x00200000u << x) : ((u32)(x - 1) << 23);
        return __int_as_float((int)bits);
    }
    static __device__ __forceinline__ void dq8(const u8* b, int g, float* v) {
        float d = e8m0_half(b[0]);
        int j0 = (g & 1) * 8, sh = (g >> 1) * 4;
#pragma unroll
        for (int i = 0; i < 8; i++) v[i] = (float)kvalues_fp4[(b[1 + j0 + i] >> sh) & 0xF] * d;
    }

    // The fp4 table holds doubled values and the scale is halved, so the table entries are the integers
    static __device__ __forceinline__ void q16(const u8* b, int g, i8* q, float& d, float& m) {
        d = e8m0_half(b[0]);
        m = 0.0f;
        int sh = g * 4;
#pragma unroll
        for (int j = 0; j < 16; j++) q[j] = kvalues_fp4[(b[1 + j] >> sh) & 0xF];
    }
};

KF_QUANT_KERNELS_I8(Mxfp4)

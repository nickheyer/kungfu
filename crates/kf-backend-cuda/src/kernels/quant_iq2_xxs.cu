#include "quant.cuh"
#include "grids.cuh"

struct Iq2Xxs {
    static constexpr int QK = 256, BYTES = 66;
    static __device__ __forceinline__ void dq8(const u8* b, int g, float* v) {
        float d = kf_f16(b);
        int ib = g >> 2, l = g & 3;
        u32 aux0 = kf_u32(b + 2 + 8 * ib), aux1 = kf_u32(b + 6 + 8 * ib);
        float db = d * (0.5f + (float)(aux1 >> 28)) * 0.25f;
        u64 grid = iq2xxs_grid[(aux0 >> (8 * l)) & 0xFF];
        u8 signs = ksigns_iq2xs[(aux1 >> (7 * l)) & 127];
#pragma unroll
        for (int j = 0; j < 8; j++) v[j] = kf_neg(db * (float)((grid >> (8 * j)) & 0xFF), signs & kmask_iq2xs[j]);
    }
};

KF_QUANT_KERNELS(Iq2Xxs)

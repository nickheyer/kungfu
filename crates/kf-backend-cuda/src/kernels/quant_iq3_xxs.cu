#include "quant.cuh"
#include "grids.cuh"

struct Iq3Xxs {
    static constexpr int QK = 256, BYTES = 98;
    static __device__ __forceinline__ void dq8(const u8* b, int g, float* v) {
        float d = kf_f16(b);
        int ib = g >> 2, l = g & 3;
        u32 aux = kf_u32(b + 66 + 4 * ib);
        float db = d * (0.5f + (float)(aux >> 28)) * 0.5f;
        u8 signs = ksigns_iq2xs[(aux >> (7 * l)) & 127];
        const u8* qs = b + 2 + 8 * ib + 2 * l;
        u32 g1 = iq3xxs_grid[qs[0]], g2 = iq3xxs_grid[qs[1]];
#pragma unroll
        for (int j = 0; j < 4; j++) {
            v[j] = kf_neg(db * (float)((g1 >> (8 * j)) & 0xFF), signs & kmask_iq2xs[j]);
            v[j + 4] = kf_neg(db * (float)((g2 >> (8 * j)) & 0xFF), signs & kmask_iq2xs[j + 4]);
        }
    }
};

KF_QUANT_KERNELS(Iq3Xxs)

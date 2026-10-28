#include "quant.cuh"
#include "grids.cuh"

struct Iq2Xs {
    static constexpr int QK = 256, BYTES = 74;
    static __device__ __forceinline__ void dq8(const u8* b, int g, float* v) {
        float d = kf_f16(b);
        int ib = g >> 2, l = g & 3;
        u8 sc = b[66 + ib];
        float db = d * (0.5f + (float)(l < 2 ? (sc & 0xF) : (sc >> 4))) * 0.25f;
        u16 q = kf_u16(b + 2 + 2 * (4 * ib + l));
        u64 grid = iq2xs_grid[q & 511];
        u8 signs = ksigns_iq2xs[q >> 9];
#pragma unroll
        for (int j = 0; j < 8; j++) v[j] = kf_neg(db * (float)((grid >> (8 * j)) & 0xFF), signs & kmask_iq2xs[j]);
    }
};

KF_QUANT_KERNELS(Iq2Xs)

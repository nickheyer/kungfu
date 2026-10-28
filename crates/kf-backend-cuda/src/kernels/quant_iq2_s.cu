#include "quant.cuh"
#include "grids.cuh"

struct Iq2S {
    static constexpr int QK = 256, BYTES = 82;
    static __device__ __forceinline__ void dq8(const u8* b, int g, float* v) {
        float d = kf_f16(b);
        int ib = g >> 2, l = g & 3;
        u8 sc = b[74 + ib];
        float dl = d * (0.5f + (float)(l < 2 ? (sc & 0xF) : (sc >> 4))) * 0.25f;
        int qh = b[66 + ib];
        int idx = (int)b[2 + 4 * ib + l] | ((qh << (8 - 2 * l)) & 0x300);
        u64 grid = iq2s_grid[idx];
        u8 signs = b[34 + 4 * ib + l];
#pragma unroll
        for (int j = 0; j < 8; j++) v[j] = kf_neg(dl * (float)((grid >> (8 * j)) & 0xFF), signs & kmask_iq2xs[j]);
    }
};

KF_QUANT_KERNELS(Iq2S)

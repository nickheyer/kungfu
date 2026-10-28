#include "quant.cuh"
#include "grids.cuh"

struct Iq1S {
    static constexpr int QK = 256, BYTES = 50;
    static __device__ __forceinline__ void dq8(const u8* b, int g, float* v) {
        float d = kf_f16(b);
        int ib = g >> 2, l = g & 3;
        int qh = kf_u16(b + 34 + 2 * ib);
        float dl = d * (float)(2 * ((qh >> 12) & 7) + 1);
        float delta = (qh & 0x8000) ? -0.125f : 0.125f;
        u64 grid = iq1s_grid[(int)b[2 + 4 * ib + l] | (((qh >> (3 * l)) & 7) << 8)];
#pragma unroll
        for (int j = 0; j < 8; j++) v[j] = dl * ((float)(i8)((grid >> (8 * j)) & 0xFF) + delta);
    }
};

KF_QUANT_KERNELS(Iq1S)

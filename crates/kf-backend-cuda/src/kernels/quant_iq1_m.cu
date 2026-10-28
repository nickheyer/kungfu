#include "quant.cuh"
#include "grids.cuh"

struct Iq1M {
    static constexpr int QK = 256, BYTES = 56;
    // Block scale is a half assembled from the top nibbles of the four scale words
    static __device__ __forceinline__ void dq8(const u8* b, int g, float* v) {
        u16 sc0 = kf_u16(b + 48), sc1 = kf_u16(b + 50), sc2 = kf_u16(b + 52), sc3 = kf_u16(b + 54);
        float d = kf_h2f((u16)((sc0 >> 12) | ((sc1 >> 8) & 0x00f0) | ((sc2 >> 4) & 0x0f00) | (sc3 & 0xf000)));
        int ib = g >> 2, l = g & 3;
        int s = kf_u16(b + 48 + 2 * (ib / 2));
        int sh = 6 * (ib % 2) + (l < 2 ? 0 : 3);
        float dl = d * (float)(2 * ((s >> sh) & 7) + 1);
        int qh = b[32 + 2 * ib + (l >> 1)];
        int idx = (int)b[4 * ib + l] | ((qh << ((l & 1) ? 4 : 8)) & 0x700);
        float delta = (qh & ((l & 1) ? 0x80 : 0x08)) ? -0.125f : 0.125f;
        u64 grid = iq1s_grid[idx];
#pragma unroll
        for (int j = 0; j < 8; j++) v[j] = dl * ((float)(i8)((grid >> (8 * j)) & 0xFF) + delta);
    }

    // The delta sign changes per eight, so it is folded into the integers as in iq1_s
    static __device__ __forceinline__ void q16(const u8* b, int g, i8* q, float& dd, float& m) {
        u16 sc0 = kf_u16(b + 48), sc1 = kf_u16(b + 50), sc2 = kf_u16(b + 52), sc3 = kf_u16(b + 54);
        float d = kf_h2f((u16)((sc0 >> 12) | ((sc1 >> 8) & 0x00f0) | ((sc2 >> 4) & 0x0f00) | (sc3 & 0xf000)));
        int ib = g >> 1, half = g & 1;
        int s = kf_u16(b + 48 + 2 * (ib / 2));
        int sh = 6 * (ib % 2) + (half == 0 ? 0 : 3);
        dd = d * (float)(2 * ((s >> sh) & 7) + 1) * 0.125f;
        m = 0.0f;
#pragma unroll
        for (int h = 0; h < 2; h++) {
            int l = half * 2 + h;
            int qh = b[32 + 2 * ib + (l >> 1)];
            int idx = (int)b[4 * ib + l] | ((qh << ((l & 1) ? 4 : 8)) & 0x700);
            int delta = (qh & ((l & 1) ? 0x80 : 0x08)) ? -1 : 1;
            u64 grid = iq1s_grid[idx];
#pragma unroll
            for (int j = 0; j < 8; j++) q[8 * h + j] = (i8)(8 * (int)(i8)((grid >> (8 * j)) & 0xFF) + delta);
        }
    }
};

KF_QUANT_KERNELS_I8(Iq1M)

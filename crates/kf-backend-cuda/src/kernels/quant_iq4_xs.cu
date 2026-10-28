#include "quant.cuh"
#include "grids.cuh"

struct Iq4Xs {
    static constexpr int QK = 256, BYTES = 136;
    static __device__ __forceinline__ void dq8(const u8* b, int g, float* v) {
        float d = kf_f16(b);
        int ib = g >> 2, w = g & 3;
        int scales_h = kf_u16(b + 2);
        int ls = (((b[4 + ib / 2] >> (4 * (ib % 2))) & 0xF) | (((scales_h >> (2 * ib)) & 3) << 4)) - 32;
        float dl = d * (float)ls;
        const u8* qs = b + 8 + 16 * ib + (w & 1) * 8;
        int sh = (w >> 1) * 4;
#pragma unroll
        for (int i = 0; i < 8; i++) v[i] = dl * (float)kvalues_iq4nl[(qs[i] >> sh) & 0xF];
    }

    // Sixteen group g is sub block g over two, low nibbles for even g and high for odd
    static __device__ __forceinline__ void q16(const u8* b, int g, i8* q, float& d, float& m) {
        int ib = g >> 1, sh = (g & 1) * 4;
        int scales_h = kf_u16(b + 2);
        int ls = (((b[4 + ib / 2] >> (4 * (ib % 2))) & 0xF) | (((scales_h >> (2 * ib)) & 3) << 4)) - 32;
        d = kf_f16(b) * (float)ls;
        m = 0.0f;
        const u8* qs = b + 8 + 16 * ib;
#pragma unroll
        for (int j = 0; j < 16; j++) q[j] = kvalues_iq4nl[(qs[j] >> sh) & 0xF];
    }
};

KF_QUANT_KERNELS_I8(Iq4Xs)

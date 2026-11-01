#include "quant.cuh"
#include "grids.cuh"

struct Iq3S {
    static constexpr int QK = 256, BYTES = 110;
    static __device__ __forceinline__ void dq8(const u8* b, int g, float* v) {
        float d = kf_f16(b);
        int ib = g >> 2, l = g & 3;
        const u8* qs = b + 2 + 8 * ib;
        int qh = b[66 + ib];
        u8 signs = b[74 + 4 * ib + l];
        int sc = (b[106 + ib / 2] >> (4 * (ib % 2))) & 0xF;
        float db = d * (float)(1 + 2 * sc);
        u32 g1 = iq3s_grid[(int)qs[2 * l] | ((qh << (8 - 2 * l)) & 256)];
        u32 g2 = iq3s_grid[(int)qs[2 * l + 1] | ((qh << (7 - 2 * l)) & 256)];
#pragma unroll
        for (int j = 0; j < 4; j++) {
            v[j] = kf_neg(db * (float)((g1 >> (8 * j)) & 0xFF), signs & kmask_iq2xs[j]);
            v[j + 4] = kf_neg(db * (float)((g2 >> (8 * j)) & 0xFF), signs & kmask_iq2xs[j + 4]);
        }
    }

    static __device__ __forceinline__ void q16(const u8* b, int g, i8* q, float& d, float& m) {
        int ib = g >> 1;
        const u8* qs = b + 2 + 8 * ib;
        int qh = b[66 + ib];
        int sc = (b[106 + ib / 2] >> (4 * (ib % 2))) & 0xF;
        d = kf_f16(b) * (float)(1 + 2 * sc);
        m = 0.0f;
#pragma unroll
        for (int h = 0; h < 2; h++) {
            int l = (g & 1) * 2 + h;
            u8 signs = b[74 + 4 * ib + l];
            u32 g1 = iq3s_grid[(int)qs[2 * l] | ((qh << (8 - 2 * l)) & 256)];
            u32 g2 = iq3s_grid[(int)qs[2 * l + 1] | ((qh << (7 - 2 * l)) & 256)];
#pragma unroll
            for (int j = 0; j < 4; j++) {
                int v1 = (int)((g1 >> (8 * j)) & 0xFF), v2 = (int)((g2 >> (8 * j)) & 0xFF);
                q[8 * h + j] = (i8)((signs & kmask_iq2xs[j]) ? -v1 : v1);
                q[8 * h + j + 4] = (i8)((signs & kmask_iq2xs[j + 4]) ? -v2 : v2);
            }
        }
    }
};

// Word form: four grid words signed in place, the low sign nibble over the first word of each pair
template <>
struct KfQ16<Iq3S> {
    static __device__ __forceinline__ void load(const u8* __restrict__ b, int g, int4& wv, float& d, float& m) {
        int ib = g >> 1, l0 = (g & 1) * 2;
        const u8* qs = b + 2 + 8 * ib;
        int qh = b[66 + ib];
        int sc = (b[106 + ib / 2] >> (4 * (ib % 2))) & 0xF;
        d = kf_f16(b) * (float)(1 + 2 * sc);
        m = 0.0f;
        u32 s0 = b[74 + 4 * ib + l0], s1 = b[74 + 4 * ib + l0 + 1];
        wv.x = (int)kf_sign4(iq3s_grid[(int)qs[2 * l0] | ((qh << (8 - 2 * l0)) & 256)], s0);
        wv.y = (int)kf_sign4(iq3s_grid[(int)qs[2 * l0 + 1] | ((qh << (7 - 2 * l0)) & 256)], s0 >> 4);
        wv.z = (int)kf_sign4(iq3s_grid[(int)qs[2 * l0 + 2] | ((qh << (8 - 2 * (l0 + 1))) & 256)], s1);
        wv.w = (int)kf_sign4(iq3s_grid[(int)qs[2 * l0 + 3] | ((qh << (7 - 2 * (l0 + 1))) & 256)], s1 >> 4);
    }
};

KF_QUANT_KERNELS_I8(Iq3S)

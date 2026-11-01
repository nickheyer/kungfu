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

    static __device__ __forceinline__ void q16(const u8* b, int g, i8* q, float& d, float& m) {
        int ib = g >> 1, half = g & 1;
        u8 sc = b[66 + ib];
        d = kf_f16(b) * (0.5f + (float)(half == 0 ? (sc & 0xF) : (sc >> 4))) * 0.25f;
        m = 0.0f;
#pragma unroll
        for (int h = 0; h < 2; h++) {
            int l = half * 2 + h;
            u16 w = kf_u16(b + 2 + 2 * (4 * ib + l));
            u64 grid = iq2xs_grid[w & 511];
            u8 signs = ksigns_iq2xs[w >> 9];
#pragma unroll
            for (int j = 0; j < 8; j++) {
                int v = (int)((grid >> (8 * j)) & 0xFF);
                q[8 * h + j] = (i8)((signs & kmask_iq2xs[j]) ? -v : v);
            }
        }
    }
};

// Word form: the two grid entries of the group signed in place as four words
template <>
struct KfQ16<Iq2Xs> {
    static __device__ __forceinline__ void load(const u8* __restrict__ b, int g, int4& wv, float& d, float& m) {
        int ib = g >> 1, half = g & 1, l0 = half * 2;
        u8 sc = b[66 + ib];
        d = kf_f16(b) * (0.5f + (float)(half == 0 ? (sc & 0xF) : (sc >> 4))) * 0.25f;
        m = 0.0f;
        u16 w0 = kf_u16(b + 2 + 2 * (4 * ib + l0)), w1 = kf_u16(b + 2 + 2 * (4 * ib + l0 + 1));
        u32 lo, hi;
        kf_grid8(iq2xs_grid[w0 & 511], ksigns_iq2xs[w0 >> 9], lo, hi);
        wv.x = (int)lo;
        wv.y = (int)hi;
        kf_grid8(iq2xs_grid[w1 & 511], ksigns_iq2xs[w1 >> 9], lo, hi);
        wv.z = (int)lo;
        wv.w = (int)hi;
    }
};

KF_QUANT_KERNELS_I8(Iq2Xs)

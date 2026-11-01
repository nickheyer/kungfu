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

    // Two eight groups of sub block ib share its scale, the grid bytes signed per lane
    static __device__ __forceinline__ void q16(const u8* b, int g, i8* q, float& d, float& m) {
        int ib = g >> 1;
        u32 aux0 = kf_u32(b + 2 + 8 * ib), aux1 = kf_u32(b + 6 + 8 * ib);
        d = kf_f16(b) * (0.5f + (float)(aux1 >> 28)) * 0.25f;
        m = 0.0f;
#pragma unroll
        for (int h = 0; h < 2; h++) {
            int l = (g & 1) * 2 + h;
            u64 grid = iq2xxs_grid[(aux0 >> (8 * l)) & 0xFF];
            u8 signs = ksigns_iq2xs[(aux1 >> (7 * l)) & 127];
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
struct KfQ16<Iq2Xxs> {
    static __device__ __forceinline__ void load(const u8* __restrict__ b, int g, int4& wv, float& d, float& m) {
        int ib = g >> 1, l0 = (g & 1) * 2;
        u32 aux0 = kf_u32(b + 2 + 8 * ib), aux1 = kf_u32(b + 6 + 8 * ib);
        d = kf_f16(b) * (0.5f + (float)(aux1 >> 28)) * 0.25f;
        m = 0.0f;
        u32 lo, hi;
        kf_grid8(iq2xxs_grid[(aux0 >> (8 * l0)) & 0xFF], ksigns_iq2xs[(aux1 >> (7 * l0)) & 127], lo, hi);
        wv.x = (int)lo;
        wv.y = (int)hi;
        kf_grid8(iq2xxs_grid[(aux0 >> (8 * (l0 + 1))) & 0xFF], ksigns_iq2xs[(aux1 >> (7 * (l0 + 1))) & 127], lo, hi);
        wv.z = (int)lo;
        wv.w = (int)hi;
    }
};

KF_QUANT_KERNELS_I8(Iq2Xxs)

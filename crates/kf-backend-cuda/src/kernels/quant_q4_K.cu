#include "quant.cuh"

struct Q4K {
    static constexpr int QK = 256, BYTES = 144;
    // Scale and min of sub block j from the twelve byte packed array
    static __device__ __forceinline__ void scale_min(int j, const u8* q, int& sc, int& m) {
        if (j < 4) {
            sc = q[j] & 63;
            m = q[j + 4] & 63;
        } else {
            sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
            m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        }
    }
    static __device__ __forceinline__ void dq8(const u8* b, int g, float* v) {
        float d = kf_f16(b), mn = kf_f16(b + 2);
        int j = g >> 3, hi = (g >> 2) & 1, l0 = (g & 3) * 8;
        int sc, m;
        scale_min(2 * j + hi, b + 4, sc, m);
        float d1 = d * (float)sc, m1 = mn * (float)m;
        uint2 w = *reinterpret_cast<const uint2*>(b + 16 + 32 * j + l0);
        int sh = hi * 4;
#pragma unroll
        for (int i = 0; i < 4; i++) {
            v[i] = d1 * (float)((w.x >> (8 * i + sh)) & 0xF) - m1;
            v[4 + i] = d1 * (float)((w.y >> (8 * i + sh)) & 0xF) - m1;
        }
    }

    // Sixteen group g is half l0 of sub block two j plus hi
    static __device__ __forceinline__ void q16(const u8* b, int g, i8* q, float& d, float& m) {
        int j = g >> 2, hi = (g >> 1) & 1, l0 = (g & 1) * 16;
        int sc, mi;
        scale_min(2 * j + hi, b + 4, sc, mi);
        d = kf_f16(b) * (float)sc;
        m = kf_f16(b + 2) * (float)mi;
        const u8* p = b + 16 + 32 * j + l0;
        int sh = hi * 4;
#pragma unroll
        for (int i = 0; i < 16; i++) q[i] = (i8)((p[i] >> sh) & 0xF);
    }
};

// Word form: the sixteen nibble bytes of the group as one aligned load, the scale and min out of the three scale words
template <>
struct KfQ16<Q4K> {
    static __device__ __forceinline__ void load(const u8* __restrict__ b, int g, int4& wv, float& d, float& m) {
        const u32* bw = reinterpret_cast<const u32*>(b);
        u32 w0 = bw[0];
        int j = g >> 2, hi = (g >> 1) & 1, l0 = (g & 1) * 16;
        int s = 2 * j + hi;
        const u32* sw = bw + 1;
        // Both scale forms from three bytes, selected without a lane divergent branch
        int bs = kf_wbyte(sw, s), bs4 = kf_wbyte(sw, s + 4), bm4 = kf_wbyte(sw, s & 3);
        int sc = s < 4 ? (bs & 63) : ((bs4 & 0xF) | ((bm4 >> 6) << 4));
        int mi = s < 4 ? (bs4 & 63) : ((bs4 >> 4) | ((bs >> 6) << 4));
        d = kf_h2f((u16)(w0 & 0xffffu)) * (float)sc;
        m = kf_h2f((u16)(w0 >> 16)) * (float)mi;
        int4 r = *reinterpret_cast<const int4*>(b + 16 + 32 * j + l0);
        u32 sh = hi * 4;
        wv.x = (int)(((u32)r.x >> sh) & 0x0f0f0f0fu);
        wv.y = (int)(((u32)r.y >> sh) & 0x0f0f0f0fu);
        wv.z = (int)(((u32)r.z >> sh) & 0x0f0f0f0fu);
        wv.w = (int)(((u32)r.w >> sh) & 0x0f0f0f0fu);
    }
};

// Sixty four columns per lane: sub block pair j of lane j, the thirty two nibble bytes as two loads, the low nibbles
// the even sub block and the high the odd, with both scales out of the three scale words
template <>
struct KfLane<Q4K> {
    static constexpr int COLS = 64, NS = 2, CSTRIDE = 16;
    static __device__ __forceinline__ int base(int gg) {
        return gg * 64;
    }
};

template <>
struct KfQw<Q4K, 64> {
    static __device__ __forceinline__ void load(const u8* __restrict__ b, int j, int4* wv, float* d, float* m) {
        const u32* bw = reinterpret_cast<const u32*>(b);
        const u32 w0 = bw[0];
        const u32* sw = bw + 1;
        const float dd = kf_h2f((u16)(w0 & 0xffffu)), dm = kf_h2f((u16)(w0 >> 16));
#pragma unroll
        for (int h = 0; h < 2; h++) {
            int s = 2 * j + h;
            int bs = kf_wbyte(sw, s), bs4 = kf_wbyte(sw, s + 4), bm4 = kf_wbyte(sw, s & 3);
            int sc = s < 4 ? (bs & 63) : ((bs4 & 0xF) | ((bm4 >> 6) << 4));
            int mi = s < 4 ? (bs4 & 63) : ((bs4 >> 4) | ((bs >> 6) << 4));
            d[h] = dd * (float)sc;
            m[h] = dm * (float)mi;
        }
        const int4 r0 = *reinterpret_cast<const int4*>(b + 16 + 32 * j);
        const int4 r1 = *reinterpret_cast<const int4*>(b + 32 + 32 * j);
        wv[0] = make_int4((int)((u32)r0.x & 0x0f0f0f0fu), (int)((u32)r0.y & 0x0f0f0f0fu), (int)((u32)r0.z & 0x0f0f0f0fu), (int)((u32)r0.w & 0x0f0f0f0fu));
        wv[1] = make_int4((int)((u32)r1.x & 0x0f0f0f0fu), (int)((u32)r1.y & 0x0f0f0f0fu), (int)((u32)r1.z & 0x0f0f0f0fu), (int)((u32)r1.w & 0x0f0f0f0fu));
        wv[2] = make_int4((int)(((u32)r0.x >> 4) & 0x0f0f0f0fu), (int)(((u32)r0.y >> 4) & 0x0f0f0f0fu), (int)(((u32)r0.z >> 4) & 0x0f0f0f0fu), (int)(((u32)r0.w >> 4) & 0x0f0f0f0fu));
        wv[3] = make_int4((int)(((u32)r1.x >> 4) & 0x0f0f0f0fu), (int)(((u32)r1.y >> 4) & 0x0f0f0f0fu), (int)(((u32)r1.z >> 4) & 0x0f0f0f0fu), (int)(((u32)r1.w >> 4) & 0x0f0f0f0fu));
    }
};

// Each group's integers sit above an offset the product takes off
template <>
struct KfOffset<Q4K> {
    static constexpr bool V = true;
};

KF_QUANT_KERNELS_I8(Q4K)

#include "quant.cuh"

struct Q6K {
    static constexpr int QK = 256, BYTES = 210;
    // Quad k of the 128 element half n combines a nibble of ql with two bits of qh
    static __device__ __forceinline__ void dq8(const u8* b, int g, float* v) {
        float d = kf_f16(b + 208);
        int e0 = g * 8, n = e0 >> 7, r = e0 & 127, k = r >> 5, l0 = r & 31;
        const u8* ql = b + 64 * n + (k & 1) * 32;
        const u8* qh = b + 128 + 32 * n;
        float dsc = d * (float)(i8)b[192 + 8 * n + (l0 >> 4) + 2 * k];
        int lsh = (k >> 1) * 4, hsh = 2 * k;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            int l = l0 + i;
            int q = (int)(i8)(u8)(((ql[l] >> lsh) & 0xF) | (((qh[l] >> hsh) & 3) << 4)) - 32;
            v[i] = dsc * (float)q;
        }
    }

    // One signed scale per sixteen
    static __device__ __forceinline__ void q16(const u8* b, int g, i8* q, float& d, float& m) {
        int e0 = g * 16, n = e0 >> 7, r = e0 & 127, k = r >> 5, l0 = r & 31;
        const u8* ql = b + 64 * n + (k & 1) * 32 + l0;
        const u8* qh = b + 128 + 32 * n + l0;
        d = kf_f16(b + 208) * (float)(i8)b[192 + 8 * n + (l0 >> 4) + 2 * k];
        m = 0.0f;
        int lsh = (k >> 1) * 4, hsh = 2 * k;
#pragma unroll
        for (int i = 0; i < 16; i++) q[i] = (i8)((int)(i8)(u8)(((ql[i] >> lsh) & 0xF) | (((qh[i] >> hsh) & 3) << 4)) - 32);
    }
};

// Word form: the sixteen low bytes and high bytes of the group, shifted into six bits and centred by taking 32 off every byte
template <>
struct KfQ16<Q6K> {
    static __device__ __forceinline__ void load(const u8* __restrict__ b, int g, int4& wv, float& d, float& m) {
        int e0 = g * 16, n = e0 >> 7, r = e0 & 127, k = r >> 5, l0 = r & 31;
        int4 lo = kf_ld16(b + 64 * n + (k & 1) * 32 + l0);
        int4 hi = kf_ld16(b + 128 + 32 * n + l0);
        d = kf_h2f(*reinterpret_cast<const u16*>(b + 208)) * (float)(i8)b[192 + 8 * n + (l0 >> 4) + 2 * k];
        m = 0.0f;
        u32 lsh = (k >> 1) * 4, hsh = 2 * k;
        wv.x = (int)__vsub4((((u32)lo.x >> lsh) & 0x0f0f0f0fu) | ((((u32)hi.x >> hsh) & 0x03030303u) << 4), 0x20202020u);
        wv.y = (int)__vsub4((((u32)lo.y >> lsh) & 0x0f0f0f0fu) | ((((u32)hi.y >> hsh) & 0x03030303u) << 4), 0x20202020u);
        wv.z = (int)__vsub4((((u32)lo.z >> lsh) & 0x0f0f0f0fu) | ((((u32)hi.z >> hsh) & 0x03030303u) << 4), 0x20202020u);
        wv.w = (int)__vsub4((((u32)lo.w >> lsh) & 0x0f0f0f0fu) | ((((u32)hi.w >> hsh) & 0x03030303u) << 4), 0x20202020u);
    }
};

// Sixty four columns per lane: lane gg takes half n = gg / 2 of the block at byte offset l0 = 16 (gg mod 2), where the
// sixteen low bytes at l0 and at l0 + 32 with the sixteen high bytes at l0 give the four thirty two groups of the half,
// so the lane's chunks sit 32 columns apart from column 128 n + l0, each with its own signed scale
template <>
struct KfLane<Q6K> {
    static constexpr int COLS = 64, NS = 4, CSTRIDE = 32;
    static __device__ __forceinline__ int base(int gg) {
        return (gg >> 1) * 128 + (gg & 1) * 16;
    }
};

template <>
struct KfQw<Q6K, 64> {
    static __device__ __forceinline__ void load(const u8* __restrict__ b, int gg, int4* wv, float* d, float* m) {
        const int n = gg >> 1, l0 = (gg & 1) * 16;
        const int4 lo0 = kf_ld16(b + 64 * n + l0);
        const int4 lo1 = kf_ld16(b + 64 * n + 32 + l0);
        const int4 hi = kf_ld16(b + 128 + 32 * n + l0);
        const float dd = kf_h2f(*reinterpret_cast<const u16*>(b + 208));
        // The eight scales of the half, byte l0 / 16 + 2 k for group k, the first four in one word and the rest in the next
        const int4 scw = kf_ld16(b + 192 + 8 * n);
        const int sb = 8 * (l0 >> 4);
        d[0] = dd * (float)(i8)(((u32)scw.x >> sb) & 0xff);
        d[1] = dd * (float)(i8)(((u32)scw.x >> (sb + 16)) & 0xff);
        d[2] = dd * (float)(i8)(((u32)scw.y >> sb) & 0xff);
        d[3] = dd * (float)(i8)(((u32)scw.y >> (sb + 16)) & 0xff);
        m[0] = m[1] = m[2] = m[3] = 0.0f;
        const u32 m4 = 0x0f0f0f0fu, m2 = 0x03030303u, c32 = 0x20202020u;
        wv[0] = make_int4((int)__vsub4(((u32)lo0.x & m4) | (((u32)hi.x & m2) << 4), c32), (int)__vsub4(((u32)lo0.y & m4) | (((u32)hi.y & m2) << 4), c32), (int)__vsub4(((u32)lo0.z & m4) | (((u32)hi.z & m2) << 4), c32), (int)__vsub4(((u32)lo0.w & m4) | (((u32)hi.w & m2) << 4), c32));
        wv[1] = make_int4((int)__vsub4(((u32)lo1.x & m4) | ((((u32)hi.x >> 2) & m2) << 4), c32), (int)__vsub4(((u32)lo1.y & m4) | ((((u32)hi.y >> 2) & m2) << 4), c32), (int)__vsub4(((u32)lo1.z & m4) | ((((u32)hi.z >> 2) & m2) << 4), c32), (int)__vsub4(((u32)lo1.w & m4) | ((((u32)hi.w >> 2) & m2) << 4), c32));
        wv[2] = make_int4((int)__vsub4((((u32)lo0.x >> 4) & m4) | ((((u32)hi.x >> 4) & m2) << 4), c32), (int)__vsub4((((u32)lo0.y >> 4) & m4) | ((((u32)hi.y >> 4) & m2) << 4), c32), (int)__vsub4((((u32)lo0.z >> 4) & m4) | ((((u32)hi.z >> 4) & m2) << 4), c32), (int)__vsub4((((u32)lo0.w >> 4) & m4) | ((((u32)hi.w >> 4) & m2) << 4), c32));
        wv[3] = make_int4((int)__vsub4((((u32)lo1.x >> 4) & m4) | ((((u32)hi.x >> 6) & m2) << 4), c32), (int)__vsub4((((u32)lo1.y >> 4) & m4) | ((((u32)hi.y >> 6) & m2) << 4), c32), (int)__vsub4((((u32)lo1.z >> 4) & m4) | ((((u32)hi.z >> 6) & m2) << 4), c32), (int)__vsub4((((u32)lo1.w >> 4) & m4) | ((((u32)hi.w >> 6) & m2) << 4), c32));
    }
};

KF_QUANT_KERNELS_I8(Q6K)

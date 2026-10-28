#include "quant.cuh"

struct Q3K {
    static constexpr int QK = 256, BYTES = 110;
    // Six bit scale of sub block is unpacked from the twelve packed scale bytes
    static __device__ __forceinline__ int scale(const u8* b, int is) {
        const u8* s = b + 96;
        int p = is & 3, k = is >> 2;
        int lo = k == 0 ? (s[p] & 0xF) : k == 1 ? (s[4 + p] & 0xF) : k == 2 ? (s[p] >> 4) : (s[4 + p] >> 4);
        int hi = (s[8 + p] >> (2 * k)) & 3;
        return (int)(i8)(u8)(lo | (hi << 4));
    }
    static __device__ __forceinline__ void dq8(const u8* b, int g, float* v) {
        float d_all = kf_f16(b + 108);
        int e0 = g * 8, n = e0 >> 7, r = e0 & 127, s = r >> 5, l0 = r & 31, h = l0 >> 4;
        float dl = d_all * (float)(scale(b, n * 8 + s * 2 + h) - 32);
        u8 m = (u8)(1u << (n * 4 + s));
        const u8* q = b + 32 + n * 32;
        const u8* hm = b;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            int l = l0 + i;
            int hi = (hm[l] & m) ? 0 : 4;
            v[i] = dl * (float)((int)((q[l] >> (2 * s)) & 3) - hi);
        }
    }

    static __device__ __forceinline__ void q16(const u8* b, int g, i8* q, float& d, float& m) {
        int e0 = g * 16, n = e0 >> 7, r = e0 & 127, s = r >> 5, l0 = r & 31, h = l0 >> 4;
        d = kf_f16(b + 108) * (float)(scale(b, n * 8 + s * 2 + h) - 32);
        m = 0.0f;
        u8 mask = (u8)(1u << (n * 4 + s));
        const u8* p = b + 32 + n * 32 + l0;
        const u8* hm = b + l0;
#pragma unroll
        for (int j = 0; j < 16; j++) q[j] = (i8)((int)((p[j] >> (2 * s)) & 3) - ((hm[j] & mask) ? 0 : 4));
    }
};

KF_QUANT_KERNELS_I8(Q3K)

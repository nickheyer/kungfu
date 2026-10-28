#include "quant.cuh"

struct Q5K {
    static constexpr int QK = 256, BYTES = 176;
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
        const u8* qh = b + 16;
        const u8* ql = b + 48 + 32 * j;
        u8 u = (u8)(1u << (2 * j + hi));
#pragma unroll
        for (int i = 0; i < 8; i++) {
            int l = l0 + i;
            int q = (hi ? (ql[l] >> 4) : (ql[l] & 0xF)) + ((qh[l] & u) ? 16 : 0);
            v[i] = d1 * (float)q - m1;
        }
    }

    static __device__ __forceinline__ void q16(const u8* b, int g, i8* q, float& d, float& m) {
        int j = g >> 2, hi = (g >> 1) & 1, l0 = (g & 1) * 16;
        int sc, mi;
        scale_min(2 * j + hi, b + 4, sc, mi);
        d = kf_f16(b) * (float)sc;
        m = kf_f16(b + 2) * (float)mi;
        const u8* qh = b + 16 + l0;
        const u8* ql = b + 48 + 32 * j + l0;
        u8 u = (u8)(1u << (2 * j + hi));
#pragma unroll
        for (int i = 0; i < 16; i++) q[i] = (i8)((hi ? (ql[i] >> 4) : (ql[i] & 0xF)) + ((qh[i] & u) ? 16 : 0));
    }
};

KF_QUANT_KERNELS_I8(Q5K)

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
};

KF_QUANT_KERNELS(Q5K)

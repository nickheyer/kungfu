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

KF_QUANT_KERNELS_I8(Q4K)

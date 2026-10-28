#include "quant.cuh"

struct Q2K {
    static constexpr int QK = 256, BYTES = 84;
    // Element e sits in half h of shift slot s within the 128 element chunk n
    static __device__ __forceinline__ void dq8(const u8* b, int g, float* v) {
        float d = kf_f16(b + 80), mn = kf_f16(b + 82);
        int e0 = g * 8, n = e0 >> 7, r = e0 & 127, s = r >> 5, h = (r & 31) >> 4, l0 = r & 15;
        u8 sc = b[n * 8 + s * 2 + h];
        float dl = d * (float)(sc & 0xF), ml = mn * (float)(sc >> 4);
        const u8* q = b + 16 + n * 32 + h * 16 + l0;
#pragma unroll
        for (int i = 0; i < 8; i++) v[i] = dl * (float)((q[i] >> (2 * s)) & 3) - ml;
    }

    // Each sixteen has its own four bit scale and minimum
    static __device__ __forceinline__ void q16(const u8* b, int g, i8* q, float& d, float& m) {
        float dd = kf_f16(b + 80), mn = kf_f16(b + 82);
        int e0 = g * 16, n = e0 >> 7, r = e0 & 127, s = r >> 5, h = (r & 31) >> 4;
        u8 sc = b[n * 8 + s * 2 + h];
        d = dd * (float)(sc & 0xF);
        m = mn * (float)(sc >> 4);
        const u8* p = b + 16 + n * 32 + h * 16;
#pragma unroll
        for (int j = 0; j < 16; j++) q[j] = (i8)((p[j] >> (2 * s)) & 3);
    }
};

KF_QUANT_KERNELS_I8(Q2K)

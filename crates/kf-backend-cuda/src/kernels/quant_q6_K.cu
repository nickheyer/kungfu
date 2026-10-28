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
};

KF_QUANT_KERNELS(Q6K)

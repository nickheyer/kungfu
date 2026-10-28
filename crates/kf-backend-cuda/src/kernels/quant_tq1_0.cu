#include "quant.cuh"

struct Tq1_0 {
    static constexpr int QK = 256, BYTES = 54;
    // Base three digit n of a packed byte as minus one, zero or one times d
    static __device__ __forceinline__ float trit(u8 q, int n, float d) {
        const u8 pow3[6] = {1, 3, 9, 27, 81, 243};
        u8 p = (u8)(q * pow3[n]);
        int xi = (int)(((u16)p * 3) >> 8);
        return (float)(xi - 1) * d;
    }
    static __device__ __forceinline__ void dq8(const u8* b, int g, float* v) {
        float d = kf_f16(b + 52);
#pragma unroll
        for (int i = 0; i < 8; i++) {
            int e = g * 8 + i, n, at;
            if (e < 160) {
                n = e / 32;
                at = e % 32;
            } else if (e < 240) {
                n = (e - 160) / 16;
                at = 32 + (e - 160) % 16;
            } else {
                n = (e - 240) / 4;
                at = 48 + (e - 240) % 4;
            }
            v[i] = trit(b[at], n, d);
        }
    }

    static __device__ __forceinline__ int trit_i(u8 v, int n) {
        const u8 pow3[6] = {1, 3, 9, 27, 81, 243};
        u8 p = (u8)(v * pow3[n]);
        return (int)(((u16)p * 3) >> 8) - 1;
    }
    static __device__ __forceinline__ void q16(const u8* b, int g, i8* q, float& d, float& m) {
        d = kf_f16(b + 52);
        m = 0.0f;
#pragma unroll
        for (int i = 0; i < 16; i++) {
            int e = g * 16 + i, n, at;
            if (e < 160) {
                n = e / 32;
                at = e % 32;
            } else if (e < 240) {
                n = (e - 160) / 16;
                at = 32 + (e - 160) % 16;
            } else {
                n = (e - 240) / 4;
                at = 48 + (e - 240) % 4;
            }
            q[i] = (i8)trit_i(b[at], n);
        }
    }
};

KF_QUANT_KERNELS_I8(Tq1_0)

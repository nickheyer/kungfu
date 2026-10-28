#include "quant.cuh"
#include "grids.cuh"

struct Nvfp4 {
    static constexpr int QK = 64, BYTES = 36;
    // Half the value of an unsigned e4m3 scale byte, zero for the two null encodings
    static __device__ __forceinline__ float ue4m3_half(u8 x) {
        if (x == 0 || x == 0x7F) return 0.0f;
        int e = (x >> 3) & 0xF, m = x & 7;
        float raw = e == 0 ? (float)m * __int_as_float((127 - 9) << 23) : (1.0f + (float)m / 8.0f) * __int_as_float((e - 7 + 127) << 23);
        return raw * 0.5f;
    }
    static __device__ __forceinline__ void dq8(const u8* b, int g, float* v) {
        int s = g >> 1, sh = (g & 1) * 4;
        float d = ue4m3_half(b[s]);
        const u8* qs = b + 4 + 8 * s;
#pragma unroll
        for (int i = 0; i < 8; i++) v[i] = (float)kvalues_fp4[(qs[i] >> sh) & 0xF] * d;
    }

    static __device__ __forceinline__ void q16(const u8* b, int g, i8* q, float& d, float& m) {
        d = ue4m3_half(b[g]);
        m = 0.0f;
        const u8* qs = b + 4 + 8 * g;
#pragma unroll
        for (int j = 0; j < 8; j++) {
            q[j] = kvalues_fp4[qs[j] & 0xF];
            q[j + 8] = kvalues_fp4[qs[j] >> 4];
        }
    }
};

KF_QUANT_KERNELS_I8(Nvfp4)

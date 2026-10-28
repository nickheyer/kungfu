#include "quant.cuh"

struct Q4_0 {
    static constexpr int QK = 32, BYTES = 18;
    // Group g holds low nibbles for g below two, high nibbles above
    static __device__ __forceinline__ void dq8(const u8* b, int g, float* v) {
        float d = kf_f16(b);
        int j0 = (g & 1) * 8, sh = (g >> 1) * 4;
#pragma unroll
        for (int i = 0; i < 8; i++) v[i] = (float)((int)((b[2 + j0 + i] >> sh) & 0xF) - 8) * d;
    }
};

KF_QUANT_KERNELS(Q4_0)

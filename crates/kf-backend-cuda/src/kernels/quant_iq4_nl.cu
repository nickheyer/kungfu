#include "quant.cuh"
#include "grids.cuh"

struct Iq4Nl {
    static constexpr int QK = 32, BYTES = 18;
    static __device__ __forceinline__ void dq8(const u8* b, int g, float* v) {
        float d = kf_f16(b);
        int j0 = (g & 1) * 8, sh = (g >> 1) * 4;
#pragma unroll
        for (int i = 0; i < 8; i++) v[i] = d * (float)kvalues_iq4nl[(b[2 + j0 + i] >> sh) & 0xF];
    }
};

KF_QUANT_KERNELS(Iq4Nl)

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

    static __device__ __forceinline__ void q16(const u8* b, int g, i8* q, float& d, float& m) {
        d = kf_f16(b);
        m = 0.0f;
        int sh = g * 4;
#pragma unroll
        for (int j = 0; j < 16; j++) q[j] = kvalues_iq4nl[(b[2 + j] >> sh) & 0xF];
    }
};

// Word form: the sixteen nibble bytes of the block as one load, each word through the table
template <>
struct KfQ16<Iq4Nl> {
    static __device__ __forceinline__ void load(const u8* __restrict__ b, int g, int4& wv, float& d, float& m) {
        d = kf_f16(b);
        m = 0.0f;
        int sh = g * 4;
        int4 r = kf_ld16(b + 2);
        wv.x = (int)kf_iq4nl4(((u32)r.x >> sh) & 0x0F0F0F0Fu);
        wv.y = (int)kf_iq4nl4(((u32)r.y >> sh) & 0x0F0F0F0Fu);
        wv.z = (int)kf_iq4nl4(((u32)r.z >> sh) & 0x0F0F0F0Fu);
        wv.w = (int)kf_iq4nl4(((u32)r.w >> sh) & 0x0F0F0F0Fu);
    }
};

KF_QUANT_KERNELS_I8(Iq4Nl)

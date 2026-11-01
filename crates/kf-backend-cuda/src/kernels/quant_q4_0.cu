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

    // Half g is the low nibbles for zero and the high nibbles for one, centred at eight
    static __device__ __forceinline__ void q16(const u8* b, int g, i8* q, float& d, float& m) {
        d = kf_f16(b);
        m = 0.0f;
        int sh = g * 4;
#pragma unroll
        for (int j = 0; j < 16; j++) q[j] = (i8)((int)((b[2 + j] >> sh) & 0xF) - 8);
    }
};

// Word form: the sixteen bytes of nibbles, half g selecting the low or high nibbles, eight taken off every byte
template <>
struct KfQ16<Q4_0> {
    static __device__ __forceinline__ void load(const u8* __restrict__ b, int g, int4& wv, float& d, float& m) {
        d = kf_h2f(*reinterpret_cast<const u16*>(b));
        m = 0.0f;
        int4 r = kf_ld16(b + 2);
        u32 sh = g * 4;
        wv.x = (int)__vsub4(((u32)r.x >> sh) & 0x0f0f0f0fu, 0x08080808u);
        wv.y = (int)__vsub4(((u32)r.y >> sh) & 0x0f0f0f0fu, 0x08080808u);
        wv.z = (int)__vsub4(((u32)r.z >> sh) & 0x0f0f0f0fu, 0x08080808u);
        wv.w = (int)__vsub4(((u32)r.w >> sh) & 0x0f0f0f0fu, 0x08080808u);
    }
};

KF_QUANT_KERNELS_I8(Q4_0)

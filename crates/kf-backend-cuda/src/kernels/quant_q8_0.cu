#include "quant.cuh"

struct Q8_0 {
    static constexpr int QK = 32, BYTES = 34;
    static __device__ __forceinline__ void dq8(const u8* b, int g, float* v) {
        float d = kf_f16(b);
        const u16* q = reinterpret_cast<const u16*>(b + 2 + 8 * g);
#pragma unroll
        for (int i = 0; i < 4; i++) {
            u16 p = q[i];
            v[2 * i] = (float)(i8)(p & 0xFF) * d;
            v[2 * i + 1] = (float)(i8)(p >> 8) * d;
        }
    }

    // Sixteen integers of half g with the block scale, no offset
    static __device__ __forceinline__ void q16(const u8* b, int g, i8* q, float& d, float& m) {
        d = kf_f16(b);
        m = 0.0f;
        const u8* p = b + 2 + 16 * g;
#pragma unroll
        for (int j = 0; j < 16; j++) q[j] = (i8)p[j];
    }
};

// Word form: the sixteen quants of half g straight from their two byte aligned run, the scale as one halfword
template <>
struct KfQ16<Q8_0> {
    static __device__ __forceinline__ void load(const u8* __restrict__ b, int g, int4& wv, float& d, float& m) {
        d = kf_h2f(*reinterpret_cast<const u16*>(b));
        m = 0.0f;
        wv = kf_ld16(b + 2 + 16 * g);
    }
};

// Thirty two columns per lane, the whole block: the scale and quants in nine aligned words around the two byte
// aligned block, funnel shifted into place
template <>
struct KfLane<Q8_0> {
    static constexpr int COLS = 32, NS = 1, CSTRIDE = 16;
    static __device__ __forceinline__ int base(int gg) {
        return 0;
    }
};

template <>
struct KfQw<Q8_0, 32> {
    static __device__ __forceinline__ void load(const u8* __restrict__ b, int gg, int4* wv, float* d, float* m) {
        const u64 a = (u64)b;
        const u32* w = reinterpret_cast<const u32*>(a & ~(u64)3);
        const u32 sh = (u32)(a & 3) * 8;
        u32 r[9];
#pragma unroll
        for (int i = 0; i < 9; i++) r[i] = w[i];
        u32 q[9];
#pragma unroll
        for (int i = 0; i < 9; i++) q[i] = __funnelshift_r(r[i], i + 1 < 9 ? r[i + 1] : 0u, sh);
        d[0] = kf_h2f((u16)(q[0] & 0xffffu));
        m[0] = 0.0f;
        // The quants start two bytes in, so each word is the high half of one word joined with the low half of the next
        u32 v[8];
#pragma unroll
        for (int i = 0; i < 8; i++) v[i] = __funnelshift_r(q[i], q[i + 1], 16);
        wv[0] = make_int4((int)v[0], (int)v[1], (int)v[2], (int)v[3]);
        wv[1] = make_int4((int)v[4], (int)v[5], (int)v[6], (int)v[7]);
    }
};

KF_QUANT_KERNELS_I8(Q8_0)

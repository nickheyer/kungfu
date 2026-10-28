#pragma once

typedef unsigned char u8;
typedef signed char i8;
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long long u64;
typedef long long i64;

#define KF_NEG_INF __int_as_float(0xff800000)

// Exact half precision to single conversion through the hardware instruction
__device__ __forceinline__ float kf_h2f(u16 h) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}

__device__ __forceinline__ u16 kf_u16(const u8* p) {
    return (u16)p[0] | ((u16)p[1] << 8);
}

__device__ __forceinline__ u32 kf_u32(const u8* p) {
    return (u32)p[0] | ((u32)p[1] << 8) | ((u32)p[2] << 16) | ((u32)p[3] << 24);
}

__device__ __forceinline__ float kf_f16(const u8* p) {
    return kf_h2f(kf_u16(p));
}

__device__ __forceinline__ float kf_bf16(const u8* p) {
    return __int_as_float((int)((u32)kf_u16(p) << 16));
}

// Half precision bits of a single with round to nearest even
__device__ __forceinline__ u16 kf_f2h(float v) {
    u16 h;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(v));
    return h;
}

// Round trip through half precision with round to nearest even
__device__ __forceinline__ float kf_round_f16(float v) {
    u16 h;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(v));
    return kf_h2f(h);
}

// Round trip through brain float with round to nearest even and NaN kept quiet
__device__ __forceinline__ float kf_round_bf16(float v) {
    u32 u = __float_as_uint(v);
    if ((u & 0x7fffffffu) > 0x7f800000u) return __uint_as_float((u | 0x00400000u) & 0xffff0000u);
    u += 0x7fffu + ((u >> 16) & 1u);
    return __uint_as_float(u & 0xffff0000u);
}

__device__ __forceinline__ float kf_neg(float v, bool n) {
    return n ? -v : v;
}

__device__ __forceinline__ float kf_warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float kf_warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// Sum across the block through shared scratch of 32 floats, every thread receives the total
__device__ __forceinline__ float kf_block_sum(float v, float* scratch) {
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
    v = kf_warp_sum(v);
    __syncthreads();
    if (lane == 0) scratch[warp] = v;
    __syncthreads();
    v = lane < nw ? scratch[lane] : 0.0f;
    return kf_warp_sum(v);
}

__device__ __forceinline__ float kf_block_max(float v, float* scratch) {
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
    v = kf_warp_max(v);
    __syncthreads();
    if (lane == 0) scratch[warp] = v;
    __syncthreads();
    v = lane < nw ? scratch[lane] : KF_NEG_INF;
    return kf_warp_max(v);
}

// Activation by code, silu gelu relu gelu_erf relu_sqr, matching the reference formulas
__device__ __forceinline__ float kf_act(float x, int act) {
    if (act == 0) return x / (1.0f + expf(-x));
    if (act == 1) return 0.5f * x * (1.0f + tanhf(0.7978846f * (x + 0.044715f * x * x * x)));
    if (act == 3) return 0.5f * x * (1.0f + erff(x * 0.70710678f));
    if (act == 4) { float r = fmaxf(x, 0.0f); return r * r; }
    return fmaxf(x, 0.0f);
}

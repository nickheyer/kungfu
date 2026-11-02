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

// Position stream of rotary pair i under the section rules of a rope kind, mode 0 one stream, 1 mrope, 2 imrope, 3 vision
__device__ __forceinline__ int kf_rope_stream(int i, int mode, int s0, int s1, int s2, int sect) {
    if (mode == 0) return 0;
    if (mode == 3) return i < s0 ? 0 : 1;
    int sector = i % sect;
    if (mode == 1) return sector < s0 ? 0 : (sector < s0 + s1 ? 1 : (sector < s0 + s1 + s2 ? 2 : 3));
    int t = sector % 3;
    if (t == 1 && sector < 3 * s1) return 1;
    if (t == 2 && sector < 3 * s2) return 2;
    if (t == 0 && sector < 3 * s0) return 0;
    return 3;
}

__device__ __forceinline__ u32 kf_smem_addr(const void* p) {
    return (u32)__cvta_generic_to_shared(p);
}

// Sixteen byte asynchronous copy into shared memory, zero filled when fill is false
__device__ __forceinline__ void kf_cp_async16(u32 dst, const void* src, bool fill) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    int n = fill ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(n));
#endif
}

// Sixteen byte asynchronous copy of the first n source bytes into shared memory with the rest zero filled
__device__ __forceinline__ void kf_cp_async16n(u32 dst, const void* src, int n) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(n));
#endif
}

__device__ __forceinline__ void kf_cp_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;");
#endif
}

template <int N>
__device__ __forceinline__ void kf_cp_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;" ::"n"(N));
#endif
}

__device__ __forceinline__ void kf_ldmatrix_x4(u32& r0, u32& r1, u32& r2, u32& r3, u32 addr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
#endif
}

__device__ __forceinline__ void kf_ldmatrix_x4_trans(u32& r0, u32& r1, u32& r2, u32& r3, u32 addr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
#endif
}

// c += a * b for one m16 n8 k16 half precision product with single precision accumulation
__device__ __forceinline__ void kf_mma_f16(float* c, u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

// Two singles rounded to halves and packed low then high
__device__ __forceinline__ u32 kf_pack_half2(float lo, float hi) {
    return (u32)kf_f2h(lo) | ((u32)kf_f2h(hi) << 16);
}

// Four byte asynchronous copy into shared memory, zero filled when fill is false
__device__ __forceinline__ void kf_cp4(u32 dst, const void* src, bool fill) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    int n = fill ? 4 : 0;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" ::"r"(dst), "l"(src), "r"(n));
#endif
}

__device__ __forceinline__ void kf_load8(const float* p, float* v) {
    float4 a = *reinterpret_cast<const float4*>(p);
    float4 b = *reinterpret_cast<const float4*>(p + 4);
    v[0] = a.x;
    v[1] = a.y;
    v[2] = a.z;
    v[3] = a.w;
    v[4] = b.x;
    v[5] = b.y;
    v[6] = b.z;
    v[7] = b.w;
}

// Bytes of one row of one super block in the fragment array
#define KF_XA_ROW 280

// One row of the int8 activation scratch with its tile of the fragment array behind the sums, see docs/mm-frag.md
struct KfActRow {
    i8* q;
    float* s16;
    float* m16;
    float* s32;
    float* m32;
    u8* xa;
    int xa_rt, xa_gid;
};

__device__ __forceinline__ KfActRow kf_act_row(i8* q, float* s16, float* m16, float* s32, float* m32, int row, int n, int rows) {
    KfActRow r;
    r.q = q + (i64)row * n;
    r.s16 = s16 + (i64)row * (n >> 4);
    r.m16 = m16 + (i64)row * (n >> 4);
    r.s32 = s32 + (i64)row * (n >> 5);
    r.m32 = m32 + (i64)row * (n >> 5);
    r.xa = nullptr;
    r.xa_rt = 0;
    r.xa_gid = row & 7;
    if (rows >= 2 && (n & 255) == 0) {
        const size_t base = (reinterpret_cast<size_t>(m32) + (size_t)rows * (n >> 5) * 4 + 15) & ~(size_t)15;
        const int t = row >> 3;
        r.xa = reinterpret_cast<u8*>(base) + (size_t)t * (n >> 8) * 8 * KF_XA_ROW;
        r.xa_rt = min(8, rows - 8 * t);
    }
    return r;
}

// Columns one block of the quantizing kernels covers, eight warps of 256
#define KF_QUANT_SPAN 2048

// Quantizes the eight columns of every lane in a 256 wide span starting at column c0 - lane * 8, writing the int8
// values and the block scale and integer sum of each 16 and each 32 wide group as floats
__device__ __forceinline__ void kf_quant_span(const float* v, int c0, int n, int kind, const KfActRow& o) {
    const int lane = threadIdx.x & 31;
    const bool in = c0 < n;
    float d = 0.0f;
    int qi[8];
    if (kind == 3) {
        float amax = 0.0f;
#pragma unroll
        for (int j = 0; j < 8; j++) amax = fmaxf(amax, fabsf(v[j]));
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
        float dd = amax / 127.0f;
        float id = dd != 0.0f ? 1.0f / dd : 0.0f;
        d = kf_round_f16(dd);
#pragma unroll
        for (int j = 0; j < 8; j++) qi[j] = (int)roundf(v[j] * id);
    } else {
        float a = -1.0f, mx = 0.0f;
        int idx = 0x7fffffff;
#pragma unroll
        for (int j = 0; j < 8; j++) {
            float aj = in ? fabsf(v[j]) : -1.0f;
            if (aj > a) {
                a = aj;
                idx = c0 + j;
                mx = v[j];
            }
        }
        for (int o = 16; o > 0; o >>= 1) {
            float oa = __shfl_xor_sync(0xffffffffu, a, o);
            int oi = __shfl_xor_sync(0xffffffffu, idx, o);
            float om = __shfl_xor_sync(0xffffffffu, mx, o);
            if (oa > a || (oa == a && oi < idx)) {
                a = oa;
                idx = oi;
                mx = om;
            }
        }
        if (a > 0.0f) {
            float iscale = -127.0f / mx;
            d = 1.0f / iscale;
#pragma unroll
            for (int j = 0; j < 8; j++) qi[j] = (int)fminf(rintf(iscale * v[j]), 127.0f);
        } else {
#pragma unroll
            for (int j = 0; j < 8; j++) qi[j] = 0;
        }
    }
    int sum = 0;
#pragma unroll
    for (int j = 0; j < 8; j++) sum += qi[j];
    // Two lanes make a group of 16, four a group of 32
    const int sum16 = sum + __shfl_xor_sync(0xffffffffu, sum, 1);
    const int sum32 = sum16 + __shfl_xor_sync(0xffffffffu, sum16, 2);
    if (in) {
        u32 lo = (u32)(qi[0] & 0xff) | ((u32)(qi[1] & 0xff) << 8) | ((u32)(qi[2] & 0xff) << 16) | ((u32)(qi[3] & 0xff) << 24);
        u32 hi = (u32)(qi[4] & 0xff) | ((u32)(qi[5] & 0xff) << 8) | ((u32)(qi[6] & 0xff) << 16) | ((u32)(qi[7] & 0xff) << 24);
        *reinterpret_cast<uint2*>(o.q + c0) = make_uint2(lo, hi);
        if (o.xa) {
            const int sb = c0 >> 8, g = (c0 >> 5) & 7, tig = (c0 >> 3) & 3;
            u8* blk = o.xa + (size_t)sb * o.xa_rt * KF_XA_ROW;
            *reinterpret_cast<uint2*>(blk + (g * o.xa_rt + o.xa_gid) * 32 + tig * 8) = make_uint2(lo, hi);
            if ((lane & 3) == 0) {
                u8* sums = blk + o.xa_rt * 256 + o.xa_gid * 16;
                sums[g] = (u8)(sum32 & 0xff);
                sums[8 + g] = (u8)((sum32 >> 8) & 0xff);
            }
            if ((c0 & 255) == 0) *reinterpret_cast<float*>(blk + o.xa_rt * 272 + o.xa_gid * 4) = d;
        }
        if ((lane & 1) == 0) {
            o.s16[c0 >> 4] = d;
            o.m16[c0 >> 4] = (float)sum16;
        }
        if ((lane & 3) == 0) {
            o.s32[c0 >> 5] = d;
            o.m32[c0 >> 5] = (float)sum32;
        }
    }
}


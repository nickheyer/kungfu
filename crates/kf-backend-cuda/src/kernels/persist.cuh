#pragma once
// One launch for a whole single row decode pass. A cooperative grid walks a list of stages, each standing for one
// kernel launch of the pass with that launch's arguments laid out as C, every block taking the stage's work items
// blockIdx.x, blockIdx.x + gridDim.x, and so on. Every block counts itself into a stage once its items are done, and
// a stage waits only for the latest stage whose buffers it touches, so stages that share no buffers run back to
// back without a barrier. A product stage issues the weight loads of its first item before that wait, so the
// weights stream while the activations they multiply are still being written, and a product whose activation is a
// norm or gated activation of the stage before computes it itself into shared memory from the f32 rows. Loads of
// tensors written earlier in the pass bypass the incoherent L1. The product bodies are separate functions so the
// attention and they are allocated registers apart.
//
// The composing host defines KF_PERSIST_MV_TYPES(X) and KF_PERSIST_EMBED_TYPES(X) as X(index, Struct) lists of the
// quant types the pass multiplies and embeds from, and KF_PERSIST_HDS(X) as X(hd) of the head sizes it attends over.
//
// The stage table and every stage's arguments live in the constant bank of the module, the headers first and the
// argument structs behind them at their byte offsets, so the bodies read their arguments as operands the way a kernel
// reads its parameters instead of holding them in registers.
#include "quant.cuh"
#include "attention_decode.cuh"

#define KF_PS_THREADS 256
#ifndef KF_PS_BLOCKS
#define KF_PS_BLOCKS 2
#endif
#define KF_PS_DATA 61440

enum { KF_PS_INPUT = 0, KF_PS_EMBED = 1, KF_PS_RMS_Q8 = 2, KF_PS_MATVEC = 3, KF_PS_ATTENTION = 4, KF_PS_GATED_Q8 = 5 };

struct KfStage {
    int kind, sel, items, gx, gy, gz, smem, argoff, dep;
};

__constant__ __align__(16) u8 kf_ps_data[KF_PS_DATA];

struct KfPsInput {
    const int* src;
    float* out;
    int n;
};

struct KfPsEmbed {
    const u8* w;
    const int* tokens;
    float* out;
    int n, n_tok;
    i64 row_bytes;
};

struct KfPsRms {
    const float* x;
    const float* w;
    const float* b;
    int n;
    float eps;
    int kind;
    i8* q;
    float* s16;
    float* m16;
    float* s32;
    float* m32;
};

struct KfPsGated {
    const float* gate;
    const float* up;
    int n, stride, act, kind;
    i8* q;
    float* s16;
    float* m16;
    float* s32;
    float* m32;
};

struct KfPsMv {
    const u8* w;
    const i8* xq;
    const float* xd16;
    const float* xs16;
    const float* xd32;
    const float* xs32;
    const float* bias;
    const float* res;
    float* y;
    int rows, n, n_out;
    i64 row_bytes;
    const long long* segs;
    int nseg, wpr;
};

struct KfPsAttn {
    const float* q;
    u16* kc;
    u16* vc;
    const int* pos;
    const int* seq;
    const int* attend;
    const float* sinks;
    float* out;
    float* parts;
    int n_head, n_head_kv;
    float scale;
    int window;
    float softcap;
    int cap, rows;
    int* counters;
    int kind;
    i8* qo;
    float* s16o;
    float* m16o;
    float* s32o;
    float* m32o;
    const float* k_in;
    const float* v_in;
    const int* rope4;
    const float2* cst;
    int half, neox;
    float mscale;
    int mode, s0, s1, s2, sect, norm_q;
    const float* wq;
    const float* bq;
    float eps_q;
    int norm_k;
    const float* wk;
    const float* bk;
    float eps_k;
    const int* tree;
};

__device__ __forceinline__ int kf_ps_ld_acquire(const int* p) {
    int v;
    asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ int kf_ps_ld_relaxed(const int* p) {
    int v;
    asm volatile("ld.relaxed.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ void kf_ps_fence_acquire() {
    asm volatile("fence.acq_rel.gpu;" ::: "memory");
}

__device__ __forceinline__ void kf_ps_red_release(int* p, int v) {
    asm volatile("red.release.gpu.global.add.s32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

// Holds the block until every block has arrived at stage s, the counts monotonic over passes and target the count of
// this pass. The poll is a relaxed load, an acquire load invalidates the L1 of the SM on every turn, and one fence
// after it orders the stage's reads.
__device__ __forceinline__ void kf_ps_wait(int* cnt, int s, unsigned target) {
    if (s < 0) return;
    if (threadIdx.x == 0) {
        while ((int)((unsigned)kf_ps_ld_relaxed(cnt + s) - target) < 0) {
#ifdef KF_PS_SLEEP
            __nanosleep(KF_PS_SLEEP);
#endif
        }
        kf_ps_fence_acquire();
    }
    __syncthreads();
}

// Counts the block into stage s once every thread's writes of the stage are issued
__device__ __forceinline__ void kf_ps_arrive(int* cnt, int s) {
    __syncthreads();
    if (threadIdx.x == 0) kf_ps_red_release(cnt + s, 1);
}

__device__ __forceinline__ i64 kf_ps_now() {
    i64 t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__device__ __forceinline__ void kf_ps_ldcg8(const float* p, float* v) {
    float4 a = __ldcg(reinterpret_cast<const float4*>(p));
    float4 b = __ldcg(reinterpret_cast<const float4*>(p + 4));
    v[0] = a.x;
    v[1] = a.y;
    v[2] = a.z;
    v[3] = a.w;
    v[4] = b.x;
    v[5] = b.y;
    v[6] = b.z;
    v[7] = b.w;
}

// Item bx of the token or position input, written as floats
__device__ __forceinline__ void kf_ps_input(const KfPsInput& a, int bx, int gx) {
    for (int i = bx * KF_PS_THREADS + threadIdx.x; i < a.n; i += gx * KF_PS_THREADS) a.out[i] = (float)(u32)a.src[i];
}

// Span bx of row row normed with the row's weight and bias and quantized, the row summed by every block of it
__device__ __noinline__ void kf_ps_rms_q8(const KfPsRms& a, int row, int spanb, float* scratch) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int span = spanb * KF_QUANT_SPAN + warp * 256;
    const float* xr = a.x + (i64)row * a.n;
    float s = 0.0f;
    for (int i = threadIdx.x; i < a.n; i += KF_PS_THREADS) {
        const float v = __ldcg(xr + i);
        s += v * v;
    }
    s = kf_block_sum(s, scratch);
    if (span >= a.n) return;
    const float inv = 1.0f / sqrtf(s / (float)a.n + a.eps);
    const KfActRow o = kf_act_row(a.q, a.s16, a.m16, a.s32, a.m32, row, a.n, 0);
    const int c0 = span + lane * 8;
    float v[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    if (c0 < a.n) {
        kf_ps_ldcg8(xr + c0, v);
        float ww[8] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
        float bb[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        if (a.w) kf_load8(a.w + c0, ww);
        if (a.b) kf_load8(a.b + c0, bb);
#pragma unroll
        for (int j = 0; j < 8; j++) v[j] = v[j] * (inv * ww[j]) + bb[j];
    }
    kf_quant_span(v, c0, a.n, a.kind, o);
}

// Span spanb of row row of the gate activation times the up rows, quantized
__device__ __noinline__ void kf_ps_gated_q8(const KfPsGated& a, int row, int spanb) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int span = spanb * KF_QUANT_SPAN + warp * 256;
    if (span >= a.n) return;
    const KfActRow o = kf_act_row(a.q, a.s16, a.m16, a.s32, a.m32, row, a.n, 0);
    const int c0 = span + lane * 8;
    float v[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    if (c0 < a.n) {
        float gg[8], uu[8];
        kf_ps_ldcg8(a.gate + (i64)row * a.stride + c0, gg);
        kf_ps_ldcg8(a.up + (i64)row * a.stride + c0, uu);
#pragma unroll
        for (int j = 0; j < 8; j++) v[j] = kf_act(gg[j], a.act) * uu[j];
    }
    kf_quant_span(v, c0, a.n, a.kind, o);
}

// The output row of a product item for this warp's row group, with the member of the segment table it falls in
struct KfPsRow {
    const u8* row;
    float* y;
    const float* bias;
    const float* res;
    int o;
    bool live;

    __device__ __forceinline__ void resolve(const KfPsMv& a, int bx, int rpb, int rw) {
        const u8* w = a.w;
        int n_out = a.n_out;
        y = a.y;
        bias = a.bias;
        res = a.res;
        o = bx * rpb + rw;
        live = true;
        if (a.segs) {
            const long long* e = a.segs;
            int i = 0;
            for (; i < a.nseg; i++, e += KF_SEG_WORDS) {
                if (o >= (int)e[4] && o < (int)(e[4] + e[5])) break;
            }
            if (i == a.nseg) {
                live = false;
                e = a.segs;
            }
            u8* yb = reinterpret_cast<u8*>(a.y);
            w += e[0];
            y = reinterpret_cast<float*>(yb + e[1]);
            bias = e[2] == KF_SEG_NONE ? nullptr : reinterpret_cast<const float*>(yb + e[2]);
            res = e[3] == KF_SEG_NONE ? nullptr : reinterpret_cast<const float*>(yb + e[3]);
            o -= (int)e[4];
            n_out = (int)e[5];
        }
        live = live && o < n_out;
        row = w + (i64)(live ? o : 0) * a.row_bytes;
    }

    // The warps of the row joined through part and the row written by the first of them
    __device__ __forceinline__ void finish(float acc, float* part, int wpr, int warp, int split, int lane) const {
        acc = kf_warp_sum(acc);
        if (wpr > 1) {
            if (split > 0 && lane == 0) part[warp] = acc;
            __syncthreads();
            if (split == 0) {
                for (int s = 1; s < wpr; s++) acc += part[warp + s];
            }
        }
        if (live && split == 0 && lane == 0) y[o] = acc + (bias ? bias[o] : 0.0f) + (res ? __ldcg(res + o) : 0.0f);
    }
};

enum { KF_PS_PRE_NONE = 0, KF_PS_PRE_RMS = 1, KF_PS_PRE_GATED = 2 };

// The prologue word and its argument struct follow a product's own arguments
#define KF_PS_PRE_OFF (sizeof(KfPsMv))
#define KF_PS_PRE_ARGS (sizeof(KfPsMv) + 8)

// Bytes of shared memory the staged activation of a product of n columns takes, the integers then the scales and
// sums per 16 and per 32
__device__ __forceinline__ int kf_ps_act_bytes(int n) {
    return n + (n >> 4) * 8 + (n >> 5) * 8;
}

// The activation row of a product computed into shared memory from the f32 rows the stage before wrote, the norm of
// the residual row or the gated activation of the gate and up rows, quantized as the product's type takes it, every
// block of the stage doing the same work for itself instead of waiting on one block's epilogue
template <typename T>
__device__ __forceinline__ void kf_ps_stage_pre(const KfPsMv& a, int pre, const u8* __restrict__ pre_args, u8* smem, float* red, const float*& xd_s, const float*& xs_s) {
    constexpr int GSH = KfLaneWords<T>::GSH;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int n = a.n;
    KfActRow o;
    o.q = reinterpret_cast<i8*>(smem);
    o.s16 = reinterpret_cast<float*>(smem + n);
    o.m16 = o.s16 + (n >> 4);
    o.s32 = o.m16 + (n >> 4);
    o.m32 = o.s32 + (n >> 5);
    o.xa = nullptr;
    o.xa_rt = o.xa_gid = 0;
    if (pre == KF_PS_PRE_RMS) {
        const KfPsRms& r = *reinterpret_cast<const KfPsRms*>(pre_args);
        float s = 0.0f;
        for (int i = tid; i < n; i += KF_PS_THREADS) {
            const float v = __ldcg(r.x + i);
            s += v * v;
        }
        s = kf_block_sum(s, red);
        const float inv = 1.0f / sqrtf(s / (float)n + r.eps);
        for (int span = warp * 256; span < n; span += KF_PS_THREADS * 8) {
            const int c0 = span + lane * 8;
            float v[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
            if (c0 < n) {
                kf_ps_ldcg8(r.x + c0, v);
                float ww[8] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
                float bb[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
                if (r.w) kf_load8(r.w + c0, ww);
                if (r.b) kf_load8(r.b + c0, bb);
#pragma unroll
                for (int j = 0; j < 8; j++) v[j] = v[j] * (inv * ww[j]) + bb[j];
            }
            kf_quant_span(v, c0, n, r.kind, o);
        }
    } else {
        const KfPsGated& g = *reinterpret_cast<const KfPsGated*>(pre_args);
        for (int span = warp * 256; span < n; span += KF_PS_THREADS * 8) {
            const int c0 = span + lane * 8;
            float v[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
            if (c0 < n) {
                float gg[8], uu[8];
                kf_ps_ldcg8(g.gate + c0, gg);
                kf_ps_ldcg8(g.up + c0, uu);
#pragma unroll
                for (int j = 0; j < 8; j++) v[j] = kf_act(gg[j], g.act) * uu[j];
            }
            kf_quant_span(v, c0, n, g.kind, o);
        }
    }
    __syncthreads();
    xd_s = GSH == 5 ? o.s32 : o.s16;
    xs_s = GSH == 5 ? o.m32 : o.m16;
}

// A block fills warm_bytes of shared memory with the rows of its coming items before it waits on the stage before,
// so the memory bus stays busy while the barrier and the small stages before a product resolve
__device__ __forceinline__ void kf_ps_prefetch_l2(const void* p) {
    asm volatile("prefetch.global.L2 [%0];" ::"l"(p));
}

// Items past the first whose rows the warm buffer holds, each item rpb rows of stride bytes
__device__ __forceinline__ int kf_ps_warm_depth(int my_items, int rpb, i64 stride, int warm_bytes) {
    const i64 per_item = (i64)rpb * stride;
    return (int)min((i64)my_items - 1, per_item > 0 ? (i64)warm_bytes / per_item : 0);
}

// Row rw of item k of the block in the warm buffer
__device__ __forceinline__ u8* kf_ps_warm_row(u8* warm, int k, int rpb, int rw, i64 stride) {
    return warm + ((i64)(k - 1) * rpb + rw) * stride;
}

// Copies a row of row_bytes into the warm buffer, the warps of the row's group taking interleaved sixteen byte
// chunks, through aligned loads for rows a multiple of sixteen bytes and funnel shifted loads otherwise with the
// odd tail in halfwords
__device__ __forceinline__ void kf_ps_warm_copy(const u8* __restrict__ src, u8* dst, i64 row_bytes, int wpr, int split, int lane) {
    const i64 chunks = row_bytes >> 4;
    const bool aligned = (row_bytes & 15) == 0;
    for (i64 c = split * 32 + lane; c < chunks; c += wpr * 32) {
        const int4 v = aligned ? *reinterpret_cast<const int4*>(src + c * 16) : kf_ld16(src + c * 16);
        *reinterpret_cast<int4*>(dst + c * 16) = v;
    }
    if (!aligned && split == 0) {
        for (i64 b = chunks * 16 + lane * 2; b < row_bytes; b += 64) *reinterpret_cast<u16*>(dst + b) = *reinterpret_cast<const u16*>(src + b);
    }
}

// Fills the warm buffer with the rows of the block's items one through depth
__device__ __forceinline__ void kf_ps_warm_fill(const KfPsMv& a, u8* warm, int depth, int rpb, int rw, int wpr, int split, int lane, i64 stride) {
    const int g = gridDim.x;
    for (int k = 1; k <= depth; k++) {
        KfPsRow r;
        r.resolve(a, blockIdx.x + k * g, rpb, rw);
        if (r.live) kf_ps_warm_copy(r.row, kf_ps_warm_row(warm, k, rpb, rw, stride), a.row_bytes, wpr, split, lane);
    }
}

// The activation row of a product copied from the quantized scratch into shared memory as int8 with the scales and
// sums the type's lane run reads
template <typename T>
__device__ __forceinline__ void kf_ps_stage_act(const KfPsMv& a, u8* smem, const float*& xd_s, const float*& xs_s) {
    constexpr int GSH = KfLaneWords<T>::GSH;
    const int tid = threadIdx.x;
    const int n = a.n, ng = n >> GSH;
    const float* xd = GSH == 5 ? a.xd32 : a.xd16;
    const float* xs = GSH == 5 ? a.xs32 : a.xs16;
    int4* sq = reinterpret_cast<int4*>(smem);
    int4* sd = reinterpret_cast<int4*>(smem + n);
    int4* ss = sd + (ng >> 2);
    for (int k = tid; k < (n >> 4); k += KF_PS_THREADS) sq[k] = __ldcg(reinterpret_cast<const int4*>(a.xq) + k);
    for (int k = tid; k < (ng >> 2); k += KF_PS_THREADS) {
        sd[k] = __ldcg(reinterpret_cast<const int4*>(xd) + k);
        ss[k] = __ldcg(reinterpret_cast<const int4*>(xs) + k);
    }
    __syncthreads();
    xd_s = reinterpret_cast<const float*>(smem + n);
    xs_s = xd_s + ng;
}

// The block index of slot j of item k of this block in the row's steps, minus one past the row's steps or for a dead row
template <typename T>
__device__ __forceinline__ int kf_ps_mv_blk(const KfPsMv& a, int bx, int step, int nsteps, int nblk, int bl, int rpb, int rw, const u8*& row) {
    constexpr int BPW = 32 / (T::QK / KfLane<T>::COLS);
    KfPsRow r;
    r.resolve(a, bx, rpb, rw);
    row = r.row;
    const int b = step * BPW + bl;
    return r.live && step < nsteps && b < nblk ? b : -1;
}

// The lane's words of block blk of a row, zero for a block index below zero
template <typename T>
__device__ __forceinline__ KfLaneWords<T> kf_ps_mv_words(const u8* row, int blk, int gg) {
    KfLaneWords<T> w;
    if (blk >= 0) w.load(row + (i64)blk * T::BYTES, gg);
    else w.zero();
    return w;
}

// Every item of a product stage. The warp's work is the flat sequence of its steps over every item of the block,
// the words of the step after the one multiplied always in flight, so the first step's weights load before the wait
// on the stage this one depends on and an item's last step overlaps the next item's first. The activation row lands
// in shared memory once per stage, copied from the quantized scratch or computed by the prologue pre names, see
// kf_ps_stage_pre. Every warp of a block takes the same number of slots per item, a slot past the row's steps
// loading nothing, so the joins of the warps line up. The two slots alternate by position so each stays in registers.
template <typename T>
__device__ __noinline__ void kf_ps_matvec(const KfPsMv& a, int pre, const u8* __restrict__ pre_args, int items, int* cnt, int dep, int s, unsigned target, u8* smem, u8* warm, int warm_bytes, float* part, float* red, i64* trace) {
    constexpr int LPB = T::QK / KfLane<T>::COLS, BPW = 32 / LPB;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = gridDim.x;
    const int wpr = a.wpr;
    const int wsh = wpr == 8 ? 3 : (wpr == 4 ? 2 : (wpr == 2 ? 1 : 0));
    const int rpb = 8 >> wsh, rw = warp >> wsh, split = warp - (rw << wsh);
    const int n = a.n, nblk = n / T::QK, nsteps = (nblk + BPW - 1) / BPW;
    const int slots = (nsteps + wpr - 1) / wpr;
    const int bl = lane / LPB, gg = lane - bl * LPB, kb = KfLane<T>::base(gg);
    const int my_items = (int)blockIdx.x < items ? (items - (int)blockIdx.x + g - 1) / g : 0;
    const u8* row = nullptr;
    int b0 = my_items > 0 ? kf_ps_mv_blk<T>(a, blockIdx.x, split, nsteps, nblk, bl, rpb, rw, row) : -1;
    KfLaneWords<T> w0 = kf_ps_mv_words<T>(row, b0, gg);
    // Rows in shared memory keep their stride a multiple of sixteen
    const i64 stride = (a.row_bytes + 15) & ~(i64)15;
    const int depth = kf_ps_warm_depth(my_items, rpb, stride, warm_bytes);
    kf_ps_warm_fill(a, warm, depth, rpb, rw, wpr, split, lane, stride);
    if (pre == KF_PS_PRE_RMS && blockIdx.x == 0) {
        // The norm's weight and bias come to the L2 for every block's prologue
        const KfPsRms& r = *reinterpret_cast<const KfPsRms*>(pre_args);
        for (int off = tid * 128; off < r.n * 4; off += KF_PS_THREADS * 128) {
            if (r.w) kf_ps_prefetch_l2(reinterpret_cast<const u8*>(r.w) + off);
            if (r.b) kf_ps_prefetch_l2(reinterpret_cast<const u8*>(r.b) + off);
        }
    }
    kf_ps_wait(cnt, dep, target);
    if (trace && tid == 0) trace[(3 * s + 1) * g + blockIdx.x] = kf_ps_now();
    if (my_items > 0) {
        const i8* xq_s = reinterpret_cast<const i8*>(smem);
        const float* xd_s;
        const float* xs_s;
        if (pre == KF_PS_PRE_NONE) {
            kf_ps_stage_act<T>(a, smem, xd_s, xs_s);
        } else {
            kf_ps_stage_pre<T>(a, pre, pre_args, smem, red, xd_s, xs_s);
        }
        if (trace && tid == 0) trace[(3 * s + 2) * g + blockIdx.x] = kf_ps_now();
        float acc = 0.0f;
        // k and j the slot multiplied, kn and jn the slot in flight
        int k = 0, j = 0, kn = 0, jn = 0;
        for (;;) {
            if (++jn == slots) {
                jn = 0;
                kn++;
            }
            const int b1 = kn < my_items ? kf_ps_mv_blk<T>(a, blockIdx.x + kn * g, split + jn * wpr, nsteps, nblk, bl, rpb, rw, row) : -1;
            if (kn >= 1 && kn <= depth) row = kf_ps_warm_row(warm, kn, rpb, rw, stride);
            const KfLaneWords<T> w1 = kf_ps_mv_words<T>(row, b1, gg);
            if (b0 >= 0) w0.dot(xq_s, xd_s, xs_s, b0 * T::QK + kb, acc);
            if (j == slots - 1) {
                KfPsRow r;
                r.resolve(a, blockIdx.x + k * g, rpb, rw);
                r.finish(acc, part + 8 * (k & 1), wpr, warp, split, lane);
                acc = 0.0f;
            }
            k = kn;
            j = jn;
            if (k >= my_items) break;
            if (++jn == slots) {
                jn = 0;
                kn++;
            }
            b0 = kn < my_items ? kf_ps_mv_blk<T>(a, blockIdx.x + kn * g, split + jn * wpr, nsteps, nblk, bl, rpb, rw, row) : -1;
            if (kn >= 1 && kn <= depth) row = kf_ps_warm_row(warm, kn, rpb, rw, stride);
            w0 = kf_ps_mv_words<T>(row, b0, gg);
            if (b1 >= 0) w1.dot(xq_s, xd_s, xs_s, b1 * T::QK + kb, acc);
            if (j == slots - 1) {
                KfPsRow r;
                r.resolve(a, blockIdx.x + k * g, rpb, rw);
                r.finish(acc, part + 8 * (k & 1), wpr, warp, split, lane);
                acc = 0.0f;
            }
            k = kn;
            j = jn;
            if (k >= my_items) break;
        }
    }
    if (trace && tid == 0) trace[(3 * s + 3) * g + blockIdx.x] = kf_ps_now();
}

#ifndef KF_PS_PFK
#define KF_PS_PFK 2
#endif
#ifndef KF_PS_PFV
#define KF_PS_PFV 4
#endif
// Query heads per attention block, the split of the standalone kernel
#ifndef KF_PS_GMAX
#define KF_PS_GMAX 4
#endif

template <int HD>
__device__ __noinline__ void kf_ps_attention(const KfPsAttn& a, int bx, int by, int bz, int gy, int gz) {
    kf_attention_decode<HD, KF_PS_PFK, KF_PS_PFV, KF_PS_GMAX>(bx, by, bz, gy, gz, a.q, a.kc, a.vc, a.pos, a.seq, a.attend, a.sinks, a.out, a.parts, a.n_head, a.n_head_kv, a.scale, a.window, a.softcap, a.cap, a.rows, a.counters, a.kind, a.qo, a.s16o, a.m16o, a.s32o, a.m32o, a.k_in, a.v_in, a.rope4, a.cst, a.half, a.neox, a.mscale, a.mode, a.s0, a.s1, a.s2, a.sect, a.norm_q, a.wq, a.bq, a.eps_q, a.norm_k, a.wk, a.bk, a.eps_k, a.tree);
}

#define KF_PS_MV_CASE(I, T) \
    case I: \
        kf_ps_matvec<T>(*reinterpret_cast<const KfPsMv*>(kf_ps_data + argoff), *reinterpret_cast<const int*>(kf_ps_data + argoff + KF_PS_PRE_OFF), kf_ps_data + argoff + KF_PS_PRE_ARGS, items, cnt, dep, s, target, kf_ps_smem, \
                        kf_ps_smem + warm_off, warm_bytes, part, scratch, trace); \
        break;
#define KF_PS_EMBED_CASE(I, T) \
    case I: kf_embed_part<T>(e.w, e.tokens, e.out, e.n, e.n_tok, e.row_bytes, bx * (i64)KF_PS_THREADS + threadIdx.x, (i64)gx * KF_PS_THREADS); break;
#define KF_PS_ATTN_CASE(HD) \
    case HD: kf_ps_attention<HD>(*reinterpret_cast<const KfPsAttn*>(kf_ps_data + argoff), bx, by, bz, gy, gz); break;

// The stages of one pass in order, see the header comment, the warm buffer of warm_bytes at warm_off bytes into the
// dynamic shared memory. With trace given, every block stamps the nanosecond clock
// into it, a row of blocks per stamp, the start first, then for stage s rows 3s + 1 to 3s + 3 as a product leaves its
// wait, finishes staging, and finishes its items, then from row 3 nstages + 1 the arrival at every stage.
extern "C" __global__ void __launch_bounds__(KF_PS_THREADS, KF_PS_BLOCKS) persist(int nstages, int warm_off, int warm_bytes, int* cnt, int* gen_ptr, i64* trace) {
    __shared__ float scratch[32];
    __shared__ float part[16];
    extern __shared__ __align__(16) u8 kf_ps_smem[];
    const unsigned g = gridDim.x;
    const unsigned gen = (unsigned)kf_ps_ld_acquire(gen_ptr);
    const unsigned target = (gen + 1u) * g;
    if (trace && threadIdx.x == 0) trace[blockIdx.x] = kf_ps_now();
    for (int s = 0; s < nstages; s++) {
        const KfStage& st = *reinterpret_cast<const KfStage*>(kf_ps_data + sizeof(KfStage) * s);
        const int kind = st.kind, sel = st.sel, items = st.items, gx = st.gx, gy = st.gy, gz = st.gz, argoff = st.argoff, dep = st.dep;
        if (kind == KF_PS_MATVEC) {
            switch (sel) { KF_PERSIST_MV_TYPES(KF_PS_MV_CASE) }
        } else {
            kf_ps_wait(cnt, dep, target);
            if (trace && threadIdx.x == 0) trace[(3 * s + 1) * g + blockIdx.x] = kf_ps_now();
            for (int item = blockIdx.x; item < items; item += g) {
                const int bx = item % gx, by = (item / gx) % gy, bz = item / (gx * gy);
                switch (kind) {
                    case KF_PS_INPUT: kf_ps_input(*reinterpret_cast<const KfPsInput*>(kf_ps_data + argoff), bx, gx); break;
                    case KF_PS_EMBED: {
                        const KfPsEmbed& e = *reinterpret_cast<const KfPsEmbed*>(kf_ps_data + argoff);
                        switch (sel) { KF_PERSIST_EMBED_TYPES(KF_PS_EMBED_CASE) }
                        break;
                    }
                    case KF_PS_RMS_Q8: kf_ps_rms_q8(*reinterpret_cast<const KfPsRms*>(kf_ps_data + argoff), bx, by, scratch); break;
                    case KF_PS_GATED_Q8: kf_ps_gated_q8(*reinterpret_cast<const KfPsGated*>(kf_ps_data + argoff), bx, by); break;
                    case KF_PS_ATTENTION:
                        switch (sel) { KF_PERSIST_HDS(KF_PS_ATTN_CASE) }
                        break;
                }
            }
        }
        kf_ps_arrive(cnt, s);
        if (trace && threadIdx.x == 0) trace[(3 * nstages + 1 + s) * g + blockIdx.x] = kf_ps_now();
    }
    // Every block has read the generation once the last stage is complete, so the next pass counts one round higher
    if (blockIdx.x == 0) {
        kf_ps_wait(cnt, nstages - 1, target);
        if (threadIdx.x == 0) *gen_ptr = (int)(gen + 1u);
    }
}

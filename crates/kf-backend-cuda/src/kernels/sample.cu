#include "common.cuh"

// The k largest values of every row with their ids, largest first and lower id first among equals, then the row's
// log sum exp, written as k ids, k values, and the log sum exp. A row is cut into slices of about eight thousand
// columns, one block each, and every slice finds its own k largest exactly by a radix select over a composite key of
// the value's order preserving bits above the complemented id, so the key is unique and no tie needs a rule of its
// own. The slice's first pass finds its maximum and sum of exponentials and the bits every key shares at the top, so
// the digits start where the keys part and spread over the bins instead of crowding into the few a row of logits
// fills at the top. Each pass takes eight bits of the key into a histogram private to every pair of warps. Once the
// keys that can still hold the k th largest fit the gather space they come into shared memory and the remaining
// digits resolve there, until exactly k keys stand above the prefix. The merge ranks the slices' candidates, which
// hold the row's k largest, the same way and joins the slices' sums.
#define KF_TOPK_THREADS 1024
#define KF_TOPK_WARPS (KF_TOPK_THREADS / 32)
#define KF_TOPK_COPIES (KF_TOPK_WARPS / 2)
#define KF_TOPK_BITS 8
#define KF_TOPK_BINS (1 << KF_TOPK_BITS)
#define KF_TOPK_MAX 128
#define KF_TOPK_SLICES 16
#define KF_TOPK_GATHER 2048
static_assert(KF_TOPK_MAX * KF_TOPK_SLICES <= KF_TOPK_GATHER, "the merge holds every slice's candidates in the gather space");

typedef unsigned long long u64;

// Calls f with the index and value of every element of the span the thread owns, four at a time through sixteen byte
// loads when the span allows, so several loads are in flight per thread
template <typename F>
__device__ __forceinline__ void kf_topk_each(const float* __restrict__ span, int len, F&& f) {
    const int tid = threadIdx.x;
    if ((reinterpret_cast<size_t>(span) & 15) == 0) {
        const float4* r4 = reinterpret_cast<const float4*>(span);
        const int n4 = len >> 2;
        for (int j = tid; j < n4; j += KF_TOPK_THREADS) {
            const float4 v = r4[j];
            f(4 * j, v.x);
            f(4 * j + 1, v.y);
            f(4 * j + 2, v.z);
            f(4 * j + 3, v.w);
        }
        for (int i = 4 * n4 + tid; i < len; i += KF_TOPK_THREADS) f(i, span[i]);
    } else {
        for (int i = tid; i < len; i += KF_TOPK_THREADS) f(i, span[i]);
    }
}

__device__ __forceinline__ u32 kf_topk_fkey(float f) {
    const u32 u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__device__ __forceinline__ float kf_topk_unkey(u32 key) {
    return __uint_as_float((key & 0x80000000u) ? (key ^ 0x80000000u) : ~key);
}

// Inclusive prefix sum over the block of one int per thread, the block total in every thread
__device__ __forceinline__ int kf_topk_scan(int v, int* red, int& total) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int n = __shfl_up_sync(0xffffffffu, v, o);
        if (lane >= o) v += n;
    }
    if (lane == 31) red[warp] = v;
    __syncthreads();
    if (warp == 0) {
        int w = red[lane];
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int n = __shfl_up_sync(0xffffffffu, w, o);
            if (lane >= o) w += n;
        }
        red[lane] = w;
    }
    __syncthreads();
    total = red[KF_TOPK_WARPS - 1];
    const int before = warp > 0 ? red[warp - 1] : 0;
    __syncthreads();
    return v + before;
}

// Bitwise and and or of one word per lane across the warp
__device__ __forceinline__ void kf_topk_warp_bits(u64& all, u64& any) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        all &= __shfl_xor_sync(0xffffffffu, all, o);
        any |= __shfl_xor_sync(0xffffffffu, any, o);
    }
}

// Bitwise and and or of one word per thread across the block, every thread receiving both
__device__ __forceinline__ void kf_topk_block_bits(u64& all, u64& any, u64* scratch) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    kf_topk_warp_bits(all, any);
    __syncthreads();
    if (lane == 0) {
        scratch[warp] = all;
        scratch[KF_TOPK_WARPS + warp] = any;
    }
    __syncthreads();
    all = scratch[lane];
    any = scratch[KF_TOPK_WARPS + lane];
    kf_topk_warp_bits(all, any);
}

// The shared state of a radix select in one block
struct KfTopkState {
    int hist[KF_TOPK_COPIES][KF_TOPK_BINS];
    int red[32];
    int sel_digit, sel_above, sel_count, slot;
    u64 bits[2 * KF_TOPK_WARPS];
    u64 keys[KF_TOPK_GATHER];
    u64 top[KF_TOPK_MAX];
};

// One digit of the select below bit hi, count adding every key that shares the prefix into its bin, the prefix and
// the count of keys above it extended, the keys at or above the new prefix returned
template <typename Count>
__device__ __forceinline__ int kf_topk_pass(KfTopkState& st, int hi, int width, int k, u64& prefix, int& above, Count&& count) {
    const int tid = threadIdx.x;
    for (int b = tid; b < KF_TOPK_COPIES * KF_TOPK_BINS; b += KF_TOPK_THREADS) (&st.hist[0][0])[b] = 0;
    __syncthreads();
    count();
    __syncthreads();
    int c = 0;
    if (tid < KF_TOPK_BINS) {
#pragma unroll
        for (int w = 0; w < KF_TOPK_COPIES; w++) c += st.hist[w][tid];
    }
    int total;
    const int incl = kf_topk_scan(c, st.red, total);
    const int need = k - above;
    if (tid < KF_TOPK_BINS && total - incl < need && total - incl + c >= need) {
        st.sel_digit = tid;
        st.sel_above = total - incl;
        st.sel_count = c;
    }
    __syncthreads();
    prefix = (prefix << width) | (u64)st.sel_digit;
    above += st.sel_above;
    const int survivors = above + st.sel_count;
    __syncthreads();
    return survivors;
}

// The k largest of the n keys in shared memory, of which survivors stand at or above prefix from bit hi with above
// keys higher, their digits resolved until exactly k stand, then ranked into st.top, largest first
__device__ __forceinline__ void kf_topk_finish(KfTopkState& st, int n, int k, int hi, u64 prefix, int above, int survivors) {
    const int tid = threadIdx.x, copy = tid >> 6;
    // Every thread has read the gathered count before the first thread resets it below
    __syncthreads();
    while (hi > 0 && survivors > k) {
        const int width = hi < KF_TOPK_BITS ? hi : KF_TOPK_BITS;
        const int shift = hi - width;
        const u64 dmask = (1ull << width) - 1ull;
        survivors = kf_topk_pass(st, hi, width, k, prefix, above, [&]() {
            for (int j = tid; j < n; j += KF_TOPK_THREADS) {
                const u64 key = st.keys[j];
                if ((key >> hi) == prefix) atomicAdd(&st.hist[copy][(int)((key >> shift) & dmask)], 1);
            }
        });
        hi = shift;
    }
    if (tid == 0) st.slot = 0;
    __syncthreads();
    for (int j = tid; j < n; j += KF_TOPK_THREADS) {
        const u64 key = st.keys[j];
        if ((key >> hi) >= prefix) st.top[atomicAdd(&st.slot, 1)] = key;
    }
    __syncthreads();
    if (tid < k) {
        const u64 mine = st.top[tid];
        int rank = 0;
        for (int q = 0; q < k; q++) rank += st.top[q] > mine;
        st.keys[rank] = mine;
    }
    __syncthreads();
    if (tid < k) st.top[tid] = st.keys[tid];
    __syncthreads();
}

// Columns of every slice but the last, a multiple of four so every slice starts on sixteen bytes
__device__ __forceinline__ int kf_topk_slice_len(int cols, int slices) {
    return ((cols + slices - 1) / slices + 3) & ~3;
}

__device__ __forceinline__ int kf_topk_id_bits(int cols) {
    int id_bits = 1;
    while ((1 << id_bits) < cols) id_bits++;
    return id_bits;
}

// The k largest keys of slice blockIdx.x of row blockIdx.y into keys_out, with the slice's maximum and sum of
// exponentials into stats
extern "C" __global__ void __launch_bounds__(KF_TOPK_THREADS) topk_slices(const float* __restrict__ x, u64* __restrict__ keys_out, float* __restrict__ stats, int cols, int k, int slices) {
    __shared__ KfTopkState st;
    __shared__ float fred[32];
    const int tid = threadIdx.x, copy = tid >> 6;
    const int slice = blockIdx.x, start = slice * kf_topk_slice_len(cols, slices);
    const int len = min(kf_topk_slice_len(cols, slices), cols - start);
    const float* span = x + (i64)blockIdx.y * cols + start;
    const int at = blockIdx.y * slices + slice;
    // Ids take the bits that hold the last column, complemented so a lower id makes a larger key
    const int id_bits = kf_topk_id_bits(cols);
    const u32 idmask = (1u << id_bits) - 1u;
    const int total_bits = 32 + id_bits;
    const auto key_of = [&](int i, float v) { return ((u64)kf_topk_fkey(v) << id_bits) | (u64)(idmask - (u32)(start + i)); };
    // The first pass, the maximum and sum of exponentials with the key bits every value shares at the top
    u64 all = ~0ull, any = 0ull;
    {
        float m = KF_NEG_INF, s = 0.0f;
        kf_topk_each(span, len, [&](int, float v) {
            const u64 key = kf_topk_fkey(v);
            all &= key;
            any |= key;
            if (v > m) {
                s = s * __expf(m - v) + 1.0f;
                m = v;
            } else if (v > KF_NEG_INF) {
                s += __expf(v - m);
            }
        });
        const float bm = kf_block_max(m, fred);
        s = kf_block_sum(m == KF_NEG_INF ? 0.0f : s * __expf(m - bm), fred);
        if (tid == 0) {
            stats[2 * at] = bm;
            stats[2 * at + 1] = s;
        }
        kf_topk_block_bits(all, any, st.bits);
    }
    // The shared top bits are the prefix every key has, the digits begin where the keys part
    const u32 differ = (u32)(all ^ any);
    const int common = differ == 0u ? 32 : __clz(differ);
    int hi = total_bits - common;
    u64 prefix = common == 0 ? 0ull : (u64)((u32)all >> (32 - common));
    int above = 0;
    int survivors = len;
    while (survivors > KF_TOPK_GATHER) {
        const int width = hi < KF_TOPK_BITS ? hi : KF_TOPK_BITS;
        const int shift = hi - width;
        const u64 dmask = (1ull << width) - 1ull;
        survivors = kf_topk_pass(st, hi, width, k, prefix, above, [&]() {
            if (shift >= id_bits) {
                // The digit lies inside the value bits, so the value key alone decides and the id bits stay out
                const u32 want = (u32)prefix;
                const int vhi = hi - id_bits, vshift = shift - id_bits;
                kf_topk_each(span, len, [&](int, float v) {
                    const u32 key = kf_topk_fkey(v);
                    if ((key >> vhi) == want) atomicAdd(&st.hist[copy][(int)((key >> vshift) & (u32)dmask)], 1);
                });
            } else {
                kf_topk_each(span, len, [&](int i, float v) {
                    const u64 key = key_of(i, v);
                    if ((key >> hi) == prefix) atomicAdd(&st.hist[copy][(int)((key >> shift) & dmask)], 1);
                });
            }
        });
        hi = shift;
    }
    // The keys at or above the prefix, which hold the k largest, come into shared memory
    if (tid == 0) st.slot = 0;
    __syncthreads();
    if (hi >= id_bits) {
        const u32 want = (u32)prefix;
        const int vhi = hi - id_bits;
        kf_topk_each(span, len, [&](int i, float v) {
            if ((kf_topk_fkey(v) >> vhi) >= want) st.keys[atomicAdd(&st.slot, 1)] = key_of(i, v);
        });
    } else {
        kf_topk_each(span, len, [&](int i, float v) {
            const u64 key = key_of(i, v);
            if ((key >> hi) >= prefix) st.keys[atomicAdd(&st.slot, 1)] = key;
        });
    }
    __syncthreads();
    kf_topk_finish(st, st.slot, k, hi, prefix, above, survivors);
    if (tid < k) keys_out[(i64)at * k + tid] = st.top[tid];
}

// Row blockIdx.x from the slices' candidates, the k largest ranked and written with the row's log sum exp
extern "C" __global__ void __launch_bounds__(KF_TOPK_THREADS) topk_merge(const u64* __restrict__ keys_in, const float* __restrict__ stats, float* __restrict__ out, int cols, int k, int slices) {
    __shared__ KfTopkState st;
    __shared__ float fred[32];
    const int tid = threadIdx.x;
    const int n = slices * k;
    const u64* src = keys_in + (i64)blockIdx.x * n;
    u64 all = ~0ull, any = 0ull;
    for (int j = tid; j < n; j += KF_TOPK_THREADS) {
        const u64 key = src[j];
        st.keys[j] = key;
        all &= key;
        any |= key;
    }
    const int id_bits = kf_topk_id_bits(cols);
    const u32 idmask = (1u << id_bits) - 1u;
    const int total_bits = 32 + id_bits;
    float* orow = out + (i64)blockIdx.x * (2 * k + 1);
    // The slices' maxima and sums joined into the row's log sum exp
    const float* sts = stats + (i64)blockIdx.x * 2 * slices;
    float m = KF_NEG_INF;
    for (int s = tid; s < slices; s += KF_TOPK_THREADS) m = fmaxf(m, sts[2 * s]);
    m = kf_block_max(m, fred);
    float sum = 0.0f;
    for (int s = tid; s < slices; s += KF_TOPK_THREADS) {
        const float ms = sts[2 * s];
        if (ms > KF_NEG_INF) sum += sts[2 * s + 1] * __expf(ms - m);
    }
    sum = kf_block_sum(sum, fred);
    if (tid == 0) orow[2 * k] = m + logf(sum);
    kf_topk_block_bits(all, any, st.bits);
    const u64 differ = all ^ any;
    const int common = differ == 0ull ? total_bits : min(total_bits, __clzll(differ) - (64 - total_bits));
    const int hi = total_bits - common;
    const u64 prefix = hi == total_bits ? 0ull : (all >> hi);
    kf_topk_finish(st, n, k, hi, prefix, 0, n);
    if (tid < k) {
        const u64 mine = st.top[tid];
        orow[tid] = (float)(idmask - (u32)(mine & (u64)idmask));
        orow[k + tid] = kf_topk_unkey((u32)(mine >> id_bits));
    }
}

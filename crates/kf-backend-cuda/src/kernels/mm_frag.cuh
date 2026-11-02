#pragma once
// Product of R int8 activation rows against sixteen native weight rows per warp on mma fragments read straight
// from a per warp cp.async ring of super blocks and from the fragment array the quantizer wrote, see docs/mm-frag.md

#ifndef KF_FRAG_STAGES
#define KF_FRAG_STAGES 0
#endif
#ifndef KF_FRAG_WARPS
#define KF_FRAG_WARPS 0
#endif

template <typename T>
struct KfFrag;

// c = a b over sixteen weight rows, eight activation rows and thirty two columns of unsigned weights
__device__ __forceinline__ void kf_frag_mma32(int* c, const u32* a, const u32* b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};"
                 : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(0));
#endif
}

// c = a b over sixteen weight rows, eight activation rows and thirty two columns, both operands unsigned
__device__ __forceinline__ void kf_frag_mma32_uu(int* c, const u32* a, const u32* b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};"
                 : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(0));
#endif
}

// c = a b over sixteen weight rows, eight activation rows and sixteen columns of signed weights
__device__ __forceinline__ void kf_frag_mma16(int* c, const u32* a, const u32* b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%7,%7,%7};"
                 : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(b[0]), "r"(0));
#endif
}

// Word at a shared memory address of two byte alignment through the two words around it
__device__ __forceinline__ u32 kf_frag_word(const u8* p) {
    const u32 a = (u32)reinterpret_cast<size_t>(p);
    if (a & 2) {
        const u32 lo = *reinterpret_cast<const u32*>(p - 2), hi = *reinterpret_cast<const u32*>(p + 2);
        return __byte_perm(lo, hi, 0x5432);
    }
    return *reinterpret_cast<const u32*>(p);
}

template <typename T>
struct KfFragSmem {
    static constexpr int STAGES = KF_FRAG_STAGES > 0 ? KF_FRAG_STAGES : KfFrag<T>::STAGES;
    static constexpr int WARPS = KF_FRAG_WARPS > 0 ? KF_FRAG_WARPS : KfFrag<T>::WARPS;
    static constexpr int BYTES = WARPS * STAGES * 16 * KfFrag<T>::PITCH;
    static constexpr int BLOCKS = BYTES <= 32 * 1024 ? 3 : (BYTES <= 49152 ? 2 : 1);
};

template <typename T, int R>
__device__ void kf_mm_frag(const u8* __restrict__ w, const u8* __restrict__ xa, const float* __restrict__ bias, const float* __restrict__ res, float* __restrict__ y, int rows, int n, int n_out, i64 row_bytes, const long long* __restrict__ segs, int nseg, int wpr) {
    extern __shared__ __align__(128) u8 kf_frag_smem[];
    typedef KfFrag<T> F;
    constexpr int STAGES = KfFragSmem<T>::STAGES, WARPS = KfFragSmem<T>::WARPS, NT = R / 8, KS = F::KSTEP, GROUPS = T::QK / KS, AR = KS / 8, BR = KS / 16;
    constexpr int PITCH = F::PITCH, PIECES = F::PIECES, STAGE = 16 * PITCH, RING = STAGES * STAGE;
    constexpr int NP = (16 * PIECES + 31) / 32;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, gid = lane >> 2, tig = lane & 3;
    const int wsh = wpr == 8 ? 3 : (wpr == 4 ? 2 : (wpr == 2 ? 1 : 0));
    const int rg = warp >> wsh, split = warp - (rg << wsh);
    const int act0 = blockIdx.y * R;
    int o0 = blockIdx.x * ((16 * WARPS) >> wsh) + rg * 16;
    bool live = true;
    if (segs) {
        const long long* e = segs;
        int i = 0;
        for (; i < nseg; i++, e += KF_SEG_WORDS) {
            if (o0 >= (int)e[4] && o0 < (int)(e[4] + e[5])) break;
        }
        if (i == nseg) {
            live = false;
            e = segs;
        }
        u8* yb = reinterpret_cast<u8*>(y);
        w += e[0];
        y = reinterpret_cast<float*>(yb + e[1]);
        bias = e[2] == KF_SEG_NONE ? nullptr : reinterpret_cast<const float*>(yb + e[2]);
        res = e[3] == KF_SEG_NONE ? nullptr : reinterpret_cast<const float*>(yb + e[3]);
        o0 -= (int)e[4];
        n_out = (int)e[5];
    }
    if (!live) o0 = 0;
    const int nsb = n / T::QK, last = rows - 1;
    const u8* end = w + (i64)n_out * row_bytes;
    u8* ring = kf_frag_smem + warp * RING;
    const u32 ring_s = kf_smem_addr(ring);
    // Pieces of a stage per lane, sixteen bytes of a row window each, rows past the end reading one that exists
    const u8* src[NP];
    u32 dst[NP];
#pragma unroll
    for (int i = 0; i < NP; i++) {
        const int p = lane + 32 * i, r = p / PIECES, part = p - r * PIECES;
        src[i] = w + (i64)min(o0 + r, n_out - 1) * row_bytes + part * 16;
        dst[i] = r * PITCH + part * 16;
    }
    auto issue = [&](int sb, int slot) {
        if (sb < nsb) {
#pragma unroll
            for (int i = 0; i < NP; i++) {
                if (lane + 32 * i < 16 * PIECES) {
                    const u8* s = src[i] + (i64)sb * T::BYTES;
                    s = reinterpret_cast<const u8*>(reinterpret_cast<size_t>(s) & ~(size_t)15);
                    const i64 left = end - s;
                    kf_cp_async16n(ring_s + slot * STAGE + dst[i], s, left >= 16 ? 16 : (left > 0 ? (int)left : 0));
                }
            }
        }
        kf_cp_commit();
    };
    const u8* row0 = w + (i64)min(o0 + gid, n_out - 1) * row_bytes;
    const u8* row1 = w + (i64)min(o0 + gid + 8, n_out - 1) * row_bytes;
    // Fragment array offsets per n tile of the B fragment, the sums, and the two C column scales with their tile rows
    u32 xb[NT], xsum[NT], xdp[NT][2];
    int brt[NT], xrt[NT][2];
#pragma unroll
    for (int nt = 0; nt < NT; nt++) {
        const int br = min(act0 + nt * 8 + gid, last), bt = br >> 3, bg = br & 7;
        brt[nt] = min(8, rows - 8 * bt);
        const u32 tile = (u32)bt * nsb * 8 * KF_XA_ROW;
        xb[nt] = tile + bg * 32 + (KS == 32 ? tig * 8 : tig * 4);
        xsum[nt] = tile + brt[nt] * 256 + bg * 16 + 4 * (tig & 1);
#pragma unroll
        for (int e = 0; e < 2; e++) {
            const int xr = min(act0 + nt * 8 + 2 * tig + e, last), xt = xr >> 3;
            xrt[nt][e] = min(8, rows - 8 * xt);
            xdp[nt][e] = (u32)xt * nsb * 8 * KF_XA_ROW + xrt[nt][e] * 272 + (xr & 7) * 4;
        }
    }
    float acc[NT][4];
#pragma unroll
    for (int nt = 0; nt < NT; nt++) acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.0f;
#pragma unroll
    for (int s = 0; s < STAGES - 1; s++) issue(split + s * wpr, s);
    int slot = 0;
    for (int sb = split; sb < nsb; sb += wpr, slot = slot + 1 == STAGES ? 0 : slot + 1) {
        kf_cp_wait<STAGES - 2>();
        __syncwarp();
        issue(sb + (STAGES - 1) * wpr, slot == 0 ? STAGES - 1 : slot - 1);
        const u8* st = ring + slot * STAGE;
        const u8* r0 = st + gid * PITCH + ((u32)reinterpret_cast<size_t>(row0 + (i64)sb * T::BYTES) & 15u);
        const u8* r1 = st + (gid + 8) * PITCH + ((u32)reinterpret_cast<size_t>(row1 + (i64)sb * T::BYTES) & 15u);
        typename F::Hdr h0, h1;
        F::header(r0, h0);
        F::header(r1, h1);
        int qacc[NT][4], mint[NT][4];
#pragma unroll
        for (int nt = 0; nt < NT; nt++) qacc[nt][0] = qacc[nt][1] = qacc[nt][2] = qacc[nt][3] = mint[nt][0] = mint[nt][1] = mint[nt][2] = mint[nt][3] = 0;
        // Activation scale of the super block for the two C columns
        float xdv[NT][2];
#pragma unroll
        for (int nt = 0; nt < NT; nt++) {
#pragma unroll
            for (int e = 0; e < 2; e++) xdv[nt][e] = __ldg(reinterpret_cast<const float*>(xa + xdp[nt][e] + sb * xrt[nt][e] * KF_XA_ROW));
        }
        // Min term of the super block as a product of the packed mins against the packed group sums
        if constexpr (F::HAS_MIN) {
            u32 am[4] = {F::min_frag(h0, tig), F::min_frag(h1, tig), 0u, 0u};
#pragma unroll
            for (int nt = 0; nt < NT; nt++) {
                const u8* p = xa + xsum[nt] + sb * brt[nt] * KF_XA_ROW;
#ifdef KF_FRAG_NO_XS
                u32 bl[2] = {(u32)sb, 0u}, bh[2] = {(u32)nt, 0u};
#else
                u32 bl[2] = {tig < 2 ? __ldg(reinterpret_cast<const u32*>(p)) : 0u, 0u};
                u32 bh[2] = {tig < 2 ? __ldg(reinterpret_cast<const u32*>(p + 8)) : 0u, 0u};
#endif
                int cl[4], ch[4];
                kf_frag_mma32_uu(cl, am, bl);
                kf_frag_mma32(ch, am, bh);
#pragma unroll
                for (int e = 0; e < 4; e++) mint[nt][e] = ch[e] * 256 + cl[e];
            }
        }
#pragma unroll
        for (int j = 0; j < GROUPS / F::GPC; j++) {
            u32 a[F::GPC][AR];
#ifdef KF_FRAG_NO_DECODE
#pragma unroll
            for (int gg = 0; gg < F::GPC; gg++) {
#pragma unroll
                for (int q = 0; q < AR; q++) a[gg][q] = *reinterpret_cast<const u32*>(reinterpret_cast<size_t>((q & 1 ? r1 : r0) + 16 * j + 4 * gg + 2 * (q & 2)) & ~(size_t)3);
            }
#else
            F::chunk(r0, r1, tig, j, a);
#endif
            // B fragments of the chunk for every n tile ahead of its products
            u32 b[F::GPC][NT][BR];
#pragma unroll
            for (int gg = 0; gg < F::GPC; gg++) {
                const int g = j * F::GPC + gg;
#pragma unroll
                for (int nt = 0; nt < NT; nt++) {
#ifdef KF_FRAG_BFAKE
                    b[gg][nt][0] = (u32)(g + sb) * 0x01010101u;
                    if constexpr (BR == 2) b[gg][nt][1] = b[gg][nt][0] ^ 0x10101010u;
#else
                    const u8* p = xa + xb[nt] + sb * brt[nt] * KF_XA_ROW;
                    if constexpr (KS == 32) {
                        const uint2 v = __ldg(reinterpret_cast<const uint2*>(p + g * brt[nt] * 32));
                        b[gg][nt][0] = v.x;
                        b[gg][nt][1] = v.y;
                    } else {
                        b[gg][nt][0] = __ldg(reinterpret_cast<const u32*>(p + (g >> 1) * brt[nt] * 32 + 16 * (g & 1)));
                    }
#endif
                }
            }
#pragma unroll
            for (int gg = 0; gg < F::GPC; gg++) {
                const int g = j * F::GPC + gg;
                const int sc0 = F::scale(h0, g), sc1 = F::scale(h1, g);
#pragma unroll
                for (int nt = 0; nt < NT; nt++) {
                    int c[4];
#ifdef KF_FRAG_NO_MMA
                    c[0] = (int)(a[gg][0] ^ b[gg][nt][0]);
                    c[1] = (int)(a[gg][1] + b[gg][nt][0]);
                    c[2] = (int)(a[gg][AR - 1] ^ b[gg][nt][BR - 1]);
                    c[3] = (int)(a[gg][AR - 2] + b[gg][nt][BR - 1]);
#else
                    if constexpr (KS == 32) kf_frag_mma32(c, a[gg], b[gg][nt]);
                    else kf_frag_mma16(c, a[gg], b[gg][nt]);
#endif
                    qacc[nt][0] += c[0] * sc0;
                    qacc[nt][1] += c[1] * sc0;
                    qacc[nt][2] += c[2] * sc1;
                    qacc[nt][3] += c[3] * sc1;
                }
            }
        }
#pragma unroll
        for (int nt = 0; nt < NT; nt++) {
            const float x0 = xdv[nt][0], x1 = xdv[nt][1];
            float m[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            if constexpr (F::HAS_MIN) {
                m[0] = (float)mint[nt][0] * h0.dmin;
                m[1] = (float)mint[nt][1] * h0.dmin;
                m[2] = (float)mint[nt][2] * h1.dmin;
                m[3] = (float)mint[nt][3] * h1.dmin;
            }
            acc[nt][0] = fmaf(x0, fmaf((float)qacc[nt][0], h0.d, -m[0]), acc[nt][0]);
            acc[nt][1] = fmaf(x1, fmaf((float)qacc[nt][1], h0.d, -m[1]), acc[nt][1]);
            acc[nt][2] = fmaf(x0, fmaf((float)qacc[nt][2], h1.d, -m[2]), acc[nt][2]);
            acc[nt][3] = fmaf(x1, fmaf((float)qacc[nt][3], h1.d, -m[3]), acc[nt][3]);
        }
    }
    kf_cp_wait<0>();
    if (wpr > 1) {
        float* join = reinterpret_cast<float*>(kf_frag_smem);
        __syncthreads();
        if (split > 0) {
#pragma unroll
            for (int nt = 0; nt < NT; nt++) {
#pragma unroll
                for (int e = 0; e < 4; e++) join[((warp * 32 + lane) * NT + nt) * 4 + e] = acc[nt][e];
            }
        }
        __syncthreads();
        if (split > 0) return;
        for (int s = 1; s < wpr; s++) {
#pragma unroll
            for (int nt = 0; nt < NT; nt++) {
#pragma unroll
                for (int e = 0; e < 4; e++) acc[nt][e] += join[(((warp + s) * 32 + lane) * NT + nt) * 4 + e];
            }
        }
    }
    if (!live) return;
#pragma unroll
    for (int half = 0; half < 2; half++) {
        const int o = o0 + gid + 8 * half;
        if (o >= n_out) continue;
        const float b = bias ? bias[o] : 0.0f;
#pragma unroll
        for (int nt = 0; nt < NT; nt++) {
#pragma unroll
            for (int e = 0; e < 2; e++) {
                const int ar = act0 + nt * 8 + 2 * tig + e;
                if (ar < rows) y[(i64)ar * n_out + o] = acc[nt][2 * half + e] + b + (res ? res[(i64)ar * n_out + o] : 0.0f);
            }
        }
    }
}

#define KF_FRAG_ENTRY(T, R)                                                                                                                                                                                                                                                                   \
    extern "C" __global__ void __launch_bounds__(32 * KfFragSmem<T>::WARPS, R >= 24 ? 1 : KfFragSmem<T>::BLOCKS) mm_frag_##R(const u8* w, const u8* xa, const float* bias, const float* res, float* y, int rows, int n, int n_out, i64 row_bytes, const long long* segs, int nseg, int wpr) { \
        kf_mm_frag<T, R>(w, xa, bias, res, y, rows, n, n_out, row_bytes, segs, nseg, wpr);                                                                                                                                                                                                     \
    }

// Fragment product entries of a type with a KfFrag plan at 8 16 24 and 32 activation rows
#define KF_FRAG_KERNELS(T) KF_FRAG_ENTRY(T, 8) KF_FRAG_ENTRY(T, 16) KF_FRAG_ENTRY(T, 24) KF_FRAG_ENTRY(T, 32)

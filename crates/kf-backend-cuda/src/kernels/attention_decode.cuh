#pragma once
// Decode attention of one row, one kv head, and one part of its keys, the body behind attention_decode and the persistent
// pass, PFK rounds of keys and PFV rounds of values in flight per warp, up to GMAX query heads per block
#include "common.cuh"

// Merges a second online softmax state into the first, both over the same value dims
__device__ __forceinline__ void kf_at_merge(float& m, float& l, float* acc, float m2, float l2, const float* acc2, int n) {
    float mn = fmaxf(m, m2);
    float c1 = m == KF_NEG_INF ? 0.0f : __expf(m - mn);
    float c2 = m2 == KF_NEG_INF ? 0.0f : __expf(m2 - mn);
    l = l * c1 + l2 * c2;
    for (int i = 0; i < n; i++) acc[i] = acc[i] * c1 + acc2[i] * c2;
    m = mn;
}

// Decode attention of one row and one kv head with every query head that shares it, over one of grid y parts of the
// row's keys, see kf_attention_decode. counters holds one count per row and kv head then one per row, each reset by
// the block that completes it. tree, when given, holds three words per row: the two halves of the mask of the keys
// the row sees among the last 64 before its last key, bit k for the key k below it, and the row's position for the
// window rule, since a tree row's cache index stands past its position.
#define KF_DEC_WARPS 8
#define KF_DEC_THREADS 256

// Norms and rotates one head held as eight dims per lane over the LPK lanes of a key group, the dims of lane dl being
// dl * 8 + i. vp holds the neox partner of each dim, the dim half away, as the caller loaded it, so no lane waits on
// another, and cs and sn hold the cosine and sine of each dim's pair angle.
template <int LPK>
__device__ __forceinline__ void kf_dec_prep(float* v, const float* vp, int dl, const float* cs, const float* sn, int half, int neox, float mscale, int norm, const float* __restrict__ w, const float* __restrict__ b, float eps) {
    constexpr int HD = LPK * 8;
    float inv = 1.0f;
    if (norm) {
        float ss = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; i++) ss += v[i] * v[i];
#pragma unroll
        for (int off = LPK >> 1; off > 0; off >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, off);
        inv = 1.0f / sqrtf(ss / (float)HD + eps);
#pragma unroll
        for (int i = 0; i < 8; i++) {
            int c = dl * 8 + i;
            v[i] = v[i] * (inv * (w ? w[c] : 1.0f)) + (b ? b[c] : 0.0f);
        }
    }
    if (neox) {
        float r[8];
#pragma unroll
        for (int i = 0; i < 8; i++) {
            int c = dl * 8 + i;
            bool first = c < half;
            int cp = first ? c + half : c - half;
            float o = vp[i];
            if (norm) o = o * (inv * (w ? w[cp] : 1.0f)) + (b ? b[cp] : 0.0f);
            r[i] = c >= 2 * half ? v[i] : (first ? (v[i] * cs[i] - o * sn[i]) * mscale : (o * sn[i] + v[i] * cs[i]) * mscale);
        }
#pragma unroll
        for (int i = 0; i < 8; i++) v[i] = r[i];
    } else {
#pragma unroll
        for (int i = 0; i < 8; i += 2) {
            int c = dl * 8 + i;
            if (c >= 2 * half) continue;
            float x0 = v[i], x1 = v[i + 1];
            v[i] = (x0 * cs[i] - x1 * sn[i]) * mscale;
            v[i + 1] = (x0 * sn[i] + x1 * cs[i]) * mscale;
        }
    }
}

// Four or fewer query heads of one kv head of one row over a part of its keys, in chunks of KF_DEC_CHUNK keys: every
// warp scores KPW keys at once against every head, LPK lanes to a key with eight dims each, and parks the scores in
// shared memory, then each head's warps take the softmax of the chunk and sum the values in the same key lane layout,
// the key lanes of a dim joined by shuffle and the warps of a head through shared memory into the head's running
// state. With k_in given the query heads and the new key arrive unnormed and unrotated, the block norms and rotates
// them from one sine and cosine table per block, the first part appends the key and value to the cache, and the new
// key and value come from registers. A short row runs in one block that writes the output itself, a long row joins
// its parts in the last block to arrive, and the finishing block quantizes its heads' slice of the row for the
// product after it when the slice holds whole quantization blocks, else the row once every kv head of the row has
// joined.
#define KF_DEC_CHUNK 128

// Eight halves of a key or value row as floats
__device__ __forceinline__ void kf_dec_h8(const uint4& u, float* f) {
    f[0] = kf_h2f(u.x & 0xffffu);
    f[1] = kf_h2f(u.x >> 16);
    f[2] = kf_h2f(u.y & 0xffffu);
    f[3] = kf_h2f(u.y >> 16);
    f[4] = kf_h2f(u.z & 0xffffu);
    f[5] = kf_h2f(u.z >> 16);
    f[6] = kf_h2f(u.w & 0xffffu);
    f[7] = kf_h2f(u.w >> 16);
}

template <int HD, int PFK, int PFV, int GMAX>
__device__ __forceinline__ void kf_attention_decode(int bx, int by, int bz, int gy, int gz, const float* __restrict__ q, u16* __restrict__ kc, u16* __restrict__ vc, const int* __restrict__ pos, const int* __restrict__ seq, const int* __restrict__ attend, const float* __restrict__ sinks, float* __restrict__ out, float* __restrict__ parts, int n_head, int n_head_kv, float scale, int window, float softcap, int cap, int rows, int* __restrict__ counters, int kind, i8* __restrict__ qo, float* __restrict__ s16o, float* __restrict__ m16o, float* __restrict__ s32o, float* __restrict__ m32o,
                                                    const float* __restrict__ k_in, const float* __restrict__ v_in, const int* __restrict__ rope4, const float2* __restrict__ cst, int half, int neox, float mscale, int mode, int s0, int s1, int s2, int sect, int norm_q, const float* __restrict__ wq, const float* __restrict__ bq, float eps_q, int norm_k, const float* __restrict__ wk, const float* __restrict__ bk, float eps_k,
                                                    const int* __restrict__ tree) {
    constexpr int LPK = HD / 8;
    constexpr int KPW = 32 / LPK;
    constexpr int KPS = KF_DEC_WARPS * KPW;
    constexpr int CHUNK = KF_DEC_CHUNK;
    // Keys one part takes at least, so a short row is one block and a long one spreads over the grid
    constexpr int KEYS_PER_PART = CHUNK;
    const bool fused = k_in != nullptr;
    __shared__ float sc_s[GMAX][CHUNK];
    __shared__ float mg_m[KF_DEC_WARPS], mg_l[KF_DEC_WARPS];
    __shared__ __align__(16) float mg_acc[KF_DEC_WARPS][HD];
    __shared__ float cs_s[HD / 2], sn_s[HD / 2];
    __shared__ float wsm[KF_DEC_WARPS][64];
    __shared__ int sm_last;
    extern __shared__ __align__(16) float rowbuf[];
    const int r = bx / n_head_kv, hk = bx - r * n_head_kv, part = by;
    const int group = n_head / n_head_kv;
    const int nsplit = gz, hs = bz;
    const int hpb = (group + nsplit - 1) / nsplit;
    const int h0 = hs * hpb, nh = min(group, h0 + hpb) - h0;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int kl = lane / LPK, dl = lane - kl * LPK;
    const int p = pos[r];
    const int sq = seq[r];
    const int hi_row = max(p, attend[r]);
    const unsigned long long tmask = tree ? ((unsigned long long)(u32)tree[3 * r] | ((unsigned long long)(u32)tree[3 * r + 1] << 32)) : ~0ull;
    const int wpos = tree ? tree[3 * r + 2] : p;
    const int lo_row = window > 0 ? max(0, wpos - (window - 1)) : 0;
    const int nkeys = hi_row - lo_row + 1;
    const int n_active = min(gy, max(1, (nkeys + KEYS_PER_PART - 1) / KEYS_PER_PART));
    if (part >= n_active || nh <= 0) return;
    const int cslot = r * (n_head_kv * nsplit + 1) + hk * nsplit + hs, crow = r * (n_head_kv * nsplit + 1) + n_head_kv * nsplit;
    const int n = n_head * HD, cols_kv = nh * HD, c_lo = (hk * group + h0) * HD;
    const bool slice = kind != 0 && ((kind == 3 && cols_kv % 32 == 0) || (kind == 4 && cols_kv % 256 == 0));
    const int per = (nkeys + n_active - 1) / n_active;
    const int lo = lo_row + part * per;
    const int hi = min(hi_row, lo + per - 1);
    const i64 kw = (i64)n_head_kv * HD, vw = kw;
    const u16* kb = kc + (i64)sq * cap * kw + (i64)hk * HD;
    const u16* vb = vc + (i64)sq * cap * vw + (i64)hk * HD;

    // The first round of keys of the first chunk loads now, so the queries' preparation below overlaps the wait
    const int ring0_first = lo % cap;
    uint4 kr[PFK];
#pragma unroll
    for (int u = 0; u < PFK; u++) {
        const int off = warp * KPW + u * KPS + kl;
        kr[u] = make_uint4(0, 0, 0, 0);
        if (lo + off <= hi && off < CHUNK) {
            int i = ring0_first + off;
            kr[u] = *reinterpret_cast<const uint4*>(kb + (i64)(i >= cap ? i - cap : i) * kw + dl * 8);
        }
    }
    bool kr_ready = true;

    // The queries of this block's heads, this lane's eight dims of each, and the new key and value when fused. With
    // the ropes folded in each warp readies one head, the query heads in the warps below nh and the new key in warp
    // nh, with the neox rotation partner of every dim, the dim half away, loaded alongside, and hands it on through
    // shared memory so no warp repeats another's work
    float qv[GMAX][8];
    float knew[8], vnew[8];
    if (fused) {
        const float* vr = v_in + ((i64)r * n_head_kv + hk) * HD + dl * 8;
#pragma unroll
        for (int i = 0; i < 8; i++) vnew[i] = kf_round_f16(__ldcg(vr + i));
        const int cp0 = dl * 8 < half ? dl * 8 + half : dl * 8 - half;
        const bool in_pairs = dl * 8 < 2 * half;
        float own[8], ownp[8];
        const float* src = warp < nh ? q + ((i64)r * n_head + hk * group + h0 + warp) * HD : (warp == nh ? k_in + ((i64)r * n_head_kv + hk) * HD : nullptr);
#pragma unroll
        for (int i = 0; i < 8; i++) {
            own[i] = src ? __ldcg(src + dl * 8 + i) : 0.0f;
            ownp[i] = src && neox && in_pairs ? __ldcg(src + cp0 + i) : 0.0f;
        }
        // The sine and cosine of every pair angle of the row out of the table of its stream's position, one per thread
        if (tid < half) {
            const int* rp = rope4 + 4 * r;
            const float2 c = cst[(i64)rp[kf_rope_stream(tid, mode, s0, s1, s2, sect)] * half + tid];
            cs_s[tid] = c.x;
            sn_s[tid] = c.y;
        }
        __syncthreads();
        float cs[8], sn[8];
#pragma unroll
        for (int i = 0; i < 8; i++) {
            int c = dl * 8 + i;
            int ip = neox ? (c < half ? c : c - half) : (c >> 1);
            bool in = c < 2 * half;
            cs[i] = in ? cs_s[ip] : 1.0f;
            sn[i] = in ? sn_s[ip] : 0.0f;
        }
        if (warp < nh) {
            kf_dec_prep<LPK>(own, ownp, dl, cs, sn, half, neox, mscale, norm_q, wq, bq, eps_q);
        } else if (warp == nh) {
            kf_dec_prep<LPK>(own, ownp, dl, cs, sn, half, neox, mscale, norm_k, wk, bk, eps_k);
#pragma unroll
            for (int i = 0; i < 8; i++) own[i] = kf_round_f16(own[i]);
            if (part == 0 && hs == 0 && kl == 0) {
                const i64 slot = (i64)sq * cap + (p % cap);
                u16* ko = kc + slot * kw + (i64)hk * HD + dl * 8;
                u16* vo = vc + slot * vw + (i64)hk * HD + dl * 8;
#pragma unroll
                for (int i = 0; i < 8; i++) {
                    ko[i] = kf_f2h(own[i]);
                    vo[i] = kf_f2h(vnew[i]);
                }
            }
        }
        if (warp <= nh && kl == 0) {
#pragma unroll
            for (int i = 0; i < 8; i++) mg_acc[warp][dl * 8 + i] = own[i];
        }
        __syncthreads();
#pragma unroll
        for (int hq = 0; hq < GMAX; hq++) {
            if (hq < nh) {
#pragma unroll
                for (int i = 0; i < 8; i++) qv[hq][i] = kf_round_f16(mg_acc[hq][dl * 8 + i]);
            }
        }
#pragma unroll
        for (int i = 0; i < 8; i++) knew[i] = mg_acc[nh][dl * 8 + i];
        __syncthreads();
    } else {
#pragma unroll
        for (int hq = 0; hq < GMAX; hq++) {
            if (hq < nh) {
                const float* qr = q + ((i64)r * n_head + hk * group + h0 + hq) * HD + dl * 8;
#pragma unroll
                for (int i = 0; i < 8; i++) qv[hq][i] = kf_round_f16(__ldcg(qr + i));
            }
        }
    }

    // Warps per head of the value pass and this warp's head and index among them
    const int wph = KF_DEC_WARPS / nh;
    const int hw = warp / wph, wi = warp - hw * wph;
    // Running state of head hw in the warps of index zero, the eight dims of lane dl of the sum
    float m_run = KF_NEG_INF, l_run = 0.0f, acc_run[8];
#pragma unroll
    for (int d = 0; d < 8; d++) acc_run[d] = 0.0f;

    for (int c0 = lo; c0 <= hi; c0 += CHUNK) {
        const int c1 = min(hi, c0 + CHUNK - 1), nc = c1 - c0 + 1;
        // Ring row of the chunk's first key, the rest following with one wrap since a row never attends past cap keys
        const int ring0 = c0 % cap;
        auto ring = [&](int off) { int i = ring0 + off; return i >= cap ? i - cap : i; };
        // Scores of the chunk: every warp takes KPW keys per step and PFK steps of keys per round, all of a round's
        // keys loading before any is scored, every lane in every step for the shuffles
        for (int jr = c0 + warp * KPW; jr <= c1; jr += PFK * KPS) {
            if (!kr_ready) {
#pragma unroll
                for (int u = 0; u < PFK; u++) {
                    const int j = jr + u * KPS + kl;
                    kr[u] = make_uint4(0, 0, 0, 0);
                    if (j <= c1) kr[u] = *reinterpret_cast<const uint4*>(kb + (i64)ring(j - c0) * kw + dl * 8);
                }
            }
            kr_ready = false;
#pragma unroll
            for (int u = 0; u < PFK; u++) {
                const int j = jr + u * KPS + kl;
                if (jr + u * KPS > c1) break;
                const bool ok = j <= c1 && (hi_row - j >= 64 || ((tmask >> (hi_row - j)) & 1ull));
                float kf[8];
                kf_dec_h8(kr[u], kf);
                if (fused && j == p) {
#pragma unroll
                    for (int i = 0; i < 8; i++) kf[i] = knew[i];
                }
#pragma unroll
                for (int hq = 0; hq < GMAX; hq++) {
                    if (hq >= nh) break;
                    float d = 0.0f;
#pragma unroll
                    for (int i = 0; i < 8; i++) d = fmaf(qv[hq][i], kf[i], d);
#pragma unroll
                    for (int off = LPK >> 1; off > 0; off >>= 1) d += __shfl_xor_sync(0xffffffffu, d, off);
                    float sc = d * scale;
                    if (softcap != 0.0f) sc = softcap * tanhf(sc / softcap);
                    if (dl == 0 && j <= c1) sc_s[hq][j - c0] = ok ? sc : KF_NEG_INF;
                }
            }
        }
        __syncthreads();
        // Softmax of the chunk, warp wi of head hw over keys wi * 32 + lane + 32 wph t, then the value sum over the
        // same keys in the key lane layout so no other warp's weights are read, PFV keys of values loading at once
        float m_c = KF_NEG_INF, l_c = 0.0f, acc_c[8];
#pragma unroll
        for (int d = 0; d < 8; d++) acc_c[d] = 0.0f;
        if (hw < nh) {
            for (int t = wi * 32 + lane; t < nc; t += 32 * wph) m_c = fmaxf(m_c, sc_s[hw][t]);
            m_c = kf_warp_max(m_c);
            for (int t = wi * 32 + lane; t < nc; t += 32 * wph) {
                float pj = __expf(sc_s[hw][t] - m_c);
                l_c += pj;
                sc_s[hw][t] = pj;
            }
            l_c = kf_warp_sum(l_c);
            __syncwarp();
            for (int tb = wi * 32; tb < nc; tb += 32 * wph) {
                const int te = min(nc, tb + 32);
#pragma unroll
                for (int t0 = tb; t0 < tb + 32; t0 += PFV * KPW) {
                    uint4 vr[PFV];
                    float pj[PFV];
#pragma unroll
                    for (int u = 0; u < PFV; u++) {
                        const int t = t0 + u * KPW + kl;
                        vr[u] = make_uint4(0, 0, 0, 0);
                        pj[u] = 0.0f;
                        if (t < te) {
                            pj[u] = sc_s[hw][t];
                            if (!(fused && c0 + t == p)) vr[u] = *reinterpret_cast<const uint4*>(vb + (i64)ring(t) * vw + dl * 8);
                        }
                    }
#pragma unroll
                    for (int u = 0; u < PFV; u++) {
                        const int t = t0 + u * KPW + kl;
                        float vf[8];
                        kf_dec_h8(vr[u], vf);
                        if (fused && t < te && c0 + t == p) {
#pragma unroll
                            for (int d = 0; d < 8; d++) vf[d] = vnew[d];
                        }
#pragma unroll
                        for (int d = 0; d < 8; d++) acc_c[d] = fmaf(pj[u], vf[d], acc_c[d]);
                    }
                }
            }
            // The key lanes of each dim join
#pragma unroll
            for (int off = LPK; off < 32; off <<= 1) {
#pragma unroll
                for (int d = 0; d < 8; d++) acc_c[d] += __shfl_xor_sync(0xffffffffu, acc_c[d], off);
            }
        }
        // The warps of a head join into its first warp, then the chunk joins the running state
        if (hw < nh && wi > 0) {
            if (lane == 0) {
                mg_m[warp] = m_c;
                mg_l[warp] = l_c;
            }
            if (kl == 0) {
#pragma unroll
                for (int d = 0; d < 8; d++) mg_acc[warp][dl * 8 + d] = acc_c[d];
            }
        }
        __syncthreads();
        if (hw < nh && wi == 0) {
            for (int w = 1; w < wph; w++) {
                float m2 = mg_m[warp + w], l2 = mg_l[warp + w], a2[8];
#pragma unroll
                for (int d = 0; d < 8; d++) a2[d] = mg_acc[warp + w][dl * 8 + d];
                kf_at_merge(m_c, l_c, acc_c, m2, l2, a2, 8);
            }
            kf_at_merge(m_run, l_run, acc_run, m_c, l_c, acc_c, 8);
        }
        __syncthreads();
    }

    // Warp zero of each head holds the head over this part, its key lane zero the dims
    if (hw < nh && wi == 0 && kl == 0) {
        const int h = hk * group + h0 + hw;
        if (n_active == 1) {
            float mx = l_run > 0.0f ? m_run : KF_NEG_INF;
            float sink = sinks ? sinks[h] : 0.0f;
            if (sinks) mx = fmaxf(mx, sink);
            float wp = l_run > 0.0f ? __expf(m_run - mx) : 0.0f;
            float tot = l_run * wp + (sinks ? __expf(sink - mx) : 0.0f);
            float* orow = out + ((i64)r * n_head + h) * HD + dl * 8;
#pragma unroll
            for (int d = 0; d < 8; d++) {
                float v = tot > 0.0f ? acc_run[d] * wp / tot : 0.0f;
                orow[d] = v;
                if (slice) rowbuf[h * HD + dl * 8 + d] = v;
            }
        } else {
            float* prow = parts + (((i64)part * rows + r) * n_head + h) * (2 + HD);
            if (dl == 0) {
                prow[0] = m_run;
                prow[1] = l_run;
            }
#pragma unroll
            for (int d = 0; d < 8; d++) prow[2 + dl * 8 + d] = acc_run[d];
        }
        if (n_active > 1 || !slice) __threadfence();
    }
    __syncthreads();
    if (n_active > 1) {
        if (tid == 0) sm_last = atomicAdd(counters + cslot, 1) == n_active - 1;
        __syncthreads();
        if (!sm_last) return;
        __threadfence();
        // Join the parts of this row and kv head: the warps share the query heads, each warp's lanes over the parts
        // for the weights, then over its share of the dims with the weights read back from shared memory
        const int n_parts = n_active;
        const int width = 2 + HD;
        const i64 stride = (i64)rows * n_head * width;
        for (int hq = warp / wph; hq < nh; hq += KF_DEC_WARPS / wph) {
            const int wj = warp % wph;
            const int h = hk * group + h0 + hq;
            const float* p0 = parts + ((i64)r * n_head + h) * width;
            float mx = KF_NEG_INF;
            for (int pi = lane; pi < n_parts; pi += 32) {
                const float* pp = p0 + pi * stride;
                if (__ldcg(pp + 1) > 0.0f) mx = fmaxf(mx, __ldcg(pp));
            }
            mx = kf_warp_max(mx);
            float sink = sinks ? sinks[h] : 0.0f;
            if (sinks) mx = fmaxf(mx, sink);
            float tot = 0.0f;
            for (int pi = lane; pi < n_parts; pi += 32) {
                const float* pp = p0 + pi * stride;
                const float l2 = __ldcg(pp + 1);
                float wp = l2 > 0.0f ? __expf(__ldcg(pp) - mx) : 0.0f;
                wsm[warp][pi] = wp;
                tot += l2 * wp;
            }
            tot = kf_warp_sum(tot) + (sinks ? __expf(sink - mx) : 0.0f);
            __syncwarp();
            for (int i = lane + 32 * wj; i < HD; i += 32 * wph) {
                float o = 0.0f;
                for (int pi = 0; pi < n_parts; pi++) o += __ldcg(p0 + pi * stride + 2 + i) * wsm[warp][pi];
                float v = tot > 0.0f ? o / tot : 0.0f;
                out[((i64)r * n_head + h) * HD + i] = v;
                if (slice) rowbuf[h * HD + i] = v;
            }
        }
        if (!slice) __threadfence();
        __syncthreads();
    }
    if (slice) {
        if (tid == 0) counters[cslot] = 0;
        const int c_hi = c_lo + cols_kv;
        const KfActRow o = kf_act_row(qo, s16o, m16o, s32o, m32o, r, n);
        for (int span = c_lo + warp * 256; span < c_hi; span += KF_DEC_WARPS * 256) {
            int c0 = span + lane * 8;
            float v[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
            if (c0 < c_hi) kf_load8(rowbuf + c0, v);
            kf_quant_span(v, c0, c_hi, kind, o);
        }
        return;
    }
    if (tid == 0) {
        counters[cslot] = 0;
        sm_last = kind != 0 && atomicAdd(counters + crow, 1) == n_head_kv * nsplit - 1;
    }
    __syncthreads();
    if (!sm_last) return;
    __threadfence();
    // Every head of the row is joined: quantize the row for the product that follows
    for (int i = tid; i < n; i += blockDim.x) rowbuf[i] = __ldcg(out + (i64)r * n + i);
    __syncthreads();
    const KfActRow o = kf_act_row(qo, s16o, m16o, s32o, m32o, r, n);
    for (int span = warp * 256; span < n; span += KF_DEC_WARPS * 256) {
        int c0 = span + lane * 8;
        float v[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        if (c0 < n) kf_load8(rowbuf + c0, v);
        kf_quant_span(v, c0, n, kind, o);
    }
    if (tid == 0) counters[crow] = 0;
}


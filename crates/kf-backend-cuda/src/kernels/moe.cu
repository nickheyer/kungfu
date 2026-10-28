#include "common.cuh"

// Warp wide arg max over candidates flagged one, the lowest index wins ties, minus one when none remain
__device__ __forceinline__ int kf_pick(const float* score, const int* flag, int n) {
    float v = KF_NEG_INF;
    int idx = 0x7fffffff;
    for (int e = threadIdx.x; e < n; e += 32) {
        if (flag[e] != 1) continue;
        float y = score[e];
        if (idx == 0x7fffffff || y > v || (y == v && e < idx)) {
            v = y;
            idx = e;
        }
    }
    for (int o = 16; o > 0; o >>= 1) {
        float ov = __shfl_xor_sync(0xffffffffu, v, o);
        int oi = __shfl_xor_sync(0xffffffffu, idx, o);
        bool take = oi != 0x7fffffff && (idx == 0x7fffffff || ov > v || (ov == v && oi < idx));
        if (take) {
            v = ov;
            idx = oi;
        }
    }
    return idx == 0x7fffffff ? -1 : idx;
}

// Gating, probability bias, grouped eligibility, top k and weight normalization for one row per warp
extern "C" __global__ void moe_route(const float* logits, const float* pbias, int* sel, float* wts, int n_expert, int n_used, int sigmoid_gate, int n_group, int n_group_used, int norm) {
    extern __shared__ float sm[];
    float* probs = sm;
    float* sc = probs + n_expert;
    int* flag = (int*)(sc + n_expert);
    float* gs = (float*)(flag + n_expert);
    int* gflag = (int*)(gs + (n_group > 0 ? n_group : 1));
    int r = blockIdx.x, lane = threadIdx.x;
    const float* lr = logits + (i64)r * n_expert;
    if (sigmoid_gate) {
        for (int e = lane; e < n_expert; e += 32) probs[e] = 1.0f / (1.0f + expf(-lr[e]));
    } else {
        float m = KF_NEG_INF;
        for (int e = lane; e < n_expert; e += 32) m = fmaxf(m, lr[e]);
        m = kf_warp_max(m);
        float s = 0.0f;
        for (int e = lane; e < n_expert; e += 32) {
            float x = expf(lr[e] - m);
            probs[e] = x;
            s += x;
        }
        s = kf_warp_sum(s);
        for (int e = lane; e < n_expert; e += 32) probs[e] = probs[e] / s;
    }
    __syncwarp();
    for (int e = lane; e < n_expert; e += 32) {
        sc[e] = probs[e] + (pbias ? pbias[e] : 0.0f);
        flag[e] = n_group > 0 ? 0 : 1;
    }
    __syncwarp();
    if (n_group > 0) {
        int per = n_expert / n_group;
        for (int g = lane; g < n_group; g += 32) {
            float b1 = KF_NEG_INF, b2 = KF_NEG_INF;
            for (int i = 0; i < per; i++) {
                float v = sc[g * per + i];
                if (v > b1) {
                    b2 = b1;
                    b1 = v;
                } else if (v > b2) {
                    b2 = v;
                }
            }
            gs[g] = per >= 2 ? b1 + b2 : b1;
            gflag[g] = 1;
        }
        __syncwarp();
        for (int k = 0; k < n_group_used; k++) {
            int best = kf_pick(gs, gflag, n_group);
            if (lane == 0 && best >= 0) gflag[best] = 2;
            __syncwarp();
        }
        for (int e = lane; e < n_group * per; e += 32) flag[e] = gflag[e / per] == 2 ? 1 : 0;
        __syncwarp();
    }
    int* srow = sel + (i64)r * n_used;
    float* wrow = wts + (i64)r * n_used;
    for (int j = 0; j < n_used; j++) {
        int best = kf_pick(sc, flag, n_expert);
        if (lane == 0) {
            srow[j] = best;
            if (best >= 0) flag[best] = 2;
        }
        __syncwarp();
    }
    if (lane == 0) {
        float sum = 0.0f;
        for (int j = 0; j < n_used; j++) {
            float w = srow[j] >= 0 ? probs[srow[j]] : 0.0f;
            wrow[j] = w;
            sum += w;
        }
        if (norm) {
            float div = fmaxf(sum, 6.103515625e-5f);
            for (int j = 0; j < n_used; j++) wrow[j] = wrow[j] / div;
        }
    }
}

// Groups routed pairs by slot into an order list with offsets, one block of threads
extern "C" __global__ void moe_group(const int* slots, int total, int n_slots, int* order, int* offsets) {
    extern __shared__ int cnt[];
    for (int i = threadIdx.x; i < n_slots; i += blockDim.x) cnt[i] = 0;
    __syncthreads();
    for (int i = threadIdx.x; i < total; i += blockDim.x) {
        int s = slots[i];
        if (s >= 0) atomicAdd(&cnt[s], 1);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int run = 0;
        for (int i = 0; i < n_slots; i++) {
            int c = cnt[i];
            offsets[i] = run;
            cnt[i] = run;
            run += c;
        }
        offsets[n_slots] = run;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < total; i += blockDim.x) {
        int s = slots[i];
        if (s >= 0) order[atomicAdd(&cnt[s], 1)] = i;
    }
}

extern "C" __global__ void moe_glu(const float* gate, const float* up, int stride, float* h, const int* slots, int n_ff, int total, int act) {
    for (i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x; i < (i64)total * n_ff; i += (i64)gridDim.x * blockDim.x) {
        i64 rj = i / n_ff, o = i - rj * n_ff;
        h[i] = slots[rj] < 0 ? 0.0f : kf_act(gate[rj * stride + o], act) * up[rj * stride + o];
    }
}

// Routing weighted sum of expert outputs plus the host partial sum when present
extern "C" __global__ void moe_combine(const float* part, const float* wts, const int* slots, const float* host, float* out, int rows, int n_used, int n_embd, float scale) {
    for (i64 i = blockIdx.x * (i64)blockDim.x + threadIdx.x; i < (i64)rows * n_embd; i += (i64)gridDim.x * blockDim.x) {
        i64 r = i / n_embd, o = i - r * n_embd;
        float acc = 0.0f;
        for (int j = 0; j < n_used; j++) {
            i64 rj = r * n_used + j;
            if (slots[rj] >= 0) acc += wts[rj] * scale * part[rj * n_embd + o];
        }
        out[i] = acc + (host ? host[i] : 0.0f);
    }
}

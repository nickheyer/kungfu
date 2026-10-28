#include "common.cuh"

// Index of the largest value in a row, the lowest index wins ties
extern "C" __global__ void argmax(const float* x, int cols, int* out) {
    __shared__ float bv[32];
    __shared__ int bi[32];
    float v = KF_NEG_INF;
    int idx = 0x7fffffff;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float y = x[i];
        if (y > v || (y == v && i < idx)) {
            v = y;
            idx = i;
        }
    }
    for (int o = 16; o > 0; o >>= 1) {
        float ov = __shfl_xor_sync(0xffffffffu, v, o);
        int oi = __shfl_xor_sync(0xffffffffu, idx, o);
        if (ov > v || (ov == v && oi < idx)) {
            v = ov;
            idx = oi;
        }
    }
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
    if (lane == 0) {
        bv[warp] = v;
        bi[warp] = idx;
    }
    __syncthreads();
    if (warp == 0) {
        v = lane < nw ? bv[lane] : KF_NEG_INF;
        idx = lane < nw ? bi[lane] : 0x7fffffff;
        for (int o = 16; o > 0; o >>= 1) {
            float ov = __shfl_xor_sync(0xffffffffu, v, o);
            int oi = __shfl_xor_sync(0xffffffffu, idx, o);
            if (ov > v || (ov == v && oi < idx)) {
                v = ov;
                idx = oi;
            }
        }
        if (lane == 0) *out = idx == 0x7fffffff ? 0 : idx;
    }
}

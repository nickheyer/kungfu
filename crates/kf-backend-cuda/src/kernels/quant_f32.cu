#include "quant.cuh"

struct QF32 {
    static constexpr int QK = 1, BYTES = 4;
    static __device__ __forceinline__ float at(const u8* p, int i) { return reinterpret_cast<const float*>(p)[i]; }
};

KF_QUANT_KERNELS(QF32)

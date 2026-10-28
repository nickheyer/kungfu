#include "quant.cuh"

struct QF16 {
    static constexpr int QK = 1, BYTES = 2;
    static __device__ __forceinline__ float at(const u8* p, int i) { return kf_h2f(reinterpret_cast<const u16*>(p)[i]); }
};

KF_QUANT_KERNELS(QF16)

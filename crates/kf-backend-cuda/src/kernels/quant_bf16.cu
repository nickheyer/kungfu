#include "quant.cuh"

struct QBf16 {
    static constexpr int QK = 1, BYTES = 2;
    static __device__ __forceinline__ float at(const u8* p, int i) { return __int_as_float((int)((u32)reinterpret_cast<const u16*>(p)[i] << 16)); }
};

KF_QUANT_KERNELS(QBf16)

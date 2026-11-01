// The KF8 product at a k step of 128 over two pipeline stages with the activation scale applied per group, the
// configuration the types with 32 wide groups run fastest at, see docs/kf8-format.md
#define KF8_BK 128
#define KF8_STAGES 2
#define KF8_DIRECT
#include "kf8.cu"

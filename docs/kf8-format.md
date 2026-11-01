# KF8, the device weight form for the tensor core product

One matmul kernel for every block quant type. The 27 per type kernels keep dequant and the decode matvec on native GGUF bytes, and gain one entry, `unpack_i8`, which writes a tensor in KF8. The prefill product reads only KF8, so there is one GEMM to make fast instead of 27.

## Why not repack at native density

The integer form of a weight is `d * q - m` per sixteen, with `q` an int8. For the codebook types the values of `q` are wide, iq2_xxs reaches 43, so no form narrower than int8 holds them. Per sixteen scales as fp16 are not exact for the K types, whose scale is an fp16 superblock scale times a six bit integer. An exact narrow form is therefore per type, which is the situation today. KF8 instead spends bytes freely, int8 plus fp32 scales, because it never serves decode. Decode is bandwidth bound and stays on native bytes. Prefill is compute bound and the extra bytes are cheap.

## Layout

For a weight of `n_out` rows and `n_in` columns, with group size `G` of 16 or 32 fixed by the type:

| array | type | shape | meaning |
|---|---|---|---|
| `q` | int8 | `[n_out][n_in]` | the integers, `n_in` contiguous per output row |
| `d` | f32 | `[n_out][n_in / G]` | scale of each group |
| `m` | f32 | `[n_out][n_in / G]` | offset of each group, absent when the type has none |

Element `(o, k)` is `d[o][k / G] * q[o][k] - m[o][k / G]`, bit exact with `dequant_row` of the type, so `kf_backend_cpu::dequant_all` is the test oracle.

`G` is 32 when the native scale and offset are constant over every aligned 32, which holds for the `_0` and `_1` types, q4_K, q5_K, q8_K, the iq types with per 32 scales, the ternary types, mxfp4 and the one and two bit types. `G` is 16 for q2_K, q3_K, q6_K, iq2_xs, iq2_s, iq1_m and nvfp4. A test checks on random bytes that `G` of 32 types give equal `q16` scales for both halves of every 32.

`q` is plain row major so a 128 by 64 tile is 128 runs of 64 contiguous bytes, which `cp.async` moves in 16 byte pieces. The swizzle for conflict free `ldmatrix` happens on the way into shared memory, not in the stored form.

## Lifetime

The unpack kernel is bandwidth bound, native bytes in and about 1.5 times the element count out, so a 0.6B model converts in under 2 ms. Two policies share one code path:

- resident, the KF8 copy is made once at load and kept, chosen when it fits the device budget next to the native bytes
- transient, the KF8 copy is made into a scratch buffer just before each product and reused by the next weight, chunked over `n_out` so the scratch stays under 256 MB

Both exist because a 0.6B model has memory to spare and a 7B Q4_K_M on a 12 GB card does not. The plan decides per weight.

## The GEMM on KF8

`mma.sync.m16n8k16.s8` with int32 accumulate, 128 by 128 by 64 block tile, 8 warps each 64 by 32, three stage `cp.async` pipeline, `ldmatrix` fragments from XOR swizzled shared memory. Activations arrive from `quant_act` as int8 with an fp32 scale and an integer sum per sixteen, exactly the integers the CPU oracle multiplies back.

The expensive part of a fine grained int8 GEMM is not the mma, it is the scalar rescaling after it. Per output per 16 k the mma does 16 multiplies and the tensor pipe sustains 64 outputs per clock per SM, while `I2F` sustains 16. Three measures keep the scalar side off the critical path:

- the mma accumulates in int32 across every k16 whose weight scale and activation scale are both constant, 32 k for G of 32, so the conversion and the fp32 FMA happen once per 32 rather than once per 16
- int32 to f32 uses the magic number add and subtract, two full rate ops, instead of `I2F`, valid because an int8 by int8 sum over 256 stays under 2 to the 22
- the activation scale is applied once per activation block, 32 for Q8_0 and Q8_1 kinds and 256 for Q8K kinds, so the per group work is one FMA with the weight scale only

The offset term `m * sum(x)` is one more FMA per group per output and only compiles in for types with offsets. The kernel is templated on `G`, the activation block and the presence of `m`, eight instantiations in one module. For G of 16 types the scalar side is three ops per output per 16 k and the ceiling is lower, about 60 percent of peak, versus about 75 percent for G of 32. A later refinement for the K types with integer sub scales is an IMAD chain on the integer sub scale with one conversion per 256.

## Paths beside KF8

- f32, f16 and bf16 weights keep the fp32 tile. An exact fp16 and bf16 mma path for those is a separate change.
- The decode matvec stays per type on native bytes. The decode megakernel addresses launch overhead, which is the decode gap, not the matvec itself.
- The host side int8 expert path uses the same `q16` decomposition in Rust and is part of the MoE work.

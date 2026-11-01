# The multi row product on native bytes, verify passes and small batches on the tensor cores

A product of 2 to 16 activation rows against a quantized weight reads each weight byte once, decodes it once, and
multiplies it against every row through `mma.sync`, so a verify pass of k drafted tokens costs little more than one
decode step. One kernel template serves every block type through the lane loaders the one row matvec already has.

## Why

The one row matvec reads a weight at 650 to 800 GB/s. The dp4a kernel it had for 2 to 8 rows dotted each weight word
against every row from L1 and ran at a quarter of that, so an 8 row verify pass cost 3.9 decode steps and a drafter had
to be right most of the time to pay for itself. The KF8 tensor core tile reads the int8 form, 1.8 times the bytes of a
four bit weight, plus the unpack that writes it, so it only wins past 32 rows.

## Shape

A warp owns eight weight rows, four lanes to a row, and walks their columns four lane widths at a time, 256 columns for
the K types, so the lanes of a row read one whole block contiguously, as the matvec lanes do, and no step depends on a
cache line surviving to the next. Each lane decodes its columns into int8 words through `KfQw`, parks them in a warp
private tile in shared memory whose rows run sixteen bytes past the step, and `ldmatrix` turns the tile into the A
fragments of `m8n8k16`.

The activation rows come in two ways. When every warp of the block stands on the same step, the block holds a window of
the activation rows over a few steps in shared memory, with their scales and sums per run, refilled together every
`wsteps` steps, and `ldmatrix` turns the window into the B fragments as it does the weights, so a step reads no
activation through the caches. When the warps of a row group split a long row and stand on steps apart, the window
would hold a step per warp and no longer fit twice per multiprocessor, so the fragments and sums of each quarter of a
step come straight from the quantized rows through L1. In both cases no fragment lives past the quarter of the step
that uses it, which is what keeps the kernel at 128 registers and two blocks per multiprocessor. Entries exist for 8,
16, and 32 rows, and the native product beats the KF8 tile up to 32.

The chunks of one run under one weight scale chain through an accumulator seeded with 0x4B400000, so the int32 sum
lands in the mantissa, and `fma(sum, d, -magic d)` recovers the scaled sum exactly, the scale having under 23
significant bits. q8_K, whose scale is f32, subtracts first. The scaled sums of the runs inside one activation block
add up, the offsets times the activation sums come off for the five types that have them, and the block's activation
scale applies once. The activation block is the super block for the 256 wide types and 32 columns otherwise, both
known at compile time from the block size.

Eight warps make a block of 64 rows, or 64 over the warps per row group when the grid would be under two blocks per
SM, those warps splitting the steps and joining through shared memory. Grid y walks tiles of 8, 16, or 32 activation
rows over the same weights. Stacked products resolve their segment per eight row tile, so members start on multiples
of eight rows.

## Measured

Llama-3.2-1B Q4_K_M on an RTX 3080 Ti, tokens per second of `kungfu bench --verify`, every drafted row standing, with
the logits staying on the device as `docs/candidates.md` describes.

| rows | dp4a kernel | first tensor core kernel | now | pass cost in decode steps |
|---|---|---|---|---|
| 1 | 577 | 577 | 604 | 1 |
| 4 | 1160 | 1550 | 1898 | 1.27 |
| 8 | 1158 | 1950 | 3479 | 1.39 |
| 16 | 2455 | 2480 | 5116 | 1.89 |
| 24 | | | 5318 | 2.73 |
| 32 | | | 6618 | 2.92 |

Device microseconds of the products alone on the model's shapes, `cargo test --test bench_kernels product_kernels`,
which swing by a tenth between runs with the clocks of a desktop card.

| product | 1 row | 8 rows | 16 rows | 32 rows | KF8 tile at 40 rows with its unpack |
|---|---|---|---|---|---|
| q4_K gate and up, 2048 by 16384 | 28 | 46 | 63 | 99 | 118 and 79 |
| q6_K down, 8192 by 2048 | 25 | 39 | 52 | 73 | 159 and 47 |
| q6_K head, 2048 by 128256 | 266 to 300 | 320 to 400 | 385 to 525 | 520 to 630 | 745 and 707 |

With the tensor core work taken out of the kernel the same reads and decodes stream at 584 GB/s on the gate and up
shape and 773 on the head, against 668 and 803 for the one row matvec, so the memory pattern is not what holds the
multi row kernel below the floor. The back end is, and it was issue bound at two blocks per multiprocessor: 128
registers, since fragments and sums of a whole step lived at once, and sixteen loads of activation words per n tile
per step through L1. The window and the quarter sized fragments took the head at 16 rows from 583 to about 400 us and
the gate and up product from 95 to 63. Forcing three blocks per multiprocessor spilled 50 to 360 bytes of stack and ran
slower everywhere. A persistent grid of two blocks per multiprocessor walking the row groups, so the weight stream
never drained between groups, ran slower than one block per group, and n tiles in groups of two with the scales
shuffled once per quarter ran slower on q4_K.

Two layouts measured earlier. Thirty two rows per warp with one lane per row ran at 100 to 200 GB/s because a warp
touched 32 half used lines per step and the L1 under a 40 KB tile could not hold them to the next step. Sixteen rows
with two lanes per row ran at 300 to 400 GB/s for the same reason with half the lines.

## Next

- The per run work, two shuffles, two mma and the fold, is per eight rows. Staging the raw block bytes through shared
  memory with `cp.async` frees the registers of the decoded step in flight, and `m16n8k16` tiles halve the mma per
  byte, which is where the next third of the gap to one pass per verify lies.

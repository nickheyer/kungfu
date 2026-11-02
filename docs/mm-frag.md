# The fragment product, verify passes on native bytes at the weight stream

A warp owns sixteen weight rows and walks their super blocks through a cp.async ring in shared memory, reading each
row's bytes in the order the `m16n8k32` and `m16n8k16` integer tensor core fragments want them, so no decoded tile,
`ldmatrix`, or shuffle stands between the bytes and the mma. The activations come from a fragment array the
quantizer writes beside the int8 rows, in the same order, so one warp load touches two cache lines instead of eight.

## Why

The multi row product of `docs/mm-i8.md` decoded every byte into a shared tile, turned it into eight row fragments,
and folded each run through shuffles and scalar multiplies per eight rows. Its sixteen row pass cost 1.9 decode steps
on Llama-3.2-1B Q4_K_M, with the reads and decodes alone streaming near the matvec rate, so the back end was the
cost. Here the back end is one mma per thirty two columns and n tile, one integer multiply per C element and group,
and one fold per super block.

## Shape

Per type a `KfFrag<T>` plan gives the super block bytes, the shared memory pitch of a row, the ring stages and warps
of a block, the mma width, the scale of a group, the packed mins of a lane, and the fragments of a chunk of raw
bytes. Q4_K reads a row's eight bytes at lane offset 8 tig of each thirty two byte chunk, low nibbles for one group
and high for the next, which is exactly the lane's two A words of `m16n8k32` when the activations take the same
column order. Q6_K blocks sit two byte aligned, so each row's super block lands in a sixteen byte window at its own
skew and words come through a byte permute of the two around them.

The integer group sums multiply by the six bit scale as integers, the whole super block converting once. The min
term of Q4_K is a second product of the packed six bit mins against the group sums of the activations split into low
and high bytes, two mma per n tile and super block. Grid y walks tiles of 8, 16, 24, or 32 activation rows, and wpr
warps of a row group split the super blocks and join through shared memory.

The fragment array holds, per tile of eight rows and super block, every lane's eight byte B fragment in group order,
then each row's group sums as low and high bytes, then each row's super block scale, 280 bytes per row present, so
it sits inside the four bytes per element every quantizing producer already holds. `kf_act_row` derives it from the
row count and `kf_quant_span` writes it for passes of two rows or more over whole super blocks.

## Measured

Device microseconds of the product alone at 2048 by 16384 on an RTX 3080 Ti, `cargo test --release --test
bench_kernels frag_kinds -- --ignored --nocapture`, the tile being the product of `docs/mm-i8.md`.

| rows | q4_K fragment | q4_K tile | q6_K fragment | q6_K tile |
|---|---|---|---|---|
| 2 | 28.1 | 44.4 | 50.6 | 53.2 |
| 8 | 30.5 | 45.4 | 52.0 | 55.2 |
| 16 | 37.8 | 63.1 | 54.1 | 65.8 |
| 24 | 42.7 | 97.2 | 52.3 | 84.4 |
| 32 | 50.5 | 98.4 | 62.0 | 86.3 |

The q6_K head of Llama-3.2-1B, 2048 by 128256, takes 329 us at sixteen rows against 389 on the tile.

Tokens per second of `kungfu bench --verify` on Llama-3.2-1B-Instruct-Q4_K_M, every drafted row standing, and the
pass cost in decode steps.

| rows | t/s | steps per pass | before |
|---|---|---|---|
| 1 | 589 | 1 | 1 |
| 4 | 2342 | 1.01 | 1.27 |
| 8 | 4539 | 1.04 | 1.39 |
| 16 | 7358 | 1.28 | 1.89 |
| 24 | 9398 | 1.50 | 2.73 |
| 32 | 10722 | 1.76 | 2.92 |

With the mma, the activation loads, and the decode taken out, the q4_K ring streams at 633 to 679 GB/s against the
matvec's 674, so what remains above the stream at sixteen rows is the activation fragment loads at 5 us and the min
product at 2 us. The q6_K ring streams at 602 GB/s, its windows reading 240 bytes per 210, and its decode and
activation loads add 7 and 9 us. `cargo test --release --test bench_kernels frag_variants -- --ignored --nocapture`
times the kernel with each part taken out.

## Next

- Q6_K's activation fragments arrive one word per sixteen columns and n tile, since its scales change every sixteen
  columns. Activations as the A operand would take all sixteen rows in one load.
- Rows of the q6_K ring could copy 224 bytes at a pitch that keeps the loads conflict free.

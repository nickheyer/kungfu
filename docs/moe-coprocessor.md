# The expert coprocessor, routed experts split between device slots and host workers

A routed layer whose experts do not fit the device keeps a share of them in device slots and computes the rest on the
host, during the same pass, without the host thread ever waiting on the device. The device drives the host: the
dispatch kernel of a layer posts what it needs to a mailbox in mapped pinned memory, worker threads compute it, and
the await kernel of the layer reads the answer back. A decode pass stays one CUDA graph replay.

## Why

Reading a 1.5 MB expert from host memory costs about 120 us over PCIe and about 30 us across the cores from DRAM, so
a missed expert is computed where its bytes are and never moved for one token. Routing is skewed: on Qwen3-30B-A3B
the most selected 62.5 percent of experts take about 95 percent of the selections, so slots holding them turn most
calls into device work and the host sees a few experts per token. The old path synchronized the host thread three
times per layer and dequantized every missed row through the reference code, which cost more than the experts.

## Device side

`moe_route` picks the experts and weights as before. `moe_dispatch` maps every routed pair through the layer's slot
table, a device array from expert to slot or minus one, writes the slot per pair for the expert kernels, and posts
the selections, the misses as row, expert and weight triples, the input rows as f32, and the pass number to the
layer's mailbox. The slot products run on the device as before. `moe_await` spins on the mailbox's `done_seq` until
it equals the pass number, copies the host partial into device scratch, and `moe_combine` adds it. A layer without
misses never waits.

The pass number is one device word the host writes before each pass and the kernels compare against, so a graph
replay needs no new arguments. An await that waits two seconds writes an error word the host reads at the end of the
pass and fails the pass.

## Mailbox

One per routed layer, pinned host memory mapped into the device address space.

| offset | content |
|---|---|
| 0 | `req_seq` u32, the pass that posted, written last with a system fence before it |
| 4 | `n_miss`, `rows` |
| 64 | `done_seq` u32, the pass answered, `err` |
| 128 | selections, 256 i32 |
| 1152 | misses, 256 of row i32, expert i32, weight f32 |
| 4224 | input rows, 8 by `n_embd` f32 |
| after | partial sum rows, 8 by `n_embd` f32 |

## Host side

Worker threads, half the hardware threads so none loses a scheduler slice mid job, spin while a pass runs and sleep
between passes. The first
thread scans the mailboxes. A posted layer with misses becomes a job: misses grouped by expert, the rows each expert
needs quantized into the int8 form the expert's quant type takes, the gate and up rows in chunks of 32 any worker
claims, the hidden rows quantized per expert, then the output columns in chunks of 64 any worker claims, each chunk
summing every missed expert into the partial rows with the routing weights. Phases end when every unit is done, not
when every thread arrives, so a thread the scheduler holds back delays only its own chunk. The leader writes
`done_seq` after the last column unit.

The dot products are AVX2 kernels per quant type in kf-quant, `dot_i8`, int8 weights unpacked once per 32 columns
and multiplied against up to four activation rows at once, exact integer sums scaled once per block. Every block
type has one. A whole 1.5 MB expert costs 65 to 80 us with 16 threads.

The expert bytes are read straight from the model mapping, which is populated at load so no page faults land in a
pass.

## Slots

The plan gives every routed layer the same share of its experts in slots. At load the slots take the most selected
experts of the routing profile when one is given. At every pass boundary, with the device idle, the host reads each
mailbox, notes the pass each selected expert was last used in, and gives the missed experts slots: free slots
without limit, held slots only to an expert missed twice within sixteen passes, which takes the slot of the expert
selected longest ago that this pass did not select, at most sixteen such replacements per pass. The host unmaps the
holder on the compute stream and hands the copy to a copier thread, which stages the expert's bytes through a pinned
ring and issues the copy and the new table entry on the copy stream, so the pass that follows misses either expert
until the bytes are in place and never reads a slot being written, and the host thread spends microseconds. A
prefill pass runs the slot resident experts in place, streams the rest through the device, and asks for its most
used experts the same way. Selection counts halve every 4096 passes and feed the routing profile written at exit.

## Speculation

The server drafts without a model: the continuation of the longest repeat of the context's tail inside the context,
else a chain through a table of three grams and the model's own most likely successor of each token learned from
every prompt and every row of logits, up to the draft length, which grows while rounds accept everything. One pass
verifies the drafted rows, the request's own sampler deciding each, so the output distribution is the model's. Passes
of up to eight rows replay from graphs like decode steps, so a verify pass of k rows costs about 1 plus 0.06 k decode
steps on Llama-3.2-1B, see `docs/mm-i8.md`.

## Measured

Qwen3-30B-A3B at 2.06 bits on an RTX 3080 Ti with 62.5 percent of the experts in slots, server measured on natural
prompts: 112 to 129 tokens per second against 16 before and 52 to 55 for llama.cpp with its best expert split, with
28 of 384 expert calls per token computed on the host after the slots adapt. Prefill 1300 to 2500 against 1340.
Per token the host workers run about 2.6 ms and the device kernels about 6 ms, of which the expert products, the
attention at short context, and the routing kernel are each far from the memory floor and are the next work.

# Candidates, the logits of a pass stay on the device

A pass hands back the top 64 candidates of every row of logits with the row's log sum exp, 129 floats a row, and
the whole row crosses only when a sampler proves it needs it. The rest of the row stays in device memory until the
next pass.

## Why

A verify pass of k rows wrote k times 128256 floats, 4 MB for 8 rows and 8 MB for 16, into pinned memory and then
into a vector the sampler read once. That was 0.8 ms of a 3.4 ms pass of 8 rows and 1.5 ms of 16, more than every
kernel of the pass but the head product, and the host then sorted 128256 values per row to sample.

## The op

`Op::TopK { k }` closes every head graph. On the device `topk_slices` cuts a row into slices of about eight thousand
columns, one block of 1024 threads each, and every slice selects its own k largest exactly by a radix select over a
composite key, the value's order preserving bits above the complemented id, so keys are unique and a tie between equal
logits goes to the lower id without a rule of its own. The slice's first pass takes its maximum and sum of exponentials
and the bits every key shares at the top, so the digits start where the keys part. Each later pass takes eight bits
into a histogram private to a pair of warps, and once the keys that can still hold the k th largest fit the gather
space they come into shared memory and the remaining digits resolve there. `topk_merge` ranks the slices' candidates,
which hold the row's k largest, and joins the slices' sums. The CPU backend selects the same way in Rust and is the
test oracle, ids and values compared exactly.

One row of 128256 logits takes 25 us, sixteen rows 60 us, against 255 us for one row through a single block with one
shared histogram, where the few bins a row of logits fills serialized the atomics.

## The sampler

`Candidates` carries the ids, their logits, which of them a bias, penalty, or mask moved, the floor, which is the k th
logit and so a bound on every other token, the log sum exp, and the row's index in its pass. `sample_sparse` returns
the dense sampler's token when the candidates settle it and nothing otherwise: it applies the repeat penalty and the
bias to the candidates it holds, orders them, and takes the prefix the row placed above every outside token, an
untouched candidate standing anywhere and a moved one only above the floor. Greedy sampling is settled when that prefix
holds one token, and a top k sampler when it holds the top k, after which the draw runs on the same values in the same
order as the dense sampler, so the token and the generator state match it bit for bit. A penalty below one, a positive
bias on a token outside the candidates, a top k of zero, or a top k past the prefix are not settled, and the caller
fetches the whole row through `Session::dense_row` and runs the dense sampler, which the test fuzzes against.

Log probabilities come from the candidates when they hold the token and the top asked for, and from the whole row
otherwise. The server settles every job's row before any job's verify pass, since a pass replaces the rows on the
device, and the chooser of a verify pass fetches rows of that pass alone.

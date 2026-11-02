# Tree verification, one pass over a tree of drafts

A drafter proposes a tree of tokens rooted at the token the sampler already chose, and one pass scores every row of
the tree. Each row attends to the slot's history and to its ancestors in the tree alone, through a mask the pass
carries per row, and takes the position its depth gives it. The sampler then walks from the root: its own pick at a
row stands when a child holds it, and the pick that no child holds is the model's next token. The accepted path's
cache rows move down to where they would have stood fed one after another and every other row leaves the cache.

## Why

A chain of drafts stands only while every token is right. A tree hedges the positions a drafter is unsure of with
its next best tokens, so a wrong guess at depth one no longer ends the round, and a verify pass of sixteen rows costs
1.3 decode steps through the fragment product of `docs/mm-frag.md`, so the rows are cheap. Every drafter kungfu has, the ngram drafter and the learned head of
`docs/drafter.md`, proposes trees.

## The pass

`Inputs::tree` holds one `TreeRow` per row: a 64 bit mask whose bit k stands for the key k below the row's last key,
every bit at or past the pass's row count set so the history is seen, and the row's position for the window rule,
since the row's cache index runs past its position. The session builds the rows from the parents, numbers the cache
indices in row order, gives every row the rotary position of its depth through the rotary streams, and points every
row's last key at the tree's last row. A tree pass holds at most 64 rows and one slot.

Both attention backends read the mask. On the device the decode kernel and the classic kernel take it as their last
argument and score a masked key as minus infinity, so it falls out of the softmax, and a tree pass always runs the
decode kernel, since the tensor core prefill kernel reads no mask. The pass keeps its own prepared graph, so a tree of
n rows and a plain pass of n rows never share kernels or captures, and tree passes up to 64 rows replay from graphs.

## The path

`Backend::compact(seq, base, keep)` moves the cache rows base plus keep[i] to base plus i in every layer and drops the
rest. On the device one block per layer reads every kept row into shared memory before any is written, since a row a
later index keeps may stand where an earlier one lands. The host backend does the same in place after reading the
rows out, and a recurrent state refuses, having no rows to keep. `Session::accept_path` runs it and rewrites the
slot's tokens and rotary positions to the path.

## Verified

`tests/attention.rs` feeds an eight row tree with three branches after 48 rows of history, on the classic kernel at
head size 16 and on the decode kernel at head size 64 with rope, and checks that every row equals the same path fed
alone on both backends, that the backends agree, and that a step after keeping one path equals a step after feeding
that path alone. The server's greedy answers with trees differ from plain decoding where two tokens lie within the
last bits of each other, since the multi row kernels and the one row kernels round differently: over 508 positions of
one sequence the eight row pass picks another token than the one row pass at 5, the sixteen row pass at 11, and one
prefill pass over the whole sequence at 8, which is the same spread llama.cpp's batched verification shows.

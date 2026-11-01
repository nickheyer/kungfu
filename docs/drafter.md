# The learned drafter, a draft head kungfu teaches itself

A draft head predicts the model's next hidden state from the current one and the token the sampler chose, so a tree of
drafts can be read off the model's own head without running the model. Nothing about it names a model family: the
head's shapes come from the embedding width and the vocabulary, its training data from the rows the server scores
anyway, and its weights live in the cache directory next to the routing profile, so a restart is warm and no file
ships with kungfu.

## What a pass provides

Every head pass keeps its residual rows before the final norm, `Graph::hidden`, through the pass and returns them
with `Backend::hidden_rows`. Once a head is started the device keeps a ring of the last 16384 scored rows with each
row's slot, position, and token, `features::FeatureRing`. Two rows of one slot at neighbouring positions in the same
epoch make a training pair, the epoch of a slot rising on every rollback so no pair crosses one. A tree pass
contributes the path compaction keeps, in the positions it then holds. `kernels/train.cu` holds the kernels a head
learns with, a single precision product with either operand transposed, a skinny product for the few rows of a draft
that reads each weight once, the gated activation and the rms norm with their backward passes, AdamW, and the smooth
L1 loss, each checked against the reference in `kf_backend_cpu::train`.

## The head

`HeadConfig::for_width` gives a model of width d one attention layer over a window of the last 64 pairs, heads of
128, and a gated MLP twice as wide, 12 d squared parameters. The head's prediction is scored against the model's own
head rows over a draft vocabulary of the 32768 tokens the ring has seen most, recounted every 256 steps, and the four
most likely tokens become a node's children at the first depth, two at the next two, one after that, `DRAFT_FANOUT`.
A training step takes four chunks of 64 consecutive rows and reports the loss before the update and the hit rate, the
share of the chunks' positions whose actual next token stood among the four children the head would have drafted.
The host head in `kf_backend_cpu::drafter` is the reference the device head is checked against step for step, draft
for draft.

## At the server

`--speculate head`, what `auto` means on a device, starts the head over the model's embedding and head weights when
the model is loaded, after the plan has kept its bytes free, `HeadConfig::device_bytes`. `--speculate ngram` or
`off` leaves the head and its ring out, and the cpu backend, the reference, takes `ngram` or `off` alone.

Training follows the rows. Every row the ring receives earns two passes over itself, so the head trains two epochs
over everything the server scores. Between requests the scheduler trains on that credit until it is spent, and while
requests run it trains one step every so many decode passes that training stays under a tenth of the decode time.
With a cache directory the head is saved every ten minutes it trained and when the server stops, as `draft_head.bin`
beside `draft_head.json`, which names the model, its width and vocabulary, and the steps, loss, and hit rate reached,
so a head is loaded only into the model it learned.

Each decode pass hands the head every slot's row. A round of a slot brings the head's window up to the slot's latest
run of positions in its epoch, drafts a tree of `--head-rows` rows from the held row and the sampler's token, grafts
the chain the ngram drafter reads off the context's repeats into it, and the pass verifies the tree as any other, see
`docs/tree-verify.md`.

Which drafter a round takes is measured, not assumed. A drafted round yields the sampler's token, the accepted path,
and the model's next token over the decode pass, the drafting, and the verify pass it took, and plain decoding yields
one token a decode pass. The scheduler keeps the measured tokens a second of the head's rounds and of the ngram
drafter's, runs a drafter without a measure until it has one, runs the head only once its hit rate reaches a quarter,
picks the fastest of head, ngram, and plain decoding for each round, and every eighth round runs the drafter it did
not pick so its measure stays fresh. An untrained head therefore costs nothing but its training time, and a head that
pays takes over as soon as it does.

`/metrics` reports `kungfu_head_steps_total`, `kungfu_head_loss`, `kungfu_head_hit`, the measured rates
`kungfu_draft_rate_head` and `kungfu_draft_rate_ngram`, and `kungfu_draft_rounds_total` and
`kungfu_draft_accepted_total` by source. The `drafter` event at start names the parameters and whether a saved head
was loaded, each save logs its steps, loss, and hit rate, and a request's event carries `head_rounds`.

## The study

`kungfu drafter-data` answers prompts greedily in parallel slots and writes every token's hidden state, so the text
token at every position is the model's own argmax and chained acceptance on it is the acceptance speculation would see.
`tools/drafter_study.py` fits three heads on those rows and chains each on held out sequences.

Llama-3.2-1B Q4_K_M, 2510 answers to generated prompts, 1.2 million response tokens, 60 answers held out, two epochs:

| head | parameters | accepted per round | top 1 at depth 1 to 5 | top 4 at depth 1 to 5 |
|---|---|---|---|---|
| ridge regression, closed form | 8 M | 0.35 | 0.28 0.24 0.14 0.13 0.15 | 0.51 0.42 0.29 0.27 0.23 |
| gated MLP, regression plus the head's KL | 21 M | 0.69 | 0.54 0.25 0.15 0.09 0.10 | 0.76 0.45 0.30 0.22 0.14 |
| one attention layer over 64 features | 75 M | 0.72 | 0.52 0.29 0.24 0.24 0.22 | 0.75 0.51 0.40 0.37 0.34 |

The attention head keeps predicting when it chains on its own output and the MLP does not, which is what a tree needs.
Both were still learning when the epochs ran out, the regression loss at 0.08 and 0.07 and falling.

## What the numbers mean here

A round of this drafter on this model accepts about 1.1 tokens with a tree of four children at depth one, against 1.9
decode steps for the sixteen row pass and the head's own reads: the head at 50 million parameters reads 200 MB per
depth as the skinny product streams it, a quarter of the model, and the draft vocabulary of the 32 thousand most
frequent tokens another 54 MB. On a 1B model the head costs more than it saves until it accepts two or more tokens
per round, which this amount of training does not reach, and the server's measure keeps the ngram drafter in charge
until it does. On an 8B model the head has 200 million parameters and reads a sixth of the model per depth, so a
round of five depths and sixteen rows costs about four decode steps and pays from about two accepted tokens, where
the published heads reach three to four. A serving process adds about two million rows an hour to the ring, so the
hours a process serves decide when the head takes over.

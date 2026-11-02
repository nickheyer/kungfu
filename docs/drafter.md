# The learned draft block, a block drafter kungfu teaches itself

A draft block predicts the model's distribution at the next five positions in one pass from the current row's
features and the token the sampler chose, so a tree of drafts comes from one read of the block's weights. Nothing
about it names a model family: the block's shapes come from the embedding width and the vocabulary, its tap layers
from the first rows the server scores, its training data from the rows the server scores anyway, and its weights
live in the cache directory next to the routing profile, so a restart is warm and no file ships with kungfu.

## What a pass provides

Every pass keeps the residual rows after the block's tap layers, `Graph::taps`, and returns them with
`Backend::tap_rows`. Once a block is started the device keeps a ring of the last 16384 scored rows with each row's
slot, position, token, and parent among the ring's rows, `kf_core::RingMeta`, the same bookkeeping on every
backend so all sample the same chunks. A plain pass's rows join the ring on the model's own path below the slot's
last row. A tree pass's rows all join as branches below the row the tree hung from, and the path compaction keeps
moves onto the model's path at the positions it then holds, so the rejected rows stay with the model's
distribution after them. A slot's epoch rises on every rollback so no chunk crosses one.

## The block

`DraftConfig::for_width` gives a model of width d one attention layer over a window of the last 64 rows, heads of
128, and a gated MLP twice as wide, with a fusion of the tap rows in front and a learned mask row per drafted
position after the first. A draft runs the accepted path's rows, each the features of a tree row with the token
that followed it, then the mask rows, which attend to the path and to each other, and reads every drafted
position's logits through the model's own head over a draft vocabulary of the 32768 tokens the ring has seen most.
The tree takes the positions' top candidates by product probability, `kf_core::tree_of_block`, up to
`--head-rows` rows.

The pass multiplies through an int8 snapshot of the block, every matrix transposed into q8_0 rows and read by the
model's own int8 product, so one round reads the block once. Training stays in single precision.

## Training

A training step takes four chunks of 64 consecutive rows of the model's path. Every row drafts two query groups
from the same features: the main group's first token is the token that followed and its targets are the model's
candidates at the next five positions along the path, the branch group's first token is a rejected row below it
and its targets follow that branch, so the block learns from the positions the model rejected as well. The main
group minimizes `--head-loss`: `edr` the expected decoding rounds, every position's rejection cost weighted by the
occupancy of the round that reaches it with the value difference of accepting there, the chunk being the
sequence; `bv` the log of the expected accepted length under block verification, the vocabulary integrated form
along the model's path; `ce` the cross entropy at every position. The branch group always minimizes the cross
entropy.

The step keeps the 40 percent of the targets whose cross entropy exceeds a reference's by the most, the reference
a moving average of the block, and trains on those alone. `--head-learners` learners step side by side at rates
spread by factors of two with optimistic AdamW, the parameters played a step of the last direction ahead of the
trained point, and a hedge over their running cross entropies weighs them into the mixture the block drafts from,
which the snapshot follows after every step. The loss reported is the mixture's, the hit rate the share of main
group rows whose actual next token stood among the four children the block would have drafted first.

The host block in `kf_backend_cpu::draft` is the reference the device block is checked against, chunk for chunk,
gradient for gradient, and tree for tree, `tests/train.rs`.

## The tap layers

Before the block starts, every pass keeps the residual rows of every layer and the top 30 percent of each pass's
rows by predictive entropy join block averages of 32 rows per layer. Once 4096 rows are kept, `--head-taps` layers
are chosen by beam search over the matrix entropy of the concatenated averages, the expansion of a set taken from
the layers of highest entropy gain under `--head-select elsi`, or from the layers of highest centered kernel
alignment transition around it under `elsi_c`. The passes are rebuilt to keep the chosen layers and the block
starts over them, the layers saved beside the weights so a restart skips the selection.

## At the server

`--speculate head`, what `auto` means on a device, reserves the block's bytes beside the plan for as many learners
as leave the model a context, selects the tap layers from the first requests, and starts the block when they are
chosen, or loads the saved block when the cache directory holds one of this model.

Training follows the rows. Every row the ring receives earns two passes over itself, so the block trains two epochs
over everything the server scores. Between requests the scheduler trains on that credit until it is spent, and while
requests run it trains one step every so many decode passes that training stays under a tenth of the decode time.
With a cache directory the block is saved every ten minutes it trained and when the server stops, as
`draft_head.bin` beside `draft_head.json`, which names the model, its width, vocabulary, tap layers, depth, and the
steps, loss, and hit rate reached.

Each round the block's pass carries the accepted path into its history and, once the hit rate reaches a quarter,
drafts a tree. Four arms play: the block's tree, the ngram drafter's, the block's tree with the ngram chain grafted
in, and plain decoding. The block's arms are awake once it is warm and the ngram's with an ngram drafter. A normal
hedge over the awake arms picks the round's arm, an arm not yet measured running first, and a sleeping arm's regret
stands. Every round opens a measurement holding each awake arm's tree, the verify pass's seconds or one decode pass
for the arm it played. Every distribution the slot sees afterwards, from a verify pass or a decode pass, walks each
open measurement one position, each depth's acceptance the mass the distribution put on the tree's children along
the tokens committed. Once every walk has left its tree or reached a leaf the measurement is observed: each arm's
loss is one less its tokens a second over the most a round could yield, so the hedge learns every arm's full
expected length each round. A round whose request has too little room left for every arm's tree runs its cut tree
without a measurement, and a request's end drops the measurements still walking.

`/metrics` reports `kungfu_head_steps_total`, `kungfu_head_loss`, `kungfu_head_hit`, `kungfu_draft_rate` and
`kungfu_draft_weight` by arm, and `kungfu_draft_rounds_total` and `kungfu_draft_accepted_total` by source. The
`drafter` event names the parameters, learners, and rows at start, the tap layers when chosen, and each save's
steps, loss, and hit rate, and a request's event carries `head_rounds`.

## The study

`kungfu drafter-data` answers prompts greedily in parallel slots and writes every token's final residual row, the
rows of the layers `--taps` names, and the model's candidates, so the text token at every position is the model's
own argmax and chained acceptance on it is the acceptance speculation would see. `tools/drafter_study.py` fits
heads on those rows and chains each on held out sequences.

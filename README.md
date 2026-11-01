# kungfu

Inference runtime based loosely on STRATA. Teaches your inference server Kung-Fu.

## Install

```sh
cargo install --path crates/kungfu
cargo install --path crates/kungfu --no-default-features
```

## Usage

```sh
kungfu inspect --json model.gguf
kungfu describe
kungfu serve --model model.gguf --host 127.0.0.1 --port 8080 --ctx auto --parallel 4
kungfu serve --model model.gguf --plan plan.json --cache-dir cache --profile cache/routing_profile.json
kungfu serve --model vl.gguf --mmproj mmproj.gguf --port 8080
kungfu serve --model model.gguf --role stage --layer-from 8 --layer-to 20 --port 9001 --admit 10.0.0.1
kungfu serve --model model.gguf --role head --peer stage=0@10.0.0.2:9001 --port 8080
kungfu serve --model model.gguf --role prefill --slot-dir slots --port 8081
kungfu serve --model model.gguf --role draft --draft-model small.gguf --port 9002
kungfu serve --model model.gguf --role head --peer draft=0@10.0.0.3:9002 --draft-max 4 --port 8080
```

## Current State

Server measured tokens per second on one RTX 3080 Ti, 1k token prompt, 128 generated, llama.cpp at its best flags for the file.

| Model | llama.cpp prefill | kungfu prefill | llama.cpp decode | kungfu decode |
|---|---|---|---|---|
| Llama-3.2-1B Q4_K_M | 26319 | 29510 | 537 | 546 to 620 |
| Qwen3-30B-A3B IQ2_XXS, experts partly on the device | 1296 | 3533 | 49 | 128 |

Greedy top 1 agreement with llama.cpp over 320 teacher forced positions: Llama-3.2-1B 99.1 percent, Qwen3-30B-A3B 94.4 percent, where llama.cpp's own CPU backend agrees with its CUDA run at 96.2 percent on that file.

The routed experts a layer's device slots miss are computed by host worker threads driven by the device through mapped memory, with no host synchronization inside a decode pass, see `docs/moe-coprocessor.md`. Prefill runs the slot resident experts in place and streams only the missing ones. Decode drafts a tree of tokens and verifies the whole tree in one pass, each row attending to its ancestors alone, see `docs/tree-verify.md`. The trees come from the context's own repeats and the model's recycled candidates, and from a draft head the server teaches itself on the rows it scores, saved beside the routing profile so a restart is warm, each round taking whichever drafter measures faster, see `docs/drafter.md`. On by default, `--speculate off` turns it off. The logits of a pass stay on the device and each row comes back as its top 64 candidates with the row's log sum exp, the whole row crossing only when a sampler proves it needs it, see `docs/candidates.md`.

What is still short

- Dense decode sits at parity. The persistent decode kernel runs 1.85 ms per token on Llama-3.2-1B against a 1.03 ms memory floor and is slower than graph replay, since every stage ends in a grid barrier. Per stage dependency counters, weight loads issued before the waits, and a short context attention path are the work.
- The device side of a routed decode layer spends about 7 ms per token in 23 small launches where the memory floor is 1.7 ms. The expert products and the attention at short context are the largest.
- A missed expert costs the host 65 to 80 us with 16 threads, bound by the codebook unpack of the two bit types, about half of memory speed. Natural text misses about 28 of 384 expert calls per token once the slots adapt.
- A verify pass of 8 rows costs 1.4 decode steps on Llama-3.2-1B, 1.9 for 16 rows and 2.9 for 32, through the tensor core product on native bytes with the activations windowed in shared memory, see `docs/mm-i8.md`. With every drafted row standing that is 3480, 5120, and 6620 tokens per second against 604 one row at a time. The server decodes text that copies its context at 2.7 times the plain rate and code edits and open ended text at the plain rate, since the ngram drafter holds only on repeats, so the pass is no longer what bounds speculation and a drafter that holds on open text is. The multi row product runs at 400 to 600 GB/s against 650 to 800 for one row, issue bound at two blocks per multiprocessor.
- The learned head drafts only where it measures faster than the ngram drafter and plain decoding, see `docs/drafter.md`. Trained offline on 1.2 million of Llama-3.2-1B's own tokens it accepts 0.72 tokens a round and is still learning, where a head pays for itself from about two accepted tokens a round, so the measure keeps the ngram drafter in charge and the head waits on the hours a process serves.

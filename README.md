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

Performance, 1k-token prompt, 128 generated, same GPU, server-measured tokens per second

┌───────────────────────────────────────────────┬───────────────────┬───────────────┬────────────┬──────────────────┬───────────────┬────────────┐
│                     Model                     │ llama.cpp prefill │ kungfu before │ kungfu now │ llama.cpp decode │ kungfu before │ kungfu now │
├───────────────────────────────────────────────┼───────────────────┼───────────────┼────────────┼──────────────────┼───────────────┼────────────┤
│ Qwen3-0.6B Q8_0                               │ 31270             │ 3903          │ 7057       │ 434              │ 313           │ 300        │
├───────────────────────────────────────────────┼───────────────────┼───────────────┼────────────┼──────────────────┼───────────────┼────────────┤
│ Llama-3.2-1B Q4_K_M                           │ 25427             │ 2327          │ 5328       │ 523              │ 265           │ 254        │
├───────────────────────────────────────────────┼───────────────────┼───────────────┼────────────┼──────────────────┼───────────────┼────────────┤
│ Qwen2.5-0.5B Q8_0                             │ 34308             │ 5583          │ 10359      │ 537              │ 377           │ 353        │
├───────────────────────────────────────────────┼───────────────────┼───────────────┼────────────┼──────────────────┼───────────────┼────────────┤
│ Gemma3-270m Q8_0                              │ 41200             │ 12794         │ 26302      │ 585              │ 403           │ 418        │
├───────────────────────────────────────────────┼───────────────────┼───────────────┼────────────┼──────────────────┼───────────────┼────────────┤
│ Qwen3-30B-A3B IQ2_XXS, experts partly on host │ 1424              │ 41            │ not rerun  │ 72               │ 10            │ not rerun  │
└───────────────────────────────────────────────┴───────────────────┴───────────────┴────────────┴──────────────────┴───────────────┴────────────┘

The prefill change is the int8 tensor-core matmul. Every block quant type now gives `q16`, sixteen integers with a scale and offset, and `matmul_i8` runs a 64 by 64 by 64 tile through `mma.sync.m16n8k16.s8` on sm_80 and above. Activations are quantized once per product by `quant_act` into int8 with per sixteen scales and integer sums, the same integers `round_act` would multiply back, so each sixteen wide step is one exact integer product scaled once and the result still matches the CPU oracle at 1e-5 for all 27 block types. Rows of nine and more take this path, fewer rows keep the warp matvec. Floats and devices below sm_80 keep the fp32 tile. All 37 workspace tests pass, with the new one covering every int8 type at partial tiles and partial k steps with and without bias.

What is still short

- Prefill is 3 to 5 times slower than llama.cpp. The int8 tile is a plain shared memory loop with no double buffering or ldmatrix, and the weight side still unpacks through scalar byte loads per sixteen. Attention is the other half at long prompts, a 2.5k prompt prefills at 4700 on Qwen3-0.6B against 7000 at 1.1k.
- The MoE file is 7 times slower on decode and 35 times slower on prefill. The host-side expert path dequantizes every row per call through the generic oracle code, so the CPU side is the bottleneck, not the device. The expert matmul on the device still uses the fp32 tile and could take the int8 one.
- Decode sits at 60 to 70 percent of llama.cpp. The remaining gap is about 750 kernel launches per token, mostly activation rounding before each matvec, which can be fused away.

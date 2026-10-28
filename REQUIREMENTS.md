# kungfu requirements

Part 1 lists what the build is missing or wrong today, by crate, with the contract each piece must meet. Part 2 specifies the finished binary from nebu's side of the table, so nebu can install, plan, launch, watch, and form it like any other runtime, and better than the ones it has.

## Part 1. Missing and broken

The build fails at every unwritten piece by calling a function that does not exist, named for the work. `cargo check --workspace` is the live list. It reports kf-quant and kf-layers first and hides the rest until those compile.

### kf-quant, 21 dequant kernels

`crates/kf-quant/src/types/<type>.rs` each call `dequant_row_<type>_unwritten(src, dst)`. Each must become a real `dequant_row` for the block layout in ggml-common.h at the commit nebu's llama.cpp recipe pins, bit exact with ggml's `dequantize_row_<type>`.

| Types | Block, bytes |
|---|---|
| q2_K, q3_K, q4_K, q5_K, q6_K, q8_K | 256, 84 110 144 176 210 292 |
| iq2_xxs, iq2_xs, iq3_xxs, iq1_s, iq3_s, iq2_s, iq4_xs, iq1_m | 256, 66 74 98 50 110 82 136 56 |
| iq4_nl, mxfp4 | 32, 18 17 |
| nvfp4 | 64, 36 |
| tq1_0, tq2_0 | 256, 54 66 |
| q1_0, q2_0 | 128 18, 64 18 |

The i-quant types need the codebook grids from ggml-common.h copied in under the MIT notice. Every type with a cheap integer dot product (q4_K, q5_K, q6_K, q8_K, iq4_nl, iq4_xs) also overrides `dot_row` so the CPU path does not dequantize per call.

Done when a row dequantized here equals ggml's output for the same bytes.

### kf-layers, 12 graph builders

Each layer type below has real weights, `weights()`, and `state()`. Its `forward` calls a graph builder that does not exist. Each builder emits ops into the graph and returns the output tensor. New ops are added to `Op` in kf-core with a CPU implementation first.

| Function | Layer | Must emit |
|---|---|---|
| `gated_delta_net_graph` | DeltaNet | fused qkvz projection, short causal conv with carried conv state, gated delta rule recurrence with carried state, gated rms norm, output projection, chunked path for prefill and single step path for decode |
| `kimi_delta_attention_graph` | Kda | separate q k v with three convs, f_a f_b and g_a g_b low rank gates with the lower bound, beta, delta recurrence, norm, output |
| `mla_attention_graph` | Mla | q through the low rank pair or full, kv through kv_a_mqa and norm, rope on the rope dims only, absorbed k_b v_b against a latent cache of rank plus rope dims, output, optional indexer |
| `sparse_attention_graph` | Attention with DsaIndexer | indexer scores every key through k, q, optional per head proj, keeps top_k, attention reads only those keys, kpool compression when present |
| `compressed_attention_graph` | CompressedAttention | DeepSeek V4.1 kv stream through the compressor at its ratio with gate and ape, low rank q, sinks, grouped low rank output, source layer sharing per `dsv4.kv_source` and `dsv4.topk_source` |
| `hyper_connection_block_graph` | HyperConnected | n residual streams, mix before the mixer and before the feed forward with fn base scale or norm down up inject, sinkhorn normalization when configured, first block expands one stream to n |
| `hyper_connection_head_graph` | HcHead | collapse n streams to one through norm down up, or fn base scale |
| `hyper_connection_mean_graph` | HcMean | mean of n streams |
| `ple_ngram_graph` | Ple | hash the last ngram tokens per head with offsets and vocab sizes into the per layer table, key value through norms, short conv, multiplied into the stream |
| `engram_graph` | Engram | DeepSeek V4.1 hashed n gram memory with its token map and pad, kv and gate per stream |
| `nextn_draft_graph` | NextN | enorm of the embedded next token and hnorm of the trunk state concatenated through eh_proj, then the block, then the shared or main head |
| `moe_latent_graph` | Moe with Latent | latent down, pre norm, routed experts in the latent width, post norm, latent up, Muse Spark |

Done when `kungfu generate` on a file of each family is greedy top 1 identical to llama.cpp on the same token ids for 32 tokens on 20 prompts, at least 99 percent agreement, and `kungfu inspect` claims every tensor.

### kf-arch

`longrope_factors_select(hp, w)` must pick rope_factors_long or rope_factors_short by the context length against the trained context and return it as the rope factors weight.

### kf-backend-cuda

The registry compiles one real module. The backend cannot run.

| Function | Must |
|---|---|
| `cuda_kernel_for_op` | map every `Op` variant and quant type to a kernel, missing kernels return none and the op falls back to the host |
| `cuda_run_graph` | execute a graph on the stream with weights uploaded once and cached by Wid, activations resident on the device, the output copied back, CUDA graph capture for the decode shape |
| `cuda_truncate_state` and `cuda_reset_state` | the kv and recurrent state the backend owns |

Kernels live as CUDA C text under `src/kernels/`, compiled through NVRTC with a PTX cache keyed by source hash and compute capability. Release builds embed PTX for sm_75 through sm_120 so no toolkit is needed at runtime. Required kernel families: quantized matmul and matvec per quant type, attention with cache, rope, norms, gated activations, routed expert matmul over a cache of resident experts, delta net recurrence, sampling on device.

### kf-engine

| Function | Must |
|---|---|
| `plan_device_placement` | from the model's weights and states, the device's free bytes, and the context, place per layer weights and state on device until the budget is spent, experts last, leaving an expert cache slot count per layer for what did not fit, and return a `Plan` whose bytes nebu can check against measurements |
| `expert_cache_warm` | fill slots from a routing profile, most used experts first |
| `expert_cache_lookup` | slot of an expert, evicting by least recent use, counting hits and misses |
| `speculative_verify` | feed drafted tokens in one pass, accept the longest prefix the model agrees with, return it plus the model's own next token |

Beyond the named functions the engine still lacks: pinned host memory for experts with device copies of the cached ones, the CPU pool computing uncached experts while the device computes cached ones, prefetch of the next layer's experts during attention, kv streaming from host past a resident window, a prefix cache across requests, parallel sequences, and a persisted routing profile.

### kf-serve

| Function | Must |
|---|---|
| `sentencepiece_from_gguf` | the `llama` tokenizer model from tokens, scores, and token types with byte fallback and the add_space_prefix rule |
| `pretokenizer_regexes_for` | every remaining `tokenizer.ggml.pre` name in llama-vocab.cpp, the exact regex list in order |
| `parse_glm_xml_tool_calls` and `parse_glm_xml_stream` | `<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>` |
| `parse_deepseek_tool_calls` and `parse_deepseek_stream` | the DeepSeek V3 unicode marker format |

The head also lacks Anthropic messages, the OpenAI responses endpoint, image input, `/metrics`, and request cancellation when the client disconnects.

### Known wrong today

- `CpuBackend` is correct and slow, it dequantizes expert weights per token. It is the parity oracle, not a serving path.
- MRope runs as Neox because inputs carry one position stream. Image positions need a multi stream input op.
- `Session` rebuilds the graph every call and the server resets the session per request. Both go when the prefix cache lands.
- `describe` reports a static parameter list. It must be generated from the same table the CLI parses.

## Part 2. What nebu needs from the finished binary

nebu runs a runtime as one process it installs, launches with flags, watches through logs and a health check, plans memory for, and joins into formations. Every item here maps to a field or method nebu already has.

### Distribution

- One static binary named `kungfu`, no Python, no runtime toolkit.
- GitHub release assets named `kungfu-<version>-<os>-<arch>-<backend>.tar.gz`, one binary each, for linux x86_64 cpu, linux x86_64 cuda, linux aarch64 cpu, windows x86_64 cpu, windows x86_64 cuda, macos arm64 cpu. Metal and HIP join the list when they exist.
- `kungfu --version` prints `kungfu <semver> protocol <n>`.
- Source builds need only cargo and, for cuda, nvcc. The nebu recipe is cargo and nvcc as tools, `cargo build --release -p kungfu` as the one step, `target/release/kungfu` as the output.

### Self description

`kungfu describe` prints one JSON document. nebu reads it instead of knowing the runtime. Fields nebu consumes and the nebu field each feeds.

| Field | Content | nebu |
|---|---|---|
| `protocol` | integer, bumped only for breaking changes | capability gate |
| `architectures` | GGUF ids the registry claims | Formats and model matching |
| `quant_types` | id, name, block, size, and which backends run it | descriptor fit |
| `backends` | name, devices probed with ordinal name total and free bytes | Unmet and device assignment |
| `params` | every launch parameter with name, type, default, solved, label, unit, min, max, step, group, flag, env, rule, choices, advanced, picks | `Params()` verbatim |
| `health` | path, interval, timeout | `Health()` |
| `api` | flavors served and their paths | `API()` and the gateway |
| `roles` | per role, phase, files, health kind, listens, single, rendezvous, head, connects_from | `Roles()` |
| `shapes` | which shapes the roles compose | `Shapes()` |
| `shape_facts` | activation bytes, chunks in flight, reductions per layer, handoff kind, draft tokens, relay overlap | `Policy().Shapes` |
| `measurements` | the log keys the binary emits with their meaning | `Measure()` |
| `errors` | the error codes the binary emits with a summary, a hint, and the fix flag | `Triage()` |
| `stop_grace_seconds` | drain time honored on SIGTERM | `StopGrace()` |

`params` is the single source of truth. The CLI parses the same table it prints, so a flag cannot exist that describe does not list.

### Model inspection

`kungfu inspect --json <gguf>` is nebu's descriptor source for these files. It must print the architecture, hyperparameters, tensor groups by kind with bytes and count per layer, and per layer state as bytes per token or fixed bytes, so nebu's estimator uses the engine's own numbers and never a formula of its own. Unknown architectures and unclaimed tensors are errors with the names.

### Launch

- `kungfu serve --model <first shard> --host H --port P` plus any described params by flag. Device visibility through `CUDA_VISIBLE_DEVICES`, nothing else.
- `--plan <json>` takes nebu's memory plan, per group kind and pool with bytes, and the engine honors it per layer. Without it the engine plans itself and prints the plan it chose as the same JSON on the first log lines.
- `--ctx auto` picks the largest context that fits the plan, up to the trained length, and prints the number.
- `--cache-dir` is where the routing profile, PTX cache, and slot files go. nebu passes its seat cache dir.
- `--profile <file>` loads a routing profile. Without one the engine warms from its own counting and writes the profile to the cache dir at exit.

### Readiness and health

- `/health` answers 503 with `{"phase": "<loading|warming>", "progress": 0..1}` until weights are placed and one token has decoded, then 200 `{"status":"ok"}`.
- SIGTERM drains in flight requests within the grace from describe and exits 0. SIGKILL after that is nebu's fallback, not the plan.

### API

The gateway speaks OpenAI, Anthropic, and Ollama to clients and OpenAI to the runtime. The runtime must offer, exactly as OpenAI shapes them:

- `POST /v1/chat/completions`, streaming and not, with `tools`, `tool_choice`, `reasoning_effort`, `chat_template_kwargs`, `seed`, `stop`, `max_tokens`, `temperature`, `top_p`, `top_k`, `min_p`, `repeat_penalty`, `logit_bias`, `response_format` json schema.
- Replies carry `reasoning_content`, structured `tool_calls` the gateway runs itself, `finish_reason` stop length or tool_calls, and `usage` with `prompt_tokens_details.cached_tokens`.
- Streaming sends one chunk per decoded piece with `reasoning_content`, `content`, or `tool_calls` deltas, a final chunk with the finish reason and usage, then `[DONE]`.
- `GET /v1/models`, `GET /health`, `GET /describe`, `GET /metrics` in Prometheus text.
- A disconnected client cancels its generation within one decode step.
- The prefix cache keeps the longest common token prefix of consecutive requests, so a multi turn conversation prefills only the new turn. This is what the gateway's continuity depends on.

### Logs

Every log line is one event on stderr, `kungfu event=<name> k=v ...`, keys stable across releases. nebu parses these and nothing else.

| Event | Keys | nebu |
|---|---|---|
| `plan` | the chosen plan as JSON | shown and compared |
| `alloc` | `pool=device:N|host`, `kind=weights|state|compute|experts`, `bytes` | `Measure()` sums by `pool.kind` |
| `alloc_failed` | `pool`, `kind`, `bytes`, `free` | `Measure()` key `pool.failed` |
| `ready` | `n_ctx`, `warm_ms` | readiness |
| `cache` | `layer`, `slots`, `hit_rate`, every 10 seconds | shown |
| `transport` | `kind=rdma|tcp`, `peer` | `Transport()` |
| `error` | `code`, plus the keys describe lists for that code | `Triage()` |
| `request` | `prompt_tokens`, `cached_tokens`, `completion_tokens`, `prefill_tps`, `decode_tps` | shown |

Measured bytes after load must match the plan within 5 percent, or the plan event states why.

### Formations

Roles the binary plays, each a `--role` with the seat flags `--rank`, `--count`, `--rendezvous`, `--peer role=rank@addr:port`, `--layer-from`, `--layer-to`, `--slot-dir`, `--cache-key`, `--aux-port`, `--admit addr`.

| Role | Shape | Health | Serves |
|---|---|---|---|
| head | solo, chain, draft | HTTP `/health` | the API, sampling, the layers not on stages |
| replica | replicas | HTTP, listens | the whole model, the gateway spreads requests |
| stage | chain | hello exchange, listens, single | layers `layer_from..layer_to` to the head over TCP or RDMA, the head connects, nothing else is admitted |
| prefill and decode | relay | HTTP, listens, canary | prefill answers the prompt and hands the cache to decode by slot file in `--slot-dir` or over the wire, decode answers the stream |
| draft | draft | HTTP, listens, single | draft tokens to the head from a draft model or next token heads |

The hello exchange is the one nebu already sends to a ggml rpc stage, so no nebu change is needed. One byte command 14, eight byte little endian size 24, 24 zero bytes. The stage answers an eight byte little endian size then that many bytes whose first three are the protocol version. Admission refuses every address not in `--admit`.

The transport a seat negotiated is logged once as the `transport` event.

### Placement, the part that makes it optimal

nebu decides placement and the engine executes it. The plan has, per layer, the weights pool, the state pool, and an expert cache slot count. The engine adds what nebu cannot know:

- Experts pinned in host memory once, device slots holding the hottest by the routing profile, swapped by least recent use, the hit rate logged.
- Uncached experts computed on the host pool while the device computes cached ones, in the same layer step.
- The next layer's cached experts prefetched over PCIe while attention runs.
- Kv kept on the host past the resident window the plan names, streamed in for attention.
- Prefill in chunks the plan sizes, decode under CUDA graphs.

Targets on the same file and hardware against llama.cpp at its best flags: decode no slower on any dense file, at least 1.5 times faster on a hybrid MoE file whose experts do not fit the device, prefill at least 1.5 times faster on both. Greedy top 1 agreement with llama.cpp at least 99 percent.

### Extension rule

A new architecture is one file in `crates/kf-arch/src/archs/` and one line in the registry. A new quant type is one file in `crates/kf-quant/src/types/` and one line. A new backend is one crate implementing `Backend`. Nothing else in the tree names them, and `cargo check` fails on the day a name leaks.

# Baselines, ceilings, and composition for single-stream LLM decode on consumer GPUs (as of 2026-11-01)

Scope: batch-1 (single request) decode tokens per second on RTX 3080 Ti (12 GB, 912 GB/s), RTX 3090, RTX 4090, RTX 5090. Every number below carries its source and date where the source gives one. Numbers marked "computed" are arithmetic from cited inputs and are listed under Inferences.

## 1. llama.cpp today: plain decode and speculative modes

### Takeaway
The best measured llama.cpp plain-decode numbers for 8B-class Q4_K_M models are about 155 t/s on an RTX 4090 (Qwen3-8B, July 2026) and about 300 t/s on an RTX 5090 (Llama 2 7B Q4_0), which is roughly 63 to 74 percent of the weight-bandwidth floor for dense 4-bit models (80 percent at Q8_0 and 87 percent at bf16) and only 38 to 51 percent for Qwen3-30B-A3B. With its newest speculative mode (draft-dspark, merged 2026-07-28) llama.cpp reaches a 1.81x geomean (up to 2.19x on GSM8K, 340.7 t/s) on Qwen3-8B Q4_K_M on an RTX 4090, so the "best configuration" bar for Qwen3-backbone 8B models on a 4090 is roughly 280 t/s geomean across math, code and chat, and about 341 t/s on the friendliest domain.

### Cited Findings

Spec-type inventory and how modes combine
- The `--spec-type` options are none, draft-simple, draft-eagle3, draft-dflash ("block-diffusion draft model that emits a block per step"), draft-dspark ("DFlash backbone plus a semi-autoregressive Markov head"), draft-mtp (MTP heads from the main model), ngram-cache, ngram-simple, ngram-map-k, ngram-map-k4v (experimental, up to four m-gram values), ngram-mod ("basic n-gram hasher with a shared pool"); `--spec-default` enables ngram-mod; default `--spec-draft-n-max` is 3 — [llama.cpp docs/speculative.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md)
- Multiple types can be given comma-separated (e.g. `ngram-mod,ngram-map-k4v`); when a draft model is combined with draftless decoding, "the draftless decoding takes precedence"; the doc gives no throughput numbers, only sample stats such as acceptance 0.57576 (171/297) for a draft model and ngram_mod 730/960 accepted — [speculative.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md)
- Recommended settings in the docs: draft-dflash example `--spec-draft-n-max 15` (clamped to the draft's trained block size); draft-dspark example `--spec-draft-n-max 7`, only Qwen3-backbone drafts supported; ngram-mod example `--spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-max 64` — [speculative.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md)

Plain decode scoreboards
- llama.cpp CUDA scoreboard (Discussion #15013, started 2025-08-01), Llama 2 7B Q4_0, tg128 with FA: RTX 3090 161.89 t/s, RTX 3090 Ti 172.26, RTX 4090 188.96, RTX 5090 300.40 (no FA: 158.16 / 171.19 / 186.21 / 290.02); no RTX 3080 Ti row — [Discussion #15013](https://github.com/ggml-org/llama.cpp/discussions/15013)
- Older (May 2024 snapshot) llama.cpp Llama 3 8B Q4_K_M tg512 / tg1024 / tg4096 / tg8192: RTX 3080 Ti 110.60 / 106.71 / 98.34 / 88.63; RTX 3090 115.42 / 111.74 / 97.31 / 87.49; RTX 4090 130.58 / 127.74 / 119.44 / 110.66; F16 on 4090 54.84 — [XiongjieDai GPU-Benchmarks-on-LLM-Inference](https://github.com/XiongjieDai/GPU-Benchmarks-on-LLM-Inference)
- Qwen3-8B on RTX 4090 (DSpark PR, July 2026) baselines: bf16 about 58.5 t/s, Q8_0 about 100.6, Q4_K_M about 155.4 — [PR #25173](https://github.com/ggml-org/llama.cpp/pull/25173)
- Llama-3.1-8B-Instruct Q4_K_M baseline 121.5 t/s and BF16 44.5 t/s on RTX A6000 48 GB (EAGLE3 PR, merged 2026-06-12) — [PR #18039](https://github.com/ggml-org/llama.cpp/pull/18039)
- Qwen3-30B-A3B Q4_K_M (17.28 GiB, 30.53 B params), tg128, FA on, 2025-11-30: RTX 4090 198.39 t/s baseline (fusion disabled) and 271.04 with fusion plus `GGML_CUDA_GRAPH_OPT=1`; RTX 5090 246.96 to 352.06; gpt-oss-20B MXFP4 on 5090 329.23 to 419.14, which the author puts at "~80%" of a ~520 t/s speed-of-light — [Discussion #17621](https://github.com/ggml-org/llama.cpp/discussions/17621)
- RTX 5090, Qwen3.5-35B-A3B UD-Q4_K_XL (~19 GB), TG average 194.0 t/s (2026-02-25); RTX 3090 Ti Qwen3.5-35B-A3B Q4_0 mainline TG 152.56 t/s (llama-sweep-bench, Mar 2026) — [Discussion #19890](https://github.com/ggml-org/llama.cpp/discussions/19890)
- LocalScore (llamafile, a llama.cpp derivative) on RTX 4090: Llama 3.2 1B Q4_K_M 200 t/s generation, Llama 3.1 8B Q4_K_M 90.5 t/s — [LocalScore RTX 4090](https://www.localscore.ai/accelerator/77)
- On RTX 3090, llama.cpp BF16 Qwen3.5-0.8B tg128 267 t/s vs a fused megakernel 413 t/s (1.55x decode) — [Lucebox hub](https://github.com/Luce-Org/lucebox-hub); [HN thread](https://news.ycombinator.com/item?id=47838788)

draft-dspark (PR #25173, merged 2026-07-28, RTX 4090, greedy, n-max 7)
- Qwen3-8B bf16 per domain, baseline / DFlash / DSpark t/s (accept rate): GSM8K 58.5 / 182.4 (53.7%) / 237.3 (78.9%); MATH500 58.5 / 195.7 (59.2%) / 223.2 (72.8%); HumanEval 59.1 / 238.8 (77.2%) / 241.4 (81.7%); MBPP 59.6 / 177.3 (53.3%) / 193.1 (63.7%); MT-Bench 58.6 / 93.5 (19.7%) / 120.4 (31.3%); geomean DFlash 2.89x, DSpark 3.35x — [PR #25173](https://github.com/ggml-org/llama.cpp/pull/25173)
- Geomean speedup by target quant: bf16 DFlash 2.89x / DSpark 3.35x; Q8_0 2.28x / 2.68x; Q4_K_M 1.54x / 1.81x; Q4_K_M GSM8K 155.4 to 340.7 t/s at 77.4% accept — [PR #25173](https://github.com/ggml-org/llama.cpp/pull/25173)
- SpeedBench qualitative split, Qwen3-8B bf16: baseline 58.15, DFlash 90.00, DSpark 109.10 t/s (accept 26.98%) — [PR #25173](https://github.com/ggml-org/llama.cpp/pull/25173)
- Qwen3-4B bf16 DSpark geomean 2.85x (baseline about 103 t/s; GSM8K 354.0 t/s) — [PR #25173](https://github.com/ggml-org/llama.cpp/pull/25173)

draft-dflash (PR #22105, opened 2026-04-19, merged 2026-06-28)
- Headline "up to 8x speedup (Qwen3)" with numerical equivalence to reference; Qwen3-8B bf16 block 16 (GPU not stated): quicksort code-only 51.9 to 419.3 t/s (8.08x, 93.3% accept) without thinking but 92.0 t/s (1.77x, 12.0%) with thinking; "1-day DC trip" 1.09x with thinking, 1.49x without — [PR #22105](https://github.com/ggml-org/llama.cpp/pull/22105)
- Qwen3.6-27B Q4_K_M on DGX Spark, SpeedBench: 12.57 to 33.76 t/s (2.69x), accept rate 0.2516 — [PR #22105](https://github.com/ggml-org/llama.cpp/pull/22105)
- MoE target GPT-OSS-20B (block 8): 0.61x to 1.27x, i.e. slower than baseline on most prompts — [PR #22105](https://github.com/ggml-org/llama.cpp/pull/22105)
- Gerganov announced DFlash support on X on Jul 8 crediting NVIDIA and Ruixiang Wang — [X post](https://x.com/ggerganov/status/2074869767689662643)

draft-eagle3 (PR #18039, merged 2026-06-12, draft_size 8, RTX A6000 unless noted)
- Llama-3.1-8B BF16: 44.5 to 146.2 / 127.1 / 113.8 t/s (3.28x / 2.85x / 2.55x; accept 80.6% / 77.4% / 80.9%) — [PR #18039](https://github.com/ggml-org/llama.cpp/pull/18039)
- Llama-3.1-8B Q4_K_M: 121.5 to 274.4 / 238.9 / 196.5 t/s (2.26x / 1.97x / 1.62x; accept 92.5% / 79.4% / 77.2%) — [PR #18039](https://github.com/ggml-org/llama.cpp/pull/18039)
- Qwen3-8B BF16 1.62x to 2.17x; Qwen3-14B BF16 1.26x to 1.46x; Qwen3-32B Q4_K_M 1.16x to 1.30x; Qwen3-30B-A3B BF16 on DGX Spark 1.25x to 1.39x; GPT-OSS-20B 0.89x to 1.06x — [PR #18039](https://github.com/ggml-org/llama.cpp/pull/18039)

draft-mtp (PR #22673, merged 2026-05-16)
- RTX 5090, Qwen3.6-27B q4_0 weights and KV: 63.72 to 105.47 t/s eval at n=3, acceptance 0.79728 — [PR #22673](https://github.com/ggml-org/llama.cpp/pull/22673)
- RTX 3090 + 3060, Qwen3.6-27B Q6_K: 22.97 to 42.45 t/s at n=3, 76.0% accept (~1.85x) — [PR #22673](https://github.com/ggml-org/llama.cpp/pull/22673)
- RTX 3060 Laptop, Qwen3.6-35B-A3B: 22.92 baseline; n=1/2/3/4 gives 27.58 / 29.39 / 27.78 / 26.09 t/s with acceptance 0.8835 / 0.815 / 0.7127 / 0.6421 — [PR #22673](https://github.com/ggml-org/llama.cpp/pull/22673)
- Follow-up PR #23269 (merged 2026-05-19) changed default `--spec-draft-n-max` from 16 to 3 and states parallel decoding with MTP "remains highly suboptimal" — [PR #23269](https://github.com/ggml-org/llama.cpp/pull/23269)

ngram family
- ngram-mod (PR #19164, merged 2026-01-30): LCG rolling hash of the last n tokens maps to next token, ~16 MB, constant memory, one pool shared across server slots; user logs: gpt-oss-120b verbatim-repeat task 0.92188 acceptance (1416/1536) at 813.71 t/s versus 181.60 t/s in a 0% run; Qwen 30B general requests 0.047 to 0.634 acceptance at about 114 to 167 t/s; no no-speculation baseline in the thread — [PR #19164](https://github.com/ggml-org/llama.cpp/pull/19164)
- Gerganov: ngram hashing drafts up to 64 tokens, "mostly effective for coding tasks" — [X post](https://x.com/ggerganov/status/2040514840687514037)
- A feature request to chain draft-mtp then ngram-mod claimed ngram-mod on top of draft-mtp gave no speedup and projected "5-10%" from chaining; it has no measured numbers and was closed not planned on 2026-07-12 — [Issue #23184](https://github.com/ggml-org/llama.cpp/issues/23184)
- No PR or benchmark table with numbers for ngram-simple, ngram-map-k, ngram-map-k4v or ngram-cache was found beyond the speculative.md sample counts (ngram_simple 73/187, ngram_map_k 968/1248) — [speculative.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md)

### Inferences
- Bandwidth efficiency of plain decode (computed from the bytes-per-token figures in Section 2): Llama 2 7B Q4_0 reads about 3.75 GB per token, so #15013 corresponds to about 607 GB/s on the 3090 (65% of 936), 709 GB/s on the 4090 (70%), 1127 GB/s on the 5090 (63%). Qwen3-8B on the 4090: bf16 58.15 t/s is about 87% of the 67 t/s floor, Q8_0 100.6 t/s about 80% of 125, Q4_K_M 155.4 t/s about 72 to 74% of 209 to 215. Lower bit widths reach a smaller fraction of bandwidth, consistent with dequantization compute starting to matter. Llama-3.1-8B Q4_K_M on the A6000 (768 GB/s) is about 73% of its 166 t/s floor.
- Qwen3-30B-A3B Q4_K_M reads about 1.91 GB of active weights per token, so 271 t/s on the 4090 is about 51% of the 527 t/s floor and 352 t/s on the 5090 is about 38% of 936; MoE plain decode has the most headroom.
- The 1B class is far from its floor in llama.cpp-derived runtimes: LocalScore's 200 t/s for Llama 3.2 1B Q4_K_M on a 4090 is about 16% of the ~1,200 t/s floor; launch and synchronization overhead dominate at this size (Qwen3-0.6B megakernel evidence in Section 3).
- Estimated llama.cpp bar on the RTX 3080 Ti, scaling 4090 numbers by 912/1008 at equal efficiency: Qwen3-8B Q4_K_M about 140 t/s plain and about 254 t/s with DSpark geomean (about 308 t/s on GSM8K-like text). Llama-3.1-8B Q4_K_M about 144 t/s plain (A6000 121.5 scaled by 912/768) and about 233 to 326 t/s with EAGLE3. A 5x target therefore means roughly 1,200 to 1,600 t/s for 8B-class models on the 3080 Ti, about 6 to 8x the Q4_K_M single-token bandwidth floor (192 to 197 t/s at 1K context) and about 3 to 4x the IQ2_XXS floor (about 409 to 435 t/s).
- The speculative modes that beat plain decode on dense 8B targets in llama.cpp are DSpark, DFlash, EAGLE3 and MTP; on MoE targets (GPT-OSS-20B, Qwen3-30B-A3B) all published llama.cpp speculative results are at or below 1.4x and several are below 1.0x.

### Gaps
- No measured llama.cpp tg numbers were found for: any model on the RTX 3080 Ti newer than the May 2024 snapshot; Llama-3.2-1B on 3090/4090/5090 with mainline llama.cpp; Qwen3-8B on 3090/5090; any IQ2-class quant (IQ2_XXS, IQ2_XS, IQ2_S, IQ2_M) of any of the four models on any of the four GPUs; Qwen3-30B-A3B on the 3090 or at IQ2 on a 12 GB card.
- No PR or discussion with acceptance or speedup tables was found for ngram-simple, ngram-map-k, ngram-map-k4v or ngram-cache; only sample counts exist.
- The DFlash PR does not name the GPU for its Qwen3-8B bf16 table.
- No llama.cpp result combining draft-dspark or draft-dflash with an ngram mode was found.

## 2. Physical floors: bytes per token and ceiling tokens per second

### Takeaway
At 1K context the Q4_K_M bandwidth floor on the RTX 3080 Ti is about 192 t/s for Llama-3.1-8B, 189 t/s for Qwen3-8B, 1,095 t/s for Llama-3.2-1B, and 453 t/s for Qwen3-30B-A3B (which does not fit 12 GB at Q4_K_M). At 32K context an f16 KV cache roughly doubles bytes per token for the 8B models (KV reads become 48 to 51 percent of traffic) and makes KV 57 to 63 percent of traffic for the 1B and 30B-A3B models. The LM head is 9 to 11 percent of weight traffic for the 8B models, 13 percent for Qwen3-30B-A3B and 27 percent for Llama-3.2-1B.

### Cited Findings
- Memory bandwidth: RTX 3080 Ti 912 GB/s (given by the requester; consistent with a 384-bit bus and 19 Gbps GDDR6X — [fpsbench](https://fpsbench.com/compare/gpu/nvidia-geforce-rtx-3080-ti-vs-nvidia-geforce-rtx-5090), [TechPowerUp forum](https://techpowerup.com/forums/posts/5397245)); RTX 3090 936 GB/s — [getdeploying](https://getdeploying.com/gpus/nvidia-rtx-3090-vs-nvidia-rtx-5080); RTX 4090 1008 GB/s — [gigagpu](https://gigagpu.com/rtx-4090-24gb-gddr6x-vram-bandwidth/); RTX 5090 1792 GB/s theoretical, 1,674 GB/s measured read (93%) — [alpindale blog](https://blog.alpindale.net/posts/5090_decode_optimization/)
- Architectures (config.json): Llama-3.2-1B hidden 2048, 16 layers, 32 Q heads, 8 KV heads, head_dim 64, FFN 8192, vocab 128256, tied embeddings — [config](https://huggingface.co/unsloth/Llama-3.2-1B-Instruct/raw/main/config.json); Llama-3.1-8B hidden 4096, 32 layers, 8 KV heads, head_dim 128, FFN 14336, vocab 128256, untied — [config](https://huggingface.co/unsloth/Meta-Llama-3.1-8B-Instruct/raw/main/config.json); Qwen3-8B hidden 4096, 36 layers, 8 KV heads, head_dim 128, FFN 12288, vocab 151936, untied — [config](https://huggingface.co/Qwen/Qwen3-8B/raw/main/config.json); Qwen3-30B-A3B hidden 2048, 48 layers, 32 Q heads, 4 KV heads, head_dim 128, 128 experts, 8 active, expert FFN 768, vocab 151936, untied — [config](https://huggingface.co/Qwen/Qwen3-30B-A3B/raw/main/config.json)
- GGUF sizes, Llama-3.1-8B-Instruct: Q8_0 8.54 GB, Q6_K 6.60, Q4_K_M 4.92, IQ4_XS 4.45, Q3_K_M 4.02, IQ3_M 3.78, Q2_K 3.18, IQ2_M 2.95 — [bartowski](https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF); Llama-3 8B IQ2_XXS 2.39 GB, IQ2_M 2.94 GB — [bartowski dolphin-2.9.1-llama-3-8b](https://HuggingFace.co/bartowski/dolphin-2.9.1-llama-3-8b-GGUF); Qwen3-30B-A3B Q4_K_M 17.28 GiB — [Discussion #17621](https://github.com/ggml-org/llama.cpp/discussions/17621)
- Measured achievable fractions on consumer parts: Qwen3-0.6B bf16 megakernel on RTX 5090 runs at 71.2% of measured bandwidth (1,192 GB/s effective of 1,674) with the LM head near 1,500 GB/s; the author attributes the remaining gap to four full-grid barriers per layer at about 2.2 us each and 211 to 216 us fixed overhead per step — [alpindale blog](https://blog.alpindale.net/posts/5090_decode_optimization/)

### Inferences
Computed with: f16 KV bytes per token = 2 x n_kv x head_dim x 2 bytes x layers (Llama-3.2-1B 32 KiB, Llama-3.1-8B 128 KiB, Qwen3-8B 144 KiB, Qwen3-30B-A3B 96 KiB); weight bytes per token = all non-embedding weights read once plus the LM head (token embedding is a row gather). Q4_K_M body at 4.8 bits per weight (derived from the 4.92 GB Llama file) with a Q6_K head; IQ2_M body at 2.7 and IQ2_XXS at 2.0625 bits with a Q4_K head; Qwen3-30B-A3B active body = attention 18.9 M per layer plus 8 experts of 4.72 M plus router, i.e. 2.73 B active body parameters plus a 311 M head.

| Model | Quant | Weight GB/token | LM head share | KV GB at 1K / 32K (f16) | 3080 Ti 1K / 32K | 3090 1K / 32K | 4090 1K / 32K | 5090 1K / 32K |
|---|---|---|---|---|---|---|---|---|
| Llama-3.2-1B | Q4_K_M | 0.80 | 27% | 0.03 / 1.07 | 1095 / 487 | 1124 / 500 | 1211 / 538 | 2152 / 957 |
| Llama-3.2-1B | IQ2_XXS | 0.40 | 37% | 0.03 / 1.07 | 2108 / 619 | 2164 / 636 | 2330 / 684 | 4143 / 1217 |
| Llama-3.1-8B | BF16 | 15.01 | 7% | 0.13 / 4.29 | 60 / 47 | 62 / 48 | 67 / 52 | 118 / 93 |
| Llama-3.1-8B | Q4_K_M | 4.62 | 9% | 0.13 / 4.29 | 192 / 102 | 197 / 105 | 212 / 113 | 377 / 201 |
| Llama-3.1-8B | IQ2_M | 2.65 | 11% | 0.13 / 4.29 | 327 / 131 | 336 / 135 | 362 / 145 | 643 / 258 |
| Llama-3.1-8B | IQ2_XXS | 2.10 | 14% | 0.13 / 4.29 | 409 / 143 | 420 / 146 | 452 / 158 | 804 / 280 |
| Qwen3-8B | BF16 | 15.14 | 8% | 0.15 / 4.83 | 60 / 46 | 61 / 47 | 66 / 50 | 117 / 90 |
| Qwen3-8B | Q4_K_M | 4.68 | 11% | 0.15 / 4.83 | 189 / 96 | 194 / 98 | 209 / 106 | 371 / 188 |
| Qwen3-8B | IQ2_M | 2.69 | 13% | 0.15 / 4.83 | 321 / 121 | 329 / 124 | 354 / 134 | 630 / 238 |
| Qwen3-8B | IQ2_XXS | 2.14 | 16% | 0.15 / 4.83 | 398 / 131 | 408 / 134 | 440 / 145 | 782 / 257 |
| Qwen3-30B-A3B | Q4_K_M (18.6 GB file, does not fit 12 GB) | 1.91 active | 13% | 0.10 / 3.22 | 453 / 178 | 465 / 182 | 500 / 196 | 889 / 349 |
| Qwen3-30B-A3B | IQ2_M (about 10.4 GB file) | 1.10 active | 16% | 0.10 / 3.22 | 761 / 211 | 782 / 217 | 842 / 233 | 1496 / 415 |
| Qwen3-30B-A3B | IQ2_XXS (about 8.0 GB file) | 0.88 active | 20% | 0.10 / 3.22 | 931 / 222 | 955 / 228 | 1029 / 246 | 1829 / 437 |

- With a q8_0 KV cache at 32K the 3080 Ti floors rise to 132 (Llama-3.1-8B Q4_K_M), 126 (Qwen3-8B Q4_K_M), 666 (Llama-3.2-1B Q4_K_M) and 352 (Qwen3-30B-A3B IQ2_XXS) t/s.
- Attention share of traffic at 1K is 3 to 5% for every model at Q4_K_M and 5 to 10% at IQ2; at 32K f16 it is 48% (Llama-3.1-8B Q4_K_M), 51% (Qwen3-8B), 57% (Llama-3.2-1B) and 63% (Qwen3-30B-A3B), rising to 67 to 79% at IQ2_XXS. Long-context decode is a KV-bandwidth problem once weights are at 2 to 4 bits.
- The Qwen3-30B-A3B IQ2 file sizes are computed from parameter counts, not taken from a published file.
- Speculation is the only listed lever that moves tokens per second above these floors without changing bytes per token: Lucebox's 207.6 t/s on Qwen3.5-27B Q4_K_M on a 3090 (Section 3) is about 3.6x the 3090's single-token floor for that model (about 57 t/s, computed assuming about 16.5 GB for 27 B parameters at 4.85 bits per weight).

### Gaps
- No published per-tensor byte breakdown for the exact Q4_K_M and IQ2 recipes of these four models was fetched; per-tensor quant types were approximated.
- Exact Qwen3-30B-A3B IQ2_XXS and IQ2_M file sizes were not retrieved.

## 3. Best published single-stream results on one consumer GPU

### Takeaway
No academic paper found reports an 8B-class model on a single consumer GPU with quantization, speculation, sparsity and KV compression stacked. The highest single-consumer-GPU 8B numbers located are llama.cpp's own: DSpark on RTX 4090 with Qwen3-8B Q4_K_M at 340.7 t/s on GSM8K (about 281 t/s geomean). The strongest custom-runtime evidence is Lucebox's DFlash plus DDTree on one RTX 3090 (Qwen3.5-27B Q4_K_M, 129.5 t/s mean on HumanEval, 207.6 t/s demo, 3.4 to 5.5x over its own AR baseline) and megakernels for sub-1B models (about 1,000 t/s for Qwen3-0.6B bf16 on an RTX 5090). Datacenter papers report lossless 4.86x average and up to 6.08x per benchmark for Qwen3-8B with DFlash on H200, and 5.46x average with DFlare.

### Cited Findings
- llama.cpp DSpark on RTX 4090: Qwen3-8B Q4_K_M GSM8K 155.4 to 340.7 t/s (77.4% accept); bf16 HumanEval 241.4 t/s; Q4_K_M geomean 1.81x — [PR #25173](https://github.com/ggml-org/llama.cpp/pull/25173)
- Lucebox on one RTX 3090: Qwen3.5-27B Q4_K_M target with BF16 DFlash draft, 207.6 t/s DFlash vs 38.0 t/s AR (5.46x) in the demo; HumanEval bench 129.5 t/s mean at DDTree budget 22, 3.43x over AR Q4_K_M (37.78 t/s); speedups "measured vs vendored llama.cpp (-fa 1, matching KV quant)"; DFlash path greedy-only — [HN thread](https://news.ycombinator.com/item?id=47838788); [Lucebox hub](https://github.com/Luce-Org/lucebox-hub)
- Lucebox later reports Qwen 3.8 27B UD-IQ4_XS with DFlash2 on an AMD R9700 at 208.1 t/s average (227.8 peak HumanEval), 6.4x over its AR and 3.8x over llama.cpp with the same drafter — [Lucebox hub](https://github.com/Luce-Org/lucebox-hub)
- Qwen3-0.6B bf16 on RTX 5090: about 1,033 t/s (0.97 ms/token) with a persistent megakernel, Feb 9 to 14, 2026; original MegaQwen ran 530 t/s on an RTX 3090 — [alpindale blog](https://blog.alpindale.net/posts/5090_decode_optimization/)
- Vendor benchmark (methodology unverified) on RTX 4090, Llama 3.1 8B batch 1: EXL2 4.0 bpw about 240 t/s, AWQ-Marlin INT4 about 225 t/s, FP16 about 95 t/s — [gigagpu](https://gigagpu.com/rtx-4090-24gb-llama-3-8b-benchmark/)
- MTP on one RTX 4090, Qwen3.6-27B: ~47 to ~133 t/s on editing turns (2.85x), blog — [Pedro Alonso](https://www.pedroalonso.net/blog/qwen-mtp-speculative-decoding-4090/)
- DFlash paper (arXiv 2602.06036, 2026-02-05), H200, temperature 0, block 16: Qwen3-8B average 4.86x speedup, tau 6.49 (MATH-500 6.08x, tau 7.87; MT-Bench 2.75x, tau 4.24) vs EAGLE-3 (60-node tree) 2.02x, tau 3.40; temperature 1: 4.03x, tau 5.48; SGLang single B200 Qwen3-8B Math500 5.1x at concurrency 1 (tau 8.01) — [DFlash HTML](https://arxiv.org/html/2602.06036v2)
- DFlare (arXiv 2606.02091) reports 5.46x average wall-clock speedup on Qwen3-8B, about 8% over DFlash — [DFlare](https://arxiv.org/html/2606.02091v1)
- EAGLE-3 (arXiv 2503.01840), temperature 0, LLaMA-3.1-8B: mean 4.44x, tau 6.23 (HumanEval 4.85x, tau 6.74; CNN/DM 3.65x); temperature 1: 3.45x, tau 4.92 — [EAGLE-3](https://arxiv.org/html/2503.01840)
- SpecEE (ISCA 2025): 2.43x on Llama2-7B in a PC setting (RTX 4060 Laptop, i7-13650HX), 2.25x on A100 — [SpecEE](https://arxiv.org/abs/2504.08850)
- BlockFFN (COLM 2025) on NVIDIA Jetson Orin NX: chunk-sparse MoE-style FFN plus EAGLE-2, 12.84 to 47.17 t/s average (3.67x) — [BlockFFN](https://arxiv.org/pdf/2507.08771)

### Inferences
- The highest 8B-class single-consumer-GPU number with near-lossless output found is about 341 t/s (llama.cpp DSpark, Qwen3-8B Q4_K_M, RTX 4090, GSM8K). Scaled to a 5090 by bandwidth this is about 600 t/s; no published 5090 8B speculative number was found.
- Datacenter tau of 6 to 8 for DFlash at block 16 on math and code would, on a consumer GPU with a verify forward costing about the same as one token, move an 8B Q4_K_M model from about 190 t/s to roughly 1,000 to 1,500 t/s; the gap between that and llama.cpp's measured 1.81x geomean at Q4_K_M is the room a new runtime would need to capture.

### Gaps
- No academic system reports 8B on a single RTX 3080 Ti, 3090, 4090 or 5090 with all of speculation, quantization, sparsity and KV compression together.
- The DFlash paper reports datacenter GPUs only (H200, B200); the DSpark paper does not name its GPU and its latency measurements use batch 128.
- No consumer-GPU numbers for DFlare, DBloom, JetSpec or DDTree for 8B targets were found.

## 4. Composition studies: do quantization, sparsity, speculation, KV compression and layer skipping multiply

### Takeaway
Every measured combination found is sub-multiplicative, and several conflict outright. Speculation's multiplier shrinks as the target is quantized (DSpark in llama.cpp: 3.35x at bf16, 2.68x at Q8_0, 1.81x at Q4_K_M) because the verify pass of a low-bit model is relatively more expensive, while acceptance barely moves. EAGLE-2 on a W4A16 Llama-3-8B on an RTX 3090 was the one tested pairing where the combination did not beat the techniques alone. Activation sparsity and speculation conflict because the union of active rows across a verify batch is 2.5x larger than one token's (BlockFFN: 12.45% to 30.86%), and for MoE the union of experts makes verify cost grow with draft length. KV compression in the verify path leaves acceptance largely intact in the papers found, but llama.cpp's quantized KV cache slows decode.

### Cited Findings

Quantization x speculation
- llama.cpp DSpark/DFlash on RTX 4090, Qwen3-8B geomean speedup over same-quant baseline: bf16 3.35x / 2.89x; Q8_0 2.68x / 2.28x; Q4_K_M 1.81x / 1.54x; GSM8K accept bf16 78.9% vs Q4_K_M 77.4% — [PR #25173](https://github.com/ggml-org/llama.cpp/pull/25173)
- llama.cpp EAGLE3 on A6000, Llama-3.1-8B: BF16 2.55x to 3.28x vs Q4_K_M 1.62x to 2.26x; Q4_K_M acceptance 77.2% to 92.5% vs BF16 77.4% to 80.9% (not lower) — [PR #18039](https://github.com/ggml-org/llama.cpp/pull/18039)
- SpecMQuant (arXiv 2505.22179, May 2025, A100 and RTX 3090): "integrating EAGLE-2 with quantization yields additional speedup compared to using either technique alone, except when EAGLE-2 is applied to W4A16 8B model on the RTX 3090"; W4A16 is the least compatible; "Quantization has minimal impact on average accepted length" with W4A16 and W8A8 "exhibiting nearly no degradation"; for W4A16 the verification-to-decoding time ratio reaches 1.8 at tree size 60 on 8B (A100) vs below 1.2 for FP16 and W8A8; fewer draft tokens give higher speedup on 4-bit models — [SpecMQuant HTML](https://arxiv.org/html/2505.22179v1), [PDF](https://arxiv.org/pdf/2505.22179)
- SpecMQuant Table 2, W4A16 Llama-3-70B on A100: AR 35.28 t/s; EAGLE-2 depth 3/4/5/6 gives tau 3.25/3.57/3.73/3.81 and 2.10x/2.12x/1.86x/1.80x (longer acceptance, lower speed); hierarchical draft (tree to sequence via an intermediate model) 98.19 t/s, 2.78x — [SpecMQuant HTML](https://arxiv.org/html/2505.22179v1)
- QSpec (EMNLP 2025): W4A4 draft verified by W4A16 of the same weights; up to 1.64x over W4A16; on vLLM batch 1 speedups 1.29x (WildChat), 1.36x (GSM8K), 1.34x (MBPP) at 93.1% / 92.1% / 95.5% acceptance; "applying GPTQ quantization to the EAGLE draft model resulted in substantial degradation of the acceptance rate"; EAGLE is best at batch 1 but degrades at batch 8 and 16 — [QSpec PDF](https://i.cs.hku.hk/~cwu/papers/jtzhao-emnlp25.pdf), [ACL Anthology](https://aclanthology.org/2025.emnlp-main.240/)
- An MLX EAGLE-3 prototype found quantizing the draft to 4-bit gave identical acceptance (0.34) while halving draft cost — [mlx-lm Discussion #890](https://github.com/ml-explore/mlx-lm/discussions/890)
- ML-SpecQD: MXFP4-quantized self-drafts reach 71.2% average acceptance on QA and 91.1% on code generation — [ML-SpecQD](https://arxiv.org/pdf/2503.13565)

Activation sparsity x speculation
- BlockFFN on Jetson Orin NX (Spec-Bench average): baseline AR 12.84 t/s; EAGLE-2 22.36 (1.74x); token-level sparsity kernel 40.36 (3.14x); sparsity plus EAGLE-2 (32-token tree) 47.17 (3.67x); on Translation the combination (39.18 t/s) is slower than sparsity alone (40.39); per-token activated expert ratio 12.45% vs 30.86% for the union over a 32-token EAGLE-2 tree; FFN time 7.8 ms (1-token sparse) vs 27 ms (32-token union) — [BlockFFN](https://arxiv.org/pdf/2507.08771)
- TEAL (ICLR 2025): Llama-3-8B single-batch decode on A6000 45.32 t/s baseline to 55.38 / 64.15 / 73.94 t/s at 25 / 40 / 50% sparsity (1.22x / 1.42x / 1.63x), on A100 1.11x / 1.25x / 1.33x; at batch 4 layer sparsity at fixed error falls from about 60% to about 38% (Llama-2-7B W_down); TEAL "does not scale as well to higher batch sizes" — [TEAL HTML](https://arxiv.org/html/2408.14690v2)
- Polar Sparsity: MLP activation sparsity degrades with batch size due to union activations while attention head sparsity stays batch-invariant; early layers stay sparse (under 1% per-token activation) even at large batch — [Polar Sparsity](https://arxiv.org/html/2505.14884v2)
- ReLU Strikes Back: activation reuse across consecutive tokens; aggregated sparsity plus speculative decoding gives 1.27x over standard speculative decoding at gamma 16 on OPT-6.7B — [ReLU Strikes Back](https://arxiv.org/html/2310.04564v1)
- Sparse verification (arXiv 2512.21911, 2025-12-26): sparsifies attention (block KV pruning scored by the first draft token's query), FFN (channel thresholding) and MoE (expert skipping) during verification on Llama3.1-8B, Qwen3-30B-A3B, DeepSeek-R1; mean acceptance length "remained largely stable"; no speedup figure on the abstract page — [alphaXiv 2512.21911](https://www.alphaxiv.org/abs/2512.21911)

MoE x speculation (expert union)
- Bounding speculation in MoE (arXiv 2609.22156): Qwen3-Coder-30B-A3B with EAGLE-3 on A100 at batch 1; verify time modeled as beta0 + alpha x unique experts; at K=8 the median round has 26.1 unique experts per layer; alpha triples from about 0.08 ms/expert (m 4 or less) to about 0.265 ms/expert (m 5 or more); an offline oracle plateaus at about 2.34x with mean accepted length near 2.1 — [arXiv 2609.22156](https://arxiv.org/html/2609.22156v1)
- MoESD: at moderate batch sizes most experts are already activated by the batch and verification cost saturates — [MoESD](https://arxiv.org/pdf/2505.19645)

Layer skipping and early exit x speculation
- SpecEE uses a speculative model to shrink the early-exit predictor's vocabulary search space (predictor overhead otherwise up to about 20% of latency); 2.43x on RTX 4060 Laptop with Llama2-7B — [SpecEE](https://arxiv.org/abs/2504.08850)
- TSS (arXiv 2609.26100): skipping target layers during verification raises Translation acceptance length from 2.70 to 4.53 and throughput from 75.6 to 127.3 t/s (1.68x), with BLEU changing from 0.131 to 0.237, i.e. the output distribution is no longer the full target's — [TSS](https://arxiv.org/abs/2609.26100)

KV compression x speculation
- QuantSpec (ICML 2025): self-speculation with a 4-bit hierarchical KV cache and 4-bit weights keeps acceptance above 90% and gives up to about 2.5x end-to-end for long context; sparse-KV self-drafts degrade faster as speculation length grows; recent KV entries kept in full precision to protect acceptance — [QuantSpec](https://arxiv.org/abs/2502.10424), [alphaXiv](https://www.alphaxiv.org/abs/2502.10424)
- MoE-XBench (arXiv 2608.21693, Aug 2026) in GGUF/llama.cpp: adding KV compression to Prune+Quant lowers decode to 0.63x of Prune+Quant; at 32K, Quant+KV decode is 0.61x of Quant-only — [MoE-XBench](https://arxiv.org/html/2608.21693)
- Lynx (2026 preprint) caps speculative tokens at 64 because acceptance diminishes beyond 64 — [Lynx](https://arxiv.org/html/2607.01831v1)

Mode stacking inside llama.cpp
- Combining a draft model with a draftless mode gives the draftless mode precedence rather than a joint draft — [speculative.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md)

### Inferences
- Composition ratios (computed): llama.cpp Qwen3-8B on a 4090, Q4_K_M alone is 155.4/58.5 = 2.66x over bf16 and DSpark alone is 3.35x, so a multiplicative model predicts 8.9x; the measured Q4_K_M plus DSpark geomean is about 155.4 x 1.81 / 58.5 = 4.8x (54% of multiplicative). Llama-3.1-8B EAGLE3 on A6000: 274.4/44.5 = 6.2x measured vs 2.73 x 3.28 = 8.9x predicted (69%). BlockFFN: 3.67x measured vs 5.46x predicted (67%).
- Two-bit target vs acceptance: no direct measurement exists; at 4-bit the evidence (SpecMQuant, llama.cpp EAGLE3 and DSpark PRs) is that acceptance is essentially unchanged and the loss comes from verify cost. By the same mechanism a 2-bit target with codebook dequantization (IQ2) would further raise verify-to-decode cost relative to its single-token cost unless the kernels stay bandwidth bound at 8 to 16 rows.
- Per-row sparsity in a verify batch: the measured union inflation (2.5x for a sparsity-trained BlockFFN, 60% to 38% sparsity at batch 4 for TEAL, 8 to 26.1 experts per layer for Qwen3-Coder-30B-A3B at K=8) means gathers keyed on one token's mask lose most of their savings at verify widths of 8 to 32; the mitigations in the literature are training for chunk-level sparsity (BlockFFN), restricting sparsity to attention heads (Polar Sparsity), or scoring sparsity from a single token for the whole batch (2512.21911).
- KV compression appears compatible with acceptance when recent tokens stay full precision (QuantSpec, 2512.21911); the measured conflict is speed, not acceptance, in llama.cpp's quantized-KV path.

### Gaps
- "FastSpec" was not found under that name on arXiv; the closest matches are Speculative Speculative Decoding (arXiv 2603.03251, up to 2x over optimized speculative baselines) and FastDraft (arXiv 2411.11055) — [SSD](https://arxiv.org/pdf/2603.03251), [FastDraft](https://arxiv.org/abs/2411.11055v3)
- No study measures acceptance with a 2-bit (IQ2, QuIP#, AQLM, QTIP) target and a draft trained on the full-precision target.
- No study measures a TEAL-style mask shared across EAGLE or DFlash verify tokens on a dense 8B model.
- No study measures acceptance against KV bit width (f16, q8_0, q4_0) for a separate draft model in llama.cpp.
- The sparse-verification paper (2512.21911) abstract page gives no speedup or hardware.

## 5. Ceilings on tokens per forward

### Takeaway
Theory bounds expected accepted tokens per iteration by roughly (mu + mu2)/mu^2 x log P, so tokens per forward grow only logarithmically with verify width and inversely with the target's entropy; measured entropies are lowest on code and math and for Qwen3 relative to Llama 3.1. Empirically, for Qwen3-8B at greedy decoding, block-diffusion drafters accept 6.5 to 9.2 tokens per verify on math and code at block 16 and up to 10.5 with a 24-token block, but only 4.2 to 5.4 on MT-Bench chat; with temperature 1 and block 7, DSpark reaches 5.0 to 6.2 on math and code and 3.2 to 3.7 on chat. Acceptance collapses when the target is in thinking mode (DFlash in llama.cpp: 12% vs 93% acceptance on the same prompt).

### Cited Findings
- Pankratov and Alistarh (arXiv 2512.11718, 2025-12-12): for any deterministic speculative algorithm, E[X] <= (mu + mu(2)) log(P)/mu^2 + O(1), with P the verifier capacity, mu the expected entropy of the target's output, mu(2) the expected second log-moment; "exponentially increasing the computational budget P will only yield a linear improvement in speedup" — [arXiv 2512.11718](https://arxiv.org/abs/2512.11718v1)
- Measured mu at temperature 1 (80 samples per dataset): Llama 3.1 8B HumanEval 0.279, MT-bench 1.088, GSM8K 0.636, CNN/DM 0.577, NQ 0.837 (mean 0.683); Qwen3 8B HumanEval 0.178, MT-bench 0.369, GSM8K 0.202, CNN/DM 0.328, NQ 0.518 (mean 0.319); Llama 3.3 70B mean 0.153; EAGLE-3 at P = 60 tracks the bound linearly with a constant gap — [arXiv 2512.11718](https://arxiv.org/abs/2512.11718v1)
- Acceptance Dynamics Across Cognitive Domains (arXiv 2604.14682, 1B draft / 7B target): mean target entropy chat 1.2517, code 0.8858, math 1.1513 nats; acceptance near 0.5 in every domain; code's lower entropy did not yield higher acceptance — [arXiv 2604.14682](https://arxiv.org/html/2604.14682)
- EAGLE-3 accepted length tau at temperature 0 (tree depth 8, 60 total draft tokens for 8B): LLaMA-3.1-8B MT-bench 6.13, HumanEval 6.74, GSM8K 6.23, Alpaca 6.70, CNN/DM 5.34; Vicuna-13B HumanEval 7.54; at temperature 1 LLaMA-3.1-8B mean 4.92 — [EAGLE-3](https://arxiv.org/html/2503.01840)
- DFlash tau at temperature 0, block 16, Qwen3-8B: GSM8K 6.54, MATH-500 7.87, AIME25 7.08, HumanEval 6.50, MBPP 5.95, LCB 7.27, MT-Bench 4.24; a block-8 drafter fully accepts the whole block 35.7% of the time on MATH-500 — [DFlash HTML](https://arxiv.org/html/2602.06036v2)
- DSpark (arXiv 2607.05147, 2026-07-06, DeepSeek-affiliated) tau at temperature 1, block 7, Qwen3-8B: EAGLE-3 / DFlash / DSpark GSM8K 5.30 / 5.33 / 6.17; MATH 4.77 / 4.91 / 5.78; AIME25 3.91 / 4.07 / 5.01; MBPP 3.96 / 4.36 / 5.16; HumanEval 4.33 / 4.64 / 5.52; LCB 4.17 / 4.39 / 5.17; MT-Bench 2.66 / 3.11 / 3.72; Alpaca 2.54 / 2.98 / 3.58; Arena-Hard 2.54 / 2.81 / 3.21 — [DSpark HTML](https://arxiv.org/html/2607.05147v1)
- DSpark positional acceptance (Qwen3-4B): position 1 Math DFlash 0.88 vs EAGLE-3 0.81, DSpark 0.93; position 1 Chat DFlash 0.72 vs EAGLE-3 0.53; DFlash decays from 0.87 to 0.78 on Code and 0.72 to 0.63 on Chat across the block; production DeepSeek-V4 per-user speedups 57% to 85% over MTP-1 at matched throughput with max draft length 5 — [DSpark HTML](https://arxiv.org/html/2607.05147v1)
- DBloom (arXiv 2608.30427, AMD, 2026-08-31), greedy, full test sets: original DFlare B16 on Qwen3-8B tau GSM8K 7.86, MATH-500 9.17, AIME21-26 8.63, HumanEval 7.07, MBPP 6.77, LCB 7.84, MT-Bench 5.00; after continuation and expansion to B24: 9.23 / 10.54 / 9.58 / 7.93 / 7.72 / 8.71 / 5.38; naive inference-time widening B16 to B24 without retraining lowers speedup (DFlare-4B -52%, DFlash-4B -33%); the "ceiling bin" (fraction of cycles that accept the full block) flags block-limited acceptance; mean E[n]+1 6.75 to 6.79 with speedup 4.88x to 4.91x across six loss weightings on Qwen3-8B — [arXiv 2608.30427](https://arxiv.org/abs/2608.30427)
- llama.cpp DFlash with thinking on vs off, Qwen3-8B quicksort: 12.0% vs 93.3% acceptance (1.77x vs 8.08x) — [PR #22105](https://github.com/ggml-org/llama.cpp/pull/22105)
- llama.cpp MTP acceptance falls with draft length: 0.8835 (n=1), 0.815 (n=2), 0.7127 (n=3), 0.6421 (n=4) on Qwen3.6-35B-A3B — [PR #22673](https://github.com/ggml-org/llama.cpp/pull/22673)
- Verbatim-repeat tasks with ngram-mod reach 0.92 acceptance with 64-token drafts — [PR #19164](https://github.com/ggml-org/llama.cpp/pull/19164)

### Inferences
- Plugging Qwen3-8B entropies into the leading term with P = 16 gives (0.178 + 0.313)/0.178^2 x ln 16, about 43 tokens for HumanEval and (0.369 + 0.814)/0.369^2 x ln 16, about 24 for MT-bench; these loose limits sit far above measured tau, so the binding constraints in practice are drafter quality and block horizon, not entropy (computed; the bound's O(1) term and log base follow the paper's notation and the absolute values are indicative only).
- Measured tokens per verify for 8B targets in 2026: about 6 to 10.5 on math and code at greedy, 5 to 6 at temperature 1; about 3.2 to 5.4 on open chat. A runtime that needs 6+ tokens per forward to beat llama.cpp by 5x at Q4 (Section 1 inference) is only plausible on code and math at greedy; chat traffic needs both fewer bytes per token and speculation.
- Thinking-mode traces are the hardest regime for block drafters in the llama.cpp data, while ngram-style drafters do best on verbatim repetition and reasoning loops.

### Gaps
- No published accepted-length histogram for EAGLE-3, DFlash or DSpark measured on a consumer GPU with a quantized target.
- No information-theoretic bound specific to code vs natural text beyond per-dataset entropy measurements.
- DSpark offline results do not state GPU type.

## 6. Accuracy drift when methods stack

### Takeaway
Stacked-method quality is measured with perplexity plus task suites (MMLU, GSM8K, HumanEval, IFEval, RULER) and, for speculation, by the claim of distribution equivalence. The one 2026 composition benchmark found (MoE-XBench, Aug 2026) shows mostly additive perplexity loss with measurable compounding on task accuracy (Qwen3-30B prune plus Q4_K_M loses about 18% more average accuracy than the sum of the parts) and finds KV-cache quantization costs more on already-compressed models. Speculative decoding is lossless only up to floating-point and batching effects that can flip near-ties.

### Cited Findings
- MoE-XBench (arXiv 2608.21693, Aug 2026, Sony AI and UVA): composes REAP expert pruning (20 to 50%), GGUF weight quant (1 to 16 bits, default Q4_K_M, also IQ1_M) and KV quant (Q8, Q4, TurboQuant) on ten MoE models; quality via MMLU, GSM8K, MuSR, IFEval, HumanEval, BFCLv3, RULER and WikiText-2 perplexity; "composable compression losses are not additive"; Qwen3-30B: "19.35% expert pruning increases PPL by 28.5%" while "69.5% Q4_K_M weight quantization increases PPL by only 0.7%"; average task score drops 0.0433 (prune) and 0.0016 (quant) but 0.0528 combined; Q4_K_M PPL penalty grows from +0.13 at 20% pruning to +0.40 at 50%; Q4/Q4 KV raises PPL by 0.12 unpruned vs 0.17 pruned; IQ1_M memory savings offset by dequantization overhead — [MoE-XBench](https://arxiv.org/html/2608.21693)
- TEAL: Llama-3-8B WikiText PPL 5.87 to 5.94 / 6.21 / 6.67 / 9.06 at 25 / 40 / 50 / 65% activation sparsity, six-task average 68.07 to 67.73 / 66.21 / 63.42 / 52.59; with 8-bit RTN, 4-bit AWQ and 2/3-bit QuIP# weights the sharp-degradation point is similar and errors "compound somewhat independently" — [TEAL HTML](https://arxiv.org/html/2408.14690v2)
- KV benchmark (arXiv 2607.05399): on Llama-3.1-8B and Mistral-7B, KV quantization (KIVI, TurboQuant) is the most stable across workloads, SnapKV pruning shows moderate degradation, summarization suffers significant loss — [arXiv 2607.05399](https://arxiv.org/pdf/2607.05399)
- QSpec: W4A4 joint quantization "suffers from substantial performance degradation on multi-step reasoning tasks", while W4A4-draft plus W4A16-verify keeps W4A16 quality — [QSpec PDF](https://i.cs.hku.hk/~cwu/papers/jtzhao-emnlp25.pdf)
- DBloom: "The greedy argmax is not bit-identical across implementations ... batching and floating-point reduction order can flip a near-tie and shift where a run reaches end-of-sequence" — [arXiv 2608.30427](https://arxiv.org/abs/2608.30427)
- TSS changes task scores (Translation BLEU 0.131 to 0.237) because skipped-layer verification replaces the full target — [TSS](https://arxiv.org/abs/2609.26100)
- SwiftKV combined with FP8 per-token KV quantization kept accuracy within 0.4 points without quantization-aware finetuning — [SwiftKV](https://arxiv.org/pdf/2410.03960)

### Inferences
- "Equal or better accuracy" under stacking has to be measured on task suites, not perplexity alone: MoE-XBench shows perplexity additivity hiding super-additive task loss.
- Speculation with strict verification adds no accuracy loss beyond what the target's own quantization, sparsity and KV compression introduce, so the accuracy budget is spent entirely by the non-speculative methods; verify-side approximations (TSS layer skipping, sparse verification) spend it too.

### Gaps
- No 2026 paper found measures compounded quality loss for a dense 8B model stacking 2-bit weights, activation sparsity, KV quantization and speculative verification.
- No IQ2-class quality numbers for Llama-3.1-8B, Qwen3-8B or Qwen3-30B-A3B were collected here.

## Thin or unexplored

- Two-bit targets and speculation: no measurement of draft acceptance or verify-to-decode cost for IQ2, QuIP#, AQLM or QTIP targets, on any GPU.
- Verify-batch kernels for codebook quants: no published consumer-GPU numbers on how IQ2 or other vector-quant matmuls scale from 1 to 16 rows, which decides whether low-bit and speculation multiply.
- Shared sparsity masks across a verify block for dense 8B models (TEAL or CATS with EAGLE-3, DFlash, DSpark): only chunk-trained BlockFFN on Jetson and a sparse-verification abstract exist.
- KV quantization vs acceptance for a separate drafter (EAGLE-3, DFlash, DSpark, MTP) in llama.cpp-style q8_0 and q4_0 caches: unmeasured; only self-speculative QuantSpec exists.
- Block-diffusion drafters on MoE targets at batch 1 on consumer GPUs: llama.cpp DFlash and EAGLE3 on GPT-OSS-20B run at or below baseline, and the only bound (2.34x oracle) is on an A100.
- Consumer-GPU results for DFlare, DBloom, DDTree, JetSpec and DSpark beyond the single llama.cpp RTX 4090 table.
- Mainline llama.cpp tg numbers for the RTX 3080 Ti, for Llama-3.2-1B on any listed GPU, and for any IQ2 quant: absent from the public scoreboards.
- Chaining draftless (ngram) and model drafters in one verify pass: llama.cpp gives draftless precedence and the one feature request has no data.
- Compounded task-accuracy loss for dense 8B models stacking more than two lossy methods: MoE-XBench covers MoE prune, quant and KV only.
- Long-context (32K) single-stream speculative decoding on consumer GPUs where KV reads exceed weight reads: no measured tau or speedup for 8B targets at 32K on these cards.
- Sub-1B and 1B decode in llama.cpp is launch-bound (about 16% of floor in LocalScore); the only fixes found are hobbyist megakernels (Qwen3-0.6B, Qwen3.5-0.8B), with no peer-reviewed consumer-GPU result for a 1B Llama.

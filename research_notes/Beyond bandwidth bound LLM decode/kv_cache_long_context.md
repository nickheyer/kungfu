# KV cache and attention bandwidth per decoded token at 8K-128K context, and training-free methods that remove it

Scope: single-stream decode for GQA models of 1B-8B parameters, target RTX 3080 Ti (12 GB, 912 GB/s). Byte tables are computed from the cited model configs and file sizes. All throughput and accuracy figures are the cited papers' own numbers on their own hardware unless stated otherwise.

## 1. Per-token byte budget: KV cache versus weights at 4K, 32K, 128K

### Takeaway
With about 4.9-bit weights (Q4_K_M density) and fp16 KV, the KV cache is 5-19% of the bytes read per decoded token at 4K, 29-66% at 32K, and 62-88% at 128K across Llama 3.2 1B/3B, Llama 3.1 8B, Qwen2.5-7B, and Qwen3-8B. Even with KV compressed to 4.5 bits, KV is 10-35% of bytes at 32K and 31-68% at 128K, and an 8B model at 128K only fits in 12 GB with KV at roughly 4.5 bits or fewer.

### Cited Findings
- Llama 3.2 1B config: 16 layers, 32 query heads, 8 KV heads, head_dim 64, tied embeddings, 128256 vocab, 131072 max positions - [HF config](https://huggingface.co/unsloth/Llama-3.2-1B/raw/main/config.json)
- Llama 3.2 3B config: 28 layers, 24 query heads, 8 KV heads, head_dim 128, tied embeddings, 131072 max positions - [HF config](https://huggingface.co/unsloth/Llama-3.2-3B/raw/main/config.json)
- Llama 3.1 8B config: 32 layers, 32 query heads, 8 KV heads, head_dim 128, untied embeddings, 131072 max positions - [HF config](https://huggingface.co/unsloth/Meta-Llama-3.1-8B/raw/main/config.json)
- Qwen3-8B config: 36 layers, 32 query heads, 8 KV heads, head_dim 128, max_position_embeddings 40960 (native window; 128K needs RoPE scaling) - [HF config](https://huggingface.co/Qwen/Qwen3-8B/raw/main/config.json)
- Qwen2.5-7B config: 28 layers, 28 query heads, 4 KV heads, hidden 3584 (head_dim 128), 131072 max positions - [HF config](https://huggingface.co/Qwen/Qwen2.5-7B/raw/main/config.json)
- Llama-3.1-8B-Instruct GGUF sizes: Q4_K_M 4.92 GB, Q4_K_S 4.69 GB, IQ4_XS 4.45 GB, IQ3_M 3.78 GB, Q8_0 8.54 GB - [bartowski GGUF repo](https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF)
- Measured llama.cpp decode on DGX Spark GB10 with Nemotron-3-Nano-30B-A3B Q4_K_XL (hybrid model): f16 KV 44.7 / 44.6 / 38.0 tok/s at about 6K / 24K / 110K depth; q8_0 KV 44.9 / 39.7 / 25.0; q4_0 KV 45.0 / 39.3 / 24.0. Quantized KV decodes slower than f16 at long depth in that kernel set - [llama.cpp discussion #20969](https://github.com/ggml-org/llama.cpp/discussions/20969)
- Measured llama.cpp decode on Apple M1 Ultra, Qwen3.5-397B-A17B: 19.98 tok/s at depth 0, 12.82 at 100K, 8.01 at 250K - [llama.cpp discussion #20969](https://github.com/ggml-org/llama.cpp/discussions/20969)

### Inferences
- fp16 K+V bytes per token = 2 x layers x kv_heads x head_dim x 2 B. Llama 3.2 1B 32 KiB; Llama 3.2 3B 112 KiB; Llama 3.1 8B 128 KiB; Qwen2.5-7B 56 KiB; Qwen3-8B 144 KiB (computed from the configs above).
- Weight bytes per token assume the Llama-3.1-8B Q4_K_M density of 4.92 GB / 8.03B params = 0.613 B/param (about 4.9 bits), applied to 1.24B, 3.21B, 7.62B, 8.19B params for the other models. Parameter counts are the published model sizes, not fetched in this pass. The full LM head is read every token, the embedding table is a single-row gather, so per-token weight bytes are slightly below file size for untied models.

| Model | KV @4K / 32K / 128K (fp16, GB) | Weights (GB) | KV share fp16 KV (4K / 32K / 128K) | KV share 8.5-bit KV (q8_0-like) | KV share 4.5-bit KV |
|---|---|---|---|---|---|
| Llama 3.2 1B | 0.134 / 1.07 / 4.29 | 0.76 | 15.0% / 58.6% / 85.0% | 8.6% / 42.9% / 75.0% | 4.7% / 28.4% / 61.4% |
| Llama 3.2 3B | 0.470 / 3.76 / 15.0 | 1.97 | 19.3% / 65.6% / 88.4% | 11.2% / 50.3% / 80.2% | 6.3% / 34.9% / 68.2% |
| Llama 3.1 8B | 0.537 / 4.29 / 17.2 | 4.92 | 9.8% / 46.6% / 77.7% | 5.5% / 31.7% / 65.0% | 3.0% / 19.7% / 49.5% |
| Qwen2.5-7B | 0.235 / 1.88 / 7.52 | 4.67 | 4.8% / 28.7% / 61.7% | 2.6% / 17.6% / 46.1% | 1.4% / 10.2% / 31.2% |
| Qwen3-8B | 0.604 / 4.83 / 19.3 | 5.02 | 10.7% / 49.0% / 79.4% | 6.0% / 33.8% / 67.2% | 3.3% / 21.3% / 52.0% |

- Fit in 12 GB at 128K (weights + KV, before activations and compute buffers): Llama 3.1 8B fp16 KV 22.1 GB no, 8.5-bit 14.0 GB no, 4.5-bit 9.75 GB yes. Qwen3-8B 4.5-bit 10.5 GB marginal. Qwen2.5-7B 8.5-bit 8.7 GB yes. Llama 3.2 3B 8.5-bit 9.95 GB yes. Llama 3.2 1B fp16 5.1 GB yes.
- Pure bandwidth floors at 912 GB/s for Llama 3.1 8B Q4_K_M at 32K: fp16 KV 9.2 GB/token, 10.1 ms, 99 tok/s; 8.5-bit KV 7.2 GB, 7.9 ms, 127 tok/s; 4.5-bit KV 6.1 GB, 6.7 ms, 149 tok/s; weights alone 4.92 GB, 5.4 ms, 185 tok/s. KV compression alone, with weights fixed, caps the 32K gain at 1.87x over fp16 KV and 1.46x over 8.5-bit KV.
- Arithmetic against the 5x goal: if the baseline is Q4_K_M weights plus 8.5-bit KV at 32K (7.2 GB/token), 5x means about 1.44 GB of traffic per emitted token, which is below the weights alone. No KV method reaches that by itself. It requires weight and KV traffic to be amortized over several emitted tokens per forward pass (speculative verify rows share one read of weights and of the KV history), plus KV compression so the KV term does not dominate the amortized pass. Example: weights 4.92 GB plus KV compressed 16x from fp16 (0.27 GB at 32K) = 5.19 GB per pass needs about 3.6 accepted tokens per pass; with 8.5-bit KV (2.28 GB) it needs 5.0. If the llama.cpp baseline also runs speculative decoding, the bar rises by its acceptance length.
- The GB10 llama.cpp numbers show the realized decode floor at long depth is set by kernel dequant cost, not bytes, for quantized KV in that implementation. A new runtime gains on llama.cpp at long context partly by making low-bit KV actually cheaper to read than f16.

### Gaps
- No published llama.cpp or other measured bs1 decode numbers for a dense 8B GQA model at 32K and 128K depth on an RTX 3080 Ti or 3090 were found; the search returned only qualitative claims or other GPUs. A local llama-bench sweep at -d 4096/32768/131072 with f16, q8_0, q4_0 KV is needed to fix the baseline.
- Weight bytes for 1B and 3B were scaled from 8B Q4_K_M density, not taken from actual GGUF file sizes.

## 2. KV quantization: bits per element at under 1 percent loss, with and without calibration, and decode kernel throughput

### Takeaway
Without calibration, about 3-3.5 effective bits per element (counting scale overhead) is the reliable under-1-percent point (TurboQuant 3.5 bits quality-neutral; InnerQBase 3.5 effective bits within 0.4 points on GSM8K; KVQuant 3-bit under 0.1 PPL). 2-bit without calibration is measurably lossy on reasoning tasks (InnerQSmall -4.7 GSM8K). With one-shot calibration AQUA-KV reaches 2-2.5 bits at under 1 percent relative error, a 6.4-8x reduction from fp16. Measured decode kernel gains are 1.7x-2.7x over fp16 attention at batch 1, with an 8x attention-logit claim for TurboQuant on H100.

### Cited Findings
- TurboQuant (Google): data-oblivious online vector quantization; absolute quality neutrality at 3.5 bits per channel and marginal degradation at 2.5 bits per channel for KV cache; distortion within about 2.7x of the information-theoretic lower bound; arXiv 2504.19874 - [arXiv](https://arxiv.org/abs/2504.19874)
- TurboQuant presented at ICLR 2026, announced March 2026; at least 6x KV memory reduction on needle-in-a-haystack with perfect retrieval - [The Elec](https://www.thelec.net/news/articleView.html?idxno=6179); [morphllm summary](https://www.morphllm.com/turboquant)
- TurboQuant 4-bit gives up to 8x faster attention-logit computation versus unquantized 32-bit keys on H100, against a JAX baseline; this applies to logit computation, not end-to-end decode - [Tom's Hardware](https://tomshardware.com/tech-industry/artificial-intelligence/googles-turboquant-compresses-llm-kv-caches-to-3-bits-with-no-accuracy-loss) (press relay of Google's blog; original post not retrieved)
- InnerQ (arXiv 2602.23200, Feb 2026): tuning-free, groups along the inner dimension; keys 3-bit symmetric in all variants; values 3-bit (Base), 2-bit (Small), 2-bit hybrid sym/asym (Hybrid). Effective bits per number including overhead: InnerQBase 3.5, InnerQHybrid 3.25, InnerQSmall 3.0, KIVI 3.0, TurboQuant 3.75 - [arXiv HTML](https://arxiv.org/html/2602.23200v2)
- InnerQ accuracy vs fp16 (averages over Llama 3.2-1B, 3.2-3B, 3.1-8B, 2-7B, Mistral 7B v0.3): GSM8K 26.52 baseline vs Base 26.16 (-0.36), Hybrid 24.19 (-2.33), Small 21.79 (-4.73); HumanEval 48.47 vs Base 49.39, Hybrid 46.64, Small 46.34; MBPP 47.80 vs Base 47.40, Hybrid 46.00, Small 45.30. LongBench: Base +0.4 over KIVI, Small 4.3 below Base - [arXiv HTML](https://arxiv.org/html/2602.23200v2)
- InnerQ kernel timing: fused dequant-GEMV CUDA kernels on Jetson Xavier NX, batch 1, contexts 512-32768, one Llama 3.1 8B layer: about 2.7x over fp16, about 1.3x over KIVI (InnerQSmall about 1.4x at 32768), 1.2-1.3x over TurboQuant. No calibration; per-channel key normalization computed at end of prefill and folded into W_Q and W_K; sink window 32 plus recent window 96 kept in full precision, group size 32 - [arXiv HTML](https://arxiv.org/html/2602.23200v2)
- KIVI: tuning-free asymmetric 2-bit (per-channel keys, per-token values); 2.6x less peak memory including weights, up to 4x larger batch, 2.35x-3.47x throughput; ICML 2024 - [arXiv](https://arxiv.org/abs/2402.02750); [GitHub](https://github.com/jy-yuan/KIVI)
- KVQuant: under 0.1 perplexity degradation at 3-bit on Wikitext-2 and C4; 1M context on one A100-80GB and 10M on 8 GPUs for LLaMA-7B; custom kernels up to about 1.7x faster than fp16 matrix-vector for LLaMA-7B; NeurIPS 2024 - [arXiv](https://arxiv.org/abs/2401.18079)
- QServe (QoQ): W4A8KV4 with SmoothAttention to limit 4-bit KV degradation; max serving throughput vs TensorRT-LLM: Llama-3-8B 1.2x on A100, 1.4x on L40S; Qwen1.5-72B 2.4x on A100, 3.5x on L40S (batched serving, not bs1) - [arXiv](https://arxiv.org/abs/2405.04532)
- AQUA-KV ("Cache Me If You Must"): near-lossless at 2-2.5 bits per value with under 1 percent relative error in perplexity and LongBench on Llama 3.2; one-shot calibration on one GPU in 1-6 hours including 70B models - [arXiv](https://arxiv.org/abs/2501.19392)
- Palu: low-rank projection compresses KV by 50 percent; up to 1.89x on RoPE attention; with quantization up to 2.91x on RoPE attention and up to 1.19 lower perplexity than quantization-only methods - [arXiv](https://arxiv.org/abs/2407.21118)
- KVTC (NVIDIA, ICLR 2026): PCA decorrelation plus adaptive quantization plus entropy coding with a basis calibrated once; up to 20x compression keeping reasoning and long-context accuracy, 40x or more for specific uses; evaluated on Llama 3.1 8B and Llama 3.3 70B; designed for compact on-GPU and off-GPU storage - [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/3fb6f10bd2784f6cfb6a6ed6280df40c-Abstract-Conference.html); [arXiv](https://arxiv.org/pdf/2511.01815v2)
- QuantWM (arXiv 2609.26425, Sept 2026, video world models): training-free causal 2-bit KV; key quantization gives smaller reconstruction error than value quantization but much larger output degradation, so it optimizes for preserving attention logits and token selection - [arXiv](https://arxiv.org/abs/2609.26425v2)
- Community llama.cpp perplexity on Radeon 7900XTX (dataset unnamed): f16 6.8976; q8_0 -0.38%; q4_0 -1.62%; tbq4_0 -1.60%; tbq3_0 +1.73%; pq3_0 +2.52%; K tbq3_0 / V q8_0 +1.75%; K q8_0 / V pq3_0 +0.37%. tq3_0 on both K and V for Llama-3.3-70B Q4_K_M on 2x RTX 3090: 4.40 vs 4.09 PPL on WikiText-2 (+7.6%) - [llama.cpp discussion #20969](https://github.com/ggml-org/llama.cpp/discussions/20969)
- Community result: skipping value dequantization for near-zero attention weights gave +22.8% decode at 32K on M5 Max, and +5% on plain q8_0 - [llama.cpp discussion #20969](https://github.com/ggml-org/llama.cpp/discussions/20969)

### Inferences
- A 4x cut from fp16 (4 effective bits) is available training-free at negligible loss on 1B-8B Llama-family models (InnerQBase, KVQuant, TurboQuant). An 8x cut (2 effective bits) currently needs calibration (AQUA-KV) or a learned transform (KVTC, which is a storage format and not a decode-time read path).
- Keys are the sensitive half. InnerQ keeps keys at 3 bits in every variant, QuantWM reports keys dominate output error, and the community mixed K/V perplexities are best when K is high precision. A runtime should spend bits asymmetrically: about 3-4 bits K and 2 bits V.
- The 7900XTX perplexities below the f16 baseline at 4 bits indicate noise in that measurement setup; KLD against fp16 logits is the better acceptance test for a checkpoint-only runtime.
- The value-skip trick (skip V rows whose softmax weight is near zero) is a cheap, training-free form of decode sparsity that stacks with any KV quant format.

### Gaps
- TurboQuant's model list, LongBench numbers, and kernel timings were not visible on the abstract page; the 8x H100 figure is only available via press relay in this pass.
- Unverified 2026 leads seen only as search hits or citations: RoPE-Aware Bit Allocation for KV-Cache Quantization ([arXiv 2606.24033](https://arxiv.org/pdf/2606.24033)), KV Cache Compression Through the Lens of Transform Coding ([arXiv 2608.14191](https://arxiv.org/pdf/2608.14191)), Oscar (arXiv 2605.17757), Kitty (arXiv 2511.18643), RotateKV ([arXiv 2501.16383](https://arxiv.org/pdf/2501.16383)), NSNQuant ([arXiv 2505.18231](https://arxiv.org/pdf/2505.18231)).
- No paper found with bs1 decode kernel timings for 2-3 bit KV on an Ampere consumer GPU (InnerQ uses Jetson Xavier NX; QServe uses A100/L40S batched serving).
- Palu's hardware and AQUA-KV's venue were not stated on the abstract pages.

## 3. Query-aware sparse attention at decode without training

### Takeaway
Decode-time page or cluster retrieval reads about 1.5-6 percent of the KV per step with under 1-3 points loss on RULER and LongBench at 32K-128K (RetroInfer RULER-128K -0.73% on Llama 3.1 8B at a 1.8% cluster budget; UNIQUE 97.5% of dense RULER-32K at 1.6% of tokens). Realized decode speedups are 2.9x-4.5x over full attention on A100 and 5.3x end-to-end on H100 at batch 80; no paper in this set reports a measured bs1 consumer-GPU result (MagicPIG's RTX 4090 number is simulated). XAttention, SpargeAttn, FlexPrefill, and MInference accelerate prefill and do not reduce decode KV reads.

### Cited Findings
- Quest (ICML 2024): per-page min/max key summaries, loads only top-K critical pages per query. The abstract as written says up to 2.23x self-attention speedup which reduces inference latency by 7.03x - [arXiv](https://arxiv.org/abs/2406.10774). The ordering is self-contradictory (attention speedup cannot be smaller than end-to-end), so the consistent reading is 7.03x attention and 2.23x end-to-end; treat as unresolved until the paper body is checked.
- Quest selects pages with the highest query-aware upper bounds from per-page min/max key summaries and is used as a selector in 2026 systems work (HiSparse) - [HiSparse arXiv 2608.07009](https://arxiv.org/html/2608.07009)
- RetroInfer (VLDB vol. 19): wave index of key clusters with GPU-CPU split; retrieval budget about 1.8% (150 of 8192 clusters at 128K) plus a 4+64 token steady zone plus an estimation zone of 23% of clusters; PCIe traffic under 2% of full attention - [arXiv HTML](https://arxiv.org/html/2505.02922v2); [VLDB PDF](https://www.vldb.org/pvldb/vol19/p1016-lu.pdf)
- RetroInfer accuracy: RULER-128K average drop vs full attention 0.73% (Llama 3.1 8B), 0.78% (Qwen2.5-7B), 1.46% (Llama3-8B-1048K), 0.27% (Qwen2.5-72B); NIAH 100% up to 1M on Llama3-8B-1048K. LongBench text claims at most 0.23% drop, but its Table 2 shows Qwen2.5-7B 35.74 vs 36.23 and Qwen2.5-72B 39.15 vs 39.70 - [arXiv HTML](https://arxiv.org/html/2505.02922v2)
- RetroInfer speed on A100-80GB (PCIe 4.0 to CPU): 4.4x / 4.4x / 4.5x decode throughput over full attention at 60K / 120K / 240K; per model 3.6x Llama 3.1 8B, 2.9x Qwen2.5-7B; 3.4x over GPU-resident Quest; 10.5x over MagicPIG at 1M. Full attention at 128K on one A100 gains little beyond batch 3 - [arXiv HTML](https://arxiv.org/html/2505.02922v2)
- UNIQUE (arXiv 2605.27740, May 2026): page-level top-k using per-page key mean and std; training-free or with optional sparsity-aware fine-tuning. On Ministral-3-8B-Instruct, RULER-32K dense 93.15 vs 90.78 at a 512-token budget (about 1.6% of 32K) and 92.53 at 2048 (about 6.3%); LongBench-Pro dense 37.70 vs 34.70 / 35.91 / 36.58 / 36.72 / 37.11 at 128 / 256 / 512 / 1024 / 2048 - [arXiv HTML](https://arxiv.org/html/2605.27740v1)
- UNIQUE baselines at budget 512: Quest RULER-32K 84.14 and LongBench-Pro 21.72 (Quest 91.67 RULER at 2048); InfLLM LongBench-Pro 34.99, RULER-32K 73.30; H2O LongBench-Pro 29.04, RULER-32K 56.00 - [arXiv HTML](https://arxiv.org/html/2605.27740v1)
- UNIQUE speed on H100: attention kernel up to 11.4x over dense FlashInfer at 32K (32 Q heads, 8 KV heads, head_dim 128, budget 512, batch 80); end-to-end up to 5.3x in vLLM at 32K batch 80; fused criticality kernel 1.73x over unfused; top-k 2.0x faster than FlashInfer's and 4.8x faster than torch.topk - [arXiv HTML](https://arxiv.org/html/2605.27740v1)
- ShadowKV: low-rank pre-RoPE keys on GPU, values offloaded to CPU, sparse reconstruction per step; up to 6x larger batch and up to 3.04x throughput on A100 across Llama-3.1-8B, Llama-3-8B-1M, GLM-4-9B-1M, Yi-9B-200K, Phi-3-Mini-128K, Qwen2-7B-128K - [arXiv](https://arxiv.org/abs/2410.21465)
- MagicPIG: LSH sampling on CPU; 54 ms decode latency for Llama-3.1-8B-Instruct at 96K on a single RTX 4090 (simulated, not measured); per-GPU throughput gains A100 1.5x, L20 5.0x, RTX 4090 3.3x - [arXiv HTML](https://arxiv.org/html/2410.16179v4)
- Twilight: top-p (nucleus) pruning layered on a base sparse selector; prunes up to 98% of tokens with nearly no accuracy loss on medium and long context tasks. v2 reports up to 15.4x over full attention and 3.9x end-to-end per-token latency; v5 (Nov 2025) reports 1.4x attention and 1.35x end-to-end over prior sparse methods and up to 15.8x over full attention - [arXiv v3](https://arxiv.org/html/2502.02770v3); [arXiv v2](https://arxiv.org/html/2502.02770v2); [arXiv v5](https://arxiv.org/html/2502.02770v5)
- The Sparse Frontier (arXiv 2504.17768, v3 June 2026): six training-free methods, up to 128K, sparsity up to 0.95; token-to-page selection is feasible at decode and gives higher sparsity tolerance than prefill; longer sequences tolerate higher sparsity, so fixed-budget methods are suboptimal; larger sparse models beat smaller dense ones at equal cost - [arXiv](https://arxiv.org/abs/2504.17768)
- LServe (MLSys 2025): converts half of attention heads to streaming heads and prunes KV pages with query-centric selection inside a unified block-sparse framework that also covers KV quantization; decoding 1.3-2.1x over vLLM; for Llama-3-8B decoding up to 3.3x over TensorRT-LLM - [HF paper page](https://huggingface.co/papers/2502.14866); [MLSys listing](https://mlsys.org/virtual/2025/3270)
- A 2026 survey states sparse decoding speedups are not automatic because of irregular memory gathers, poor coalescing, synchronization overhead, and control-flow fragmentation - [arXiv 2603.20397](https://arxiv.org/html/2603.20397v1)
- HySparse argues inference-only sparsity creates a training-inference mismatch that can accumulate error in long decoding - [arXiv 2602.03560](https://arxiv.org/pdf/2602.03560)
- Prefill-side methods: MInference reduces pre-filling latency up to 10x at 1M tokens on A100 - [arXiv 2407.02490](https://arxiv.org/abs/2407.02490); XAttention reports up to 13.5x attention acceleration in block-sparse prefill - [arXiv 2503.16428](https://arxiv.org/abs/2503.16428); FlexPrefill (ICLR 2025) adapts sparse patterns per head during prefill - [arXiv 2502.20766](https://arxiv.org/abs/2502.20766); SpargeAttn is a training-free sparse plus quantized attention kernel for language, image, and video models - [arXiv 2502.18137](https://arxiv.org/abs/2502.18137)

### Inferences
- For a single 8B stream at 32K, reading 1.6-6% of KV turns the 4.3 GB fp16 KV term into 70-270 MB, so KV falls below 5% of per-token bytes and weights dominate again. Sparse decode attention is the larger lever than quantization at 32K-128K, and the two multiply.
- Accuracy depends heavily on the selector: at the same 1.6% budget, UNIQUE's mean/std page score holds 90.78 RULER-32K while Quest's min/max bound drops to 84.14 and H2O to 56.00. Page statistics richer than min/max are worth the metadata bytes.
- The batch-80 and A100 offload numbers do not transfer directly to bs1 on a 12 GB card. At bs1 the selection kernel (scoring every page summary, top-k) is itself a full pass over the metadata, so metadata must be small relative to KV (Quest-style 2 vectors per page of 16 tokens is 1/8 of the fp16 key bytes).
- Prefill sparsity methods matter for time-to-first-token at 128K on the target GPU but do not change decoded tokens per second.

### Gaps
- No measured bs1 decode result on an RTX 30-series or 40-series GPU was found for Quest, RetroInfer, ShadowKV, UNIQUE, or Twilight.
- Quest's correct attention vs end-to-end number pairing needs the paper body.
- RULER at 128K for 1B-3B models under decode sparsity was not found.

## 4. Token eviction and merging, and the accuracy cliffs

### Takeaway
Eviction holds near full LongBench accuracy at 10-20 percent retention in single-query settings (PyramidKV parity at 12 percent; R-KV near 100 percent at 10 percent on reasoning traces) and gives about 2x decode latency at 3-4x reduction (KVzip). The cliffs are at budgets around 1-3 percent on retrieval-heavy RULER (H2O 56.00 vs 93.15 dense at 1.6 percent), below about one-third retention for some gated schemes, and whenever a cache compressed for one query is reused for another. A 1K-step retrofit (DMS) reaches 8x and beats training-free eviction.

### Cited Findings
- H2O at a 512-token budget on RULER-32K: 56.00 vs 93.15 dense; LongBench-Pro 29.04 vs 37.70 - [UNIQUE arXiv HTML](https://arxiv.org/html/2605.27740v1)
- PyramidKV: matches full KV performance retaining 12% of the KV cache; at 0.7% retention outperforms other compression methods by up to 20.5 absolute points on TREC - [arXiv](https://arxiv.org/abs/2406.02069)
- LookaheadKV (arXiv 2603.10899, Mar 2026) LongBench tables: SnapKV and PyramidKV within 1-2 points at 1024 and 2048 token budgets; SnapKV ahead by about 2 points on average at 512 - [arXiv](https://arxiv.org/html/2603.10899v1)
- PAGE (arXiv 2609.22157): PyramidKV leads SnapKV on Qwen2.5-1.5B, Qwen2.5-3B, Mistral-7B; H2O beats both on Qwen2.5-14B; its gated method beats SnapKV above about one-third retention and loses below it - [arXiv](https://arxiv.org/pdf/2609.22157)
- Nexus Sampling (arXiv 2606.23961): at 80% eviction (20% density) matches or exceeds SnapKV and PyramidKV on average across three models in the prefill-only regime - [arXiv](https://arxiv.org/pdf/2606.23961)
- KVzip (NeurIPS 2025 oral): query-agnostic eviction via context reconstruction; 3-4x KV reduction and about 2x lower FlashAttention decode latency with negligible loss on QA, retrieval, reasoning, code; up to 170K tokens; LLaMA3.1, Qwen2.5, Gemma3 - [arXiv](https://arxiv.org/abs/2505.23416)
- KVzip reports SnapKV accuracy declines significantly when the cache compressed for the first query is reused for later queries - [arXiv](https://arxiv.org/pdf/2505.23416)
- R-KV (NeurIPS 2025): redundancy-aware decode-time compression for reasoning models; nearly 100% of full-KV performance at 10% KV and 105% at 16%, while prior baselines reach about 60%; up to 13x larger batch and over 9x speedup in long generation vs full KV - [arXiv](https://arxiv.org/pdf/2505.24133); [project page](https://zefan-cai.github.io/R-KV.page)
- DMS ("Inference-Time Hyper-Scaling with KV Cache Compression", NVIDIA, NeurIPS 2025): retrofits existing LLMs with about 1K training steps for 8x KV compression; delays eviction and implicitly merges representations; beats training-free TOVA and H2O; at matched memory-read budget improves Qwen-R1 32B by 9.1 AIME24, 7.6 GPQA, 9.6 LiveCodeBench - [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/0d781fa5f639bf2caf728a68e9678362-Abstract-Conference.html); [arXiv](https://arxiv.org/abs/2506.05345v2)
- Dustin argues permanent eviction loses information because token importance shifts during decoding - [arXiv](https://arxiv.org/abs/2606.24957)
- QuantSpec reports sparse KV methods are much more lossy than quantization on tasks where the whole context matters - [arXiv](https://arxiv.org/pdf/2502.10424)

### Inferences
- For a runtime that must match accuracy on arbitrary prompts, permanent eviction is the riskiest tier: multi-turn reuse and shifting importance both break it. Query-aware retrieval that keeps the full cache (Section 3) reaches lower read fractions with smaller cliffs, at the cost of holding all KV in memory (which forces quantization at 128K on 12 GB).
- DMS shows light post-hoc adaptation (about 1K steps) buys 8x with better accuracy than training-free eviction, which fits the brief's allowance for light adaptation.
- R-KV's regime (long generated reasoning traces) is the one case where eviction is near-lossless at 10%, because the redundancy is in self-generated tokens.

### Gaps
- No single benchmark defines an accuracy cliff across eviction methods; the cliff points above are assembled from different papers, models, and metrics.
- Eviction results for 1B-3B models are sparse (PAGE has Qwen2.5-1.5B and 3B).

## 5. Cross-layer and cross-head KV sharing post-hoc, and the retraining needed

### Takeaway
Training-free cross-layer methods reach 1.3x to 8x: KVSharer about 28-30 percent savings, MiniCache 5.02x only when combined with 4-bit quantization (merging alone is about 1.25x), xKV up to 8x with up to 4.23x end-to-end speedup (ICML 2026). GQA-to-MLA conversion (TransMLA) reports 68.75 percent KV reduction training-free with 1.65 percent average drop; larger reductions use fine-tuning. Architecture-level cross-layer attention (CLA-style) requires training.

### Cited Findings
- xKV (ICML 2026): post-training joint low-rank factorization of grouped layers' KV into a shared subspace, exploiting aligned dominant singular vectors across layers; up to 8x compression preserving accuracy on long-context and multi-turn tasks; up to 4.23x end-to-end speedup over full attention with Selective Reconstruction; 30% higher throughput than baselines at similar accuracy - [arXiv](https://arxiv.org/abs/2503.18893)
- MiniCache: training-free merging of similar KV states in adjacent middle-to-deep layers; with 4-bit quantization reaches up to 5.02x compression on LLaMA-2-7B ShareGPT, about 5x throughput, 41% memory reduction vs fp16 full cache - [arXiv](https://arxiv.org/abs/2405.14366)
- KVSharer: shares a layer's cache with a dissimilar layer, which hurts less than sharing similar layers; one summary reports 25% of layers shared gives 28% memory savings, 1.65x generation acceleration, and at most 10% PPL increase - [arXiv](https://arxiv.org/pdf/2410.18517); [Medium summary](https://medium.com/@techsachin/kvsharer-method-for-efficient-llm-inference-via-dissimilar-kv-cache-sharing-between-layers-b8d595ad1b0a)
- Aggregator summary: training-free xKV and CommonKV reach 50-75% cache reduction with 0.4-0.9% perplexity increase, versus CLA and FusedKV which need retraining for similar compression - [emergentmind](https://www.emergentmind.com/topics/cross-layer-kv-cache-sharing); CommonKV - [ResearchGate](https://www.researchgate.net/publication/394920409_CommonKV_Compressing_KV_Cache_with_Cross-layer_Parameter_Sharing)
- TransMLA: converts a pretrained GQA model to MLA. Earlier version: compressing 93% of LLaMA-2-7B KV gives 10.6x inference speedup at 8K (measured in vLLM). Later version: 68.75% KV reduction training-free with 1.65% average drop on 6 benchmarks. Its authors report MHA2MLA suffered a 21.85% decline at comparable compression - [arXiv v4](https://arxiv.org/html/2502.07864v4); [arXiv v3](https://arxiv.org/html/2502.07864v3)

### Inferences
- MiniCache's 5.02x includes a 4x factor from 4-bit quantization, so cross-layer merging contributes about 1.25x by itself; xKV's low-rank cross-layer factorization is the only training-free sharing result here near 8x.
- GQA-to-MLA conversion attacks the KV term structurally and is a natural target for light post-hoc adaptation; it composes with quantization (the latent vector is quantizable) and with page-sparse retrieval on the latent.
- Low-rank KV (Palu, xKV, ShadowKV keys) adds a reconstruction matmul per step; at bs1 on Ampere this compute is cheap relative to the bytes saved, but the kernel has to fuse reconstruction into the attention read to realize it.

### Gaps
- xKV model names and accuracy deltas were not on the abstract page.
- TransMLA's two version numbers conflict and the amount of fine-tuning behind the 93% figure was not confirmed.
- No retrained cross-layer attention conversion (CLA from a pretrained checkpoint) with measured token counts was found in this pass.

## 6. Attention computation itself at 32K on the target GPU, and kernel floors

### Takeaway
By byte accounting, at 32K an 8B GQA model with Q4_K_M weights spends 47-49 percent of its per-token traffic on fp16 KV (4.7 ms of a 10.1 ms floor at 912 GB/s for Llama 3.1 8B) and 32-34 percent with 8.5-bit KV. Single-stream GQA decode attention is memory-bound by a wide margin, so its floor is KV bytes divided by bandwidth; but measured quantized-KV kernels can run slower than f16 at long depth, so dequant cost, not bytes, sets the realized floor in current implementations.

### Cited Findings
- Llama 3.1 8B: 32 query heads over 8 KV heads (GQA ratio 4), head_dim 128, 32 layers - [HF config](https://huggingface.co/unsloth/Meta-Llama-3.1-8B/raw/main/config.json)
- Qwen3-8B: 32 query heads over 8 KV heads, 36 layers - [HF config](https://huggingface.co/Qwen/Qwen3-8B/raw/main/config.json)
- Llama-3.1-8B Q4_K_M is 4.92 GB - [bartowski GGUF repo](https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF)
- llama.cpp on DGX Spark: q8_0 and q4_0 KV decode at 25.0 and 24.0 tok/s vs 38.0 for f16 at about 110K depth - [llama.cpp discussion #20969](https://github.com/ggml-org/llama.cpp/discussions/20969)
- InnerQ fused dequant-GEMV kernels: about 2.7x over fp16 per layer at bs1 up to 32K on Jetson Xavier NX - [arXiv HTML](https://arxiv.org/html/2602.23200v2)
- KVQuant kernels: up to about 1.7x over fp16 matrix-vector for LLaMA-7B - [arXiv](https://arxiv.org/abs/2401.18079)
- UNIQUE sparse decode kernel: up to 11.4x over dense FlashInfer at 32K with a 512-token budget (H100, batch 80) - [arXiv HTML](https://arxiv.org/html/2605.27740v1)
- Dustin: self-attention 27.85x faster at 32K with a 512-token budget (batch 16, H200) - [arXiv HTML](https://arxiv.org/html/2606.24957)
- Full attention at 128K on one A100 gains little beyond batch 3, consistent with attention being bandwidth-bound per sequence - [RetroInfer arXiv HTML](https://arxiv.org/html/2505.02922v2)

### Inferences
- Arithmetic intensity of bs1 GQA decode attention with fp16 KV and GQA ratio 4: each 2-byte K element feeds 4 query rows (4 multiply-adds, 8 FLOPs), so about 4 FLOP/byte for QK and the same for PV, far below the FLOP/byte ridge of any recent NVIDIA GPU. With 2-bit KV the byte count drops 8x and intensity rises to about 32 FLOP/byte plus dequant ALU work; dequant must be done once per element and reused across all query rows (GQA group and speculative verify rows) to stay bandwidth-bound.
- Floor at 912 GB/s for Llama 3.1 8B at 32K: fp16 KV 4.71 ms, 8.5-bit 2.50 ms, 4-bit 1.18 ms (no scale overhead), 2-bit 0.59 ms, 2-bit with 5% page-sparse reads about 0.03 ms. Weights stay at 5.4 ms for Q4_K_M, so beyond about 4-bit KV plus sparse reads, attention stops mattering and the 5x target is entirely a weight-traffic and tokens-per-pass problem.
- Speculative verify with k rows multiplies the useful work per KV byte by k without changing KV bytes, so verify-time attention costs nearly the same as a single decode row while bandwidth-bound; this is why combining speculative decoding with KV compression is multiplicative in principle.

### Gaps
- No measured decode-time breakdown (attention vs GEMV vs other) for an 8B GQA model at 32K on an RTX 3080 Ti or other Ampere consumer card was found.
- RTX 3080 Ti tensor-core throughput was not fetched, so the exact compute ridge point for dequant-heavy kernels is not quantified here.

## 7. Combined frontier: quantization with sparse attention, and with speculative decoding

### Takeaway
Pairwise stacks exist (low-rank plus quant in Palu, merging plus 4-bit in MiniCache, sparse heads plus quantized KV in LServe, 4-bit hierarchical KV plus self-speculation in QuantSpec, sparse-KV self-drafting in MagicDec and SparseSpec-L, sparse verification in Dustin), but no paper found reports a full quant x sparse x speculative stack with multiplicative gains at bs1 on a consumer GPU. The bs1 evidence is weak: SparseSpec-L gets 1.10x at batch 1 versus 2.45x at batch 4 at 16K on an A40.

### Cited Findings
- QuantSpec: self-speculative decoding where the draft shares the target's architecture with a hierarchical 4-bit quantized KV cache and 4-bit weights; over 90% acceptance; up to about 2.5x end-to-end speedup; about 1.3x less memory than sparse-KV self-speculation - [arXiv](https://arxiv.org/abs/2502.10424)
- MagicDec: self-speculation with a sparse-KV draft; up to 2.51x speedup for Llama3.1-8B at batch sizes 32-256 - [arXiv](https://arxiv.org/abs/2408.11049)
- SparseSpec-L (arXiv 2607.27735, submitted to AAAI 2027): training-free self-speculation; draft uses a sparsified, recallable KV with sink, recent, and top middle tokens at about 10% of context, re-indexed from the previous full verification pass; verification uses full KV, so output distribution is preserved. Single A40, HF Transformers, dense non-FlashAttention verify, contexts up to 64K - [arXiv HTML](https://arxiv.org/html/2607.27735v1)
- SparseSpec-L results: average accepted tokens per step 11.26 (Llama 3.1 8B, LongBench v2), 8.41 (InfiniteBench), 8.03 (Qwen2.5-7B LongBench v2); end-to-end speedup 2.79x / 2.60x on Llama 3.1 8B and 1.95x / 1.86x on Qwen2.5-7B; at 16K batch 1 / 2 / 4: 1.10x / 1.52x / 2.45x; MagicDec (SnapKV) 1.74x / 1.68x and EAGLE-3 0.89x / 0.92x on Llama 3.1 8B in the same setup - [arXiv HTML](https://arxiv.org/html/2607.27735v1)
- Dustin (arXiv 2606.24957): target verifies draft tokens against a 512-token KV subset (about 1.6% of 32K: 4 sinks, 16 recent, top draft-lookahead and historical scores); not lossless. On H200 (bf16, 4-GPU pipeline for 70B-class): end-to-end 6.61x (Qwen2.5-72B, batch 8, 32K) and 9.17x (batch 16), Llama-3.3-70B 6.28x / 7.18x; acceptance length 2.2-2.9. LongBench: Qwen2.5-72B 55.23 vs 55.81 (-0.58) at budget 512, 52.41 (-3.40) at 128; Llama-3.3-70B 53.23 vs 50.94 (+2.29). Requires offline profiling of retrieval heads and a 11.9-35.4 minute config search per target-draft pair - [arXiv HTML](https://arxiv.org/html/2606.24957)
- "Strong Drafts Need Compact Memories" (arXiv 2608.30252): independent draft model with a compressed KV memory built by a lightweight adaptor; target keeps full KV so output stays lossless - [arXiv HTML](https://arxiv.org/html/2608.30252)
- Hidden-state drift study (arXiv 2604.26412): draft reuse of the target's KV improves long-range acceptance, but end-to-end speedups remain marginal under current training pipelines - [arXiv](https://arxiv.org/pdf/2604.26412)
- SpecStream (arXiv 2609.33184): starts verification before the full KV history is restored and drafts during KV transfers - [arXiv HTML](https://arxiv.org/html/2609.33184v1)
- VeriCache (blog write-up): compressed cache for drafting, full cache restored only for verification, up to 4x throughput over full-KV with identical outputs - [Medium](https://devshahs.medium.com/vericache-making-lossy-kv-compression-exact-84ba36c318ec) (paper figure not confirmed)
- Palu low-rank plus quantization: up to 2.91x RoPE attention speedup - [arXiv](https://arxiv.org/abs/2407.21118); MiniCache merging plus 4-bit: 5.02x compression - [arXiv](https://arxiv.org/abs/2405.14366); LServe sparse plus quantized KV in one block-sparse framework - [HF paper page](https://huggingface.co/papers/2502.14866)
- Q-Hitter selects tokens by both attention importance and quantization robustness, producing a sparse and quantized KV - cited in [QuantSpec](https://arxiv.org/pdf/2502.10424)

### Inferences
- The lossless pattern is "compressed KV for drafting, full KV for verification". On a 12 GB card at 128K the full KV must still be held (quantized), so lossless self-speculation needs the verify path to read low-bit KV too; the accuracy bar then rests on the KV quantizer's KLD.
- SparseSpec-L's 11.26 accepted tokens per step on Llama 3.1 8B shows self-drafting with 10% sparse KV can amortize the full-KV verify read over many tokens; its weak bs1 speedup (1.10x at 16K) comes from a dense non-FlashAttention HF verify path where the verify pass is not bandwidth-optimal, so a bandwidth-bound verify kernel is the missing piece rather than the algorithm.
- Dustin shows verify rows can attend to a 1.6% subset with under 1 point LongBench change on 70B models, giving 27.85x attention speedup; applied to 1B-8B targets on a consumer GPU this has not been measured.
- Multiplicative bound for an 8B model at 32K on the target GPU: 2-bit KV (8x) times 5% page sparsity (20x) gives a KV term of about 27 MB per pass; with weights at 4.92 GB, tokens per second then scales almost purely with accepted tokens per pass.

### Gaps
- No paper found combining 2-3 bit KV quantization with page-sparse retrieval and speculative verification, measuring the product at bs1.
- No consumer-GPU (RTX 30/40-series) measurements for any of the speculative long-context methods above.
- MagicDec and QuantSpec venues and hardware were not confirmed from the abstract pages.

## Thin or unexplored

- Measured bs1 long-context decode on consumer GPUs: the strongest sparse-attention results are on A100 (RetroInfer), H100 at batch 80 (UNIQUE), H200 multi-GPU (Dustin), A40 with HF kernels (SparseSpec-L), and a simulated RTX 4090 (MagicPIG). No paper reports measured tokens per second for an 8B GQA model at 32K-128K on an RTX 3080 Ti, 3090, or 4090 with sparse decode attention.
- One fused kernel for low-bit KV plus page-sparse reads: page selection metadata computed from quantized keys, top-k, gather, and dequant-attend in one pass. Twilight, LServe, and ShadowKV touch pieces; nothing found measures 2-3 bit KV times 2-6% page reads in a single bs1 kernel.
- Lossless sparse verification: Dustin's sparse verification is lossy; lossless schemes keep full-KV verify. A verify pass that attends sparsely with an exactness check (bounding the omitted softmax mass per row and falling back to full reads only when the bound fails) was not found in the literature.
- Shared page selection across speculative verify rows and across layers: verify rows attend to the same history, so one selection could serve all k rows and adjacent layers. No paper quantifies selection reuse across verify rows or layers at bs1.
- Low-bit KV formats designed for multi-row reuse: the llama.cpp GB10 data show q8_0 and q4_0 KV decoding slower than f16 at 110K, i.e. dequant-bound kernels. No paper designs a KV format around dequant-once, reuse-across-GQA-group-and-verify-rows on Ampere tensor cores.
- Accuracy cliffs for 1B-3B models: nearly all sparse attention and eviction evaluations use 7B-8B or larger; small models (where KV is 58-88% of traffic at 32K-128K) are almost unstudied under combined compression.
- Checkpoint-only automatic bit and budget allocation: AQUA-KV needs 1-6 h calibration, Dustin needs 12-35 min of profiling, and fixed budgets are suboptimal per The Sparse Frontier. A fast per-checkpoint calibration that picks K bits, V bits, page size, and sparsity budget per layer and head from a few minutes of data was not found.
- Value-side sparsity in decode: skipping V dequant for near-zero softmax weights gave +22.8% at 32K in a llama.cpp community test; no paper measures it alongside low-bit V and page-sparse K.
- GQA-to-MLA conversion combined with quantization and sparse retrieval on the latent, under light adaptation, measured on consumer GPUs.
- Sparse decode under RoPE scaling (YaRN) on models whose native window is shorter than the deployed context (Qwen3-8B native 40960): no study of whether page statistics remain reliable past the native window.
- The 4K-16K regime: KV is under 20% of traffic for 8B models at 4K, so every KV method above yields little there; no paper in this set pairs KV compression with weight-traffic amortization and reports gains across the whole 4K-128K range.

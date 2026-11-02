# Beyond the weight read: decoupling per-token decode cost from weight bytes (published work through 2026-11-01)

Scope: research directions in which a transformer's single-stream decode cost per token is no longer "stream every weight byte once per token". Target platform in mind: RTX 3080 Ti (12 GB, 912 GB/s, 6 MB L2), 24-thread CPU, 62 GB host RAM, beating llama.cpp best config (heavy quant plus speculative drafting) by 5x single-stream with equal accuracy, for arbitrary checkpoints with light post-hoc adaptation. arXiv IDs of the form 26MM.xxxxx are 2026 preprints (month MM). Speculative decoding and plain weight quantization are adjacent fronts and appear here only where they combine with a decoupling method.

Organizing frame used throughout (my accounting, not from a single source): per-token time is roughly (bytes touched per token) / (achieved bandwidth) + fixed overheads. Every direction below attacks one of five terms: (a) fewer weights touched per token (sparsity, pruning, memory layers, MoE-fication, layer reuse), (b) more tokens per weight read (multi-token, diffusion, parallel branches, superposition, wavefront batching), (c) weights generated or looked up instead of read (seeds, LUTs), (d) higher achieved bandwidth and fewer bubbles (megakernels), (e) a second bandwidth pool (host RAM over PCIe, CPU).

## 1. Replacing computation with retrieval or hashing (kNN-LM, memory layers, UltraMem, PKM, LSH neuron selection, LookupFFN, activation memoization). Which can be fitted post-hoc to a trained dense model?

### Takeaway
Retrieval and hash-based FFN replacements that cut per-token weight reads (memory layers, UltraMem, PEER, LookupFFN, fast feedforward networks) all require training from scratch or substantial recovery training; the post-hoc versions that work today on arbitrary checkpoints are dynamic-sparsity and memoization methods (TEAL family, SpAx, GRIFFIN, CMoE or ExpertWeaver MoE carving, LoRA-Drop), which deliver 1.2x to 2.6x decode speedups at small accuracy cost. kNN-LM-style retrieval adds work per token and does not reduce weight reads.

### Cited Findings
Retrieval-augmented LMs (add cost, do not remove weight reads)
- kNN-LM interpolates a pretrained LM with a nearest-neighbor datastore over cached hidden states, improving WikiText-103 perplexity to 15.79 (2.9 points) with no additional training; the datastore search is extra work per token on top of the full forward pass — [Khandelwal et al., ICLR 2020](https://arxiv.org/abs/1911.00172)
- kNN-LM's perplexity gains do not translate into better open-ended generation quality — [Wang et al., 2023, "kNN-LM Does Not Improve Open-Ended Text Generation"](https://arxiv.org/abs/2305.14625)
- RETRO with a 2-trillion-token retrieval database matched GPT-3 and Jurassic-1 on the Pile with 25x fewer parameters, i.e. retrieval can substitute for parameter bytes, but only when the model is trained with retrieval — [Borgeaud et al., DeepMind 2021](https://arxiv.org/abs/2112.04426)

Memory layers and product-key memories (trained architectures)
- Memory Layers at Scale (Meta FAIR, ICML 2025) uses a trainable product-key key-value lookup that adds parameters without adding compute; the two product-key halves factorize the key space so the full key set is never instantiated — [arXiv 2412.09764](https://arxiv.org/abs/2412.09764), [Moonlight review](https://www.themoonlight.io/review/memory-layers-at-scale)
- Reported: over 100% improvement on factual QA versus dense baselines at compute parity, outperforming dense and MoE on factual tasks; FLOP-parity gains hold from 134M to 8B parameters but are larger early in training, so benefit for heavily over-trained inference models is unconfirmed — [Graphcore Research summary](https://graphcore-research.github.io/memory-layers), [HF abstract](https://huggingface.co/papers/2412.09764)
- UltraMem (ByteDance, ICLR 2025) reports inference up to 6x faster than MoE at the same parameter count and compute, because a memory layer touches a few table rows rather than whole expert matrices — [arXiv 2411.12364](https://arxiv.org/abs/2411.12364)
- UltraMemV2 (ICLR 2026) reaches parity with 8-active-expert MoE at equal compute and parameters, validated up to 120B total and 2.5B activated parameters; +1.6 on long-context memorization, +6.2 on multi-round memorization, +7.9 on in-context learning; speed claims in the abstract are qualitative ("state-of-the-art inference speed", "lower latency and power") — [ICLR 2026 poster](https://iclr.cc/virtual/2026/poster/10009579), [HyperAI listing of arXiv 2508.18756](https://hyper.ai/en/papers/2508.18756)
- Post-hoc retrofit evidence is small-scale only: one 2026 paper swaps a few FFN layers of Qwen2.5-0.5B for sparse key-value memory, discards the dense FFN weights, and must run a recovery stage training only memory parameters because the swap initially degrades the model — [arXiv 2604.05248](https://arxiv.org/html/2604.05248)
- Another 2026 paper inserts PKM modules at layers 6, 12, 18 of Qwen and compares replacing the MLP against adding a zero-initialized memory branch beside the intact MLP (the latter adds bytes, it does not remove them) — [arXiv 2605.03229](https://arxiv.org/pdf/2605.03229)
- Fast-weight PKM (FwPKM, Zhao and Jones, revised Feb 2026) replaces the MLP with a sparse fast-weight memory, as a training-time architecture — [arXiv 2601.00671](https://arxiv.org/pdf/2601.00671)
- Hierarchical-memory pretraining (Oct 2025) frames PKM as a sparser MoE and claims memories can be added after pretraining — [arXiv 2510.02375](https://arxiv.org/pdf/2510.02375)

Hash or tree-based FFN selection
- LookupFFN (ICML 2023) replaces GEMM FFNs with learnable hash-table lookups (softmax-weighted average of buckets for differentiability), 2.51x faster than a vanilla FFN for CPU inference with similar RoBERTa pretraining quality; trained from scratch — [PMLR](https://proceedings.mlr.press/v202/zeng23a.html), [Moonlight review](https://www.themoonlight.io/en/review/lookupffn-making-transformers-compute-lite-for-cpu-inference)
- UltraFastBERT (fast feedforward networks) uses 12 of 4095 neurons per layer at inference with a 78x FFN speedup on CPU over an optimized baseline; BERT-scale, trained from scratch — [Belcak and Wattenhofer, arXiv 2311.10770](https://arxiv.org/abs/2311.10770)
- 2025 GPU LSH work concentrates on attention and KV selection, not FFN neurons: MagicPIG uses LSH sampling for attention with up to 5x decode throughput; Spotlight Attention hashes queries and keys — [MagicPIG arXiv 2410.16179](https://arxiv.org/abs/2410.16179), [Spotlight arXiv 2508.19740](https://web3.arxiv.org/pdf/2508.19740)
- SelectInfer (2026) does dynamic neuron selection via top-k sorting rather than LSH for on-device LLMs — [arXiv 2607.18081](https://arxiv.org/pdf/2607.18081)

Post-hoc dynamic sparsity (the practical "retrieve only the neurons you need" path)
- Deja Vu (ICML 2023) trains small post-hoc predictors of contextual sparsity and cut OPT-175B latency more than 2x versus FasterTransformer; relies on the high natural sparsity of ReLU-family OPT — [Liu et al., arXiv 2310.17157](https://arxiv.org/abs/2310.17157)
- TEAL (ICLR 2025), training-free magnitude sparsification of hidden states across the whole model: 40% to 50% model-wide sparsity, up to 1.53x and 1.8x wall-clock decode speedup at 40% and 50%, compatible with weight quantization — [arXiv 2408.14690](https://arxiv.org/html/2408.14690v3)
- R-Sparse (ICLR 2025), rank-aware training-free sparsity: 50% model-level sparsity with about 43% end-to-end efficiency gain — [arXiv 2504.19449](https://arxiv.org/abs/2504.19449)
- LaRoSA rotates activations per layer before top-k selection and reports more consistent wall-clock gains than TEAL at 50% on LLaMA3-8B and Qwen2.5-7B — [arXiv 2507.01299](https://arxiv.org/html/2507.01299)
- Prox (Jul 2026), training-free FFN sparsity via approximate intermediate-channel salience: 1.51x to 1.99x at 60% to 70% FFN sparsity on an A6000, throughput within 2.9% of TEAL, main gain is accuracy at high sparsity — [arXiv 2607.27591](https://arxiv.org/pdf/2607.27591)
- SpAx (Kim, Durvasula, Vijaykumar, Oct 2026), training-free, three-tier per-activation choice: full weight, 2-bit approximate weight, or skip. Built on llama.cpp, batch 1, RTX 4090 over PCIe 4.0 x8 with all weights offloaded — [arXiv 2610.02598](https://arxiv.org/html/2610.02598)
  - Llama-3.1-8B at no more than 10% perplexity increase: 3.86x average (up to 5.57x) over dense and 2.07x over TEAL for BF16 from CPU memory; 2.06x over dense (up to 2.74x) for Q4_K_M; with Q4_K_M weights the gain over TEAL shrinks to 1.07x average
  - Bytes per token (BF16): dense 13.96 GB, TEAL 7.95 GB, SpAx 3.76 GB; (Q4_K_M): dense 4.18 GB, TEAL 2.36 GB, SpAx 1.76 GB
  - Accuracy at +10% perplexity (WikiText-2 6.24 to 6.87): MMLU 65.4 to 62.1, GSM8K 50.2 to 43.9, ARC-C 52.1 to 49.1, WinoGrande 75.1 to 73.4
  - Models: Llama-3.1-8B, Gemma-4-31B, Qwen3.6-27B
- Turbo Sparse and ProSparse obtain about 90% FFN sparsity by ReLU-fication with continued pretraining (on the order of 150B tokens for dReLU), i.e. not "light" adaptation — [Turbo Sparse, arXiv 2406.05955](https://arxiv.org/abs/2406.05955)

Post-hoc dense-to-MoE carving
- CMoE, retitled "Analytical FFN-to-MoE Restructuring via Activation Pattern Analysis": minutes of processing, optional lightweight fine-tune on 2k samples, up to 1.17x speedup in compute-bound settings; earlier abstract: lossless at 75% activation with modest speedup, 25% activation maximizes efficiency, evaluated on Llama-2 7B and Llama-3 8B — [arXiv 2502.04416](https://arxiv.org/abs/2502.04416), [HF page](https://huggingface.co/papers/2502.04416)
- ExpertWeaver (Feb 2026), training-free with calibration, initializes routers from centroids of GLU gate weights; at 25% sparsity on Llama3-8B average 60.4 versus CMoE 57.2 — [arXiv 2602.15521](https://arxiv.org/html/2602.15521v1)
- L0-MoE: up to 2.5x speedup over dense on Llama-3-8B, Mistral-7B, Qwen2-7B, but needs about 30B tokens of training — [arXiv 2609.21672](https://arxiv.org/abs/2609.21672), [Lacuna summary](https://lacuna.tiptreesystems.com/work/accelerating-dense-llms-via-l0-regularized-mixture-of-experts/wrk_8fe7eb20b0624464744a992ddfc9a1c7)

Activation memoization across tokens and requests
- LoRA-Drop (Jan 2026): consecutive-token hidden states have cosine similarity about 0.6 to 0.85 in all evaluated models; on most steps a fixed subset of intermediate layers reuses the previous token's hidden state plus a low-rank correction, with periodic full refresh steps. Up to 2.6x faster decoding, 45% to 55% KV-cache reduction, within 0.5 percentage points of baseline on GSM8K, MATH, BBH, HumanEval, MBPP, LongBench, XNLI, XCOPA; LLaMA2-7B, LLaMA3-8B, Qwen2.5-7B, Qwen2.5-14B — [arXiv 2601.02569](https://arxiv.org/abs/2601.02569), [PDF](https://arxiv.org/pdf/2601.02569)
- GateSkip (ICLR 2026) learns gates for token-wise skipping of attention and MLP branches, carrying hidden state and KV forward for skipped tokens — [ICLR 2026 paper](https://proceedings.iclr.cc/paper_files/paper/2026/file/7e3491e922bfd199ea34ecafeb7380f0-Paper-Conference.pdf)
- Cross-request reuse exists only at the KV level: RelayCaching reuses decoding KV caches across agents with more than 80% reuse rates, training-free — [arXiv 2603.13289](https://arxiv.org/pdf/2603.13289)

### Inferences
- Memory layers are the only retrieval family that genuinely reduces bytes read per token while adding capacity (a few rows of a huge table instead of full FFN matrices), but every positive result is from pretraining; the only retrofit evidence is a 0.5B model with recovery training. For "any checkpoint plus light adaptation" they are not usable yet.
- On a 12 GB card the practical decoupling knob from this family is dynamic sparsity. Its marginal value collapses once weights are already 4-bit: SpAx's per-token bytes go 4.18 GB to 1.76 GB (2.4x) for Q4_K_M, but its speedup over TEAL is only 1.07x, and GSM8K loses 6.3 points at the operating point. Reasoning and math degrade first.
- LoRA-Drop is the largest post-hoc single-stream number in this family (2.6x) and is orthogonal to quantization, but whether its low-rank corrections need training is not stated in the abstract; it also interacts with speculative drafting because skipped layers do not update KV.
- kNN-LM and RETRO-style methods are accuracy tools, not speed tools, for an existing checkpoint.

### Gaps
- No published post-hoc retrofit of memory layers or UltraMem into a 7B+ dense model with measured decode speed and accuracy.
- No 2025 to 2026 GPU result for LSH-based FFN neuron selection at LLM scale; LookupFFN and UltraFastBERT remain BERT-scale CPU results.
- No published "semantic cache of layer outputs across requests" (memoizing FFN outputs keyed by hidden-state similarity) for LLM decode; only KV reuse.
- Exact LoRA-Drop training requirement and hardware were not in the abstract.

## 2. Approximate matrix multiplication by lookup (MADDNESS, LUT-NN, PQ matmul, hashing-based GEMV, weight generation from seeds). Bytes read per output versus a 2-bit dense GEMV.

### Takeaway
Product-quantization lookup matmul trades multiplies for table reads, but the table for a fixed weight matrix is at least as large as a 2-bit weight matrix unless activations are encoded at well under 0.25 bits per dimension, and no LLM-scale accuracy result exists. The lookup direction that does reduce bytes is weight-side vector quantization (codebooks, AQLM-style) plus accelerator co-design (EVA), and the inverse idea, generating weights from pseudo-random seeds (SeedLM), trades bytes for compute but has only FPGA measurements.

### Cited Findings
- MADDNESS learns hash functions (decision trees) to encode activation subvectors into one of 16 prototypes per codebook and sums precomputed 8-bit dot-product tables; reports up to 100x faster than exact matmul and 10x faster than prior approximate methods on its benchmarks; table storage is K*C*M = 16*C*M bytes for C codebooks and M outputs — [Blalock and Guttag, ICML 2021](https://proceedings.mlr.press/v139/blalock21a/blalock21a.pdf), [search summary of table storage](http://proceedings.mlr.press/v139/blalock21a/blalock21a-supp.pdf)
- "Look-ups are not (yet) all you need for deep learning inference" argues MADDNESS-style lookup does not yet deliver usable accuracy when substituted into deep networks — [arXiv 2207.05808](https://arxiv.org/pdf/2207.05808)
- Stella Nera is a differentiable MADDNESS hardware accelerator (halutmatmul) — [arXiv 2311.10207](https://arxiv.org/pdf/2311.10207), [GitHub](https://github.com/joennlae/halutmatmul)
- A GPU MADDNESS library uses warp-wide data sharing to avoid random lookups, reporting about 103x average speedup on matrix-multiply benchmarks, not end-to-end LLM inference — [IEEE 2024](https://ieeexplore.ieee.org/document/10475989/)
- LUT-NN (MobiCom 2023) applies learned table lookup to whole-network inference (CNN and BERT class) — [arXiv 2302.03213](https://arxiv.org/abs/2302.03213)
- T-MAC uses table lookup to execute low-bit weight GEMV on CPUs; it still reads every weight's bits, lookup only replaces dequantize-multiply — [arXiv 2407.00088](https://arxiv.org/html/2407.00088v1)
- EVA (May 2026) targets VQ-compressed LLM decode: computes input-times-codebook dot products directly (turning GEMV into GEMM) and does structured lookups from an output buffer to avoid bank conflicts; up to 11.17x speedup and 7.17x energy efficiency over the prior best lookup-based architecture; hardware and software co-design, "2-bit-level" VQ weights — [arXiv 2605.24144](https://arxiv.org/abs/2605.24144)
- EVA's related work lists GPU and accelerator lookup GEMM designs: VQ-LLM (optimizes hot codebook entries on GPU), CodeGEMM, LUT Tensor Core, FIGLUT, Platinum — [arXiv 2605.24144 PDF](https://arxiv.org/pdf/2605.24144)
- SeedLM (Apple, ICLR 2025) stores each weight block as an LFSR seed plus a few coefficients and regenerates the block at inference, trading compute for memory traffic; data-free post-training; retains zero-shot accuracy at 4 and 3 bits on Llama 3 70B; FPGA tests show 4-bit SeedLM approaching 4x speedup over FP16 as model size grows to 70B — [arXiv 2410.10714](https://arxiv.org/abs/2410.10714), [Apple ML Research](https://machinelearning.apple.com/research/seedlm-compressing)
- 2026 seed-based follow-ups: S-Quant adapts the number of generated bases per block; Seed-Q allocates seed budget by weight sensitivity within an LFSR framework — [arXiv 2609.38477](https://arxiv.org/html/2609.38477)

### Inferences
- Bytes per output element, weight matrix D_in x D_out (my arithmetic from the MADDNESS table formula):
  - 2-bit dense GEMV: D_in/4 bytes per output (0.25 bytes per weight). For D_in = 4096 that is 1024 bytes per output.
  - PQ lookup with C codebooks, 16 prototypes, 8-bit table entries: 16*C bytes per output. Matching 2-bit dense needs C <= D_in/64, i.e. 64-dimensional activation subvectors encoded into 4 bits (0.0625 bits per activation dimension). With 4-bit table entries, C <= D_in/32 (32-dim subvectors, 0.125 bits per dimension). The accuracy published for MADDNESS uses far shorter subvectors, so at equal bytes the activation encoding is extremely lossy.
  - Conclusion: activation-side lookup AMM cannot beat a 2-bit weight stream on bytes for LLM-width layers; the table is a re-encoded weight matrix. It wins only on compute-bound hardware (CPU, MCU) where multiplies dominate, which is the opposite of the 3080 Ti decode regime.
- Weight-side VQ (codebook indices) is the lookup form that reduces bytes, and that is the quantization front: 2-bit-level VQ plus fast codebook GEMV. EVA's 11x is over a prior lookup architecture on custom hardware, not over a GPU 2-bit GEMV.
- SeedLM-style procedural weights are the only "generate instead of read" idea with LLM-scale accuracy; at 3 to 4 bits equivalent it is no better in bytes than existing 2 to 4-bit quants, and its GPU cost (LFSR generation plus a small projection per block) has no published GPU decode measurement.

### Gaps
- No published GPU result for activation-side PQ or MADDNESS matmul inside a 1B+ LLM with accuracy numbers.
- No GPU kernel measurement for SeedLM-style decode; only FPGA.
- EVA's hardware platform and baseline identities were not in the abstract.

## 3. Single-request semantic parallelism (Skeleton-of-Thought, APAR, PASTA, Multiverse, Hogwild!, Group Think, ParaThinker, APR, Parallel-R1, ThreadWeaver, 2026 work). Wall-clock speedups, quality, training; which work training-free?

### Takeaway
Measured wall-clock gains are 1.1x to 3.6x, concentrated on math and structured reasoning: Hogwild! is training-free and reached 3.1x to 3.6x aggregate tokens per second with 4 workers on QwQ-32B (forward latency only up 17% to 22%), but solution-time speedup depends on non-redundant collaboration; trained methods (Multiverse, PASTA, ThreadWeaver, Parason) report 1.14x to 2x with accuracy parity or small losses. Open-ended and code tasks are thinly measured, and SoT-style training-free decomposition fails on harder reasoning problems.

### Cited Findings
Training-free
- Skeleton-of-Thought (ICLR 2024): prompts the model for a skeleton then expands points in parallel; up to 2.39x speedup across 11 LLMs (v1 abstract), and can improve quality on several question categories — [arXiv 2307.15337v1](https://arxiv.org/abs/2307.15337v1), [v3](https://arxiv.org/abs/2307.15337)
- Hogwild! Inference (Yandex, HSE, IST Austria; NeurIPS 2025): multiple instances of the same LLM share one KV cache via RoPE position shifting, see each other's tokens immediately, and decide how to collaborate by prompting; works "out of the box" on reasoning models without fine-tuning — [arXiv 2504.06261](https://arxiv.org/abs/2504.06261), [NeurIPS poster](https://neurips.cc/virtual/2025/poster/115744)
  - Timing (L40S, QwQ-32B-AWQ): 1 worker 20.1 to 18.3 tok/s (1k to 16k context); 2 workers 36.3 to 34.3; 4 workers 68.9 to 60.3. Per-forward latency 49.7 to 54.5 ms (1 worker), 55.1 to 58.3 (2), 58.1 to 66.4 (4). Authors: 2 workers nearly 2x, 4 workers 3.2x to 3.6x — [arXiv 2504.06261v4 HTML](https://arxiv.org/html/2504.06261v4)
  - Attention overhead: at 16k KV tokens, 100 us (FlashAttention) versus 140 us (2 workers) and 203 us (4 workers) — [same](https://arxiv.org/html/2504.06261v4)
  - Accuracy is reported as accuracy versus number of sequential forward passes (LIMO, LiveCodeBench v5, OlympiadBench, AIME'25) in figures; models QwQ-32B, Qwen3 1.7B to 235B-A22B, Phi-4-Reasoning-Plus, DeepSeek-R1. Failure modes: Qwen3-1.7B and 4B get distracted, less robust at long contexts, workers sometimes miss redundant work; SoT could not solve the harder problems alone — [same](https://arxiv.org/html/2504.06261v4)
- Group Think (MediaTek, May 2025): one LLM acts as several concurrent thinkers with token-level visibility into each other's partial output; includes a modification letting existing open models (not trained for it) run Group Think on a local GPU, motivated by idle capacity at small batch on edge GPUs; evaluated on enumeration, divide-and-conquer, and programming tasks — [arXiv 2505.11107](https://arxiv.org/html/2505.11107v1)
- Parallel decoding of parallelizable reasoning branches within one sequence via a tree-like attention mask reports up to nearly 100% decoding speedup while basically maintaining answer quality — [Yu, arXiv 2503.20533](https://arxiv.org/abs/2503.20533)

Trained
- APAR: instruction-tuning on hierarchically structured data; up to 2x speedup alone, up to 4x with speculative decoding; 20% to 70% throughput gain and 20% to 35% latency reduction in high-throughput serving — [arXiv 2401.06761](https://arxiv.org/abs/2401.06761)
- PASTA (MIT/Google, 2025): model learns to emit its own async annotations; on 805 AlpacaEval prompts, geometric-mean speedups 1.21x to 1.93x with length-controlled win-rate changes of +2.2% to -7.1% versus sequential (arithmetic-mean equivalent 1.57x to 2.6x); Pareto frontier still improving after two rounds of preference optimization — [arXiv 2502.11517](https://arxiv.org/html/2502.11517v2)
- Multiverse (CMU and NVIDIA, NeurIPS 2025): MapReduce-style Map, Process, Reduce stages internalized by the model; Multiverse-32B trained by SFT on 1K examples; AIME24 54%, AIME25 46%; up to 2x wall-clock speedup per generated token across batch sizes; one summary reports average parallelism degree about 1.15 to 1.18 on AIME — [arXiv 2506.09991](https://arxiv.org/pdf/2506.09991), [NeurIPS poster](https://neurips.cc/virtual/2025/poster/115810)
- ParaThinker (Tsinghua, Sep 2025): native parallel paths then a summary stage reusing KV caches; with 8 paths, +12.3% (1.5B) and +7.5% (7B) average accuracy over sequential, +4.3% and +2.0% over majority voting, with 7.1% latency overhead; this is accuracy-at-equal-latency, not speedup — [arXiv 2509.04475](https://arxiv.org/html/2509.04475v1)
- Adaptive Parallel Reasoning (Berkeley, 2025): spawn() and join() child threads, SFT on traces then end-to-end RL; Countdown 83.4% versus 60.0% at 4k context, 80.1% versus 66.6% at 20k total tokens, 75.2% versus 57.3% at about 5000 ms latency — [arXiv 2504.15466](https://arxiv.org/abs/2504.15466)
- Parallel-R1 (Sep 2025): SFT on easy-task parallel trajectories then RL on hard problems; +8.4% over a sequential RL model, +42.9% on AIME25 when used as a mid-training exploration scaffold; no wall-clock figures in the abstract — [arXiv 2509.07980](https://arxiv.org/abs/2509.07980)
- ThreadWeaver (ICML 2026 oral), on Qwen3-8B: 71.9% average across six math benchmarks versus 72.2% sequential, 79.9% AIME24; token-latency speedup up to 1.53x average and 3.56x max, critical path 15.1k to 13.2k tokens; measured wall-clock on 50 MATH problems 162.34 s to 142.21 s (1.14x); no acceleration for inherently sequential tasks — [arXiv 2512.07843](https://arxiv.org/pdf/2512.07843), [ICML 2026](https://icml.cc/virtual/2026/oral/71140)
- Parason (Aug 2026): distinguishes subtask parallelism from trial parallelism (parallel speculative attempts), the latter being 65.5% of parallelizable steps in DeepSeek-V4 traces on HLE; Parallelism-Aware GRPO; about 1.7x average wall-clock acceleration on AIME24 and AIME25 with competitive accuracy — [arXiv 2608.24658](https://arxiv.org/abs/2608.24658)
- Divide-and-Conquer CoT (Stanford, Jan 2026): director plus spawned workers, 1.5B long-CoT model, SFT then RL, optimizes longest-path length as latency proxy — [arXiv 2601.23027](https://www.alphaxiv.org/abs/2601.23027)
- Parallel-Probe (Feb 2026): online control of parallel thinking cuts sequential tokens more than 30% versus self-consistency — [arXiv 2602.03845](https://arxiv.org/html/2602.03845v2)
- Message-passing alternative to fork-join (Jul 2026) keeps worker contexts persistent across iterations; reports nearly all wall-clock inside vLLM — [arXiv 2607.01077](https://arxiv.org/html/2607.01077)
- Agentic setting: reasoning in parallel with tool actions cut task time 13.4% and median per-task latency 10.9% — [arXiv 2608.13667](https://arxiv.org/html/2608.13667v1)

### Inferences
- Hogwild!'s timing table gives a direct consumer-relevant calibration: 4 rows cost 1.17x to 1.22x a single-row step on an L40S with a 4-bit 32B model, consistent with the target's 8 to 16 rows costing 1.4 to 1.9 steps. On a 3080 Ti, 4 to 8 collaborating workers would deliver 3x to 5x aggregate tokens per second; the realized speedup in time-to-answer equals aggregate throughput times the fraction of non-redundant work, which the papers only report through accuracy-versus-sequential-steps curves.
- Training-free options: SoT (prompting; could not solve harder reasoning problems alone per the Hogwild! evaluation), Hogwild! (reasoning models only, fails on small models), Group Think's local-GPU variant, and Yu's in-sequence branching. Everything with 1.5x+ and accuracy parity on math (Multiverse, ThreadWeaver, Parason, APR) needed SFT and usually RL on the specific model.
- For open-ended chat the only quantified result is PASTA on AlpacaEval (1.21x to 1.93x geometric mean with up to 7.1% win-rate loss). For code, Hogwild! reports LiveCodeBench Pass@1 improvements with 2 workers but not a latency figure for code.
- These methods are architecture-agnostic at the kernel level (they only need batched attention with custom masks or position offsets), but their benefit is task-dependent: zero on sequential short answers, largest on long reasoning traces, which is also where llama.cpp spec drafting is weakest at temperature above 0.

### Gaps
- Wall-clock speedups for Group Think, Parallel-R1, and Multiverse at batch 1 on a consumer GPU were not found.
- No paper measures semantic parallelism stacked with speculative decoding plus quantization at batch 1 on consumer hardware (APAR's 4x with spec is the only combination figure, from 2024).
- Quality impact on open-ended generation beyond AlpacaEval, and on code latency, is largely unmeasured.

## 4. Superposed decoding and multi-stream single-pass generation (Superposed Decoding, Mixture of Inputs, successors)

### Takeaway
Superposed Decoding gives k alternative drafts for the price of one pass (at least 2.44x faster for k >= 3) but produces short alternatives, not one long answer faster; Mixture of Inputs is a training-free quality method with no speed benefit; the 2026 successor in spirit, Multi-Stream LLMs, trains models to emit several role streams per forward pass and cuts reasoning delay by up to about 47% (MATH500) with flat or better accuracy.

### Cited Findings
- Superposed Decoding (NeurIPS 2024) feeds a weighted superposition of the k most recent token embeddings through one forward pass and resolves k drafts with n-gram filtering; drafts at least as coherent and factual as nucleus and greedy decoding while at least 2.44x faster for k >= 3; training-free — [arXiv 2405.18400](https://www.emergentmind.com/papers/2405.18400), [GitHub](https://github.com/RAIVNLab/SuperposedDecoding), [NeurIPS 2024](https://mlanthology.org/neurips/2024/shen2024neurips-superposed/)
- Mixture of Inputs ("Text Generation Beyond Discrete Token Sampling", 2025): replaces the one-hot next input with a Bayesian blend of the sampled token and the discarded distribution; training-free; improves text quality, math, code, PhD-level QA on QwQ-32B, Nemotron-Super-49B, Gemma-3-27B, DAPO-Qwen-32B with negligible overhead; no speedup claim — [arXiv 2505.14827](https://arxiv.org/abs/2505.14827)
- Multi-Stream LLMs (Su, Yang, Li, Geiping; May 2026): each forward pass reads all input streams and emits one token per output stream, with per-stream RoPE counters, learned stream embeddings, and a cross-stream causal mask; trained by instruction-tuning on multi-stream data (not training-free) — [arXiv 2605.12460](https://arxiv.org/html/2605.12460v1)
  - Delay versus vanilla (units not given in the excerpt) and accuracy: Qwen3-1.7B MATH500 43.14 to 22.94 delay, 48.2% to 51.6%; GSM8K 14.93 to 11.29, 90.6% to 89.5%; Qwen3-4B MATH500 45.82 to 26.51, 60.8% to 64.0%; SQuAD 9.68 to 5.48, 75.1% to 74.7% — [same](https://arxiv.org/html/2605.12460v1)
- CALM (Tencent WeChat AI, Oct 2025) compresses K tokens into one continuous vector (over 99.9% reconstruction) and predicts the next vector with a likelihood-free head, reducing generative steps K-fold; requires training a new model — [arXiv 2510.27688](https://arxiv.org/html/2510.27688v1)

### Inferences
- Superposition exploits the same "spare rows are nearly free" property as batching, but packs streams into one row. It is useful for autocomplete-style multiple suggestions; for a single long answer it offers nothing, because each draft is a separate continuation.
- Multi-stream generation is a general architectural pattern (only masks, position IDs, a stream embedding), applicable to any RoPE transformer after fine-tuning; its decode benefit comes from overlapping thinking with answering, i.e. it shortens the critical path rather than reducing bytes per token.

### Gaps
- No 2025 to 2026 direct successor to Superposed Decoding surfaced in search.
- No work combines superposed inputs with speculative verification to get one long answer faster.
- Multi-Stream LLMs' hardware, units, and training cost were not in the excerpt read.

## 5. Layer-parallel or staggered single-stream execution (StagFormer, layer parallelism, speculation across depth, PipeDec, hidden-state prediction)

### Takeaway
Running layer L+k before layer L finishes for the same token has three published forms: architectures trained for it (StagFormer, Ladder Residual, pipeline-native transformers), post-hoc layer pairing or FFN fusion (1.2x to 1.7x, mostly multi-device or very large models), and depth-wise pipelining combined with speculation (PipeDec, Speculative Pipeline Decoding, AdaDecode, WaveFront Decoding). On a single GPU, depth parallelism does not reduce bytes; the only single-device variant with large gains is WaveFront Decoding on looped models (2.4x to 4.8x, training-free) because weight sharing turns mixed-depth states into one batched call.

### Cited Findings
- StagFormer (Google; arXiv 2501.15665v2, revised Aug 2025): upper stack attends to lower-stack outputs only up to the previous time step, so stacks can run concurrently; trained from scratch on the Pile (about 300B to 327B tokens); 2 x 18-layer stacks (2.9B) reach Pile perplexity 3.756 and average 47.0 versus 3.780 and 45.0 for a 36-layer 2.8B baseline; speedup is simulated on twice the chips ignoring communication; shared-weight 18L variant 3.896 perplexity, 44.8 average; p=3 and p=4 stacks degrade (average 37.8 and 41.2); no method to convert a pretrained model — [arXiv 2501.15665v2](https://arxiv.org/html/2501.15665v2)
- Layer Parallelism ("Leveraging the true depth of LLMs", Feb 2025): runs consecutive layer pairs in parallel without retraining; Llama 3.2 3B effective depth reduced 18% for up to 1.22x speedup, 1.38x at aggressive settings; implemented through tensor parallelism — [arXiv 2502.02790](https://arxiv.org/pdf/2502.02790), [MarkTechPost](https://www.marktechpost.com/2025/02/14/layer-parallelism-enhancing-llm-inference-efficiency-through-parallel-execution-of-transformer-layers/)
- AdaDecode (ICML 2025): emits tokens from intermediate layers when confident and runs deferred layers in parallel with the next token, with output consistency and no changes to original parameters; up to 1.73x — [arXiv 2506.03700](https://arxiv.org/abs/2506.03700)
- FFN Fusion (NVIDIA, Mar 2025): sequences of FFN layers left after attention removal can be fused and run in parallel; Llama-3.1-405B to Llama-Nemotron-Ultra-253B-Base with 1.71x inference latency speedup and 35x lower per-token cost; effectiveness grows with scale (49B to 253B tested) — [arXiv 2503.18908](https://arxiv.org/abs/2503.18908)
- Ladder Residual (Together, 2025): reroutes residuals so communication overlaps compute; about 29% end-to-end speedup for a 70B model with 8-way tensor parallelism; parts of Llama-3.1-8B converted with about 3B tokens of retraining at minimal accuracy loss — [arXiv 2501.06589](https://arxiv.org/abs/2501.06589)
- PipeDec (Apr 2025): 14-stage pipeline serving Llama-3.1-70B with a Llama-3.2-1B drafter and a dynamic prediction tree so all stages work on one request; 4.46x to 7.79x over plain pipeline parallelism, 2.2x to 2.69x over tree speculative decoding (average 6.17x and 2.53x); multi-GPU — [arXiv 2504.04104](https://arxiv.org/html/2504.04104v1), [Emergent Mind](https://www.emergentmind.com/papers/2504.04104)
- Speculative Pipeline Decoding (May 2026): splits the target into n pipeline stages so n tokens of one sequence are in flight at different depths; a draft module fuses features from several depths to predict one token per cycle, hiding draft latency; beats EAGLE-3 in theoretical and wall-clock speedup at moderate width; deliberately avoids multi-token drafting to prevent compounding feature drift — [arXiv 2605.30852](https://arxiv.org/abs/2605.30852), [HTML](https://arxiv.org/html/2605.30852)
- SPEED (2023) speculatively runs future tokens using early-layer predictions and notes that in a standard decoder this does not reduce memory traffic — [arXiv 2310.12072](https://arxiv.org/html/2310.12072)
- Hidden Transfer (2024) predicts pseudo hidden states of future tokens at an intermediate layer with a trained linear map, then runs the remaining layers to draft several tokens per pass — [arXiv 2404.12022](https://arxiv.org/pdf/2404.12022)
- WaveFront Decoding (Ha and Kim, Sep 2026): for looped LMs, arranges tokens at different recursion depths on a diagonal wavefront and processes them in one batched recurrent-block call, drafting new positions at shallow depth while deepening earlier ones to full-depth verification; training-free; across six Spec-Bench categories 2.42x on Ouro-2.6B, 3.54x on Huginn-3.5B, 4.81x on Huginn-3.5B with cross-recurrence KV sharing; consistently above draft-then-verify; its analysis bounds speedup below T/T_d because prelude and coda layers are paid per committed token — [arXiv 2609.23033](https://arxiv.org/abs/2609.23033), [search snippet with bound](https://arxiv.org/pdf/2609.23033)
- Pipeline-Native Transformers (Aug 2026, CPU): architectures whose layer dependencies allow a stage-major schedule, weights stored as L2-sized tiles in consumption order; best candidate halves critical-path weight bandwidth (9.00 to 4.50 MB per token) within 0.24 perplexity on TinyStories; a 30.9B pipeline-native MoE decodes at 5.94 tok/s on a 32-vCPU Ice Lake versus 4.75 tok/s for llama.cpp on a comparable dense model — [arXiv 2608.23841](https://arxiv.org/abs/2608.23841)

### Inferences
- On one GPU, two layers running concurrently still read both layers' weights; the gain is latency hiding (sync and launch bubbles, which megakernels already remove) rather than fewer bytes. Expect at most the 1.2x to 1.4x range seen in Layer Parallelism and AdaDecode for single-device use.
- The structurally interesting result is WaveFront Decoding: weight sharing converts depth into batch, so one weight read serves several tokens at several depths. For an arbitrary checkpoint this requires first converting it to a looped model (Section 8), which no published work has paired with WFD.
- Pipeline-native layouts with L2-sized tiles are a CPU result; the 3080 Ti's 6 MB L2 cannot hold a meaningful tile schedule for a 7B model (see Section 6).

### Gaps
- No post-hoc conversion of a pretrained dense checkpoint into a StagFormer-style staggered model.
- Speculative Pipeline Decoding numbers, models, and whether it applies on a single GPU were not in the abstract.
- WFD hardware and quality numbers were not in the abstract (lossless verification is claimed via full-depth correction).

## 6. Weight residency and streaming (persistent megakernels, host-to-device streaming, L2-resident shards)

### Takeaway
Persistent megakernels raise achieved batch-1 bandwidth from roughly 50% (llama.cpp, vLLM, SGLang) to 74% to 86% on H100 and a self-reported 94% of achievable bandwidth on an RTX 5080, worth 1.5x to 2.1x over llama.cpp on sub-1B models on consumer RTX cards but only 1.03x to 1.3x at serving level for 4B to 13B models; host-to-device streaming over PCIe is about 30x slower than VRAM and only helps when combined with aggressive sparsity (SpAx: 2.5 tok/s for fully offloaded BF16 Llama-3.1-8B); L2 residency is irrelevant on a 6 MB-L2 card.

### Cited Findings
Megakernels
- Hazy Research "No Bubbles" (May 2025), Llama-3.2-1B BF16, batch 1, no speculation: under 1 ms per forward on H100 (78% of memory bandwidth versus at most 50% for vLLM and SGLang), almost 2.5x faster than vLLM and over 1.5x faster than SGLang; on B200 under 680 us and over 3.5x faster than vLLM. B200 600 us breakdown: about 250 us storing, syncing and loading activations; 200 us RMSNorm and matvec; 30 us waiting on weights; 40 us warp sync; 80 us setup. Sources of bubbles: about 100 kernels per forward, stragglers, launch cost about 1.3 to 2.1 us — [Hazy Research blog](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles)
- Mirage Persistent Kernel (OSDI 2026): compiler builds an SM-level task graph and runs it in one persistent kernel with decentralized scheduling; up to 1.7x lower end-to-end latency than kernel-per-operator serving; A100, H100, B200; five models — [arXiv 2512.22219](https://arxiv.org/pdf/2512.22219), [OSDI 2026](https://www.usenix.org/conference/osdi26/presentation/cheng)
- ForgeMegakernel (Tsinghua, ModelBest; Sep 2026), agent-generated megakernels, H100, BF16 only — [arXiv 2609.12379](https://arxiv.org/html/2609.12379v1)
  - 14 cells, 0.6B to 13B (Llama, Qwen3, Qwen2.5, MiniCPM, NanBeige): 50.5% to 85.9% MBU versus SGLang 36.5% to 78.9% and MPK 20.6% to 75.9%; kernel-level 1.21x geometric mean over SGLang, 1.54x over MPK
  - Llama-3.2-1B: 74.0% MBU, 1.84x over llama.cpp, 1.50x over vLLM
  - One Qwen3-4B example: llama.cpp 4768 us/token (50.5% MBU), SGLang 3932 us (61.2%), megakernel 3781 us (63.7%)
  - Serving-level inside SGLang: 1.03x to 1.28x, geometric mean 1.11x; Qwen3-8B batch 1 5601 versus 6009 us/token
  - Where time goes in non-megakernel stacks: 280 to 448 launches per token (301 to 653 us), small zero-weight kernels (norm, RoPE, KV store) 17% to 21% of step time, intermediate DRAM writes 7.7 MB per token versus 1.4 MB
  - Cost: median 14.7 agent-hours and about 177 dollars per megakernel
- Consumer cards (self-reported READMEs): Luce megakernel on RTX 3090, Qwen3.5-0.8B 413 tok/s versus llama.cpp 267 tok/s (1.55x), 250 tok/s on an RTX 2080 Ti — [lucebox-hub](https://github.com/Luce-Org/lucebox-hub); SamuelReeder megakernels on RTX 5080, Qwen3.5-0.8B INT8 955 tok/s (2.1x llama.cpp), 1205 tok/s with MTP speculation, sustaining 94% of achievable bandwidth — [GitHub](https://github.com/SamuelReeder/megakernels)
- AutoMegaKernel (Jun 2026): INT8 megakernel 1.19x to 1.23x over cuBLAS BF16 at batch-1 decode on an RTX 5090 (laptop) — [HF paper page](https://huggingface.co/papers/2606.09682.md)
- Fleet extends megakernel task abstractions to multi-die GPUs — [arXiv 2604.15379](https://arxiv.org/pdf/2604.15379)
- Community summary of the megakernel debate — [Latent Space](https://latent.space/p/ainews-megakernels-are-so-dead-and/)

Host-to-device streaming
- PCIe 4.0 x16 tops out at 31.5 GB/s per direction; spilling weights to system RAM costs 10x to 50x throughput (example: 14B 4-bit at about 94 tok/s ceiling in VRAM versus about 5 tok/s spilled) — [DEV Community](https://dev.to/axrisi/vram-for-local-llms-why-memory-bandwidth-sets-your-tokens-per-second-h4h)
- SpAx (RTX 4090, PCIe 4.0 x8, all weights offloaded, Llama-3.1-8B): derived from per-token totals, CPU-memory BF16 dense about 0.87 tok/s, TEAL about 1.47, SpAx about 2.52; flash BF16 0.40, 0.65, 0.97 — [arXiv 2610.02598](https://arxiv.org/html/2610.02598)
- LLM in a Flash (Apple, ACL 2024): windowing and row-column bundling with sparsity prediction let models up to 2x DRAM size run with 4x to 5x (CPU) and 20x to 25x (GPU) speedup over naive loading — [arXiv 2312.11514](https://arxiv.org/abs/2312.11514)
- 2026 self-published engines claim PCIe-overlapped MoE streaming on gaming PCs (FreeToken: 39.3 tok/s on an 8 GB RTX 4060 laptop at a measured 52.7 GB/s PCIe ceiling; Strata: 125B at 94 tok/s), not independently benchmarked — [FreeToken review](https://www.zhongzhuzhou.org/blog/2026-08-23-freetoken-technical-review-en/), [Strata vs llama.cpp](https://medium.com/data-science-in-your-pocket/strata-vs-llama-cpp-the-best-way-to-run-a-125b-llm-on-your-gaming-pc-7e4afe3c1760)

L2 residency
- GA102 (RTX 3080 Ti class) has 6 MB of L2 — [NVIDIA Ampere GA102 whitepaper](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf)
- L2-sized weight tiles in consumption order cut L1 read misses 7.29x on CPU in the pipeline-native design — [arXiv 2608.23841](https://arxiv.org/abs/2608.23841)

### Inferences
- Megakernel gains shrink with model size: 1.55x to 2.1x over llama.cpp at 0.8B (launch and sync overhead dominate small models) versus about 1.1x to 1.3x at 4B to 8B. For a 7B to 8B target on a 3080 Ti, a megakernel is worth roughly 1.2x to 1.4x over llama.cpp, an estimate scaled from the ForgeMegakernel Qwen3-4B and Qwen3-8B data.
- None of the published megakernels use 2 to 4-bit weights on Ampere consumer GPUs for 7B+ models; ForgeMegakernel is BF16 on H100, the consumer ones are 0.8B.
- Streaming arithmetic for the target: 912 GB/s versus about 25 GB/s practical PCIe 4.0 x16 is about 36x. Streamed bytes per token hide behind on-device reads only if they are at most about 1/36 of the on-device bytes per token. Host RAM therefore extends capacity (bigger models) only when per-token host bytes are tiny, which means sparsity or MoE-like routing on the streamed part. It is not a route to higher single-stream speed for a model that already fits.
- 6 MB of L2 holds about 0.13% of a 4-bit 8B model (about 4.5 GB); L2 residency can at most keep norms, small embeddings or LM-head slices hot. It matters only for activation traffic, which Hazy measured at about 40% of B200 step time (250 of 600 us) for a 1B model.

### Gaps
- No published bandwidth-fraction measurements for megakernels on an RTX 3080 Ti or any Ampere consumer card with 7B+ models.
- No peer-reviewed measurement of dense-model PCIe weight streaming with compute overlap at batch 1 on consumer GPUs beyond SpAx.
- MPK per-model, per-GPU numbers were not extracted.

## 7. Model surgery for width (post-hoc narrowing, width pruning with distillation, Minitron and successors)

### Takeaway
Width pruning plus distillation reliably halves parameters with a few to 7 MMLU points lost and needs 45B to 380B distillation tokens, far beyond "light" adaptation; training-free width removal (SliceGPT) tops out around 25% removal. Throughput gains are measured at large batch on H100, not single stream.

### Cited Findings
- Minitron (NVIDIA, NeurIPS 2024): pruning plus KD-only retraining derives smaller models with more than 40x less compute than training from scratch; width pruning consistently outperforms depth pruning at a fixed parameter budget; a shared linear map lifts student hidden states to teacher width during distillation — [arXiv 2408.11796](https://arxiv.org/pdf/2408.11796)
- Mistral-NeMo-Minitron 8B: hidden 5120 to 4096, MLP 14336 to 11520, distilled on 380B tokens; GSM8K 55.7 to 58.5 and HumanEval 23.8 to 36.2 (above the teacher on these two) — [NVIDIA blog](https://developer.nvidia.com/blog/mistral-nemo-minitron-8b-foundation-model-delivers-unparalleled-accuracy/)
- Llama-3.1-Minitron 4B: width variant hidden 4096 to 3072, MLP 14336 to 9216; depth variant drops 16 of 32 layers; teacher corrected on 94B tokens; MMLU 65.3 (8B) versus 58.7 (width) and 60.5 (depth) per the blog table (header garbled in source); TensorRT-LLM H100 throughput 1.8x (width) and 2.7x (depth) at batch 32 to 64 — [NVIDIA blog](https://developer.nvidia.com/blog/how-to-prune-and-distill-llama-3-1-8b-to-an-nvidia-llama-3-1-minitron-4b-model/)
- Puzzle (NVIDIA): Llama-3.1-Nemotron-51B from Llama-3.1-70B via blockwise local distillation, MIP architecture search, global KD; 2.17x H100 throughput, 98.4% of parent accuracy, 45B tokens (under 50B for the whole pipeline) — [arXiv 2411.19146](https://web3.arxiv.org/pdf/2411.19146), [Moonlight review](https://themoonlight.io/review/puzzle-distillation-based-nas-for-inference-optimized-llms)
- SliceGPT (ICLR 2024), training-free PCA-based deletion of rows and columns: up to 25% parameters removed for Llama-2-70B, OPT-66B, Phi-2 while keeping 99%, 99%, 90% of zero-shot performance; on 24 GB consumer GPUs total inference compute for Llama-2-70B drops to 64% of dense — [arXiv 2401.15024](https://arxiv.org/abs/2401.15024)
- 2026 applications and variants: Bielik-Minitron-7B applies the recipe to a Polish model — [arXiv 2603.11881](https://arxiv.org/pdf/2603.11881); LayerTrim trims MLP features of redundant layers in reasoning models training-free for 1.24x prefill and 19.7% lower peak memory on Qwen3-14B — [arXiv 2610.19121](https://arxiv.org/abs/2610.19121); SparseDecoding calibrates pruning on decode-step activations for up to 1.48x end-to-end decode on A100 with N:M kernels — [arXiv 2610.12327](https://arxiv.org/html/2610.12327)
- Structured pruning hurts reasoning under test-time scaling while unstructured pruning can help — [arXiv 2604.25098](https://arxiv.org/html/2604.25098v1)

### Inferences
- A 2x parameter cut by width gives at most 2x fewer bytes per token at batch 1 and costs about 7 MMLU points at 4B scale even after 94B tokens; that fails the "equal or better accuracy" requirement unless the pruned model is also given the parent's capability through heavier distillation (Puzzle's 98.4% at 1.37x fewer parameters used 45B tokens).
- Training-free width surgery (SliceGPT 25%) is about a 1.3x byte reduction, comparable to TEAL-style dynamic sparsity but static, and stacks multiplicatively with quantization.

### Gaps
- No single-stream (batch 1) latency measurements on consumer GPUs for Minitron or Puzzle models.
- No 2026 successor that reduces the distillation budget below about 10B tokens for 2x width reduction at parity was found.

## 8. Genuinely new decode paradigms in 2025 to 2026 (recurrent-depth conversion, SSM distillation, per-prompt weight compression or pruning, latent or multi-token generation)

### Takeaway
The highest published single-stream multipliers come from changing what one forward pass emits: TiDAR (4.71x to 5.91x tokens per second, 7.45 to 8.25 tokens per pass, trained), Apple's gated-LoRA multi-token prediction (about 5x code and math, 2.5x chat, light adaptation, frozen base), Uno (up to 3x lossless with frozen AR weights plus lightweight diffusion weights), and WaveFront Decoding on looped models (2.4x to 4.8x training-free). Per-prompt pruning is training-free but modest (GRIFFIN 1.25x to 1.29x); trained per-prompt selection reaches 41% lower decode time. SSM distillation reduces KV traffic, not weight bytes, so its large speedups are long-context and batch phenomena.

### Cited Findings
Recurrent depth and looped models
- Huginn-3.5B (Geiping et al., Feb 2025) is a depth-recurrent model with a prelude, a looped core block, and a coda, trained from scratch on about 800B tokens — [arXiv 2502.05171](https://arxiv.org/abs/2502.05171)
- Ouro (ByteDance Seed, Oct 2025): looped LMs of 1.4B and 2.6B pretrained on 7.7T tokens match 4B and 8B dense transformers on nearly all benchmarks (2x to 3x parameter efficiency) at 4 recurrent steps; better at knowledge manipulation, not knowledge storage — [arXiv 2510.25741](https://arxiv.org/html/2510.25741v2), [HF](https://huggingface.co/papers/2510.25741)
- Retrofitted recurrence (McLeish et al., NeurIPS 2025): converts TinyLlama-1.1B, OLMo-2-1B, Llama-3.2-1B into depth-recurrent models (early layers become prelude, later layers become recurrent block and coda, middle removed) with a depth curriculum and a healing phase; beats post-training the original non-recurrent model at equal compute on GSM8K and MATH; Muon reported to stabilize training — [arXiv 2511.07384](https://arxiv.org/html/2511.07384v1), [Emergent Mind summary](https://www.emergentmind.com/papers/2511.07384)
- Relaxed Recursive Transformers (Google, ICLR 2025): convert pretrained Gemma into layer-tied recursive models with per-layer LoRA; recursive Gemma 1B beats TinyLlama 1.1B and Pythia 1B and recovers most of Gemma 2B; continuous depth-wise batching with early exit offers a theoretical 2x to 3x throughput gain — [arXiv 2410.20672](https://arxiv.org/abs/2410.20672)
- WaveFront Decoding: training-free 2.42x (Ouro-2.6B), 3.54x and 4.81x (Huginn-3.5B) single-stream — [arXiv 2609.23033](https://arxiv.org/abs/2609.23033)

State-space or linear-attention distillation
- Mamba in the Llama (NeurIPS 2024): reuses attention projection weights to distill Llama into hybrid linear RNNs; with a quarter of attention layers kept it matches the original on chat benchmarks; adds a hardware-aware speculative decoding algorithm for hybrids — [arXiv 2408.15237](https://arxiv.org/abs/2408.15237v3)
- Jet-Nemotron (NVIDIA, NeurIPS 2025): PostNAS starts from a pretrained full-attention model, freezes MLP weights, and searches attention-layer placement and linear-attention blocks; Jet-Nemotron-2B reports up to 53.6x generation throughput and 6.1x prefill speedup while matching or beating Qwen3, Qwen2.5, Gemma3, Llama3.2 — [NeurIPS 2025](https://papers.nips.cc/paper_files/paper/2025/hash/4364fbe773a77fb373df5c7f1d4cbb53-Abstract-Conference.html), [NVIDIA project page](https://research.nvidia.com/labs/eai/publication/jetnemotron/)

Per-prompt or test-time weight specialization
- GRIFFIN (2024): training-free and calibration-free; FFN activations "flock" within a sequence, so the prompt selects FF neurons reused for all generated tokens; at 50% FF parameters 1.29x (Gemma 7B) and 1.25x (Llama 2 13B) on an NVIDIA L40 — [arXiv 2404.01365](https://arxiv.org/html/2404.01365v3)
- Instruction-Following Pruning (Apple, 2025): a trained predictor selects a prompt-specific parameter subset reused during decoding; 8B or 9B pruned to 3B activated, time-to-first-token down up to 57%, decoding time down up to 41% — [arXiv 2501.02086](https://arxiv.org/pdf/2501.02086)
- SlimWise (Sep 2026), MoE: prefill with the full model, decode with an expert-pruned model sharing the KV cache; up to 1.81x decode throughput at 50% expert pruning on Qwen3.6-35B-A3B — [arXiv 2609.34117](https://arxiv.org/abs/2609.34117)

Multiple tokens per forward pass (latent, mask-token, diffusion)
- Your LLM Knows the Future (Apple, Jul 2025): mask tokens appended to the input, gated LoRA (base weights frozen, behavior preserved for normal tokens), a lightweight sampler, and speculative verification; nearly 5x faster on code and math, almost 2.5x on chat and knowledge, without quality loss — [arXiv 2507.11851](https://arxiv.org/pdf/2507.11851)
- ESP (Mar 2026): training-free multi-token prediction by probing with mask tokens drawn from embedding space, building a pruned speculative tree, verified losslessly; acceptance length +7% to 11% over Lookahead on LLaMA3, +7% to 8% on Qwen3; throughput up to 15% to 19% above the strongest baseline — [arXiv 2603.17942](https://arxiv.org/abs/2603.17942)
- TiDAR (NVIDIA, Nov 2025): one forward pass drafts by diffusion and verifies autoregressively using spare "free token slots"; 1.5B lossless versus its AR counterpart at 4.71x tokens per second and 7.45 tokens per network evaluation; 8B 5.91x and 8.25 tokens per evaluation with minimal loss; requires training the hybrid model — [arXiv 2511.08923](https://arxiv.org/html/2511.08923v1)
- Uno (Sep 2026): attaches lightweight diffusion weights to a frozen AR model via a short diffusion-distillation phase and verifies drafts with speculative rejection sampling (lossless); up to 3x over the base AR model, including at the largest batch the device supports; 8B Uno beats 26B DiffusionGemma and Mercury 2 on agentic, coding, long-context benchmarks; secondary report of about 3000x fewer training tokens than AR pretraining and of speedup sensitivity to temperature — [arXiv 2609.04010](https://arxiv.org/abs/2609.04010), [Pith review](https://pith.science/paper/2609.04010), [Mervin Praison summary](https://mer.vin/news/uno-diffusion-augmented-llms-deliver-lossless-3x-inference-speedups/)
- DiffusionGemma (Aug 2026): diffusion variant of Gemma 4 26B-A4B refining 256-token blocks in about 12 forward passes (about 20 tokens per pass), about 1500 tok/s on one H100 — [arXiv 2608.00146](https://arxiv.org/html/2608.00146v1)
- ZAYA1-8B-Diffusion-Preview (Zyphra): AR-converted diffusion model, 4.6x speedup with a lossless sampler and 7.7x with a non-lossless logit-mixing sampler — [Zyphra](https://www.zyphra.com/our-work/zaya1-8b-diffusion-preview)
- OPDLM (Jun 2026): on-policy distillation to convert AR models to diffusion with far fewer tokens — [arXiv 2606.06712](https://arxiv.org/pdf/2606.06712)
- Diffusion LMs push attention into the compute-bound regime, breaking the memory-bound assumption — [FOCUS, arXiv 2601.23278](https://arxiv.org/pdf/2601.23278)
- PRISM (Feb 2026) disaggregates each drafting step across different parameter sets so drafter capacity grows at flat per-step cost; more than 2.6x decode throughput over an optimized engine — [arXiv 2602.01762](https://arxiv.org/pdf/2602.01762)
- Energy per output token is predicted by active weight bytes plus KV traffic, not parameter count — [Token-Energy Law, arXiv 2610.28576](https://arxiv.org/html/2610.28576)

### Inferences
- Multi-token-per-pass methods are the only family whose published single-stream multipliers (3x to 6x versus plain AR) approach the 5x target by themselves, but they are measured against plain AR on server GPUs. Against llama.cpp with speculative drafting (already 1.5x to 2.5x over AR on favorable text, a typical range rather than a cited figure), the residual multiplier is about 1.5x to 3x.
- Spare token slots are finite. On the target, 8 to 16 rows cost 1.4 to 1.9 steps (consumer's measurement), so a pass that verifies about 8 tokens costs about 1.4 steps; TiDAR-level acceptance (7 to 8 tokens per pass) would give about 5x over AR at batch 1. Heavier quantization lowers bytes per step and therefore lowers the number of free slots before compute binds (fewer bytes per FLOP), so 2-bit weights plus wide verification can turn compute-bound sooner on a GeForce card.
- Uno and Apple's gated LoRA are the two approaches that keep the original weights frozen and add light trainable parameters, which matches "any checkpoint plus light post-hoc adaptation". TiDAR, DiffusionGemma, ZAYA require substantial training.
- Recurrent-depth conversion reduces unique weight bytes (shared block) and, with WaveFront Decoding, turns depth into batch; the published conversions are 1B scale and need continued pretraining, so a general 7B+ "checkpoint to looped plus WFD" pipeline does not exist yet.
- SSM and linear-attention distillation keep MLP weights (Jet-Nemotron freezes them), so weight bytes per token at short context are unchanged; the 53.6x figure is a KV-bound, long-context throughput result, not a single-stream short-context one.
- GRIFFIN shows that per-prompt neuron selection is training-free on gated FFNs but caps at about 1.25x to 1.3x for 50% FF removal; Apple's trained version reaches about 1.7x decode (41% lower time).

### Gaps
- No published consumer-GPU, batch-1, quantized-weight evaluation of TiDAR, Uno, or gated-LoRA MTP.
- Retrofitted-recurrence token budgets and any 7B+ conversion were not found.
- No 2026 paper proposing per-prompt weight compression (re-quantizing or re-coding weights conditioned on the prompt) was found.

## 9. Cross-direction summary: numbers, training cost, generality

### Takeaway
No single published direction reaches 5x over llama.cpp best config at equal accuracy for arbitrary checkpoints. Reaching it requires stacking roughly independent factors: multi-token-per-pass with frozen-base adapters (about 1.5x to 3x over spec-drafted llama.cpp), megakernel efficiency (about 1.2x to 1.4x at 7B to 8B), training-free activation sparsity or per-prompt pruning on top of 4-bit weights (about 1.1x to 1.3x at small accuracy cost), and on reasoning workloads semantic parallelism (about 1.5x to 3x).

### Cited Findings
- Megakernel efficiency: 1.84x over llama.cpp at 1B on H100, 1.03x to 1.28x serving-level at 1B to 13B — [ForgeMegakernel](https://arxiv.org/html/2609.12379v1); 1.55x (RTX 3090) and 2.1x (RTX 5080) at 0.8B — [lucebox-hub](https://github.com/Luce-Org/lucebox-hub), [megakernels](https://github.com/SamuelReeder/megakernels). Generality: any architecture, per-model engineering.
- Activation sparsity: TEAL 1.53x to 1.8x (training-free, Llama and Mistral families) — [TEAL](https://arxiv.org/html/2408.14690v3); SpAx 1.07x over TEAL with Q4_K_M weights — [SpAx](https://arxiv.org/html/2610.02598). Generality: any transformer, gated FFN best; math accuracy falls first.
- Temporal reuse: LoRA-Drop up to 2.6x within 0.5 pp on Llama and Qwen — [LoRA-Drop](https://arxiv.org/abs/2601.02569). Generality: decoder-only, layer subset chosen per model.
- Per-prompt pruning: GRIFFIN 1.25x to 1.29x training-free; IFPruning up to 41% lower decode time trained — [GRIFFIN](https://arxiv.org/html/2404.01365v3), [IFPruning](https://arxiv.org/pdf/2501.02086). Generality: gated-FFN models.
- MoE carving: CMoE up to 1.17x training-free; L0-MoE up to 2.5x with 30B tokens — [CMoE](https://arxiv.org/abs/2502.04416), [L0-MoE](https://arxiv.org/abs/2609.21672).
- Multi-token per pass: Apple gated LoRA about 5x code and math, 2.5x chat (frozen base) — [arXiv 2507.11851](https://arxiv.org/pdf/2507.11851); Uno up to 3x lossless (frozen base) — [arXiv 2609.04010](https://arxiv.org/abs/2609.04010); TiDAR 4.71x to 5.91x (trained) — [arXiv 2511.08923](https://arxiv.org/html/2511.08923v1).
- Looped plus wavefront: 2.42x to 4.81x training-free on natively looped models — [WFD](https://arxiv.org/abs/2609.23033).
- Semantic parallelism: Hogwild! 3.1x to 3.6x aggregate throughput with 4 workers, training-free, reasoning models — [Hogwild!](https://arxiv.org/html/2504.06261v4); Parason about 1.7x wall-clock, ThreadWeaver 1.14x measured wall-clock, Multiverse up to 2x, PASTA 1.21x to 1.93x — [Parason](https://arxiv.org/abs/2608.24658), [ThreadWeaver](https://arxiv.org/pdf/2512.07843), [Multiverse](https://arxiv.org/pdf/2506.09991), [PASTA](https://arxiv.org/html/2502.11517v2).
- Width surgery: 1.8x throughput at 7 MMLU points lost with 94B tokens; SliceGPT 25% training-free — [Minitron 4B](https://developer.nvidia.com/blog/how-to-prune-and-distill-llama-3-1-8b-to-an-nvidia-llama-3-1-minitron-4b-model/), [SliceGPT](https://arxiv.org/abs/2401.15024).
- Layer parallel: 1.22x to 1.73x training-free (Layer Parallelism, AdaDecode); StagFormer simulated only, from scratch — [arXiv 2502.02790](https://arxiv.org/pdf/2502.02790), [AdaDecode](https://arxiv.org/abs/2506.03700), [StagFormer](https://arxiv.org/html/2501.15665v2).
- Retrieval, LUT-AMM, memory layers, SeedLM: no published post-hoc single-stream GPU decode speedup at LLM scale — [Memory Layers](https://arxiv.org/abs/2412.09764), [MADDNESS](https://proceedings.mlr.press/v139/blalock21a/blalock21a.pdf), [SeedLM](https://arxiv.org/abs/2410.10714).

### Inferences
- Illustrative stacking for an 8B model at about 4.5 GB of 4-bit weights on 912 GB/s (my arithmetic): bandwidth floor about 200 forward passes per second. llama.cpp at an assumed 60% to 70% efficiency gives 120 to 140 passes per second; with speculative drafting at 2 tokens per pass, about 250 to 280 tok/s. A 5x target is about 1250 to 1400 tok/s. A plausible stack: megakernel to about 85% efficiency (170 passes per second) times about 7 tokens per pass at about 1.4x step cost (TiDAR-class acceptance via frozen-base adapter) gives about 850 tok/s; adding 1.3x from activation sparsity or per-prompt FFN pruning gives about 1100 tok/s; reasoning workloads with 2 to 4 Hogwild-style workers add aggregate throughput but share the same spare rows as multi-token verification, so the two do not multiply freely. The 5x target is reachable on paper only on favorable text (code, math, structured) and only if 7-token acceptance holds at the user's sampling temperature and with 4-bit weights, neither of which has been measured.
- The binding constraint for all "more tokens per weight read" methods on a GeForce Ampere card is the finite spare-row budget before compute binds; multi-token verification, parallel branches, superposition, and wavefront batching all draw on the same budget.

### Gaps
- No paper measures any stack of these methods together on a consumer GPU.
- Acceptance rates of multi-token adapters at temperature above 0 on quantized weights are unpublished.

## Thin or unexplored (ranked by potential multiplier if closed)

1. Frozen-checkpoint multi-token adapters evaluated at batch 1 on consumer GPUs with 2 to 4-bit weights (potential 3x to 6x over plain AR, 1.5x to 3x over spec-drafted llama.cpp). Uno and gated-LoRA MTP keep base weights frozen and report 3x to 5x, but on server GPUs, unquantized, mostly at temperature 0. Nobody has measured how quantization noise affects their acceptance, or how the spare-row budget of a GeForce card caps tokens per pass — [Uno](https://arxiv.org/abs/2609.04010), [Apple MTP](https://arxiv.org/pdf/2507.11851), [TiDAR](https://arxiv.org/html/2511.08923v1)
2. Checkpoint-to-looped conversion plus wavefront decoding (potential 2x to 3x fewer unique weight bytes times 2.4x to 4.8x depth-as-batch). WaveFront Decoding works only on natively looped models (Ouro, Huginn); retrofitted recurrence exists only at 1B with continued pretraining; no paper composes the two for a 7B+ dense checkpoint — [WFD](https://arxiv.org/abs/2609.23033), [Retrofitted recurrence](https://arxiv.org/html/2511.07384v1), [Relaxed Recursive](https://arxiv.org/abs/2410.20672)
3. A unified spare-row scheduler that allocates the batched-decode budget across multi-token verification, parallel reasoning branches, and draft trees for one request (potential 1.5x to 2x on top of either alone). Hogwild!, TiDAR, and Speculative Pipeline Decoding each consume spare rows independently; no paper co-schedules them — [Hogwild!](https://arxiv.org/html/2504.06261v4), [TiDAR](https://arxiv.org/html/2511.08923v1), [SPD](https://arxiv.org/abs/2605.30852)
4. Training-free semantic parallelism for non-reasoning tasks (code, chat, structured output) at batch 1 (potential 1.5x to 2x). Training-free methods are either reasoning-only (Hogwild!) or fail on harder reasoning (SoT); PASTA, the only open-ended result, needs training — [PASTA](https://arxiv.org/html/2502.11517v2), [SoT](https://arxiv.org/abs/2307.15337v1)
5. Temporal activation reuse (memoization across tokens) and cross-request layer-output caching (potential 1.5x to 2.6x). LoRA-Drop is the only published single-request temporal-reuse decoder; semantic caching of layer outputs across requests has no published LLM work; hidden states change little between tokens (cosine 0.6 to 0.85) — [LoRA-Drop](https://arxiv.org/abs/2601.02569)
6. Activation sparsity inside a low-bit megakernel on GPU-resident weights (potential 1.2x to 1.5x). SpAx shows sparsity gains collapse to 1.07x over TEAL once weights are Q4_K_M, partly because quantized row-gather kernels are inefficient; no published 2 to 4-bit sparse GEMV megakernel for consumer GPUs — [SpAx](https://arxiv.org/html/2610.02598), [TEAL](https://arxiv.org/html/2408.14690v3)
7. Quantized 7B+ megakernels on Ampere consumer GPUs (potential 1.2x to 1.4x over llama.cpp). Published megakernels are BF16 on H100 (ForgeMegakernel) or sub-1B on RTX cards — [ForgeMegakernel](https://arxiv.org/html/2609.12379v1), [lucebox-hub](https://github.com/Luce-Org/lucebox-hub)
8. Per-prompt weight specialization beyond neuron selection (potential 1.3x to 2x). GRIFFIN-style training-free selection is 1.25x to 1.29x; nobody re-quantizes, re-codes, or drops layers per prompt using prefill statistics and measures single-stream decode — [GRIFFIN](https://arxiv.org/html/2404.01365v3), [IFPruning](https://arxiv.org/pdf/2501.02086)
9. Post-hoc memory-layer or PKM retrofits at 7B+ scale (potential large byte reduction per token if FFNs can be replaced by row lookups, but recovery training cost unknown). Only a 0.5B Qwen retrofit with recovery training exists — [arXiv 2604.05248](https://arxiv.org/html/2604.05248), [UltraMemV2](https://iclr.cc/virtual/2026/poster/10009579)
10. Post-hoc staggered or depth-pipelined conversion for single-GPU use (potential 1.1x to 1.3x, latency hiding only, since bytes are unchanged on one device). StagFormer is from-scratch only — [StagFormer](https://arxiv.org/html/2501.15665v2)
11. GPU kernels for seed-generated or procedural weights (potential under 1.2x versus existing 2 to 4-bit quants, because SeedLM's bytes per weight match 3 to 4-bit quantization). Only FPGA results exist — [SeedLM](https://arxiv.org/abs/2410.10714)
12. Activation-side lookup matmul (MADDNESS, PQ-AMM) for LLM decode (potential at most 1x on GPU by the bytes argument in Section 2, since the table is at least as large as a 2-bit weight matrix at usable accuracy) — [MADDNESS](https://proceedings.mlr.press/v139/blalock21a/blalock21a.pdf), [Look-ups are not (yet) all you need](https://arxiv.org/pdf/2207.05808)
13. L2-resident weight shards on consumer GPUs (negligible on a 6 MB L2) — [GA102 whitepaper](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf)

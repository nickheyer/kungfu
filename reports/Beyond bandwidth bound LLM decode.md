# Closing 5x Requires Taming the Verify Wall

No published or preprint method through November 2026 reaches 5x the single-stream speed of llama.cpp's best configuration at equal accuracy on an RTX 3080 Ti. No single new mechanism comes close. llama.cpp already ships the strongest frozen-target drafters (EAGLE-3, MTP, DFlash, DSpark). The papers that report 5x or more (TiDAR 5.91x, DFlare 5.46x, JetSpec 9.64x on math, DiffusionGemma about 7x) measure against plain autoregressive decode on datacenter GPUs, and each either gives up accuracy or needs verify widths that a GA102 cannot run for free. For Qwen3-8B on this card the bar is about **1,270 t/s geomean, 1,540 t/s on GSM8K-style math and 780 t/s on MT-Bench chat**. For Llama-3.1-8B it is 1,165 to 1,630 t/s. Those targets are 6 to 8x the Q4_K_M bandwidth floor and above the 804 to 909 t/s floor of a 1-bit 8B model, so the multiplier has to come from accepted tokens per weight pass. That runs into a wall the literature has not studied: the per-byte decode and issue cost of multiplying low-bit weights against several verify rows. Tensor peak is not that wall. An INT8 verify kernel on this card has 272 TOPS of dense tensor peak, which puts the FLOP ridge near 55 to 66 rows at 4 bits. The reader's 1.9 steps for 16 rows is issue-bound: decode work per byte plus multiply-issue work per eight rows per byte, both set by the weight format and the kernel design. Exactly one stack of published results closes 5x on paper:

- a DFlare-class block-16 drafter quantized to 4 bits
- a verify kernel at about 1.15 steps for 16 rows
- a 4.0-bpw trellis code that beats Q4_K_M perplexity
- a persistent kernel at 85 percent of bandwidth
- TEAL-class sparsity inside verification

At greedy decoding and short context the stack reaches about 5.2x on MATH-500 and 5.3x on MT-Bench, but only 3.8 to 4.6x on HumanEval, GSM8K and MBPP. It closes on all five benchmarks once more than about 13K tokens of context sit in a 4-bit, page-sparse KV cache. It fails in thinking mode and with a bf16 drafter. It also fails at temperature 1 unless llama.cpp's drafter degrades by the same factor, which no one has measured. The research is thinnest, and an original multiplier most likely, at that decode-and-issue wall:

- a sub-4-bit weight format whose per-byte decode is cheap enough for a multi-row kernel to sit at the bandwidth floor, about 1.6x at 4 bpw and about 2.4x if it also reaches Q4_K_M accuracy at 2.5 bpw
- lossless acceptance at low-margin prose positions, 1.5 to 2x on chat
- KV formats built for multi-row reuse, 1.2 to 1.9x from 8K to 32K context
- quantization-aware drafters, about 1.3x

## The bar is 1,165 to 1,630 tokens per second for an 8B model

llama.cpp exposes every drafter family in the brief through `--spec-type`: draft-eagle3, draft-mtp, draft-dflash, draft-dspark, and five ngram modes (ngram-cache, ngram-simple, ngram-map-k, ngram-map-k4v, ngram-mod). When a model drafter is combined with a draftless mode, the draftless mode takes precedence; the two do not form a joint draft ([speculative.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md)). The strongest measured 8B configuration is DSpark, merged 2026-07-28. On an RTX 4090, Qwen3-8B Q4_K_M decodes at 155.4 t/s plain and **1.81x geomean with DSpark, peaking at 340.7 t/s on GSM8K** at 77.4 percent acceptance ([PR #25173](https://github.com/ggml-org/llama.cpp/pull/25173)). EAGLE-3 lifts Llama-3.1-8B Q4_K_M on an RTX A6000 from 121.5 t/s to 196.5-274.4 t/s ([PR #18039](https://github.com/ggml-org/llama.cpp/pull/18039)). MTP takes Qwen3.6-27B q4_0 on an RTX 5090 from 63.72 to 105.47 t/s ([PR #22673](https://github.com/ggml-org/llama.cpp/pull/22673)). A request to chain MTP with ngram-mod was closed with no measurements ([Issue #23184](https://github.com/ggml-org/llama.cpp/issues/23184)).

The regularity that defines the bar is that **llama.cpp's drafter gain shrinks as its weights shrink**. On the same 4090, DSpark gives 3.35x at bf16, 2.68x at Q8_0 and 1.81x at Q4_K_M, while GSM8K acceptance barely moves (78.9 vs 77.4 percent) ([PR #25173](https://github.com/ggml-org/llama.cpp/pull/25173)). Heavy quantization taxes the verify pass, not the draft.

| Model and quant | GPU | Plain t/s | Best measured drafter | Bar t/s | 5x target t/s |
|---|---|---|---|---|---|
| Qwen3-8B Q4_K_M | RTX 4090, measured | 155.4 | DSpark 1.81x geomean, 340.7 on GSM8K | 281 geomean / 341 GSM8K | 1,406 / 1,704 |
| Qwen3-8B Q4_K_M | RTX 3080 Ti, scaled by 912/1008 | 141 | same ratios | 254 / 308 | 1,272 / 1,541 |
| Qwen3-8B Q4_K_M | RTX 3090, scaled by 936/1008 | 144 | same ratios | 261 / 316 | 1,306 / 1,582 |
| Llama-3.1-8B Q4_K_M | RTX A6000, measured | 121.5 | EAGLE-3 196.5-274.4 | 274 | 1,372 |
| Llama-3.1-8B Q4_K_M | RTX 3080 Ti, scaled by 912/768 | 144 | EAGLE-3 233-326 | 326 | 1,165-1,630 |
| Llama 3 8B Q4_K_M | RTX 3080 Ti, measured May 2024 | 110.6 (tg512), 88.6 (tg8192) | none published | 111 | 553 before drafting |
| Llama-3.2-1B Q4_K_M | RTX 4090, llamafile | 200 | none published | 200 | 1,000 |
| Qwen3-30B-A3B Q4_K_M | RTX 4090 / 5090, measured | 198.4-271.0 / 247.0-352.1 | llama.cpp MoE drafting at or below 1.4x | 271 / 352 | 1,355 / 1,760 |
| Qwen3-30B-A3B, 2-bit | RTX 3080 Ti, reference in the research brief | 52-55 | none | 55 | 260-275 |

Sources for the table: [PR #25173](https://github.com/ggml-org/llama.cpp/pull/25173), [PR #18039](https://github.com/ggml-org/llama.cpp/pull/18039), [XiongjieDai benchmarks](https://github.com/XiongjieDai/GPU-Benchmarks-on-LLM-Inference), [LocalScore](https://www.localscore.ai/accelerator/77), [Discussion #17621](https://github.com/ggml-org/llama.cpp/discussions/17621). Scaled rows are bandwidth-proportional estimates at llama.cpp's measured efficiency. The 2-bit MoE row uses the reference figures the research brief gives for the reader's own coprocessor (112 to 129 t/s) against llama.cpp; it is not a published result.

Three features of this table decide where 5x is meaningful.

First, on verbatim or code-only output llama.cpp already runs near the block horizon. DFlash on a code-only quicksort prompt reaches 419.3 t/s bf16, 8.08x at 93.3 percent acceptance ([PR #22105](https://github.com/ggml-org/llama.cpp/pull/22105)). ngram-mod hits 0.92 acceptance with 64-token drafts on a verbatim-repeat task ([PR #19164](https://github.com/ggml-org/llama.cpp/pull/19164)). Beating those by 5x would need about 40 accepted tokens per pass against a 16-token block, which no drafter can supply.

Second, llama.cpp is weakest exactly where drafting is weak. Scale DSpark's bf16 per-domain gains by the measured Q4_K_M-to-bf16 ratio (0.54 for both GSM8K and the geomean) and its MT-Bench gain comes out near 1.11x. Every published llama.cpp speculative result on MoE targets sits at or below 1.4x. DFlash on GPT-OSS-20B runs at 0.61 to 1.27x and EAGLE-3 at 0.89 to 1.06x ([PR #22105](https://github.com/ggml-org/llama.cpp/pull/22105), [PR #18039](https://github.com/ggml-org/llama.cpp/pull/18039)).

Third, the 1B class is launch-bound. Llama 3.2 1B Q4_K_M runs at 200 t/s on a 4090, about 16 percent of its 1,211 t/s floor ([LocalScore](https://www.localscore.ai/accelerator/77)). A 5x over plain decode there sits below the bandwidth floor, so it is a kernel-efficiency question, not a research one.

The per-domain bars that the stack section tests against come from applying that 0.54 ratio to DSpark's measured bf16 gains on the 4090 ([PR #25173](https://github.com/ggml-org/llama.cpp/pull/25173)):

| Domain, greedy | DSpark gain at bf16 (measured) | Gain at Q4_K_M | 3080 Ti bar t/s | 5x target t/s |
|---|---|---|---|---|
| GSM8K | 4.06x | 2.19x measured | 308 | 1,541 |
| MATH-500 | 3.82x | 2.06x scaled | 290 | 1,448 |
| HumanEval | 4.08x | 2.21x scaled | 311 | 1,554 |
| MBPP | 3.24x | 1.75x scaled | 246 | 1,230 |
| MT-Bench | 2.05x | 1.11x scaled | 156 | 780 |
| Geomean | 3.35x | 1.81x measured | 254 | 1,272 |

Scaling by a constant ratio is justified because DSpark runs at a fixed draft budget and acceptance barely changes with target quantization, so the round-cost ratio between bf16 and Q4_K_M is the same for every domain.

**The notes disagree on the plain 3080 Ti rate.** Scaling the 2026 runs on the 4090 and A6000 gives 141 to 144 t/s. Scaling the RTX 3090 Q4_0 scoreboard gives 125 to 137 t/s ([Discussion #15013](https://github.com/ggml-org/llama.cpp/discussions/15013)). The only direct measurement is 110.6 t/s from a 2024 snapshot ([XiongjieDai](https://github.com/XiongjieDai/GPU-Benchmarks-on-LLM-Inference)). This report uses 141 to 144 t/s because those runs use 2026 kernels, and the higher plain rate makes the bar harder to beat, not easier.

**The MoE bar has a second disagreement.** The brief's reference puts llama.cpp at 52 to 55 t/s for Qwen3-30B-A3B at 2 bits. A fully resident IQ2_XXS build (about 8.0 GB, computed) running at the 38 to 51 percent of floor that llama.cpp reaches on a 4090 and 5090 would instead decode at about 350 to 475 t/s (computed from the 931 t/s floor and [Discussion #17621](https://github.com/ggml-org/llama.cpp/discussions/17621)). That would move the 5x target to 1,750 to 2,400 t/s. No public run settles which bar is real.

Custom runtimes on consumer cards already show how much llama.cpp's speculative path leaves unused. Lucebox's DFlash plus DDTree runs Qwen3.5-27B Q4_K_M at 129.5 t/s mean on HumanEval on one RTX 3090, 3.43x over an AR baseline that matches llama.cpp settings. A later Lucebox run of Qwen 3.8 27B on an AMD R9700 reports **3.8x over llama.cpp with the same drafter**. These figures are self-reported ([Lucebox hub](https://github.com/Luce-Org/lucebox-hub), [HN thread](https://news.ycombinator.com/item?id=47838788)).

## Three walls cap any runtime on this card, and the third is unstudied

### Bandwidth puts the 5x target above the 1-bit floor

Bytes per token follow from the published configs. Qwen3-8B has 36 layers with 8 KV heads of dimension 128 and an untied 151,936-token head; Llama-3.1-8B has 32 layers and a 128,256-token head ([Qwen3-8B config](https://huggingface.co/Qwen/Qwen3-8B/raw/main/config.json), [Llama-3.1-8B config](https://huggingface.co/unsloth/Meta-Llama-3.1-8B-Instruct/raw/main/config.json)). File sizes come from GGUF quant repos ([bartowski GGUF](https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF)). The floors below count every non-embedding weight once, the LM head once, and f16 KV:

| Model, weights | GB read per token at 1K context | 3080 Ti floor at 1K, t/s | Floor at 32K, f16 KV, t/s |
|---|---|---|---|
| Llama-3.1-8B Q4_K_M | 4.62 + 0.13 KV | 192 | 102 |
| Llama-3.1-8B IQ2_M | 2.65 + 0.13 | 327 | 131 |
| Llama-3.1-8B IQ2_XXS | 2.10 + 0.13 | 409 | 143 |
| Qwen3-8B Q4_K_M | 4.68 + 0.15 | 189 | 96 |
| Qwen3-8B IQ2_XXS | 2.14 + 0.15 | 398 | 131 |
| Llama 3 8B at 1.0 bpw body, 2- to 4-bit head | 1.00-1.14, no KV | 804-909 | n/a |
| Llama-3.2-1B Q4_K_M | 0.80 + 0.03 | 1,095 | 487 |
| Qwen3-30B-A3B IQ2_XXS, active weights | 0.88 + 0.10 | 931 | 222 |

The 5x targets of 1,165 to 1,630 t/s are above every dense 8B floor in the table, the 1-bit floor included. Even 1-bit weights at full bandwidth would still need 1.3 to 2x from tokens per pass (computed). Weight compression is a multiplier on tokens per pass, not a substitute for it. At 32K context, f16 KV is 48 to 51 percent of 8B traffic, so long-context decode becomes a KV problem. The LM head is 9 to 16 percent of weight traffic for the 8B models.

**The notes disagree on this floor** for Llama-3.1-8B Q4_K_M: 185 t/s using the 4.92 GB file size, 192 t/s using 4.62 GB of per-token weights plus 0.13 GB of KV, and 197 t/s using the 4.62 GB weights alone.

### Accepted tokens per pass saturate near 10 on math and 6 on chat

The one formal bound comes from Pankratov and Alistarh: E[X] <= (mu + mu2) log P / mu^2 + O(1), where P is verifier capacity and mu is the target's expected entropy. In their words, exponentially more verifier compute buys only linearly more speedup ([arXiv 2512.11718](https://arxiv.org/abs/2512.11718v1)). Their measured temperature-1 entropies for Qwen3-8B are 0.178 on HumanEval and 0.369 on MT-Bench. Plugging those in at P = 16 gives loose limits of about 43 and 24 tokens (computed, indicative only).

The binding limits are empirical:

- **Block horizon.** DBloom tracks a "ceiling bin", the fraction of rounds that accept the whole block. That fraction predicts the gain from widening a block from 16 to 24 (Spearman rho 0.88 to 0.93). MT-Bench stays at tau 5.00 to 5.38 while MATH-500 reaches 10.54. The authors stop at 24 because the ceiling bin empties ([arXiv 2608.30427](https://arxiv.org/abs/2608.30427)).
- **Per-position decay.** On AIME25, JetSpec's per-position acceptance falls from about 99 percent at depth 1 to about 50 percent at depth 8 ([JetSpec](https://arxiv.org/html/2606.18394v3)).
- **Domain.** Across DFlash, DSpark, JetSpec, DBloom, post-hoc MTP heads and gated LoRA, chat tau is 45 to 60 percent of math tau at the same budget ([DSpark](https://arxiv.org/html/2607.05147v1), [Apple MTP](https://arxiv.org/html/2507.11851v1)).
- **Temperature.** DFlash's average tau falls from 6.49 to 5.48 at temperature 1 ([DFlash](https://arxiv.org/html/2602.06036v2)).
- **Thinking mode.** DFlash acceptance falls from 93.3 to 12.0 percent on the same prompt when thinking is on ([PR #22105](https://github.com/ggml-org/llama.cpp/pull/22105)).

The best measured lossless figures are tau 10.54 (DBloom B24) and 10.76 (JetSpec, 256-node tree) on MATH-500 at greedy decoding, 5.38 and 5.94 on MT-Bench, and 9.01 and 5.22 for JetSpec at temperature 1 ([JetSpec](https://arxiv.org/html/2606.18394v3)).

**The notes disagree on how entropy relates to acceptance.** The bound and its measurements say low-entropy code supports the longest acceptance. A 1B-draft, 7B-target study found chat had the highest target entropy (1.25 nats) and the highest acceptance (0.565), with Spearman |rho| < 0.21 between entropy and acceptance ([arXiv 2604.14682](https://arxiv.org/html/2604.14682)). That study's TinyLlama drafter is weak, which limits how far its result carries.

### Per-byte decode and issue cost, not tensor peak, sets the multi-row verify wall

GA102's tensor rates do not bind a multi-row verify. Per the GA102 whitepaper, an RTX 3090 (82 SMs at 1695 MHz) peaks at 142 dense FP16 tensor TFLOPS with FP16 accumulate, 71 TFLOPS with FP32 accumulate (GeForce half rate) and 284 dense INT8 TOPS with INT32 accumulate ([GA102 whitepaper](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf)). Scaled to the 3080 Ti's 80 SMs at 1665 MHz, those are **136 TFLOPS, 68 TFLOPS and 272 TOPS**. The reader's verify kernel (docs/mm-i8.md in their repo) runs `mma.sync` m8n8k16 INT8 with INT32 accumulate over natively quantized weights, so its tensor peak is 272 TOPS, and at 60 percent it achieves about 160 TOPS. An 8B dense model costs 15 to 16 GOP per verified row, about 0.1 ms per row (computed).

The FLOP ridge is the row count at which tensor time equals the weight read at 85 percent of 912 GB/s. It sits far above any useful block (computed):

| Weight format | FLOP ridge, rows per pass |
|---|---|
| Q4_K_M | about 60-66 |
| 4.0 bpw | about 55 |
| IQ2_M | about 35-37 |
| 2.0 bpw | about 27 |
| 1.0 bpw | about 14 |

**Block 16, block 24 and 32-row trees are not FLOP-bound at any bit width above about 1.5 bpw.**

The reader's measurements (1.4 steps at 8 rows, 1.9 at 16, 2.73 at 24, 2.92 at 32) are documented in their notes as issue-bound, not tensor-bound:

- With the `mma` work removed, the decode alone streams at 584 GB/s on the gate-and-up shape, against 668 GB/s for the one-row matvec.
- The kernel sits at 128 registers and two blocks per SM.
- Its per-run work (two shuffles, two `mma`, a fold) is paid once per eight rows per weight run.

**The wall is the per-byte decode cost plus the per-eight-rows-per-byte issue work of multiplying decoded weights against the rows.** The weight format and the kernel design set it; tensor peak does not.

That mechanism, not a FLOP budget, is why the usable window narrows as bits fall:

- **More weights per byte.** A 2-bit format carries 4 weights per byte against about 1.7 at Q4_K_M, so decode work and multiply-issue work per byte both rise (computed).
- **More work per weight.** Codebook formats (IQ2, trellis, VQ) add lookup or bit-manipulation work to every weight. That is why the 2-bit kernels in the notes run at 26 to 62 percent of the floor even at one row ([QTIP](https://arxiv.org/html/2406.11235), [Q-Palette](https://arxiv.org/html/2509.20214)).

A codebook kernel that stays near the floor at 16 rows is therefore harder to build than a Q4 one.

Near-overlap is reachable. On an RTX 4090, the FLUTE 2-bit kernel delivers 2,432 t/s at batch 16 against 173 t/s at batch 1, which means 16 rows cost 1.14 steps; FP16 manages 1.06 steps ([HIGGS](https://arxiv.org/html/2411.17525v1)). FLUTE's one-row baseline runs near 48 percent of the floor, so part of that ratio reflects a slow single-row kernel. Hogwild! measures 4 rows at 1.17 to 1.22x a single-row step on an L40S ([Hogwild!](https://arxiv.org/html/2504.06261v4)).

Two consequences follow.

- **Long-context attention is borderline, not a wall.** At 32K, attention adds about 19.3 GFLOP per row for Qwen3-8B, so 16 rows cost about 309 GFLOP (computed). Against a weight plus 4-bit KV read of about 6.7 ms, that attention takes:
  - about 3.8 ms at FP16 accumulate (136 TFLOPS peak, about 82 achieved)
  - about 7.5 ms at FP32 accumulate (68 TFLOPS peak, about 41 achieved)

  Whether dense attention reaches the critical path therefore depends on accumulate precision. Page-sparse reads keep it off the critical path.
- **Large trees lose on this card.** Datacenter tree budgets of 128 to 256 nodes exceed the FLOP ridge even at INT8 rates. 256 rows x 15 GOP is 3.84 TOP, about 24 ms at 160 TOPS, against a 5 to 6 ms weight read (computed). SpecMQuant found EAGLE-2 trees added nothing on a W4A16 8B model on an RTX 3090, and found verification on 4-bit weights costs 1.8 decode steps at tree size 60 against below 1.2 at FP16 ([SpecMQuant](https://arxiv.org/html/2505.22179v1)).

## Each axis reports 1.1x to 9.6x, mostly against plain decode on datacenter GPUs

### Parallel generation: frozen-target drafters lead, conversions trade accuracy for width

| Method | Base weights and adaptation | Tokens per pass | Reported speed | Accuracy vs parent |
|---|---|---|---|---|
| DFlash, block 16, Qwen3-8B | frozen; about 800K regenerated samples | tau 6.49 average greedy (MATH-500 7.87, MT-Bench 4.24); 5.48 at temp 1 | 4.86x average, 6.08x MATH-500, 2.75x MT-Bench (H200) | lossless |
| DFlare, block 16 | frozen; 2.4M samples | GSM8K 7.86, MATH-500 9.17, HumanEval 7.07, MBPP 6.77, MT-Bench 5.00 | 5.46x average | lossless |
| DBloom, block 24 | frozen; +30K prompts | GSM8K 9.23, MATH-500 10.54, HumanEval 7.93, MT-Bench 5.38 | 4.88-4.91x | lossless |
| JetSpec, 256-node tree | frozen; about 800K examples on 8 H100 | MATH-500 10.76, MT-Bench 5.94 greedy | 9.64x MATH-500, 4.58x MT-Bench | lossless |
| DSpark, block 7, temp 1 | frozen; 1.3M prompts, 10 epochs | GSM8K 6.17, HumanEval 5.52, MT-Bench 3.72 | +26.7% tau over EAGLE-3 | lossless |
| Gated LoRA, 8 masks, Tulu3-8B | frozen; LoRA only on mask slots, 50K iterations on 8 A100 | HumanEval 5.35, GSM8K 5.22, AlpacaEval 2.52 | about 5x code and math, 2.5x chat | NTP unchanged |
| Uno | frozen AR plus light diffusion weights | not reported | up to 3x | lossless |
| Post-hoc MTP heads, K=3, Qwen3-8B | frozen; about 40 h on 32 A100 | 2.32-3.05 | not reported | lossless; relaxed verify cut AIME25 from 76.7 to 60.0 |
| TiDAR 8B | 150B tokens of continual pretraining | 8.25 tokens per forward | 5.91x (H100) | average 65.31 vs 68.09 |
| I-DLM 8B | full fine-tune, 4.5B tokens | TPF 1.80-4.01 | 3.1x at matched MATH-500 | AIME-25 60.8 vs 65.4 |
| Set Block Decoding, Qwen3-8B | full fine-tune, 70B tokens | forward passes cut 2.5-5.4x | roofline only | AIME25 33.3 to 26.6 at high gamma |
| Jacobi Forcing, Qwen2.5-Coder-7B | full fine-tune, about 20K steps | TPF 4.01-4.09 | 3.86-3.97x (A100) | HumanEval 87.8 to 83.5 |
| DiffusionGemma, 3.8B active | fine-tune plus RL | TPF 18.8-19.7 | 1,479 vs 204 t/s AR (H100) | AIME 2026 69.1 vs 84.2 |
| ESP | training-free | tau 1.66-1.71 | not reported | lossless |

Sources: [DFlash](https://arxiv.org/html/2602.06036v2), [DFlare](https://arxiv.org/abs/2606.02091), [DBloom](https://arxiv.org/pdf/2608.30427), [JetSpec](https://arxiv.org/html/2606.18394v3), [DSpark](https://arxiv.org/html/2607.05147), [gated LoRA](https://arxiv.org/html/2507.11851v1), [Uno](https://arxiv.org/abs/2609.04010), [post-hoc MTP](https://arxiv.org/html/2610.00888v1), [TiDAR](https://arxiv.org/html/2511.08923v1), [I-DLM](https://arxiv.org/html/2604.11035v1), [SBD](https://arxiv.org/pdf/2509.04185), [Jacobi Forcing](https://arxiv.org/pdf/2512.14681), [DiffusionGemma](https://arxiv.org/html/2608.00146v1), [ESP](https://arxiv.org/html/2603.17942v1).

**What lies beyond llama.cpp's drafters is narrow.**

- DFlare improves average speed about 8 percent over DFlash. DBloom's block 24 adds 17 percent tau on GSM8K (9.23 vs 7.86). At the reader's measured issue cost those 24 rows do not pay; the stack section gives the arithmetic.
- JetSpec's 256-node trees exceed the FLOP ridge even at INT8 tensor rates (about 24 ms of tensor work at 160 TOPS against a 5 to 6 ms weight read, computed). At a budget of 16, JetSpec merely ties DFlash (MT-Bench 2.68x).
- Frozen-base single-model adapters work: gated LoRA, Uno, and I-DLM's lossless R-ISD variant. Orthrus claims 7.8x lossless with no published detail ([arXiv 2605.12825](https://arxiv.org/abs/2605.12825)). None of these has been measured with quantized weights or on consumer hardware.
- **Every conversion that changes base weights loses 1 to 5 points, and up to 15 on hard reasoning.** Those losses fail the equal-accuracy requirement.

Without training, self-speculation in the 8B class is weak. An independent ICML 2026 study measured SWIFT at 1.07 to 1.17x on LayerSkip-Llama-3-8B, against 1.3 to 1.6x claimed by its authors ([ICML 2026 study](https://www.eecg.utoronto.ca/~ganesa10/assets/pdfs/selfspecdecode-icml2026.pdf)).

### Activation sparsity buys 1.1x to 1.6x and dynamic depth buys nothing at batch 1

| Method | Under-1-point operating point | Measured decode |
|---|---|---|
| TEAL, Llama-3-8B | 25% model-wide sparsity, -0.34 average | A6000 1.22x / 1.42x / 1.63x at 25 / 40 / 50%; A100 1.11x / 1.25x / 1.33x |
| WiSparse, Llama-3.1-8B Instruct | 30% (-0.17), 40% (-0.87) | H20 +17.2% at 50% |
| Prox (FFN only), Qwen3-8B | 40% FFN sparsity, -0.4 | RTX 3090 54.7 to 95.1 t/s at 70% FFN, which costs 7.5 points |
| R-Sparse, Llama-2-7B | 40%, -0.88 | up to about 40% faster (A6000, FP32) |
| SpQt, Q4_K weights, Apple GPUs | Llama-2-7B PPL 5.28 to 5.27 at 25% | 1.32-1.55x, the only measured quantized sparse decode |
| FlexiDepth, Llama-3-8B-Instruct | 6.25 of 32 layers skipped at 102.1% retention (own claim) | 21% slower at batch 1 |
| MindSkip (attention skipped per sequence), Llama-3-8B | -0.3 at 1.21x | measured at saturating batch sizes |
| LoRA-Drop, Llama and Qwen 7-14B | within 0.5 points | up to 2.6x |

Sources: [TEAL](https://arxiv.org/html/2408.14690v3), [WiSparse](https://arxiv.org/pdf/2602.14452), [Prox](https://arxiv.org/html/2607.27591), [R-Sparse](https://arxiv.org/html/2504.19449v1), [SpQt](https://arxiv.org/html/2511.04477), [FlexiDepth](https://arxiv.org/html/2503.23798v3), [vSkipper](https://arxiv.org/html/2609.37062v1), [Router-Tuning](https://arxiv.org/html/2410.13184v1), [LoRA-Drop](https://arxiv.org/abs/2601.02569).

**A defensible equal-accuracy budget is about 25 percent of block bytes**, a 1.30x ceiling on Llama-3-8B. Math and code degrade first. In WINA, base Llama-3-8B loses 8 GSM8K points at 40 percent, while WiSparse holds GSM8K at 40 percent on the instruct model ([WINA](https://arxiv.org/html/2505.19427v1)).

Three obstacles stand between that ceiling and a 4-bit consumer runtime:

- **The A100's realization ratio is the right prior.** The A6000 realizes 87 to 94 percent of the byte-bound gain; the A100 realizes much less. A 4-bit step on a 3080 Ti is as short as an FP16 step on an A100, and TEAL-type kernels carry 1.0 to 1.5 ms of fixed overhead on an A6000 ([arXiv 2609.33889](https://arxiv.org/html/2609.33889v1)). That overhead would be 20 to 30 percent of a 5 ms step (computed).
- **Standard quant layouts block input-channel skipping.** Row-major quantization groups make skipping input channels useless. SpQt needed a zigzag Q4_K layout, and without it naive sparse kernels ran below 0.2x ([SpQt](https://arxiv.org/html/2511.04477)). TEAL, Prox and R-Sparse leave quantized sparse kernels to future work.
- **Multi-row verify batches erode sparsity.** TEAL's W_down sparsity falls from about 60 to 38 percent at batch 4 ([TEAL](https://arxiv.org/html/2408.14690v3)). BlockFFN's active-expert fraction grows from 12.45 to 30.86 percent over a 32-token tree ([BlockFFN](https://arxiv.org/pdf/2507.08771)). Active neurons persist 82 to 85 percent between consecutive tokens ([NeuroPrefetcher](https://arxiv.org/html/2608.22643)), which suggests half to two thirds of single-row sparsity survives in an 8- to 16-row verify.

Dynamic depth fares worse. No retrofitted per-token router speeds up batch-1 GPU decode. **The notes disagree on FlexiDepth's accuracy**: its authors report near-100 percent retention, while vSkipper served the same checkpoint and measured -3.64 GSM8K points and BBH falling from 68.04 to 59.65 ([vSkipper](https://arxiv.org/html/2609.37062v1)).

### Sub-2-bit codes beat GGUF by 0.45 bpw but need 4 bpw to match Q4_K_M

| Format, Llama 3 or 3.1 8B | bpw | Adaptation | Perplexity vs FP16 | Decode evidence |
|---|---|---|---|---|
| GGUF Q4_K_M | 4.90 whole file | none | 1.024x | about 72-74% of floor (computed) |
| EXL3 (QTIP-derived) | 4.0 decoder | none | 7.729 vs Q4_K_M 7.810, same harness | not reported |
| GGUF IQ2_M / IQ2_XXS / IQ1_S | 2.94 / 2.39 / 2.01 whole file | none | 1.31x / 2.07x / 8.67x | no public tg tables |
| YAQA-B plus fine-tune | 2 | 1-30 GPU-h Hessians plus fine-tune | 1.17x | not reported |
| LLVQ plus scale fine-tune | 2 | about 52M tokens | 1.22x | slightly slower than QTIP |
| PV-Tuning | 2.01 / 1.01 | 2-8 days on 8 GPUs | 1.26x / 2.02x | not reported |
| QTIP | 2 | QuIP# recipe | 1.32x | 61% of byte bound (L40S GEMV); 127 t/s Llama 2 7B on RTX 3090 |
| ParetoQ QAT | 2 / 1.58 / 1 | about 30B tokens | 1.30x / 1.40x / 1.54x | minimal kernel gains at 1-8B |
| BTC-LLM | 1.11 | none | 1.25x, unreproduced | not reported |

Sources: [exllamav3 8B chart](https://github.com/turboderp-org/exllamav3/blob/master/doc/llama31_8b_instruct_bpw.png), [GGUF perplexities](https://huggingface.co/ThomasBaruzier/Meta-Llama-3.1-8B-Instruct-GGUF), [YAQA](https://arxiv.org/html/2505.22988), [LLVQ](https://arxiv.org/html/2603.11021), [PV-Tuning](https://arxiv.org/html/2405.14852), [QTIP](https://arxiv.org/html/2406.11235), [Leech unfolding](https://arxiv.org/html/2609.02652), [ParetoQ](https://arxiv.org/html/2502.02631), [BTC-LLM](https://arxiv.org/html/2506.12040v1).

**On the one same-harness chart, trellis codes match GGUF quality with about 0.45 fewer bpw near 2 to 2.7 bpw** on 8B: EXL3 at 2.25 bpw scores 9.62 against IQ2_M's 9.93 at about 2.7 ([exllamav3 chart](https://github.com/turboderp-org/exllamav3/blob/master/doc/llama31_8b_instruct_bpw.png)). No published method reaches Q4_K_M accuracy below about 3.5 to 4 bpw. **Compression is therefore worth about 1.2x at equal accuracy against llama.cpp's matched quant.** YAQA, LLVQ, XOR-Trellis and CurveTQ all stop at 2 bpw, and no paper combines output-KL rounding with Leech or rotation-free trellis codes ([XOR-Trellis](https://arxiv.org/html/2610.00432)).

**Kernels are a first-order term.** Measured 2-bit decode runs at 16 percent of the byte floor for AQLM and 26 to 62 percent for trellis, lattice and LUT kernels ([QTIP](https://arxiv.org/html/2406.11235), [Q-Palette](https://arxiv.org/html/2509.20214)). QTIP 2-bit on an RTX 3090 runs at 127 t/s, slower than llama.cpp Q4_0 at 158 t/s on the same GPU model.

Restructuring is weaker still:

- Low-rank decomposition removes only 5 to 10 percent of parameters at under 1 point of loss; LEMS loses 1.5 points at 10 percent on Llama3-8B ([LEMS](https://github.com/lems-svd/lems)).
- CALDERA's low-rank patch on 2-bit weights halves batch-1 speed ([CALDERA](https://arxiv.org/html/2405.18886)).
- Recursive weight tying needs about 60B tokens of uptraining and still loses 3 to 9 points ([Relaxed Recursive Transformers](https://arxiv.org/html/2410.20672)).
- Halving width costs about 7 MMLU points after 94B distillation tokens ([Minitron](https://developer.nvidia.com/blog/how-to-prune-and-distill-llama-3-1-8b-to-an-nvidia-llama-3-1-minitron-4b-model/)).

### Decoupling cost from weight bytes works only where weights are shared or rows are free

| Direction | Best published number | Fit for any checkpoint with light adaptation on this card |
|---|---|---|
| WaveFront Decoding on looped LMs | 2.42x Ouro-2.6B; 3.54x and 4.81x Huginn-3.5B, training-free | no: checkpoint-to-looped conversion exists only at 1B, with 26-52B tokens |
| Hogwild! (workers sharing one KV cache) | 4 workers 68.9 vs 20.1 t/s aggregate, per-forward latency +17% (L40S) | reasoning models only; time-to-answer gain unreported |
| Persistent megakernels | 1.55x (RTX 3090, 0.8B), 2.1x (RTX 5080, 0.8B), 1.07x serving Qwen3-8B (H100) | yes, as engineering |
| Memory layers, UltraMem | up to 6x faster than MoE at equal parameters | no: trained from scratch, one 0.5B retrofit |
| PQ or LUT matmul (MADDNESS) | lookup table at least as large as 2-bit weights unless activations use under 0.25 bit per dimension | no: loses to 2-bit weights on bytes |
| SeedLM | FPGA 3.98x of a 4x ideal at 3-4 bpw | no GPU kernel; no better than 3-4-bit quants on bytes |
| LoRA-Drop temporal reuse | up to 2.6x within 0.5 points | untested with speculation |
| GRIFFIN, IFPruning | 1.25-1.29x training-free; 41% lower decode time trained | GRIFFIN yes |
| Semantic parallelism (ThreadWeaver, Parason, PASTA) | 1.14x measured wall clock; about 1.7x; 1.21-1.93x | no: SFT plus RL per model |
| PCIe weight streaming | about 36x slower than VRAM | capacity only |

Sources: [WaveFront Decoding](https://arxiv.org/abs/2609.23033), [retrofitted recurrence](https://arxiv.org/html/2511.07384v1), [Hogwild!](https://arxiv.org/html/2504.06261v4), [lucebox-hub](https://github.com/Luce-Org/lucebox-hub), [megakernels](https://github.com/SamuelReeder/megakernels), [ForgeMegakernel](https://arxiv.org/html/2609.12379v1), [UltraMem](https://arxiv.org/abs/2411.12364), [memory retrofit](https://arxiv.org/html/2604.05248), [MADDNESS](https://proceedings.mlr.press/v139/blalock21a/blalock21a.pdf), [SeedLM](https://arxiv.org/html/2410.10714), [LoRA-Drop](https://arxiv.org/abs/2601.02569), [GRIFFIN](https://arxiv.org/html/2404.01365v3), [IFPruning](https://arxiv.org/pdf/2501.02086), [ThreadWeaver](https://arxiv.org/pdf/2512.07843), [PASTA](https://arxiv.org/html/2502.11517v2).

The pattern is consistent. **The only directions with large single-device gains turn one weight read into several useful rows.** WaveFront turns depth into batch through weight sharing; Hogwild! runs parallel workers. Both draw on the same spare-row budget as speculative verification, so on GA102 they compete with it instead of multiplying.

With only 6 MB of L2 on GA102 ([GA102 whitepaper](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf)), weight residency in cache is irrelevant.

**The notes disagree on what a megakernel is worth at 7 to 8B.** One estimate puts it at 1.2 to 1.4x over llama.cpp. The measured serving-level gain is 1.03 to 1.28x, and 1.07x for Qwen3-8B (5,601 vs 6,009 us per token) ([ForgeMegakernel](https://arxiv.org/html/2609.12379v1)).

### KV compression is worth nothing at 4K and up to 1.9x at 32K

With Q4_K_M weights and f16 KV, KV is 9.8 percent of Llama 3.1 8B's per-token bytes at 4K, 46.6 percent at 32K and 77.7 percent at 128K (computed from configs).

| Method | Bits or budget | Accuracy | Decode evidence |
|---|---|---|---|
| TurboQuant | 3.5 bits per channel | quality-neutral; 2.5 bits marginal | 8x attention-logit speedup on H100 (press relay) |
| InnerQ Base / Small | 3.5 / 3.0 effective bits | GSM8K -0.36 / -4.73 | about 2.7x over fp16 per layer at batch 1 (Jetson Xavier NX) |
| AQUA-KV | 2-2.5 bits | under 1% relative error | needs 1-6 h of calibration |
| RetroInfer | 1.8% cluster budget | RULER-128K -0.73% (Llama 3.1 8B) | 3.6x decode throughput (A100) |
| UNIQUE | 1.6% / 6.3% of a 32K context | RULER-32K 90.78 / 92.53 vs 93.15 | 5.3x end to end (H100, batch 80) |
| DMS | 8x compression | beats TOVA and H2O | about 1K training steps |
| SparseSpec-L | self-draft over 10% of KV | lossless | 11.26 accepted per step; 1.10x at batch 1 and 16K (A40) |
| Dustin | 512-token sparse verify | LongBench -0.58 (lossy) | 6.61-9.17x (H200, batch 8-16) |
| llama.cpp q8_0 / q4_0 KV (baseline) | 8.5 / 4.5 bits | n/a | 25.0 / 24.0 vs 38.0 t/s f16 at about 110K (GB10) |

Sources: [TurboQuant](https://arxiv.org/abs/2504.19874), [Tom's Hardware](https://tomshardware.com/tech-industry/artificial-intelligence/googles-turboquant-compresses-llm-kv-caches-to-3-bits-with-no-accuracy-loss), [InnerQ](https://arxiv.org/html/2602.23200v2), [AQUA-KV](https://arxiv.org/abs/2501.19392), [RetroInfer](https://arxiv.org/html/2505.02922v2), [UNIQUE](https://arxiv.org/html/2605.27740v1), [DMS](https://arxiv.org/abs/2506.05345v2), [SparseSpec-L](https://arxiv.org/html/2607.27735v1), [Dustin](https://arxiv.org/html/2606.24957), [Discussion #20969](https://github.com/ggml-org/llama.cpp/discussions/20969).

**About 4 effective bits is lossless without training.** Keys are the sensitive half; InnerQ keeps keys at 3 bits in every variant. Page-sparse reads at 1.6 to 6.3 percent cost under 1 point of RULER only when the selector uses richer page statistics: at the same budget, Quest falls to 84.14 and H2O to 56.00 ([UNIQUE](https://arxiv.org/html/2605.27740v1)).

Verify rows share one read of the KV history, so KV compression multiplies with speculation in principle. llama.cpp's quantized KV decodes slower than f16 at long depth, which leaves this lever open to a new runtime. **The notes disagree on Quest's speedups**: the abstract's ordering of its 2.23x and 7.03x figures is self-contradictory ([Quest](https://arxiv.org/abs/2406.10774)).

### MoE gains come from changing routes, and verify unions eat speculation

| Lever (Qwen3-30B-A3B unless noted) | Best published point | Accuracy | Speed evidence |
|---|---|---|---|
| Training-free expert skipping (ACE) | 20% of routed slots | 80.25 vs 80.67 | 1.31-1.41x decode at higher skip rates (A100) |
| Self-distilled zero experts (ZEDA) | 51.2% of slots zeroed | 74.2 vs 74.9; LCB v6 53.2 vs 57.1 | 1.19-1.25x (H200); 30 h on 8 H200 |
| Top-k halving with k2 renormalization | 8 to 4 experts (Qwen3.6-35B-A3B) | MMLU -0.35 | not reported |
| One-shot pruning (REAP / PRIME, 25%) | 96 of 128 experts | MC -7.8% relative (REAP); 70.46 vs 71.88 (PRIME) | active bytes unchanged |
| Per-expert mixed precision (DuoRate) | 2.9 bpw read | 76.16 vs 77.14 | 656.8 MB per token, computed; speed unmeasured |
| Route prediction (SeqMoE) | 84.6-97.0% hit rate at 15-45% residency | n/a | 84.82% of full-load speed at 45% residency (RTX 4090) |
| Prefetch-steered routing (PSR) | OLMoE at 50% residency | -0.03 points | 2.3x over exact LRU; only about 1.25x on Qwen3.6-35B-A3B |
| Verify union | 19.5 unique experts for 4 tokens vs 29.1 uniform | n/a | batch-1 target efficiency 0.860 at K=3 |
| Residency-restricted verify (AcceptMoE) | cached experts only | -0.27 points | 2.06x under offloading |
| Self-draft with fewer experts (S2-MoE) | 3 of 8 experts | lossless | 1.2-2.9x (RTX 4090, offloading) |

Sources: [ACE](https://arxiv.org/html/2609.05228), [ZEDA](https://arxiv.org/html/2605.18643), [k2 renormalization](https://arxiv.org/abs/2609.04575), [REAP](https://arxiv.org/html/2510.13999v3), [PRIME](https://arxiv.org/html/2610.13982v1), [DuoRate](https://arxiv.org/html/2610.13328), [SeqMoE](https://arxiv.org/html/2609.12978), [PSR](https://arxiv.org/html/2610.34721), [union study](https://arxiv.org/html/2610.38981), [AcceptMoE](https://arxiv.org/abs/2608.02989), [S2-MoE](https://arxiv.org/html/2608.15018v2).

**At 2-bit experts, Qwen3-30B-A3B's non-expert bytes dominate.** Per token it reads 453 MB of experts against 510 MB of Q4 attention and 255 MB of LM head, so experts are about 37 percent of bytes at short context. No MoE paper quantizes attention or the head below 4 bits at under 1 percent loss.

**Under-1-percent PTQ needs 2.9 to 4.3 bits per weight read** ([DuoRate](https://arxiv.org/html/2610.13328)). The 2-bit settings that fit this card lose 2.5 to 6.5 points with the best allocators, and uniform GPTQ at 2.0 bits loses 12.3 points ([ProQuant](https://arxiv.org/html/2610.30779)).

**The notes disagree on MoE speculation headroom.** An oracle analysis of Qwen3-Coder-30B-A3B with EAGLE-3 plateaus at 2.34x with a mean accepted length near 2.1 ([arXiv 2609.22156](https://arxiv.org/html/2609.22156v1)). Against that, routing overlap gives Qwen3-30B a batch-1 target efficiency of 0.860 ([arXiv 2610.38981](https://arxiv.org/html/2610.38981)), and DFlare reaches 3.91x on GPT-OSS-20B in a datacenter setting ([DFlare](https://arxiv.org/abs/2606.02091)).

## Measured stacks deliver 54 to 69 percent of the multiplicative prediction

| Pairing | Measured | Multiplicative prediction | Fraction realized | Source |
|---|---|---|---|---|
| Q4_K_M x DSpark, Qwen3-8B, RTX 4090 | 4.8x over bf16 AR | 2.66 x 3.35 = 8.9x | 54% | [PR #25173](https://github.com/ggml-org/llama.cpp/pull/25173) |
| Q4_K_M x EAGLE-3, Llama-3.1-8B, A6000 | 6.2x | 2.73 x 3.28 = 8.9x | 69% | [PR #18039](https://github.com/ggml-org/llama.cpp/pull/18039) |
| Chunk sparsity x EAGLE-2, BlockFFN, Jetson Orin NX | 3.67x | 3.14 x 1.74 = 5.46x | 67% | [BlockFFN](https://arxiv.org/pdf/2507.08771) |
| W4A16 x EAGLE-2 tree, 8B, RTX 3090 | no gain over either alone | n/a | at most 50% | [SpecMQuant](https://arxiv.org/html/2505.22179v1) |
| W4A4 draft x W4A16 verify (QSpec), batch 1 | 1.29-1.36x | n/a | n/a | [QSpec](https://i.cs.hku.hk/~cwu/papers/jtzhao-emnlp25.pdf) |
| Sparse-KV self-speculation at batch 1, 16K, A40 | 1.10x (2.45x at batch 4) | n/a | n/a | [SparseSpec-L](https://arxiv.org/html/2607.27735v1) |
| KV quantization added to prune plus quant, GGUF | decode 0.61-0.63x | n/a | n/a | [MoE-XBench](https://arxiv.org/html/2608.21693) |
| Accuracy: expert pruning x Q4_K_M, Qwen3-30B | -0.0528 | -0.0449 if additive | 18% super-additive | [MoE-XBench](https://arxiv.org/html/2608.21693) |
| Accuracy: R-Sparse 50% x INT4, Llama-2-7B | -2.34 | -1.38 if additive | 70% super-additive | [R-Sparse](https://arxiv.org/html/2504.19449v1) |

Every measured combination is sub-multiplicative. The losses come from three per-row taxes:

- **Dequantization work per verified row.** SpecMQuant's W4A16 verify costs 1.8 decode steps at tree size 60, against below 1.2 at FP16.
- **Union growth.** Sparse channels and routed experts accumulate across verify rows.
- **Drafter overhead.** A fixed-size drafter becomes a larger share of a target pass that compression has shrunk.

Target quantization does not reduce acceptance. SpecMQuant finds nearly no change in accepted length at W4A16 and W8A8, and llama.cpp's EAGLE-3 acceptance at Q4_K_M (77.2 to 92.5 percent) is no lower than at bf16.

**The notes disagree on whether the drafter can be quantized.** QSpec reports that GPTQ on an EAGLE draft substantially degrades acceptance ([QSpec](https://i.cs.hku.hk/~cwu/papers/jtzhao-emnlp25.pdf)). An MLX EAGLE-3 prototype found a 4-bit draft kept identical acceptance (0.34) at half the cost ([mlx-lm #890](https://github.com/ml-explore/mlx-lm/discussions/890)).

Verify-side approximations also spend the accuracy budget. Skipping layers during verification raised a translation benchmark's BLEU from 0.131 to 0.237, so the outputs are no longer the target's ([TSS](https://arxiv.org/abs/2609.26100)). Sparse verification that prunes two thirds of FFN activations keeps acceptance length largely stable ([arXiv 2512.21911](https://www.alphaxiv.org/abs/2512.21911)).

**Accuracy has to be measured on task suites, not perplexity.** MoE-XBench shows perplexity losses adding up while task losses compound.

## One stack closes 5x on paper: math and greedy chat at short context, everything past 13K

This is the only combination of published results whose arithmetic reaches 5x over the per-domain bars above. Each factor is a published number or arithmetic on one. The drafter is the only component llama.cpp already has a version of. The rest of the multiplier is runtime physics that llama.cpp does not implement.

| Factor | Value used | Basis |
|---|---|---|
| Bandwidth efficiency vs llama.cpp | 0.85 / 0.74 = 1.15x | llama.cpp runs Qwen3-8B Q4_K_M at 72-74% of floor (computed from [PR #25173](https://github.com/ggml-org/llama.cpp/pull/25173)); megakernels reach 74-86% MBU on H100 ([ForgeMegakernel](https://arxiv.org/html/2609.12379v1)) |
| Bytes per pass at equal accuracy | 4.83 / 4.03 GB = 1.20x | 4.0-bpw trellis beats Q4_K_M perplexity on one harness, 7.729 vs 7.810 ([exllamav3](https://github.com/turboderp-org/exllamav3/blob/master/doc/llama31_8b_instruct_bpw.png)); 4-bit KV is lossless ([TurboQuant](https://arxiv.org/abs/2504.19874), [InnerQ](https://arxiv.org/html/2602.23200v2)) |
| TEAL-class sparsity inside verify | 1.10x | 25% sparsity costs -0.34 points; about half survives a 16-row union ([TEAL](https://arxiv.org/html/2408.14690v3)) |
| Drafter tau at block 16, greedy | DFlare: GSM8K 7.86, MATH-500 9.17, HumanEval 7.07, MBPP 6.77, MT-Bench 5.00 | DBloom's remeasurement ([DBloom](https://arxiv.org/pdf/2608.30427)) |
| Round cost, verify plus draft | 1.15 + 0.13 = 1.28 steps | FLUTE's 16-row cost of 1.14 steps on a 4090 ([HIGGS](https://arxiv.org/html/2411.17525v1)); a 5-layer drafter at 4 bits, about 15% of target parameters |

These factors yield 165 verify rounds per second (912 GB/s x 0.85 / 4.03 GB x 1.10 / 1.28). Tokens per second is that rate times tau:

| Domain, greedy, 1K context | Stack t/s | llama.cpp bar t/s | Ratio | Verdict |
|---|---|---|---|---|
| MATH-500 | 1,516 | 290 | 5.2x | holds |
| MT-Bench | 827 | 156 | 5.3x | holds, on an estimated bar |
| MBPP | 1,119 | 246 | 4.5x | fails |
| GSM8K | 1,299 | 308 | 4.2x | fails |
| HumanEval | 1,169 | 311 | 3.8x | fails |

**Long context closes the rest.** llama.cpp decodes fastest with f16 KV at long depth (its q8_0 path is slower), so its bytes per pass grow by 0.147 GB per 1K tokens of Qwen3-8B context. With 4-bit KV read through about 6.3 percent of pages, the stack's bytes barely move. The added advantage over the 1K case is computed:

| Context | Extra factor over 1K |
|---|---|
| 4.5K | 1.10x |
| 7.5K | 1.19x |
| 12.6K | 1.33x |
| 32K | 1.87x |

The extra factor closes MBPP at about 4.5K, GSM8K at about 7.5K and HumanEval at about 12.6K. At 32K the five domains reach 7.0 to 9.9x. Dense attention at 32K is borderline. Sixteen verify rows of attention cost about 309 GFLOP: about 3.8 ms at FP16 accumulate or about 7.5 ms at FP32 accumulate, against a 6.7 ms weight plus 4-bit KV read (computed). Whether it reaches the critical path depends on accumulate precision. The page-sparse read keeps attention off the critical path and supplies the byte advantage in the table above. It costs 0.62 RULER-32K points at a 6.3 percent budget ([UNIQUE](https://arxiv.org/html/2605.27740v1)).

**Where the stack fails:**

- **Temperature 1.** DFlash's tau falls 16 percent and its MT-Bench speedup 10 percent. If llama.cpp's gain held at its greedy level, MATH-500 would drop to about 4.4x and MT-Bench to 4.8x. The PR tables are greedy only, so the temperature-1 ratio is unknown.
- **Thinking mode.** Block-drafter acceptance collapses to 12 percent ([PR #22105](https://github.com/ggml-org/llama.cpp/pull/22105)).
- **The reader's current verify cost.** At 1.9 steps for 16 rows, every ratio shrinks by 0.63 (MATH-500 3.3x).
- **A bf16 drafter.** At about 0.5 steps of drafting cost, every ratio shrinks by 0.78.
- **DBloom's block 24, under the measured issue cost.** At 272 TOPS block 24 is not FLOP-bound, so the verdict rests on the reader's measured issue cost. From 16 to 24 rows that cost rises 1.44x (1.9 to 2.73 steps), while DBloom's 24-row tau rises only 1.15x on MATH-500 (9.17 to 10.54). With 0.13 steps of drafting, tau per step falls by 17 to 24 percent:
  - MATH-500: 4.52 to 3.69
  - GSM8K: 3.87 to 3.23
  - HumanEval: 3.48 to 2.77
  - MT-Bench: 2.46 to 1.88

  Going from 24 to 32 rows adds only 0.19 steps (2.73 to 2.92). A kernel that held 24 rows near the floor (about 1.2 steps) would reverse the verdict: MATH-500 tau per step would rise from 7.16 to 7.92 (computed).
- **Accuracy is equal only on paper.** The claim nets the trellis code's 1.0 percent perplexity surplus against TEAL's -0.34 points and 4-bit KV's -0.36 GSM8K. Measured compounding is 18 to 70 percent super-additive.
- **Long-context tau is unmeasured.** No paper reports tau for DFlare-class drafters at 8K to 32K. SparseSpec-L's 8.03 to 11.26 accepted tokens per step on long inputs is the only nearby evidence ([SparseSpec-L](https://arxiv.org/html/2607.27735v1)).

**MoE variant.** Multiplying the reader's 112 to 129 t/s by S2-MoE's lossless 1.2 to 2.9x gives 134 to 374 t/s. That clears the 260 to 275 t/s reference bar only at the top of S2-MoE's range, which on Jetson was HumanEval at 2.60x ([S2-MoE](https://arxiv.org/html/2608.15018v2)). PSR plus a block drafter reports 2.1 to 2.9x over the fastest AR configuration at a small accuracy cost ([PSR](https://arxiv.org/html/2610.34721)). Against a resident-IQ2 llama.cpp bar of 350 to 475 t/s, nothing closes: 5x would need about twice the 931 t/s floor, while verify unions grow 2.4 unique experts per verified token.

**1B class.** Megakernels give 1.55 to 2.1x over llama.cpp at 0.8B, and 5x over plain llama.cpp sits below the 1,095 t/s floor. Compression and wide blocks are the wrong tools there. Llama 3.2 1B at IQ2_XXS degrades to 3.7x its 8-bpw perplexity ([exllamav3 1B chart](https://github.com/turboderp-org/exllamav3/blob/master/doc/llama32_1b_instruct_bpw.png)), and SDAR found 1.7B and 4B models degrade above block 4 ([SDAR](https://arxiv.org/html/2510.06303v1)).

## The verify wall is the thinnest and richest research gap

The ranking below orders gaps by the multiplier closing each would add on this card at equal accuracy with light adaptation. Looped conversion has the largest raw multiplier but ranks low because it needs pretraining-scale conversion.

| Rank | Gap | What exists | Multiplier if closed (computed) |
|---|---|---|---|
| 1 | A sub-4-bit weight format whose per-byte decode is cheap enough for a multi-row kernel to sit at the bandwidth floor at 16 rows, with Q4_K_M accuracy and a compressed LM head | no paper measures the 1-to-16-row scaling curve for any IQ, VQ or trellis code on any GPU; the closest evidence is two endpoints (batch 1 and 16) for HIGGS grids through FLUTE's lookup kernel on a 4090 (1.14 steps); SpecMQuant's RTX 3090 null on W4A16 trees; 2-bit kernels at 26-62% of the floor even at one row; nothing within 3% of FP16 perplexity below about 3.5 bpw; YAQA, LLVQ, XOR-Trellis and CurveTQ stop at 2 bpw and no paper stacks them; heads left at FP16 | 1.6x from verify cost at 4 bpw (16-row round from 2.03 to 1.28 steps); about 1.75x if the kernel also holds 24 rows near the floor so DBloom's block 24 pays. About 1.5x more if the same property holds for a 2.5-bpw format at Q4_K_M accuracy (1.60x fewer bytes, minus a 4-bit drafter that grows from 0.13 to 0.21 of a pass). About 2.4x combined. Every stack depends on it |
| 2 | Lossless acceptance at low-margin positions in prose and at temperature 1 | chat tau 45-60% of math tau in every method; DBloom's ceiling bin empties on chat; DSpark truncation lifts chat acceptance from 45.7% to 95.7%, so survival is predictable; relaxed verify cut AIME25 from 76.7 to 60.0 | 1.5-2x on chat and temperature 1 (tau from 5 to 8-10), where llama.cpp's bar is lowest |
| 3 | KV formats built for multi-row reuse, page selection shared across verify rows and layers, lossless sparse verify with a bound on omitted softmax mass | no measured bs1 consumer result; llama.cpp quantized KV slower than f16; no fused 2-3-bit plus page-sparse kernel; Dustin's sparse verify is lossy | 1.2x at 8K, 1.4x at 16K, 1.9x at 32K; it also keeps 32K attention off the verify critical path |
| 4 | Quantization-aware drafters trained on the quantized target, and drafters that transfer across checkpoints | no tau data for 2026 drafters on 3-4-bit targets; QSpec and MLX disagree on quantized drafts; every drafter trained per target on 0.8-2.4M samples; training-free ESP stops at tau 1.71 | about 1.3x (drafting cost from about 0.5 to 0.13 steps), plus coverage of arbitrary checkpoints. Self-distillation on target outputs added 0.14-0.41 accepted length in post-hoc MTP heads |
| 5 | One spare-row budget shared by drafting and parallel reasoning branches | Hogwild!, TiDAR and Speculative Pipeline Decoding each use rows alone; no co-scheduler | 1.3-2x on long reasoning, where block drafters collapse |
| 6 | MoE verify-batch routing and non-expert bytes | largest single-sequence union measured is 4 tokens; OEA halves the expert union only across independent sequences; no sub-4-bit attention or head study for MoE | 1.5-2x on MoE speculation; 1.36x from re-quantizing Qwen3-30B-A3B attention to 2.5 bpw and its head to 4 bits |
| 7 | Activation sparsity inside verify on low-bit weights | no NVIDIA quantized sparse kernel; union density vs draft length unmeasured on dense SwiGLU | 1.15-1.4x, plus fewer decoded bytes and less issue work per verify pass |
| 8 | Temporal activation reuse inside verify | LoRA-Drop alone; its training requirement is not stated | 1.3-2.6x, unmeasured with speculation |
| 9 | Checkpoint-to-looped conversion plus wavefront decoding | wavefront decoding works only on natively looped models; retrofits exist only at 1B | 2.4-4.8x raw, but fails light adaptation and equal accuracy today |

Sources for the table: [SpecMQuant](https://arxiv.org/html/2505.22179v1), [HIGGS](https://arxiv.org/html/2411.17525v1), [DBloom](https://arxiv.org/pdf/2608.30427), [DSpark](https://arxiv.org/html/2607.05147), [post-hoc MTP](https://arxiv.org/html/2610.00888v1), [Discussion #20969](https://github.com/ggml-org/llama.cpp/discussions/20969), [Dustin](https://arxiv.org/html/2606.24957), [YAQA](https://arxiv.org/html/2505.22988), [Leech unfolding](https://arxiv.org/html/2609.02652), [QSpec](https://i.cs.hku.hk/~cwu/papers/jtzhao-emnlp25.pdf), [ESP](https://arxiv.org/html/2603.17942v1), [Hogwild!](https://arxiv.org/html/2504.06261v4), [OEA](https://arxiv.org/html/2511.02237), [Prox](https://arxiv.org/html/2607.27591), [LoRA-Drop](https://arxiv.org/abs/2601.02569), [WaveFront Decoding](https://arxiv.org/abs/2609.23033).

**Gap 1 is the thinnest area in the literature.** The verify-cost problem and the sub-4-bit accuracy problem are one problem: a format is only worth its bytes if a multi-row kernel can decode it at the bandwidth floor. No paper measures 1-to-16-row scaling for any IQ, VQ or trellis code on any GPU.

**The first four gaps share one root.** Datacenter drafter research maximizes tau at budgets that cost nothing on an H200. On a GA102 the quantity that matters is accepted tokens per decoded weight byte and per unit of decode and issue work in a multi-row kernel, and almost nothing published optimizes it.

The reader is unusually well placed on the top gaps:

- Their tree-verify runtime, with its documented issue-bound costs, is the diagnostic for gap 1.
- Their on-device training kernels can train a drafter against the quantized target's own outputs (gaps 2 and 4).
- The light fine-tuning budgets in published 2-bit work (52M tokens for LLVQ's scale tuning, 1 to 30 GPU-hours for YAQA) are the right scale for the accuracy half of gap 1.

Several directions rank below the table:

- **Hybrid diffusion with 256-token canvases.** One 256-row step costs 256 x 15 GOP = 3.84 TOP, about 24 ms at 160 TOPS. At about 20 tokens per step that is roughly 830 t/s for an 8B dense model (computed): near the chat bar but below the math and code bars, and only after a conversion that cost DiffusionGemma up to 15 AIME points.
- **LUT matmul.** It loses to 2-bit weights on bytes.
- **Seed-generated weights.** Bytes are no better than 3 to 4-bit quants.
- **L2 residency.** Irrelevant on 6 MB.
- **Staggered layers on one GPU.** Latency hiding only, 1.1 to 1.3x.
- **Memory-layer retrofits.** Shown only at 0.5B.
- **PCIe streaming for a model that already fits.** About 36x slower than VRAM.

## Where the notes disagree on numbers

| Quantity | One figure | Other figure |
|---|---|---|
| llama.cpp 8B Q4_K_M on the 3080 Ti | 110.6 t/s measured, Llama 3 8B, 2024 ([XiongjieDai](https://github.com/XiongjieDai/GPU-Benchmarks-on-LLM-Inference)) | 125-137 t/s scaled from the 3090; 141-144 t/s scaled from the 4090 and A6000 |
| Q4_K_M 8B floor on the 3080 Ti | 185 t/s from the 4.92 GB file | 192 t/s with 1K of KV; 197 t/s weights only |
| llama.cpp bandwidth efficiency | 63-74% of floor, dense 4-bit, consumer GPUs | 50.5% MBU, Qwen3-4B, H100 BF16 ([ForgeMegakernel](https://arxiv.org/html/2609.12379v1)) |
| Megakernel gain at 7-8B | 1.2-1.4x estimated | 1.03-1.28x serving-level, 1.07x for Qwen3-8B |
| TEAL speedup at 40 / 50% | 1.53x / 1.8x headline (Llama-2) | 1.42x / 1.63x Llama-3-8B on A6000, 1.25x / 1.33x on A100 ([TEAL](https://arxiv.org/html/2408.14690v3)) |
| FlexiDepth retention | about 100% at 8 skipped layers | GSM8K -3.64 points, BBH 68.04 to 59.65 ([vSkipper](https://arxiv.org/html/2609.37062v1)) |
| GRIFFIN at 50% | near dense on Llama-2-7B ([Polar Sparsity](https://arxiv.org/pdf/2505.14884)) | 45.11 vs 69.44 on Llama-3-8B ([R-Sparse](https://arxiv.org/html/2504.19449v1)) |
| ReLU-fication on Llama-2-7B | near dense (Polar Sparsity) | 35.34 vs 65.88 (R-Sparse) |
| CATS | within 1-2% at 50% MLP sparsity (own) | perplexity 6.78 vs TEAL 5.94 at 25% model-wide sparsity |
| SWIFT | 1.3-1.6x (authors) | 1.07-1.17x (independent ICML 2026 study) |
| CLaSp | 1.3-1.7x (authors); 1.24x greedy in its own table ([CLaSp](https://arxiv.org/html/2505.24196v1)) | 1.08x non-greedy (independent review) |
| Quantized draft acceptance | GPTQ draft degrades acceptance (QSpec) | 4-bit draft acceptance unchanged at 0.34 (MLX) |
| MoE speculation headroom | 2.34x oracle plateau, accepted length about 2.1 | 0.860 batch-1 efficiency; DFlare 3.91x on GPT-OSS-20B |
| MoE-SpeQ speedup | 3.3x (one review) | 2.34x (another summary) ([arXiv 2511.14102](https://arxiv.org/abs/2511.14102)) |
| DeepSeek-V3 MTP second-token acceptance | 85-90% ([report](https://arxiv.org/pdf/2412.19437)) | 80-90% ([follow-up](https://arxiv.org/pdf/2505.09343)) |
| Entropy vs acceptance | lowest-entropy code bounds tau highest | code's lower entropy did not raise acceptance |
| Quest | 2.23x attention, 7.03x latency (abstract order) | consistent reading: 7.03x attention, 2.23x end to end |
| TransMLA | 93% KV compression, 10.6x at 8K (earlier version) | 68.75% training-free at -1.65% (later version) ([TransMLA](https://arxiv.org/html/2502.07864v4)) |
| RetroInfer on LongBench | at most 0.23% drop (text) | 35.74 vs 36.23 for Qwen2.5-7B (table) |
| MoDES | 97.33% retention at 88% skipping, VL model | 56.61 at 50% on text Qwen3-30B (ACE) |
| 5x target for 8B on the 3080 Ti | 1,200-1,600 t/s; 1,250-1,400 t/s | 625-685 t/s before drafting; 10-30 tokens per forward on prose. Same arithmetic, different llama.cpp baselines |

## Conclusion

On this card the binding constraint is the per-byte decode and issue cost of multi-row verification over low-bit weight formats. No published work measures or optimizes it, and bandwidth remains the floor it must reach. Tensor peak is not the limit: at INT8 rates the FLOP ridge sits near 55 to 66 rows at 4 bits. The datacenter literature optimizes tau at budgets that are free on an H200, which is why its 5x-plus headlines do not transfer. Under the reader's measured issue cost, block 16 beats block 24, and a kernel that held 24 rows near the floor would reverse that. Large trees lose on FLOPs even at INT8. Compression narrows the usable verify window only to the extent that a format spends more decode and issue work per byte, which is why codebook and trellis codes are where the problem is hardest. The same physics says where 5x is even definable. llama.cpp's drafter already sits near the block horizon on code-only and verbatim output, so 5x there is impossible. Its drafter roughly halves at Q4_K_M and nearly vanishes on chat, which is where an equal-accuracy 5x is reachable.

The published stack that closes 5x draws its multiplier mainly from verify cost, bytes, bandwidth efficiency and KV, which are runtime physics llama.cpp does not implement, not from a better drafter. The original contribution most likely to deliver the missing factor is therefore one no paper evaluates: a low-bit weight format and kernel designed for 16-row verification, paired with a 4-bit drafter trained against that quantized target on the device itself. Long-context KV reuse would extend the 5x from math and greedy chat to every domain. The reader's own 1.9-step measurement is the first number such work would move.

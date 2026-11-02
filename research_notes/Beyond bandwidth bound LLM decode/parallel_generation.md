# Parallel generation from one forward pass of a pretrained AR transformer (survey through 2026-11-01)

Units used throughout. TPF = tokens committed per target forward pass (also NFE reduction). tau = mean accepted length per verify round including the bonus token (speculative-style methods). "Lossless" = output distribution identical to the base model under the stated sampling rule. Unless stated, speed numbers are batch 1 on datacenter GPUs (A100, H100, H200, B200) in BF16 or FP8. No paper surveyed reports measurements on an RTX 3080 Ti or any 12 GB Ampere GeForce card except SpecMQuant (RTX 3090) and one community fork report (RTX 3090).

## 1. Block-diffusion conversion of AR checkpoints (SDAR, D2F, Dream, DiffuLLaMA, DiffuCoder, Fast-dLLM v1/v2, dKV-Cache, BD3-LM, Seed Diffusion, Mercury, Gemini Diffusion, LLaDA 2.0, TiDAR, I-DLM, DiffusionGemma, newer)

### Takeaway
Converting an AR checkpoint to block diffusion costs 1B to 150B tokens of full-weight continued training, and none of the published conversions leaves the base weights untouched. In the 1B to 8B class the realized batch-1 gain over the AR base at near-equal quality is about 2.5x to 6x: Fast-dLLM v2 gets 2.5x, I-DLM 3.1x and TiDAR 5.9x. Every one of them gives up some accuracy versus its own AR parent, typically 1 to 5 points and up to 15 points on hard reasoning. The highest TPF reported anywhere is about 20 (DiffusionGemma, 7.1x over AR on an H100), and it comes with drops as large as -15 AIME points.

### Cited Findings

**Conversion cost, weights, speed and quality per method**

| Method (venue) | AR base | Conversion data / compute | Base weights | TPF / speed | Quality vs AR parent | Source |
|---|---|---|---|---|---|---|
| SDAR (ACL Findings 2026) | Qwen3 1.7B, 4B, 8B, 30B-A3B | 50B-token continued pretraining at 4k ctx + 4B-token SFT; 20B to 40B tokens gave "slightly lower" results; GPU-hours not reported | Continued training initialized from AR weights; paper does not state whether every parameter is updated | TPF rises with block size and model size but numeric TPF appears only in figures; no wall-clock numbers reported | SDAR-Chat minus AR-Chat: 1.7B GSM8K -1.0, MATH500 +1.2, HumanEval -4.3, MBPP -0.8; 8B GSM8K -1.5, MATH500 +0.2, HumanEval -2.0, MBPP -3.1; 8B LMB-Hard -6.8; 1.7B AIME-25 -5.9; small models (1.7B, 4B) degrade for block size > 4 | [arXiv 2510.06303](https://arxiv.org/html/2510.06303v1), [ACL Anthology](https://aclanthology.org/2026.findings-acl.1110/) |
| Fast-dLLM v2 (2025) | Qwen2.5-1.5B, Qwen2.5-7B (Instruct) | about 1B tokens; batch 256; 1.5B: 6,000 steps, 64 A100 x 8 h (512 A100-h); 7B: 2,500 steps, 64 A100 x 12 h (768 A100-h) | Fine-tuned with block-diffusion objective | Threshold 0.9 on GSM8K: 39.1 to 101.7 tok/s (2.6x vs its own threshold 1.0); 2.54x throughput vs Qwen2.5-7B-Instruct on A100; batch 1 102.5 tok/s, batch 4 217.5 tok/s; block 32, sub-block 8 | 7B average 60.3 vs 59.6 for an NTP fine-tune on same data, but vs Qwen2.5-7B row: MATH 61.6 vs 73.3, IFEval 61.4 vs 70.8, GSM8K 83.7 vs 71.4, HumanEval 63.4 vs 51.2 (table evaluated with parallel decoding disabled) | [arXiv 2509.26328](https://arxiv.org/pdf/2509.26328), [project page](https://nvlabs.github.io/Fast-dLLM/v2/), [HF card](https://huggingface.co/Efficient-Large-Model/Fast_dLLM_v2_7B) |
| Dream 7B (2025) | Qwen2.5-7B | 580B tokens | Continued pretraining from Qwen2.5 weights | No AR-relative speed given; in Jacobi Forcing's A100 comparison Dream-Base runs at 0.23x to 0.49x of AR | Base: GSM8K 77.2 vs 78.9, MATH 39.6 vs 41.1, HumanEval 57.9 vs 56.7, MBPP 56.2 vs 63.6, MMLU 69.5 vs 71.9 | [arXiv 2508.15487](https://arxiv.org/html/2508.15487v1), [Jacobi Forcing Table 1/2](https://arxiv.org/pdf/2512.14681) |
| DiffuLLaMA (ICLR 2025) | GPT-2 and LLaMA, 127M to 7B | under 200B tokens | Adapted by continued training | No speed claims in abstract | "Competitive with AR counterparts" (no numbers in abstract) | [arXiv 2410.17891](https://arxiv.org/abs/2410.17891) |
| DiffuCoder (Apple, 2025) | 7B code model | 130B code tokens | Trained 7B dLLM (base not named on abstract) | No decoding-speed numbers in abstract | coupled-GRPO adds +4.4% EvalPlus | [arXiv 2506.20639](https://arxiv.org/abs/2506.20639) |
| D2F (ICLR 2026) | Applied to dLLMs (Dream-Base-7B, LLaDA-Instruct-8B), not to AR checkpoints; asymmetric distillation from a bidirectional dLLM teacher, LoRA adapters released | Training cost not stated | Student adapters on dLLM | GSM8K: D2F-Dream 119.9 tok/s vs LLaMA3-Instruct-8B 48.0 tok/s (2.5x), 150.9 tok/s at 71.2 score (3.1x) on A100-PCIe-40GB; >50x vs vanilla dLLMs | Jacobi Forcing's A100 comparison: D2F HumanEval 54.3 at 1.77x vs AR Qwen2.5-Coder-7B 87.8; MATH 35.4 vs 77.0 | [GitHub](https://github.com/SJTU-DENG-Lab/Discrete-Diffusion-Forcing), [ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/04bb76a98d9f32226b3beba7bd26a51f-Abstract-Conference.html), [Jacobi Forcing](https://arxiv.org/pdf/2512.14681) |
| Fast-dLLM v1 (ICLR 2026) | Training-free for existing dLLMs (LLaDA, Dream) | none | unchanged | Up to 27.6x throughput vs vanilla dLLM via block approximate KV cache + confidence-threshold parallel decoding | "minimal accuracy loss" | [arXiv 2505.22618](https://arxiv.org/abs/2505.22618), [project](https://nvlabs.github.io/Fast-dLLM/) |
| dKV-Cache (2025) | Training-free for dLLMs | none | unchanged | 2x to 10x vs vanilla dLLM | Decode variant "almost lossless"; Greedy variant loses accuracy | [arXiv 2505.15781](https://arxiv.org/abs/2505.15781) |
| BD3-LM (ICLR 2025 oral) | Block-diffusion formulation that SDAR, Fast-dLLM v2, DFlash, DFlare build on | n/a | n/a | n/a | n/a | [arXiv 2503.09573](https://arxiv.org/abs/2503.09573), [DBloom related work](https://arxiv.org/pdf/2608.30427) |
| Seed Diffusion Preview (ByteDance, 2025) | Not disclosed | Not disclosed | n/a | 2,146 tok/s on H20 (code) | "Competitive" on code benchmarks; no numbers in abstract | [arXiv 2508.02193](https://arxiv.org/abs/2508.02193) |
| Mercury Coder (Inception, 2025) | Not disclosed | Not disclosed | n/a | Mini 1,109 tok/s, Small 737 tok/s on H100 (Artificial Analysis) | Second on Copilot Arena quality | [arXiv 2506.17298](https://arxiv.org/abs/2506.17298) |
| Gemini Diffusion (Google demo, May 2025) | Not disclosed | Not disclosed | n/a | 1,479 tok/s excluding 0.84 s per-request overhead; independent tester about 857 tok/s | HumanEval 89.6 vs 90.2 and MBPP 76.0 vs 75.8 for Flash-Lite; GPQA Diamond 40.4 vs 56.5 | [Gigazine](https://gigazine.net/gsc_news/en/20250522-google-gemini-diffusion), [Fortune](https://fortune.com/2025/05/21/gemini-diffusion-google-io-sleeper-hit-blazing-speed-ai-model-wars/) |
| LLaDA 2.0 (Ant, Dec 2025) | Converted AR MoE (16B mini, 100B flash with about 6.1B active) | Three-phase block-level schedule; token count not on abstract | Updated by conversion | Flash with CAP: 535 tok/s, 2.1x over Ling-flash-2.0 (256 tok/s); TPF 2.55 at threshold 0.95, up to 3.31 at 0.85 | Claims parity; details in paper | [arXiv 2512.15745](https://arxiv.org/pdf/2512.15745), [alphaXiv](https://www.alphaxiv.org/abs/2512.15745), [HF card](https://huggingface.co/inclusionAI/LLaDA2.0-flash) |
| TiDAR (NVIDIA, MLSys 2026 oral) | Qwen2.5-1.5B, Qwen3-8B | 50B tokens (1.5B), 150B tokens (8B); 2M-token batch, H100s | Continual pretraining from the AR checkpoint | T/NFE 7.45 (1.5B), 8.25 (8B); batch-1 H100 speedup 4.71x (1.5B), 5.91x (8B); draft length 4/8/16 gives 3.47/5.49/6.97 T/NFE | 1.5B average 44.03 vs 41.64 (wins HumanEval, loses MBPP, GSM8K, Minerva); 8B average 65.31 vs 68.09 (HumanEval 57.93 vs 64.63); not lossless | [arXiv 2511.08923](https://arxiv.org/html/2511.08923v1), [MLSys](https://mlsys.org/virtual/2026/oral/3751) |
| I-DLM (Apr 2026) | Qwen3-8B, Qwen3-32B | 4.5B tokens on 8 H100 (paper says SDAR needed about 54B, 12x more) | Full fine-tune (ZeRO-2); lossless variant adds rank-128 LoRA with base-only introspection positions | TPF by stride N=2/3/4/8: 1.80/2.48/2.96/4.01; batch-1 H100 TPS 209.6/281.9/324.5/445.1; MATH-500 batch 1: I-DLM 341 tok/s, lossless 310, EAGLE-3 238; 3.1x throughput vs Qwen3-8B at matched MATH-500 quality | vs Qwen3-8B: MMLU 82.4 vs 83.5, GPQA-D 55.6 vs 58.9, AIME-25 60.8 vs 65.4, LCB-v6 45.7 vs 50.3, HumanEval 93.3 vs 95.1, MATH-500 96.8 vs 95.8, IFEval equal | [arXiv 2604.11035](https://arxiv.org/html/2604.11035v1) |
| DiffusionGemma (Google, Aug 2026) | Gemma 4 MoE, 25.2B total, 3.8B active | Under 10% of AR training token budget; SFT then sampler-distillation + RL | Fine-tuned | TPF 19.74 (thinking), 18.76 (no-think); per task HumanEval 23.0, GSM8K 23.2, IFEval 17.2, Codeforces 15.1; 1,479 tok/s vs Gemma 4 AR 204 and AR+MTP 303 tok/s (single H100, FP8, batch 1); 256-token canvas, about 12 effective steps | AIME 2026 69.1 vs 84.2, BigBench EH 47.6 vs 59.1, GPQA-D 73.2 vs 79.8, LCB-v6 69.1 vs 71.4, GSM8K 96.3 vs 96.6, IFEval 97.4 vs 97.2 | [arXiv 2608.00146](https://arxiv.org/html/2608.00146v1) |
| MARS (Apr 2026) | Qwen2.5-0.5B/7B-Instruct | Same 2M-example SFT data, 5 epochs; 7B costs 202 H200-h (2.0x AR SFT) | Continued training from the AR-SFT checkpoint, no new parameters | TPF 1.68 avg (7B, B=4, tau 0.95): GSM8K 1.79, HumanEval 1.60, BBH 2.60; wall-clock 1.71x at batch 4 with block KV cache (0.23x to 0.90x without it) | Multi-token mode average 56.8 vs 56.6 AR-SFT; IFEval -4.0 | [arXiv 2604.07023](https://arxiv.org/html/2604.07023v1) |
| NBDiff-7B (Dec 2025) | Off-the-shelf AR weights | Gradual block-size growth with context-causal prefix | Updated | Numbers not on abstract | Claims SOTA among 7B DLMs | [arXiv 2512.06776](https://arxiv.org/abs/2512.06776) |

- SDAR, per I-DLM's analysis, needs a KV-commit forward pass per block that yields no new tokens, capping TPF at N/2 (2.0 at N=4) even with perfect acceptance. TiDAR's compute efficiency is capped at N/(N+1) and it spends about 7.8x compute overhead at TPF 2.5 against I-DLM's 2.5x. These are third-party claims from the I-DLM authors — [I-DLM](https://arxiv.org/html/2604.11035v1)
- DiffusionGemma's sampler distillation raised TPF from about 5 (SFT only) to nearly 20 and halved output length; at about 32 concurrent requests AR regains the throughput lead; a 256-token MoE canvas touches about 84 unique experts versus 8 per AR token, making each step about 4.3x slower — [DiffusionGemma](https://arxiv.org/html/2608.00146v1)
- An independent analysis of DiffusionGemma finds commitment happens "in a short late burst" with structured JSON committed in essentially arbitrary order; commit confidence tracks correctness on math but carries no signal on factual recall — [arXiv 2606.14620](https://arxiv.org/abs/2606.14620)
- SPEED (ICML 2026) sharpened-teacher distillation reports up to 12.2x on LLaDA-8B-Instruct and 6.7x on Dream-7B-Instruct, relative to those dLLMs' own decoding, not to AR — [ICML poster](https://icml.cc/virtual/2026/poster/62409)
- ParallelBench finds dLLMs under parallel decoding "suffer dramatic quality degradation" on tasks easy for AR models because the conditional independence assumption ignores token dependencies, and current strategies cannot adapt parallelism to difficulty — [arXiv 2510.04767](https://arxiv.org/abs/2510.04767)

**Conversions that do not modify base weights**
- None of the block-diffusion conversions above keeps the base frozen. The frozen-base routes are: DFlash, DFlare, JetSpec and DSpark drafters, which train a separate small module reading frozen target hidden states (section 4) — [DFlash](https://arxiv.org/html/2602.06036v2), [JetSpec](https://arxiv.org/html/2606.18394v3), [DSpark](https://arxiv.org/html/2607.05147); Apple's gated LoRA, which is active only on mask positions so the NTP path is unchanged — [arXiv 2507.11851](https://arxiv.org/html/2507.11851v1); I-DLM's lossless R-ISD LoRA variant — [I-DLM](https://arxiv.org/html/2604.11035v1); and Orthrus, which adds a lightweight trainable module to a frozen LLM and claims up to 7.8x with lossless consensus. Orthrus's base models, benchmarks and hardware are not on its abstract — [arXiv 2605.12825](https://arxiv.org/abs/2605.12825)

### Inferences
- In the target size class, the best lossy conversion at about 3 points of average loss is TiDAR-8B at 5.91x. The best near-lossless one is I-DLM-8B at about 3.1x (0 to 4.6 points below Qwen3-8B per benchmark). Neither result was measured with quantized weights or on consumer hardware.
- Block-diffusion conversion is costly per checkpoint: 1B tokens and about 770 A100-hours for Fast-dLLM v2 7B at the low end, 150B tokens for TiDAR-8B at the high end. That conflicts with the requirement to work on any checkpoint with light adaptation. I-DLM at 4.5B tokens on 8 H100s is the cheapest conversion reported to reach near-parity at 8B.
- High TPF (about 20) has been shown only together with RL and sampler distillation and multi-point losses on reasoning (DiffusionGemma). Those losses fail the "equal or better accuracy" requirement unless decoding falls back to AR on hard spans, and no one has built or measured that hybrid.
- TPF numbers for diffusion models count committed tokens per forward whose cost grows with block width. On a bandwidth-bound consumer GPU a 16 to 32 token block costs about the same as one token, but a 256-token canvas does not (DiffusionGemma's step costs about 3.2x a single AR step even on an H100).

### Gaps
- SDAR's numeric TPF and wall-clock values are only in figures that were not machine-readable; GPU-hours for SDAR, SBD, TiDAR, LLaDA 2.0 and DiffusionGemma conversions are not reported.
- Dream 7B GPU-hours, D2F training cost, LLaDA 2.0 conversion token count and NBDiff numbers were not retrieved.
- No conversion paper reports results on prose or creative writing TPF; DiffusionGemma's closest category is IFEval (TPF 17.2).
- No block-diffusion conversion reports quantized-weight (4-bit) TPF or accuracy.

## 2. Set Block Decoding (Meta, 2025) and successors

### Takeaway
SBD fine-tunes an NTP model with a joint NTP plus masked-token loss over 70B tokens (full weights) and cuts forward passes 2.2x to 5.4x on Llama-3.1-8B and Qwen3-8B at near-NTP accuracy. Chat-style tasks (GSM8K, HumanEval+, MBPP) gain only 2.2x to 3.0x; long reasoning traces gain 3.4x to 5.4x. Wall-clock gains are argued from a roofline model only.

### Cited Findings
- Setup: Llama-3.1 8B base and Qwen-3 8B base fine-tuned on 70B tokens (reasoning + instruct mix, 32k context), AdamW lr 3e-4, 2M-token batch, 34k iterations, block size sampled uniformly from 2 to 16 during training; EB-Sampler entropy threshold gamma at inference, temperature 0; exact KV caching retained; no architecture change — [arXiv 2509.04185](https://arxiv.org/pdf/2509.04185)
- Table 1 Llama-3.1 8B (NTP baseline then SBD gamma_low then SBD gamma_high, with NFE reduction): MATH500 80.2 / 81.0 (3.55x) / 80.4 (3.23x); AIME25 33.3 / 30.0 (3.35x) / 23.3 (4.50x); LCB v6 31.5 / 31.7 (3.72x) / 29.9 (4.59x); GSM8K 85.3 / 84.2 (2.20x) / 84.0 (2.34x); HumanEval+ 56.7 / 57.9 (2.61x) / 54.9 (2.95x); MBPP 70.4 / 69.5 (2.49x) / 67.2 (2.66x) — [arXiv 2509.04185](https://arxiv.org/pdf/2509.04185)
- Table 1 Qwen-3 8B: MATH500 86.6 / 85.0 (3.41x) / 85.4 (3.54x); AIME25 33.3 / 33.3 (3.85x) / 26.6 (5.06x); LCB v6 36.6 / 37.2 (3.92x) / 33.3 (5.36x); GSM8K 90.1 / 90.1 (2.77x) / 88.7 (2.86x); HumanEval+ 69.5 / 66.5 (2.51x) / 65.2 (2.72x); MBPP 78.0 / 77.5 (2.61x) / 74.6 (3.03x). Average NFE reduction 3.0x/3.4x (Llama) and 3.2x/3.8x (Qwen) — [arXiv 2509.04185](https://arxiv.org/pdf/2509.04185)
- gamma values: reasoning 0.1 (low) and 0.35 (high); chat 0.35 and 0.6 — [arXiv 2509.04185](https://arxiv.org/pdf/2509.04185)
- SBD needs more steps than NTP to reach parity in SFT and closes the gap after about 34k iterations; a 3B ablation trained SBD on the last 10% of a 1T-token pretraining run — [arXiv 2509.04185](https://arxiv.org/pdf/2509.04185)
- Roofline analysis (H100 FP8, 8B model) says block-16 forwards cost only a few percent more than a 1-token forward at batch 1, so NFE reduction translates to about the same wall-clock gain; gains shrink as batch or block grows — [arXiv 2509.04185](https://arxiv.org/pdf/2509.04185)
- Third-party review: if a block forward costs 1.3x a token forward, a 3x NFE reduction becomes about 2.3x wall-clock; reported range 2.2x to 5.4x — [Pith review](https://pith.science/paper/2509.04185)
- Successors in the same family (NTP + masked prediction in one network, KV-cache exact): MARS uses one LM head for all block positions and reaches TPF 1.68 at 7B with 202 H200-hours — [MARS](https://arxiv.org/html/2604.07023v1); TiDAR drafts by diffusion and samples AR in one pass at 8.25 T/NFE — [TiDAR](https://arxiv.org/html/2511.08923v1); I-DLM verifies its previous draft while drafting the next in the same pass — [I-DLM](https://arxiv.org/html/2604.11035v1)

### Inferences
- SBD's measured gain on short chat-style answers (2.2x to 3.0x) is the relevant number for prose-like workloads; the 4x to 5x figures come from long thinking traces.
- SBD is not lossless (accuracy drops up to 10 points on AIME25 at gamma_high). Its accuracy-preserving setting gives about 3x, lower than DFlash's lossless 4.9x average on Qwen3-8B (section 4), and SBD costs 70B tokens of full fine-tuning against DFlash's frozen-target draft.

### Gaps
- No measured wall-clock numbers for SBD; no SBD release of weights or code was found; no successor paper titled as SBD v2 was found.
- No SBD results on open-ended prose, quantized weights or GPUs other than the H100 roofline.

## 3. Consistency and Jacobi decoding (CLLM, lookahead, PEARL, ParallelSpec, 2026 work)

### Takeaway
Plain Jacobi decoding on an unmodified model yields about 1.0x. Training-free n-gram and lookahead methods yield TPF of about 1.3 to 1.7 on 3B to 8B models. Consistency-distilled models reach 2x (CLLM) to 4x (Jacobi Forcing, ICML 2026) on code and math at a few points of accuracy loss, after a cheap fine-tune of about 20k steps.

### Cited Findings
- Jacobi Forcing (ICML 2026), A100, batch 1, Qwen2.5-Coder-7B-Instruct base: AR 41.3 tok/s, HumanEval 87.8; vanilla Jacobi TPF 1.03, 0.97x; CLLM* TPF 2.68, 2.50x, 87.8; Jacobi Forcing TPF 4.01, 3.86x, 83.5; with multi-block + rejection recycling TPF 4.09, 3.97x, 83.5. MBPP: AR 74.3; CLLM* 1.94x at 71.4; Jacobi Forcing (MR) TPF 2.84, 2.62x at 70.4 — [arXiv 2512.14681](https://arxiv.org/pdf/2512.14681)
- Jacobi Forcing on Qwen2.5-Math-7B-Instruct: GSM8K AR 92.4, JF (MR) TPF 4.04, 3.71x at 91.4; MATH AR 77.0, JF (MR) TPF 3.98, 3.68x at 77.4. On B200 HumanEval JF (MR) TPF 4.21, 328 tok/s, 3.95x — [arXiv 2512.14681](https://arxiv.org/pdf/2512.14681)
- Jacobi Forcing training: 450k prompts for trajectory generation, lr 1e-6, batch 4, max new length 2048, 10k steps at block 16 then 10k steps at block 32 (progressive distillation adds about 20% speedup), on 8x A100-80GB or 8x H200; full weights; speedup saturates with training steps at large block sizes — [arXiv 2512.14681](https://arxiv.org/pdf/2512.14681), [ICML 2026](https://icml.cc/virtual/2026/poster/64353)
- CLLM (ICML 2024): 2.4x to 3.4x generation speedup with consistency distillation on domain and open-domain benchmarks — [arXiv 2403.00835](https://arxiv.org/abs/2403.00835)
- Lookahead decoding (ICML 2024), training-free, no draft model: up to 1.8x on MT-bench, up to 4x on code with multi-GPU strong scaling — [arXiv 2402.02057](https://arxiv.org/abs/2402.02057)
- ESP, training-free embedding-space mask probing (ICML 2026), single A100, SpecBench, 100-token generations: tau (tok/s) on Llama-3.1-8B-Instruct at branch budget 60: PLD 1.30 (32.7), LADE 1.51 (35.6), ESP 1.71 (40.5); Qwen3-8B: PLD 1.24, LADE 1.46, ESP 1.66; Llama-3.2-3B: LADE 1.49, ESP 1.67. No plain-AR tok/s reported — [arXiv 2603.17942](https://arxiv.org/html/2603.17942v1)
- PEARL (ICLR 2025), parallel draft-and-verify with a separate draft model: up to 4.43x over AR and 1.50x over vanilla speculative decoding — [arXiv 2408.11850](https://arxiv.org/abs/2408.11850)
- ParallelSpec (2024): parallel drafter producing multiple tokens in one pass; up to 62% latency reduction over baseline speculative methods, 2.84x overall on Llama-2-13B — [arXiv 2410.05589](https://arxiv.org/abs/2410.05589)
- Parallel Token Prediction (ICLR 2026): moves sampling randomness into input variables so several future tokens are deterministic functions of one pass; can be distilled from an existing model; 2.4x on a diverse-task speculative decoding benchmark — [arXiv 2512.21323](https://arxiv.org/abs/2512.21323)
- Progressive Tree Drafting (Jul 2026): training-free, model-agnostic self-drafting inside the target, building on Lookahead and Self-Draft; numbers not retrieved — [arXiv 2607.10661](https://arxiv.org/html/2607.10661)
- MARS paper measured training-free Jacobi on its 0.5B AR-SFT model at TPF 1.07 — [MARS](https://arxiv.org/html/2604.07023v1)

### Inferences
- The training-free ceiling in this family at 8B is about TPF 1.7 on mixed tasks, far below the 5x requirement.
- Jacobi Forcing is the strongest Jacobi-family result at 7B (about 4x wall-clock on code and math). It changes the base weights and loses about 4 points on HumanEval and MBPP. Its fine-tune is small (about 20k steps at batch 4, roughly 160M generated tokens by my arithmetic from the stated settings), so it is the cheapest route to TPF about 4 reported in this survey.
- PEARL and ParallelSpec need a separate draft model or drafter, so they fall outside "no separate draft model". Their parallel draft-verify overlap is the part that transfers to a single-model design.

### Gaps
- No Jacobi Forcing, CLLM or lookahead results on open-ended chat or prose were found; all Jacobi Forcing results are code and math.
- No 2026 consistency-distillation paper on Llama-3.x 8B or Qwen3 8B general chat was found.

## 4. Multi-token prediction heads and attached drafters trained post-hoc (DeepSeek MTP, Medusa-2, EAGLE-3, HASS, DFlash, DSpark, Falcon, Clover, 2026 heads)

### Takeaway
Frozen-target, post-hoc drafters are the strongest lossless family in 2026. On Qwen3-8B with greedy decoding, the best results are about tau 6.5 for DFlash (block 16), about tau 8 for DFlare and DBloom, and about tau 10.8 for JetSpec with a 256-node tree, giving 4.9x to 9.6x on math. On open-ended chat the same methods fall to tau 4 to 6 (2.7x to 4.6x), and per-depth MTP heads reach only tau 2.3 to 2.7 on chat and summarization. Acceptance on prose is consistently about half of acceptance on math.

### Cited Findings

**DFlash (z-lab, 2026; block-diffusion drafter, 5 layers, frozen target)**
- Qwen3-8B, thinking off, greedy (speedup / tau): GSM8K 5.15x / 6.54, MATH-500 6.08x / 7.87, AIME25 5.62x / 7.08, HumanEval 5.14x / 6.50, MBPP 4.65x / 5.95, LCB 5.51x / 7.27, MT-Bench 2.75x / 4.24; average 4.86x / 6.49. Qwen3-4B average 4.91x / 6.54 — [arXiv 2602.06036](https://arxiv.org/html/2602.06036v2)
- Temperature 1: Qwen3-8B average 4.03x / 5.48; Qwen3-4B 4.24x / 5.69; MT-Bench 2.47x (8B) — [arXiv 2602.06036](https://arxiv.org/html/2602.06036v2)
- EAGLE-3 on the same Qwen3-8B harness: budget 16 average 1.76x / 2.96, budget 60 2.02x / 3.40; MT-Bench 1.63x / 2.83 (16) — [arXiv 2602.06036](https://arxiv.org/html/2602.06036v2)
- LLaMA-3.1-8B-Instruct on SGLang, single B200, concurrency 1 (block 10): GSM8K 2.4x / tau 4.32, HumanEval 2.8x / 4.91, Alpaca 2.2x / 3.73; EAGLE-3 (60) 1.8x to 2.0x with tau 4.07 to 4.65 — [arXiv 2602.06036](https://arxiv.org/html/2602.06036v2)
- SGLang Qwen3-8B MATH-500 throughput: baseline 230 tok/s, DFlash 1,175 tok/s at concurrency 1 (5.1x, tau 8.01), dropping to 2.8x at concurrency 32 — [arXiv 2602.06036](https://arxiv.org/html/2602.06036v2)
- Training: about 800K target-regenerated samples (Nemotron Post-Training V2 + CodeAlpaca), 6 epochs, lr 6e-4, seq 3072; target embedding and LM head shared and frozen; GPU-hours and total parameter count not reported — [arXiv 2602.06036](https://arxiv.org/html/2602.06036v2)

**DFlare, DBloom, DDTree, JetSpec (2026 follow-ups)**
- DFlare: each draft layer attends to its own learned mix of target layers; data scaled from 800K to 2.4M samples; average speedup 5.46x on Qwen3-8B (8% over DFlash), 5.52x on Qwen3-4B (11%), 3.91x on GPT-OSS-20B — [arXiv 2606.02091](https://arxiv.org/abs/2606.02091)
- DBloom (AMD, Aug 2026): expanding DFlare from block 16 to 24 with about 30K extra prompts (1% of DFlare's data) and continuation training raises Qwen3-8B tau: GSM8K 7.86 to 9.23, MATH-500 9.17 to 10.54, HumanEval 7.07 to 7.93, MBPP 6.77 to 7.72, LCB 7.84 to 8.71, MT-Bench 5.00 to 5.38 — [arXiv 2608.30427](https://arxiv.org/pdf/2608.30427)
- DDTree (tree over DFlash per-position distributions) on Qwen3-8B greedy, budget 256: MATH-500 8.78x / 9.81, HumanEval 6.31x / 6.96, MT-Bench 4.26x / 5.41 — [JetSpec Table 1](https://arxiv.org/html/2606.18394v3), [DDTree](https://arxiv.org/pdf/2604.12989)
- JetSpec (Hao AI Lab, 2026): causal parallel draft head on fused target hidden states emits a scored tree in one pass; target frozen; Qwen3-8B greedy budget 256: GSM8K 7.82x / 8.62, MATH-500 9.64x / 10.76, AIME25 8.78x / 9.82, HumanEval 7.12x / 7.78, MBPP 6.73x / 7.43, LCB 7.67x / 8.79, MT-Bench 4.58x / 5.94; budget 16 is about equal to DFlash (MT-Bench 2.68x / 3.98) — [arXiv 2606.18394](https://arxiv.org/html/2606.18394v3)
- JetSpec at temperature 1, budget 256: MATH-500 7.83x / 9.01, MT-Bench 4.06x / 5.22; vLLM batch 1 budget 16 to 128 raises throughput from 443 to 968 tok/s (3.09x to 6.75x); trained on about 800K examples on 8x H100; on AIME25 per-position acceptance is about 99% at depth 1 and about 50% at depth 8 — [arXiv 2606.18394](https://arxiv.org/html/2606.18394v3), [JetSpec project](https://jetspec-project.github.io/jetspec-web/)
- DBloom's linear block of 24 shows no statistically significant committed-length loss versus JetSpec at any tree budget up to 256 nodes on Qwen3-8B — [arXiv 2608.30427](https://arxiv.org/pdf/2608.30427)

**DSpark (DeepSeek, Jul 2026)**
- Semi-autoregressive drafter: parallel backbone (5 layers offline) plus lightweight sequential head plus confidence head that schedules per-request verify length; block 7 offline; target, embedding and LM head frozen; trained on about 1.3M prompts (39.4% math, 38.9% code, 17.6% chat) with target-regenerated responses, 10 epochs — [arXiv 2607.05147](https://arxiv.org/html/2607.05147)
- Qwen3-4B tau (EAGLE-3 / DFlash / DSpark, block 7): GSM8K 5.14 / 5.40 / 6.11; HumanEval 4.16 / 4.74 / 5.38; MT-Bench 2.39 / 3.07 / 3.64; Alpaca 2.26 / 2.96 / 3.54; Arena-Hard 2.55 / 2.83 / 3.29. DSpark domain averages: math 5.57, code 5.12, chat 3.49 — [arXiv 2607.05147](https://arxiv.org/html/2607.05147)
- Macro-average tau gain on Qwen3-8B: +26.7% over EAGLE-3, +18.4% over DFlash. In DeepSeek-V4 production, per-user speed rose 60% to 85% over the MTP-1 baseline at matched throughput — [arXiv 2607.05147](https://arxiv.org/html/2607.05147), [LMSYS blog](https://www.lmsys.org/blog/2026-07-06-dspark-sglang/)
- DPara overlaps a DSpark-style drafter's backbone with verification by precomputing drafts for every acceptance boundary; average 3.21x on Qwen3-8B and 3.52x on Qwen3-14B over AR across seven benchmarks — [arXiv 2609.27396](https://arxiv.org/abs/2609.27396)

**Per-depth MTP heads**
- DeepSeek-V3 MTP: second-token acceptance 85% to 90% across topics, giving 1.8x TPS — [DeepSeek-V3 report](https://arxiv.org/pdf/2412.19437); a follow-up states 80% to 90% — [arXiv 2505.09343](https://arxiv.org/pdf/2505.09343)
- Gemma 4 with its MTP serving path: TPF 1.40, 303 tok/s vs 204 tok/s AR on H100 FP8 — [DiffusionGemma](https://arxiv.org/html/2608.00146v1)
- Post-hoc MTP heads on frozen Qwen3-8B (Microsoft, Oct 2026): K=3 chained heads, each about 200M parameters (one transformer layer at backbone width; about 600M total, about 7.5% of the backbone); 1.89M prompts, about 2.5B tokens per epoch, 5 epochs, about 40 h on 32 A100-80GB. Mean accepted length: GSM8K 3.05, MATH500 3.00, MBPP 2.84, MT-Bench 2.67, Alpaca 2.47, CNN/DailyMail 2.32; matches or exceeds MiMo-7B's jointly pretrained MTP. Per-head acceptance on GSM8K is 86.5/67.4/51.1%. Self-distillation on target outputs is the biggest driver (+0.14 to +0.41 MAL). A chain-aware relaxed verifier adds +12% to +16% MAL, but AIME25 accuracy dropped to 60.0 from 76.7 at one threshold — [arXiv 2610.00888](https://arxiv.org/html/2610.00888v1)
- Apple gated-LoRA mask-token MTP on Tulu3-8B (frozen base, LoRA active only on mask positions, rank 128, 2-layer MLP sampler; about 1M Tulu3 examples, 50k iterations on 8 A100): average accepted tokens per step with 1/2/4/8 masks are 1.67/2.14/2.88/3.17; with 8 masks HumanEval 5.35, GSM8K 5.22, AlpacaEval 2.52, IFEval 2.39, MMLU 2.38, PopQA 1.91. NTP accuracy and loss are unchanged, but gated LoRA cannot be fused into base weights — [arXiv 2507.11851](https://arxiv.org/html/2507.11851v1)

**Earlier heads (2024 to 2025)**
- Medusa-1 (frozen backbone) over 2.2x; Medusa-2 (heads + backbone jointly fine-tuned) 2.3x to 3.6x — [arXiv 2401.10774](https://arxiv.org/abs/2401.10774)
- EAGLE-3: up to 6.5x, about 1.4x over EAGLE-2, 1.38x SGLang throughput at batch 64 — [arXiv 2503.01840](https://arxiv.org/abs/2503.01840)
- HASS (ICLR 2025): 2.81x to 4.05x averaged over three datasets on four LLaMA models, 8% to 20% over EAGLE-2 — [arXiv 2408.15766](https://arxiv.org/abs/2408.15766)
- Falcon (AAAI 2025): semi-autoregressive drafter equal to about two transformer layers; 2.91x to 3.51x on Vicuna and LLaMA2-Chat — [arXiv 2412.12639](https://arxiv.org/abs/2412.12639)
- Kangaroo: 67M-parameter adapter on a shallow sub-network, up to 1.68x on Spec-Bench, beating Medusa-1 (591M parameters) — [arXiv 2404.18911](https://arxiv.org/abs/2404.18911)

### Inferences
- Head size: DFlash and DSpark use about 5 layers, roughly 15% of a 36-layer Qwen3-8B by layer count. That is an estimate, since parameter counts are not reported. Per-depth MTP heads cost about 7.5% of the backbone. Kangaroo is about 1% of a 7B model.
- Prose acceptance: across DFlash, DSpark, JetSpec, DBloom, Microsoft's MTP heads and Apple's gated LoRA, chat or summarization tau is 45% to 60% of math tau at the same block or tree budget. The best lossless chat result is tau 5.94 (JetSpec, 256-node tree, greedy) and 5.22 at temperature 1.
- These drafters are the only frozen-target route with tau above 5. They need one training run per target, about 0.8M to 2.4M regenerated samples. That is light next to conversion but still per checkpoint, so a runtime that must support any checkpoint needs a fast drafter-training pipeline. A universal drafter does not exist yet.
- Tree budgets of 128 to 256 nodes (JetSpec, DDTree) raise verify FLOPs 8x to 16x over block 16. See section 7 for why that cost is not free on a 912 GB/s, roughly 70 TFLOPS-class consumer GPU with 4-bit weights.

### Gaps
- Clover and Clover-2 numbers were not retrieved; DFlash, DSpark and JetSpec parameter counts and GPU-hours are not reported.
- No published tau for these drafters on creative writing or long-form prose beyond MT-Bench, Alpaca, Arena-Hard and CNN/DailyMail.
- No 2026 drafter result on Llama-3.x 1B to 3B or on non-Qwen dense 8B checkpoints except DFlash's LLaMA-3.1-8B SGLang table.

## 5. Self-speculative decoding without extra weights (LayerSkip, SWIFT, Kangaroo, Draft and Verify, early exit)

### Takeaway
With no training, layer-skipping self-speculation gives 1.04x to 1.5x. An independent ICML 2026 characterization on LayerSkip-Llama-3-8B measured SWIFT at only 1.07x to 1.17x. Early-exit methods reach 1.5x to 2.2x but require training: LayerSkip retraining or Kangaroo's adapter.

### Cited Findings
- ICML 2026 characterization (Toronto + Fujitsu), LayerSkip-Llama-3 8B, tok/s: CNN/DM baseline 58.4, SWIFT 65.5 (1.12x, acceptance 0.95, draft is 61% of model), LayerSkip 88.7 (1.52x), DEL 130.0 (2.23x); HumanEval baseline 57.6, SWIFT 67.6 (1.17x), LayerSkip 105.3 (1.83x), DEL 128.0 (2.22x); GSM8K baseline 58.2, SWIFT 62.0 (1.07x), LayerSkip 94.7 (1.63x), DEL 114.9 (1.97x) — [ICML 2026 paper](https://www.eecg.utoronto.ca/~ganesa10/assets/pdfs/selfspecdecode-icml2026.pdf)
- Same study on LayerSkip-Llama-2: layer-skipping methods (SWIFT, Draft and Verify) 1.04x to 1.48x; early-exit methods 1.49x to 2.67x mean. Conclusions: draft cost, not acceptance rate, dominates throughput; layer skipping cannot reuse draft computation in verification, so the full model runs again; early exit requires retraining — [ICML 2026 paper](https://www.eecg.utoronto.ca/~ganesa10/assets/pdfs/selfspecdecode-icml2026.pdf)
- LayerSkip (ACL 2024): layer dropout + shared early-exit loss during pretraining, continual pretraining or fine-tuning; up to 2.16x CNN/DM, 1.82x coding, 2.0x TOPv2 — [arXiv 2404.16710](https://arxiv.org/abs/2404.16710)
- SWIFT (ICLR 2025), training-free: authors report 1.3x to 1.6x — [arXiv 2410.06916](https://arxiv.org/abs/2410.06916)
- CLaSp (ACL 2025), training-free in-context layer skipping: authors report 1.3x to 1.7x on LLaMA3; an independent review found 1.08x on LLaMA-3-8B non-greedy — [HF paper page](https://huggingface.co/papers/2505.24196), [Pith review](https://pith.science/paper/2505.24196)
- ConfLayers (Apr 2026), training-free confidence-based layer skipping: up to 1.4x — [arXiv 2604.14612](https://arxiv.org/pdf/2604.14612)
- SparseSpec-L (Jul 2026), training-free sparse-to-full self-speculation: 1.55x to 2.79x across 3B to 14B; 2.79x with 72.9% acceptance on Llama-3.1-8B-Instruct, LongBench v2 (long inputs) — [arXiv 2607.27735](https://arxiv.org/html/2607.27735)
- Kangaroo: early exit + 67M adapter, up to 1.68x on Spec-Bench — [arXiv 2404.18911](https://arxiv.org/abs/2404.18911)
- QuantSpec (ICML 2025): the draft is the same model with 4-bit weights and a hierarchical 4-bit KV cache; acceptance above 90%, up to about 2.5x end to end, mainly at long context — [arXiv 2502.10424](https://arxiv.org/html/2502.10424v1)
- Component-aware self-speculation exists for hybrid (attention + SSM) models; numbers not retrieved — [arXiv 2605.01106](https://arxiv.org/pdf/2605.01106)

### Inferences
- With no training, self-speculation in the 8B class tops out near 1.2x to 1.5x at short context. The cheapest early-exit option (a Kangaroo-style adapter) reaches about 1.7x, and DEL on a LayerSkip-trained checkpoint about 2.2x. None of these approaches 5x.
- Self-drafting from a quantized copy of the same weights (QuantSpec) pays off when KV dominates, i.e. long context. At short context on a 12 GB card the weights already dominate, so its value depends on whether the runtime already serves 4-bit weights. If it does, the draft and target cost almost the same.

### Gaps
- Draft and Verify numbers were not separately retrieved; no training-free self-speculation results on Qwen3 8B at short context were found.

## 6. Ceiling analyses: where acceptance saturates and why

### Takeaway
No paper gives an information-theoretic upper bound on tokens per forward for natural text. The empirical evidence is consistent. For the best 2026 drafters, tau is limited by the block horizon on math (35.7% of block-8 rounds fully accept on MATH-500) but not on chat, where ceiling-bin mass is low and tau sits near 4 to 6. Prose acceptance is limited by the target's own per-token uncertainty (margin), not by drafter capacity or block length.

### Cited Findings
- DBloom: the "ceiling bin" (rounds where the whole block is accepted) correlates with gain from widening block 16 to 24 (Spearman rho +0.88 to +0.93); MT-Bench is consistently the low-ceiling benchmark (tau 5.00 to 5.38 on Qwen3-8B vs 9.17 to 10.54 on MATH-500); after expansion to 24 the ceiling fraction falls to a few percent, so the authors stop at 24 — [arXiv 2608.30427](https://arxiv.org/pdf/2608.30427)
- DFlash's Table 8 (cited by DBloom): a block-8 drafter fully accepts the block 35.7% of the time on MATH-500; drafters generalize to smaller but not larger inference blocks; naive widening at inference lowers speedup (DFlare Qwen3-4B MATH-500 6.5x to 3.0x) — [arXiv 2608.30427](https://arxiv.org/pdf/2608.30427)
- JetSpec: on AIME25 per-position acceptance is about 99% at depth 1 and about 50% at depth 8 — [JetSpec project](https://jetspec-project.github.io/jetspec-web/), [arXiv 2606.18394](https://arxiv.org/html/2606.18394v3)
- Domain gap measurements: DSpark Qwen3-4B tau math 5.57, code 5.12, chat 3.49 — [arXiv 2607.05147](https://arxiv.org/html/2607.05147); Apple gated LoRA 8 masks code 5.35, math 5.22, chat 2.46, knowledge QA 2.16 — [arXiv 2507.11851](https://arxiv.org/html/2507.11851v1); Microsoft MTP heads GSM8K 3.05 vs CNN/DailyMail 2.32 — [arXiv 2610.00888](https://arxiv.org/html/2610.00888v1); DiffusionGemma TPF IFEval 17.2 vs GSM8K 23.2 — [arXiv 2608.00146](https://arxiv.org/html/2608.00146v1)
- DSpark's static-threshold sweep on Qwen3-4B raised chat acceptance from 45.7% to 95.7% by truncating at confident positions. Per-token survival probability is therefore predictable, and the loss is in tokens per round, not in correctness — [arXiv 2607.05147](https://arxiv.org/html/2607.05147)
- Theory (Jun 2026): acceptance under greedy, relaxed and tree rules depends on local ranking and margin events of the target distribution; tree acceptance reduces to whether the target's greedy token lies in the drafter's top-m; validated on Qwen3 only; bounds not stated numerically on the abstract — [arXiv 2606.30265](https://arxiv.org/abs/2606.30265)
- EACL 2026: ties the expected number of accepted tokens to the expected entropy of the target model via branching random walks (described in a search summary as a lower bound inversely related to target entropy) — [ACL Anthology](https://aclanthology.org/2026.eacl-long.301.pdf)
- ParallelBench: an information-theoretic analysis shows parallel (conditionally independent) decoding degrades when inter-token dependencies are strong, and fixed parallelism cannot match task difficulty — [arXiv 2510.04767](https://arxiv.org/abs/2510.04767)
- A small study (Llama-2-7B GPTQ target, TinyLlama draft, depth-3 trees) found chat had the highest target entropy (1.25 nats) yet the highest acceptance (0.565), and entropy to acceptance Spearman |rho| < 0.21. The weak drafter limits how far this generalizes — [arXiv 2604.14682](https://arxiv.org/html/2604.14682)
- SpecTr-GBV gives a multi-draft bound on expected acceptance length converging to the maximum length as drafts grow — [arXiv 2604.25925](https://arxiv.org/pdf/2604.25925)
- Self-speculation study: acceptance rate is a poor standalone predictor of throughput; draft and verification costs dominate — [ICML 2026 paper](https://www.eecg.utoronto.ca/~ganesa10/assets/pdfs/selfspecdecode-icml2026.pdf)

### Inferences
- Lossless per-forward parallelism on chat saturates near tau 5 to 6 even with 256-node trees, and the ceiling indicator says longer blocks will not lift it. The remaining lever for prose is to raise acceptance at low-margin positions. Relaxed or typical acceptance does that, but it is no longer lossless; Microsoft's +14% MAL came with accuracy hits on AIME at some thresholds.
- Requirement check (my arithmetic). For an 8B model at about 4.8 bits per weight (about 4.8 GB), one weight pass at 912 GB/s takes about 5.3 ms, so the AR ceiling is about 190 tok/s before KV traffic. If llama.cpp's best configuration already realizes a drafter speedup of N on a task, beating it by 5x at equal bytes per pass needs about 5N tokens per forward. With N about 2 to 4 on chat, that is roughly 10 to 30 tokens per forward on prose, at least twice the best published lossless chat tau (5.94). The only methods reporting TPF about 20 (DiffusionGemma) lose up to 15 benchmark points.

### Gaps
- No published entropy-based upper bound on tau for natural-language prose versus code; no creative-writing or long-form fiction acceptance study for 2026 drafters.
- The MoE speculation-bound paper (Qwen3-Coder + EAGLE-3) frames budget choice as a stochastic shortest path; its numeric saturation values were not on the abstract — [arXiv 2609.22156](https://arxiv.org/html/2609.22156v1)

## 7. Composability with quantization, sparsity and KV compression

### Takeaway
Gains do not multiply cleanly on bandwidth-bound consumer hardware. The one controlled study on an RTX 3090 found EAGLE-2 tree drafting added no speedup on a W4A16 Llama-3-8B. Tree verification makes low-bit dequant kernels compute-bound, which erases the bandwidth savings. Linear or small drafts on top of 4-bit weights still help, and KV compression composes with speculation mainly at long context.

### Cited Findings
- SpecMQuant: EAGLE-2 on 4-bit-weight targets (W4A16, W4A8) gives limited extra speedup versus FP16 or W8A8 targets because tree verification compute offsets 4-bit bandwidth gains. Integrating EAGLE-2 with quantization adds speedup over either alone "except when EAGLE-2 is applied to W4A16 8B model on the RTX 3090". Fewer drafts give higher speedup on 4-bit models. A hierarchical scheme (small intermediate model turning tree drafts into sequence drafts) reaches 2.78x on W4A16 Llama-3-70B on A100, 1.31x over EAGLE-2 — [arXiv 2505.22179](https://arxiv.org/abs/2505.22179)
- QSPEC (EMNLP 2025): complementary quantization schemes for draft and verify, up to 1.64x without quality loss — [QSPEC PDF](https://i.cs.hku.hk/~cwu/papers/jtzhao-emnlp25.pdf)
- QuantSpec (ICML 2025): 4-bit weights + hierarchical 4-bit KV as self-draft, acceptance above 90%, up to about 2.5x, about 1.3x less memory than sparse-KV self-speculation (MagicDec, TriForce) — [arXiv 2502.10424](https://arxiv.org/html/2502.10424v1)
- SparseSpec-L: sparse-attention self-draft gives 2.79x at long context on Llama-3.1-8B — [arXiv 2607.27735](https://arxiv.org/html/2607.27735)
- DiffusionGemma ran FP8 weights with diffusion decoding on one H100 for 1,479 tok/s vs 303 tok/s for FP8 AR+MTP: diffusion parallelism and FP8 combined — [arXiv 2608.00146](https://arxiv.org/html/2608.00146v1)
- DSpark runs in production on DeepSeek-V4 (with V4's compressed KV design) and adds 60% to 85% per-user speed over MTP-1 — [arXiv 2607.05147](https://arxiv.org/html/2607.05147), [DeepSeek-V4.1-Flash report](https://arxiv.org/pdf/2609.19969)
- DFlash speedup decays with concurrency (5.1x at 1 to 2.8x at 32 on Qwen3-8B SGLang), showing verification compute starts to bind as total tokens per pass grow — [arXiv 2602.06036](https://arxiv.org/html/2602.06036v2)
- Community report: BeeLlama.cpp fork pairing DFlash with TurboQuant KV compression on a single RTX 3090 reached 164 tok/s on Qwen 3.6 27B (4.4x) and 177.8 tok/s on Gemma 4 31B (4.93x); gains confined to token generation; secondary source, unverified — [aiweekly](https://aiweekly.co/alerts/beellama-dflash-hits-nearly-5x-token-speed-on-rtx-3090)
- I-DLM keeps AR-style causal KV so paged-KV and other AR serving optimizations carry over; its stationary-batch scheduler is part of its 2.2x to 4.5x concurrency gains over SDAR and LLaDA-2.1-mini — [arXiv 2604.11035](https://arxiv.org/html/2604.11035v1)

### Inferences
- Roofline on the target GPU (my arithmetic). The bandwidth figure (912 GB/s) is given; the compute figure is an assumption of about 70 dense FP16 tensor TFLOPS with FP32 accumulate for GA102 class parts, about 45 TFLOPS achieved, which should be checked against NVIDIA's GA102 specs. An 8B model costs about 16 GFLOP per token, about 0.36 ms at 45 TFLOPS, while one 4.8 GB weight pass costs about 5.3 ms. Verification is therefore nearly free up to roughly 12 to 16 tokens per pass and turns compute-bound beyond that, earlier if dequant runs on CUDA cores instead of tensor cores. Linear blocks of 16 to 24 (DFlash, DBloom) and SBD blocks of 16 fit this budget. JetSpec and DDTree 128 to 256-node trees cost 8x to 16x more compute per pass and would lose much of their datacenter speedup here. SpecMQuant's RTX 3090 null result for trees on W4A16 is consistent with this.
- Multiplicative stacking is plausible only when the verify width stays inside the free compute window. Per-forward parallelism is roughly orthogonal to bytes per weight, but each extra verified token raises dequant work, so lower-bit formats shrink the free window.

### Gaps
- No published measurement of DFlash, DSpark, JetSpec, SBD, Jacobi Forcing or I-DLM on any 12 GB Ampere GeForce card or with GGUF-style 2 to 4 bit weights.
- No paper measures whether 3 to 4 bit quantization of the target lowers drafter acceptance (tau) for 2026 drafters.
- No study quantizes a block-diffusion-converted model (SDAR, Fast-dLLM v2, TiDAR, I-DLM) or an SBD or Jacobi-Forcing model and reports TPF and accuracy; targeted searches returned none.

## Thin or unexplored

- Consumer-GPU measurements. Every 2026 parallel-decoding result (DFlash, DFlare, DBloom, JetSpec, DSpark, SBD, Jacobi Forcing, TiDAR, I-DLM, DiffusionGemma) is on A100, H100, H200 or B200. The only consumer-card data are SpecMQuant on an RTX 3090, which found no gain for trees on W4A16 ([arXiv 2505.22179](https://arxiv.org/abs/2505.22179)), and one unverified community fork ([aiweekly](https://aiweekly.co/alerts/beellama-dflash-hits-nearly-5x-token-speed-on-rtx-3090)).
- Choosing the verify budget for low-bit weights on compute-poor, bandwidth-rich cards. CaDDTree and BASTION choose tree budgets by latency on datacenter GPUs ([DBloom related work](https://arxiv.org/pdf/2608.30427)). Nobody models dequant cost per verified token for 2 to 4 bit formats.
- Quantizing converted or parallel-trained models. No paper reports how 2 to 4 bit weights affect TPF or accuracy for SDAR, Fast-dLLM v2, TiDAR, I-DLM, SBD or Jacobi Forcing models, or whether a target's quantization lowers a DFlash, JetSpec or DSpark drafter's tau.
- Prose and creative writing. Chat tau is about half of math tau for every method. No method targets prose acceptance directly, no 2026 paper reports tau on long-form fiction or essay writing, and DiffusionGemma reports no chat category.
- An information-theoretic ceiling for natural text. There is only an entropy-linked lower bound (EACL 2026) and margin-based certificates ([arXiv 2606.30265](https://arxiv.org/abs/2606.30265)). Nobody gives an upper bound on lossless tokens per forward for prose against code, or measures the target's own per-position margin distribution as a tau ceiling.
- Drafters that work across checkpoints. Every frozen-target drafter is trained per target on 0.8M to 2.4M regenerated samples. No published drafter transfers across architectures, or even across fine-tunes of one base, without retraining. ESP is training-free but stops at tau about 1.7 ([arXiv 2603.17942](https://arxiv.org/html/2603.17942v1)).
- Lossless frozen-base parallel decoding with high tau. Apple's gated LoRA reaches 5.35 on code and 2.46 on chat ([arXiv 2507.11851](https://arxiv.org/html/2507.11851v1)), and Orthrus claims 7.8x lossless with no published detail ([arXiv 2605.12825](https://arxiv.org/abs/2605.12825)). Neither has been replicated independently, and neither has been combined with block-diffusion drafting.
- Stacking a parallel-trained target with an attached drafter. No paper puts a DFlash, DSpark or JetSpec drafter on an SBD, Jacobi-Forcing or I-DLM model, or uses such a model's own block predictions as the draft inside a lossless verifier. TiDAR and I-DLM's lossless R-ISD come closest.
- Hybrid diffusion-AR fallback for accuracy. DiffusionGemma keeps an AR mode and proposes dynamic routing but does not build or measure it ([arXiv 2608.00146](https://arxiv.org/html/2608.00146v1)). Routing hard spans such as AIME-style reasoning to AR while diffusing easy spans has no published evaluation.
- Block horizons beyond 24. DBloom stops at 24 because the ceiling bin empties, and DFlash leaves adaptive block size for future work ([arXiv 2608.30427](https://arxiv.org/pdf/2608.30427)). Per-request block size chosen from survival probability exists only in DSpark's production scheduler, which uses a maximum block of 5.
- Architectures beyond dense and MoE transformers. Results are almost all on Qwen, Llama and Gemma attention models. Hybrid SSM-attention self-speculation has a single 2026 paper ([arXiv 2605.01106](https://arxiv.org/pdf/2605.01106)), and no conversion recipe has been shown to apply unchanged across architecture families.
- Composing with KV compression at short context on a single GPU. Only DeepSeek-V4 production (DSpark) and the long-context work (QuantSpec, SparseSpec, MagicDec) show it; there is no open 1B to 8B study at short context.
- Small models. Few 2026 drafter or conversion results cover Llama-3.2 1B/3B or Qwen3 1.7B. SDAR found 1.7B and 4B degrade for block size above 4 ([arXiv 2510.06303](https://arxiv.org/html/2510.06303v1)), and no one has explained why small models tolerate parallel blocks worse.

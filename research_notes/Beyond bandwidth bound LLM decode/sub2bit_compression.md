# Sub-2-bit weight compression and structural weight reduction of pretrained transformers (literature as of 2026-11-01)

Conventions used throughout:
- PPL = WikiText-2 perplexity. "ctx" = evaluation context length. Absolute PPL is only comparable inside one paper or harness, so most comparisons below also give the ratio to that paper's own FP16/BF16 baseline ("x FP16").
- ZS = zero-shot average accuracy (task set differs per paper).
- bpw = bits per weight. Most papers count only the decoder linear layers and keep embeddings and lm_head at 16 bit. GGUF file sizes include everything.
- "Derived" marks arithmetic done for these notes from cited numbers; it is not stated by the source.
- Bandwidth floors use spec bandwidth: RTX 3080 Ti 912 GB/s, RTX 3090 936 GB/s, RTX 4090 1008 GB/s, RTX 6000 Ada 960 GB/s, L40S 864 GB/s.
- Llama 3 8B shape used for derived floors: 6.98e9 decoder linear weights + 0.525e9 lm_head weights read per token (token embedding reads one row only).

## 1. Objective: the 2025-2026 accuracy vs bytes-per-weight frontier against GGUF reference points, and realized GPU decode throughput

### Takeaway
At about 2.0 bpw the best published methods (YAQA, LLVQ, and PV-Tuning, all with light fine-tuning) hold Llama 3 8B at 1.17x to 1.27x FP16 perplexity. GGUF IQ2_XXS sits at about 2.07x and IQ1_S at about 8.7x on Llama 3.1 8B Instruct. In the one same-harness chart, a QTIP-derived format (EXL3) matches GGUF quality with about 0.15 to 0.5 fewer decoder bpw. No published method reaches Q4_K_M-level accuracy below about 3 bpw on an 8B model.

Measured batch-1 decode for 2-bit kernels is 16% of the byte-bound floor for AQLM and 26% to 62% for trellis, lattice and LUT kernels. llama.cpp's Q4_0 kernel reaches about 63% to 69%. So a 2-bit trellis model is not automatically faster than llama.cpp 4-bit on the same card.

### Cited Findings

**Same-harness comparison: exllamav3 perplexity-vs-bpw charts**
- The x-axis is decoder-only bpw, read from the chart to about +-0.05. The chart does not state the dataset or ctx.
- EXL3 is "a streamlined variant of QTIP" -- [exllamav3 README](https://github.com/turboderp-org/exllamav3)

| Model | Method @ bpw (approx) -> PPL | Source |
|---|---|---|
| Llama 3.1 70B Instruct | EXL3 1.6 -> 11.358; 1.8 -> 8.187; 2.0 -> 6.608; 2.25 -> 6.053; 2.5 -> 5.528; 3.0 -> 4.502; 4.0 -> 3.853; 6.0 -> 3.656 | [chart](https://github.com/turboderp-org/exllamav3/blob/master/doc/llama31_70b_instruct_bpw.png) |
| Llama 3.1 70B Instruct | GGUF imatrix IQ1_S ~1.67 -> 11.692; IQ1_M ~1.84 -> 9.704; IQ2_XXS ~2.1 -> 7.518; IQ2_S ~2.45 -> 6.487; IQ3_XXS ~3.07 -> 5.417; Q4_K_M ~4.8 -> 3.806 | same chart |
| Llama 3.1 70B Instruct | AQLM 2bit 1x16-g1 ~2.07 -> 6.546; VPTQ v8-k65536 ~2.07 -> 7.124; VPTQ v16 ~2.14 -> 7.212; AWQ 4bit ~4.15 -> 4.191 | same chart |
| Llama 3.1 8B Instruct | EXL3 1.9 (H3) -> 13.230; 2.0 -> 10.516; 2.25 -> 9.619; 2.5 -> 9.087; 3.0 -> 8.159; 3.5 -> 7.931; 4.0 -> 7.729; 5.0 -> 7.665; 8.0 -> 7.637 | [chart](https://github.com/turboderp-org/exllamav3/blob/master/doc/llama31_8b_instruct_bpw.png) |
| Llama 3.1 8B Instruct | GGUF imatrix IQ2_XS ~2.36 -> 12.191; IQ2_S ~2.48 -> 10.818; IQ2_M ~2.7 -> 9.929; IQ3_XS ~3.27 -> 8.512; IQ4_XS ~4.25 -> 7.840; Q4_K_M ~4.8 -> 7.810; Q6_K ~6.56 -> 7.671 | same chart |
| Llama 3.1 8B Instruct | AQLM 2x8-g8-pv 2.0 -> 11.363; AQLM 1x16-g8-pv ~2.27 -> 10.122; VPTQ v8-k65536-4096 ~3.8 -> 8.049 | same chart |
| Llama 3.2 1B Instruct | EXL3 2.0 -> 23.273; 2.25 -> 19.491; 2.5 -> 17.904; 3.0 -> 15.423; 4.0 -> 14.514; 8.0 -> 14.224. GGUF IQ2_XXS ~2.12 -> 52.214; IQ2_XS ~2.36 -> 34.808; IQ2_M ~2.68 -> 23.140. AQLM 2x8-g1-pv ~2.0 -> 22.887 | [chart](https://github.com/turboderp-org/exllamav3/blob/master/doc/llama32_1b_instruct_bpw.png) |

**GGUF reference points from llama.cpp users and maintainers**
- **Llama 3.1 8B Instruct, imatrix GGUF** -- [ThomasBaruzier HF card](https://huggingface.co/ThomasBaruzier/Meta-Llama-3.1-8B-Instruct-GGUF)
  - The card does not state the dataset or ctx.
  - Whole-file bpw is derived from the listed MiB over 8.03e9 parameters.

  | Quant | Size | Whole-file bpw (derived) | PPL | x F16 (derived) |
  |---|---|---|---|---|
  | F16 | 15325 MiB | 16 | 7.3226 | 1.00 |
  | Q4_K_M | 4693 MiB | 4.90 | 7.4975 | 1.024 |
  | IQ2_M | 2812 MiB | 2.94 | 9.5838 | 1.31 |
  | IQ2_XS | 2486 MiB | 2.60 | 11.771 | 1.61 |
  | IQ2_XXS | 2289 MiB | 2.39 | 15.1538 | 2.07 |
  | IQ1_M | 2062 MiB | 2.15 | 26.8862 | 3.67 |
  | IQ1_S | 1927 MiB | 2.01 | 63.4502 | 8.67 |

- **Llama 3 70B GGUF** (wiki.test, ctx not stated, posted by dranger003) -- [llama.cpp discussion #6901](https://github.com/ggml-org/llama.cpp/discussions/6901)
  - F16 2.8308.
  - Q4_K_M: 39.60 GiB, 2.9674 (+4.83%).
  - IQ2_M: 22.46 GiB, 4.8959 (+72.95%).
  - IQ2_XS: 19.69 GiB, 5.7486 (+103%).
  - IQ2_XXS: 17.79 GiB, 6.6705 (+135.6%).
  - IQ1_M: 15.60 GiB, 8.5193 (+201%).
  - IQ1_S: 14.29 GiB, 9.8655 (+248.5%).
  - Derived whole-file bpw over 70.6e9 parameters: Q4_K_M 4.82, IQ2_M 2.73, IQ2_XS 2.40, IQ2_XXS 2.17, IQ1_M 1.90, IQ1_S 1.74.
- **IQ1_M / IQ1_S on Llama 2 at ctx 4096** -- [llama.cpp PR #6302](https://github.com/ggml-org/llama.cpp/pull/6302)
  - LLaMA-v2-7B: IQ1_S 11.86, IQ1_M 9.335. IQ1_M is about 1.96 bpw whole-model.
  - LLaMA-v2-70B: IQ1_S 5.211, IQ1_M 4.829. IQ1_M is about 1.85 bpw whole-model.
  - The IQ1_M tensor type itself is "exactly 1.75 bpw".

**Llama 3 / 3.1 8B frontier points from papers (ratio to each paper's own FP16)**

| Method (venue) | bpw | Fine-tune | PPL (ctx) | x FP16 | ZS vs FP16 | Source |
|---|---|---|---|---|---|---|
| YAQA-B + QuIP#-style FT, QTIP quantizer (ICML 2026), Llama 3.1 8B Instruct | 2 | yes | 7.60 (8192) | 1.17 | 66.38 vs 69.82 | [arXiv 2505.22988](https://arxiv.org/html/2505.22988) |
| LLVQ shape-gain, input-scale FT (ICML 2026) | 2 | ~52M tokens | 6.99 vs 5.75 | 1.22 | MMLU 55.7 vs 65.5 | [arXiv 2603.11021](https://arxiv.org/html/2603.11021) |
| PV-Tuning (NeurIPS 2024) | 2.01 | yes | 6.99 vs 5.54 | 1.26 | 64.36 vs 68.61 | [arXiv 2405.14852](https://arxiv.org/html/2405.14852) |
| QTIP (NeurIPS 2024) | 2 | per QuIP# recipe | 7.33 (8192) vs 5.54 | 1.32 | ArcC 44.2, Wino 70.7 | [arXiv 2406.11235](https://arxiv.org/html/2406.11235) |
| QuIP# (ICML 2024), as reported in QTIP | 2 | yes | 7.84 (8192) | 1.42 | ArcC 39.2, Wino 68.2 | [arXiv 2406.11235](https://arxiv.org/html/2406.11235) |
| XOR-Trellis, no rotation (arXiv Sep 2026), Llama 3.1 8B base | ~2.25 incl. scales | no | 8.00 (4096) vs 5.84 | 1.37 | 64.90 vs 71.03 | [arXiv 2610.00432](https://arxiv.org/html/2610.00432) |
| CALDERA, 2-bit Q + 4-bit rank-256 LR (NeurIPS 2024) | 2.4 | LR-factor FT | 7.34 vs 5.54 | 1.32 | - | low-rank/sharing subagent, [arXiv 2405.18886](https://arxiv.org/html/2405.18886) |
| VPTQ (EMNLP 2024) | 2.08 / 2.24 | yes | 9.29 / 9.19 vs 6.14 | 1.51 / 1.50 | 60.2 / 62.7 vs 68.6 | [arXiv 2409.17066](https://arxiv.org/html/2409.17066) |
| EfficientQAT w2g128 | 2.14 | QAT | 9.80 (2048) vs 6.14 | 1.60 | 59.37 vs 68.58 | [arXiv 2407.11062](https://arxiv.org/html/2407.11062) |
| GPTAQ + QuaRot, W2A16 per-channel (ICML 2025) | ~2 | no | 13.4 vs 6.14 (GPTQ 23.2) | 2.18 | - | [arXiv 2504.02692](https://arxiv.org/html/2504.02692) |
| ParetoQ QAT, ~30B tokens | 2 / 1.58 / 1 | QAT | 8.0 / 8.6 / 9.5 vs 6.15 | 1.30 / 1.40 / 1.54 | 71.2 / 69.0 / 67.8 vs 74.6 | [arXiv 2502.02631](https://arxiv.org/html/2502.02631) |
| PV-Tuning | 1.01 | yes | 11.17 vs 5.54 | 2.02 | 50.01 vs 68.61 | [arXiv 2405.14852](https://arxiv.org/html/2405.14852) |
| BTC-LLM (ACL 2026) | 1.11 / 0.8 | no | 7.70 / 9.49 vs 6.14 | 1.25 / 1.55 | 63.71 / 59.3 vs 68.18 | [arXiv 2506.12040](https://arxiv.org/html/2506.12040v1) |
| PTQ1.61 | 1.61 | no | 22.90 vs 6.14 | 3.73 | 37.20 vs 65.98 | [arXiv 2502.13179](https://arxiv.org/html/2502.13179) |
| LittleBit (NeurIPS 2025) | 1.0 / 0.1 | QAT | 16.30 / 26.11 vs 6.10 | 2.67 / 4.28 | - | [arXiv 2506.13771](https://arxiv.org/html/2506.13771v3) |

**70B-class frontier points**

| Method | bpw | PPL (ctx) vs FP16 | x FP16 | ZS vs FP16 | Source |
|---|---|---|---|---|---|
| YAQA-B, QTIP quantizer, no FT, Llama 3.1 70B Instruct | 2 | 5.30 vs 3.52 (8192) | 1.51 | 66.19 vs 67.67 | [arXiv 2505.22988](https://arxiv.org/html/2505.22988) |
| XOR-Trellis, no rotation, Llama 3.1 70B Instruct | ~2.25 | 5.62 vs 3.52 (8192); QTIP-3INST 6.31 | 1.60 | 78.43 vs 80.63 (QTIP 77.96) | [arXiv 2610.00432](https://arxiv.org/html/2610.00432) |
| PV-Tuning, Llama 3 70B | 2.07 / 1.01 | 4.57 / 8.67 vs 2.59 | 1.76 / 3.35 | 70.38 / 51.47 vs 75.37 | [arXiv 2405.14852](https://arxiv.org/html/2405.14852) |
| QTIP, Llama 3 70B | 2 | 4.97 vs 2.59 (8192); QuIP# 5.77 | 1.92 | ArcC 28.0 (QuIP# 18.3) | [arXiv 2406.11235](https://arxiv.org/html/2406.11235) |
| VPTQ, Llama 3 70B | 2.02 | 5.6 vs 2.9 | 1.93 | 70.9 vs 75.3 | [arXiv 2409.17066](https://arxiv.org/html/2409.17066) |
| EfficientQAT w2g128, Llama 3 70B | 2.14 | 6.38 vs 2.85 (2048) | 2.24 | 67.57 vs 75.33 | [arXiv 2407.11062](https://arxiv.org/html/2407.11062) |
| ARB-LLM_RC, Llama 3 70B | 1.06 | 11.10 vs 2.86 | 3.88 | - | [arXiv 2410.03129](https://arxiv.org/html/2410.03129) |
| QTIP-HYB + FT, Llama 2 70B | 2 | 3.70 vs 3.12 (4096) | 1.19 | - | [arXiv 2406.11235](https://arxiv.org/html/2406.11235) |
| PV-Tuning, Llama 2 70B | 1.14 | 5.52 vs 3.12 | 1.77 | 64.58 vs 72.40 | [arXiv 2405.14852](https://arxiv.org/html/2405.14852) |

**Realized batch-1 decode throughput on single consumer or workstation GPUs**
- Floors are derived. The "% of floor" column is derived unless marked "paper".

| Model, format | GPU | tok/s | FP16 tok/s same setup | % of byte floor | Source |
|---|---|---|---|---|---|
| Llama 2 7B, llama.cpp Q4_0 (reference) | RTX 3090 | 158.16 (tg128) | - | ~63% | [llama.cpp #15013](https://github.com/ggml-org/llama.cpp/discussions/15013) |
| Llama 2 7B, llama.cpp Q4_0 (reference) | RTX 4090 | 186.21 | - | ~69% | same |
| Llama 2 7B, QTIP 2-bit | RTX 3090 | 127 (batch not stated) | 52.5 | ~26% (FP16 head) | [QTIP Table 17](https://arxiv.org/html/2406.11235) |
| Llama 2 7B, QTIP 2-bit / 3-bit / 4-bit | RTX 6000 Ada, batch 1 | 188 / 161 / 140 | 55.9 | ~37% at 2-bit | [QTIP Table 4](https://arxiv.org/html/2406.11235) |
| Llama 2 7B, QuIP# 2-bit / 4-bit | RTX 4090 | 170.50 / 117.73 | - | 29.6% / 40.87% (paper) | [QuIP# Table 5](https://arxiv.org/html/2402.04396) |
| Llama 2 70B, QuIP# 2-bit | RTX 4090 | 32.74 | OOM | 56.84% (paper) | same |
| Llama 2 7B, AQLM 2-bit | RTX 6000 Ada | 81.5 | 55.9 | ~16% | [QTIP Table 4](https://arxiv.org/html/2406.11235) |
| Llama 3.1 8B, HIGGS/FLUTE 2 / 3 / 4-bit | RTX 4090, batch 1 | 173 / 150 / 139 | 57 | ~48% at 2-bit with FP16 head, ~34% with 4-bit head | [HIGGS Table 1](https://arxiv.org/html/2411.17525v1) |
| Llama 2 7B, Q-Palette 2-bit (QTIP measured in same run) | RTX 4090, batch 1 | 272 (QTIP 209) | 71 | ~51% | seed/mixed-precision subagent, [arXiv 2509.20214](https://arxiv.org/html/2509.20214) |
| Llama 3.1 8B, Q-Palette mixed TCQ ~2-bit | RTX 4090, batch 1 | 223 (3.57x FP16) | - | ~62% with FP16 head | same |
| Qwen3-4B, SAGE VQ 2.25 bpw payload (2.50 with radius grids) | RTX 4090, batch 1 | 265.2 | 101.0 | - | [arXiv 2610.14745](https://arxiv.org/html/2610.14745) |
| Bonsai 8B (Qwen3-8B architecture), Q1_0 1.125 bpw, vendor numbers | RTX 4090 | 368 (TG128) | 59 | ~42% | [prism-ml model card](https://huggingface.co/prism-ml/Bonsai-8B-gguf) |
| Llama2-7B ternary (CAT-Q, llama.cpp TQ1_0/TQ2_0) | RTX 4090, batch 1 | 281.66 | - | ~45-53% | [arXiv 2606.26650](https://arxiv.org/html/2606.26650) |
| Llama 2 7B, VPTQ 2.02 bpw (naive Torch/Triton) | not stated | 39.9 | 38.32 | - | [VPTQ Table 2](https://arxiv.org/html/2409.17066) |

**Kernel-level byte-bound efficiency at batch 1 (L40S, one token through 252 Qwen3-4B projections)** -- [arXiv 2609.02652](https://arxiv.org/html/2609.02652)

| Kernel | Time | % of byte bound | Speedup vs FP16 |
|---|---|---|---|
| QTIP 2-bit | 2.246 ms | 61% | 4.89x |
| AWQ w4g128 | 3.252 ms | 88% | 3.38x |
| FP16 | 10.994 ms | - | 1.00x |
| No-weights launch control | 2.306 ms | - | - |

- The paper attributes the largest share of its own lattice kernels' gap to launch geometry and occupancy.

### Inferences
- **The sub-2-bit gap GGUF leaves open.** At about 2.0 to 2.4 bpw, published trellis and VQ methods with light fine-tuning sit at 1.17x to 1.32x FP16 PPL on Llama 3 8B, while IQ2_XXS (2.39 bpw whole-file) sits at 2.07x and IQ1_M (2.15 bpw whole-file) at 3.67x.
- **IQ1_S and IQ1_M are dominated** at 8B by every trellis or VQ method at about 2 decoder bpw.
  - Their whole-file sizes (2.01 and 2.15 bpw) include quantized embeddings and output.
  - Most papers keep those tensors at 16 bit and count decoder linears only, so whole-file comparisons favour GGUF.
  - The exllamav3 chart places all formats on a decoder-only bpw axis, which is the fair comparison.
- **Same-harness gap.** On Llama 3.1 70B:
  - EXL3 at 1.8 bpw (8.19) beats IQ1_M at about 1.84 (9.70).
  - EXL3 at 2.0 (6.61) beats IQ2_XXS at about 2.1 (7.52).
  - EXL3 at 2.0 nearly matches IQ2_S at about 2.45 (6.49).
  - Interpolating the EXL3 curve gives roughly a 0.15 to 0.45 bpw advantage at equal quality.
  - On 8B the advantage is about 0.45 bpw near 2 to 2.7 bpw: EXL3 2.0 at 10.52 vs IQ2_S about 2.48 at 10.82, and EXL3 2.25 at 9.62 vs IQ2_M about 2.7 at 9.93.
- **Q4_K_M-level accuracy below 3 bpw does not exist in published work.** On the same harness EXL3 needs about 3.5 to 4.0 bpw to match Q4_K_M on 8B (7.931 at 3.5, 7.729 at 4.0, vs 7.810). The best QAT result at 2 bits (ParetoQ, about 30B tokens) is still 3.4 ZS points under FP16 on Llama 3 8B.
- **Bandwidth arithmetic for a 3080 Ti (912 GB/s), Llama 3 8B, derived:**

  | Decoder bpw | lm_head | Bytes per token | Floor |
  |---|---|---|---|
  | 4.8 (Q4_K_M-like) | 6.56-bit | 4.62 GB | 197 tok/s |
  | 2.0 | 4-bit | 2.01 GB | 454 tok/s |
  | 2.0 | FP16 | 2.80 GB | 326 tok/s |
  | 1.6 | 4-bit | 1.66 GB | 550 tok/s |
  | 1.0 | 4-bit | 1.14 GB | 804 tok/s |
  | 1.0 | 2-bit | 1.00 GB | 909 tok/s |

  - Scaling the RTX 3090 Q4_0 result by bandwidth puts llama.cpp Q4_K_M Llama 3 8B on a 3080 Ti at roughly 125 to 137 tok/s without drafting (derived estimate, not measured).
  - A 5x target is therefore 625 to 685 tok/s before llama.cpp's drafting gain is counted. That already exceeds the 1.6 bpw floor.
  - If llama.cpp drafting doubles its throughput, the target exceeds even the 1-bit floor of an 8B dense model.
  - Conclusion: weight compression alone cannot reach 5x with equal accuracy. Multi-token verification per weight pass has to supply most of the multiplier, with compression shrinking the bytes per pass.
- **Kernel efficiency is a first-order term.** Published 2-bit trellis and VQ kernels reach 26% to 62% of the byte floor on consumer cards. llama.cpp Q4_0 reaches 63% to 69%, and AWQ-4 on L40S 88%.
  - QTIP 2-bit on an RTX 3090 (127 tok/s, 7B) is slower than llama.cpp Q4_0 on the same GPU model (158 tok/s, 7B). The two numbers come from different software stacks.
  - The fastest 2-bit consumer-GPU numbers (Q-Palette 272 tok/s on 7B, about 51%) are 1.46x llama.cpp Q4_0 on a 4090.
- **70B on a 12 GB card.** The Llama 3 70B decoder holds 68.45e9 linear weights, so it fits in 12 GB only at about 1.25 bpw or less (10.7 GB at 1.25 bpw, derived).
  - The published frontier there (PV-Tuning 1.01 to 1.14 bpw) gives Llama 2 70B ZS 64.58, about Llama 2 7B FP16 (64.80). Llama 3 70B at 1.01 bpw gives ZS 51.47, well under Llama 3 8B FP16 (68.61).
  - On the EXL3 harness, 70B at 1.8 bpw (8.187, about 15.4 GB) is no better than 8B at 3.0 bpw (8.159, about 3 GB).
  - Per byte, the 8B class wins on this card.
- **1B models degrade far more.** Llama 3.2 1B at EXL3 2.0 bpw is 1.64x its 8 bpw PPL, against 1.38x for 8B and 1.81x for 70B relative to 6 to 8 bpw. GGUF IQ2_XXS on 1B is 3.7x.

### Gaps
- No primary source gives measured batch-1 decode for any sub-2-bit trellis or VQ format on an RTX 3080 Ti or 3090. The nearest are QTIP 2-bit on a 3090 and Bonsai 1-bit on a 3060 Laptop (81 tok/s, vendor).
- exllamav3 publishes no tokens/s tables. Its perplexity charts do not state the dataset or ctx.
- llama.cpp publishes no tg128 table for IQ1_S, IQ1_M or IQ2_XXS on CUDA. Only the Q4_0 scoreboard exists.
- No paper reports perplexity for every GGUF reference type and a 2025-2026 VQ or trellis method on identical evaluation code apart from the exllamav3 charts.

## 2. Vector and trellis quantization: QTIP, QuIP#, AQLM, PV-Tuning, VPTQ, HIGGS, GPTAQ, CDQuant and 2026 work

### Takeaway
Trellis-coded quantization (QTIP and its 2026 successors YAQA, XOR-Trellis, CurveTQ, BCJR-QAT) and lattice VQ (LLVQ, ICML 2026) hold the 2-bit accuracy frontier. YAQA's output-KL objective cuts KL by about 30% over LDLQ at equal bits.

Fine-tuning costs range from about 1 to 50 GPU-hours for 8B-70B Hessian or recovery work up to several 8-GPU-days for PV-Tuning convergence. Decode kernels run at 2 to 4.9x FP16. That is well short of the 7 to 8x a 2-bit byte count would allow.

### Cited Findings

**QTIP (Tseng et al., NeurIPS 2024)** -- [arXiv 2406.11235](https://arxiv.org/html/2406.11235)
- **Llama 2, ctx 4096, HYB code with fine-tuning, PPL (W2 / C4):**
  - 7B: 2-bit 5.86 / 7.73; 3-bit 5.28 / 6.87; 4-bit 5.17 / 6.69. FP16 5.12 / 6.63.
  - 70B: 2-bit 3.70 / 5.48. FP16 3.12 / 4.97.
  - QuIP# at 2-bit: 6.19 (7B), 3.91 (70B). AQLM at 2-bit: 6.14 (7B), 3.83 (70B).
- **Llama 2 at 2 bits, no fine-tuning:**
  - 7B: 1MAD 7.05, 3INST 6.82.
  - 70B: 1MAD 3.87, 3INST 3.90.
  - QuIP# without fine-tuning: 8.22 (7B), 4.16 (70B).
- **Llama 3, ctx 8192:**
  - 8B: 2-bit 7.33 / 8.62; 3-bit 6.01; 4-bit 5.67. BF16 5.54 / 7.10.
  - 70B: 2-bit 4.97 / 6.80; 3-bit 3.18; 4-bit 2.75. BF16 2.59 / 5.78.
- **Zero-shot, Llama 3 8B at 2-bit (ArcC / ArcE / BoolQ / PiQA / Wino):** QTIP 44.2 / 75.2 / 76.7 / 77.6 / 70.7. QuIP# 39.2 / 72.9 / 76.6 / 75.6 / 68.2.
- **Decode cost per weight:**
  - 1MAD: 4 or fewer instructions.
  - 3INST: 3 ALU instructions plus 1 data movement.
  - HYB: about 2 instructions amortized, with a lookup table.
- **Throughput:** RTX 6000 Ada batch 1 and RTX 3090 / A6000 tables, given in Section 1.
- **Fine-tuning cost:** not reported. The paper uses the QuIP# recipe. Hessians come from 6144 sequences of 2048 tokens for Llama 2, and 4096 sequences of 8192 tokens for Llama 3.

**QuIP# (Tseng et al., ICML 2024)** -- [arXiv 2402.04396](https://arxiv.org/html/2402.04396)
- **Method:** E8P lattice codebook plus randomized Hadamard incoherence.
- **Llama 2 70B at 2-bit:** 3.91 / 5.71 at ctx 4096; 4.16 / 6.12 at ctx 2048.
- **Fine-tuning:** "around 50 GPU-hours for a 70B parameter model".
- **Throughput:** RTX 4090 at 2-bit, 7B 170.50 tok/s (29.6% of peak bandwidth) and 70B 32.74 tok/s (56.84%).
- **In the HF Llama implementation:** QuIP# 106.3 tok/s vs FP16 33.1 vs AQLM 20.6 (7B).

**AQLM (ICML 2024) and PV-Tuning (Malinovskii et al., NeurIPS 2024)** -- [arXiv 2405.14852](https://arxiv.org/html/2405.14852)

| Model | Method | bpw | W2 / C4 | ZS | FP16 W2 / ZS |
|---|---|---|---|---|---|
| Llama 2 7B | PV-Tuning | 2.02 | 5.84 / 7.62 | 61.35 | 5.12 / 64.80 |
| Llama 2 7B | AQLM | 2.02 | 6.64 | 56.47 | |
| Llama 2 7B | QuIP# | 2.01 | 6.19 | 57.51 | |
| Llama 2 7B | PV-Tuning | 1.02 | 8.28 | 50.66 | |
| Llama 2 7B | OneBit | 1.01 | 9.73 | 50.06 | |
| Llama 2 7B | BiLLM | 1.08 | 32.48 | 41.68 | |
| Llama 2 13B | PV-Tuning | 1.97 | 5.12 | 64.92 | ZS 67.82 |
| Llama 2 70B | PV-Tuning | 2.07 | 3.78 | 70.72 | 3.12 / 72.40 |
| Llama 2 70B | PV-Tuning | 1.14 | 5.52 | 64.58 | |
| Llama 3 8B | PV-Tuning | 2.01 | 6.99 / 8.29 | 64.36 | 5.54 / 68.61 |
| Llama 3 8B | PV-Tuning | 1.01 | 11.17 | 50.01 | |
| Llama 3 8B | DB-LLM | 2.01 | 12.77 | 51.8 | |
| Llama 3 70B | PV-Tuning | 2.07 | 4.57 | 70.38 | 2.59 / 75.37 |
| Llama 3 70B | PV-Tuning | 1.01 | 8.67 | 51.47 | |
| Mistral 7B | PV-Tuning | 2.01 | 5.29 | 66.32 | 4.78 / 69.38 |
| Phi-3 Mini 3.8B | PV-Tuning | 2.03 | 6.88 | 65.70 | 5.83 / 70.5 |

- **PV-Tuning cost:**
  - "Our longest training run took 2 days on 8 GPUs to outperform all baselines and 8 days to fully converge".
  - 70B runs need 8xA100 or RAM offloading.
  - Wall-clock is up to 1.5x QuIP# fine-tuning.
- **AQLM 70B quantization cost:** 183 h, listed in the VPTQ comparison table measured on 4xA100-80GB -- [VPTQ Table 2](https://arxiv.org/html/2409.17066)

**VPTQ (Liu et al., EMNLP 2024)** -- [arXiv 2409.17066](https://arxiv.org/html/2409.17066)
- **Llama 2 7B:**
  - 2.02 bpw: 6.13 / 8.07, ZS 58.2, 39.9 tok/s, 2 h.
  - 2.26 bpw: 5.95.
  - FP16 5.12 / 6.63, ZS 62.2, 38.32 tok/s.
- **Llama 2 70B at 2.07 bpw:** 3.93, ZS 68.6, 9.7 tok/s, 19 h. For comparison, AQLM 70B took 183 h and QuIP# 70B 25 h.
- **Llama 3 8B (FP16 6.14, ZS 68.6):**
  - 4.03 bpw: 6.42.
  - 3.03 bpw: 6.97, ZS 66.7.
  - 2.08 bpw: 9.29, ZS 60.2.
- **Llama 3 70B (FP16 2.9, ZS 75.3):** 2.02 bpw gives 5.6, ZS 70.9.
- **Setup notes:**
  - Throughput uses naive Torch and Triton kernels without CUDA graphs. The GPU is not stated.
  - Costs are on 4xA100-80GB.

**HIGGS (Malinovskii, Panferov et al., arXiv 2411.17525)** -- [arXiv 2411.17525v1](https://arxiv.org/html/2411.17525v1)
- **Method:** data-free, Hadamard rotation plus MSE-optimal grids, with dynamic-programming bit allocation.
- **Llama 3.1 8B (FP16 5.607, ZS 69.31, MMLU 65.35):**
  - 3.25 bits, dynamic data-free: 6.388, ZS 66.74, MMLU 61.62.
  - 3.25 bits, GPTQ: 7.133.
  - 4.0 bits, dynamic: 5.910.
- **FLUTE kernels, RTX 4090, Llama 3.1 8B, tok/s:**

  | Batch | FP16 | 2-bit | 3-bit | 4-bit |
  |---|---|---|---|---|
  | 1 | 57 | 173 | 150 | 139 |
  | 16 | 862 | 2432 | - | - |

- **Bit range:** the portion of the paper read here contains no 2-bit or 2.5-bit Llama 3.1 8B accuracy row. The lowest dynamic budget reported is 3.25 bits (seed/mixed-precision subagent, same source).

**GPTAQ (Li et al., ICML 2025)** -- [arXiv 2504.02692](https://arxiv.org/html/2504.02692)
- **Method:** asymmetric calibration against full-precision outputs. It is about 20 more lines of code than GPTQ.
- **W2A16 per-channel with QuaRot, PPL (GPTAQ vs GPTQ):**

  | Model | FP16 | GPTQ | GPTAQ |
  |---|---|---|---|
  | L3-8B | 6.14 | 23.2 | 13.4 |
  | L3-70B | 2.85 | 18.5 | 10.7 |
  | L2-7B | 5.47 | 20.7 | 9.02 |
  | L2-70B | 3.32 | 5.60 | 5.18 |

- **W3A16 per-channel:** L3-8B 7.25 vs GPTQ 7.59.
- **Cost (W4A4):** 0.3 GPU-h for 8B and 2.7 GPU-h for 70B on one A100.

**CDQuant (Nair and Suggala, Google, arXiv 2406.17542)** -- [arXiv 2406.17542](https://arxiv.org/html/2406.17542)
- **Method:** coordinate descent replaces GPTQ's update.
- **INT2 g128, FFN weights only, PPL (GPTQ -> BCD):**
  - Gemma-1 7B: 375.153 -> 68.732.
  - Gemma-2 9B: 13.785 -> 13.662.
  - Gemma-2 27B: 12.181 -> 11.873.
  - PaLM2-Otter: 10.816 -> 9.822.
- **As a subroutine for QuIP and FrameQuant:** CD and BCD give about a 5% boost.
- **Coverage:** no Llama results, no fine-tuning, no kernel.

**YAQA, "Model-Preserving Adaptive Rounding" (Tseng, Sun, De Sa, ICML 2026)** -- [arXiv 2505.22988](https://arxiv.org/html/2505.22988); [ICML listing](https://icml.cc/virtual/2026/poster/64269)
- **Method:** Kronecker-factored Hessians of the full-model KL.
- **No-FT results, QTIP quantizer, ctx 8192:**

  | Model | Bits | LDLQ (KL / PPL) | YAQA-B (KL / PPL) | ZS (YAQA-B / BF16) |
  |---|---|---|---|---|
  | Llama 3.1 8B Instruct (BF16 PPL 6.50) | 2 | 0.356 / 9.39 | 0.241 / 8.39 | 64.32 / 69.82 |
  | Llama 3.1 8B Instruct | 3 | - | 0.044 / 6.85 | 69.31 |
  | Llama 3.1 70B Instruct (BF16 PPL 3.52) | 2 | 0.497 / 6.02 | 0.335 / 5.30 | 66.19 / 67.67 |

- **With QuIP# fine-tuning, 8B Instruct at 2-bit:** LDLQ 7.82, YAQA-B 7.60 (ZS 66.38).
- **Hessian cost:**
  - Sketch A: about 20 GPU-hours for a 10B model on 20M tokens.
  - Sketch B: about 30 GPU-hours, but works from as few as 2K sequences, about 1 GPU-hour.
- **QAT comparison:** the authors estimate Google's Gemma 3 QAT used 100x more data.

**LLVQ, Leech-lattice VQ (Qualcomm, ICML 2026)** -- [arXiv 2603.11021](https://arxiv.org/html/2603.11021); [ICML listing](https://icml.cc/virtual/2026/75287)
- **Method:** codebook-free, 24-dimensional, multi-shell.
- **2 bpw results (W2 / MMLU / CSR):**

  | Model, setting | Fine-tune | LLVQ (best config) | QTIP-3INST | QuIP# |
  |---|---|---|---|---|
  | Llama 3 8B (baseline 5.75 / 65.5 / 74.6) | no | 8.50 / 52.6 / 69.3 | 9.59 / 50.1 / 66.4 | - |
  | Llama 3 8B | input scales only, about 52M tokens | 6.99 / 55.7 / 70.7 | 7.28 / 53.5 / 69.1 | 7.92 / 48.1 / 66.7 |
  | Llama 2 7B | yes | 5.33 W2 (best) | - | - |

- **Kernel:** fused dequant plus matvec on a 4096x4104 matrix takes 11.94 us vs FP16 16.3 to 17.69 us. The GPU is not named. The authors call it "slightly slower than QTIP".

**Unfolding the Leech lattice for decode (arXiv 2609.02652, Sep 2026)** -- [arXiv 2609.02652](https://arxiv.org/html/2609.02652)
- **Format:** a 2-bit code whose served kernel stores an unfolded 4.804 bits per weight in VRAM, to avoid a codebook too large for a lookup table.
- **Qwen3-8B:** PPL 10.97 vs FP16 8.99 vs QTIP 11.17.
- **End-to-end on L40S, 8B:** 26.5 tok/s dense, 34.1 fused with FP16 head, 68.2 served with int8 head. The paper says most of the served gain comes from replacing the output head.
- **A100:** all lattice arms fall below FP16 (0.79x), being "bound by instruction issue rather than by DRAM".

**XOR-Trellis (arXiv 2610.00432, Sep 2026)** -- [arXiv 2610.00432](https://arxiv.org/html/2610.00432)
- **Format:** about 2.25 bpw, including an E5M3 scale per 32 weights. No rotation and no fine-tuning.
- **Llama 3.1 8B (ctx 4096):** 8.00 / 11.34 vs QTIP-3INST 8.48 / 11.93 vs BF16 5.84 / 8.43. ZS 64.90 vs QTIP 63.82 vs BF16 71.03.
- **Llama 3.1 70B Instruct (ctx 8192):** 5.62 vs QTIP 6.31 vs BF16 3.52.
- **Against YAQA-3INST on Llama 3.1 8B Instruct:** 8.96 vs 9.57.
- **Decode cost:** about 14 XOR2 gates plus a 4x4 FP4 palette lookup per weight, vs about twelve 32-bit adds plus logic for QTIP-3INST.
- **Throughput:** no GPU throughput is reported.

**Other 2026 trellis and VQ work**
- **CurveTQ (arXiv 2610.09212, Oct 2026):**
  - Rotation-free trellis with curvature-weighted Viterbi search.
  - 1 to 3 points higher mean downstream accuracy than QTIP and Proteus on three 4-8B Instruct models.
  - Weighted native search matches a full randomized Hadamard within about 1 point.
  - Claims the fastest decoder of the three, with no kernel timings in the abstract -- [arXiv 2610.09212](https://arxiv.org/abs/2610.09212)
- **BCJR-QAT (arXiv 2605.10655, May 2026):**
  - A differentiable BCJR relaxation of trellis coding.
  - Single-layer QAT beats QTIP-PTQ by 0.084 PPL on Llama-3.2-1B at 2 bpw.
  - Fused Triton kernel 6.57x faster than its reference -- [arXiv 2605.10655](https://arxiv.org/abs/2605.10655)
- **UniSVQ (ICML 2026):**
  - Codewords are affine transforms of integer lattices, to keep integer kernels.
  - Claims to beat SQ methods and match VQ methods with higher throughput. The abstract gives no numbers -- [arXiv 2606.10520](https://arxiv.org/abs/2606.10520)
  - Companion paper **LC-QAT (ICML 2026):** data-efficient 2-bit QAT via a linear-constrained VQ mapping -- [arXiv 2606.10531](https://arxiv.org/abs/2606.10531)
- **SAGE (arXiv 2610.14745, Oct 2026):**
  - Activation-guided spherical VQ at 2.25 bpw payload (2.50 with FP16 radius grids).
  - Qwen3-4B-Instruct average 72.81 vs BF16 82.67 vs QTIP 66.03 (QTIP at 2.00 bpw). Qwen3-30B-A3B 82.48 vs 86.51.
  - RTX 4090 batch 1: 265.2 vs 101.0 tok/s FP16.
  - Calibration: 1,920 sequences of 2,048 tokens, 20 Adam epochs -- [arXiv 2610.14745](https://arxiv.org/html/2610.14745)
- **CCQ, convolutional-code quantization (Baidu, arXiv 2507.07145, Jul 2025):**
  - 2.0 to 2.75 bits with a lookup-free, bit-shift decode.
  - ERNIE-4.5-300B-A47B in 89 GB on a single GPU; DeepSeek-V3 in 184 GB. No accuracy numbers in the abstract -- [arXiv 2507.07145](https://arxiv.org/abs/2507.07145)
- **Q-Palette (NeurIPS 2025):**
  - A family of fractional-bit TCQ and VQ quantizers with mixed-scheme allocation.
  - Llama 2 7B at 2.0 bits: 6.47 vs QTIP 6.84 (ctx 4096, no FT).
  - RTX 4090 batch 1: 272 tok/s vs QTIP 209 -- seed/mixed-precision subagent, [arXiv 2509.20214](https://arxiv.org/html/2509.20214)

### Inferences
- **Ranking at 2 bpw on Llama 3 8B, ratio to own FP16:**
  - YAQA plus FT about 1.17 (on an Instruct model)
  - LLVQ plus scale-FT 1.22
  - PV-Tuning 1.26
  - QTIP 1.27 to 1.32
  - QuIP# 1.38 to 1.42
  - VPTQ 1.51
  - GPTAQ/GPTQ-class scalar 2.2 to 3.8
  - The output-KL objective (YAQA) and better codes (trellis, Leech) are orthogonal and could stack. No paper combines YAQA with LLVQ.
- **2026 work drops random rotations.** XOR-Trellis and CurveTQ remove the Hadamard rotation and its runtime transform, which removes a per-token cost from decode.
- **Light fine-tuning is cheap at 8B.** LLVQ's scale-only FT uses about 52M tokens, and YAQA's Hessians take 1 to 30 GPU-hours. PV-Tuning's multi-day 8-GPU budget buys the 1-bit-class points.
- **Codebook decode cost decides kernel speed.**
  - AQLM's 1x16 codebook runs at 81.5 tok/s vs QTIP's 188 on the same GPU.
  - The unfolding paper had to spend 4.8 bits per weight in VRAM to make a Leech code fast, which removes the byte advantage.
  - Computed codes (QTIP 1MAD/3INST/HYB, XOR-Trellis) are the only family with both frontier accuracy and about 2 to 4 ALU ops per weight.

### Gaps
- No 2-bit Llama 3 8B numbers from HIGGS, and no Llama numbers from CDQuant.
- No QTIP or PV-Tuning fine-tuning GPU-hours for 8B specifically.
- No GPU tokens/s for XOR-Trellis, CurveTQ, UniSVQ or LLVQ end to end.
- No 1.5-bit points for trellis methods. EXL3's 1.6 and 1.8 bpw chart points are the only trellis data below 2 bpw found.

## 3. Rotations and outlier handling as enablers of lower bit widths

### Takeaway
Incoherence processing (randomized Hadamard in QuIP#, QTIP, HIGGS) is what made weight-only 2-bit PTQ viable in 2024. QuaRot, SpinQuant and FlatQuant mainly serve W4A4.

2025-2026 results split. Learned rotations help scalar 2-bit (ButterflyQuant). Curvature-weighted trellis search now matches or beats Hadamard-rotated trellis without any rotation (XOR-Trellis, CurveTQ), while YAQA collapses without it.

### Cited Findings
- **SpinQuant (ICLR 2025)** -- [arXiv 2405.16406](https://arxiv.org/abs/2405.16406)
  - W4A4KV4 Llama-2 7B: zero-shot gap to full precision 2.9 points.
  - Llama-3 8B: up to 45.1% gap reduction vs QuaRot.
  - Random rotations vary by up to 13 points.
- **FlatQuant (ICML 2025)** -- [arXiv 2410.09426](https://arxiv.org/abs/2410.09426)
  - W4A4 Llama-3-70B: under 1% accuracy drop, 7.5% over SpinQuant.
  - Up to 2.3x prefill and 1.7x decode speedup vs FP16.
  - No weight-only 2-bit results in the abstract.
- **QuaRot + GPTAQ at W4A4** -- [arXiv 2504.02692](https://arxiv.org/html/2504.02692)
  - Llama 3 8B: 7.36 vs FP16 6.44, ZS 69.6 vs 74.3.
  - Llama 3 70B: 6.94, ZS 69.1 vs 80.4.
- **ButterflyQuant (arXiv 2509.09679, v3 Feb 2026)** -- [arXiv 2509.09679](https://arxiv.org/html/2509.09679)
  - Learnable butterfly (Givens) rotations, W2A16.
  - LLaMA-2-7B: 15.40 vs SpinQuant 16.43 vs GPTQ 36.77 (FP16 5.47).
  - LLaMA-2-13B: 10.24 vs QuIP 13.75 vs SpinQuant 17.10.
  - Calibration takes 5 to 10 minutes on 128 samples, vs hours for SpinQuant.
  - These scalar-grid results are far behind trellis and VQ at the same bits (QTIP no-FT 6.82 on 7B).
- **Ablation inside LLVQ (Llama 2 7B, no fine-tuning)** -- [arXiv 2603.11021](https://arxiv.org/html/2603.11021)
  - E8P (QuIP#) with input and output Hadamard: 7.96.
  - LLVQ shape-gain with no rotation: 7.27.
- **XOR-Trellis** -- [arXiv 2610.00432](https://arxiv.org/html/2610.00432)
  - Without rotation, curvature (D-weighted) search beats Hadamard-rotated QTIP-3INST on Llama 3.1 8B (8.00 vs 8.48) and 70B Instruct (5.62 vs 6.31).
  - YAQA-3INST without rotation "collapses to thousands of perplexity points".
- **CurveTQ** -- [arXiv 2610.09212](https://arxiv.org/abs/2610.09212)
  - Weighted native search matches a full-dimension randomized Hadamard within about one point of downstream accuracy.
- **"Transforms for LLM Quantization: The Great Inversion and Format Co-Design" (arXiv 2608.25188, Aug 2026)** -- [arXiv 2608.25188](https://arxiv.org/abs/2608.25188)
  - Proves that energy-concentrating transforms (KLT-like, favoured by variable bit allocation) and flattening transforms (Hadamard-like, favoured by uniform grouped formats) conflict.
  - Concludes that the best transform depends jointly on allocation regime and number format.
  - Surveys 43 transform methods across 200 works.
- **Spectral rotations with auto-round, W2A16, models of 1.5B or less (arXiv 2605.25203)** -- [arXiv 2605.25203](https://arxiv.org/abs/2605.25203)
  - 15 to 58% PPL reduction over vanilla auto-round.
  - Qwen2.5-1.5B: 36.93 -> 21.84. No comparison against QuaRot or SpinQuant.

### Inferences
- For a runtime targeting weight-only sub-2-bit, activation-outlier rotations (QuaRot/SpinQuant/FlatQuant) matter little unless activations are also quantized for compute-bound verification.
- The live choice is between Hadamard incoherence, which needs an online fast transform per layer input or output, and 2026 rotation-free curvature-weighted trellis search, which needs none at decode.
- The Great Inversion result implies that a mixed-precision allocator and a Hadamard rotation pull in opposite directions. A runtime doing per-row or per-layer allocation at a 1.8-bit average should not assume Hadamard is optimal.

### Gaps
- No published W2A16 or sub-2-bit study compares QuaRot, SpinQuant, FlatQuant or ButterflyQuant against trellis or VQ at matched bits on Llama 3 8B.
- No measured decode overhead of the online Hadamard in QuIP#/QTIP kernels at batch 1 vs a rotation-free trellis kernel.

## 4. Binary and ternary conversion of pretrained models, and decode kernels

### Takeaway
Converting a pretrained model to 1 to 1.6 bpw without training still costs most of its quality (Llama 3 8B PTQ: PPL 3.7x to 9x FP16), with BTC-LLM as an unreproduced outlier. With roughly 10B to 30B tokens of QAT or distillation, ternary or binary Llama 3 8B reaches about 1.4 to 1.55x FP16 PPL and loses 5.6 to 6.8 ZS points (ParetoQ).

The fastest measured consumer-GPU numbers for 1 to 1.6 bpw are 281.66 tok/s (ternary Llama2-7B) and 368 tok/s (vendor 1-bit Bonsai 8B) on an RTX 4090. Both run through llama.cpp-style kernels at about 42 to 53% of the byte floor.

### Cited Findings
Material in this section comes from the binary/ternary subagent's primary-source reads unless marked otherwise.

**Post-training binarization (no or light training)**
- **OneBit (NeurIPS 2024):**
  - About 1.0073 bpw. LLaMA2-7B 9.73 vs 5.47; LLaMA2-13B 8.76 vs 4.88.
  - ZS LLaMA2-7B 52.23 vs 63.70.
  - Distillation on 132k sequences for 50 epochs, about 7 days on 8xA100 for 7B. No kernel -- [arXiv 2402.11295](https://arxiv.org/html/2402.11295)
- **BiLLM (ICML 2024):**
  - 1.08 bpw. LLaMA2-7B 32.48; LLaMA2-70B 8.41 vs 3.32. Llama-3-8B reported by later papers at 38.5 to 55.8.
  - 0.5 h on one A100. No kernel -- [arXiv 2402.04291](https://arxiv.org/html/2402.04291)
  - PTQ1.61's accounting puts BiLLM at an effective 2.1 bpw -- [arXiv 2502.13179](https://arxiv.org/html/2502.13179)
- **PTQ1.61:**
  - 1.61 bpw. LLaMA-3-8B 22.90, ZS 37.20 vs 65.98, MMLU 22.9. LLaMA-2-70B 6.94.
  - About 2 h for 7B. Fake quantization only -- [arXiv 2502.13179](https://arxiv.org/html/2502.13179)
- **ARB-LLM_RC:**
  - 1.06 to 1.08 bpw. LLaMA-3-8B 27.42; LLaMA-2-70B 6.16; LLaMA-3-70B 11.10.
  - 67 to 76 min for 7B -- [arXiv 2410.03129](https://arxiv.org/html/2410.03129)
- **STBLLM, binary plus N:M sparsity:**
  - 0.8 bpw: LLaMA-3-8B 33.44; LLaMA-2-7B 13.06.
  - 0.55 bpw: LLaMA-3-8B 253.76.
  - Its 2:4 sparse-tensor-core kernel was measured only at sequence lengths 4096 to 8192 on an RTX 4090 -- [arXiv 2408.01803](https://arxiv.org/html/2408.01803)
- **BTC-LLM (ACL 2026):**
  - 1.11 bpw: LLaMA-3-8B 7.70 vs 6.14, ZS 63.71 vs 68.18.
  - 0.8 bpw: 9.49, ZS 59.3.
  - 36 min for LLaMA-2-7B on one H800. Speed claims lack a setup -- [arXiv 2506.12040](https://arxiv.org/html/2506.12040v1)
- **DBellQuant:**
  - 1.15 bpw. LLaMA-2-7B 17.91; LLaMA-2-70B 6.84. LLaMA-3-8B ZS 44.27 vs 68.64.
  - BitBLAS latency measured only at prefill shapes -- [arXiv 2507.01027](https://arxiv.org/html/2507.01027v2)
- **PT2-LLM, post-training ternarization (ICLR 2026):**
  - 1.58 bpw, ctx 2048. LLaMA-3-8B 32.19, ZS 37.79 vs 65.59. LLaMA-2-70B 6.27, ZS 55.87 vs 69.01.
  - 32 min for 7B.
  - "Up to 2.1x" end-to-end vs 2-bit in llama.cpp on LLaMA-65B -- [arXiv 2510.03267](https://arxiv.org/html/2510.03267v1)
- **CAT-Q (Intel, ICML 2026 oral per arXiv comment):**
  - PTQ to ternary (g128) with about 1M calibration tokens, 8 to 60 h on 8xA100.
  - ZS: Qwen3-8B 61.76 vs 71.57; Llama2-70B 72.72 vs 76.53.
  - RTX 4090 batch 1 via llama.cpp TQ1_0/TQ2_0: Llama2-7B 281.66 tok/s, Qwen3-4B 320.34 tok/s -- [arXiv 2606.26650](https://arxiv.org/html/2606.26650)
- **LittleBit (NeurIPS 2025):**
  - Binarized low-rank factors. Llama3-8B: 1.0 bpw 16.30, 0.55 bpw 18.47, 0.1 bpw 26.11 (FP16 6.10).
  - Llama2-70B fits in 1.98 GB at 0.1 bpw.
  - Kernel tested only on an A100 MLP layer (up to 11.6x vs FP16) -- [arXiv 2506.13771](https://arxiv.org/html/2506.13771v3)
  - LittleBit-2 (ICML 2026) Llama-3-8B numbers are secondary-source only -- [arXiv 2603.00042](https://arxiv.org/abs/2603.00042)

**Ternary or binary via QAT or distillation from pretrained weights**
- **ParetoQ (Meta, arXiv 2502.02631):**
  - LLaMA-3-8B (FP 74.6 ZS, 6.15 PPL): 2-bit 71.2 / 8.0; ternary 69.0 / 8.6; binary 67.8 / 9.5.
  - 1, 1.58 and 2 bits saturate at about 30B tokens.
  - LLaMA-3.2-3B ternary: 61.9 / 9.9 vs 65.2 / 7.7.
  - 2-bit CUTLASS kernel on H100 is 4.14x FP16 on a 16384^2 matrix, with "minimal" gains at 1B to 8B sizes -- [arXiv 2502.02631](https://arxiv.org/html/2502.02631)
- **BitNet Distillation (BitDistill, Microsoft, Oct 2025):**
  - Ternary Qwen3 0.6B / 1.7B / 4B, 10B tokens of continued pretraining on 8xMI300X nodes.
  - Task-specific only: MNLI Qwen3-4B 91.40 vs FP16-SFT 91.48.
  - CPU, 16 threads: 1,135 vs 427 tok/s (2.65x). No perplexity or MMLU -- [arXiv 2510.13998](https://arxiv.org/html/2510.13998)
- **HF "Fine-tuning LLMs to 1.58bit", Llama3-8B-1.58-100B-tokens (Sep 2024):**
  - 10B-token stage reaches WikiText PPL 12.2.
  - The 100B-token benchmark table is image-only.
  - 2-bit storage with a Triton unpack kernel -- [HF blog](https://huggingface.co/blog/1_58_llm_extreme_quantization); [model](https://huggingface.co/HF1BitLLM/Llama3-8B-1.58-100B-tokens)
- **Tequila (Tencent, ICLR 2026):**
  - Ternary LLaMA-3.2 1B / 3B with 10B tokens on 16 GPUs. 3B ZS 0.576 vs 0.636 (excluding GPQA).
  - CPU LUT kernel 3.0x -- [arXiv 2509.23809](https://arxiv.org/html/2509.23809v2)
- **EfficientQAT w2g128 (2.14 bpw):**
  - Llama-3-8B 9.80, ZS 59.37 vs 68.58. Llama-3-70B 6.38, ZS 67.57 vs 75.33.
  - 4.8 h for 7B and 40.9 h for 70B on one A100.
  - BitBLAS INT2 GEMV on A100 2.9 to 4.4x FP16 -- [arXiv 2407.11062](https://arxiv.org/html/2407.11062)
- **Bonsai 8B (PrismML, Apr 2026, vendor card):**
  - Qwen3-8B architecture, Q1_0 at 1.125 bpw (1.15 GB).
  - Average 70.5 vs Qwen3-8B 79.3; GSM8K 88 vs 93.
  - Conversion recipe and cost not stated -- [HF card](https://huggingface.co/prism-ml/Bonsai-8B-gguf)

**Kernels**
- **bitnet.cpp (CPU):**
  - Formats: I2_S 2 bpw, TL1 2 bpw, TL2 1.67 bpw.
  - 7B on an i7-13700H: about 20.6 to 20.7 tok/s vs FP16 3.30.
  - 7B on M2 Ultra: TL2 55.42 tok/s. A 100B model reaches 7.45 tok/s on M2 Ultra -- [arXiv 2502.11880](https://arxiv.org/html/2502.11880)
- **bitnet GPU W2A8 (dp4a, A100):**
  - 2.6 to 3.6x over BF16 at the kernel level.
  - BitNet b1.58 2B4T end to end: 221 ms vs 683 ms for 64 in / 64 out -- [BitNet gpu README](https://github.com/microsoft/BitNet/blob/main/gpu/README.md)
- **T-MAC (EuroSys 2025):**
  - BitNet-b1.58-3B on M2-Ultra: 30 tok/s on 1 core, 71 tok/s on 8 cores. 11 tok/s on a Raspberry Pi 5 -- [arXiv 2407.00088](https://arxiv.org/html/2407.00088)
- **LUT-GEMM:**
  - OPT-175B on 1 A100: 30.4 ms per token at 1-bit, vs FP16 on 8 GPUs at 42.4 ms -- [arXiv 2206.09557](https://arxiv.org/html/2206.09557)
- **FLUTE:**
  - LLaMA-3-8B batch 1 on A6000: W3 108.1 vs FP16 44.8 tok/s. The main FLUTE paper reports no 1- or 2-bit results -- [arXiv 2407.10960](https://arxiv.org/html/2407.10960)
  - The HIGGS paper reports FLUTE 2-bit at 173 tok/s on a 4090 (Section 2).
- **LUT Tensor Core (ISCA 2025):** simulated hardware, up to 8.2x vs FP16 Tensor Core -- [arXiv 2408.06003](https://arxiv.org/html/2408.06003)
- **TriRun (Spectra 1.1):**
  - TQ1 packs 5 trits per byte (1.6 bpw).
  - 4.9x end-to-end decode for 70B on one L40S -- [arXiv 2506.23025](https://arxiv.org/html/2506.23025)
- **Intel BITCOS ternary layout (arXiv 2609.16338, Sep 2026):**
  - 2 - z bpw, where z is the zero fraction, reaching 1.485 bpw.
  - Decode up to 1.27x on Intel Xe2 GPUs. No NVIDIA numbers -- [arXiv 2609.16338](https://arxiv.org/abs/2609.16338)

### Inferences
- **Training-free 1 to 1.6 bpw conversion of a modern 8B model is not usable** at Q4_K_M-equal accuracy. The best PTQ at about 1 bpw that has been checked against several baselines (PV-Tuning 1.01, with fine-tuning) is at 2.0x FP16 PPL and 18.6 ZS points down.
- **BTC-LLM's 1.25x at 1.11 bpw** has a PPL excess of 0.25x FP16. That is 4x smaller than PV-Tuning 1.01 bpw (1.02x), 7x smaller than LittleBit 1.0 bpw (1.67x), and 11x to 30x smaller than PTQ1.61 and BiLLM on Llama 3 8B. It needs independent reproduction before use.
- **QAT budget for ternary at 8B is roughly 10B to 30B tokens,** which is far beyond "light post-hoc fine-tuning" (ParetoQ, BitDistill, HF 1.58-bit). Even then ternary Llama 3 8B (ZS 69.0) is under 2-bit QAT (71.2) and FP16 (74.6).
- **Ternary vs 2-bit at the same byte count.** Ternary in a 1.6 to 1.69 bpw layout costs 15 to 20% fewer bytes than 2-bit, yet carries more accuracy loss than 2-bit trellis PTQ with light FT.
- **Measured 1-bit-class decode on a 4090 is 42 to 53% of the byte floor.** If kernels at 1.125 bpw reached the 88% that AWQ-4 reaches on L40S, Bonsai-class decode would be about 770 tok/s on a 4090 and about 700 on a 3080 Ti (derived).

### Gaps
- No measured RTX 3080 Ti or 3090 decode for any sub-2-bit binary or ternary format. The only sub-2-bit consumer-GPU numbers are vendor or llama.cpp-based 4090 figures.
- The BiLLM family (BiLLM, ARB-LLM, PTQ1.61, DBellQuant) has no decode GEMV implementation.
- Training cost in GPU-hours is unpublished for ParetoQ, Tequila, the HF 1.58-bit model and Bonsai.

## 5. Low-rank and structured decomposition

### Takeaway
Low-rank or structured decomposition alone does not reach the 2-bit regime's compression ratios. On 7B-8B models no training-free method keeps loss under 1 ZS point at 20% or more parameter reduction. The best 8B point is LEMS at 10% with -1.5 points. The only sub-1-point result near 25% is Llama-2-70B with SliceGPT plus LoRA recovery.

As an additive correction to 2-bit quantization (CALDERA, ODLRI) it helps accuracy at 2.2 to 2.4 bpw. It halves batch-1 decode speed vs plain QuIP#.

### Cited Findings
Material in this section comes from the low-rank/sharing subagent's primary-source reads.
- **SliceGPT (ICLR 2024)** -- [arXiv 2401.15024](https://arxiv.org/html/2401.15024)
  - Llama-2-70B ZS (with RFT): dense 76.57; 20% 75.82; 25% 75.62 (-0.95); 30% 74.30.
  - Llama-2-7B at 25% with RFT: 63.04 vs 69.00.
  - PPL at 25%: Llama-2-7B 7.24 vs 5.47; 70B 4.60 vs 3.32.
  - RFT: LoRA r=32 on about 5k Alpaca sequences.
  - Batch-1 70B: 125 -> 110 ms per token on 4 -> 3 A100s.
- **ASVD** -- [arXiv 2312.05821](https://arxiv.org/html/2312.05821)
  - Llama-2-7B: 0.95 kept 5.64; 0.90 kept 5.93; 0.80 kept 8.91 (dense 5.47).
  - Llama-2-13B: 0.90 kept 5.12 vs 4.88.
  - At 0.95 kept plus INT4 AWQ: 5.82 vs AWQ alone 5.59.
- **SVD-LLM (ICLR 2025)** -- [arXiv 2403.07378](https://arxiv.org/html/2403.07378)
  - LLaMA-7B: 20% 7.73; 40% 9.27; 60% 15.00 (dense 5.68).
  - 40% plus 2-bit QuIP#: 9.83 at 1.3 GB, vs OneBit 10.20 at 1.5 GB.
  - About 3.5 h on one A100.
- **SVD-LLM V2 (NAACL 2025)** -- [arXiv 2503.12340](https://arxiv.org/html/2503.12340v1)
  - Llama-3-8B at 20%: 8.01 vs 6.14, accuracy 0.59 vs 0.61.
  - A100, batch 4: 1.29x at 20%, 2.71x at 80%.
- **Dobi-SVD (ICLR 2025)** -- [arXiv 2502.02723](https://arxiv.org/html/2502.02723v1)
  - LLaMA-7B at 0.8 kept: 6.08 vs 5.68.
  - Llama-3.1-8B at 0.6 kept: ZS 0.63 vs 0.69. At 0.8 kept: MMLU 60.1 vs 63.3.
  - About 8 GPU-h.
- **MoDeGPT (ICLR 2025)** -- [arXiv 2408.09632](https://arxiv.org/html/2408.09632)
  - Llama-2-70B at 30%: ZS 66.79 vs 70.02.
  - Llama-3-8B at 25%: 63.96 vs 72.75.
  - No backward pass. 4 h for 7B on one A100.
- **Basis Sharing (ICLR 2025, cross-layer shared SVD basis)** -- [arXiv 2410.03765](https://arxiv.org/html/2410.03765)
  - LLaMA-7B at 20%: 7.74, ZS 0.48 vs 0.52. With LoRA: 6.75.
  - 1.57x throughput at 50% (A100, batch 512).
- **LoSparse (ICML 2023):** BERT/DeBERTa/BART only, with no LLM-scale results -- [arXiv 2306.11222](https://arxiv.org/html/2306.11222v2)
- **CALDERA (NeurIPS 2024), 2-bit Q plus 4-bit low rank** -- [arXiv 2405.18886](https://arxiv.org/html/2405.18886)
  - Llama-3-8B at 2.4 bpw: 8.22 without FT and 7.34 with factor FT, vs QuIP# 13.8 (FP16 5.54).
  - Llama-2-70B at 2.2 bpw: 3.98 vs QuIP# 4.16.
  - A10G batch 1, 7B: CALDERA 45.89 tok/s vs QuIP# 87.74 vs FP16 31.75. 70B: 4.66 vs 18.18.
- **ODLRI (ACL Findings 2025)** -- [ACL Anthology](https://aclanthology.org/2025.findings-acl.746.pdf)
  - Llama-3-8B rank 256: 8.12 vs CALDERA 8.70.
  - Llama-2-70B at 2.2 bpw: 3.94.
- **MPIFA (arXiv 2501.19090)** -- [arXiv 2501.19090](https://arxiv.org/html/2501.19090v3)
  - At 90% density: Llama-2-7B 5.69; Llama-2-70B 3.54; Llama-3-8B 6.93 vs 6.14.
- **FLAT-LLM** -- [arXiv 2505.23966](https://arxiv.org/html/2505.23966v3)
  - At 20%: Llama-3-8B 8.15, ZS 61.47 vs 68.14.
  - 1.04 to 1.23x throughput (A100, batch 128).
- **2026 work:**
  - LEMS (ICML 2026), Llama3-8B at 0.9: 6.51, ZS 61.83 vs 63.34 -- [GitHub](https://github.com/lems-svd/lems)
  - SAES-SVD (arXiv 2602.03051), LLaMA-7B at 20%: 7.17 -- [arXiv 2602.03051](https://arxiv.org/html/2602.03051)
  - AA-SVD (arXiv 2604.02119), Llama-3-8B at 0.8: 9.58 -- [arXiv 2604.02119](https://arxiv.org/html/2604.02119)
  - ADMM global low rank (NeurIPS 2026), Qwen3-8B at 30%: 9.85 -> 10.05 -- [NeurIPS listing](https://neurips.cc/virtual/2026/poster/153795)

### Inferences
- **Parameter reduction at under 1% loss is about 5 to 10%** on 7-8B models. A 10% byte reduction is worth roughly 0.2 bpw at the 2-bit level, smaller than the gain from switching IQ2_XXS to a trellis code.
- **For an 8B decode runtime, low rank is useful mainly as an accuracy patch on a 2-bit backbone.** Even there it needs a fused kernel. CALDERA's separate low-rank matmuls cut batch-1 throughput in half.
- **Composition below 2.0 bpw total is unpublished on Llama 3 8B.** SVD-LLM's 40%-plus-QuIP# point is on LLaMA-1-7B only, at 9.83 PPL.

### Gaps
- No low-rank plus quantization result below 2.0 bpw total on Llama 3 8B. No fused-kernel batch-1 throughput for any low-rank-plus-2-bit format apart from CALDERA on A10G.
- Speedups for most 2025-2026 SVD methods are batched (batch 64 to 512) or figure-only.

## 6. Cross-layer sharing and recursion conversions

### Takeaway
After uptraining, about 50% of non-embedding weights can be tied across layers (Relaxed Recursive Transformers on Gemma 2B). It needs about 60B tokens and still loses 3 to 9 few-shot points against the original checkpoint.

Every "continuous depth-wise batching" speedup is a batched-throughput figure. RRT's 2.66x is simulated with oracle early exit; MoR's 2.06x is measured on from-scratch 360M models. No recursion paper reports batch-1 decode, and tied weights do not reduce bytes read per token at batch 1 unless the shared block stays cache-resident.

### Cited Findings
Material in this section comes from the low-rank/sharing subagent's primary-source reads.
- **Relaxed Recursive Transformers (ICLR 2025)** -- [arXiv 2410.20672](https://arxiv.org/html/2410.20672)
  - Sharing: Gemma 2B 1.99B non-embedding -> 0.99B (2 shared blocks looped). Layer-wise LoRA rank 512 brings it to 1.60B.
  - Uptraining: 15B tokens, or 60B with distillation (about 2% of Gemma's 3T).
  - Few-shot: original 61.7; full-size uptrained 58.6; relaxed rank-512 58.4; pure recursive 52.5; same-size reduced baseline 38.2.
  - Depth-wise batching: 2877 vs 1080 tok/s (2.66x), simulated with oracle exiting and dummy weights on one A100 40GB. Accuracy in that early-exit setting 54.0 vs 57.3.
- **Mixture-of-Recursions (NeurIPS 2025)** -- [arXiv 2507.10524](https://arxiv.org/html/2507.10524)
  - Trained from scratch only; conversion is future work.
  - 2 recursions: -47% non-embedding parameters, few-shot 43.1 vs vanilla 42.3.
  - MoR-4: 2.06x throughput measured at maximum batch on 360M models.
- **Retrofitted recurrence (arXiv 2511.07384)** -- [arXiv 2511.07384](https://arxiv.org/html/2511.07384v1)
  - TinyLlama keeps 72.7% of parameters; Llama-3.2-1B 87.5% of body parameters.
  - About 26B to 52B training tokens.
  - At 32 recurrences each token costs 264 layer applications vs 22. This adds compute; it does not compress.
- **DeltaLLM** -- [arXiv 2501.18596](https://arxiv.org/html/2501.18596)
  - MLP sharing across up to 8-9 blocks plus low-rank deltas, 30 to 40M tokens.
  - From Llama 3.2 3B: -21% parameters, ZS 0.58 -> 0.51.
  - No speed numbers.
- **MASA, matrix atom sharing (AAAI 2026)** -- [arXiv 2508.04581](https://arxiv.org/html/2508.04581v2)
  - Training-free on 20% of attention weights of Llama 3.1 8B (about 3.3% of total): 70.93 -> 70.09.
  - 8.3% slower (A100, batch 16).
- **SHARP** -- [arXiv 2502.07832](https://arxiv.org/html/2502.07832)
  - Llama2-7B MLP storage -38 to -65% with about 0.7B recovery tokens. GSM8K 14.2 -> 3.6.
- **Depth pruning baselines:**
  - ShortGPT, 27% of Llama-2-7B layers removed: 86.3% of average retained -- [arXiv 2403.03853](https://arxiv.org/html/2403.03853)
  - SLEB, Llama-2-70B at 20%: 76.57 -> 70.81 -- [arXiv 2402.09025](https://arxiv.org/html/2402.09025)
  - LaCo -- [arXiv 2402.11187](https://arxiv.org/html/2402.11187)
  - LLM-Streamline -- [arXiv 2403.19135](https://arxiv.org/html/2403.19135)
- **Minitron:**
  - Llama 3.1 8B -> 4.5B with about 194B tokens. Depth variant MMLU 58.7 vs 65.3, GSM8K 16.8 vs 48.6 -- [arXiv 2408.11796](https://arxiv.org/html/2408.11796)
- **LayerMerge (ICML 2024):** convolutional networks only -- [arXiv 2406.12837](https://arxiv.org/abs/2406.12837)

### Inferences
- **A shared block is cheap to re-read only if it fits on chip.** At batch 1 each recursion re-reads the shared block unless it is resident. A 3080 Ti has 6 MB of L2 (GA102 spec), so a 2-bit Llama 3 8B layer (about 55 MB, derived) does not fit. Tied weights then save VRAM capacity, not bandwidth per token.
- **Light fine-tuning does not recover full quality.** 60B tokens is about 1% to 2% of modern pretraining. Sharing beyond about 20% net reduction did not recover the original model's accuracy.

### Gaps
- No batch-1 latency for any recursive or looped model, converted or from scratch.
- No recursion conversion at 7B or larger.
- No combination of cross-layer sharing with 2-bit or lower quantization.

## 7. Hashing, seed-based and procedural weight reconstruction

### Takeaway
Seed and PRNG reconstruction (SeedLM, S-Quant) works at 3 to 4 bpw. SeedLM's FPGA reaches 99.5% of the bandwidth-bound speedup. No 2025-2026 seed or hash method reports usable quality below 3 bpw, and none reports GPU decode throughput.

Computed trellis codes (QTIP 1MAD/3INST, XOR-Trellis) are the procedural codebooks that do work at 2 bpw.

### Cited Findings
Material in this section comes from the seed/mixed-precision subagent's primary-source reads unless marked otherwise.
- **SeedLM (Apple, ICLR 2025)** -- [arXiv 2410.10714](https://arxiv.org/html/2410.10714)
  - Each block of C weights is stored as a 16-bit LFSR seed, a 4-bit exponent and P 4-bit coefficients.
  - Llama 3 8B: 4-bit 7.0, 3-bit 10.1 (FP16 6.1, ctx 2048). Llama 3 70B: 4-bit 3.8, 3-bit 5.7 (FP16 2.9).
  - ZS Llama 3 70B 3-bit: 74.68 vs 79.51.
  - FPGA: 3.98x speedup on 2048x2048 against a theoretical 4x, about 99.5% of the bandwidth-bound gain (derived by subagent). It uses 120,105 LUTs vs 20,800.
  - No GPU kernel.
- **S-Quant (ICML 2026 poster):** LFSR basis matrices per block, about 3 to 4 bits, with an ASIC claim of "4x". Details not accessible -- [ICML listing](https://icml.cc/virtual/2026/poster/65030)
- **Hyper-Compression (arXiv 2409.00592, v4 Feb 2026):**
  - w = frac(theta * a) per 2-weight group.
  - LLaMA2-7B: 5.82 at a 2.60x compression ratio (about 6.2 bpw, derived by subagent).
  - Latency is reported only for UNet -- [arXiv 2409.00592](https://arxiv.org/html/2409.00592v4)
- **UltraSketchLLM (DAC 2026):**
  - Hash and sketch index-free sharing at 0.5 bpw.
  - Runs a compressed Llama-3.1-8B on an RTX 3080 Ti. Accuracy values are figure-only -- [arXiv 2506.17255](https://arxiv.org/html/2506.17255)
- **QTIP computed codes:** 1MAD at 4 or fewer instructions per weight, 3INST at 3 ALU instructions plus 1 data movement. 3INST at 2 bits without fine-tuning: Llama 2 7B 6.82, 70B 3.90 -- [arXiv 2406.11235](https://arxiv.org/html/2406.11235)
- **XOR-Trellis:** about 14 XOR2 gates plus a 4x4 FP4 palette per weight, at 2.25 bpw -- [arXiv 2610.00432](https://arxiv.org/html/2610.00432)

### Inferences
- **The working "hash" at 2 bpw is a trellis or computed code.** Pure seed search (SeedLM) needs a 16-bit seed per 8 to 12 weights plus 4-bit coefficients, which does not scale below about 3 bpw at acceptable quality.
- **The FPGA result is the only evidence that procedural reconstruction can run at the bandwidth floor.** On GPUs the analogous evidence is QTIP 2-bit at 61% of byte bound (L40S GEMV, Section 1).

### Gaps
- No GPU or CPU decode throughput for SeedLM, S-Quant or Hyper-Compression.
- No seed or LFSR method at or below 2.5 bpw on Llama-class models.

## 8. Sensitivity-driven mixed precision, and what non-uniform allocation buys near 1.8 bits average

### Takeaway
Against weak uniform scalar baselines, non-uniform allocation buys 2 to 8x lower PPL at about 2 bits. Against a strong uniform trellis quantizer at the same 2.0 bits, the gain shrinks to about 0.37 PPL and 1.4 ZS points on Llama 2 7B (Q-Palette vs QTIP), at no throughput cost.

No paper reports a matched uniform vs non-uniform comparison at exactly 1.8 bpw average. The nearest points are IQ1_M (about 1.85 to 1.96 bpw whole-model) and PTQ1.61.

### Cited Findings
Material in this section comes from the seed/mixed-precision subagent's primary-source reads unless marked otherwise.
- **Q-Palette (NeurIPS 2025)** -- [arXiv 2509.20214](https://arxiv.org/html/2509.20214)
  - Llama 2 7B at 2.0 bits: MSQ 6.47 vs QTIP 6.84 (ctx 4096). ZS 60.3 vs 58.9 (FP16 64.9).
  - Llama 3.1 8B, about 2 bits, data-free, ctx 8192: fusion-aware MSQ 7.69 vs single-scheme TCQ-2 37.95. Both run at 223 tok/s on an RTX 4090.
  - Llama 2 13B at 2.0 bits: 5.35 vs QTIP 5.62.
  - Mixed vs uniform throughput: 223 vs 223 tok/s.
- **SliM-LLM (ICML 2025)** -- [arXiv 2405.14917](https://arxiv.org/html/2405.14917)
  - Llama 2 7B at 2 bits: 11.95 (g32) or 16.01 (g128) vs GPTQ 60.45.
  - Llama 2 70B at 2 bits: 6.28.
  - A800 throughput: 61.2 vs GPTQ 83.9 tok/s at 2-bit on LLaMA-7B (-27%).
- **ScaleBITS (arXiv 2602.17698)** -- [arXiv 2602.17698](https://arxiv.org/html/2602.17698)
  - Llama 3 8B at about 2.1 bits: 11.80 vs GPTQ 340, ZS 58.53.
  - Llama 3 70B at about 2.1 bits: 7.80.
  - Mixed-kernel latency +0.35 to +1.4% vs uniform 4-bit at batch 16 to 32.
- **SqueezeLLM (ICML 2024), Llama 2 70B** -- [arXiv 2306.07629](https://arxiv.org/html/2306.07629)
  - 2.01 bits dense-only: 10.86.
  - 2.05 bits with 0.1% sparse outliers: 5.04.
  - 2.22 bits with 0.45% sparse: 4.71 (ctx 4096).
  - The dense-and-sparse kernel adds 4 to 13% latency at 3-bit.
- **llama.cpp IQ1_M (per-tensor mixed types)** -- [PR #6302](https://github.com/ggml-org/llama.cpp/pull/6302)
  - Llama 2 70B: 4.829 at about 1.85 bpw whole-model.
  - Llama 2 7B: 9.335 at about 1.96 (ctx 4096).
  - An intermediate commit with only the output tensor at Q4_K gave about 34 on 7B, per the subagent's read of the PR history.
- **PTQ1.61** -- [arXiv 2502.13179](https://arxiv.org/html/2502.13179)
  - 20% of channels at 4-bit, the rest binary, 1.61 bpw: Llama 3 8B 22.90.
  - A 30% salient fraction (1.91 bpw) performed best but was rejected for budget.
- **Squeeze10-LLM (arXiv 2507.18073):** 1.6 bpw average (80% of weights at 1 bit, 20% at 4 bits). Six-task average rises from 43% to 56% over prior PTQ -- [arXiv 2507.18073](https://arxiv.org/abs/2507.18073)
- **AMQ (EMNLP 2025)** -- [arXiv 2509.12019](https://arxiv.org/html/2509.12019v1)
  - Llama 2 7B: 2.35 bits 11.49; 2.5 bits 9.24.
  - Llama 2 70B: 2.35 bits 5.17.
  - Up to 2.67x over FP16 on L40S.
- **GAMMA (arXiv 2605.18475, QTIP backend)** -- [arXiv 2605.18475](https://arxiv.org/html/2605.18475v2)
  - Qwen3-8B ZS at 2.5 bits: 76.77 vs AMQ 72.11 vs AQLM 64.53.
  - At 3.0 bits: 81.16 vs uniform QTIP 76.33.
- **CASA (arXiv 2609.25916)** -- [arXiv 2609.25916](https://arxiv.org/html/2609.25916v1)
  - Llama 3.1 8B at 2.25 bits: 20.56 vs uniform 2-bit GPTQ 103.52.
- **HIGGS dynamic (Llama 3.1 8B at 3.25 bits):** 6.388 vs best uniform 6.643. No dynamic result below 3.25 bits -- [arXiv 2411.17525](https://arxiv.org/html/2411.17525v1)
- **Any-Precision LLM (ICML 2024)** -- [arXiv 2402.10517](https://arxiv.org/html/2402.10517)
  - Bitplane GEMV on RTX 4090, 4096^2: 3-bit 9.67 us vs FP16 38.59 us (3.99x).
  - At 4-bit within 1.2% of TRT-LLM. Nothing below 3 bits.
- **BitStack** -- [arXiv 2410.23918](https://arxiv.org/html/2410.23918)
  - Llama 3.1 70B at 22,531 MB: 8.50, ZS 71.1 vs 80.0.
  - About 13% slower than GPTQ-w2 at equal size on H800.
- **Matryoshka Quantization (ICML 2025):** int2 with OmniQuant on Mistral 7B, task average 62.75 vs 59.74 for independently trained int2 -- [arXiv 2502.06786](https://arxiv.org/html/2502.06786)

### Inferences
- **What mixed allocation buys at 1.8 bpw average.** Allocation over outlier rows and sensitive layers rescues scalar formats: SqueezeLLM's 0.1% sparse outliers cut Llama 2 70B from 10.86 to 5.04 for 0.04 bpw.
  - On a trellis base the marginal gain is small (Q-Palette vs QTIP: 6.84 -> 6.47 at 2.0).
  - Extrapolating, the bulk of the 1.8-bit benefit comes from the base code. A sensitivity allocator adds a few tenths of PPL, plus the ability to hit fractional averages like 1.8 by mixing 1.5-, 2- and 2.5-bit tensors (Q-Palette's fractional-bit palette).
- **Kernel throughput cost is avoidable.** It is 0% in Q-Palette and GAMMA (within 3%) when each tensor is uniform internally, and 27% in SliM-LLM's group-wise mixed kernel.

### Gaps
- No matched uniform vs mixed comparison at exactly 1.8 bpw on any Llama model.
- No sensitivity allocator paired with a trellis base below 2.0 bpw average.
- No per-row allocation result with a fused GPU kernel below 2 bits.

## 9. Composition: sub-2-bit weights with speculative decoding or activation sparsity

### Takeaway
Results so far come from 4-bit or higher weights; no paper measures speculative-decoding acceptance or activation-sparsity tolerance with a sub-2-bit target. At 4 bits:
- Target quantization barely changes acceptance (W4A16, W8A8).
- Quantizing the draft with GPTQ hurts acceptance.
- Quantized self-drafts reach above 90% acceptance.
- Tree verification on a 4-bit target costs 1.8x a decode step, vs 1.2x at FP16.
- Activation sparsity's degradation knee with QuIP# 2/3-bit is similar to FP16, but no fused sparse-plus-VQ kernel exists.

### Cited Findings
- **"Speculative Decoding Meets Quantization" (arXiv 2505.22179, May 2025; A100 and RTX 3090)** -- [arXiv 2505.22179](https://arxiv.org/html/2505.22179v1)
  - "Quantization has minimal impact on average accepted length", with W4A16 and W8A8 "exhibiting nearly no degradation".
  - GPTQ on the draft "leads to substantial degradation of the acceptance rate".
  - On 8B (A100, tree size 60), the verification-to-decoding ratio is below 1.2 for FP16 and W8A8, while W4A16 reaches 1.8.
  - Llama-3-70B W4A16 (A100): vanilla 35.28 tok/s; EAGLE-2 74.93 (2.12x); HierSpec with an intermediate W4A16 8B model 98.19 (2.78x).
- **QuantSpec (ICML 2025):** self-speculative draft with 4-bit weights and hierarchical 4-bit KV. Above 90% acceptance and up to about 2.5x end to end, in long-context settings -- [arXiv 2502.10424](https://arxiv.org/abs/2502.10424); [ICML listing](https://icml.cc/virtual/2025/poster/46326)
- **ML-SpecQD (arXiv 2503.13565):**
  - MXFP4 weight-only copies of a BF16 target as plug-and-play drafts: up to 2x.
  - Recursive multi-level drafting: up to 2.72x -- [arXiv 2503.13565](https://arxiv.org/abs/2503.13565)
- **TEAL (ICLR 2025)** -- [arXiv 2408.14690](https://arxiv.org/html/2408.14690v2)
  - Compatibility tested with 8-bit RTN, 4-bit AWQ and 2/3-bit QuIP# on Llama-2-7B: "the point of sharp perplexity degradation is similar across bitwidths".
  - Combined sparse-plus-quantized kernels are left to future work.
  - Batch-1 speedups at 50% sparsity: Llama-3-8B 1.63x on A6000, 1.33x on A100.
- **SpQt (arXiv 2511.04477):**
  - A zigzag layout aligns llama.cpp-style Q4_K groups with column-wise activation sparsity.
  - Up to 1.55x decode on Apple A19 Pro; 1.46x M2, 1.32x M2 Pro, 1.45x M2 Max.
  - TEAL-50% on Q4_K: 22.79 tok/s vs 14.78 baseline (1.54x) on one device -- [arXiv 2511.04477](https://arxiv.org/pdf/2511.04477)
- **SpAx (arXiv 2610.02598):** for offloaded weights, abstract-level figures give 3.86x average decode speedup with 16-bit weights and 2.06x with 4-bit weights, at no more than 10% PPL increase. These come from the abstract as indexed by search and were not checked against tables -- [arXiv 2610.02598](https://arxiv.org/html/2610.02598)
- **SPARQLe (arXiv 2606.00365):** sub-precision activation sparsity under W4A8 and W2A8, with larger gains in prefill than decode, as listed by search -- [arXiv 2606.00365](https://arxiv.org/html/2606.00365)

### Inferences
- **Self-drafting is where a sub-2-bit format fits in a speculative stack.** Combining a strong 2-bit or 1.5-bit copy as the drafter with a 3-4-bit verifier, or a 2-bit target with a tiny draft head, is untested. ML-SpecQD and QuantSpec show self-drafts from a 4-bit copy of the same model already accept well.
- **At 2 bits the verification penalty probably grows.** The 1.8x verification ratio at W4A16 comes from per-token dequantization work scaling with tree size.
  - With 2 to 4 ALU ops per weight for computed codes, a kernel must decode each weight once per step and reuse it across all tree tokens (GEMM, not repeated GEMV) to keep verification near the cost of one decode step.
- **Activation sparsity and low-bit weights compound weakly in bandwidth terms.** SpAx's gain falls from 3.86x at 16-bit to 2.06x at 4-bit. A plausible reading is that skipped columns save fewer bytes, while per-group metadata or codebook granularity limits what can be skipped. A trellis code that decodes sequentially along a row cannot skip arbitrary columns without a layout change (SpQt's zigzag addresses this only for Q4_K).

### Gaps
- No measured acceptance length or verification cost with a 1-2-bit target or drafter, for EAGLE-style heads or tree verification.
- No fused activation-sparse GEMV for VQ or trellis codes.
- No study of whether sub-2-bit weight error raises the activation-sparsity degradation threshold beyond TEAL's QuIP# figure.

## Thin or unexplored

- **Sub-2-bit decode kernels on Ampere consumer GPUs.**
  - No published trellis, VQ, binary or ternary kernel has a measured batch-1 tokens/s on an RTX 3080 Ti, 3090 or 3080 below 2 bpw.
  - Published 2-bit kernels reach 26 to 62% of the byte floor, vs 88% for AWQ-4 on L40S. A sub-2-bit GEMV at over 80% of byte bound on GA102 is unpublished.
- **lm_head and embedding compression at low bits.**
  - Most frontier methods keep lm_head at FP16. For Llama 3 8B at 2 bpw that head is 1.05 GB of 2.8 GB read per token.
  - The Leech unfolding paper found that swapping in an int8 head gave most of its end-to-end gain. No paper studies 2-bit or trellis-coded output heads with vocabulary-aware error targets.
- **Q4_K_M-equal accuracy below 2 bpw on 8B.**
  - No PTQ or light-FT method gets within 3% PPL of FP16 below about 3.5 bpw on Llama 3 8B.
  - The gap between 2-bit QAT (ParetoQ, about 30B tokens) and a short output-KL fine-tune (YAQA plus FT) at about 100M tokens or less is unexplored.
- **Trellis or lattice codes below 2 bpw.** Apart from EXL3's 1.6 and 1.8 bpw chart points there is no paper-grade trellis or lattice result at 1.0 to 1.8 bpw on Llama 3. YAQA, LLVQ, XOR-Trellis and CurveTQ all stop at 2 bpw.
- **Stacking orthogonal 2026 advances.** No paper combines:
  - output-KL rounding (YAQA)
  - Leech or rotation-free curvature trellis (LLVQ, XOR-Trellis, CurveTQ)
  - fractional-bit mixed palettes (Q-Palette)
- **Speculative decoding with sub-2-bit models.**
  - No measurement of acceptance length when the target or the drafter is at 1 to 2 bpw.
  - No measurement of multi-token verification cost for computed-code dequantization.
  - No study of self-drafting from a 1.5-bit copy against a 3-4-bit verifier.
- **Activation sparsity on VQ or trellis weights.** TEAL showed similar degradation knees on QuIP# 2/3-bit but built no kernel. No fused activation-sparse GEMV exists for any codebook or trellis format.
- **Recursion and sharing.**
  - No conversion of a 7B or larger pretrained model to a recursive or looped form.
  - No batch-1 decode numbers for any recursive model.
  - No combination of cross-layer sharing with 2-bit quantization.
  - No on-chip residency analysis of shared blocks on consumer GPUs.
- **Low rank plus sub-2-bit with fused kernels.** CALDERA and ODLRI stop at 2.2 to 2.4 bpw with unfused kernels at half QuIP#'s speed. No result exists below 2.0 bpw total on Llama 3 8B.
- **Seed and hash reconstruction at 2 bpw or below.** SeedLM and S-Quant stop at 3 to 4 bpw. No GPU kernel for LFSR reconstruction exists, and no seed method has been compared head to head with computed trellis codes at equal decode instruction budgets.
- **Matched 1.8-bit-average allocation studies.**
  - No paper compares uniform vs per-layer vs per-row allocation at exactly 1.8 bpw average on a strong base code.
  - The Great Inversion result implies the rotation choice should change with the allocation regime, which no empirical sub-2-bit paper tests.
- **1B-class models.**
  - Sub-2-bit papers rarely evaluate 1B models. EXL3's chart shows Llama 3.2 1B at 2.0 bpw is 1.64x its 8 bpw PPL, and IQ2_XXS 3.7x.
  - No method is tuned for small-model fragility at under 2 bpw.
- **Standardized bytes-per-token accounting.**
  - Papers report bpw over decoder linears only and use different ctx (2048, 4096, 8192) and FP16 baselines. For Llama 3 8B these baselines range from 5.54 to 6.14.
  - No benchmark reports bytes actually read per decoded token, including heads, scales and codebooks, alongside quality.

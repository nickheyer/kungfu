# MoE inference science for models whose experts exceed consumer GPU memory (published through Nov 2026)

Target box: RTX 3080 Ti (12 GB, 912 GB/s, PCIe 4.0 x16), 24-thread CPU, 62 GB host RAM. Reference point: a host-side expert coprocessor already reaching 112-129 tok/s on Qwen3-30B-A3B at 2 bits versus 52-55 tok/s for llama.cpp. The 5x goal is about 260-275 tok/s, a budget of roughly 3.6-3.8 ms per decoded token. Model shapes used for the arithmetic below come from the official configs: Qwen3-30B-A3B has 48 layers, hidden 2048, 32 Q heads / 4 KV heads of dim 128, 128 experts with top-8, expert intermediate 768, vocab 151,936, untied embeddings ([config](https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/config.json)). Qwen3-235B-A22B has 94 layers, hidden 4096, 64 Q / 4 KV heads, 128 experts top-8, expert intermediate 1536 ([config](https://huggingface.co/Qwen/Qwen3-235B-A22B/blob/main/config.json)). gpt-oss-120b has 36 layers, hidden 2880, expert intermediate 2880, 128 experts top-4, 64 Q / 8 KV heads of dim 64, alternating 128-token sliding-window and full attention, vocab 201,088, MXFP4 experts with attention, router, embeddings and lm_head kept unquantized ([config](https://huggingface.co/openai/gpt-oss-120b/blob/main/config.json)).

## 1. Expert route prediction: hit rates and the prefetch lead they buy

### Takeaway
Next-layer prediction from the current layer's gate input is accurate (about 96% top-1 on coarse models, 97% prefetch accuracy on edge-class fine-grained models), and 2026 sequence predictors reach about 91% recall at top-(k+3) on Qwen3-30B-A3B and gpt-oss-120B while forecasting up to 8 future tokens with only a 3.6-point recall drop. The binding constraint is lead time versus PCIe, not accuracy: one layer of compute hides only about 1.4 FP8 expert transfers on a 4090, so raising prefetch depth from 1 to 3 adds under 2 points of hit rate (MoE-CORE), and the large remaining wins in 2026 come from changing the route to fit the cache (cache-aware routing, residency-restricted verification, lightweight router adaptation), which cut misses 2-3x at 0.0 to 0.4 points of accuracy.

### Cited Findings
Classic and 2025 predictors (mostly coarse or small fine-grained models)
- Pre-gated MoE (Hwang et al.) changes the gate so that each layer selects the experts of the next layer, trading a fine-tuning step for deterministic prefetch of one layer ahead; the abstract claims lower GPU memory and better performance at unchanged quality but gives no numbers on the abstract page — [arXiv 2308.12066](https://arxiv.org/abs/2308.12066)
- HOBBIT reports next-layer top-1 expert prediction accuracy averaging 96% across layers and about 90% for 2 or 3 layers ahead, using adjacent-layer gate inputs; it replaces less critical missed experts with low-precision copies and reports up to 9.93x decode speedup over prior offloading systems on edge devices, built on llama.cpp — [HOBBIT HTML](https://arxiv.org/html/2411.01433v2), [abstract](https://arxiv.org/abs/2411.01433)
- Fate (cross-layer gate) reports 97.15% prefetch accuracy and a 99.08% expert hit rate with a shallow-favoring cache, up to 14 tok/s on Qwen1.5-MoE on edge hardware — [arXiv 2502.12224](https://arxiv.org/html/2502.12224v2)
- ProMoE trains a small per-layer predictor on the layer input; it reports average decode speedup 2.07x (up to 5.02x) and prefill 2.20x (up to 3.21x) over existing offloading — [arXiv 2410.22134](https://arxiv.org/abs/2410.22134). A secondary summary gives its predictor accuracy as 84.7% versus 58.3% for token-based and 66.9% for skip-based prediction — [alphaXiv overview](https://www.alphaxiv.org/overview/2410.22134v3) (secondary source, not checked against the paper)
- AdapMoE (ICCAD 2024 DOI 10.1145/3676536.3676741) uses the next layer's gate directly for prefetch plus sensitivity-based adaptive gating; it reduces activated experts by 25% on average "without accuracy degradation" and reports 1.35x speedup — [arXiv 2408.10284](https://arxiv.org/abs/2408.10284)
- FineMoE (formerly fMoE; EuroSys '26) uses semantic hints from the prompt plus fine-grained expert maps; reports 47% lower latency and 39% higher expert hit rate than prior systems on a six-GPU testbed — [arXiv 2502.05370](https://arxiv.org/abs/2502.05370), [EuroSys paper PDF](https://intellisys.haow.us/assets/pdf/Hanfei_FineMoE_EuroSys26.pdf)
- ExpertFlow (DAC '26) predicts expert usage across all MoE layers in one forward pass of a small transformer, then groups tokens by predicted route; reports up to 93.72% GPU memory savings and up to 10x throughput — [arXiv 2410.17954](https://arxiv.org/abs/2410.17954)
- MoE-Beyond trains a lightweight transformer on 66M activation traces of DeepSeek-V2-Lite; 97.5% accuracy, 86.6% F1, GPU cache hit rate raised from 17% to 72% when only 10% of experts fit (simulation only) — [arXiv 2508.17137](https://arxiv.org/abs/2508.17137)
- DAOP (DATE 2025) splits experts between GPU and CPU by per-sequence activation patterns and pre-computes predicted experts on the CPU; up to 8.20x over caching/prefetching and 1.35x over offloading — [arXiv 2501.10375](https://arxiv.org/abs/2501.10375)
- eMoE reruns its predictor only every few prompts because reusing experts across consecutive prompts has minimal perplexity impact; reports up to 80% memory reduction and 17% lower latency — [arXiv 2503.06823](https://arxiv.org/abs/2503.06823)
- MoE-Infinity uses activation-trace matching for cache replacement and prefetch at batch 1; 3.1x to 16.7x per-token latency improvement over vLLM, Ollama, DeepSpeed and BrainStorm — [arXiv 2401.14361](https://arxiv.org/abs/2401.14361). MoE-Beyond argues trace-similarity prediction degrades under domain shift — [arXiv 2508.17137](https://arxiv.org/abs/2508.17137)
- Throughput-oriented baselines: Klotski (multi-batch expert-aware pipeline) up to 85.12x throughput — [arXiv 2502.06888](https://arxiv.org/abs/2502.06888); MoE-Lightning up to 10.3x throughput for Mixtral 8x7B on a single 16 GB T4 — [arXiv 2411.11217](https://arxiv.org/abs/2411.11217); Fiddler (ICLR 2025, CPU computes experts) 1.26x single-batch, 1.30x long prefill, 11.57x beam search — [arXiv 2402.07033](https://arxiv.org/abs/2402.07033)
- Importance-driven scheduling substitutes low-importance missed experts with similar cached ones, raising GPU hit rate to 71% for Qwen2-57B-A14B on an A6000 — [arXiv 2508.18983](https://arxiv.org/html/2508.18983v2)

2026 predictors measured on the target model classes
- SeqMoE (arXiv Sep 2026, cs.OS) treats routing as sequence modelling with a Mamba2 seq2seq predictor over the full layer stack, triggered at L/2 and L, forecasting c=8 future steps. Recall at top-(k+3): Qwen3-30B-A3B-FP8 90.90%, Qwen3.6-35B-A3B 88.39%, gpt-oss-120B-MXFP4 90.76%, DeepSeek-V4-Flash 85.81%; 8-step rollout loses only 3.58 points on average. In the motivation study top-11 recall on Qwen3-30B and top-7 recall on gpt-oss both exceed 90% — [arXiv 2609.12978](https://arxiv.org/abs/2609.12978)
- SeqMoE hit rate versus residency (average across models), SeqMoE vs LRU: 15% residency 84.56% vs 60.02%; 25% 91.72% vs 74.24%; 35% 95.09% vs 84.49%; 40% 96.11% vs 86.37%; 45% 96.97% vs 88.50%. At 45% residency it reaches 84.82% of full-load speed on Qwen3-30B (RTX 4090, PCIe 4.0), 79.20% on gpt-oss-120B, average 80.22%, versus at most 28.03% for llama.cpp, MoE-Infinity and KTransformers baselines and 51.50% for FreeToken. Predictor cost 3.46% of time, CUDA management kernels 2.05%, predictor memory 112-228 MB; the design is CUDA-graph compatible — [arXiv 2609.12978](https://arxiv.org/html/2609.12978)
- SeqMoE measures that on Qwen3-30B-FP8, RTX 4090 over PCIe 4.0, one layer of computation overlaps the transfer of only 1.39 experts on average, and says prior cross-layer methods are limited to about 3 layers of lookahead — [arXiv 2609.12978](https://arxiv.org/html/2609.12978)
- MoE-CORE (DeepSeek-V4-Flash W4A8 on Ascend 950PR) prefetch depth vs mean exact hit rate and TPOT: depth 0 89.05% / 42.6 ms, depth 1 92.43% / 36.0 ms, depth 2 93.81% / 35.9 ms, depth 3 94.05% / 36.5 ms; capacity sweep from 12 to 120 expert slots per layer moves hit rate from 47.47% to 96.43% (12->24 adds 21.01 points, 84->120 adds 5.13). With approximate expert substitution plus MTP depth 2 it reaches 21.5 ms TPOT versus 38.0-44.8 ms mean — [arXiv 2610.01950](https://arxiv.org/html/2610.01950)
- SpecPrefetch uses a 6.48M-parameter low-rank next-layer predictor on Qwen3-VL-30B-A3B (versus 207M for ProMoE) and leaves the native top-K untouched; up to 20% decode throughput gain on a Snapdragon 8 Elite — [arXiv 2607.24787](https://arxiv.org/abs/2607.24787), details via [Pith summary](https://pith.science/paper/2607.24787). The same summary says top-8 demand exceeds the roughly two-transfer prefetch window in 89% of events (secondary)
- DALI (Feb 2026, local PCs) argues LRU and score-based caches miss high-workload experts and adds residual-based prefetching, greedy 0-1 CPU/GPU expert assignment and workload-aware replacement; the abstract gives no numbers — [arXiv 2602.03495](https://arxiv.org/abs/2602.03495)
- SPICE uses a draft model architecturally matched to the target to predict experts of later layers, serves low-confidence misses with low-rank expert surrogates plus exact residual work on the CPU, and reports up to 3.12x TPOT speedup on DeepSeek-V2-Lite and Qwen2-57B-A14B — [arXiv 2608.21240](https://arxiv.org/abs/2608.21240)

Changing the route to fit the cache (cuts the miss rate instead of predicting it)
- Mixture of Cache-Conditional Experts (2024) biases routing toward resident experts: over 50% fewer cache misses at 0.1-3% perplexity cost and under 0.1% downstream accuracy change — [arXiv 2412.00099](https://arxiv.org/html/2412.00099v2)
- Prefetch-Steered Routing (Oct 2026) conditions routing only on a deterministic logical cache filled from the next layer's pre-gate ranking; RTX 4090, PCIe 4.0 x16 measured at 22.9 GB/s. OLMoE at 50% residency: 10.7 ms/token versus 24.7 ms for exact LRU (2.3x) at -0.03 points mean over 8 tasks; at 25% residency 19.2 vs 42.5 ms (2.2x) at -0.36 points. Moonlight-16B-A3B at 37.5%: 22.6 vs 43.6 ms at -0.48 points. Qwen3.6-35B-A3B at 25%: only about 1.25x over exact LRU — [arXiv 2610.34721](https://arxiv.org/html/2610.34721)
- The same paper shows that physical-cache-dependent routing makes outputs timing-dependent: OLMoE perplexity moves 8.755 / 8.674 / 8.627 with 1 / 2 / 4 prefetch copies in flight and varies by up to 0.054 run to run, while the logical-cache variant is constant at 8.509 — [arXiv 2610.34721](https://arxiv.org/html/2610.34721)
- Cache-Aware Joint Router Adaptation (Sep 2026) post-trains the backbone plus small cache routers with LM loss plus cache-coverage loss (4 epochs on GSM8K, MATH, CSQA train splits). Qwen3-30B-A3B-Instruct-2507 with 20 cache slots per layer: LRU hit 61.19% on GSM8K vs 73.13% for the Temporal router at 0.00 accuracy change, expert traffic 1407 -> 974 MB/token; CSQA hit 56.44% -> 91.61% at -1.23 points and traffic 1579 -> 304 MB/token; the Spatio-Temporal router reaches 90.62% hard hit at -2.04 points. gpt-oss-20B with 8 slots: 65.58% -> 72.84% on GSM8K at +2.12 points. Traffic is simulated; no latency is measured — [arXiv 2609.04895](https://arxiv.org/html/2609.04895v1)
- BuddyMoE substitutes an offline-calibrated similar "buddy" expert on a miss — [arXiv 2511.10054](https://arxiv.org/pdf/2511.10054); CommitMoE (AAAI) argues mispredicted experts need not be discarded when router certainty is low and reports 1.3x to 9.4x over prior offloading frameworks at maintained quality — [AAAI proceedings](https://ojs.aaai.org/index.php/AAAI/article/view/39454/43415)
- SliceMoE (DAC 2026) caches experts at bit-slice granularity with truncation-compatible "Matryoshka" quantization so a miss can be served by loading only high-order slices; decode latency up to 1.81x (DeepSeek-V2-Lite) and 1.64x (Qwen1.5-MoE), energy up to 2.37x / 2.85x lower — [arXiv 2512.12990](https://arxiv.org/abs/2512.12990)

Host-side baselines for the large classes (cited only as baselines)
- OSDI '26 CPU-GPU hybrid system (2x EPYC 9355, 24-channel DDR5, 2x RTX 5090): FP8 DeepSeek-R1 21.5 tok/s; Q4_K_M DeepSeek-R1 28 tok/s versus KTransformers 22 and ik_llama.cpp 14 — [arXiv 2606.10493](https://arxiv.org/html/2606.10493)
- KTransformers project notes cite up to 16 tok/s on DeepSeek-R1/V3 with dual Xeon, one 24 GB RTX 4090 and 382 GB DRAM — [KTransformers tutorial](https://kvcache-ai.github.io/ktransformers/en/DeepseekR1_V3_tutorial.html)

### Inferences
- Expert sizes and PCIe transfer times at the PSR-measured 22.9 GB/s: a Qwen3-30B-A3B expert is 3 x 2048 x 768 = 4.72M parameters, 1.18 MB at 2.0 bpw, about 52 us per transfer; a Qwen3-235B-A22B expert is 3 x 4096 x 1536 = 18.9M parameters, 4.72 MB at 2.0 bpw, about 206 us; a gpt-oss-120b expert is 3 x 2880 x 2880 = 24.9M parameters, 13.2 MB at MXFP4 (4.25 bpw), about 577 us (all derived from the configs above).
- At the reference 120 tok/s on Qwen3-30B-A3B the per-layer time is about 174 us, so one layer of lead hides about 3.4 transfers of 2-bit experts; with a 90% hit rate (about 0.8 misses per layer for top-8) one layer of lead is sufficient, which matches MoE-CORE's finding that depth 1 captures nearly all the latency gain.
- For Qwen3-30B-A3B the whole expert store is 48 x 128 x 4.72M = 29.0B parameters, 7.25 GB at 2.0 bpw; with about 0.5 GB of Q4 attention and 0.26 GB of lm_head the model can be fully device-resident on 12 GB with room for KV, which makes miss science irrelevant for that model at 2 bits and decisive again only at the 2.9-4.3 bpw needed for under-1% loss (Section 4).
- For Qwen3-235B-A22B the expert store is 94 x 128 x 18.9M = 227B parameters, 56.8 GB at 2.0 bpw, which nearly exhausts 62 GB of host RAM; active experts per token are 14.2B parameters (3.55 GB at 2 bpw), so at PCIe speeds any miss rate above a few percent dominates and host-side compute plus cache-aware routing is the only route consistent with the published data.
- For DeepSeek-V3 class (671B total per the [DeepSeek-V3 report](https://arxiv.org/abs/2412.19437)) even 1.5 bpw weights exceed 62 GB, so the target box needs NVMe streaming or more than 50% expert removal before any of the cited predictors apply.
- The 2026 trend is that routing-time interventions (PSR, cache-conditional routing, joint router adaptation, AcceptMoE in Section 6) produce 2x-class gains at under 0.5 points, while pure predictors top out near 92-97% hit rates at 25-45% residency.

### Gaps
- No paper measures prediction accuracy versus lead time in microseconds on a 12 GB, PCIe 4.0 consumer card for Qwen3-235B-A22B or gpt-oss-120B; SeqMoE uses 24-96 GB cards.
- ProMoE, HOBBIT, fMoE, ExpertFlow, DAOP, Klotski and eMoE abstracts do not give per-model hit rates on 128-expert models; the HOBBIT 96%/90% figures are for coarse-grained models.
- No published head-to-head of predictors on one model and one machine; SeqMoE compares accuracy only against Patterns-MoE and reports that in a figure.
- DynMoE-style post-hoc conversion results for route prediction were not found.

## 2. Dynamic top-k and expert skipping at inference

### Takeaway
On Qwen3-30B-A3B, training-free per-token skipping stays within about 0.5% relative accuracy at 20% of routed slots skipped (6.4 experts average) and about 1.1% at 30%, then collapses (-3.0% at 40%, -7.9% at 50%). Light self-distillation (ZEDA, about 30 H200-hours) moves the frontier to about 3.9 experts average (51% of slots zeroed) at -0.7 points average (about -0.9% relative), with code benchmarks losing the most; across a batch, piggyback routing halves unique experts per layer at no significant loss on Qwen3-30B.

### Cited Findings
- ACE (Sep 2026, training-free, calibration-free, always keeps top-1) on Qwen3-30B-A3B-Instruct-2507, WikiText-2 PPL / average accuracy, BF16 7.52 / 80.67: skip 10% 7.55 / 80.39; 20% 7.66 / 80.25; 30% 7.86 / 79.77; 40% 8.19 / 78.21; 50% 8.85 / 74.30; 60% 10.86 / 63.35 — [arXiv 2609.05228](https://arxiv.org/html/2609.05228)
- ACE baselines at 50% skip on Qwen3-30B (PPL / acc): router-score 12.03 / 66.69; NAEE 12.48 / 64.40; MoDES 12.25 / 56.61; Top-PP 8.85 / 74.17; AIMER 8.86 / 73.40; ExpertSparsity 10.98 / 73.67; XShare 19.63 / 10.98; DiEP 136.92 / 3.96 — [arXiv 2609.05228](https://arxiv.org/html/2609.05228)
- ACE measured on an A100 at 60% skip, length 1024, batch 1: TPOT 85.1 -> 63.0 ms; across settings 1.31-1.41x decode and 1.72-2.25x prefill — [arXiv 2609.05228](https://arxiv.org/html/2609.05228)
- ZEDA (light adaptation) adds parameter-free zero-output experts and runs two self-distillation stages against the frozen original; on Qwen3-30B-A3B the zero-expert ratio is 51.2% and the 11-benchmark average is 74.2 versus 74.9 original; Dynamic Skipping (NAEE-style) scores 68.1 at 43.8% and AdaMoE 54.8 at 51.9%. Per-benchmark: AIME24 69.1 vs 71.0, GSM8K 95.5 vs 95.4, MATH-500 95.2 vs 94.4, LCB v5 58.2 vs 61.5, LCB v6 53.2 vs 57.1, HumanEval+ 88.5 vs 85.6, IFEval 84.3 vs 86.3. Training used 60k prompts and 30.12 hours on 8x H200; decode speedup 1.19-1.25x on one H200 in SGLang — [arXiv 2605.18643](https://arxiv.org/html/2605.18643)
- Training-free halving (Chen and Yao, Sep 2026): activate top-k1 but normalize by the probability mass of the top-k2 experts; Qwen3.6-35B-A3B 8 -> 4 experts costs 0.35 MMLU points with k2=16 versus 4.65 with standard renormalization; Qwen3.5-397B-A17B 10 -> 5 costs 0.55 points; removing renormalization is catastrophic; only MMLU reported — [arXiv 2609.04575](https://arxiv.org/abs/2609.04575)
- OEA (batch-aware piggyback routing, no retraining): each token keeps top-k0 and adds lower-ranked experts only if already in the batch union. Qwen3-30B-A3B on H100, batch 16: unique experts per layer 48.8 -> 25.1 at k0=3; MoE-layer latency 175.7 -> 106.8 us (-39%); AIME24 80.0 vs 80.4, GPQA 58.6 vs 60.2, LiveCodeBench 61.2 vs 62.1, MATH-500 93.5 vs 92.8, reported as not statistically significant. Qwen3-235B-A22B needs k0=5 (-2.4 on LiveCodeBench) — [arXiv 2511.02237](https://arxiv.org/html/2511.02237)
- MoDES (multimodal) retains 97.33% of performance at 88% skipping on Qwen3-VL-MoE-30B-A3B — [arXiv 2511.15690](https://arxiv.org/html/2511.15690); ACE measures MoDES much lower on text Qwen3-30B (56.61 at 50%) — [arXiv 2609.05228](https://arxiv.org/html/2609.05228)
- AdapMoE reduces activated experts 25% without accuracy degradation (1.35x) — [arXiv 2408.10284](https://arxiv.org/abs/2408.10284)
- Not All Experts are Equal (NAEE) on Mixtral 8x7B: pruning 2 of 8 experts allows a single 80 GB GPU with 1.2x speedup at about 2.9 points loss task-agnostic, plus a dynamic skipping rule — [arXiv 2402.14800](https://arxiv.org/html/2402.14800v2), [code](https://github.com/Lucky-Lance/Expert_Sparsity)
- MoE-CORE uses approximate expert substitution as part of its 21.5 ms TPOT configuration on DeepSeek-V4-Flash — [arXiv 2610.01950](https://arxiv.org/html/2610.01950)

### Inferences
- Under the 1% budget, the training-free ceiling on Qwen3-30B-A3B is about 20-25% fewer expert reads per token; ZEDA-style adaptation at about 30 GPU-hours roughly doubles that to about 50%, which cuts active expert bytes per token by about 2x at the same precision.
- Skipping reduces bytes only for experts that would have been read; on a host-coprocessor design the best target is skipping slots that are both low-weight and non-resident, which combines ACE/ZEDA with the cache-aware routing results of Section 1, but no paper evaluates that conjunction on Qwen3-30B-A3B.
- The k2 renormalization trick is a one-integer change that should be tested before any skipping rule because it isolates the gain-calibration error that makes naive top-k reduction look worse than it is.

### Gaps
- No skipping results on gpt-oss-20B/120B (top-4 of 32/128) were found; top-4 leaves less room than top-8.
- No published accuracy curve at 2-bit weights plus skipping; all skipping numbers are on BF16 or FP8 weights.
- DynMoE post-hoc conversion numbers on Qwen3-class models were not found.

## 3. Post-hoc expert pruning and merging

### Takeaway
On Qwen3-30B-A3B no published one-shot pruning or merging configuration stays within 1% across task families: REAP at 25% keeps code, math and creative writing within about 1.3% but loses 7.8% relative on multiple-choice knowledge, PRIME at 25% loses about 2% on MC and coding, and MoE-XBench finds 19.35% pruning already raises PPL by 28.5% and no pruned configuration within 1%. Pruning also leaves per-token active bytes unchanged (top-8 still runs); its only benefit for this hardware is a smaller resident set and therefore fewer misses.

### Cited Findings
- REAP (router-weighted expert activation pruning, arXiv v3 May 2026) on Qwen3-30B-A3B, baseline Code 0.558 / Creative 0.811 / Math 0.887 / MC 0.721: REAP 25% 0.551 / 0.804 / 0.888 / 0.665; REAP 50% 0.541 / 0.718 / 0.857 / 0.503. At 25% Frequency 0.553 / 0.807 / 0.888 / 0.600, EAN 0.554 / 0.811 / 0.892 / 0.603, M-SMoE 0.537 / 0.805 / 0.886 / 0.558, HC-SMoE 0.505 / 0.497 / 0.849 / 0.674 — [arXiv 2510.13999](https://arxiv.org/html/2510.13999v3)
- REAP on Kimi-K2-Instruct W4A16, baseline Code 0.631 / MC 0.780: 25% 0.640 / 0.773; 50% 0.624 / 0.643 — [arXiv 2510.13999](https://arxiv.org/html/2510.13999v3). REAP reports 4-bit AWQ of the full model beats a 50%-pruned 16-bit REAP model at half the checkpoint size (per search summary of the paper) — [REAP PDF](https://arxiv.org/pdf/2510.13999)
- Cerebras reports REAP at 25% within about 1 point and 95.9% code retention at 50% on Qwen3-30B-A3B (vendor-reported) — [Cerebras blog](https://www.cerebras.ai/blog/reap)
- The vLLM llm-compressor REAP example on Qwen3-30B-A3B-Instruct-2507 reports baseline 0.9672, 25% 0.9559, 50% 0.9653, with weights 56.93 -> 29.92 GiB at 50% (single-metric example, figures taken from a search summary of the docs page) — [llm-compressor docs](https://docs.vllm.ai/projects/llm-compressor/en/latest/examples/reap_expert_pruning/)
- PRIME (Oct 2026, frequency-aware unified pruning and merging via ILP) on Qwen3-30B-A3B at 96 experts: MC avg 70.46 vs 71.88 original (REAP 68.95, HC-SMoE 65.88); coding avg 82.49 vs 84.18 (REAP 82.00); GSM8K 91.21 vs 91.43; LongBench 41.75 vs 41.00; MMLU 74.89 vs 77.73 (REAP 72.03); removing frequency weighting drops MC to 66.77 — [arXiv 2610.13982](https://arxiv.org/html/2610.13982v1)
- MoE-XBench (Aug 2026) on 10 MoE models: 19.35% pruning raises PPL 28.5% on Qwen3-30B-A3B; no pruned configuration within 1% (prune-only Score_perf 96.96); expert pruning is the dominant degradation source; composable compression is not predictable from standalone results; prune-only gives 0.94-1.03x speed on H100 while Q4_K_M gives 1.38-1.51x; Q4_K_M quant-only on Qwen3-30B-A3B is within 1% (PPL 7.41 vs 7.36, Score_perf 99.91) — [arXiv 2608.21693](https://arxiv.org/html/2608.21693)
- ConMoE on Qwen3-30B-A3B at 50%: original 73.58, frequency pruning 70.31, M-SMoE 68.38, HC-SMoE 45.86, ConMoE 71.19 (per search summary) — [arXiv 2605.29350](https://arxiv.org/html/2605.29350)
- Router KD (Feb 2026) updates only the router to match the original next-token distribution after pruning, editing or merging; recovery is larger for fine-grained MoEs; the abstract gives no numbers — [arXiv 2603.02217](https://arxiv.org/abs/2603.02217)
- Training-free halving paper reports that balanced, domain-specialized routing leaves limited room for expert pruning and that expert identity matters more than expert weighting — [arXiv 2609.04575](https://arxiv.org/abs/2609.04575)
- MergeMoE (expert-output merging) evaluates Qwen3-30B-A3B at 128 -> 64 experts per merged layer — [arXiv 2510.14436](https://arxiv.org/pdf/2510.14436)
- ProQuant jointly decides pruning (0 bits) and 2/3-bit assignment by ILP on Qwen3-30B-A3B (Section 4) — [arXiv 2610.30779](https://arxiv.org/html/2610.30779)

### Inferences
- The fraction removable at under 1% loss on Qwen3-30B-A3B class models, one-shot, is below 20% and likely near 10% when MC/knowledge benchmarks count; generation-heavy tasks (code, math) tolerate 25%.
- Because pruning does not reduce top-8 reads, its value here is residency: 25% fewer experts turns a 2.9 bpw expert store from about 10.5 GB to about 7.9 GB for Qwen3-30B-A3B, which may restore full device residency at near-1% quality; for Qwen3-235B-A22B the same 25% turns a 2 bpw store from 56.8 GB to about 42.6 GB, freeing host RAM.
- Router KD is the cheapest light-training step to pair with any pruning and is untested on Qwen3-30B-A3B in the found sources.

### Gaps
- EEP (evolutionary expert pruning), MC-SMoE and MoE-Pruner results on Qwen3-30B-A3B, gpt-oss and Qwen3-235B were not found in the sources examined.
- No pruning results for gpt-oss-20B/120B were found.

## 4. Per-expert mixed precision

### Takeaway
Published PTQ on Qwen3-30B-A3B reaches under 1% average loss only near 2.9-4.3 bits per weight read: DuoRate at 2.9 bpw read is 0.98 points below BF16 (about 1.3% relative), MXFP4 at 4.3 bpw is 0.64 below, and the best published results between 2.0 and 2.6 bpw lose 2.5 to 6.5 points. Usage- and sensitivity-weighted allocation helps most at 1.5-2.5 bits (ProQuant +16.0 to +42.6 points over MxMoE, +5.9 over uniform GPTQ at 2.0 bits), but the 2-bit setting the reference system runs is already outside the 1% budget by these benchmarks.

### Cited Findings
- DuoRate (Oct 2026) on Qwen3-30B-A3B-Base, average of MMLU-Pro, MMLU, GSM8K with BF16 77.14: DuoRate 2.3 bpw 73.63 (PPL 9.97); 2.6 bpw 74.64; 2.9 bpw 76.16 (PPL 9.07); uniform p2 2.27 bpw 73.55; uniform p3 3.27 bpw 75.20; MXFP4 4.3 bpw 76.50; NVFP4 4.5 bpw 76.71; HQQ 2.3 37.98; GPTQ 2.3 26.62; MiLo 2.3 34.93; MoEQuant 2.5 57.68; BitsMoE 2.54 stored 67.32 — [arXiv 2610.13328](https://arxiv.org/html/2610.13328)
- DuoRate stores one 4.27 bpw tier and reads a per-expert prefix at p2/p3/p4 chosen by a Lagrangian with one shared multiplier; routing frequency cancels from each expert's decision and only sets the budget. Computed decode traffic: BF16 3623.9 MB/token, DuoRate 2.3 / 2.6 / 2.9 bpw 520.9 / 588.9 / 656.8 MB/token, resident 14.4 GiB; throughput not measured (needs a fused prefix-read kernel) — [arXiv 2610.13328](https://arxiv.org/html/2610.13328)
- ProQuant (Oct 2026) ILP over widths {0, 2, 3} weighted by routing, on Qwen3-30B-A3B-Instruct-2507 with BF16 average 80.94: 2.5 bits 78.45 (MxMoE 62.41, GEMQ 60.88); 2.0 bits 74.49 (GPTQ 68.64, MxMoE 49.89); 1.5 bits 67.30 (MxMoE 24.69); model size 61.06 GB -> 8.09 GB at 1.5 bits — [arXiv 2610.30779](https://arxiv.org/html/2610.30779)
- MxMoE (2025): 2.4 lower WikiText-2 PPL than GPTQ at 2.25 bits; up to 3.4x over full precision and up to 29.4% over uniform quantization at equal accuracy (W5A5), with generated mixed-precision GroupGEMM kernels — [arXiv 2505.05799](https://arxiv.org/abs/2505.05799)
- Q-Strata hierarchical allocation over 18,432 expert linear layers of Qwen3-30B-A3B reports lower WikiText-2 PPL than MxMoE at all three budgets — [arXiv 2608.30564](https://arxiv.org/html/2608.30564v1)
- BitsMoE allocates bits in spectral space, comparing against MiLo, MxMoE and GEMQ at 2 bits on Qwen3-30B-A3B, with 49.00x / 16.47x faster offline quantization than MxMoE / GEMQ — [arXiv 2606.00079](https://arxiv.org/html/2606.00079)
- DynaExq switches expert precision at runtime on Qwen3-30B-A3B-Instruct rather than fixing a per-expert map offline — [arXiv 2511.15015](https://arxiv.org/pdf/2511.15015); MODE assigns per-expert bits by ILP for Qwen3-VL-30B-A3B — [arXiv 2606.17118](https://arxiv.org/html/2606.17118v1)
- HOBBIT loads missed, less critical experts at low precision (up to 9.93x decode) — [arXiv 2411.01433](https://arxiv.org/abs/2411.01433); SliceMoE serves misses from high-order bit slices (DAC 2026) — [arXiv 2512.12990](https://arxiv.org/abs/2512.12990)
- MoE-XBench: Q4_K_M on Qwen3-30B-A3B within 1% (PPL 7.41 vs 7.36), but Qwen3-235B quant-only is 2.3% worse in PPL (4.41 vs 4.31); IQ1_M is dequantization-bound on H100 — [arXiv 2608.21693](https://arxiv.org/html/2608.21693)

### Inferences
- Under-1% active expert bytes for Qwen3-30B-A3B by PTQ alone are about 650-1000 MB/token (2.9-4.3 bpw), versus about 453 MB/token at the reference 2.0 bpw; combining 2.9 bpw with ZEDA's about 49% retained slots gives roughly 320 MB/token at an unmeasured combined loss.
- A DuoRate-style prefix layout maps directly onto a host coprocessor: the device holds high-order slices of many experts and fetches low-order slices only for heavily weighted experts, which is SliceMoE's mechanism applied with DuoRate's allocator; no paper combines the two.
- The reference system's 2-bit setting likely costs about 3.5-6.5 points versus BF16 on Qwen3-30B-A3B if its quantizer matches the best published 2-bit allocators, and more if it behaves like uniform GPTQ (-12.3 points at 2.0 bits in ProQuant's table); the quality side of the 5x goal needs its own measurement before byte reductions are counted.

### Gaps
- No under-1% mixed-precision result for gpt-oss-120B below its native MXFP4 was found; no MoQE or MiLo result on Qwen3-235B-A22B was found.
- DuoRate, ProQuant and Q-Strata report no measured throughput.

## 5. Expert-internal activation sparsity

### Takeaway
Pretrained MoE experts are highly sparse internally (84.5-90.8% of neurons maskable at 95% accuracy retention on Qwen3.5 and Llama-4-Maverick), but that operating point costs about 3.4 points, and even a 60% target costs 2.7 points on Qwen3.5-35B-A3B; no published point exists at under 1% loss. Realized gains are 1.5-2.5x in the MoE layer and only about 1.1-1.2x end to end, because the gate projection stays dense and the MoE layer is under half of runtime.

### Cited Findings
- Intra-expert activation sparsity (Park, Kim, Gu, Stoica, Cheung, May 2026) on 8 MoE models from 1B to 400B: masking cutoffs at 95% accuracy retention are Granite-1B-A400M 26.5%, Qwen3.5-35B-A3B 84.5%, Qwen3.5-122B-A10B 87.4%, Llama-4-Maverick 90.8%; gpt-oss-20B is tested but its cutoff appears only in a figure. Gate projection stays dense and produces the mask; only up and down projections are sparsified; shared experts are not sparsified — [arXiv 2605.08575](https://arxiv.org/html/2605.08575)
- Same paper, Qwen3.5-35B-A3B on MI355X at batch 128, dense average 70.9%: target 60% -> 68.2%, MoE time 0.354 -> 0.383 ms (0.92x); 80% -> 67.7%, 1.42x; 85% -> 67.5%, 1.55x; 87% -> 65.7%, 1.8x. MoE-layer peak speedups 2.5x on RTX 4090, 2.0x on H200, 1.8x on MI355X at batch 16-128; end-to-end up to 1.2x on H200 and 1.1x on MI355X, bounded because the MoE layer is about 45% of total time — [arXiv 2605.08575](https://arxiv.org/html/2605.08575)
- Universal Properties of Activation Sparsity studies Qwen3-30B-A3B with a top-p rule over all inputs and intermediates; individual experts show outlier, higher sparsity than layer averages, and critical sparsity is stable across tasks — [arXiv 2509.00454](https://arxiv.org/html/2509.00454v2)
- Mixture of Neuron Experts reports that most of Qwen3-30B-A3B's neuron-level "experts" carry negligible gate weights — [arXiv 2510.05781](https://arxiv.org/html/2510.05781)
- TEAL (ICLR 2025) reaches 40-50% model-wide activation sparsity on dense Llama-2/3 and Mistral with minimal degradation using layer-dependent magnitude thresholds on all matrix inputs — [arXiv 2408.14690](https://arxiv.org/abs/2408.14690); CATS thresholds SiLU(x W_gate) and applies the mask to W_up and W_down, reaching 56.2% MLP intermediate sparsity at 25% overall on Llama-2-7B — [arXiv 2404.08763](https://arxiv.org/pdf/2404.08763), comparison per [TEAL](https://arxiv.org/pdf/2408.14690v2)

### Inferences
- A CATS-style mask computed from the dense gate output is the variant validated inside experts; at a 60% sparsity target (already -2.7 points on Qwen3.5-35B-A3B) up and down reads shrink to 40% while the gate stays dense, so expert bytes fall to 1/3 + 2/3 x 0.4 = 60% of dense, a 40% cut at a loss above 1%.
- On a host coprocessor the more promising use is on the CPU side: a missed expert computed on the host can skip down-projection rows for masked neurons, cutting host memory traffic for misses without touching device kernels.
- Low-bit packed formats store rows in blocks, so row skipping at 2 bits needs block-aligned masks; no kernel or accuracy study exists for that combination.

### Gaps
- No intra-expert sparsity result at under 1% loss for any MoE model was found.
- No result combines activation sparsity with 2-3 bit expert weights.
- Qwen3-30B-A3B critical sparsity values are only in figures of 2509.00454, not in text.

## 6. Speculative decoding for MoE and its interaction with routing and host misses

### Takeaway
Routing overlap among consecutive draft tokens is strong in fine-grained MoEs: verifying 4 tokens on Qwen3-30B touches 19.5 unique experts per layer (2.4 x k) instead of 29.1 under uniform routing, so speculation works better than byte models predict at batch 1. Under offloading the verify union is the cost, and the 2026 answers are to size the draft tree by expert cost (SpecFit, 7-22% over cached EAGLE-3, lossless), restrict verifier experts to the resident set (AcceptMoE, 2.06x under offloading at -0.27 points, 74-77% less H2D traffic), or use self-drafts from a subset of experts (S2-MoE, DraftExpert).

### Cited Findings
- "Why Speculative Decoding Works Better Than Predicted on Sparse MoE Models" (NeurIPS 2026, 10 MoE models from Mixtral to Kimi-K2.5): unique experts per layer when verifying 4 tokens (K=3) — Qwen3-30B empirical 19.5 vs independence 26.9 vs uniform 29.1; Kimi-K2.5 23.3 vs 30.2 vs 31.0; Mixtral 5.0 vs 5.5 vs 5.5. An IPF max-entropy model predicts unique expert counts within 4%. Batch-1 target efficiency at K=3: Qwen3-30B 0.860, Qwen3-235B 0.889, Mixtral 0.488 — [arXiv 2610.38981](https://arxiv.org/html/2610.38981), [abstract](https://arxiv.org/abs/2610.38981)
- SpecFit (Oct 2026) on Qwen3-30B-A3B-Instruct-2507 with RedHat EAGLE-3: A100 40 GB with 48 expert slots per layer, summarization 8.27 tok/s versus 6.80 cached EAGLE-3, 6.64 cached AR and 4.67 uncached AR; verifies 2.88 nodes and commits 2.21 tokens versus EAGLE-3's 7.00 and 2.54. GSM8K 10.79 vs 10.08 tok/s. RTX 4090 with 32 slots, GSM8K 6.70 tok/s, 10.8% over cached EAGLE-3 and 2.51x uncached AR — [arXiv 2610.15417](https://arxiv.org/html/2610.15417)
- AcceptMoE (Aug 2026) selects verifier experts from router scores times offline commitment probabilities and, under offloading, limits eligibility to cached experts; across 12 model-task pairs mean accuracy is 0.27 points below EAGLE-3; SGLang batch 1 throughput 1.290x with all experts on GPU and 2.06x under physical offloading; H2D traffic -73.6% to -77.1% — [arXiv 2608.02989](https://arxiv.org/abs/2608.02989)
- DraftExpert (Jul 2026) self-drafts with shared + top-1 + one trained draft expert per layer: draft acceptance 85.0% (DeepSeek-V2-Lite) and 83.0% (Moonlight) versus 22.5% / 21.8% for top-1 self-drafting; verify cost normalized to one AR token falls from 2.35x (fixed K) to 1.48x (confidence and expansion truncation) to 1.25x with draft-router prefetching (88% hit / 12% waste); decode 2.99 vs 2.19 tok/s (1.36x) on an RTX 4090 with experts in host memory; average 1.45x — [arXiv 2607.24434](https://arxiv.org/html/2607.24434v1)
- S2-MoE (Aug 2026, built on llama.cpp) drafts with fewer experts in the same model (Qwen3-30B-A3B 3 of 8, gpt-oss-120B 1 of 4) and a shared KV cache; RTX 4090 with offloading 1.2-2.9x over AR; Jetson Orin NX HumanEval Qwen3 2.60x vs EAGLE-3 1.87x, gpt-oss 1.74x vs 1.19x — [arXiv 2608.15018](https://arxiv.org/html/2608.15018v2)
- SP-MoE prefetches experts predicted during drafting with a cutoff-layer policy; 1.07-3.5x TPOT over prior methods — [arXiv 2510.10302](https://arxiv.org/abs/2510.10302)
- MoE-SpeQ uses a quantized draft to predict future experts for prefetch; one review reports up to 3.3x over Mixtral-Offloading-SC on Phi-3.5-MoE and over 96% hit rate on an A100-40G — [Moonlight review](https://www.themoonlight.io/en/review/moe-speq-speculative-quantized-decoding-with-proactive-expert-prefetching-and-offloading-for-mixture-of-experts); another summary gives 2.34x on Phi-MoE; the arXiv ID now carries the title "Specter" — [arXiv 2511.14102](https://arxiv.org/abs/2511.14102) (conflicting secondary figures)
- MoESD shows that once experts saturate at moderate batch, extra verify tokens add little memory traffic — [arXiv 2505.19645](https://arxiv.org/pdf/2505.19645); Cohere makes the same saturation argument — [Cohere blog](https://cohere.com/blog/mixture-of-experts-models-get-more-from-speculative-decoding)
- OEA shows that across 16 independent sequences Qwen3-30B touches 48.8 unique experts per layer, reducible to 25.1 by piggyback routing without significant loss — [arXiv 2511.02237](https://arxiv.org/html/2511.02237)
- PSR with a block drafter on Qwen3.6-35B-A3B is 2.1-2.9x faster than the fastest AR configuration within the same GPU memory — [arXiv 2610.34721](https://arxiv.org/html/2610.34721); MoE-CORE's best TPOT uses MTP depth 2 — [arXiv 2610.01950](https://arxiv.org/html/2610.01950); DeepSeek-V3's MTP module is reported at 85-90% second-token acceptance and 1.8x TPS — [DeepSeek-V3 report](https://arxiv.org/abs/2412.19437)

### Inferences
- With 2.4 x k unique experts for 4 tokens, a verify of 8-32 rows on Qwen3-30B-A3B plausibly touches 35-60 of 128 experts per layer (extrapolated from the 4-token and 16-sequence numbers); at 2 bpw that is 41-71 MB per layer, 2.0-3.4 GB per verify pass, so if those experts are device-resident the verify is still one pass over at most the full 7.25 GB store, and speculation pays when committed tokens exceed the union growth ratio.
- With host misses, a verify pass multiplies misses roughly by the union ratio; AcceptMoE's residency-restricted verification (lossy, -0.27 points) and SpecFit's cost-aware tree (lossless) are the two published answers, and the lossless one gives only 7-22% over a cached EAGLE-3.
- A verify batch is the setting where OEA-style piggybacking is cheapest, since consecutive tokens of one sequence share more experts than independent requests; no paper evaluates piggyback routing inside a single-sequence verify batch.

### Gaps
- No published unique-expert counts for verify batches of 8-32 rows on Qwen3-30B-A3B, gpt-oss or Qwen3-235B; the largest published single-sequence measurement is 4 tokens.
- No paper reports speculative decoding on a 12 GB card with a host expert coprocessor.

## 7. Attention side of MoE decode and the non-expert share of bytes

### Takeaway
At batch 1 on an H100 in FP8, MoE layers take only 33% of Qwen3-30B forward time despite holding 67% of weights, because attention kernels run at 22-25% of peak HBM bandwidth versus 45-49% for MoE kernels; at 2-bit experts the non-expert bytes (attention projections, lm_head, KV) exceed expert bytes per token, so the next 2-3x on Qwen3-30B-A3B depends at least as much on attention, lm_head and kernel efficiency as on expert science, and almost no MoE-specific work targets that side.

### Cited Findings
- Kernel profiling at batch 1, H100 FP8: Qwen3-30B MoE share of weights 66.7%, of GPU time 33%; attention HBM bandwidth 22-25% of peak, MoE 45-49%; Mixtral-8x7B MoE 89.4% of weights and 75% of time with attention at 62% of peak; L2 cache absorbs repeated reads of co-routed experts — [arXiv 2610.38981](https://arxiv.org/html/2610.38981)
- In vLLM serving of Qwen3.5-35B-A3B, the dense MoE path is about 45% of total time, capping end-to-end gains from MoE-only speedups at about 15-20% — [arXiv 2605.08575](https://arxiv.org/html/2605.08575)
- OEA measures the Qwen3-30B MoE layer at 175.7 us per layer on H100 BF16 at batch 16 and states that attention and dense kernels are not affected by batch contents the way MoE kernels are — [arXiv 2511.02237](https://arxiv.org/html/2511.02237)
- MoE-XBench: adding KV-cache compression at 32K context drops decode to 0.61x of quant-only on H100 (dequant overhead), while Q4_K_M prune+quant gives 1.30x decode over BF16 — [arXiv 2608.21693](https://arxiv.org/html/2608.21693)
- gpt-oss-120b alternates 128-token sliding-window and full attention and keeps attention, router, embeddings and lm_head in BF16 — [config](https://huggingface.co/openai/gpt-oss-120b/blob/main/config.json)
- SeqMoE attributes 2.05% of time to CUDA management kernels and is designed to stay inside CUDA graphs — [arXiv 2609.12978](https://arxiv.org/html/2609.12978)

### Inferences
- Per-token parameter reads for Qwen3-30B-A3B from the config: experts 48 x 8 x 4.72M = 1.81B; attention projections 48 x (2048x4096 + 2x2048x512 + 4096x2048) = 0.91B; lm_head 151,936 x 2048 = 0.31B; router 12.6M. At 2.0 bpw experts, Q4 (4.5 bpw) attention and Q6 (6.56 bpw) lm_head this is about 453 + 510 + 255 = 1,218 MB, plus KV at 96 KiB per token of context (48 x 2 x 4 x 128 x 2 bytes in FP16), about 384 MiB at 4K context. Experts are about 37% of bytes at short context and about 28% at 4K.
- The byte floor at 912 GB/s for that mix at 4K context is about 1.8 ms/token (about 550 tok/s); the 5x target of 3.6-3.8 ms leaves room only if kernels run near 50% of peak bandwidth across about 48 x (attention + router + experts) launches, so CUDA-graph capture and attention bandwidth efficiency are first-order.
- Qwen3-235B-A22B attention projections are 94 x 71.3M = 6.7B parameters (64 heads x 128 dim against hidden 4096), about 3.8 GB at Q4, more than half of a 12 GB card before any expert is cached; KV is about 188 KiB per token.
- gpt-oss-120b non-expert reads are about 0.96B attention plus 0.58B lm_head parameters in BF16, about 3.1 GB per token, larger than its 1.9 GB of active MXFP4 expert bytes (36 x 4 x 24.9M = 3.58B parameters at 4.25 bpw); requantizing attention and lm_head is the largest single byte lever for that model.
- lm_head reads (0.31B for Qwen3, 0.58B for gpt-oss) can be cut by vocabulary-subset or draft-guided partial logits during speculative decode; no MoE-specific paper measures that.

### Gaps
- No batch-1 per-kernel breakdown of Qwen3-30B-A3B, gpt-oss-20B or gpt-oss-120B decode on a consumer GPU was found; the only batch-1 profile is H100 FP8.
- No study of sub-4-bit attention projections or lm_head in MoE models at under 1% loss was found; MoE quantization papers (DuoRate, ProQuant, Q-Strata, MxMoE) quantize experts only.

## Thin or unexplored

- Residency-aware skipping: skipping only slots that are both low-weight and non-resident on Qwen3-30B-A3B, Qwen3-235B-A22B or gpt-oss, with accuracy measured under real timing; PSR and cache-conditional routing exist only for OLMoE, Moonlight and Qwen3.6.
- Stacked compression at a total 1% budget: no paper reports skipping plus mixed precision plus pruning on Qwen3-30B-A3B with a single combined accuracy figure, and MoE-XBench shows losses do not add predictably.
- Sub-4-bit attention projections, lm_head and KV for MoE models at batch 1, where these now exceed expert bytes once experts are at 2-3 bits.
- Intra-expert activation sparsity at under 1% loss, and any combination of it with 2-3 bit packed expert weights or with host-side computation of missed experts.
- Verify-batch routing science for 8-32 rows of a single sequence: unique-expert counts, lossless piggybacking inside the verify batch, and draft-tree shaping by expert residency on a consumer card.
- Device-driven host expert computation through mapped memory: no paper models the device-polls-host design; published hybrids (Fiddler, DAOP, KTransformers, the OSDI '26 system) use host-driven scheduling and server CPUs.
- Prefix-read precision on misses: DuoRate's prefix tiers and SliceMoE's bit slices together would let the device hold high-order slices of every expert and fetch only refinement slices; no kernel or accuracy study exists.
- Route prediction at consumer PCIe lead times for 235B and 120B classes, reported as hit rate versus microseconds of lead; SeqMoE and MoE-CORE use 24-128 GB devices.
- gpt-oss below native MXFP4: no per-expert mixed-precision or pruning study on gpt-oss-20B/120B was found, and skipping results for top-4 routing are absent.
- Qwen3-235B-A22B on 12 GB plus 62 GB: at 2 bpw the expert store nearly fills host RAM; no paper evaluates pruning plus low-bit allocation targeted at fitting this envelope.
- DeepSeek-V3 class on 62 GB RAM: requires NVMe streaming or more than 50% expert removal plus about 1.5 bpw; flash-offloading work is mobile-focused and REAP's 50% Kimi-K2 result (code 0.624 vs 0.631, MC 0.643 vs 0.780) is the only large-model pruning datum found.
- Determinism of cache-aware routing in asynchronous host-prefetch systems: PSR shows output drift with prefetch timing; no other work audits this.
- Router KD or ZEDA-style light adaptation applied jointly with residency objectives on Qwen3-30B-A3B at 2-3 bits; the joint router adaptation paper measures traffic only, not latency.

# Citation Graph Review: modded-nanogpt and slowrun Ecosystems

**Date:** 2026-03-25
**Sources:** modded-nanogpt README (KellerJordan), slowrun README (qlabs-eng), arxiv pages, Semantic Scholar (rate-limited)

This review traces the citation graph outward from the two key repositories for the Parameter Golf competition, organizing papers by topic and assessing relevance to the 16MB / 10-minute constraint.

---

## 1. Training Efficiency / Optimizer Innovations

### 1a. Old Optimizer, New Norm: An Anthology (Muon theoretical foundation)
- **Authors:** Jeremy Bernstein, Laker Newhouse
- **Year:** 2024 | **arXiv:** [2409.20325](https://arxiv.org/abs/2409.20325)
- **Citation count:** High (foundational for Muon adoption in speedrun community)
- **Summary:** Shows Adam, Shampoo, and Prodigy are equivalent to steepest descent under particular norms when EMAs are disabled. Proposes assigning different operator norms to different tensor types based on architectural role.
- **Relevance:** This is the theoretical backbone of the Muon optimizer used by every top parameter golf entry. Understanding the norm perspective helps tune Muon's behavior for small models (e.g., the choice of spectral norm for weight matrices vs. standard norm for embeddings).

### 1b. Shampoo: Preconditioned Stochastic Tensor Optimization
- **Authors:** Vineet Gupta, Tomer Koren, Yoram Singer
- **Year:** 2018 | **arXiv:** [1802.09568](https://arxiv.org/abs/1802.09568) | ICML 2018
- **Citation count:** ~500+
- **Summary:** Maintains per-dimension preconditioning matrices that contract over remaining dimensions, providing second-order optimization at competitive per-iteration cost.
- **Relevance:** Shampoo is a precursor to Muon. Understanding its preconditioning structure explains why Muon works: the Newton-Schulz orthogonalization in Muon is a cheaper approximation of Shampoo's full preconditioning. Not directly actionable since Muon already supersedes it for parameter golf, but useful for understanding optimizer ablations.

### 1c. Scalable Second Order Optimization for Deep Learning
- **Authors:** Rohan Anil, Vineet Gupta, Tomer Koren, Kevin Regan, Yoram Singer
- **Year:** 2020 | **arXiv:** [2002.09018](https://arxiv.org/abs/2002.09018)
- **Citation count:** ~400+
- **Summary:** Practical full-matrix Adagrad variant for heterogeneous hardware (CPU+accelerator). Validated on Transformers, BERT, and ImageNet.
- **Relevance:** Shows how to scale second-order methods to real hardware. The distributed implementation patterns inform how Muon's Newton-Schulz iteration is parallelized across 8xH100s. Background reference.

### 1d. NorMuon: Making Muon More Efficient and Scalable
- **Authors:** Zichong Li, Liming Liu, Chen Liang, Weizhu Chen, Tuo Zhao
- **Year:** 2025 | **arXiv:** [2510.05491](https://arxiv.org/abs/2510.05491)
- **Citation count:** Early paper, growing
- **Summary:** Discovers that Muon's orthogonalization produces highly non-uniform neuron norms, causing certain neurons to dominate. Adds row-wise normalization after orthogonalization plus per-neuron second-order momentum. 21.74% better efficiency than Adam, 11.31% better than Muon on 1.1B pretraining. FSDP2-compatible.
- **Relevance:** Directly actionable as a Muon upgrade. The 11.31% improvement over standard Muon could translate to -0.003 to -0.005 BPB within the same 10-minute training budget. Already referenced in modded-nanogpt record history. The FSDP2 compatibility means it works with multi-GPU training.

### 1e. Cautious Optimizers: Improving Training with One Line of Code
- **Authors:** Kaizhao Liang, Lizhang Chen, Bo Liu, Qiang Liu
- **Year:** 2024 | **arXiv:** [2411.16085](https://arxiv.org/abs/2411.16085)
- **Citation count:** Growing rapidly
- **Summary:** Single-line modification to momentum-based optimizers (C-AdamW, C-Lion) that preserves convergence guarantees while speeding up training. Proven consistent across transformer pretraining tasks.
- **Relevance:** Already used in modded-nanogpt as "Cautious Weight Decay." The paper validates the theoretical soundness. Could be combined with NorMuon for a "Cautious NorMuon" variant.

### 1f. Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations
- **Authors:** Alexander Hagele, Elie Bakouch, Atli Kosson, Loubna Ben Allal, Leandro Von Werra, Martin Jaggi
- **Year:** 2024 | **arXiv:** [2405.18392](https://arxiv.org/abs/2405.18392) | NeurIPS 2024 Spotlight
- **Citation count:** ~150+
- **Summary:** Shows constant LR + cooldown achieves comparable results to cosine scheduling while enabling reusable training runs. Stochastic weight averaging enhances performance for free.
- **Relevance:** Directly informs the parameter golf training schedule. The current SOTA uses a warmdown (cooldown) phase at step 3500. This paper provides the theoretical basis and suggests SWA as an additional free improvement. The "reusable runs" insight means hyperparameter search can share early training.

### 1g. Averaging Weights Leads to Wider Optima and Better Generalization (SWA)
- **Authors:** Pavel Izmailov, Dmitrii Podoprikhin, Timur Garipov, Dmitry Vetrov, Andrew Gordon Wilson
- **Year:** 2018 | **arXiv:** [1803.05407](https://arxiv.org/abs/1803.05407) | UAI 2018
- **Citation count:** ~2000+
- **Summary:** Simple averaging of SGD checkpoints finds flatter minima and improves generalization with almost no computational overhead.
- **Relevance:** EMA (exponential moving average) is already in the SOTA parameter golf stack. SWA is the more general framework. The paper's insight about wider optima explains why EMA helps: quantization is a perturbation, and flatter minima are more robust to it.

### 1h. Pre-training under Infinite Compute
- **Authors:** Konwoo Kim, Suhas Kotha, Percy Liang, Tatsunori Hashimoto
- **Year:** 2025 | **arXiv:** [2509.14786](https://arxiv.org/abs/2509.14786)
- **Citation count:** Recent
- **Summary:** When compute exceeds available data, optimal weight decay is 30x larger than standard practice. Ensembling independently trained models outperforms single models. Ensemble benefits distillable at 8x parameter reduction retaining 83% of gains.
- **Relevance:** Referenced by slowrun for optimizer selection. The heavy weight decay finding is directly actionable: current SOTA uses WD=0.04, but this paper suggests much higher values could help when data is being reused (multi-epoch). The ensemble distillation is less relevant given the 10-minute constraint.

---

## 2. Architecture Innovations (Weight Sharing, Depth Recurrence, Residual Connections)

### 2a. Value Residual Learning (ResFormer)
- **Authors:** Zhanchao Zhou, Tianyi Wu, Zhiyun Jiang, Fares Obeid, Zhenzhong Lan
- **Year:** 2024 | **arXiv:** [2410.17897](https://arxiv.org/abs/2410.17897)
- **Citation count:** Growing
- **Summary:** Adds value residual connections alongside hidden state residuals. ResFormer achieves same validation loss with 16.11% fewer parameters and 20.3% less training data. SVFormer variant shares first layer's value embedding across all layers, halving KV cache.
- **Relevance:** Already adopted in modded-nanogpt as "value embeddings." The SVFormer variant (shared value embedding) is particularly interesting for parameter golf: sharing the value projection across layers saves parameters while the residual connection maintains quality. The 16% parameter savings directly translates to either a smaller artifact or more capacity elsewhere.

### 2b. Hyper-Connections
- **Authors:** Defa Zhu, Hongzhi Huang, Zihao Huang, Yutao Zeng, Yunyao Mao, Banggu Wu, Qiyang Min, Xun Zhou
- **Year:** 2024 | **arXiv:** [2409.19606](https://arxiv.org/abs/2409.19606)
- **Citation count:** Recent
- **Summary:** Alternative to residual connections that adjusts connection strength between features at different depths and dynamically rearranges layers. Addresses the "seesaw effect between gradient vanishing and representation collapse." Improves LLM pre-training (dense and MoE) and vision tasks.
- **Relevance:** Already identified as "Partitioned Hyperconnections" in modded-nanogpt records. Generalizes the U-Net skip connection pattern already in the SOTA. The dynamic rearrangement could be particularly useful with depth recurrence: when layers are shared, hyperconnections let each iteration of a shared layer behave differently based on depth position.

### 2c. Universal Transformers
- **Authors:** Mostafa Dehghani, Stephan Gouws, Oriol Vinyals, Jakob Uszkoreit, Lukasz Kaiser
- **Year:** 2019 | **arXiv:** [1807.03819](https://arxiv.org/abs/1807.03819) | ICLR 2019
- **Citation count:** ~900+
- **Summary:** Combines transformer parallelizability with RNN-style recurrent inductive bias. Processes sequences iteratively through shared layers with dynamic per-position halting. Turing-complete under certain conditions.
- **Relevance:** The foundational paper for depth recurrence in transformers. The per-position halting mechanism is too expensive for parameter golf's training budget, but the core idea of looping a shared layer block is directly applicable. Modern implementations (Cycle(rev), Relaxed Recursive) build on this.

### 2d. Gemma 2: Improving Open Language Models at a Practical Size
- **Authors:** Gemma Team (Google)
- **Year:** 2024 | **arXiv:** [2408.00118](https://arxiv.org/abs/2408.00118)
- **Citation count:** ~500+
- **Summary:** 2B-27B parameter models with interleaving local-global attentions and group-query attention. Smaller variants (2B, 9B) use knowledge distillation instead of next-token prediction.
- **Relevance:** Already cited in modded-nanogpt for the sliding window attention pattern. The interleaving of local (sliding window) and global attention is the inspiration for the current SOTA's attention configuration. The distillation finding for small models (2B, 9B) suggests knowledge distillation is particularly effective at smaller scales.

### 2e. Exclusive Self Attention (XSA)
- **Authors:** Shuangfei Zhai
- **Year:** 2025 | **arXiv:** [2603.09078](https://arxiv.org/abs/2603.09078)
- **Citation count:** Very recent
- **Summary:** Constrains attention to capture only information orthogonal to the token's own value vector (excluding self-position), encouraging better context modeling. Consistently outperforms standard attention up to 2.7B parameters, with gains increasing with sequence length.
- **Relevance:** Already in the SOTA parameter golf stack (applied on the last 4 layers). The finding that gains increase with sequence length suggests XSA may benefit more from longer training sequences. Could potentially be extended to more layers, though the current 4-layer application was presumably optimized.

---

## 3. Quantization for Small Models

### 3a. GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers
- **Authors:** Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh
- **Year:** 2023 | **arXiv:** [2210.17323](https://arxiv.org/abs/2210.17323) | ICLR 2023
- **Citation count:** ~1500+
- **Summary:** Quantizes 175B-parameter GPT models in ~4 GPU hours to 3-4 bits per weight with negligible accuracy degradation. Achieves 3.25x-4.5x inference speedup.
- **Relevance:** The SOTA parameter golf stack uses "GPTQ-lite" (a simplified version trying 5 clip percentiles per row) for post-training quantization refinement after int6 QAT. Understanding the full GPTQ algorithm could reveal better clip search strategies. The original GPTQ uses Hessian-based importance to determine per-column quantization order, which might improve on the simpler percentile search.

### 3b. LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale
- **Authors:** Tim Dettmers, Mike Lewis, Younes Belkada, Luke Zettlemoyer
- **Year:** 2022 | **arXiv:** [2208.07339](https://arxiv.org/abs/2208.07339)
- **Citation count:** ~2000+
- **Summary:** Identifies systematic outlier features in transformers and handles them via mixed-precision decomposition (99.9% of values in 8-bit, outliers in 16-bit).
- **Relevance:** The outlier feature insight is important even at int6: certain dimensions consistently have large values and need special treatment. The current SOTA's per-row clip search partially addresses this, but explicit outlier handling (keeping a few dimensions in higher precision) could improve quantization quality.

### 3c. BitNet: Scaling 1-bit Transformers for Large Language Models
- **Authors:** Hongyu Wang, Shuming Ma, Li Dong, et al.
- **Year:** 2023 | **arXiv:** [2310.11453](https://arxiv.org/abs/2310.11453)
- **Citation count:** ~300+
- **Summary:** BitLinear replaces standard linear layers for 1-bit weight training from scratch. Competitive with 8-bit quantization and FP16 baselines while reducing memory and energy.
- **Relevance:** Foundational for ternary quantization approaches. The BitLinear module is the building block for more advanced approaches (BitNet b1.58, Spectra). At parameter golf scale (~20M params), the quality gap between 1-bit and int6 may be too large, but the training methodology informs how to train with aggressive quantization.

### 3d. The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits (BitNet b1.58)
- **Authors:** Shuming Ma, Hongyu Wang, et al.
- **Year:** 2024 | **arXiv:** [2402.17764](https://arxiv.org/abs/2402.17764)
- **Citation count:** ~500+
- **Summary:** Ternary {-1, 0, 1} weights achieve performance parity with full-precision at large scale. Uses absmean quantization and straight-through estimator.
- **Relevance:** If ternary quantization works at small scale, it would allow ~67M parameters in 16MB (vs ~21M at int6). The 3x parameter increase could more than compensate for per-parameter quality loss. High-risk, high-reward direction. See arxiv-deep-dive.md for detailed analysis.

---

## 4. RoPE / Positional Encoding Extensions

### 4a. RoFormer: Enhanced Transformer with Rotary Position Embedding
- **Authors:** Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, Yunfeng Liu
- **Year:** 2021 | **arXiv:** [2104.09864](https://arxiv.org/abs/2104.09864)
- **Citation count:** ~2500+
- **Summary:** Encodes absolute position via rotation matrices while naturally incorporating relative position dependencies. Flexible sequence lengths, decaying inter-token dependency with distance.
- **Relevance:** RoPE is the standard positional encoding in parameter golf. The current SOTA uses "Partial RoPE" (only 16 of 64 head dimensions get RoPE). Understanding the original formulation helps reason about why partial application works: most dimensions learn relative position implicitly, so only a few need explicit positional encoding.

### 4b. YaRN: Efficient Context Window Extension of Large Language Models
- **Authors:** Bowen Peng, Jeffrey Quesnelle, Honglu Fan, Enrico Shippole
- **Year:** 2023 | **arXiv:** [2309.00071](https://arxiv.org/abs/2309.00071)
- **Citation count:** ~400+
- **Summary:** Extends RoPE-based model context windows requiring 10x less tokens and 2.5x less training steps than prior methods. Demonstrates extrapolation beyond fine-tuning context length.
- **Relevance:** Referenced in modded-nanogpt for "window size warmup with YaRN." The key technique is frequency-dependent scaling of RoPE dimensions during context extension. For parameter golf, this enables training on shorter sequences initially (faster throughput) then extending to longer sequences later (better quality), without position encoding mismatch.

### 4c. Mistral 7B (Sliding Window Attention)
- **Authors:** Albert Q. Jiang, Alexandre Sablayrolles, et al.
- **Year:** 2023 | **arXiv:** [2310.06825](https://arxiv.org/abs/2310.06825)
- **Citation count:** ~3000+
- **Summary:** Introduces sliding window attention (SWA) for efficient variable-length sequence handling alongside GQA. 7B model outperforms LLaMA 2 13B.
- **Relevance:** SWA is the single biggest technique in parameter golf (-0.033 BPB from sliding window eval). Mistral is the original popularizer. The stride-64 sliding window evaluation in the SOTA directly traces to this paper. Understanding the theoretical basis (attention beyond the window is approximated via hidden state propagation) suggests optimal window/stride ratios.

---

## 5. Multi-Token Prediction

### 5a. Better & Faster Large Language Models via Multi-token Prediction
- **Authors:** Fabian Gloeckle, Badr Youbi Idrissi, Baptiste Roziere, David Lopez-Paz, Gabriel Synnaeve (Meta)
- **Year:** 2024 | **arXiv:** [2404.19737](https://arxiv.org/abs/2404.19737)
- **Citation count:** ~200+
- **Summary:** Training LMs to predict multiple future tokens simultaneously via independent output heads on shared backbone. 12% more HumanEval, 17% more MBPP, up to 3x faster inference via speculative decoding. Higher sample efficiency at no additional training overhead.
- **Relevance:** Already referenced in modded-nanogpt records as a technique. The prediction heads are discarded at inference, so they add zero bytes to the 16MB artifact. The improved sample efficiency is critical for the 10-minute training budget. Implementation: add 2-3 linear heads (512 -> 1024 vocab each, ~2MB total during training only).

---

## 6. Test-Time Training / Adaptation

### 6a. Learning to (Learn at Test Time): RNNs with Expressive Hidden States (TTT)
- **Authors:** Yu Sun, Xinhao Li, Karan Dalal, et al.
- **Year:** 2024 | **arXiv:** [2407.04620](https://arxiv.org/abs/2407.04620)
- **Citation count:** ~150+
- **Summary:** TTT layers treat the hidden state as a machine learning model updated through self-supervised learning steps. TTT-MLP matches Transformer performance on long context while TTT-Linear and TTT-MLP keep reducing perplexity with more tokens (unlike Mamba).
- **Relevance:** The TTT concept is the theoretical foundation for the LoRA TTT approach already tried in parameter golf (baseline entry got -0.037 BPB). The key insight: the hidden state is literally a model that adapts at test time. For parameter golf, the most practical variant is using standard transformer layers with small LoRA adapters that are updated via next-token prediction on validation data during the eval window.

---

## 7. Efficient Attention Mechanisms

### 7a. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness
- **Authors:** Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Re
- **Year:** 2022 | **arXiv:** [2205.14135](https://arxiv.org/abs/2205.14135)
- **Citation count:** ~4000+
- **Summary:** IO-aware tiling algorithm for exact attention that reduces memory transfers. 15% speedup on BERT, 3x on GPT-2, enables 64K token sequences.
- **Relevance:** Foundational infrastructure for parameter golf. All competitive entries use FlashAttention. The IO-awareness principle (minimize HBM reads/writes via tiling) applies to all custom kernels in the training pipeline.

### 7b. FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning
- **Authors:** Tri Dao
- **Year:** 2023 | **arXiv:** [2307.08691](https://arxiv.org/abs/2307.08691)
- **Citation count:** ~1500+
- **Summary:** 2x speedup over FlashAttention via reduced non-matmul ops, better parallelism across thread blocks, and optimized warp-level work distribution. 50-73% of theoretical max on A100.
- **Relevance:** The actual implementation used in parameter golf training. The 225 TFLOPs/s throughput sets the compute ceiling for the 10-minute budget. Understanding the parallelism trade-offs helps optimize custom attention patterns (sliding window, XSA).

### 7c. GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints
- **Authors:** Joshua Ainslie, James Lee-Thorp, Michiel de Jong, et al.
- **Year:** 2023 | **arXiv:** [2305.13245](https://arxiv.org/abs/2305.13245)
- **Citation count:** ~800+
- **Summary:** Grouped-query attention uses an intermediate number of KV heads (between 1 and full multi-head). Achieves quality close to multi-head with speed close to multi-query. Requires only 5% of original pre-training compute to convert.
- **Relevance:** The SOTA uses GQA with 4 KV heads. This paper validates the quality/efficiency trade-off. At 16MB scale, the KV head count directly affects parameter allocation: 4 KV heads with 512 dim means 4*64=256 dim for KV (vs 512 for full MHA), saving ~50% of attention KV parameters.

---

## 8. Data and Training Infrastructure

### 8a. The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale
- **Authors:** Guilherme Penedo, Hynek Kydlicek, Loubna Ben Allal, et al.
- **Year:** 2024 | **arXiv:** [2406.17557](https://arxiv.org/abs/2406.17557)
- **Citation count:** ~300+
- **Summary:** 15-trillion token dataset from 96 Common Crawl snapshots. FineWeb-Edu subset (1.3T tokens) shows improved knowledge/reasoning benchmarks.
- **Relevance:** FineWeb is the evaluation dataset for parameter golf. Understanding its composition and curation (deduplication, quality filtering) helps reason about what the model needs to learn. The FineWeb-Edu subset is not used for eval, but its quality filtering methodology could inform training data selection.

### 8b. Deep Learning is Not So Mysterious or Different
- **Authors:** Andrew Gordon Wilson
- **Year:** 2025 | **arXiv:** [2503.02113](https://arxiv.org/abs/2503.02113) | ICML 2025
- **Citation count:** Recent
- **Summary:** Argues deep learning generalization phenomena (double descent, overparametrization success) are not unique to neural networks. "Soft inductive biases" (flexible hypothesis spaces with simplicity preferences) provide unifying explanation.
- **Relevance:** Referenced by slowrun for conceptual framing. The "soft inductive bias" perspective suggests that techniques like weight decay, EMA, and quantization all serve similar purposes: they constrain the hypothesis space while allowing flexible solutions. This supports the heavy weight decay finding from Kim et al. (2509.14786).

---

## 9. Additional Techniques from modded-nanogpt Records

These techniques appear in the modded-nanogpt record history but trace to specific papers or concepts:

### 9a. ReLU-squared Activation
- **Origin:** Various works on activation functions; popularized by Primer (So et al., 2021)
- **Relevance:** Used in the SOTA parameter golf architecture. ReLU^2 provides better gradient flow than standard ReLU and is cheaper than GELU/SiLU. The squaring creates a softer sparsity pattern.

### 9b. Paired Head Attention
- **Origin:** modded-nanogpt community innovation
- **Relevance:** Pairs attention heads to share computation. Reduces the effective number of independent attention computations while maintaining representational capacity.

### 9c. Bigram Hash Embedding
- **Origin:** Character-level and n-gram embedding literature
- **Relevance:** Already in SOTA at 2048 buckets. Increasing to 10240 buckets is identified as a top quick-win opportunity. The hash embedding captures character-pair statistics that the BPE tokenizer misses.

### 9d. Smear Module (Token Look-back)
- **Origin:** modded-nanogpt community; related to causal convolution literature
- **Relevance:** A lightweight mechanism that lets each token attend to its immediate predecessor without full attention. Captures local context cheaply.

### 9e. Polar Express (Newton-Schulz Replacement)
- **Origin:** Numerical methods literature (Higham 2008, Schulz 1933)
- **Relevance:** A faster method for computing the matrix polar decomposition needed by the Muon optimizer. Replaces iterative Newton-Schulz with a direct computation, reducing optimizer overhead.

---

## Top 10 Most Actionable Papers for Parameter Golf

Ranked by direct relevance to the 16MB artifact / 10-minute training / BPB optimization constraint:

| Rank | Paper | Key Insight for Parameter Golf | Expected Impact |
|------|-------|-------------------------------|----------------|
| **1** | **NorMuon** (2510.05491) | Drop-in Muon replacement with 11% better efficiency. More training progress in same 10 minutes. | -0.003 to -0.006 BPB |
| **2** | **Multi-token Prediction** (2404.19737) | Extra prediction heads during training (discarded at eval = 0 bytes). Better sample efficiency. | -0.003 to -0.008 BPB |
| **3** | **Value Residual Learning** (2410.17897) | SVFormer shares value embeddings across layers: fewer params, same quality. Already partially adopted. | -0.002 to -0.005 BPB (if extended) |
| **4** | **YaRN** (2309.00071) | Window size warmup: train on short seqs (fast), extend to long seqs (quality). More steps in 10 min. | -0.002 to -0.004 BPB |
| **5** | **Scaling Laws + Cooldown** (2405.18392) | Optimal warmdown schedule + SWA for free generalization improvement. Already partially used. | -0.001 to -0.003 BPB |
| **6** | **Hyper-Connections** (2409.19606) | Generalizes U-Net skip pattern. Dynamic depth-wise connection strength. Complements depth recurrence. | -0.002 to -0.004 BPB |
| **7** | **XSA** (2603.09078) | Orthogonal attention captures only context info (not self). Already in SOTA on 4 layers; extending to more may help. | -0.001 to -0.003 BPB (if extended) |
| **8** | **Pre-training under Infinite Compute** (2509.14786) | Weight decay should be 30x higher in data-constrained regimes. Directly testable as a hyperparameter change. | -0.001 to -0.004 BPB |
| **9** | **Cautious Optimizers** (2411.16085) | One-line modification to any momentum optimizer. Already used for weight decay; verify it's fully applied. | -0.001 to -0.002 BPB |
| **10** | **GPTQ** (2210.17323) | Full Hessian-based column ordering (vs current 5-percentile search) could reduce quantization error. | -0.001 to -0.002 BPB |

### Honorable Mentions (high potential but higher risk/effort)

| Paper | Why Notable | Risk Factor |
|-------|------------|-------------|
| **BitNet b1.58** (2402.17764) | 3x more params in 16MB via ternary weights | Unproven at 20M-scale, may regress |
| **TTT Layers** (2407.04620) | Theoretical basis for LoRA TTT (-0.037 on baseline) | Untested on SOTA base |
| **Universal Transformers** (1807.03819) | Depth recurrence = 12 effective layers at 6-layer cost | Training speed overhead |
| **Mistral SWA** (2310.06825) | Already gives -0.033 BPB; deeper understanding of stride optimization | Already exploited |
| **GQA** (2305.13245) | 4 KV heads already optimal; paper validates trade-off | Already exploited |

---

## Cross-Cutting Themes

### Theme 1: Orthogonal Compression Axes
The citation graph reveals three independent compression axes, each with its own literature:
- **Bit-width compression** (GPTQ, BitNet, int6 QAT): fewer bits per parameter
- **Structural compression** (depth recurrence, value sharing, MASA): fewer unique parameters via sharing
- **Training efficiency** (Muon, NorMuon, MTP, curriculum): more learning per second

The SOTA primarily exploits axis 1 (int6) with some axis 3 (Muon). Axis 2 is largely untapped.

### Theme 2: The Small Model Regime is Different
Multiple papers note that techniques behave differently at small scale:
- Spectra: ternary models need more data at small scale
- Pre-training under Infinite Compute: weight decay should be much higher when data-constrained
- MTP Curriculum: small models need gradual introduction of multi-token prediction
- BitNet: validated only at 3B+, unclear at 20M

This suggests that blindly applying large-model recipes will fail. Parameter golf requires small-model-specific tuning.

### Theme 3: Eval-Time Budget is Underexploited
The 10-minute eval budget is separate from the 10-minute training budget. Currently only sliding window eval uses this time. The TTT literature (Sun et al., Tandon et al.) suggests significant BPB gains from adapting the model during evaluation. The LoRA TTT experiment (-0.037 BPB on baseline) confirms this potential.

---

## Papers Not Found / Rate-Limited

The Semantic Scholar API was consistently rate-limited during this review (HTTP 429). Citation counts above are estimates based on known publication venues and dates. A follow-up with Semantic Scholar could enrich the citation graph with:
- Exact citation counts for all papers
- Forward citations (papers citing these) that may contain newer relevant work
- Reference overlap analysis to find papers commonly cited by multiple entries

---

## Source Repositories

- **modded-nanogpt**: https://github.com/KellerJordan/modded-nanogpt -- 70+ contributors, records back to GPT-2 baseline. Primary source for training efficiency techniques and architecture innovations.
- **slowrun**: https://github.com/qlabs-eng/slowrun -- Philosophy of careful optimization over speed. References Kim et al. for optimizer choices and XSA for attention innovation.
- **Muon optimizer**: https://github.com/KellerJordan/Muon -- Standalone optimizer repo.
- **llm.c baseline**: https://github.com/karpathy/llm.c -- Original baseline for the speedrun community.

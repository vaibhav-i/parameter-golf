# Advanced Literature Review: Cutting-Edge Papers for Parameter Golf

**Date:** 2026-03-25
**Search methodology:** Queried both Semantic Scholar API and arxiv API across 10 topic areas. Papers already covered in `arxiv-survey.md` and `arxiv-deep-dive.md` are excluded. Only papers with direct applicability to the Parameter Golf challenge (16MB model, 10 min training on 8xH100s, current SOTA 1.1228 bpb) are included.

---

## 1. Extreme Compression for Small Transformers

### What Makes Low-Bit QAT Work for Reasoning LLMs? A Systematic Study
- **Authors:** Keyu Lv, Manyi Zhang, Xiaobo Xia, et al.
- **Year:** 2026 | **ID:** [2601.14888](https://arxiv.org/abs/2601.14888)
- **Summary:** Systematic study of what makes QAT succeed at low bit widths for reasoning LLMs. Key findings: (1) knowledge distillation from a full-precision teacher during QAT is more important than the quantization method itself, and (2) initializing QAT from PTQ-quantized weights consistently outperforms random or full-precision initialization.
- **Applicability:** The PTQ-initialization finding is directly actionable -- train at full precision first, apply PTQ to get initial quantized weights, then fine-tune with QAT. This two-stage approach could improve the SOTA's int6 QAT quality without changing the bit width.

### MEC-Quant: Maximum Entropy Coding for Extremely Low Bit QAT
- **Authors:** Junbiao Pang, Tianyang Cai, Baochang Zhang
- **Year:** 2025 | **ID:** [2509.15514](https://arxiv.org/abs/2509.15514)
- **Summary:** Formulates quantization as an entropy coding problem, maximizing information content in the quantized representation. The maximum entropy criterion produces non-uniform quantization levels that better preserve model capacity at extremely low bit widths (2-3 bits).
- **Applicability:** Non-uniform quantization levels could extract more information per bit than the SOTA's uniform int6 grid. If applied at int5 or int4, this could enable fitting more effective parameters into 16MB. The entropy-maximizing criterion is principled and avoids the need for per-row clip search (currently 5 percentiles in GPTQ-lite).

### Bielik-Q2-Sharp: Extreme 2-bit Quantization Methods
- **Authors:** Jakub Prejzner
- **Year:** 2026 | **ID:** [2603.04162](https://arxiv.org/abs/2603.04162)
- **Summary:** First systematic academic comparison of six 2-bit quantization methods on an 11B model. Finds that importance-weighted methods and calibration language matching significantly affect 2-bit quality. Best 2-bit variant preserves higher-order reasoning better than expected.
- **Applicability:** If 2-bit quantization can preserve meaningful quality, a model with 3x more parameters than the current SOTA could fit in 16MB. The comparative findings help choose among 2-bit methods. However, 11B is far from our scale -- results may not transfer to 20-50M models.

---

## 2. Training Efficiency / Sample Efficiency

### Influence-driven Curriculum Learning for Pre-training on Limited Data
- **Authors:** Loris Schoenegger, Lukas Thoma, Terra Blevins
- **Year:** 2025 | **ID:** [2508.15475](https://arxiv.org/abs/2508.15475)
- **Summary:** Replaces conventional difficulty metrics with training data influence (how individual examples affect model outputs) for curriculum ordering. Models trained on influence-driven curricula outperform random ordering by over 10 percentage points on downstream tasks. Effect is strongest when pre-training data is limited.
- **Applicability:** Highly relevant. The 10-minute training budget means we see limited data. If the training data can be pre-ordered by influence scores (computed offline), the model could converge faster. The "limited data" regime is exactly parameter golf's setting. Requires a one-time offline preprocessing pass.

### Prompt Curriculum Learning for Efficient LLM Post-Training
- **Authors:** Zhaolin Gao, Joongwon Kim, Wen Sun
- **Year:** 2025 | **ID:** [2510.01135](https://arxiv.org/abs/2510.01135)
- **Summary:** Lightweight RL-based algorithm that selects intermediate-difficulty prompts during training. Achieves either highest performance or significantly less time to reach comparable performance. Key finding: focusing on intermediate-difficulty examples (not too easy, not too hard) maximizes learning efficiency.
- **Applicability:** The intermediate-difficulty principle translates to pre-training: weight the loss toward tokens/sequences of moderate perplexity. Easy tokens (common words) waste gradient signal; impossible tokens (noise, rare entities) also waste signal. A simple loss-weighting schedule based on token perplexity could improve convergence.

---

## 3. Weight Sharing / Parameter Tying Innovations

### ResidualTransformer: Residual Low-Rank Learning with Weight-Sharing
- **Authors:** Yiming Wang, Jinyu Li
- **Year:** 2023 | **ID:** [2310.02489](https://arxiv.org/abs/2310.02489)
- **Summary:** Shares base transformer weights across layers but adds unique low-rank residual matrices per layer. The residuals capture layer-specific specialization at minimal parameter cost. Designed for deployment on memory-constrained always-on devices, achieving strong performance with drastically reduced model size.
- **Applicability:** This is a refined version of the Relaxed Recursive Transformer approach already identified. The key difference: residual matrices are learned directly (not LoRA adapters), potentially offering better expressiveness. For parameter golf: share 4-6 base layers, add rank-8 residual matrices per "virtual" layer. Each residual is ~8KB at int6, so 12 virtual layers add only ~96KB overhead.

### Sharing Attention Weights for Fast Transformer
- **Authors:** Tong Xiao, Yinqiao Li, Jingbo Zhu
- **Year:** 2019 | **ID:** [1906.11024](https://arxiv.org/abs/1906.11024)
- **Summary:** Shares attention weights across layers while keeping FFN weights unique. Shows that attention patterns are often similar across adjacent layers, making attention sharing nearly lossless. FFN layers, which contain the model's "knowledge," need to remain unique.
- **Applicability:** In the SOTA, attention parameters (Q/K/V/O projections) are a significant fraction of total size. If attention weights are shared across layers while keeping FFN unique, the parameter budget freed from attention can be reinvested in wider FFNs. This is complementary to MASA (attention dictionary decomposition from the deep-dive).

---

## 4. Post-Training Quantization Advances

### OCTAV: Optimal Clipping and Magnitude-aware Differentiation for Improved QAT
- **Authors:** Charbel Sakr, Steve Dai, Rangharajan Venkatesan (Intel)
- **Year:** 2022 | **ID:** [2206.06501](https://arxiv.org/abs/2206.06501)
- **Summary:** Proposes a recursive Newton-Raphson algorithm to determine MSE-optimal clipping scalars for quantization. Replaces heuristic clip search (like the SOTA's 5-percentile GPTQ-lite) with a mathematically optimal solution. Also introduces magnitude-aware gradient scaling that improves STE quality.
- **Applicability:** Directly replaces the GPTQ-lite clip search with an optimal solution. The SOTA currently tries 5 clip percentiles per row -- OCTAV computes the MSE-optimal clip in closed form. This could both improve quantization quality AND speed up the export pipeline. The magnitude-aware gradient scaling also improves training-time QAT.

### Benford's Law as a Distributional Prior for PTQ of LLMs
- **Authors:** Arthur Negrao, Pedro Silva, Vander L. S. Freitas
- **Year:** 2026 | **ID:** [2602.00165](https://arxiv.org/abs/2602.00165)
- **Summary:** Uses Benford's Law (the empirical observation that leading digits follow a logarithmic distribution) as a prior for weight quantization. Standard uniform quantizers assume evenly distributed parameters, which is wrong -- LLM weights are log-normally distributed. Benford-aware quantization assigns more levels to smaller magnitudes.
- **Applicability:** Novel quantization grid design. Instead of uniform int6 levels, use a Benford-distributed grid that gives more resolution to the frequent small-magnitude weights and less to rare large-magnitude weights. Could improve effective precision at the same bit width. Simple to implement: just change the quantization grid.

### FPTQuant: Function-Preserving Transforms for LLM Quantization
- **Authors:** Boris van Breugel, Yelysei Bondarenko, Paul Whatmough, Tijmen Blankevoort
- **Year:** 2025 | **Citations:** 9 | **ID:** [2506.04985](https://arxiv.org/abs/2506.04985)
- **Summary:** Applies lightweight linear transforms to weights and activations before quantization to make the distributions more quantization-friendly. Enables static INT4 quantization with minimal overhead and up to 3.9x speedups on H100s. Key insight: the same model can be equivalently represented in a more quantization-friendly basis.
- **Applicability:** If applied before the SOTA's int6 quantization, FPTQuant's basis transforms could reduce quantization error without changing bit width. The H100-specific speedup results are directly relevant. Could potentially enable going from int6 to int5 or int4 without quality loss.

### Opt4GPTQ: Co-Optimizing Memory and Computation for 4-bit GPTQ
- **Authors:** Yaozheng Zhang, Wei Wang, Jie Kong
- **Year:** 2025 | **ID:** [2511.19438](https://arxiv.org/abs/2511.19438)
- **Summary:** Practical optimization of the GPTQ algorithm for 4-bit inference across heterogeneous platforms. Addresses memory layout and kernel design for fast 4-bit dequantization.
- **Applicability:** If the SOTA moves from int6 to int4, this provides the optimized inference kernels. Lower priority unless the bit width changes, but relevant as an enabler for more aggressive quantization.

---

## 5. Evaluation-Time Optimization

### TTQ: Activation-Aware Test-Time Quantization
- **Authors:** Toshiaki Koike-Akino, Jing Liu, Ye Wang
- **Year:** 2026 | **ID:** [2603.19296](https://arxiv.org/abs/2603.19296)
- **Summary:** Performs quantization calibration at test time, adapting to the specific distribution of the evaluation data. Eliminates reliance on calibration data from training distribution. Online calibration adjusts quantization parameters per-batch during inference.
- **Applicability:** Interesting twist: instead of fixed quantization parameters, adapt them during evaluation. The 10-minute eval budget provides time for this. Could be combined with LoRA TTT -- while adapting model weights via TTT, also adapt quantization scales to the test distribution.

### Unsupervised Layer-Wise Dynamic Test Time Adaptation for LLMs
- **Authors:** Longhuan Xu, Cunjian Chen, Feng Yin
- **Year:** 2026 | **ID:** [2602.09719](https://arxiv.org/abs/2602.09719)
- **Summary:** Adapts model parameters independently for each prompt at test time using only the prompt itself (no labels). Different layers are adapted at different rates based on their sensitivity. Uses entropy minimization as the unsupervised objective.
- **Applicability:** Layer-wise adaptation rates could complement LoRA TTT. Instead of uniform learning rates across all LoRA adapters, use per-layer rates determined by layer sensitivity. Entropy minimization as the TTT objective is simpler than next-token prediction and could be faster.

### Adaptive Decoding via Test-Time Policy Learning
- **Authors:** Asmita Bhardwaj, Yuya Jeremy Ong, Eelaaf Zahid
- **Year:** 2026 | **ID:** [2603.18428](https://arxiv.org/abs/2603.18428)
- **Summary:** Learns a lightweight policy to adjust sampling parameters (temperature, top-p) at test time while keeping LLM weights frozen. Treats decoding as sequential decision-making.
- **Applicability:** For BPB evaluation, the decoding strategy matters less since we're computing log-probabilities, not generating. However, the principle of adapting inference-time hyperparameters per-instance could apply to sliding window stride, context length, or other eval parameters.

---

## 6. Embedding Compression / Factorization

### DeFINE: DEep Factorized INput Token Embeddings
- **Authors:** Sachin Mehta, Rik Koncel-Kedziorski, Mohammad Rastegari (Apple)
- **Year:** 2019 | **Citations:** ~85 | **ID:** [1911.12385](https://arxiv.org/abs/1911.12385)
- **Summary:** For models with large vocabularies, input/output embeddings dominate parameter count. DeFINE uses a deep factorization: tokens are first projected to a small dimension, then passed through a stack of grouped linear transforms that progressively widen the representation. Reduces embedding parameters by 2-4x while improving perplexity vs standard embeddings.
- **Applicability:** The SOTA uses a 1024-token vocabulary with tied embeddings (input=output), so embedding size is already small (~512K params). However, DeFINE's factorization could enable a larger vocabulary (e.g., 4096 tokens) without proportional parameter increase. A larger vocabulary could reduce sequence length (more information per token), enabling processing more text in the 10-minute window.

### Improving Word Embedding Factorization for Compression Using Distilled Nonlinear Neural Decomposition
- **Authors:** Vasileios Lioutas, Ahmad Rashid, Krtin Kumar
- **Year:** 2019 | **ID:** [1910.06720](https://arxiv.org/abs/1910.06720)
- **Summary:** Improves embedding compression by using nonlinear decomposition instead of linear (SVD) factorization. Distillation from the full embedding table into the compressed version preserves quality better than direct training.
- **Applicability:** If vocabulary size increases (per DeFINE insight), this provides a better compression method for the embedding table. Nonlinear decomposition captures more structure than the low-rank approaches typically used.

### Hash Embeddings for Efficient Word Representations
- **Authors:** Dan Svenstrup, Jonas Meinertz Hansen, Ole Winther
- **Year:** 2017 | **ID:** [1709.03933](https://arxiv.org/abs/1709.03933)
- **Summary:** Original hash embedding paper. Uses multiple hash functions to map tokens to shared embedding vectors, then combines via learned weights. Interpolates between full embeddings (one vector per token) and hashing (shared vectors, collisions). Achieves near-full-embedding quality with 10-50x fewer parameters.
- **Applicability:** The SOTA already uses BigramHash (2048 buckets). This paper's multi-hash approach could improve the bigram embedding quality at the same bucket count, or allow reducing bucket count. The learned combination weights add expressiveness beyond simple hash lookup.

### Multi Hash Embeddings in spaCy
- **Authors:** Lester James Miranda, Akos Kadar, Adriane Boyd
- **Year:** 2022 | **ID:** [2212.09255](https://arxiv.org/abs/2212.09255)
- **Summary:** Practical implementation of multi-hash embeddings showing that combining 2-4 hash functions with different seeds reduces collision effects and improves embedding quality. Production-tested in spaCy framework.
- **Applicability:** Directly applicable to BigramHash improvement. Instead of a single hash function mapping bigrams to 2048-10240 buckets, use 2-3 hash functions and combine. This reduces collision-induced noise with minimal parameter overhead (just combination weights).

---

## 7. Activation Functions for Compressed Models

### S2D: Selective Spectral Decay for Quantization-Friendly Conditioning
- **Authors:** Arnav Chavan, Nahush Lele, Udbhav Bamba, et al.
- **Year:** 2026 | **Citations:** 0 | **ID:** [2602.14432](https://arxiv.org/abs/2602.14432)
- **Summary:** Identifies that activation outliers correlate with dominant singular values in weight matrices. Proposes spectral regularization during training that suppresses outlier-causing singular values, making activations more quantization-friendly. Improves PTQ accuracy by up to 7% under W4A4 quantization.
- **Applicability:** Adding spectral regularization during QAT training could reduce the quantization error at int6, effectively getting int7-equivalent quality at int6 storage cost. Implementation: add a penalty term on the top singular values of weight matrices to the training loss. Simple regularization change, no architecture modification needed.

### Integer Arithmetic-Based and Activation-Aware GELU Optimization
- **Authors:** Zou, Zhang, Chen, Kou, Liu
- **Year:** 2024
- **Summary:** Replaces GELU with an activation-aware integer-arithmetic approximation, achieving 2.14x energy efficiency improvement with less than 1% accuracy loss. The approximation is designed for quantized inference.
- **Applicability:** The SOTA uses SiLU activation. If inference is done in integer arithmetic (post-quantization), an integer-friendly activation replacement could reduce dequantization overhead. Lower priority -- activation function changes have small effects compared to weight quantization.

---

## 8. Knowledge Distillation into Tiny Models

### ERNIE-Tiny: Progressive Distillation Framework for Pretrained Transformer Compression
- **Authors:** Weiyue Su, Xuyi Chen, Shikun Feng
- **Year:** 2021 | **ID:** [2106.02241](https://arxiv.org/abs/2106.02241)
- **Summary:** Multi-stage progressive distillation: instead of distilling directly from teacher to tiny student, uses a chain of intermediate-size models. Each stage distills into a slightly smaller model. The progressive approach avoids the large capacity gap problem.
- **Applicability:** If contest rules allow pre-training a teacher, progressive distillation through 3-4 stages (e.g., 200M -> 100M -> 50M -> 25M quantized) could produce a better 16MB model than training the tiny model directly. The 10-minute constraint applies to the final training only; teacher pre-training can happen offline.

### Dynamic Self-Distillation via Previous Mini-batches
- **Authors:** Yao Fu, Yin Yu, Xiaotian Han
- **Year:** 2024 | **ID:** [2411.16991](https://arxiv.org/abs/2411.16991)
- **Summary:** Self-distillation without a separate teacher: uses the model's own predictions from previous mini-batches as soft targets. The "teacher" is the same model slightly earlier in training, providing a smoothing/regularization effect.
- **Applicability:** Zero-cost distillation -- no teacher needed, no extra compute. Simply maintain a running average of the model's output distribution from the last N batches and add a KL-divergence term to the loss. Acts as regularization that prevents the small model from overfitting to sharp distributions. Could improve generalization of the 16MB model.

### Distilling Large Language Models into Tiny and Effective Students using pQRNN
- **Authors:** Prabhu Kaliamoorthi, Aditya Siddhant, Edward Li (Google)
- **Year:** 2021 | **ID:** [2101.08890](https://arxiv.org/abs/2101.08890)
- **Summary:** Distills large multilingual models into tiny pQRNN students (projection-based quasi-RNN). The pQRNN architecture uses hashed projections instead of embeddings, achieving extreme compression. Students are 300x smaller than teachers with competitive quality.
- **Applicability:** The pQRNN architecture itself is interesting -- hashed projections replacing learned embeddings is exactly the BigramHash approach in the SOTA. The 300x compression ratio shows that distillation into hash-embedding-based tiny models is viable. If a teacher is available, this recipe could be adapted.

---

## 9. Sparse Attention for Small Models

### MiniCPM-SALA: Hybridizing Sparse and Linear Attention
- **Authors:** MiniCPM Team, Wenhao An, Yingfa Chen
- **Year:** 2026 | **ID:** [2602.11761](https://arxiv.org/abs/2602.11761)
- **Summary:** Combines sparse attention (for precision on critical tokens) with linear attention (for efficiency on routine tokens) in a hybrid architecture. Different layers use different attention types based on their role. Achieves strong performance at greatly reduced compute cost.
- **Applicability:** The SOTA uses XSA (exclusive self-attention) on the last 4 layers and standard attention on the first 7. Replacing the first 7 layers with linear attention could speed up training (more steps in 10 minutes) while XSA handles the critical later layers. Linear attention is O(n) vs O(n^2), so the speedup is proportional to sequence length.

### Sparse Query Attention (SQA): Query Heads Reduction
- **Authors:** Adam Filipek
- **Year:** 2025 | **ID:** [2510.01817](https://arxiv.org/abs/2510.01817)
- **Summary:** Reduces the number of query heads (not KV heads) in attention, complementary to GQA which reduces KV heads. Shows that many query heads compute redundant patterns and can be safely removed.
- **Applicability:** The SOTA uses GQA with 4 KV heads. SQA could further reduce attention compute by pruning redundant query heads. If the model uses 8 query heads grouped into 4 KV groups, SQA might show that only 4-6 query heads are needed, saving parameters and compute.

---

## 10. Data Ordering / Curriculum for Limited Training

### Influence-driven Curriculum Learning for Pre-training on Limited Data
- **Authors:** Loris Schoenegger, Lukas Thoma, Terra Blevins
- **Year:** 2025 | **ID:** [2508.15475](https://arxiv.org/abs/2508.15475)
- **Summary:** (Also listed in Section 2.) Models trained on influence-ordered curricula outperform random ordering by >10 percentage points. Effect is strongest with limited pre-training data -- exactly the parameter golf regime.
- **Applicability:** Pre-compute influence scores on FineWeb training data using a small proxy model. Order training batches by influence score (highest influence first). This is a preprocessing step that can be done offline. Expected to improve convergence in the limited 10-minute window.

### Do Syntactic Categories Help in Curriculum Learning for Language Models?
- **Authors:** Arzu Burcu Guven, Anna Rogers, Rob van der Goot
- **Year:** 2025 | **ID:** [2511.08199](https://arxiv.org/abs/2511.08199)
- **Summary:** Examines syntactic properties for curriculum ordering. Finds that syntactically categorizable data (clean, well-formed text) improves model performance compared to training on noisy full corpora.
- **Applicability:** Simple data quality filter: prioritize well-formed, syntactically clean FineWeb documents during early training. Later training phases can include noisier data. This is a data selection strategy that costs nothing at training time.

### VCRL: Variance-based Curriculum Reinforcement Learning
- **Authors:** Guochao Jiang, Wenfeng Feng, Guofeng Quan
- **Year:** 2025 | **ID:** [2509.19803](https://arxiv.org/abs/2509.19803)
- **Summary:** Uses reward variance to determine sample difficulty. Samples with moderate variance (not too easy, not too hard) produce the most learning signal. Too-easy and too-hard samples both have low variance and contribute less.
- **Applicability:** Translates to pre-training: compute per-sequence loss variance over a few model snapshots. Prioritize sequences with moderate loss variance -- these are the ones where the model is actively learning. Easy (low loss, low variance) and impossible (high loss, low variance) sequences can be downweighted.

---

## Bonus: Muon Optimizer Optimization

### Effective Quantization of Muon Optimizer States
- **Authors:** Aman Gupta, Rafael Celente, Abhishek Shivanna, et al.
- **Year:** 2025 | **Citations:** 2 | **ID:** [2509.23106](https://arxiv.org/abs/2509.23106)
- **Summary:** Introduces 8-bit blockwise quantization of Muon optimizer states, achieving 62% memory reduction with performance parity. Muon is inherently more tolerant to quantization than Adam due to its orthogonalized update structure.
- **Applicability:** The SOTA uses the Muon optimizer. Quantizing its states to 8-bit frees GPU memory for larger batch sizes or wider models during training. Since training is memory-bound on 8xH100s with a 10-minute budget, any freed memory translates to more tokens processed per step. Could enable 1.5x larger batch size.

### To Use or not to Use Muon: How Simplicity Bias in Optimizers Matters
- **Authors:** Sara Dragutinovic, Rajesh Ranganath
- **Year:** 2026 | **ID:** [2603.00742](https://arxiv.org/abs/2603.00742)
- **Summary:** Analyzes why Muon outperforms Adam: Muon has lower "simplicity bias," learning more complex features earlier in training. Adam tends to first learn simple patterns and only later develops complex representations. For short training runs, Muon's faster development of complex features is critical.
- **Applicability:** Validates that Muon is the right optimizer for parameter golf's short training budget. Also suggests that techniques that further reduce simplicity bias (e.g., feature diversity regularization) could compound with Muon's advantage.

### Advancing Model Refinement: Muon-Optimized Distillation and Quantization for LLM Deployment
- **Authors:** Jacob Sander, Brian Jalaian, Venkat R. Dasari
- **Year:** 2026 | **ID:** [2601.09865](https://arxiv.org/abs/2601.09865)
- **Summary:** Combines Muon optimization with knowledge distillation and quantization in a unified pipeline. Shows that Muon-optimized distillation produces better quantized models than Adam-optimized distillation.
- **Applicability:** If distillation is pursued, using Muon for the distillation training (not just the original pre-training) improves the final quantized model quality. Validates the existing optimizer choice extends to distillation settings.

---

## Top 15 Most Actionable Papers

Ranked by direct applicability, considering: expected BPB impact, implementation effort, compatibility with SOTA, and novelty (not already in existing research).

| Rank | Paper | Key Technique | Expected Impact | Effort |
|------|-------|--------------|----------------|--------|
| 1 | **OCTAV** (2206.06501) | MSE-optimal clipping for quantization | -0.001 to -0.005 BPB | Low |
| 2 | **S2D** (2602.14432) | Spectral regularization for quantization-friendly activations | -0.002 to -0.005 BPB | Low |
| 3 | **Dynamic Self-Distillation** (2411.16991) | Self-distillation from previous mini-batches | -0.001 to -0.004 BPB | Low |
| 4 | **Influence-driven Curriculum** (2508.15475) | Data ordering by influence scores | -0.002 to -0.006 BPB | Medium |
| 5 | **Benford's Law PTQ** (2602.00165) | Non-uniform quantization grid matching weight distribution | -0.001 to -0.003 BPB | Low |
| 6 | **ResidualTransformer** (2310.02489) | Weight sharing + per-layer low-rank residuals | -0.005 to -0.012 BPB | Medium |
| 7 | **FPTQuant** (2506.04985) | Pre-quantization basis transforms | -0.002 to -0.005 BPB | Medium |
| 8 | **Multi Hash Embeddings** (2212.09255) | Multiple hash functions for BigramHash | -0.001 to -0.003 BPB | Low |
| 9 | **8-bit Muon States** (2509.23106) | Quantize optimizer states for larger batches | -0.001 to -0.003 BPB | Low |
| 10 | **MEC-Quant** (2509.15514) | Maximum entropy quantization levels | -0.001 to -0.004 BPB | Medium |
| 11 | **Unsupervised Layer-Wise TTT** (2602.09719) | Per-layer adaptive TTT rates | -0.005 to -0.015 BPB | Medium |
| 12 | **MiniCPM-SALA** (2602.11761) | Hybrid sparse/linear attention | -0.002 to -0.005 BPB | High |
| 13 | **DeFINE** (1911.12385) | Deep factorized embeddings for larger vocab | -0.002 to -0.006 BPB | High |
| 14 | **TTQ** (2603.19296) | Adaptive quantization at test time | -0.001 to -0.003 BPB | Medium |
| 15 | **Muon Simplicity Bias** (2603.00742) | Feature diversity regularization | -0.001 to -0.002 BPB | Low |

---

## Novel Ideas Not in Existing Research

These are techniques or combinations suggested by the papers above that have NOT been identified in `arxiv-survey.md`, `arxiv-deep-dive.md`, or any other existing research files:

### 1. OCTAV Optimal Clipping (replaces GPTQ-lite heuristic)
The SOTA's GPTQ-lite tries 5 clip percentiles per row. OCTAV computes the mathematically optimal clip via Newton-Raphson iteration -- guaranteed MSE-optimal, no search needed. This is a strict upgrade to the export pipeline with zero training-time cost. **Priority: immediate.**

### 2. Spectral Regularization During QAT (S2D)
Add a loss penalty on top singular values of weight matrices during training. This suppresses activation outliers that cause quantization error, effectively making the trained model more quantization-friendly. The regularization costs ~5% extra compute per step but could reduce quantization error by up to 7%. **Priority: high -- integrates naturally with existing QAT.**

### 3. Benford-distributed Quantization Grid
Replace uniform int6 levels with log-spaced levels matching Benford's Law. LLM weights follow a log-normal distribution; uniform grids waste resolution on large magnitudes that rarely occur. The Benford grid assigns more levels to the dense small-magnitude region. Implementation: change the quantization and dequantization functions (5 lines of code). **Priority: high -- trivial implementation, principled improvement.**

### 4. Self-Distillation from Previous Mini-Batches
No teacher needed. Maintain an exponential moving average of the model's softmax outputs. Add KL(current, EMA) to the loss with a small coefficient (0.1-0.3). This acts as regularization that smooths the model's predictions and prevents overfitting. Zero extra memory (EMA is just one extra tensor of vocab-size logits). **Priority: high -- free regularization.**

### 5. Multi-Hash BigramHash
The current BigramHash uses a single hash function. Using 2-3 hash functions with different seeds and learning a weighted combination reduces collision-induced noise. For 10240 buckets with 3 hashes, collision probability drops from ~1/10240 to ~1/10240^3 for exact collisions, with graceful degradation for partial collisions. Adds only the combination weights (~30K params). **Priority: medium -- simple modification to existing BigramHash.**

### 6. 8-bit Muon Optimizer States
The Muon optimizer's momentum states are stored in FP32. Quantizing to INT8 frees ~62% of optimizer memory. For a 20M-param model, this frees ~120MB of GPU RAM -- enough for ~50% larger batch size. Larger batches improve training throughput and convergence stability. **Priority: medium -- implementation exists, just integrate.**

### 7. PTQ-Initialized QAT (Two-Stage Training)
Train at full precision for 80% of the budget, apply PTQ to get initial quantized weights, then fine-tune with QAT for the remaining 20%. The PTQ initialization gives QAT a better starting point than learning quantization from scratch. Validated by the systematic QAT study as the most impactful single change. **Priority: medium -- requires restructuring the training loop.**

### 8. Influence-Ordered Training Data
Pre-compute influence scores for FineWeb training chunks using a small proxy model (offline, unlimited compute). During the 10-minute training, feed high-influence data first. The >10 percentage point improvement from influence ordering in limited-data regimes could translate to significant BPB gains. **Priority: medium -- requires offline preprocessing.**

### 9. Test-Time Quantization Adaptation (TTQ)
During the 10-minute eval window, recalibrate quantization scales to match the validation data distribution (which may differ from training data). This is orthogonal to LoRA TTT (which adapts weights) and could be combined. Per-layer scale recalibration takes seconds and costs nothing. **Priority: medium -- simple addition to eval loop.**

### 10. Linear Attention in Early Layers
Replace softmax attention in layers 1-4 with linear attention (no softmax, O(n) complexity). Keep XSA/softmax in later layers where attention precision matters most. The speedup in early layers (~2x per layer) allows either more training steps or longer sequences. The MiniCPM-SALA hybrid validates this approach at scale. **Priority: lower -- architecture change, higher risk.**

---

## Summary

The most impactful gap in current research is **quantization quality optimization** -- the SOTA's int6 quantization uses a heuristic clip search (GPTQ-lite with 5 percentiles) and uniform quantization levels. Three papers (OCTAV, Benford PTQ, S2D) independently suggest that the quantization pipeline has significant room for improvement without changing the bit width. Together, these could yield -0.003 to -0.010 BPB purely from better quantization at the same model size.

The second major gap is **training efficiency** -- self-distillation, influence-ordered data, and 8-bit optimizer states could squeeze more learning into the 10-minute window without any architectural changes.

Both of these categories are low-to-medium implementation effort and compatible with all other techniques in the pipeline.

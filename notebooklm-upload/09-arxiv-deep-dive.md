# Arxiv Deep Dive: Targeted Research for Parameter Golf Gaps

**Date:** 2026-03-24
**Purpose:** Follow-up to the initial arxiv survey, targeting specific gaps and untried combinations identified in `evolution-chain.md` and `modded-nanogpt-insights.md`. Papers already covered in `arxiv-survey.md` are excluded.

---

## 1. Depth Recurrence / Weight Sharing

**Why this matters for parameter golf:** Depth recurrence (looping a block of layers multiple times) is listed as "completely untried" in the evolution chain. A model with 4 unique layers looped 3 times would have 12 effective layers but only 4 layers worth of parameters -- a massive compression win within 16MB. The current SOTA uses 11 unique layers; recurrence could achieve similar effective depth at a fraction of the parameter cost.

### Key Papers

**a. Relaxed Recursive Transformers: Effective Parameter Sharing with Layer-wise LoRA**
- Authors: Sangmin Bae, Adam Fisch, Hrayr Harutyunyan, Ziwei Ji, Seungyeon Kim, Tal Schuster
- Date: 2024-10-28 | ID: [2410.20672v3](https://arxiv.org/abs/2410.20672)
- Converts pretrained LLMs into "Recursive Transformers" that share parameters across layers using a single repeated block, enhanced with depth-wise LoRA modules for per-loop differentiation. Demonstrates significant parameter savings while maintaining quality.
- **Applicability:** Highly relevant. Instead of 11 unique layer blocks (~14MB), use 3-4 unique blocks with tiny per-loop LoRA adapters (a few KB each). The LoRA adapters let each "virtual layer" specialize despite sharing base weights. Implementation: define 3-4 transformer blocks, loop each 2-3 times, add LoRA(rank=4) per loop iteration. Estimated savings: 6-8MB freed for wider layers or larger embeddings.

**b. Lessons on Parameter Sharing across Layers in Transformers**
- Authors: Sho Takase, Shun Kiyono
- Date: 2021-04-13 (updated 2023-06-02) | ID: [2104.06022v4](https://arxiv.org/abs/2104.06022)
- Proposes Sequence, Cycle, and Cycle(rev) sharing strategies that go beyond Universal Transformer's uniform repetition. Cycle(rev) -- where layers are shared in a palindromic pattern (1,2,3,3,2,1) -- consistently outperforms naive repetition.
- **Applicability:** Direct application. The Cycle(rev) pattern is a drop-in replacement for the current 11 unique layers. For example, 6 unique layers in a Cycle(rev) pattern gives 12 effective layers with 6 layers of parameters (~55% size reduction). The palindromic structure naturally creates a U-Net-like information flow, complementing the existing U-Net skip connections.

**c. Subformer: Exploring Weight Sharing for Parameter Efficiency in Generative Transformers**
- Authors: Machel Reid, Edison Marrese-Taylor, Yutaka Matsuo
- Date: 2021-01-01 | ID: [2101.00234v3](https://arxiv.org/abs/2101.00234)
- Combines "sandwich-style" parameter sharing (unique first/last layers, shared middle layers) with self-attentive embedding factorization. Outperforms standard Transformers with significantly fewer parameters on MT and summarization.
- **Applicability:** The sandwich pattern is intuitive for parameter golf: keep the first 2 and last 2 layers unique (they handle input/output processing), share the middle layers. Combined with int6 QAT, this could enable a 14-16 effective layer model in 16MB. The embedding factorization could also replace or complement BigramHash.

**d. Parallel Loop Transformer for Efficient Test-Time Computation Scaling**
- Authors: Bohong Wu, Mengzhao Chen, Xiang Luo, Shen Yan, et al.
- Date: 2025-10-28 | ID: [2510.24824v1](https://arxiv.org/abs/2510.24824)
- Enables looped transformer execution with cross-loop parallelism, eliminating the sequential bottleneck of naive looping. Uses gated sliding-window attention for efficient KV cache sharing across loop iterations.
- **Applicability:** Solves the key concern about looped transformers: that sequential loops slow down training. If loops can be parallelized, the training time cost of depth recurrence drops significantly. Critical for fitting in the 10-minute budget.

**e. Dynamic Layer Tying for Parameter-Efficient Transformers**
- Authors: Tamir David Hay, Lior Wolf
- Date: 2024-01-23 | ID: [2401.12819v1](https://arxiv.org/abs/2401.12819)
- Uses RL to dynamically select which layers to tie during training, finding optimal sharing patterns automatically. Reduces trainable parameters substantially while also acting as regularization.
- **Applicability:** Rather than hand-picking sharing patterns, let the training process discover which layers should share weights. However, the RL overhead may not fit the 10-minute budget. More practical as a one-time hyperparameter search.

### Applicability Assessment

Depth recurrence is the single most promising untried direction for parameter golf. The Cycle(rev) pattern from Takase & Kiyono is the lowest-effort implementation: replace 11 unique layers with 6 unique layers in a (1,2,3,4,5,6,6,5,4,3,2,1) pattern to get 12 effective layers. The Relaxed Recursive approach with per-loop LoRA is more sophisticated but offers finer control. **Expected impact: -0.005 to -0.015 BPB** by enabling deeper effective models within the same parameter budget.

---

## 2. Extreme Low-Bit Quantization (Ternary / Int4)

**Why this matters for parameter golf:** Current SOTA uses int5-int6 quantization (5-6 bits per weight). Going to ternary (1.58 bits) or int4 (4 bits) would either dramatically shrink the model (freeing space for more parameters) or allow a much wider/deeper model at the same 16MB size. A ternary model could theoretically pack ~67M parameters into 16MB vs ~21M at int6.

### Key Papers

**a. Spectra: Surprising Effectiveness of Pretraining Ternary Language Models at Scale**
- Authors: Ayush Kaushal, Tejas Vaidhya, Arnab Kumar Mondal, et al.
- Date: 2024-10-11 | ID: [2407.12327v5](https://arxiv.org/abs/2407.12327)
- Shows that ternary {-1, 0, 1} language models pretrained from scratch consistently outperform quantize-after-training approaches at scales above 1B parameters. At smaller scales, ternary models benefit more from additional training data than from scaling parameters.
- **Applicability:** Critical insight for parameter golf: at the ~20-50M parameter scale, ternary models may underperform int6 models unless given significantly more training data. The 10-minute budget limits data throughput. However, the "more data > more params" finding suggests maximizing batch size and sequence length with ternary weights could work.

**b. Spectra 1.1: Scaling Laws and Efficient Inference for Ternary Language Models**
- Authors: Tejas Vaidhya, Ayush Kaushal, et al.
- Date: 2025-06-28 | ID: [2506.23025v1](https://arxiv.org/abs/2506.23025)
- Provides scaling laws specific to ternary models and introduces packing schemes achieving up to 5x faster inference. Reveals that ternary models have fundamentally different scaling behavior: they benefit more from data than parameters.
- **Applicability:** The scaling laws are directly useful for deciding optimal model size at 1.58 bits within 16MB. The packing schemes (5x faster inference) could speed up evaluation, leaving more time budget for test-time training.

**c. The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits (BitNet b1.58)**
- Authors: Shuming Ma, Hongyu Wang, et al.
- Date: 2024-02-27 | ID: [2402.17764v1](https://arxiv.org/abs/2402.17764)
- Every parameter is ternary {-1, 0, 1}. Achieves performance parity with full-precision at large scale. Uses absmean quantization function and straight-through estimator for training.
- **Applicability:** The absmean quantization function is simpler than the current int6 STE approach and could be implemented in ~20 lines of code. Key question: does BitNet b1.58 work at 20-50M parameter scale? The paper only tests at 3B+.

**d. Hybrid Gated Flow (HGF): Stabilizing 1.58-bit LLMs via Selective Low-Rank Correction**
- Authors: David Alejandro Trejo Pizzo
- Date: 2026-02-05 | ID: [2602.05269v1](https://arxiv.org/abs/2602.05269)
- Combines a ternary backbone with a learned low-rank FP16 correction path controlled by adaptive gates. Recovers ~55% of the quality gap between pure ternary and full-precision.
- **Applicability:** This is a practical middle ground: use ternary for most weights (massive compression) but keep a small FP16 correction path for quality-sensitive layers. The correction path could be stored as fp16 within the 16MB budget while the bulk of parameters are at 1.58 bits. Estimated: ~45M ternary params + 2M fp16 correction params = ~12MB + 4MB = 16MB.

**e. Token-Scaled Logit Distillation for Ternary Weight Generative Language Models**
- Authors: Minsoo Kim, Sihwa Lee, et al.
- Date: 2023-12-03 | ID: [2308.06744v4](https://arxiv.org/abs/2308.06744)
- First evaluation of ternary QAT for generative LMs specifically. Uses token-scaled distillation to maintain output quality during aggressive quantization.
- **Applicability:** If distillation from a larger teacher is allowed (train a big model, distill into ternary), this provides the recipe. However, the 10-minute constraint means the teacher must also be trained quickly, or a pre-existing teacher model must be available (which contest rules may prohibit).

### Applicability Assessment

Ternary quantization is high-risk, high-reward. The key concern is that Spectra's scaling laws suggest ternary models need much more training data to compensate at small scales -- and the 10-minute budget limits data throughput. The HGF approach (ternary backbone + FP16 correction) is the most practical path: it maintains quality while still achieving ~2.5x better compression than int6. **Expected impact: -0.005 to -0.020 BPB** if the wider model compensates for quantization loss, or **+0.010 BPB regression** if ternary is too aggressive at this scale.

---

## 3. Multi-Token Prediction

**Why this matters for parameter golf:** Multi-token prediction increases learning signal per training step without proportionally increasing compute. In a 10-minute training budget, every bit of sample efficiency matters. The prediction heads are discarded at inference time, so they add zero cost to the 16MB artifact.

### Key Papers

**a. Better & Faster Large Language Models via Multi-token Prediction**
- Authors: Fabian Gloeckle, Badr Youbi Idrissi, Baptiste Roziere, David Lopez-Paz, Gabriel Synnaeve (Meta)
- Date: 2024-04-30 | ID: [2404.19737v1](https://arxiv.org/abs/2404.19737)
- Training LMs to predict 4 future tokens simultaneously improves sample efficiency and downstream performance. Models are up to 3x faster at inference (via self-speculative decoding). The key insight: multi-token prediction acts as implicit regularization, encouraging the model to learn longer-range dependencies.
- **Applicability:** Directly implementable. Add 2-3 small linear prediction heads (one per future token) on top of the transformer's hidden states during training. Each head shares the backbone but has its own output projection. The heads are ~1M params each (512 x 1024 vocab) and are discarded post-training. Expected training slowdown: ~15-20% per step, but the improved convergence should more than compensate.

**b. Pre-Training Curriculum for Multi-Token Prediction in Language Models**
- Authors: Ansar Aynetdinov, Alan Akbik
- Date: 2025-05-28 | ID: [2505.22757v1](https://arxiv.org/abs/2505.22757)
- Proposes gradually transitioning from next-token to multi-token prediction during training (a curriculum). The forward curriculum lets smaller models better leverage the MTP objective -- larger models benefit from MTP immediately, but smaller models need the ramp-up.
- **Applicability:** Highly relevant for parameter golf's small model regime. Start training with standard next-token prediction, then introduce 2-token prediction at 30% of training, then 3-token at 60%. This avoids the instability of starting MTP from scratch at small scale. The curriculum adds zero implementation complexity beyond a training phase flag.

**c. Future Token Prediction: Causal Language Modelling with Per-Token Semantic State Vector**
- Authors: Nicholas Walker
- Date: 2024-10-23 | ID: [2410.18160v1](https://arxiv.org/abs/2410.18160)
- Instead of separate prediction heads, generates embedding vectors that encode the next N tokens' semantics. These "future state vectors" are more token-focused and improve text classification and coherence.
- **Applicability:** Lower priority. The per-token semantic state approach is more complex to implement and its benefits are more relevant to downstream tasks than raw BPB on language modeling.

### Applicability Assessment

Multi-token prediction with curriculum scheduling is a strong candidate. The Meta paper (Gloeckle et al.) provides a well-validated recipe, and the curriculum approach (Aynetdinov & Akbik) specifically addresses the small-model regime. Implementation is straightforward: add 2 extra linear heads during training, apply a curriculum, discard heads before compression. **Expected impact: -0.003 to -0.008 BPB** from improved convergence within the same 10-minute budget.

---

## 4. Test-Time Training / Adaptation

**Why this matters for parameter golf:** LoRA TTT gave -0.037 BPB on the weak baseline but was never combined with the current SOTA. The 10-minute eval budget provides significant headroom for test-time adaptation. The key question is what form of TTT works best with modern quantized models.

### Key Papers

**a. End-to-End Test-Time Training for Long Context**
- Authors: Arnuv Tandon, Karan Dalal, Xinhao Li, et al.
- Date: 2025-12-29 | ID: [2512.23675v2](https://arxiv.org/abs/2512.23675)
- Uses a standard transformer with sliding-window attention, augmented by test-time next-token prediction training. Achieves comparable scaling to full attention while maintaining constant inference latency. 2.7x faster than full attention approaches.
- **Applicability:** Extremely relevant. The parameter golf eval already uses sliding window attention. This paper shows that continuing to train (adapt) the model on test data via next-token prediction during evaluation can substitute for longer context. Since the eval budget is 10 minutes and the model is small, there is ample time for TTT passes over the validation data.

**b. Learning to Discover at Test Time (TTT-Discover)**
- Authors: Mert Yuksekgonul, Daniel Koceja, et al.
- Date: 2026-01-22 | ID: [2601.16175v2](https://arxiv.org/abs/2601.16175)
- Applies RL at test time to optimize solutions for specific problems. Achieves SOTA on math, algorithm design, and scientific discovery benchmarks.
- **Applicability:** The RL-at-test-time concept is too expensive for BPB evaluation but the principle of continued optimization at test time is sound. More relevant is the simpler next-token prediction TTT from paper (a).

**c. Sleep-time Compute: Beyond Inference Scaling at Test-time**
- Authors: Kevin Lin, Charlie Snell, et al.
- Date: 2025-04-17 | ID: [2504.13171v1](https://arxiv.org/abs/2504.13171)
- Pre-computes model adaptations offline ("sleeping" on potential inputs) to reduce test-time compute by 5x while improving accuracy up to 18%.
- **Applicability:** Interesting concept but not applicable -- parameter golf evaluates on specific validation data that cannot be pre-processed before the eval window.

### Applicability Assessment

The End-to-End TTT paper is the most actionable. It validates the approach already tried in the LoRA TTT branch but with a more modern framing: use sliding-window attention + continuous next-token prediction adaptation during evaluation. The key implementation detail: use a small learning rate and update only a subset of parameters (e.g., LayerNorm scales, or a small LoRA adapter) to avoid catastrophic forgetting. **Expected impact: -0.010 to -0.025 BPB** when combined with current SOTA, based on the -0.037 BPB seen on the weaker baseline.

---

## 5. State-Space Models in Tiny Regimes

**Why this matters for parameter golf:** Mamba and other SSMs have O(n) complexity vs O(n^2) for attention, potentially allowing longer sequences or more training steps within the 10-minute budget. If SSMs match transformer quality at small scale, they could be a better architecture choice.

### Key Papers

**a. Differential Mamba**
- Authors: Nadav Schneider, Itamar Zimerman, Eliya Nachmani
- Date: 2025-07-08 | ID: [2507.06204v2](https://arxiv.org/abs/2507.06204)
- Introduces differential mechanisms to mitigate Mamba's "overallocation" of attention to irrelevant context. Improves retrieval capabilities and language modeling while maintaining efficiency.
- **Applicability:** The differential mechanism could be combined with a transformer backbone -- use Mamba-style layers for early processing (cheap, O(n)) and transformer attention for later layers (expensive, but only on a few layers). A hybrid approach could maximize throughput.

**b. Exploring the Limitations of Mamba in COPY and CoT Reasoning**
- Authors: Ruifeng Ren, Zhicong Li, Yong Liu
- Date: 2024-10-04 | ID: [2410.03810v3](https://arxiv.org/abs/2410.03810)
- Shows that constant-sized Mamba struggles with COPY operations and when scaled to match transformers on complex reasoning, loses its efficiency advantage.
- **Applicability:** Important negative result. For BPB evaluation on general web text, Mamba's limitations on copying and exact recall could hurt. Suggests pure Mamba is risky, but hybrid (Mamba + attention) could capture benefits of both.

**c. Mamba-based PKD for Efficient Knowledge Compression**
- Authors: Jose Medina, Amnir Hadachi, et al.
- Date: 2025-03-03 | ID: [2503.01727v2](https://arxiv.org/abs/2503.01727)
- Uses Mamba within progressive knowledge distillation to create extremely compact student models retaining only 1-5% of teacher FLOPs.
- **Applicability:** If a teacher model is available, distilling into a tiny Mamba student could produce a very compact model. However, the 10-minute training constraint makes this difficult unless the teacher is pre-trained.

### Applicability Assessment

Pure Mamba is risky for parameter golf due to limitations on exact recall tasks. A hybrid approach (Mamba layers for early processing, transformer attention for later layers) is more promising but adds significant implementation complexity. The modded-nanogpt competition has not adopted Mamba, suggesting it may not outperform well-optimized transformers at this scale. **Expected impact: uncertain, possibly -0.002 to -0.005 BPB** from the throughput improvement allowing more training steps, but could regress on recall-heavy validation passages.

---

## 6. Novel Attention Mechanisms

**Why this matters for parameter golf:** XSA (Exclusive Self Attention) is the current attention innovation in the SOTA. Linear attention or other efficient attention variants could further reduce compute cost (more steps in 10 min) or improve quality.

### Key Papers

**a. Kimi Linear: An Expressive, Efficient Attention Architecture**
- Authors: Kimi Team
- Date: 2025-10-30 | ID: [2510.26692v2](https://arxiv.org/abs/2510.26692)
- Hybrid linear attention combining "Kimi Delta Attention" with specialized algorithms. Outperforms full MLA (Multi-head Latent Attention) while reducing KV cache usage and improving decoding throughput.
- **Applicability:** The hybrid linear attention approach could replace standard attention in early layers (where full attention is less critical) while keeping softmax attention in later layers. This is a more principled version of the "training-time sliding window" idea from modded-nanogpt insights. The reduced KV cache also helps with longer sequences.

**b. Gated Sparse Attention: Combining Computational Efficiency with Training Stability**
- Authors: Alfred Shen, Aaron Shen
- Date: 2026-01-12 | ID: [2601.15305v1](https://arxiv.org/abs/2601.15305)
- Merges sparse and gated attention for 12-16x speedup at 128K context while reducing perplexity. Mitigates attention sink phenomena through gating.
- **Applicability:** The 128K context speedup is less relevant (parameter golf uses 2048 tokens), but the gating mechanism that mitigates attention sinks could improve quality at short context too. The sparse pattern could complement XSA -- use gated sparse attention for early layers and XSA for late layers.

**c. MASA: Share Your Attention via Matrix-based Dictionary Learning**
- Authors: Magauiya Zhussip, Dmitriy Shopkhoev, et al.
- Date: 2025-08-06 | ID: [2508.04581v2](https://arxiv.org/abs/2508.04581)
- Decomposes attention projection matrices into shared dictionary atoms, reducing attention parameters by 66.7%. Drop-in replacement, no distillation needed.
- **Applicability:** A 66.7% reduction in attention parameters is enormous. For parameter golf, this could free up ~4-5MB of the 16MB budget for wider MLPs or more layers. The dictionary atoms act as a form of weight sharing specifically for attention, complementing the depth recurrence approach (which shares entire layers). Could be combined: share attention via dictionary atoms AND share layers via Cycle(rev).

**d. Linear-InfSA: Towards Linear Transformers with Infinite Self-Attention**
- Authors: Giorgio Roffo, Luke Palmer, Nilli Lavie
- Date: 2026-02-26 | ID: [2603.00175v4](https://arxiv.org/abs/2603.00175)
- Treats attention as diffusion steps on token graphs with linear-time approximation. Supports stable training at large resolutions.
- **Applicability:** Lower priority. Designed for vision transformers with spatial structure; less directly applicable to language modeling where token relationships are less geometric.

### Applicability Assessment

MASA (attention dictionary decomposition) is the standout paper. A 66.7% reduction in attention parameters directly addresses the 16MB constraint. Combined with existing int6 QAT, it could enable a 14-15 layer model within budget. Kimi Linear is second priority for hybrid attention in early layers. **Expected impact of MASA: -0.003 to -0.008 BPB** from parameter savings reinvested in depth/width.

---

## 7. Curriculum Learning / Data Ordering

**Why this matters for parameter golf:** With only 10 minutes of training, the order and selection of training data matters enormously. Curriculum learning could front-load the most informative examples, improving convergence speed.

### Key Papers

**a. Pre-Training Curriculum for Multi-Token Prediction in Language Models**
- Authors: Ansar Aynetdinov, Alan Akbik
- Date: 2025-05-28 | ID: [2505.22757v1](https://arxiv.org/abs/2505.22757)
- (Also listed in Section 3.) Proposes curriculum strategies specifically for the MTP objective. The forward curriculum (easy-to-hard) helps smaller models, while larger models can use reverse curriculum.
- **Applicability:** Two-for-one: curriculum + multi-token prediction. The curriculum addresses both data ordering and learning objective scheduling.

**b. Efficient Pre-training via Concept-based Curriculum Masking**
- Authors: Mingyu Lee, Jun-Hyung Park, et al.
- Date: 2022-12-15 | ID: [2212.07617v1](https://arxiv.org/abs/2212.07617)
- Masks tokens based on linguistic difficulty and knowledge graph relationships, achieving comparable BERT performance at half the training cost.
- **Applicability:** The masking curriculum is for BERT-style models, not autoregressive LMs. However, the principle translates: weight the loss on harder/rarer tokens more heavily during early training, then shift to uniform loss later. This could be implemented as a simple loss scaling schedule.

**c. Is Child-Directed Speech Effective Training Data for Language Models?**
- Authors: Steven Y. Feng, Noah D. Goodman, Michael C. Frank
- Date: 2024-08-07 | ID: [2408.03617v2](https://arxiv.org/abs/2408.03617)
- Finds that local properties of training data affect model results more than global properties. Sequential arrangement matters more than overall composition.
- **Applicability:** Validates that data ordering matters for small model training. For parameter golf, this suggests shuffling strategy and batch composition could be tuned. Currently, data is likely randomly shuffled; a structured ordering (e.g., by document perplexity under a simple model) might help.

### Applicability Assessment

Curriculum learning is a moderate-effort, moderate-reward direction. The most practical approach for parameter golf is combining the MTP curriculum from Aynetdinov & Akbik with the existing training loop. Data ordering by difficulty would require a preprocessing step (score FineWeb documents by difficulty), which may not fit the constraints if data access is restricted. **Expected impact: -0.001 to -0.003 BPB** from curriculum scheduling; data ordering effects are uncertain.

---

## 8. Knowledge Distillation into Tiny Models

**Why this matters for parameter golf:** If a larger teacher model can be trained (or already exists), distilling its knowledge into the 16MB student could dramatically improve quality. The contest allows unlimited pre-computation for hyperparameter tuning.

### Key Papers

**a. BitNet Distillation (BitDistill)**
- Authors: Xun Wu, Shaohan Huang, et al.
- Date: 2025-10-15 | ID: [2510.13998v1](https://arxiv.org/abs/2510.13998)
- Fine-tunes full-precision LLMs into 1.58-bit precision using SubLN modules and attention distillation. Achieves 10x memory savings and 2.65x faster CPU inference with performance parity.
- **Applicability:** If contest rules allow starting from a pre-trained model and distilling into 1.58 bits within 10 minutes, this is a strong approach. The SubLN modules (learnable normalization per sublayer) could be adopted independently even without distillation.

**b. FBI-LLM: Scaling Up Fully Binarized LLMs from Scratch via Autoregressive Distillation**
- Authors: Liqun Ma, Mingjie Sun, Zhiqiang Shen
- Date: 2024-07-09 | ID: [2407.07093v1](https://arxiv.org/abs/2407.07093)
- Trains binary (1-bit) LLMs from scratch at 130M-7B parameter scale using autoregressive distillation. Shows that pre-trained weights are not needed for binary LLMs -- training from scratch with a distillation loss works.
- **Applicability:** Training a binary LLM from scratch with autoregressive distillation from a teacher could work IF a teacher is available. A 130M binary model would be ~16MB. The autoregressive distillation loss is a drop-in replacement for standard cross-entropy.

**c. DistillLens: Symmetric Knowledge Distillation Through Logit Lens**
- Authors: Manish Dhakal, Uthman Jinadu, et al.
- Date: 2026-02-14 | ID: [2602.13567v1](https://arxiv.org/abs/2602.13567)
- Aligns intermediate representations (not just output logits) between student and teacher using the logit lens technique. The symmetric divergence prevents overconfidence in compressed models.
- **Applicability:** Intermediate-layer alignment during distillation could help maintain quality when compressing to int5/int6. However, requires a teacher model and adds complexity.

**d. DQ-BART: Joint Distillation and Quantization**
- Authors: Zheng Li, Zijian Wang, et al.
- Date: 2022-03-21 | ID: [2203.11239v1](https://arxiv.org/abs/2203.11239)
- Jointly distills and quantizes sequence-to-sequence models, achieving 16.5x compression with minimal quality loss.
- **Applicability:** The joint distillation+quantization framework directly applies. Instead of separate QAT and training, jointly optimize both objectives. This could be combined with the existing int6 QAT by adding a distillation loss term.

**e. Oh! We Freeze: Improving Quantized Knowledge Distillation for LLMs**
- Authors: Kartikeya Bhardwaj, et al.
- Date: 2024-03-28 | ID: [2403.18159v2](https://arxiv.org/abs/2403.18159)
- Proposes "ov-freeze" for stable QKD (quantization + knowledge distillation) on LLMs. Achieves near floating-point precision at 4 bits on LLaMA.
- **Applicability:** The stability technique is relevant if pursuing 4-bit QAT with distillation. The freezing strategy (selectively freeze layers during QKD) could prevent the instability seen at very low bit widths.

### Applicability Assessment

Knowledge distillation is powerful but constrained by contest rules. If a pre-trained teacher is allowed, FBI-LLM's autoregressive distillation from scratch is the best approach (train a binary 130M model with teacher guidance in 10 min). If no teacher is allowed, self-distillation (train two models, use the ensemble as teacher) is possible but doubles compute cost. **Expected impact with teacher: -0.010 to -0.025 BPB. Without teacher: -0.002 to -0.005 BPB** from self-distillation.

---

## Top 10 Most Actionable Papers

Ranked by direct applicability to parameter golf's 16MB / 10-minute constraint, considering implementation difficulty, expected BPB improvement, and compatibility with the current SOTA approach.

| Rank | Paper | Key Technique | Expected BPB Impact | Implementation Effort | Why It Ranks Here |
|------|-------|--------------|--------------------|-----------------------|-------------------|
| 1 | **Relaxed Recursive Transformers** (2410.20672) | Layer sharing with per-loop LoRA | -0.005 to -0.015 | Medium | Directly addresses the 16MB constraint by enabling 12+ effective layers with 4-6 unique layer blocks. Per-loop LoRA adds minimal parameters. Completely untried in parameter golf. |
| 2 | **Better & Faster LLMs via Multi-token Prediction** (2404.19737) | Multi-token prediction training | -0.003 to -0.008 | Low | Zero cost at inference (heads discarded). Improves sample efficiency, critical for 10-min training. Well-validated at Meta. |
| 3 | **End-to-End TTT for Long Context** (2512.23675) | Test-time training via next-token prediction | -0.010 to -0.025 | Medium | Validates the LoRA TTT approach on modern architectures. The -0.037 BPB seen on baseline could partially transfer to SOTA. Uses the same sliding-window setup already in place. |
| 4 | **Lessons on Parameter Sharing** (2104.06022) | Cycle(rev) layer sharing pattern | -0.003 to -0.010 | Low | Simplest implementation of depth recurrence: just change layer ordering. Palindromic pattern naturally creates U-Net flow. Drop-in change to layer definitions. |
| 5 | **MASA: Attention via Matrix-based Dictionary** (2508.04581) | Attention parameter sharing via dictionary atoms | -0.003 to -0.008 | Medium | 66.7% reduction in attention parameters frees massive budget for depth/width. Drop-in replacement for attention projections. |
| 6 | **Pre-Training Curriculum for MTP** (2505.22757) | Curriculum scheduling for multi-token prediction | -0.002 to -0.004 | Low | Specifically addresses the small-model regime. Combines naturally with paper #2. The forward curriculum is a simple training schedule change. |
| 7 | **HGF: Stabilizing 1.58-bit LLMs** (2602.05269) | Ternary backbone + FP16 low-rank correction | -0.005 to -0.020 | High | Enables ~45M ternary params + 2M FP16 correction in 16MB. High risk (ternary at small scale is unproven) but enormous upside if it works. |
| 8 | **Spectra 1.1: Ternary Scaling Laws** (2506.23025) | Scaling laws for ternary LMs | Informational | N/A | Essential reference for deciding optimal ternary model size within 16MB. The "data > params" finding guides architecture choices. |
| 9 | **Parallel Loop Transformer** (2510.24824) | Parallelized looped execution | -0.002 to -0.005 | High | Solves the training speed bottleneck of depth recurrence. Critical enabler for paper #1 to fit in 10 minutes. |
| 10 | **BitNet Distillation** (2510.13998) | Distillation into 1.58-bit models | -0.010 to -0.025 | High | If a teacher model is available, this is the fastest path to a competitive 1.58-bit model. The SubLN technique is independently useful. |

### Recommended Implementation Priority

**Phase 1 (lowest effort, highest confidence):**
1. Add multi-token prediction (2-3 heads) with curriculum scheduling (papers #2 + #6)
2. Try Cycle(rev) layer sharing pattern (paper #4) -- literally just redefine layer order

**Phase 2 (moderate effort, high potential):**
3. Implement Relaxed Recursive Transformers with per-loop LoRA (paper #1)
4. Add TTT next-token prediction during evaluation (paper #3)
5. Apply MASA attention dictionary decomposition (paper #5)

**Phase 3 (high effort, speculative):**
6. Explore ternary quantization with HGF correction path (paper #7)
7. If teacher available, try BitNet distillation (paper #10)

### Key Insight

The biggest untapped opportunity is the combination of **depth recurrence** (papers #1, #4) with **multi-token prediction** (papers #2, #6) and **test-time training** (paper #3). These three techniques are orthogonal:
- Depth recurrence reduces parameter count per effective layer, freeing space within 16MB
- Multi-token prediction improves convergence within the 10-minute training budget
- Test-time training improves evaluation quality within the 10-minute eval budget

None of these have been tried in parameter golf. Together, they could yield -0.020 to -0.040 BPB improvement over current SOTA.

---

## 9. Novel Quantization Approaches (Additional)

### TurboQuant: PolarQuant + QJL Residuals
- **Source:** [Google Research Blog](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)
- **What it does:** Two-stage compression pipeline. Stage 1 (PolarQuant) converts weight vectors from Cartesian to polar coordinates (radius + angles), mapping them onto a fixed circular grid. This eliminates the clipping/outlier problems of standard linear quantization — no need to find optimal clip percentiles per row. Stage 2 (QJL) applies 1-bit sign quantization to the residual errors, recovering lost precision at near-zero memory cost.
- **Key result:** Achieves 3-bit quantization with no accuracy loss, **post-training** (no QAT needed). 4-bit achieves 8x performance increase on H100s.
- **Original context:** Designed for KV cache compression at inference time, not weight compression.

### Applicability to Parameter Golf

**The opportunity:** If PolarQuant's coordinate transform works for weight compression (not just KV cache), it could replace the current per-row linear int6 quantization with 3-4 bit quantization — roughly **doubling** the effective parameter count from ~20M to ~40M within 16MB.

**What makes it interesting vs current approach:**
- The SOTA uses GPTQ-lite (trying 5 clip percentiles per row) to minimize int6 quantization error. PolarQuant's circular grid avoids the clipping problem entirely — the grid is uniform by construction.
- Post-training means it could be a drop-in replacement for the export pipeline, no QAT changes needed.
- Could potentially stack with QAT: train with STE fake-quantize at int6, then compress further with PolarQuant at export time.

**Limitations:**
- Untested on weight compression (only validated on KV cache)
- Benchmarked on large models (Gemma, Mistral), not tiny 16MB models
- The polar coordinate transform adds decode overhead at inference — may slow eval
- At tiny model scales, every bit of precision matters more, so 3-bit may degrade quality more than on large models

**Verdict:** Secondary priority behind StableQAT and BitNet approaches, but the PolarQuant idea is novel enough to warrant a quick experiment. Easiest test: apply PolarQuant to the SOTA's exported weights (post-training, no retraining) and measure BPB delta vs standard int6+zstd-22. If competitive, try PolarQuant-aware training.

**Expected impact:** If it enables stable 3-bit weight compression: -0.010 to -0.020 BPB (from doubled parameter count). If limited to post-training polish: -0.001 to -0.002 BPB (from better quantization of existing weights).

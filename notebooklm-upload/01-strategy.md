# Parameter Golf: Comprehensive Competition Strategy

**Generated:** 2026-03-24
**Target audience:** Beginner-intermediate ML practitioner
**Goal:** Beat the current SOTA of 1.1228 BPB on the 10min/16MB track

---

## 1. Top Solutions Summary

### Current Top 5 at a Glance

| Rank | BPB | Author | Core Innovation |
|------|-----|--------|-----------------|
| 1 | 1.1228 | signalrush | GPTQ-lite clip search + EMA stacked on full technique stack |
| 2 | 1.1248 | jfprincz | Partial RoPE (16/64 dims) + LN Scale -- zero-parameter wins |
| 3 | 1.1271 | jfprincz | XSA on last 4 layers + EMA replacing SWA |
| 4 | 1.1307 | unnir | Efficient GQA-aware XSA implementation + FA3 |
| 5 | 1.1428 | thwu1 | Mixed int5/int6 quantization + BigramHash 10240 buckets |

**Entry #1 (1.1228)** is a culmination of every prior innovation. Its two unique contributions are (a) GPTQ-lite, which tries 5 clip percentiles per weight row at export time and picks the one minimizing reconstruction error (-0.0006 BPB, zero training cost), and (b) stacking EMA (decay=0.997) on top of tight SWA for smoother weights that quantize better (-0.0006 BPB). The rest is inherited: 11 layers, XSA on last 4 layers, Partial RoPE, SmearGate, BigramHash, int6+zstd-22, Muon with WD=0.04, sliding window eval. This entry proves that marginal gains from post-training polish still matter.

**Entry #2 (1.1248)** introduced two zero-parameter architectural changes. Partial RoPE applies positional encoding to only 25% of head dimensions (16 of 64), letting the other 75% learn position-invariant patterns like "the always follows of." LN Scale dampens deeper layer outputs by 1/sqrt(layer_idx+1), stabilizing training. Both are free in parameters and compute. Notably, the Late QAT flag in this entry was dead code (constant-folded by torch.compile), so the entire gain came from these two simple ideas.

**Entry #3 (1.1271)** introduced Exclusive Self Attention (XSA) from arXiv:2603.09078 -- after standard attention, subtract the component aligned with each token's own value vector, forcing the model to learn from context rather than self-reinforcing. Applied only to the last 4 layers where self-attention bias is highest. Also replaced SWA with EMA, producing smoother weight distributions that compress better under int6.

**Entry #4 (1.1307)** solved a key engineering bottleneck: the naive XSA implementation with GQA required expensive `repeat_interleave` operations. This entry uses a free reshape into KV head groups plus broadcasting, cutting XSA overhead from ~7ms/step to ~2ms/step. This made XSA practical for the 10-minute budget.

**Entry #5 (1.1428)** took a different approach to the size budget. Instead of uniform int6, it used int5 for MLP weights (which tolerate lower precision) and int6 for attention weights. The ~1.86MB savings funded a 10th layer. It also scaled BigramHash from 4096 to 10240 buckets, reducing hash collisions for an additional -0.001 BPB.

**The common thread:** Every top entry shares a core stack -- int6+zstd-22 compression, MLP 3x expansion, sliding window eval, SmearGate, BigramHash, Muon with WD=0.04, and weight averaging. The differentiation comes from attention innovations (XSA, Partial RoPE), quantization refinements (GPTQ-lite, int5), and weight averaging strategies (EMA vs SWA).

---

## 2. Technique Impact Ranking

All techniques ranked by BPB improvement per unit of implementation complexity. "Complexity" is rated 1-5 (1 = config change, 5 = novel architecture). "Risk" is the chance of no improvement or regression.

| Technique | BPB Impact | Complexity | Risk | Notes |
|-----------|-----------|------------|------|-------|
| **Sliding window eval (stride=64)** | -0.033 | 1 | Very Low | Biggest single free win. Every token gets 960+ tokens of context. Eval takes ~70-160s instead of ~16s. |
| **MLP 3x expansion (hidden=1536)** | -0.015 | 2 | Low | Requires int6 compression to fit. Largest capacity-based gain. |
| **Int6 QAT (STE fake-quantize)** | -0.015 | 3 | Low | Eliminates quantization gap entirely. Enables fitting more params in 16MB. ~28% training overhead. |
| **Lower LR (0.04 -> 0.02)** | -0.006 | 1 | Very Low | Pure hyperparameter sweep. Reduces quant penalty too. |
| **FP16 tied embedding** | -0.006 | 1 | Very Low | Skip int8/int6 quant on embedding. Huge per-byte impact since embed is most sensitive tensor. |
| **zstd-22 compression** | -0.005 (indirect) | 1 | Very Low | Saves ~1.5MB vs zlib, funding more parameters. Drop-in replacement. |
| **Muon WD=0.04** | -0.004 | 1 | Low | Keeps weights small for quantization. Sweet spot from 0.01-0.05 sweep. |
| **SmearGate** | -0.003 | 2 | Low | ~512 learned params. Blends current/previous token embeddings. Free bigram context. |
| **BigramHash (2048-10240 buckets)** | -0.003 | 2 | Low | Hash table for token pairs. ~260K-1.3M params. Complements SmearGate. |
| **EMA (decay=0.997)** | -0.003 | 2 | Low | Smoother weights than SWA. Better generalization AND better quantization. |
| **11th layer + U-Net skips** | -0.003 | 2 | Low | Funded by int6 savings. U-Net connects encoder to decoder layers. |
| **XSA (last 4 layers)** | -0.002 | 3 | Low | Forces attention to use context, not self-reference. Zero new params. |
| **Partial RoPE (16/64 dims)** | -0.002 | 1 | Very Low | One-line change. 75% of dims attend without positional bias. |
| **Muon momentum 0.99 + warmup** | -0.002 | 1 | Low | Warmup from 0.92 over 1500 steps prevents instability. |
| **Longer warmdown (3000-3500)** | -0.002 | 1 | Very Low | Extended LR decay phase. Tune based on total training steps. |
| **Orthogonal init** | -0.001 | 2 | Very Low | Uniform gradient flow at init. Helps with limited training steps. |
| **SWA (stochastic weight averaging)** | -0.001 | 2 | Very Low | Average checkpoints during warmdown. Largely superseded by EMA. |
| **LN Scale (1/sqrt(layer+1))** | -0.001 | 1 | Very Low | Dampens deep layer activations. Stabilizes 11L training. |
| **Int5 for MLP weights** | -0.001 | 2 | Low | Only helps if you use the byte savings for more capacity (e.g., extra layer). |
| **GPTQ-lite (clip percentile search)** | -0.001 | 2 | Very Low | Post-training only. Tries 5 clip levels per weight row. Zero training cost. |
| **Late QAT (threshold=0.15)** | -0.0001 | 1 | Very Low | Start fake-quantize slightly earlier in training. Marginal but free. |
| **FlashAttention 3** | indirect | 2 | Very Low | More steps in 10 min. No direct BPB gain but enables longer seq/more layers. |
| **Seq 2048** | -0.005 | 1 | Low | More context per training example. Costs ~20% more per step. |

---

## 3. Integration Opportunities

### Low-Risk Combinations (proven techniques, just not tried together)

**A. BigramHash 10240 + current SOTA stack**
- What: Change BIGRAM_VOCAB_SIZE from 2048 to 10240 in the SOTA config.
- Why it should work: The 10240-bucket version showed -0.001 BPB in a weaker 10L model (entry #5). The current SOTA uses only 2048 buckets. Fewer hash collisions means cleaner bigram features. The parameter cost is ~1MB, which should fit within the 16MB budget given the SOTA artifact is 15.55MB (450KB headroom).
- Expected BPB gain: -0.001 to -0.002
- Effort: One config line change.

**B. GPTQ-lite clip search on Int5 MLP weights**
- What: Apply the per-row clip percentile search (currently used for int6) to int5-quantized MLP weights.
- Why it should work: Int5 has more quantization error per row than int6 (32 levels vs 64). The clip search finds the optimal "zoom level" per row, which should yield a proportionally larger improvement with coarser quantization.
- Expected BPB gain: -0.001 to -0.002
- Effort: Extend existing GPTQ-lite code to handle int5.

**C. NTK-RoPE eval extrapolation + Partial RoPE**
- What: At eval time, scale RoPE frequencies to handle 1.375x the training context length (e.g., eval at 1408 if trained at 1024, or 2816 if trained at 2048). Combine with Partial RoPE.
- Why it should work: NTK-RoPE extrapolation showed benefit on early entries. Partial RoPE changes the dynamics -- with only 16/64 dims carrying positional info, extrapolation applies to fewer dimensions and may be more stable.
- Expected BPB gain: -0.001 to -0.003
- Effort: A few lines in the eval code.

### Medium-Risk Combinations (require engineering but plausible)

**D. LoRA TTT on current SOTA base**
- What: Apply per-document rank-8 LoRA test-time training at eval, adapting c_q, c_v, and lm_head per document.
- Why it should work: LoRA TTT gave -0.003 BPB improvement on the *baseline* model (entry #14). The SOTA model is much stronger, so some of that signal may already be captured. But per-document adaptation should still help because documents have unique statistical patterns the base model cannot memorize. The eval budget is 10 minutes, and entry #14 used only ~1 minute.
- Risk: The 11L model is heavier per forward pass. LoRA TTT requires multiple forward+backward passes per document. May not fit in 10-minute eval budget. Need to benchmark timing.
- Expected BPB gain: -0.005 to -0.015 (high variance)
- Effort: 1-2 days. Port the LoRA TTT code from entry #14, adapt to 11L architecture, optimize for speed.

**E. Simplified hyperconnections replacing U-Net skips**
- What: Replace the hardcoded encoder-decoder U-Net skip connections with learned skip weights between arbitrary layer pairs, following modded-nanogpt record #77.
- Why it should work: U-Net skips are a special case of hyperconnections. Learning which layer pairs should connect could find better information routing. The parameter cost is tiny (one scalar per connection). Modded-nanogpt showed this helps at similar model scales.
- Risk: May not improve if U-Net skips are already near-optimal. Could slightly slow training if activation memory increases.
- Expected BPB gain: -0.002 to -0.004
- Effort: 2-3 days. Requires modifying the transformer forward pass.

**F. 12 layers with Int5 MLP + GPTQ-lite**
- What: Push to 12 transformer layers by using int5 for all MLP weights (saving ~1.86MB) and compensating quality loss with GPTQ-lite clip search.
- Why it should work: Going from 9 to 11 layers gave cumulative -0.003 BPB. A 12th layer should give another -0.001 to -0.002 BPB. Int5 MLP is proven. GPTQ-lite should recover some int5 quality loss.
- Risk: Training speed. At ~85ms/step for 11L, 12L would be ~93ms/step, yielding ~6,450 steps in 10 minutes (vs ~7,100 for 11L). Fewer steps may negate the depth benefit.
- Expected BPB gain: -0.001 to -0.003 (net, after accounting for fewer steps)
- Effort: 1-2 days. Architecture change + timing verification.

### High-Risk / High-Reward

**G. Multi-token prediction training**
- What: Add 1-2 additional prediction heads that predict tokens 2 and 3 steps ahead. Heads are discarded at export (zero size cost). The extra loss signal improves training efficiency.
- Why it should work: Modded-nanogpt record #53 showed significant speedup from multi-token prediction. In parameter golf, the benefit is not speed but *better model quality per training step*. With only ~7K steps, extracting more learning signal per step is valuable.
- Risk: Extra heads add compute per step (~15-30% overhead), meaning fewer total steps. The quality-per-step improvement must outweigh the step count reduction. This trade-off is unverified in the parameter golf setting.
- Expected BPB gain: -0.003 to -0.005 if positive; could be neutral or slightly negative.
- Effort: 2-3 days.

**H. Ultra-low-bit quantization (2-4 bit) with StableQAT**
- What: Replace int6 STE QAT with the Fourier-based gradient surrogate from StableQAT (arXiv:2601.19320). Train with 3-4 bit weights, packing ~40-50M effective parameters into 16MB.
- Why it should work: Current SOTA uses int6 (6 bits/weight) for ~20M params. At 3 bits, you could fit ~40M params -- double the capacity. StableQAT specifically addresses training stability at ultra-low bits, which is the main failure mode.
- Risk: High. 3-bit QAT has never been tried in this competition. The Fourier surrogate is a research technique, not battle-tested. Training instability could waste the entire 10-minute budget.
- Expected BPB gain: -0.010 to -0.030 if it works (transformative). But 50%+ chance of failure.
- Effort: 3-5 days.

**I. LoRA TTT + sliding window + current SOTA (triple eval stack)**
- What: Combine all three eval-time techniques: sliding window (stride=64), per-document LoRA TTT, and NTK-RoPE extrapolation.
- Why it should work: Each targets a different aspect -- sliding window gives context, LoRA TTT gives per-document adaptation, NTK-RoPE gives position generalization. They should be roughly orthogonal.
- Risk: Eval time budget. Sliding window alone takes ~70-160s. LoRA TTT takes ~60-120s on the baseline. On 11L with both, it may exceed 10 minutes. Requires aggressive optimization (smaller LoRA rank, fewer TTT steps, parallel processing).
- Expected BPB gain: -0.005 to -0.020 (if it fits in time)
- Effort: 2-3 days.

---

## 4. Unexplored Directions

### 4.1 Alternative Architectures

**State-space models (Mamba / S4) within 16MB**
- Mamba replaces attention with selective state spaces, giving linear-time sequence processing. For a 16MB model, this could enable much longer effective context without the quadratic attention cost. However, no one has tried training a Mamba model from scratch in 10 minutes on this dataset. The risk is that Mamba may underperform transformers at this tiny scale. The Muon optimizer (designed for matrix operations) may not pair well with Mamba's recurrent structure.
- Verdict: High risk, high potential. Would require building a new training pipeline from scratch.

**RWKV-style linear attention**
- RWKV uses a linear attention mechanism with time-decay weighting. The arxiv paper on TTT with KV Binding (2602.21204) shows that test-time training architectures can be reframed as linear attention. A hybrid approach -- linear attention for most layers, full attention for the last 2-3 -- could save compute and enable more layers.
- Verdict: Medium risk. Could be tested by replacing attention in early layers only.

### 4.2 Novel Tokenizers

**Byte-level with pooling**
- Instead of the 1024-token BPE vocabulary, use raw bytes (vocab=256) with a learned pooling/upsampling layer. This eliminates tokenizer artifacts and could improve compression on unusual text. The challenge is that byte sequences are ~4x longer than BPE, requiring more compute per sequence.
- Verdict: Low priority. The 1024-BPE tokenizer is already very efficient for this dataset size.

**Larger BPE vocabulary**
- The current 1024-token BPE may be suboptimal. A 2048 or 4096 vocabulary would better cover common word fragments but doubles/quadruples the embedding table size. With FP16 embeddings, 4096 vocab at dim=512 = 4MB (25% of budget).
- Verdict: Worth a quick ablation. The embedding cost may be prohibitive, but the quality gain per token could offset it.

### 4.3 Depth Recurrence / Weight Sharing

**Looped layers (use 4 layers, loop 3 times = 12 effective depth)**
- Explicitly noted as "promising but needs more steps" in entry #20 and "worst result (+0.051 BPB)" in the single-GPU entry. The problem is that each loop pass counts as a training step's worth of compute but does not see new data, effectively halving throughput.
- However, with int5/int6 compression, a 4-layer model that loops 3x would use only ~4 layers of parameters (~5-6MB) while having 12 effective layers. The remaining 10MB could fund a massive MLP (5-6x expansion) or additional embedding tables.
- Verdict: The prior negative results were on the unoptimized baseline. With current compression techniques freeing parameter budget, it deserves a re-evaluation. But budget 3+ days for implementation and tuning.

### 4.4 Knowledge Distillation

**Is this allowed?**
- The rules state "no external downloads, network calls, or training data access during evaluation." During *training*, the constraint is 10 minutes on 8xH100. You could pre-train a larger teacher model (outside the 10 minutes), then distill into the 16MB student during the 10-minute window. However, the spirit of the competition seems to be training from scratch.
- The 4-hour entry (1.2074 BPB) showed that simply training longer without quantization awareness does not help much. A teacher model would need to be quantization-aware itself to provide useful guidance.
- Verdict: Likely allowed technically but ethically gray. The NanoNet paper (2602.06093) suggests mutual learning between small models as an alternative that avoids a pre-trained teacher.

### 4.5 Better Quantization

**Int4 / mixed-precision per-layer adaptive**
- The Attn-QAT paper (2603.00040) demonstrates stable 4-bit QAT for attention on FP4-capable GPUs. At 4 bits/weight, a 16MB model could hold ~32M parameters -- 60% more than int6. The HESTIA paper (2601.20745) uses Hessian-guided sensitivity to allocate precision per layer.
- A practical approach: use the current layer-stratified quantization idea (int8 for early/late layers, int6 for middle) but push further -- int4 for the least sensitive middle layers, int6 for attention-critical layers, FP16 for embeddings.
- Verdict: Medium risk. Requires implementing the Fourier/Hessian QAT approaches from recent papers. Could be transformative if stable.

**Ternary (1.58-bit) with Sparse-BitNet**
- Sparse-BitNet (2603.05168) shows 1.58-bit models are naturally compatible with N:M structured sparsity. At 1.58 bits/weight, ~80M parameters would fit in 16MB. Combined with sparsity, effective capacity could reach ~60M dense-equivalent parameters.
- Verdict: Very high risk. Ternary quantization from scratch in 10 minutes has never been attempted. But the potential capacity gain is enormous.

### 4.6 Test-Time Compute

**LoRA TTT on current SOTA**
- The original LoRA TTT entry only tested on the weak baseline. On the current SOTA, the absolute gain may be smaller, but per-document adaptation should still capture signal that a fixed model cannot. The SR-TTT paper (2603.06642) suggests focusing adaptation on high-surprisal tokens for efficiency.
- Verdict: High priority. This is the single most promising untried combination. See Integration Opportunity D above.

**Test-Time Control (TTC) layers**
- The TTC paper (2603.09221) adds inference-time LQR planning. This is more relevant for reasoning tasks than compression, but the FineWeb dataset contains diverse text where adaptive inference could help.
- Verdict: Low priority for BPB on FineWeb. More relevant if the competition adds reasoning benchmarks.

### 4.7 Ensemble Methods Within 16MB

**Two models, half size each**
- Train two ~8MB models with different random seeds or architectures. At eval, average their logits. Ensemble averaging typically improves by -0.005 to -0.010 BPB on language modeling tasks.
- The challenge: each 8MB model would be much weaker individually (only ~10M params each at int6). The ensemble benefit must overcome the capacity loss.
- Verdict: Worth testing but unlikely to beat a single strong model. The capacity penalty is too severe at 16MB.

**Single model with dropout-based implicit ensemble**
- Use MC Dropout at eval time -- run multiple forward passes with different dropout masks, average the logit outputs. This is a free ensemble that requires no extra parameters.
- Verdict: Low effort, low risk. May give -0.001 to -0.002 BPB. Worth a quick test.

### 4.8 Curriculum Learning / Data Ordering

**Easy-to-hard training**
- Order FineWeb training data by difficulty (e.g., perplexity under a simpler model), training on easy examples first. This could improve convergence in the limited 7K-step budget by building foundations before tackling hard examples.
- Verdict: Requires pre-computing difficulty scores (outside the 10-minute budget, which is likely allowed). The data loading pipeline would need modification. Medium effort, uncertain reward.

**Deduplication / quality filtering**
- Remove near-duplicate training sequences or low-quality text. With only ~5.6B tokens seen in training, quality matters more than quantity.
- Verdict: The FineWeb dataset is already curated. Gains from further filtering are likely small.

### 4.9 Multi-Token Prediction

- From modded-nanogpt record #53 onward. Predict 2-3 future tokens per step using small additional heads. The heads are discarded at export, so zero size cost. The extra gradient signal improves sample efficiency.
- In parameter golf, this means better model quality from the same number of training tokens. The trade-off is ~15-30% more compute per step (fewer total steps in 10 minutes).
- Verdict: Medium-high priority. Untried in parameter golf. See Integration Opportunity G above.

### 4.10 QK-Norm

- From modded-nanogpt record #5. Apply RMSNorm to Q and K before computing attention scores. Stabilizes attention dot products in deep models. The current SOTA uses LN Scale (1/sqrt(layer+1)) which is related but less targeted.
- QK-Norm could replace or complement LN Scale. It is a zero-parameter change (RMSNorm has no learnable params when applied to Q/K this way).
- Verdict: High priority, low effort. Should be tested immediately.

### 4.11 Simplified Hyperconnections

- From modded-nanogpt record #77. Learned skip connections between arbitrary layer pairs, generalizing U-Net skips. Uses a single saved activation reused across multiple connections. Tiny parameter cost (one scalar per connection).
- Current SOTA uses fixed U-Net skip connections. Hyperconnections could find better routing.
- Verdict: Medium priority, medium effort. See Integration Opportunity E above.

---

## 5. Concrete Next Steps

### Quick Wins (less than 1 day each)

#### Step 1: Reproduce baseline locally
- **What:** Run the baseline `train_gpt.py` on your hardware. If on Apple Silicon, use `train_gpt_mlx.py` for smoke testing. For real results, use RunPod with an H100.
- **Commands:**
  ```bash
  # Local smoke test (Apple Silicon)
  python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1
  RUN_ID=baseline ITERATIONS=200 TRAIN_BATCH_TOKENS=8192 python3 train_gpt_mlx.py

  # Full run (RunPod H100)
  python3 data/cached_challenge_fineweb.py --variant sp1024
  RUN_ID=baseline torchrun --standalone --nproc_per_node=1 train_gpt.py
  ```
- **Expected BPB:** ~1.2244 (matches leaderboard baseline)
- **Compute:** MLX smoke test: 5-10 min. Full H100 run: 10 min.
- **Dependencies:** None. This is step 0.

#### Step 2: Reproduce current SOTA
- **What:** Run the SOTA entry's `train_gpt.py` from `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015/` on 8xH100.
- **Expected BPB:** ~1.1228
- **Compute:** 8xH100, 10 min training + 10 min eval.
- **Dependencies:** Step 1 (familiarity with the codebase).

#### Step 3: BigramHash bucket size sweep
- **What:** In the SOTA config, change BIGRAM_VOCAB_SIZE from 2048 to 4096, 8192, and 10240. Run 3 seeds each.
- **Expected BPB improvement:** -0.001 to -0.002
- **Compute:** 4 runs x 3 seeds x 20 min = ~4 GPU-hours on 8xH100.
- **Dependencies:** Step 2.

#### Step 4: Warmdown sweep
- **What:** Try warmdown values of 4000, 4500, 5000, and 7000 (currently 3500). Run 3 seeds each.
- **Expected BPB improvement:** -0.0005 to -0.002
- **Compute:** 4 runs x 3 seeds x 20 min = ~4 GPU-hours.
- **Dependencies:** Step 2.

#### Step 5: QK-Norm addition
- **What:** Add RMSNorm to Q and K projections before attention. Test both (a) QK-Norm alone and (b) QK-Norm + LN Scale together.
- **Expected BPB improvement:** -0.001
- **Compute:** 2 runs x 3 seeds x 20 min = ~2 GPU-hours.
- **Dependencies:** Step 2.

#### Step 6: Partial RoPE dimension sweep
- **What:** Try 8/64, 24/64, and 32/64 RoPE dims (currently 16/64). The optimal fraction may not be 25%.
- **Expected BPB improvement:** -0.0005 to -0.001
- **Compute:** 3 runs x 3 seeds x 20 min = ~3 GPU-hours.
- **Dependencies:** Step 2.

### Medium Effort (1-3 days each)

#### Step 7: LoRA TTT on current SOTA
- **What:** Port the LoRA TTT eval code from entry #14 (`records/track_10min_16mb/2026-03-17_LoRA_TTT/`). Adapt to the 11L architecture. Key changes:
  - Target c_q, c_v, and lm_head with rank-8 LoRA
  - Process documents in parallel across 8 GPUs
  - Use Adam with lr=0.01, 1 step per 256-token chunk
  - Reset LoRA params between documents
- **Critical constraint:** Must fit in 10-minute eval budget. Profile the per-document time first. If too slow, reduce LoRA rank to 4 or skip lm_head.
- **Expected BPB improvement:** -0.005 to -0.015
- **Compute:** Debugging on 1xH100 (1 day), final runs on 8xH100.
- **Dependencies:** Step 2.

#### Step 8: Hyperconnections replacing U-Net skips
- **What:** Replace the fixed U-Net skip connections with learned skip weights. Implementation:
  - Add a small matrix of learnable scalars (one per layer pair)
  - During forward pass, each layer's input is a weighted sum of all prior layer outputs
  - Initialize to match U-Net pattern (encoder-decoder pairs get weight 1.0, others 0.0)
- **Expected BPB improvement:** -0.002 to -0.004
- **Compute:** 1xH100 for debugging, 8xH100 for final.
- **Dependencies:** Step 2.

#### Step 9: 12 layers with Int5 MLP
- **What:** Add a 12th transformer layer. Use int5 for all MLP weights, int6 for attention. Apply GPTQ-lite clip search to int5 weights.
- **First:** Benchmark training speed. 12L must complete ~6,000+ steps in 10 minutes. If it is too slow, try reducing MLP expansion from 3x to 2.75x to speed up each step.
- **Expected BPB improvement:** -0.001 to -0.003 (net)
- **Compute:** 8xH100 for timing verification, then 3-seed runs.
- **Dependencies:** Step 2.

#### Step 10: Systematic hyperparameter optimization
- **What:** Run a grid search over the key hyperparameters that have not been fully swept:
  - EMA decay: [0.995, 0.997, 0.999]
  - Weight decay: [0.03, 0.04, 0.05, 0.06]
  - Learning rate: [0.020, 0.025, 0.030]
  - Late QAT threshold: [0.10, 0.15, 0.20, 0.25]
  - XSA layers: [last 2, last 3, last 4, last 5]
- **Expected BPB improvement:** -0.001 to -0.003 (cumulative)
- **Compute:** ~30 runs x 20 min = 10 GPU-hours on 8xH100.
- **Dependencies:** Step 2.

### Ambitious (3+ days each)

#### Step 11: Multi-token prediction training
- **What:** Add 1-2 additional linear prediction heads after the transformer. Head k predicts token at position t+k. The total loss is a weighted sum: `loss = loss_1 + 0.3*loss_2 + 0.1*loss_3`. Heads are discarded at export.
- **Implementation considerations:**
  - Heads share the same hidden representation (no extra transformer computation)
  - Each head is a single linear layer (dim=512 -> vocab=1024), adding ~500K params during training only
  - The shifted targets need careful masking at sequence boundaries
- **Expected BPB improvement:** -0.003 to -0.005 (if compute trade-off is favorable)
- **Compute:** 8xH100 for implementation and testing. Budget 1-2 days for debugging.
- **Dependencies:** Step 2.

#### Step 12: Ultra-low-bit quantization (3-4 bit)
- **What:** Implement 4-bit QAT using the Fourier-based gradient surrogate from StableQAT. If stable, push to 3-bit. This would roughly double the effective parameter count within 16MB.
- **Implementation:**
  - Replace the STE quantize/dequantize with the Fourier surrogate from the paper
  - Start with 4-bit for MLP weights only (safest)
  - If successful, extend to attention weights
  - Keep FP16 for embeddings
- **Expected BPB improvement:** -0.010 to -0.030 (if it works)
- **Compute:** 8xH100. Budget 3-5 days. High risk of training instability.
- **Dependencies:** Step 2, solid understanding of QAT mechanics.

#### Step 13: Depth recurrence re-evaluation
- **What:** Design a 4-layer transformer block that loops 3 times (12 effective layers, 4 layers of parameters). Use the freed parameter budget for 5-6x MLP expansion.
- **Key challenges:**
  - Previous attempts showed +0.051 BPB regression, but on the unoptimized baseline
  - With current techniques (int5 compression, GPTQ-lite, EMA), the smaller parameter set may quantize much better
  - Need to handle positional encoding carefully (shared RoPE across loops?)
  - Layer-specific normalization scales are essential (each loop iteration should have its own LN Scale)
- **Expected BPB improvement:** -0.005 to -0.015 (if the convergence issue is solved); could still regress.
- **Compute:** 8xH100. Budget 3-5 days.
- **Dependencies:** Step 2, Step 9 (understanding of depth vs. width trade-offs).

#### Step 14: Sparse-BitNet (1.58-bit + N:M sparsity)
- **What:** Implement ternary weight quantization with N:M structured sparsity, following the Sparse-BitNet paper. This is the most radical compression approach.
- **Expected model size:** ~80M params at 1.58 bits = ~16MB. With 2:4 sparsity, effective capacity is ~60M dense-equivalent.
- **Risk:** Very high. Nobody has attempted this in the competition. Training stability is the main concern.
- **Expected BPB improvement:** Potentially -0.030+ (transformative) or complete failure.
- **Compute:** 8xH100. Budget 5+ days.
- **Dependencies:** Strong understanding of quantization and sparsity. Steps 1-2 for baseline comparison.

---

## Priority Summary

If you have limited GPU time, here is the recommended order:

| Priority | Step | Expected Gain | Time | Compute |
|----------|------|--------------|------|---------|
| 1 | Reproduce SOTA (Step 2) | -- | 0.5 day | 8xH100, 20 min |
| 2 | BigramHash 10240 (Step 3) | -0.001 to -0.002 | 0.5 day | 8xH100, 4 hrs |
| 3 | QK-Norm (Step 5) | -0.001 | 0.5 day | 8xH100, 2 hrs |
| 4 | Warmdown sweep (Step 4) | -0.001 | 0.5 day | 8xH100, 4 hrs |
| 5 | LoRA TTT (Step 7) | -0.005 to -0.015 | 2 days | 8xH100, 8 hrs |
| 6 | Multi-token prediction (Step 11) | -0.003 to -0.005 | 3 days | 8xH100, 10 hrs |
| 7 | Hyperconnections (Step 8) | -0.002 to -0.004 | 2 days | 8xH100, 6 hrs |
| 8 | Ultra-low-bit QAT (Step 12) | -0.010 to -0.030 | 5 days | 8xH100, 20 hrs |

The quick wins (Steps 3-6) could collectively yield -0.003 to -0.006 BPB with minimal risk, potentially pushing below 1.120. LoRA TTT (Step 7) is the single highest-expected-value medium-effort experiment. Ultra-low-bit QAT (Step 12) is the moonshot that could reset the leaderboard if it works.

**Target: 1.115-1.118 BPB** from quick wins + LoRA TTT. **Stretch target: below 1.100 BPB** from ultra-low-bit quantization or multi-token prediction breakthroughs.

# Modded-NanoGPT Insights & Parameter Golf Cross-Analysis

**Date**: 2026-03-24
**Purpose**: Map techniques from the modded-nanogpt speedrun to the parameter-golf competition, identify what's been tried, and surface untried opportunities.

---

## Part A: Modded-NanoGPT Overview

### What Is Modded-NanoGPT?

Modded-nanogpt is a collaborative speedrun to train a 124M-parameter GPT-2-equivalent model to a target validation loss (3.28 nats) as fast as possible on 8xH100 GPUs. Starting from a 45-minute baseline (May 2024), the community has driven the time down to **1.435 minutes** (March 2026) -- a 31x speedup through purely algorithmic and systems innovations.

The competition optimizes for **training speed at fixed model size**. This contrasts with parameter golf, which optimizes for **model quality at fixed model size (16MB compressed)**. However, many of the same techniques apply because both competitions reward getting the most learning out of limited resources.

### Record Progression (Selected Milestones)

| # | Time | Date | Key Innovation |
|---|------|------|----------------|
| 1 | 45 min | May 2024 | llm.c baseline |
| 3 | 24.9 min | Oct 2024 | **Muon optimizer** |
| 5 | 15.2 min | Oct 2024 | **ReLU-squared, QK-Norm**, zero-init |
| 12 | 5.03 min | Nov 2024 | FlexAttention, 64K context |
| 20 | 2.99 min | Jan 2025 | Merged QKV, long-short attention, batched Muon |
| 29 | 2.73 min | Sep 2025 | **Flash Attention 3**, max_doc_len 2048 |
| 40 | 2.36 min | Oct 2025 | **Backout architecture** (layer skipping) |
| 53 | ~2.0 min | Dec 2025 | **Multi-token prediction** |
| 58 | 1.82 min | Jan 2026 | **Paired head attention** |
| 62 | 1.66 min | Jan 2026 | **Bigram hash embeddings** |
| 77 | 1.44 min | Mar 2026 | **Simplified hyperconnections** |

### Technique Deep-Dives

#### 1. Muon Optimizer

**What it does**: Muon (MomentUm Orthogonalized by Newton-Schulz) runs SGD with Nesterov momentum, then post-processes each weight matrix's gradient update by projecting it onto the nearest orthogonal matrix via 5 iterations of Newton-Schulz. This is equivalent to steepest descent under the spectral norm -- it normalizes the update so that all singular values contribute equally, preventing any single direction from dominating.

**Why it matters**: Muon converges faster than Adam for weight matrices because it treats the optimization landscape more uniformly. It pairs naturally with orthogonal initialization (starting weights already have equal singular values).

**Parameter golf relevance**: HIGH. Muon is already the dominant optimizer across all top parameter-golf entries. It allows reaching lower loss within the 10-minute training budget. The key parameter golf insight is that Muon's weight decay (WD=0.04) also keeps weight magnitudes small, directly benefiting int6 quantization quality.

#### 2. ReLU-Squared Activation

**What it does**: Replaces standard GELU with `max(0, x)^2`. The squaring amplifies the nonlinearity and creates sparser activations (more near-zero values). Despite its simplicity, it matches or exceeds GELU performance in this regime.

**Why it matters**: Sparser activations can help the model learn more selective features. The squared output also has a smoother gradient landscape than plain ReLU, aiding optimization.

**Parameter golf relevance**: HIGH. Already adopted universally across parameter-golf submissions (labeled as "relu^2" or "relu-squared"). The 3x MLP expansion (hidden=1536 for dim=512) combined with ReLU-squared is standard.

#### 3. QK-Norm

**What it does**: Applies RMSNorm to the query and key projections before computing attention scores. This stabilizes the attention distribution by preventing the dot products from growing too large in magnitude.

**Why it matters**: Without QK-Norm, attention logits can explode in deeper models, causing training instability. With it, you can train deeper models more reliably.

**Parameter golf relevance**: MODERATE. Not explicitly mentioned in parameter-golf READMEs, but the LN Scale technique (scaling RMSNorm outputs by `1/sqrt(layer_idx+1)`) used in the top entries serves a similar stabilization purpose. The modded-nanogpt version targets Q and K specifically, which could be more surgical.

#### 4. Bigram Hash Embeddings

**What it does**: Maps consecutive token pairs (prev_token, curr_token) through a hash function into a fixed-size embedding table (e.g., 4096 or 10240 buckets, dim=128). The resulting embedding is projected to model dimension and added to the token representation. This gives the model direct access to two-token sequential patterns without attention.

**Why it matters**: Language has strong local dependencies (e.g., "of the", "in a"). By injecting bigram features at the embedding layer, the model doesn't waste attention capacity rediscovering these patterns.

**Parameter golf relevance**: HIGH. Already widely adopted. The top entries use BigramHash with 2048 buckets (dim=128). One entry experimented with 10240 buckets, gaining ~0.001 BPB. The parameter cost is modest (~260K-1.3M params depending on bucket count).

#### 5. Paired Head Attention

**What it does**: Groups attention heads into complementary pairs that share computational resources. Some heads operate in projected subspaces with reduced dimensionality, while their partners compensate. This reduces FLOPs while maintaining representational capacity.

**Why it matters**: In the speed-focused modded-nanogpt setting, this saves wall-clock time. The insight is that not all heads need full dimensionality -- some can be "cheap" heads that capture simple patterns.

**Parameter golf relevance**: LOW-MODERATE. Parameter golf is not directly FLOPs-constrained (10 minutes is ample for the model sizes used). However, if paired heads allow you to fit more total heads or more layers within the 16MB budget, it could help. Worth exploring if the parameter savings can be reallocated to depth or width.

#### 6. Backout Architecture / Simplified Hyperconnections

**What it does**: Enables skip connections between non-adjacent layers -- for example, from layer 0 directly to layer 6, bypassing layers 1-5. The model learns which pathways to use, effectively allowing it to "back out" of unhelpful intermediate computations. The simplified version uses a single saved activation reused across multiple skip connections.

**Why it matters**: Standard residual connections only connect adjacent layers. Hyperconnections allow the model to route information more efficiently through the network, potentially learning to skip layers that aren't helpful for certain inputs.

**Parameter golf relevance**: MODERATE-HIGH. Parameter-golf entries already use U-Net skip connections (connecting encoder layers to decoder layers). Hyperconnections are a generalization of this idea. The key question is whether the additional skip connection parameters (small) justify the quality improvement within the 16MB budget. This is an **untried opportunity**.

#### 7. Value Embeddings

**What it does**: Injects learned embeddings directly into the attention value projections, providing a secondary information channel. Later variants use a "U-Net pattern" with skip connections and gating to modulate the value embedding contribution per layer.

**Why it matters**: Value embeddings give the model additional capacity to inject task-specific features into the attention mechanism without increasing the Q/K computation.

**Parameter golf relevance**: MODERATE. The top parameter-golf entry uses "Shared Value Embedding (dim=128, layers 9,10) with per-layer learned scales." This is a direct adoption from modded-nanogpt. The technique adds a small number of parameters for a meaningful quality boost.

#### 8. Flash Attention 3

**What it does**: A hardware-optimized attention implementation that exploits Hopper GPU features (H100) for faster attention computation with reduced memory bandwidth requirements. Uses block-wise algorithms to avoid materializing the full attention matrix.

**Why it matters**: Faster attention means more training steps in the 10-minute budget, and lower memory means you can use larger batch sizes or longer sequences.

**Parameter golf relevance**: HIGH. Already adopted in top entries. FlashAttention 3 with Hopper optimizations is standard. The key parameter-golf-specific insight is combining FA3 with sliding window attention for efficient evaluation.

#### 9. Multi-Token Prediction

**What it does**: Instead of predicting only the next token, the model predicts multiple future tokens per training step. This increases the learning signal per gradient update without proportionally increasing compute.

**Why it matters**: More efficient use of each training example, improving sample efficiency.

**Parameter golf relevance**: MODERATE. Not currently used in any parameter-golf entry. This is an **untried opportunity**. The concern is that additional prediction heads add parameters that must fit in 16MB, but the technique could improve convergence enough to offset this.

---

## Part B: Cross-Analysis Table

### Technique Adoption Matrix

| Modded-NanoGPT Technique | Used in Param Golf? | Where / How | Impact in Param Golf |
|---|---|---|---|
| **Muon optimizer** | YES (all top entries) | WD=0.04, momentum 0.99, warmup 0.92->0.99 over 1500 steps | Core optimizer; enables faster convergence |
| **ReLU-squared** | YES (all entries) | Standard MLP activation since early submissions | Universal; no one uses GELU anymore |
| **Bigram hash embeddings** | YES (all top entries) | 2048-10240 buckets, dim=128, projected to 512 | ~0.003-0.005 BPB improvement |
| **Flash Attention 3** | YES (top entries) | Hopper-optimized, direct flash_attn_func calls | More steps/10min, enables seq_len=2048 |
| **Value embeddings** | PARTIAL (1 entry) | Shared value emb (dim=128, layers 9-10) in top entry | Small but positive contribution |
| **QK-Norm** | NO (related: LN Scale) | LN Scale (1/sqrt(layer+1)) serves similar purpose | LN Scale gives ~0.002 BPB; QK-Norm untried |
| **Paired head attention** | NO | -- | Untried; could free parameters for more depth |
| **Backout / hyperconnections** | NO (related: U-Net skips) | U-Net skip connections are a simpler version | Full hyperconnections untried |
| **Multi-token prediction** | NO | -- | Untried; could improve sample efficiency |
| **Long-short sliding window attn** | PARTIAL | Sliding window used for eval, not training | Training-time sliding windows untried |
| **Logit softcap** | YES | softcap=30.0 in top entries | Stabilizes output distribution |
| **Orthogonal init** | YES (all top entries) | OrthoInit + muP-scaled output projections | Pairs with Muon; standard practice |
| **NorMuon** | NO | -- | Gradient norm normalization variant untried |

### Techniques Unique to Parameter Golf (Not From Modded-NanoGPT)

These techniques are specific to the size-constrained parameter golf setting:

| Technique | Description | Used In |
|---|---|---|
| **Int6/Int5 quantization** | Per-row quantization to 5-6 bits; primary size reduction | All competitive entries |
| **QAT (STE fake-quant)** | Train with simulated quantization noise | Multiple entries (though one found it was dead-code-eliminated by torch.compile) |
| **GPTQ-lite** | Post-training optimal clip percentile search per row | Top entry only (1.1233) |
| **EMA weight averaging** | Exponential moving average (decay=0.997) before quantization | Top 3 entries |
| **SWA (stochastic weight averaging)** | Average checkpoints during warmdown | Earlier entries (replaced by EMA in top) |
| **SmearGate** | Learned per-dim gate blending current + previous token embeddings | All top entries |
| **Exclusive Self Attention (XSA)** | Subtracts self-value-aligned component from attention output | Top 4 entries (last 3-4 layers) |
| **zstd-22 compression** | High-ratio compression for int6-packed weights | All competitive entries |
| **Partial RoPE** | Apply RoPE to only 16/64 head dims (25%) | Top 2 entries |

---

## Part C: Untried Combinations & Opportunities

### High-Priority Opportunities (likely beneficial, moderate effort)

#### 1. QK-Norm Instead of / In Addition to LN Scale

**Current state**: Top entries use LN Scale (`1/sqrt(layer_idx+1)`) to dampen deep layers.
**Opportunity**: Apply RMSNorm directly to Q and K before attention, as in modded-nanogpt. This is more targeted -- it stabilizes the specific dot-product computation that causes issues, rather than damping the entire layer output. Could be combined with LN Scale or replace it.
**Expected impact**: Small positive (~0.001 BPB). Low risk, easy to implement.

#### 2. Simplified Hyperconnections (Beyond U-Net Skips)

**Current state**: All top entries use U-Net skip connections (encoder-decoder pattern).
**Opportunity**: Replace or augment U-Net skips with full hyperconnections -- learned skip weights between arbitrary layer pairs. The "simplified" variant from modded-nanogpt record #77 uses a single saved activation for efficiency.
**Key concern**: Additional learned skip weights add parameters, but they are small (one scalar per connection). The activation memory overhead is the bigger concern during training, though the 10-minute budget with 8xH100s provides ample room.
**Expected impact**: Moderate positive (~0.002-0.004 BPB). The U-Net pattern is a special case of hyperconnections; generalizing it could find better information routing.

#### 3. Multi-Token Prediction

**Current state**: Not used in any parameter-golf entry.
**Opportunity**: Predict 2-3 future tokens per step using small additional prediction heads. During training only -- the extra heads are discarded at inference time (no size cost). The learning signal improvement could allow reaching lower loss within the same 10-minute budget.
**Key concern**: Extra heads add compute per step, meaning fewer total training steps. Need to verify the per-step quality improvement outweighs the step count reduction.
**Expected impact**: Potentially significant (~0.003-0.005 BPB) if the trade-off is favorable.

#### 4. Training-Time Sliding Window Attention (Long-Short Pattern)

**Current state**: Sliding window is used only at evaluation time (stride=64). During training, all layers use full attention over the 2048-token sequence.
**Opportunity**: Apply the Gemma-2-style long-short pattern: early layers use restricted local windows (e.g., 256-512 tokens) while later layers use full context. This reduces attention compute in early layers, potentially allowing more training steps or longer sequences.
**Key concern**: Restricting early-layer context during training could hurt quality. Needs careful tuning of which layers get which window sizes.
**Expected impact**: Uncertain. Could be neutral if the early layers don't need full context, or slightly negative if they do.

### Medium-Priority Opportunities (interesting but uncertain)

#### 5. Paired Head Attention for Parameter Efficiency

**Current state**: 8 heads with 4 KV heads (GQA). All heads use full 64-dim.
**Opportunity**: Pair heads so that some operate at reduced dimensionality (e.g., 32-dim), freeing parameters for additional heads or layers. In modded-nanogpt this saves FLOPs; in parameter golf, the saving is bytes.
**Key concern**: The parameter savings per head pair are modest (~16K params). May not move the needle unless enabling an extra layer.
**Expected impact**: Small positive if it enables architectural changes; neutral otherwise.

#### 6. NorMuon (Normalized Gradient Muon)

**Current state**: Standard Muon with momentum warmup and weight decay.
**Opportunity**: Add gradient norm normalization to Muon, as introduced in modded-nanogpt record #41. This could improve training stability, especially at higher learning rates.
**Expected impact**: Small positive (~0.001 BPB). Low risk, easy to try.

#### 7. Asymmetric Logit Rescaling

**Current state**: Logit softcap at 30.0.
**Opportunity**: Apply different scaling factors to positive vs. negative logits before the softmax. This can stabilize the cross-entropy loss computation and improve training dynamics.
**Expected impact**: Small positive. Low effort to implement.

### Speculative Combinations

These combine multiple untried techniques. Higher risk, higher potential reward.

#### Combo A: Hyperconnections + Multi-Token Prediction
Replace U-Net skips with hyperconnections AND add 2-token prediction. The hyperconnections let the model route information more flexibly, while multi-token prediction provides richer gradients. Both techniques are zero-cost at inference (hyperconnection weights are tiny; prediction heads are discarded).

#### Combo B: QK-Norm + Partial RoPE + Paired Heads
QK-Norm stabilizes attention; Partial RoPE (already used) frees head dimensions from positional encoding; paired heads use those freed dimensions more efficiently. This is a coherent attention-focused improvement stack.

#### Combo C: Int5 Everywhere + Extra Layer + Hyperconnections
The top entry uses int6 for attention and int5 exploration exists for MLP. Push int5 more aggressively (accept slightly worse per-layer quality) to fund a 12th layer, connected via hyperconnections for better information flow. Use GPTQ-lite clip search to minimize the int5 quality hit.

### Summary: Where to Focus Next

| Priority | Technique | Effort | Risk | Expected BPB Gain |
|---|---|---|---|---|
| 1 | QK-Norm | Low | Low | ~0.001 |
| 2 | Hyperconnections | Medium | Medium | ~0.002-0.004 |
| 3 | Multi-token prediction | Medium | Medium | ~0.003-0.005 |
| 4 | NorMuon | Low | Low | ~0.001 |
| 5 | Training-time sliding windows | Medium | High | Uncertain |
| 6 | Paired head attention | Medium | Medium | ~0.001 |
| 7 | Asymmetric logit rescaling | Low | Low | ~0.0005 |

The lowest-hanging fruit is **QK-Norm** (easy to add, well-validated in modded-nanogpt). The highest-upside opportunity is **multi-token prediction** (significant convergence improvement if it works within the compute budget). **Hyperconnections** represent the most architecturally interesting opportunity, generalizing the U-Net skip pattern that is already proven effective.

---

## Appendix: Leaderboard Snapshot (as of 2026-03-24)

| Date | Entry | val_bpb | Key Techniques |
|---|---|---|---|
| 03-22 | 11L EMA + GPTQ-lite + warmdown3500 | **1.1233** | GPTQ-lite, EMA, late QAT, XSA4, Partial RoPE, value emb |
| 03-21 | 11L XSA4 + EMA + Partial RoPE | 1.1248 | Partial RoPE, LN Scale, XSA4, EMA |
| 03-20 | 11L XSA4 + EMA + Int6 MLP3x | 1.1271 | XSA4, EMA (replacing SWA) |
| 03-20 | 11L Efficient Partial XSA + FA3 | 1.1307 | Efficient GQA-aware XSA, FA3 |
| 03-20 | Int6 MLP3x + SmearGate + BigramHash | 1.1458 | Full technique stack, 9 layers |
| 03-20 | 10L Int5-MLP + BigramHash(10240) | 1.1428 | Int5 for MLP, larger bigram hash |
| 03-19 | SmearGate + OrthoInit + Muon WD | 1.1556 | First SmearGate/BigramHash/OrthoInit entry |
| 03-19 | 11L MLP3x + QAT + Int6 + Sliding | 1.1502 | 11 layers, QAT, zstd-22 |
| 03-17 | Naive Baseline | 1.2244 | 9L, int8+zlib, no tricks |

**Total progression**: 1.2244 -> 1.1233 = **0.1011 BPB improvement** in 5 days.

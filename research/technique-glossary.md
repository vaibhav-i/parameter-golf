# Parameter Golf Technique Glossary

A comprehensive reference for every major technique used in the OpenAI Parameter Golf challenge, written for a beginner-intermediate ML audience. Each entry explains what the technique is, why it matters in this specific competition, how it is implemented, what you trade off, and which leaderboard entries used it.

**Competition context:** Train the best language model that fits in a 16MB artifact (code + compressed model) in under 10 minutes on 8xH100 GPUs. The metric is bits-per-byte (BPB) on the FineWeb validation set. Lower is better. The current SOTA is **1.1147 BPB** (abaybektursun, 2026-03-25).

---

## Table of Contents

1. [Quantization](#quantization)
   - [Int8 Quantization (Baseline)](#int8-quantization-baseline)
   - [Int6 Quantization](#int6-quantization)
   - [Int5 Quantization](#int5-quantization)
   - [Mixed-Precision Quantization](#mixed-precision-quantization)
   - [QAT with STE (Quantization-Aware Training)](#qat-with-ste-quantization-aware-training)
   - [Late QAT vs Early QAT](#late-qat-vs-early-qat)
   - [GPTQ-lite (Clip Percentile Search)](#gptq-lite-clip-percentile-search)
   - [Full Hessian GPTQ + AR Self-Gen Calibration](#full-hessian-gptq--ar-self-gen-calibration)
   - [Selective ±1 Pruning](#selective-1-pruning)
   - [Compression: zstd-22 vs zlib vs LZMA](#compression-zstd-22-vs-zlib-vs-lzma)
2. [Weight Averaging](#weight-averaging)
   - [EMA (Exponential Moving Average)](#ema-exponential-moving-average)
   - [SWA (Stochastic Weight Averaging)](#swa-stochastic-weight-averaging)
   - [Interaction with Quantization](#interaction-with-quantization)
3. [Attention Innovations](#attention-innovations)
   - [XSA (Exclusive Self Attention)](#xsa-exclusive-self-attention)
   - [Partial XSA](#partial-xsa)
   - [Partial RoPE](#partial-rope)
   - [GQA (Grouped Query Attention)](#gqa-grouped-query-attention)
   - [Flash Attention 3](#flash-attention-3)
4. [Embeddings](#embeddings)
   - [BigramHash Embeddings](#bigramhash-embeddings)
   - [SmearGate](#smeargate)
   - [Value Embeddings](#value-embeddings)
   - [Tied Embeddings](#tied-embeddings)
   - [FP16 Embeddings](#fp16-embeddings)
5. [Architecture](#architecture)
   - [U-Net Skip Connections](#u-net-skip-connections)
   - [MLP Expansion Ratio](#mlp-expansion-ratio)
   - [LeakyReLU(0.5)^2 Activation](#leakyrelu052-activation)
   - [Layer Count Tradeoffs](#layer-count-tradeoffs)
   - [Overtone/Spectral Initialization](#overtonespectral-initialization)
   - [Orthogonal Initialization](#orthogonal-initialization)
   - [QK-Norm](#qk-norm)
   - [LayerNorm Scale Factors](#layernorm-scale-factors)
6. [Optimizers](#optimizers)
   - [Muon Optimizer](#muon-optimizer)
   - [Parallel Muon + Parameter Banking](#parallel-muon--parameter-banking)
   - [Muon Momentum Warmup](#muon-momentum-warmup)
   - [Muon Weight Decay](#muon-weight-decay)
   - [AdamW for Embeddings](#adamw-for-embeddings)
   - [Learning Rate Schedules](#learning-rate-schedules)
   - [Warmdown Timing Effects](#warmdown-timing-effects)
7. [Evaluation Tricks](#evaluation-tricks)
   - [Sliding Window Evaluation](#sliding-window-evaluation)
   - [Stride-16 vs Stride-64](#stride-16-vs-stride-64)
   - [Temperature Scaling](#temperature-scaling)
   - [Sequence Length and Eval](#sequence-length-and-eval)
8. [Test-Time Training](#test-time-training)
   - [LoRA TTT](#lora-ttt)
   - [Full-Model SGD TTT (Score-First)](#full-model-sgd-ttt-score-first)
9. [Confirmed Dead Ends](#confirmed-dead-ends)
   - [Depth Recurrence](#depth-recurrence)

---

## Quantization

Quantization is the single most important family of techniques in parameter golf. The challenge constrains your artifact to 16MB, which means a model with ~26M float32 parameters (~104MB raw) must be compressed by roughly 6-7x. Quantization reduces each weight from 32 or 16 bits down to 8, 6, or even 5 bits, directly shrinking model size.

### Int8 Quantization (Baseline)

**What it is.** Each weight is mapped from its floating-point value to the nearest integer in the range [-127, 127]. This is done per-row: for each row of a weight matrix, you find the maximum absolute value, compute a scale factor `scale = max_abs / 127`, then quantize as `q = round(weight / scale)`. To reconstruct: `weight_approx = q * scale`. The scale factors (one float16 per row) are stored alongside the int8 weights.

**Why it helps.** Int8 halves the storage compared to float16 (1 byte per weight instead of 2). For the baseline 9-layer, 512-dim model with ~17M params, this brings the model from ~34MB (fp16) down to ~17MB, plus scales. With zlib compression, the artifact fits under 16MB.

**Tradeoffs.** The baseline int8 scheme introduces a "quantization gap" -- the difference in BPB between the floating-point model and the quantized model. For the naive baseline, this gap was ~0.007 BPB. That sounds small, but in this competition where records are separated by 0.005 BPB, it is significant.

**Who used it.** The Naive Baseline (1.2244 BPB) uses uniform int8 for all weights including embeddings.

### Int6 Quantization

**What it is.** Instead of mapping to [-127, 127] (256 levels), weights are mapped to [-31, 31] (63 levels) or [-32, 31] (64 levels). The quantized values are still stored in int8 containers (since there is no native 6-bit integer type), but the upper 2 bits are always zero. This means compression algorithms like zstd can efficiently compress away the unused bit patterns.

**Why it helps.** Int6 packs weights into fewer effective bits, so after compression the file is significantly smaller. This frees up bytes in the 16MB budget that can be "spent" on a larger model -- for example, going from a 2x MLP expansion to 3x, or adding extra transformer layers. The key insight is that the marginal model quality from a wider MLP or deeper network more than compensates for the precision loss from int6 vs int8.

**Implementation.** Per-row quantization:
```python
row_max = weight_row.abs().max()
scale = (row_max / 31.0).clamp_min(1.0 / 31.0)  # fp16
q = round(weight_row / scale).clamp(-32, 31)      # int8 container
```
The scale is stored as float16. Reconstruction: `weight_approx = q.float() * scale.float()`.

**Tradeoffs.** Compared to int8, int6 has a larger quantization error per weight. Without QAT, int6 can add ~0.016 BPB of quantization penalty. With QAT (see below), this penalty shrinks to near-zero. The byte savings from int6 typically fund enough extra model capacity (wider MLP, more layers) to more than offset the precision loss.

**Who used it.** Nearly all entries from 2026-03-19 onward use int6 for transformer block weights. The SmearGate+OrthoInit entry (1.1556), MLP3x+QAT (1.1502), and all subsequent records use int6 on MLP and attention weights.

### Int5 Quantization

**What it is.** Weights mapped to [-16, 15] (32 levels). Even more aggressive than int6 -- each weight has only 5 bits of precision. Stored in int8 containers where the upper 3 bits are zero, yielding excellent compression ratios with zstd.

**Why it helps.** Int5 achieves a ~1.88x compression ratio with zstd (compared to ~1.51x for int6). This saves an additional ~1.86MB versus uniform int6, which is enough to fund an entire additional transformer layer. The trade-off works when applied selectively to the least precision-sensitive weights.

**Implementation.** Same pattern as int6 but with `clip_range = 15`:
```python
scale = (row_max / 15.0).clamp_min(1.0 / 15.0)
q = round(weight / scale).clamp(-16, 15)
```

**Tradeoffs.** Int5 is too aggressive for attention weights (which are precision-sensitive), but works well for MLP weights which tend to be more compressible. Must be combined with mixed-precision quantization (see below).

**Who used it.** The 10L Int5-MLP entry (1.1428 BPB, thwu1) uses int5 for MLP weights and int6 for attention weights, funding a 10th layer with the byte savings.

### Mixed-Precision Quantization

**What it is.** Different weight matrices get different quantization bit-widths based on their sensitivity to precision loss. Rather than applying a uniform scheme across the entire model, you assign int5, int6, int8, or fp16 to each tensor based on how much quality it loses from quantization.

**Why it helps.** Not all weights are equally sensitive. The tied embedding matrix (which serves double duty as both input embeddings and output head) is extremely sensitive -- quantization errors affect both directions. Attention key projections in the final layers are also sensitive. MLP weights, by contrast, are relatively robust to quantization. By allocating precision where it matters most, you get better overall quality for the same byte budget.

**Typical allocation in top entries:**
| Component | Precision | Rationale |
|-----------|-----------|-----------|
| MLP weights | Int5 or Int6 | Least sensitive, most compressible |
| Attention weights (Q, K, V, proj) | Int6 | Moderately sensitive |
| Last-layer key projection | FP16 | Very sensitive to late-layer attention |
| Tied embeddings | FP16 or Int8 | Extremely sensitive (dual role) |
| Control tensors (scales, norms, gates) | FP32 | Tiny; precision matters |

**Tradeoffs.** More complex export pipeline. You need to carefully categorize each tensor and test the quality impact. The first mixed-precision entry reduced the quantization penalty from +0.048 BPB to +0.0015 BPB -- a 32x improvement.

**Who used it.** The Mixed Quant entry (1.1630, aquariouseworkman) introduced int6 blocks + int8 embeddings. The 10L Mixed Precision entry (1.2147, Nan Liu) used int6 for middle layers and int8 for early/late layers. All subsequent top entries use mixed schemes.

### QAT with STE (Quantization-Aware Training)

**What it is.** QAT (Quantization-Aware Training) simulates quantization during training so the model learns weight configurations that survive quantization well. The key mechanism is the Straight-Through Estimator (STE): during the forward pass, weights are fake-quantized (quantize then dequantize), so the model "sees" quantized weights. During the backward pass, gradients flow through the quantization operation as if it were the identity function -- they are passed "straight through."

**Implementation (from CastedLinear):**
```python
# Forward pass: fake quantize
w32 = self.weight.float()
row_max = w32.abs().amax(dim=1)
scale = (row_max / 31.0).clamp_min(1.0 / 31.0)
w_q = (torch.clamp(torch.round(w32 / scale[:, None]), -32, 31) * scale[:, None])

# STE trick: add the quantization error as a detached tensor
# Forward uses quantized weights; backward uses original gradients
w = w + (w_q - w).detach()
```

The expression `w + (w_q - w).detach()` is the STE trick: in the forward pass, `w_q - w` cancels with `w`, yielding `w_q`. But `.detach()` means the gradient of `w_q - w` is zero, so the backward pass only sees gradients through the original `w`.

**Why it helps.** Without QAT, there is a "quantization gap" between the pre-quantization and post-quantization model quality. QAT can eliminate this gap entirely. The Seq2048 entry reported a quant gap of exactly 0.0000 BPB with STE QAT.

**Tradeoffs.** QAT adds ~28% overhead per training step (extra quantize/dequantize in every forward pass). With a 10-minute wall-clock budget, this means fewer total training steps. The net benefit depends on whether the eliminated quantization gap outweighs the lost training steps. For int6, where the quant gap can be 0.016 BPB, QAT is usually worth it. The tied embedding is typically excluded from QAT (kept in fp16) because it never sees fake quantization during training.

**Who used it.** The SmearGate+OrthoInit entry (1.1556), MLP3x+QAT entry (1.1502), and all top entries incorporate some form of QAT.

### Late QAT vs Early QAT

**What it is.** "Early QAT" means fake-quantizing from the start of training. "Late QAT" means enabling fake-quantization only in the final portion of training, when the learning rate has decayed significantly.

**Why Late QAT.** During early training when learning rates are high, fake-quantization noise can interfere with learning -- the model is trying to learn large-scale weight configurations, and the quantization noise adds unhelpful perturbation. Late QAT activates only when the LR scale drops below a threshold (e.g., 0.15 or 0.10), so the model has already converged to a good region of weight space before QAT fine-tunes it for quantization robustness.

**Implementation:**
```python
if scale < args.late_qat_threshold and not CastedLinear._qat_enabled:
    CastedLinear._qat_enabled = True
```

**Tradeoffs.** Late QAT preserves full training quality during the majority of training, only introducing quantization noise at the end. However, if the model is compiled with `torch.compile`, the `_qat_enabled` flag may be constant-folded at trace time, making the late activation a no-op (as discovered in the PR #287 entry). This is a practical pitfall to watch for.

**Who used it.** The GPTQ-lite entry (1.1233, signalrush) uses late QAT at threshold 0.15. The Partial RoPE entry (1.1248, jfprincz) intended to use late QAT at 0.1 but discovered `torch.compile` eliminated the branch.

### GPTQ-lite (Clip Percentile Search)

**What it is.** Instead of using the row maximum as the quantization scale (which is sensitive to outliers), GPTQ-lite tries multiple clip percentiles and picks the one that minimizes reconstruction error. For each row of a weight matrix, it evaluates 5 candidate clip values (99.90th, 99.95th, 99.99th, 99.999th percentile, and the full max) and selects the one with the lowest MSE between the original and reconstructed weights.

**Implementation:**
```python
for pct in [0.9990, 0.9995, 0.9999, 0.99999, 1.0]:
    if pct < 1.0:
        row_clip = torch.quantile(t32.abs(), pct, dim=1)
    else:
        row_clip = t32.abs().amax(dim=1)
    scale = (row_clip / clip_range).clamp_min(1.0 / clip_range).to(torch.float16)
    q = torch.clamp(torch.round(t32 / scale[:, None]), -clip_range, clip_range).to(torch.int8)
    recon = q.float() * scale.float()[:, None]
    err = (t32 - recon).pow(2).mean().item()
    if err < best_err:
        best_q, best_s, best_err = q, s, err
```

**Why it helps.** Weight outliers inflate the quantization scale, wasting precision on the rare extreme values while increasing error for the majority of weights. By clipping at a lower percentile, you accept small clipping error on outliers but dramatically reduce rounding error for the bulk of the distribution. This is a zero-cost post-training optimization -- it adds no training overhead.

**Tradeoffs.** The search is deterministic and fast (5 candidates per row). It adds a few seconds to the export step. The improvement is modest (~0.0006 BPB) but free.

**Who used it.** The signalrush entry (1.1228) introduced GPTQ-lite. Superseded by Full Hessian GPTQ in the current SOTA (1.1147).

### Full Hessian GPTQ + AR Self-Gen Calibration

**What it is.** Full Hessian GPTQ is a strictly better quantizer than GPTQ-lite. Instead of just searching over clip percentiles (diagonal approximation), it uses the full Hessian matrix H = X^T X to understand weight correlations, then applies Cholesky error compensation with column reordering to minimize total reconstruction error.

The "AR self-gen" part solves the calibration data problem: previous Full GPTQ implementations used training data for calibration, which was ruled illegal after the 600s training window. The solution is to have the model autoregressively generate its own calibration data (64 sequences x 2048 tokens, temperature=0.8, fixed seed). No validation data and no training data are accessed during quantization.

**How it works.**
1. After training, the model generates 64 sequences of 2048 tokens using temperature sampling
2. Forward hooks collect H = X^T X (input correlations) for each linear layer
3. The Hessian is damped: H += 0.01 * diag(H).mean() * I
4. GPTQ quantizes column-by-column, using the Hessian to compensate: after quantizing column j, the error is distributed to remaining columns proportional to their correlation with column j via Cholesky decomposition

```python
# Simplified GPTQ core loop
for j in range(cols):
    q_j = round_and_clip(W[:, j], scale)
    err = (W[:, j] - dequant(q_j)) / H_inv[j, j]
    W[:, j+1:] -= err.outer(H_inv[j, j+1:])  # distribute error
```

**Impact.** The GPTQ-lite to Full Hessian GPTQ upgrade is the single biggest change in the new SOTA (1.1147), contributing roughly -0.003 to -0.005 BPB. The AR self-gen approach is what makes it legal.

**Tradeoffs.** Adds ~30-60s to the export step for Hessian collection and GPTQ solve. Requires generating calibration sequences (a few seconds). Code complexity is significant. The Hessian matrices are collected on CPU to avoid GPU memory pressure.

**Who used it.** Current SOTA (1.1147, abaybektursun, PR #1019). First introduced with training-data calibration in PRs #535/#569/#593/#609, made legal via AR self-gen in this entry.

### Selective ±1 Pruning

**What it is.** After int6 quantization, some quantized values are ±1 (one step away from zero). Selective pruning sets these to 0 if the reconstruction error improves. This creates more zeros in the weight matrix, which compresses better with LZMA/zstd.

**Why it helps.** Weights at ±1 have small magnitude -- they contribute little to the output but cost entropy in compression. Setting them to zero when it reduces reconstruction error gives a dual benefit: slightly better model quality AND smaller compressed artifact.

**Tradeoffs.** Minimal. It's a post-training optimization that takes seconds. The pruning is conservative (only prunes when it reduces error), so quality never decreases.

**Who used it.** Current SOTA (1.1147), inherited from PR #609 by @saml212.

### Compression: zstd-22 vs zlib vs LZMA

**What it is.** After quantization, the weight tensors are serialized and compressed with a lossless compression algorithm. Three options have been explored: zlib (Python built-in, level 9), zstandard (zstd, level 22), and LZMA (preset 9).

**Why it helps.** Better compression means more parameters fit in the 16MB budget. For a model on the edge of the size limit, the compression choice can determine whether an extra layer fits.

**Compression comparison on int6 quantized weights:**
| Algorithm | Relative Size | Speed | Package |
|-----------|--------------|-------|---------|
| zlib-9 | Baseline | Fast | Built-in |
| zstd-22 | ~5% smaller | Medium | `zstandard` |
| LZMA-9 | ~8-10% smaller | Slow | Built-in (`lzma`) |

Note: earlier analysis suggested LZMA was worse on int8 data, but the current SOTA (1.1147) uses LZMA-9 on int6 data and achieves better compression than zstd-22. The effectiveness depends on the quantization format and weight distribution.

**Who used it.** Entries from MLP3x+QAT (1.1502) through signalrush (1.1228) used zstd-22. The current SOTA (1.1147) switched to LZMA-9. The ternary/binary entries also use LZMA.

---

## Weight Averaging

Weight averaging techniques produce smoother weight distributions that generalize better and quantize with less error. Two main variants are used in parameter golf.

### EMA (Exponential Moving Average)

**What it is.** A shadow copy of the model weights is maintained throughout training. After each optimizer step, the shadow weights are updated as a weighted average of the old shadow and the current weights:

```python
ema_state[name] = decay * ema_state[name] + (1 - decay) * current_weight
```

With `decay = 0.997`, each EMA update blends 99.7% of the old average with 0.3% of the new weights. At the end of training, the EMA weights replace the current weights for quantization and evaluation.

**Why it helps.** Training produces noisy weight trajectories -- the optimizer bounces around in weight space, especially with the high learning rates used in this challenge. EMA smooths out this noise, producing weights that are closer to the "center" of recent good weight configurations. Smoother weights have tighter distributions (fewer outliers), which directly benefits quantization quality and compression ratio.

**Tradeoffs.** EMA requires storing a full copy of the model in float32 on GPU, roughly doubling memory usage. With ~26M parameters, this is ~100MB of extra GPU memory -- manageable on 80GB H100s. The per-step update cost is negligible. EMA provides smoother averaging than SWA (below) because it updates every step rather than at discrete checkpoints.

**Who used it.** Introduced in the XSA4+EMA entry (1.1271, jfprincz) to replace SWA, and used in all subsequent top entries including the current SOTA (1.1228/1.1233).

### SWA (Stochastic Weight Averaging)

**What it is.** During the warmdown phase of training, periodic snapshots of the model weights are saved. At the end of training, all snapshots are uniformly averaged to produce the final weights. "Tight SWA" refers to collecting snapshots frequently (e.g., every 50 steps) during the late warmdown phase.

**Implementation:**
```python
# Collect checkpoint when LR scale < threshold
if swa_enabled and scale < 0.2 and step % swa_every == 0:
    if swa_state is None:
        swa_state = {name: t.clone() for name, t in model.state_dict().items()}
        swa_count = 1
    else:
        for name, t in model.state_dict().items():
            swa_state[name] += t
        swa_count += 1

# At end: swa_state[name] /= swa_count
```

Variants include adjusting `swa_start_frac` (what fraction of warmdown to start collecting) and `swa_every` (collection frequency). The 10L Int5-MLP entry found `start_frac=0.4` optimal -- collecting only from the last 40% of warmdown, when weights are most converged.

**Why it helps.** Like EMA, SWA smooths the weight trajectory. The uniform average of multiple checkpoints tends to find a flatter minimum in the loss landscape, which generalizes better. SWA is particularly effective when combined with a warmdown schedule, since the declining learning rate causes later checkpoints to cluster tightly.

**Tradeoffs.** SWA stores checkpoints on CPU (unlike EMA which is on GPU), so memory is less of a concern. However, SWA is a coarser approximation than EMA -- it only captures snapshots at discrete intervals, missing the continuous trajectory between them. In head-to-head comparisons, EMA (decay=0.997, every step) outperformed SWA by ~0.005 BPB.

**Who used it.** The SmearGate+BigramHash entry (1.1458, Raahil Shah) used SWA every 50 steps over the last 50% of training. The Efficient Partial XSA entry (1.1307, unnir) used SWA every 120 steps with 13 checkpoints. The GPTQ-lite SOTA entry (1.1233) uses both EMA and Tight SWA together.

### Interaction with Quantization

Weight averaging and quantization interact in a virtuous cycle. Averaged weights have tighter distributions (lower variance, fewer outliers), which means:

1. **Lower quantization error:** The gap between the floating-point value and its nearest quantization level is smaller on average.
2. **Better compression:** Tighter distributions map to fewer distinct quantized values, producing more compressible byte patterns for zstd.
3. **Less need for clipping:** With fewer outliers, the quantization scale is closer to the typical weight magnitude, allocating more precision to the values that matter.

The current SOTA stacks both EMA and SWA: EMA provides continuous smoothing during all of training, while Tight SWA captures discrete checkpoints during the final warmdown. After training, EMA weights are used as the base, providing the best of both approaches.

---

## Attention Innovations

### XSA (Exclusive Self Attention)

**What it is.** Exclusive Self Attention (from arXiv:2603.09078) modifies the attention output by subtracting the component aligned with each token's own value vector. In standard attention, the output for each token is a weighted sum of all value vectors, including the token's own value. XSA removes this "self" component, forcing the attention mechanism to only capture information from other tokens.

Mathematically, after computing the standard attention output `y`, XSA projects out the self-value component:
```
y_xsa = y - (y . v_hat) * v_hat
```
where `v_hat` is the unit-normalized value vector for the current token.

**Implementation (efficient GQA-aware version):**
```python
def _xsa_efficient(self, y, v):
    # y: [B, T, H, D], v: [B, T, Hkv, D]
    B, T, H, D = y.shape
    Hkv = v.size(-2)
    group = H // Hkv
    y_g = y.reshape(B, T, Hkv, group, D)       # free reshape, no copy
    vn = F.normalize(v, dim=-1).unsqueeze(-2)    # [B, T, Hkv, 1, D]
    proj = (y_g * vn).sum(dim=-1, keepdim=True) * vn
    return (y_g - proj).reshape(B, T, H, D)
```

**Why it helps.** The XSA paper shows that in standard transformers, deeper layers exhibit increasing "self-attention bias" -- attention outputs become increasingly aligned with the token's own value. This is redundant, since the residual stream already carries the token's own information. By removing this component, XSA encourages attention to capture genuinely new contextual information, improving the model's use of context.

**Tradeoffs.** Adds ~2ms per step with the efficient implementation (down from ~7ms with naive repeat_interleave). Zero new parameters. The improvement is ~0.002 BPB when applied to the deepest layers.

**Who used it.** Introduced in the Efficient Partial XSA entry (1.1307, unnir) on last 3 layers. Extended to last 4 layers in the XSA4+EMA entry (1.1271, jfprincz). **The current SOTA (1.1147) applies XSA to ALL 11 layers** (`XSA_LAST_N=11`), after @gowtham0992 (PR #478) showed this forces cross-position information mixing from layer 0 at zero parameter cost.

### Partial XSA

**What it is.** Rather than applying XSA to every layer, it is only applied to the deepest N layers. Early entries used last 3-4 layers.

**UPDATE (2026-03-25):** The current SOTA (1.1147) uses **XSA on ALL 11 layers** (`XSA_LAST_N=11`). The previous assumption that shallow layers have lower self-attention bias and don't benefit from XSA was overturned. XSA-all is now the standard.

**Note on looped architectures:** The depth recurrence paper (PR #363) found that XSA-all was slightly harmful on looped models (+0.001 worse), because "all layers" in a looped model means only 3 unique computations repeated 3 times, unlike 11 unique computations in a flat model.

**Who used it.** Evolution: last 3 (1.1307) -> last 4 (1.1271) -> all 11 (1.1147, current SOTA).

### Partial RoPE

**What it is.** Rotary Position Embeddings (RoPE) encode position information by rotating query and key vectors in attention. Standard RoPE applies rotation to all head dimensions. Partial RoPE applies rotation to only a subset -- for example, 16 out of 64 dimensions (25%). The remaining dimensions attend without any positional bias.

**Implementation:**
```python
def apply_rotary_emb(x, cos, sin, rope_dims=0):
    if rope_dims > 0 and rope_dims < x.size(-1):
        x_rope, x_pass = x[..., :rope_dims], x[..., rope_dims:]
        # Apply rotation only to x_rope
        half = rope_dims // 2
        x1, x2 = x_rope[..., :half], x_rope[..., half:]
        x_rope = torch.cat((x1*cos + x2*sin, x1*(-sin) + x2*cos), dim=-1)
        return torch.cat((x_rope, x_pass), dim=-1)  # x_pass is unchanged
```

**Why it helps.** By leaving most head dimensions position-free, the model can learn position-invariant patterns (like "the word after 'the' is usually a noun") in those dimensions, while still using the RoPE dimensions for position-dependent patterns. This division of labor improves representational efficiency. The improvement is ~0.002 BPB with zero new parameters.

**Tradeoffs.** Reduces the model's positional resolution (fewer dimensions encoding position). For the 1024-2048 sequence lengths used in this challenge, 16 dimensions provide sufficient positional information. For much longer sequences this might be insufficient.

**Who used it.** The Partial RoPE entry (1.1248, jfprincz) introduced `ROPE_DIMS=16` (16 of 64), carried forward to the SOTA.

### GQA (Grouped Query Attention)

**What it is.** Grouped Query Attention uses fewer key-value (KV) heads than query heads. In the challenge models, there are 8 query heads but only 4 KV heads. Each KV head serves a group of 2 query heads. This means K and V projections are half the size of the Q projection.

**Why it helps.** GQA reduces parameter count (smaller K and V weight matrices) and KV cache size. In parameter golf, the smaller weight matrices free up bytes in the 16MB budget. The quality loss from shared KV heads is minimal for these model sizes.

**Implementation.** The query, key, and value projections have different output dimensions:
```python
self.c_q = CastedLinear(dim, dim, bias=False)         # 512 -> 512
self.c_k = CastedLinear(dim, kv_dim, bias=False)      # 512 -> 256
self.c_v = CastedLinear(dim, kv_dim, bias=False)      # 512 -> 256
```
where `kv_dim = num_kv_heads * head_dim = 4 * 64 = 256`.

**Who used it.** All entries use `NUM_HEADS=8, NUM_KV_HEADS=4` (GQA with group size 2).

### Flash Attention 3

**What it is.** Flash Attention is a hardware-aware algorithm for computing attention that avoids materializing the full N x N attention matrix in GPU memory. Flash Attention 3 is the Hopper-architecture-optimized version, using H100-specific hardware features (TMA, warp-group pipelining) for maximum throughput.

**Why it helps.** Faster attention computation means more training steps within the 10-minute budget. Flash Attention 3 is called directly via `flash_attn_func` from the `flash_attn_interface` package, bypassing PyTorch's slower default attention implementations.

**Implementation:**
```python
from flash_attn_interface import flash_attn_func as flash_attn_3_func
# In attention forward:
y = flash_attn_3_func(q, k, v, causal=True)
```

**Tradeoffs.** Requires the flash-attn package compiled for Hopper. Not available on older GPU architectures. The function signature expects specific tensor layouts (`[B, T, H, D]`).

**Who used it.** All entries from the Efficient Partial XSA entry (1.1307) onward use Flash Attention 3.

---

## Embeddings

### BigramHash Embeddings

**What it is.** A hash table that maps pairs of consecutive tokens to learned embeddings. Given the previous token `prev` and current token `cur`, a hash function maps the pair to one of N buckets:

```python
bucket = (36313 * cur ^ 27191 * prev) % (bigram_vocab_size - 1)
```

Each bucket has a learned embedding vector (dim=128), which is projected to the model dimension (512) via a linear layer and added to the token embedding.

**Why it helps.** Language is highly dependent on token pairs. In English, knowing the current token is "at" and the previous was "the" gives much more information than "at" alone. BigramHash provides this signal at the embedding layer, before the transformer even starts processing. This gives the model a head start on bigram-dependent predictions, which are very common.

The hash-based approach is parameter-efficient: with 2048 buckets at dim=128, it costs only ~262K parameters (plus the 128->512 projection). Even with hash collisions, the signal is valuable because common bigrams dominate.

**Tradeoffs.** Hash collisions mean some bigram pairs share embeddings. Increasing the bucket count reduces collisions but costs more parameters. The 10L Int5-MLP entry found that going from 4096 to 10240 buckets improved BPB by 0.002, showing that collision reduction matters.

**Who used it.** Introduced in the SmearGate+OrthoInit entry (1.1556) with 4096 buckets. Scaled to 10240 buckets in the 10L Int5-MLP entry (1.1428). The top entries use 2048 buckets (a balance between size and quality).

### SmearGate

**What it is.** A learned per-dimension gate that blends each token's embedding with the previous token's embedding:

```python
gate = sigmoid(self.gate)  # shape [dim], ~512 params
output = (1 - gate) * current_emb + gate * prev_emb
```

The gate is initialized at `sigmoid(0) = 0.5` (or sometimes biased toward 0.95 to start near-identity), and the model learns per-dimension how much previous-token information to mix in.

**Why it helps.** SmearGate provides lightweight bigram context at essentially zero parameter cost (~512 parameters for a 512-dim model). It complements BigramHash: SmearGate provides a multiplicative blending of the full embedding vectors, while BigramHash provides an additive signal from a hash-based lookup. Together they give the model rich bigram information before the first attention layer.

**Tradeoffs.** The blending is purely local (only the immediately preceding token). It adds negligible compute. The gate initialization matters -- starting too close to 0 or 1 can slow convergence.

**Who used it.** Introduced in the SmearGate+OrthoInit entry (1.1556). Used in all subsequent top entries.

### Value Embeddings

**What it is.** Token identity is reinjected into the attention value vectors at specific deep layers. A shared embedding table maps each token to a low-dimensional vector (dim=128), projected to the KV dimension, and added to the value output at designated layers (e.g., layers 9 and 10). Each target layer has its own learned scale factor.

```python
# In attention forward:
v = self.c_v(x)
if v_embed is not None:
    v = v + v_embed  # add value embedding
v = v.reshape(bsz, seqlen, num_kv_heads, head_dim)
```

**Why it helps.** In deep layers, the residual stream representations have become abstract and may have "forgotten" the original token identity. Value embeddings reintroduce this information, helping the model make better predictions about specific tokens. This is particularly useful for the output layers that need to produce precise token-level predictions.

**Tradeoffs.** Adds ~131K parameters (1024 vocab x 128 dim embedding, plus projection and scale params). The shared table with per-layer scales is more parameter-efficient than having separate tables per layer.

**Who used it.** The GPTQ-lite SOTA entry (1.1233) uses value embeddings on layers 9 and 10 with `ve_dim=128`.

### Tied Embeddings

**What it is.** The input embedding matrix and the output projection head share the same weight matrix. Instead of having a separate `lm_head` linear layer that maps from model_dim to vocab_size, the input embedding matrix `tok_emb.weight` is reused as the output projection via `F.linear(hidden, tok_emb.weight)`.

**Why it helps.** Tying saves an entire `[vocab_size x model_dim]` matrix worth of parameters. For vocab_size=1024 and model_dim=512, that is 524K parameters saved (~1MB in fp16). In a 16MB budget, this is significant.

**Tradeoffs.** The embedding must be good for both roles: input lookup (where you want distinctive, well-separated token representations) and output projection (where you want the hidden state to clearly point toward the correct token). These objectives can conflict, but in practice with a small vocabulary (1024), tying works well.

**Who used it.** All entries use tied embeddings. It is the default configuration (`TIE_EMBEDDINGS=1`).

### FP16 Embeddings

**What it is.** During model export and quantization, the tied embedding matrix is kept in its full fp16 precision instead of being quantized to int8 or int6.

**Why it helps.** The tied embedding is the most quantization-sensitive tensor in the model because errors propagate in both directions: degraded input representations feed through all layers, and degraded output projection weights affect every token's prediction. The FP16 embed entry found that keeping embeddings in fp16 reduced the quantization penalty from ~0.007 BPB to ~0.0005 BPB. The cost is ~500KB extra in the artifact.

**Implementation.** During the export step, the embedding weight is classified as a "passthrough" tensor that skips quantization:
```python
if "tok_emb" in name:
    result[name] = t.to(torch.float16)  # skip int8 quantization
```

**Tradeoffs.** ~500KB extra artifact size. This is offset by reducing MLP hidden dimension slightly (e.g., 1024 -> 992) or by the byte savings from int6 compression elsewhere.

**Who used it.** Introduced in the FP16 Embed entry (1.2197, Renier Velazco). Used in all subsequent entries.

---

## Architecture

### U-Net Skip Connections

**What it is.** Borrowed from the U-Net architecture used in image segmentation, this adds learnable skip connections between the "encoder" (first half) and "decoder" (second half) layers of the transformer. The N layers are split into `N//2` encoder layers and `N - N//2` decoder layers. Each decoder layer receives the output of its corresponding encoder layer (in reverse order), scaled by a learned weight:

```python
# During forward:
skips = []
for i in range(num_encoder_layers):
    x = blocks[i](x, x0)
    skips.append(x)

for i in range(num_decoder_layers):
    if skips:
        x = x + skip_weights[i] * skips.pop()  # pop = reverse order
    x = blocks[num_encoder + i](x, x0)
```

**Why it helps.** Skip connections give the decoder direct access to earlier representations without relying solely on the residual stream. In a model with only 9-11 layers, the residual stream can become "crowded" with information from all layers. Skip connections provide dedicated channels for specific encoder-to-decoder communication.

**Tradeoffs.** Adds `num_skip_weights * model_dim` parameters (typically ~2560 for 5 skip weights x 512 dim). Negligible compute overhead. The `skips` list uses extra memory during the forward pass.

**Who used it.** All entries from the SmearGate+OrthoInit entry (1.1556) onward. The 11-layer entries use 5 encoder + 6 decoder layers.

### MLP Expansion Ratio

**What it is.** The MLP (feed-forward) block in each transformer layer expands the hidden dimension from `model_dim` to `mlp_mult * model_dim`, applies an activation function, then projects back down. The baseline uses `mlp_mult=2` (hidden=1024); top entries use `mlp_mult=3` (hidden=1536).

The activation function used is ReLU-squared (`relu(x)^2`), which is both effective and fast:
```python
def forward(self, x):
    x = torch.relu(self.fc(x))
    return self.proj(x.square())
```

**Why it helps.** The MLP is where most of the model's nonlinear computation happens. A wider MLP provides more expressiveness per layer. Going from 2x to 3x expansion is the single largest contributor to improvement in many entries -- the SmearGate+BigramHash entry called it "the single largest contributor."

**Interaction with quantization.** A 3x MLP at fp16 would be too large for 16MB. But int6 quantization makes the wider MLP fit. This is the key insight: int6 saves bytes, which "fund" the wider MLP, and the wider MLP more than compensates for any int6 precision loss.

**Tradeoffs.** More parameters per layer means either fewer layers or more aggressive quantization. The 3x MLP adds ~50% more parameters to each layer's MLP block. With int6+zstd, the 11-layer 3x-MLP model fits in ~15.5MB.

**Who used it.** All entries from the Mixed Quant entry (1.1630) onward use `MLP_MULT=3`.

### LeakyReLU(0.5)^2 Activation

**What it is.** A drop-in replacement for the standard `relu(x)^2` activation in the MLP. Instead of zeroing negative inputs, LeakyReLU allows a fraction through:

```python
# Standard: dead gradient for x < 0
x = torch.relu(self.fc(x)).square()

# LeakyReLU(0.5)^2: preserves negative gradient flow
x = F.leaky_relu(self.fc(x), negative_slope=0.5).square()
```

With `negative_slope=0.5`, negative inputs are multiplied by 0.5 instead of being zeroed. After squaring, the output is always positive but the gradient flows through both positive and negative inputs.

**Why it helps.** In standard ReLU^2, neurons that receive negative pre-activation are "dead" -- they contribute nothing to the output and receive no gradient signal. With a 3x MLP (hidden=1536), a significant fraction of neurons can be dead at any given time. LeakyReLU preserves gradient flow through all neurons, leading to more effective use of the MLP capacity. The improvement is roughly **-0.003 BPB**.

**Tradeoffs.** Essentially free -- negligible compute overhead. One line of code change. The negative slope of 0.5 was found empirically; the depth recurrence paper (PR #363) notes this improvement is more reliable on 8xH100 with 80 data shards than on smaller setups.

**Who used it.** First introduced by @parinzee (PR #493). Used by the LeakyReLU+TTT SOTA (1.1194) and current SOTA (1.1147).

### Layer Count Tradeoffs

**What it is.** The number of transformer layers in the model. The baseline uses 9 layers. Top entries use 10 or 11.

**Tradeoffs at each count:**
| Layers | Parameters | Fits in 16MB with | Notes |
|--------|-----------|-------------------|-------|
| 9 | ~17M (2x MLP), ~22M (3x MLP) | Int8 (2x), Int6 (3x) | Baseline. Fast steps, more total steps in 10 min. |
| 10 | ~19M (2x MLP), ~24M (3x MLP) | Int6 required | The 10L Int5-MLP entry used int5 on MLP to fund the 10th layer. |
| 11 | ~21M (2x MLP), ~27M (3x MLP) | Int6+zstd-22 required | All top-5 entries use 11 layers. Slower steps (~85ms vs ~44ms). |

More layers provide better language modeling capacity (deeper feature hierarchies, more attention patterns). But more layers mean slower forward passes, hence fewer training steps in the 10-minute budget. The optimal layer count depends on all other techniques in play.

**Who used it.** 9L: baseline and early entries. 10L: Mixed Precision (1.2147), Muon WD (1.1748), Int5-MLP (1.1428). 11L: all entries from MLP3x+QAT (1.1502) onward.

### Overtone/Spectral Initialization

**What it is.** A specialized initialization for the embedding matrix using SVD power-law spectrum shaping. The singular values follow `S_k ~ k^{-0.5}`, mimicking the spectral structure of natural language embeddings. This is sometimes called "overtone" initialization because the decaying spectrum resembles harmonic overtones.

**Why it helps.** Standard random initialization gives the embedding matrix a flat singular value spectrum, which is unrealistic. Real-world embeddings have a few dominant directions and many weak ones. Spectral initialization starts the embedding in a more natural configuration, potentially accelerating early learning.

**Tradeoffs.** The improvement is small and was mainly observed in combination with other techniques. Later entries replaced it with orthogonal initialization of the transformer weights, which proved more impactful.

**Who used it.** The Muon WD entry (1.1748, notapplica) used overtone spectral embedding initialization.

### Orthogonal Initialization

**What it is.** All large weight matrices are initialized using `nn.init.orthogonal_(weight, gain=1.0)`, which produces a matrix whose rows are orthonormal. Output projection weights are additionally scaled by `1/sqrt(2 * num_layers)` following muP (maximal update parametrization) conventions.

```python
if module.weight.ndim == 2 and module.weight.shape[0] >= 64:
    nn.init.orthogonal_(module.weight, gain=1.0)
    if ".proj." in name:
        module.weight.mul_(1.0 / math.sqrt(2 * num_layers))
```

**Why it helps.** Orthogonal matrices have all singular values equal to 1, meaning gradients flow through the network with uniform magnitude at initialization -- no vanishing or exploding gradients. This is especially important when combined with the Muon optimizer, which orthogonalizes gradient updates via Newton-Schulz iteration. Starting from an already-orthogonal matrix means early Muon updates are immediately useful rather than spent correcting a random initialization. With only ~7000 steps available in 10 minutes, faster convergence is critical.

**Tradeoffs.** Only applicable to 2D weight matrices with both dimensions >= 64. The muP-scaled output projections (divided by `sqrt(2*num_layers)`) ensure that the residual stream does not grow in magnitude as you add layers.

**Who used it.** Introduced in the SmearGate+OrthoInit entry (1.1556). Used in all subsequent top entries.

### QK-Norm

**What it is.** RMSNorm is applied independently to the query and key vectors before computing attention scores:

```python
q = F.rms_norm(q, (q.size(-1),))
k = F.rms_norm(k, (k.size(-1),))
```

Additionally, a per-head learned gain (`q_gain`, initialized to 1.5) scales the query after normalization, controlling the sharpness of attention:
```python
q = q * self.q_gain[None, None, :, None]
```

**Why it helps.** Without QK-Norm, the scale of attention logits depends on the magnitude of Q and K vectors, which can vary wildly during training. Normalizing Q and K decouples the attention temperature from the representation magnitude, stabilizing training. The learned `q_gain` lets the model control attention sharpness per-head.

**Tradeoffs.** Adds 8 parameters (one gain per head). The normalization adds a small compute cost but improves training stability significantly, especially with aggressive learning rates.

**Who used it.** All entries use QK-Norm with `qk_gain_init=1.5`.

### LayerNorm Scale Factors

**What it is.** The output of each layer's RMSNorm is scaled by a factor that decreases with layer depth: `1/sqrt(layer_idx + 1)`. Layer 0 has scale 1.0, layer 1 has ~0.707, layer 10 has ~0.302.

```python
self.ln_scale_factor = 1.0 / math.sqrt(layer_idx + 1) if ln_scale else 1.0
# In forward:
attn_out = self.attn(self.attn_norm(x) * self.ln_scale_factor)
```

**Why it helps.** In deep models, the residual stream accumulates contributions from all previous layers, growing in magnitude. This can destabilize training in deeper layers. Damping the deeper layers' input to attention and MLP via a decreasing scale factor prevents this accumulation from causing issues, improving convergence. Zero new parameters.

**Tradeoffs.** The fixed decay schedule may not be optimal for all configurations. It reduces the effective learning signal in deeper layers, which could theoretically hurt if those layers need to learn complex patterns. In practice, the stability gain outweighs any capacity reduction.

**Who used it.** Introduced in the Partial RoPE entry (1.1248, jfprincz). Contributed ~0.001 BPB improvement. Used in the SOTA.

---

## Optimizers

### Muon Optimizer

**What it is.** Muon (MomentUm Orthogonalized by Newton-Schulz) is a specialized optimizer for matrix-shaped parameters. It works like SGD with Nesterov momentum, but post-processes each parameter's gradient update by replacing it with the nearest orthogonal matrix via Newton-Schulz iteration. This is equivalent to steepest descent under the spectral norm.

**How it works (simplified):**
1. Compute gradient `g` for a 2D weight matrix.
2. Update momentum buffer: `buf = momentum * buf + g`
3. Apply Nesterov: `update = g + momentum * buf`
4. Orthogonalize `update` via 5-step Newton-Schulz iteration (find nearest orthogonal matrix).
5. Scale: `update *= sqrt(max(1, rows/cols))`
6. Apply: `weight -= lr * update`

The Newton-Schulz iteration (`zeropower_via_newtonschulz5`) uses the coefficients `(a, b, c) = (3.4445, -4.7750, 2.0315)`:
```python
X = G / G.norm()  # normalize
for _ in range(5):
    A = X @ X.T
    B = b * A + c * A @ A
    X = a * X + B @ X
```

**Why it helps.** Muon's orthogonalization equalizes the update across all singular value directions of the weight matrix, preventing the optimizer from over-updating in a few dominant directions while neglecting others. This leads to better-conditioned training and faster convergence. Muon was borrowed from modded-nanogpt and is particularly effective for short training runs (few thousand steps).

**Tradeoffs.** Only works for 2D parameters (weight matrices). Embedding weights, biases, norms, and other scalars use AdamW instead. The Newton-Schulz iteration adds compute per step, but is highly parallelizable on GPUs.

**Who used it.** All entries use Muon for matrix parameters. It is the default optimizer in the challenge codebase.

### Parallel Muon + Parameter Banking

**What it is.** An enhanced version of Muon introduced by abaybektursun (PR #399). Two key ideas:

1. **Parallel Muon**: Instead of sequential Newton-Schulz orthogonalization across parameters, all weight matrices are orthogonalized in parallel using batched operations. This reduces per-step overhead.

2. **Parameter Banking**: During training, a "bank" of recent parameter checkpoints is maintained. The optimizer can interpolate between the current parameters and the bank to stabilize training and improve convergence.

**Why it helps.** Parallel execution reduces the serial bottleneck of Newton-Schulz iterations. Parameter banking acts as a form of implicit regularization, preventing the model from straying too far from recent good solutions. Together they enable faster, more stable training.

**Who used it.** All entries from the LeakyReLU+TTT entry (1.1194) onward, including the current SOTA (1.1147).

### Muon Momentum Warmup

**What it is.** The Muon momentum coefficient starts at a lower value (0.92) and linearly increases to the target value (0.99) over the first 1500 steps:

```python
frac = min(step / warmup_steps, 1.0)
momentum = (1 - frac) * warmup_start + frac * target_momentum
# e.g., 0.92 -> 0.99 over 1500 steps
```

**Why it helps.** High momentum at the start of training can cause the optimizer to overshoot, especially when the gradient landscape is still being explored. Starting with lower momentum allows the model to quickly find a good region, then high momentum helps it fine-tune within that region. The baseline used `momentum=0.95` with warmup from 0.85 over 500 steps; top entries use `momentum=0.99` with warmup from 0.92 over 1500 steps.

**Who used it.** All entries from the MLP3x+QAT entry (1.1502) onward. The increased momentum (0.99 vs 0.95) and longer warmup (1500 vs 500 steps) are part of the tuned hyperparameter set.

### Muon Weight Decay

**What it is.** Decoupled weight decay applied to all matrix parameters handled by Muon. Before each gradient update, weights are shrunk: `weight *= (1 - lr * wd)`.

**Why it helps.** Weight decay serves dual purposes in parameter golf:
1. **Regularization:** Prevents overfitting during the short training window, improving generalization to the validation set.
2. **Quantization-friendliness:** Weight decay keeps weight magnitudes small and distributions tight, which directly reduces quantization error. Smaller weights need less dynamic range, so the quantization grid is finer relative to typical weight values.

The optimal value was found to be `wd=0.04` through sweeps from 0.01 to 0.05. The baseline had no Muon weight decay.

**Tradeoffs.** Too much weight decay can prevent the model from learning large-scale features. The optimal value depends on other hyperparameters, especially the learning rate schedule.

**Who used it.** Introduced with `wd=0.01` in the SmearGate+OrthoInit entry (1.1556). Increased to `wd=0.04` in later entries. All top-5 entries use `wd=0.04`.

### AdamW for Embeddings

**What it is.** While Muon handles 2D matrix parameters, embedding weights and scalar parameters (layer norms, gates, skip weights) use the AdamW optimizer with separate learning rates and weight decay.

Typical configuration:
- Tied embedding: `lr=0.035`, `wd=0.04`
- Scalar parameters: `lr=0.025`, `wd=0.04`
- Bigram embedding: same group as tied embedding

**Why separate optimizers.** Muon's Newton-Schulz orthogonalization is designed for 2D weight matrices. Embedding matrices, while technically 2D, have a different structure (each row is an independent token vector). Scalar parameters are 1D. AdamW's adaptive per-parameter learning rates are better suited for these parameter types.

**Who used it.** All entries use the Muon + AdamW dual-optimizer setup.

### Learning Rate Schedules

**What it is.** The learning rate follows a warmup-then-warmdown schedule:

1. **Warmup** (first 20 steps): The model does 20 steps of training, then resets to the initial weights and optimizer states. This "primes" the optimizer's momentum buffers without moving the weights, so the first real training steps benefit from a warmed-up optimizer.

2. **Constant phase:** LR stays at peak until warmdown begins.

3. **Warmdown:** LR decays linearly to zero over the final N iterations. The warmdown is wallclock-aware: it estimates when the training will hit the 10-minute cap based on the current step time, and begins decay accordingly.

```python
def lr_mul(step, elapsed_ms):
    step_ms = elapsed_ms / max(step, 1)
    warmdown_ms = warmdown_iters * step_ms
    remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
    if remaining_ms <= warmdown_ms:
        return remaining_ms / warmdown_ms
    return 1.0
```

**Why it helps.** The warmdown is critical for producing smooth weights that quantize well. As the learning rate approaches zero, the model converges to a stable minimum with tight weight distributions. The wallclock-aware scheduling ensures the warmdown completes regardless of how many total steps fit in the time budget.

**Who used it.** All entries use some form of warmup + warmdown. The specific values have evolved: warmdown went from 1200 (baseline) to 3000-3500 (top entries), with warmup at 20 steps.

### Warmdown Timing Effects

**What it is.** The duration of the warmdown phase significantly affects both training quality and quantization quality. Different entries explored different warmdown lengths, revealing a non-obvious tradeoff.

**Key findings:**
- `WARMDOWN_ITERS=1200` (baseline): Too short. The model is still learning when training ends, producing noisy weights that quantize poorly.
- `WARMDOWN_ITERS=3000`: Sweet spot for most configurations. Provides enough decay for smooth weights while allowing sufficient high-LR training.
- `WARMDOWN_ITERS=3500`: Used in the SOTA. Slight improvement over 3000.
- `WARMDOWN_ITERS=20000`: An extreme explored by the Warmdown-Quantization entry, where the entire run is in decay. This produced the best post-quantization quality for int8 but may sacrifice some training quality.

The Warmdown-Quantization entry discovered that `WARMDOWN_ITERS=20000` (far beyond actual steps) reduced the int8 quantization penalty from 0.014 BPB to 0.005 BPB, because the always-decaying LR produces dramatically tighter weight distributions with fewer outliers.

**Who used it.** All entries tune warmdown. The current SOTA (1.1147) uses `WARMDOWN_ITERS=4000`.

---

## Evaluation Tricks

### Sliding Window Evaluation

**What it is.** Instead of evaluating the validation set by splitting it into non-overlapping chunks (where the first tokens in each chunk have zero context), sliding window evaluation uses overlapping windows. The window advances by `stride` tokens (typically 64), but each window is `seq_len` tokens long (typically 2048). Only the last `stride` tokens in each window are scored -- these tokens have nearly the full `seq_len - stride` tokens of preceding context.

**How it works:**
```
Window 1: [t0 ... t2047]  -> score t1984-t2047 (64 tokens, each with 1984+ context)
Window 2: [t64 ... t2111] -> score t2048-t2111 (64 tokens, each with 1984+ context)
Window 3: [t128 ... t2175] -> score t2112-t2175
...
```

Every token in the validation set is scored exactly once, but each scored token gets 1984+ tokens of context instead of the 0-1023 average in non-overlapping evaluation.

**Why it helps.** This is purely an evaluation-time technique -- it does not change the model at all. Language models are better at predicting tokens when they have more context. The sliding window ensures every scored token benefits from near-maximum context. The improvement is ~0.032-0.034 BPB, which is enormous (the entire gap between the baseline and the current SOTA is only 0.10 BPB).

**Tradeoffs.** Much slower evaluation: with stride=64 and seq_len=2048, each token requires 32x more compute than non-overlapping evaluation. Evaluation takes ~70-370 seconds depending on batch size, but this is within the 10-minute evaluation budget. Smaller strides give better scores but take longer. stride=64 provides 1984 context tokens, close to the maximum of 2047.

**Implementation detail:** The `forward_logits` method is compiled with `torch.compile` for efficient batch inference during sliding window eval.

**Who used it.** Introduced in the Sliding Window Eval entry (1.1925, Matthew Li). Used in all subsequent entries. Originally with `EVAL_STRIDE=64`, now `EVAL_STRIDE=16` in top entries.

### Stride-16 vs Stride-64

**What it is.** The sliding window stride determines how many tokens are uniquely scored per window. Stride-64 scores 64 new tokens per window (each with 1984 context). Stride-16 scores only 16 new tokens per window (each with 2032 context).

**Why stride-16 is better.** Each scored token gets 48 more tokens of context (2032 vs 1984). This small increase compounds across the entire validation set for roughly **-0.015 BPB** improvement. The tradeoff is 4x more windows to evaluate, meaning 4x longer eval time.

**Timing.** On 8xH100, stride-16 evaluation takes ~4-6 minutes (vs ~1-2 minutes for stride-64). Still well within the 10-minute eval budget, especially if you're not using TTT.

**Who used it.** The ternary/binary entries (FIIZiK_) pioneered stride-16. The depth recurrence paper (PR #363) confirmed -0.015 BPB gain. All competitive entries now use stride-16.

### Temperature Scaling

**What it is.** Before computing the evaluation loss, logits are divided by a temperature parameter T. Lower temperature (T<1) sharpens the probability distribution; higher temperature (T>1) softens it.

```python
# In eval:
logits = model.forward_logits(x)
logits = logits / temperature  # temperature scaling
loss = F.cross_entropy(logits, targets)
```

**How to find the optimal temperature.** Run a grid search over [0.90, 0.95, 1.00, 1.05, 1.10] using a fast eval (stride=64), then apply the best temperature to the final stride-16 eval.

**Why it helps.** Models trained with aggressive optimization (Muon, high momentum) can produce slightly overconfident or underconfident logits. Temperature scaling corrects the calibration at zero parameter cost. The optimal T is typically around 0.90-0.95 for models using relu^2 activations (the squared activation produces sharper distributions). SwiGLU models tend to prefer T=1.0.

**Impact.** Roughly -0.001 to -0.003 BPB depending on the model. Free at eval time.

**Who used it.** The ternary/binary entries (FIIZiK_) introduced temperature search. The depth recurrence paper (PR #363) confirmed T=0.90 is optimal for relu^2 and that it's activation-dependent.

### Sequence Length and Eval

**What it is.** The sequence length used during training and evaluation can differ. Training at longer sequences (2048 vs 1024) exposes the model to longer-range dependencies, while evaluation at different lengths trades off context quality against positional extrapolation errors.

**Key findings:**
- Training at `seq_len=2048` (vs 1024) gives the model experience with longer contexts, though it slows each step by ~50% due to attention's O(N^2) cost.
- Evaluation at `seq_len=2048` with models trained at 1024 uses NTK-aware RoPE scaling to extrapolate to unseen positions.
- Well-trained models are sensitive to NTK extrapolation -- the Warmdown-Quantization entry found that eval at 1408 (1.375x training length) was optimal for well-trained models, while undertrained models benefited from 2048.
- With sliding window eval, the evaluation sequence length determines the maximum context each token sees.

**Who used it.** The Long Context entry (1.206) first explored `seq_len=2048`. Top entries train at 2048 and evaluate at 2048 with sliding window stride=64.

---

## Test-Time Training

### LoRA TTT

**What it is.** Test-Time Training (TTT) adapts the model to each validation document at evaluation time. For each document:
1. Find document boundaries using BOS tokens.
2. Split the document into overlapping chunks.
3. For each chunk: first score it (accumulate loss for BPB), then train rank-8 LoRA adapters on that chunk's loss.
4. Reset LoRA parameters between documents.

LoRA (Low-Rank Adaptation) adds small trainable matrices to the model's projections: for a weight W, it adds `W + A @ B` where A is (dim, rank) and B is (rank, dim). Only A and B are trained, keeping the original W frozen.

**Why it helps.** Each validation document has its own distribution of tokens and patterns. By adapting the model to the current document (on tokens it has already been scored on -- no data leakage), subsequent predictions within that document can be more accurate. The LoRA adapters target `lm_head`, `c_q`, and `c_v` projections.

**Ablation results from the LoRA TTT entry:**
| Change | val_bpb | Delta |
|--------|---------|-------|
| Baseline (cross-doc, flat stream) | 1.2278 | -- |
| + Document-isolated eval | 1.2168 | -0.0110 |
| + Strided evaluation (chunk=256) | 1.1941 | -0.0337 |
| + LoRA TTT | 1.1910 | -0.0368 |

Notably, most of the improvement came from document isolation and strided evaluation, not the TTT itself (-0.003 BPB from TTT alone).

**Tradeoffs.** TTT adds significant eval-time compute. The LoRA approach enables batched evaluation (64 documents in parallel), making it ~5x faster than full fine-tuning. Uses ~1/10 of the evaluation budget. The technique was not combined with other improvements (it uses the naive baseline model), so there is potential for stacking.

**Who used it.** The LoRA TTT entry (1.1928, samacqua). Largely superseded by Full-Model SGD TTT.

### Full-Model SGD TTT (Score-First)

**What it is.** Instead of training small LoRA adapters, this approach trains the full model weights using SGD with momentum. The key innovation is the "score-first" protocol that ensures legality: for each 32K-token chunk of the validation set, first score all tokens in the chunk (under `inference_mode()`), then train the model on that chunk. Every token is scored BEFORE any update that could use it.

**How it works:**
1. Divide validation tokens into 32K-token chunks
2. For each chunk:
   - **Phase 1 (Score):** Sliding window eval on this chunk's tokens (inference_mode, no gradients)
   - **Phase 2 (Train):** SGD on this chunk's tokens (gradient enabled), with cosine LR decay across chunks
3. The model progressively adapts to the validation data distribution

**Hyperparameters (from PR #549):**
- Optimizer: SGD(lr=0.002, momentum=0.9)
- 3 epochs per chunk
- Cosine LR decay across chunks
- Freeze first 2 blocks for stability
- Gradient clipping at 1.0

**Impact.** On the LeakyReLU+Parallel Muon stack (PR #549): **-0.0025 BPB** improvement. Combined with other changes for a total of 1.1194 BPB.

**IMPORTANT UPDATE (2026-03-25):** The current SOTA author (abaybektursun) **dropped TTT** after 25 failed attempts on the newer Full Hessian GPTQ stack. TTT was neutral or slightly negative when combined with better quantization. The hypothesis is that Full GPTQ already captures much of the distributional information that TTT exploits, making the two techniques redundant.

**Who used it.** PR #549 (1.1194, abaybektursun). Dropped in PR #1019 (1.1147) due to negative interactions with Full GPTQ.

---

## Confirmed Dead Ends

### Depth Recurrence

**What it is.** Reusing the same transformer blocks multiple times in a forward pass (weight sharing across loop iterations). The idea is to get more effective depth per byte of artifact. For example, a "3x3" config: 3 stem blocks + 3 core blocks repeated 3 times + 3 tail blocks = 12 effective layers from 9 unique blocks.

**Why it was promising.** In a 16MB-capped competition, sharing weights across loop iterations should give more effective depth per parameter. Samsung's TRM, Alibaba's Huginn, and Relaxed Recursive Transformers all suggested potential.

**Why it failed (conclusively).** Three independent researchers arrived at the same conclusion:

| Researcher | Best Flat | Best Looped | Gap |
|-----------|-----------|-------------|-----|
| evangelinehelsinki (PR #363) | 1.1648 | 1.1787 | +0.014 |
| Controlled comparison (same config) | 1.1648 | 1.1894 | +0.025 |
| Frosty40 (PR #499) | - | - | "recursion is a bust" |
| FIIZiK_ (250+ experiments) | - | - | "pure noise" across 5 repeat counts |

**The two taxes of recurrence:**
1. **Quantization compounding:** Shared weights are quantized once but errors propagate through every repeat. For 3 repeats, error is seen 3x. For 5 repeats, 5x. Errors compound nonlinearly.
2. **Step time overhead:** Each loop adds wall-clock time. Looped 3x3 = 144ms/step vs flat 11L = 112ms/step. That's 22% fewer training steps in 10 minutes (4175 vs 5375 steps).

**One useful finding: Noisy QAT.** For looped architectures, inject uniform noise calibrated to quantization step size during training (only on shared blocks). This collapsed the quantization gap from 0.37 BPB to 0.002 BPB. The technique doesn't help flat architectures but is novel for any depth-recurrent system.

**Verdict:** Do not pursue depth recurrence for Parameter Golf. The gap is structural, not a tuning problem.

---

## Summary: Technique Impact Table

A rough ordering of techniques by their contribution to the final score, based on ablation data from various entries (updated 2026-03-30):

| Technique | Approximate Impact | Type | Status |
|-----------|-------------------|------|--------|
| Sliding window eval (stride=64) | -0.032 to -0.034 BPB | Eval | Standard |
| MLP 3x expansion (with int6) | -0.025 to -0.030 BPB | Architecture | Standard |
| 11 layers (from 9, with int6+zstd) | -0.015 to -0.020 BPB | Architecture | Standard |
| Stride-16 eval (vs stride-64) | ~-0.015 BPB | Eval | New standard |
| QAT STE (eliminate quant gap) | -0.010 to -0.016 BPB | Quantization | Standard |
| FP16 embeddings | -0.006 to -0.010 BPB | Quantization | Standard |
| EMA weight averaging | -0.005 to -0.006 BPB | Training | Standard |
| Full Hessian GPTQ (vs GPTQ-lite) | -0.003 to -0.005 BPB | Quantization | **New SOTA** |
| Weight decay 0.04 | -0.003 to -0.005 BPB | Training | Standard |
| SmearGate + BigramHash | -0.003 to -0.005 BPB | Embeddings | Standard |
| LeakyReLU(0.5)^2 (vs relu^2) | ~-0.003 BPB | Architecture | New standard |
| XSA on all 11 layers | -0.002 to -0.003 BPB | Attention | **New SOTA** |
| Full-model SGD TTT | -0.0025 BPB | Eval | **Dead on new stack** |
| Partial RoPE (16/64) | -0.002 BPB | Attention | Standard |
| Temperature scaling (T=0.90) | -0.001 to -0.003 BPB | Eval | New standard |
| Orthogonal initialization | -0.001 to -0.003 BPB | Architecture | Standard |
| LZMA-9 (vs zstd-22) | Enables larger model | Compression | **New SOTA** |
| Warmdown 4000 | -0.001 to -0.002 BPB | Training | **New SOTA** |
| LN Scale factors | -0.001 BPB | Architecture | Standard |
| Selective ±1 pruning | Enables smaller artifact | Quantization | **New SOTA** |
| GPTQ-lite clip search | -0.0006 BPB | Quantization | Superseded |
| Depth recurrence | +0.025 BPB WORSE | Architecture | **Dead** |

Note: impacts are approximate and context-dependent. Many techniques interact (e.g., int6 enables MLP 3x; Full GPTQ makes TTT redundant).

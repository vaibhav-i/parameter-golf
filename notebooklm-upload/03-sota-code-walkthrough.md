# SOTA Code Walkthrough: 11L EMA + GPTQ-lite (1.1233 BPB)

**Script**: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`
**Baseline reference**: `records/track_10min_16mb/2026-03-17_NaiveBaseline/train_gpt.py`

This document walks through the current state-of-the-art Parameter Golf training script line by line. It is written for someone with beginner-to-intermediate ML knowledge who wants to understand not just *what* the code does, but *why* each piece exists and how all the pieces fit together.

---

## Section 1: Architecture Overview

### ASCII Architecture Diagram

```
                     INPUT TOKEN IDS  [B, T]
                            |
                   +--------v--------+
                   |   tok_emb       |  nn.Embedding(1024, 512)
                   |   (tied w/ LM)  |  Shared with output projection
                   +--------+--------+
                            |
                   +--------v--------+
                   |  BigramHash      |  2048 buckets, dim=128, proj to 512
                   |  Embedding       |  Adds bigram context (additive)
                   +--------+--------+
                            |
                   +--------v--------+
                   |  RMSNorm         |  Normalize combined embedding
                   +--------+--------+
                            |
                   +--------v--------+
                   |  SmearGate       |  Learned mix of current + previous position
                   +--------+--------+
                            |
                       x0 = x  (saved for residual mixing in every block)
                            |
          ==================|==================
          |    ENCODER (5 layers: blocks 0-4)  |
          |                                     |
          |  For each layer i in [0..4]:        |
          |    x_in = mix[0]*x + mix[1]*x0      |  <-- residual mix with original embedding
          |    x = x_in + attn_scale * Attn(     |
          |          LN(x_in) * ln_scale_factor) |  <-- LN scale = 1/sqrt(i+1)
          |    x = x + mlp_scale * MLP(          |
          |          LN(x) * ln_scale_factor)    |
          |    skips.append(x)                   |  <-- save for U-Net
          |                                     |
          |  Value Embeddings injected at        |
          |  layers 9,10 only (decoder side)     |
          ======================================
                            |
          ==================|==================
          |    DECODER (6 layers: blocks 5-10)  |
          |                                     |
          |  For each layer i in [0..5]:        |
          |    x = x + skip_weight * skips.pop()|  <-- U-Net skip from encoder
          |    (same Block structure as above)   |
          |                                     |
          |  Layers 7-10: XSA enabled           |  <-- last 4 layers only
          |  Layers 9-10: Value Embedding       |  <-- shared VE + per-layer scale
          ======================================
                            |
                   +--------v--------+
                   |  final RMSNorm   |
                   +--------+--------+
                            |
                   +--------v--------+
                   |  Logit Projection|  F.linear(x, tok_emb.weight)  (tied)
                   |  + Softcap(30.0) |  30 * tanh(logits / 30)
                   +--------+--------+
                            |
                   +--------v--------+
                   |  Cross-Entropy   |
                   |  Loss            |
                   +-----------------+
```

### Key Dimensions

| Parameter | Value | Notes |
|-----------|-------|-------|
| `vocab_size` | 1024 | SentencePiece BPE, very small vocab |
| `model_dim` | 512 | Width of residual stream |
| `num_layers` | 11 | 5 encoder + 6 decoder |
| `num_heads` | 8 | Query heads |
| `num_kv_heads` | 4 | Key/Value heads (GQA, 2:1 ratio) |
| `head_dim` | 64 | = model_dim / num_heads |
| `mlp_mult` | 3.0 | MLP hidden = 1536 |
| `rope_dims` | 16 | Only 16 of 64 head dims get RoPE |
| `train_seq_len` | 2048 | Context window |
| `logit_softcap` | 30.0 | Caps logit magnitudes |
| `bigram_vocab_size` | 2048 | Hash buckets for bigram embedding |
| `bigram_dim` | 128 | Bigram embedding width before projection |
| `ve_dim` | 128 | Value embedding width |
| `xsa_last_n` | 4 | XSA on layers 7, 8, 9, 10 |

### Parameter Count Breakdown

Here is an approximate breakdown (all values in parameters, not bytes):

| Component | Calculation | Params |
|-----------|-------------|--------|
| **tok_emb** | 1024 x 512 | 524,288 |
| **BigramHash embed** | 2048 x 128 | 262,144 |
| **BigramHash proj** | 128 x 512 | 65,536 |
| **BigramHash scale** | 1 | 1 |
| **SmearGate** | 512 | 512 |
| **Per block (x11):** | | |
| - attn Q proj | 512 x 512 | 262,144 |
| - attn K proj | 512 x 256 | 131,072 |
| - attn V proj | 512 x 256 | 131,072 |
| - attn out proj | 512 x 512 | 262,144 |
| - MLP fc | 512 x 1536 | 786,432 |
| - MLP proj | 1536 x 512 | 786,432 |
| - attn_scale | 512 | 512 |
| - mlp_scale | 512 | 512 |
| - resid_mix | 2 x 512 | 1,024 |
| - q_gain | 8 | 8 |
| **Block subtotal** | per block | ~2,361,352 |
| **All 11 blocks** | | ~25,974,872 |
| **skip_weights** | 5 x 512 | 2,560 |
| **ValueEmbed shared** | 1024 x 128 + 128 x 256 + 1 | 163,841 |
| **VE layer scales** | 2 | 2 |
| **final_norm** | (parameterless) | 0 |
| **Total (approx)** | | ~26.99M |

The model trains at ~27M parameters but ships as an int6-quantized artifact under 16MB. The MTP heads (if enabled) are excluded from the export.

### Baseline vs. SOTA: Key Differences at a Glance

| Feature | Baseline | SOTA |
|---------|----------|------|
| Layers | 9 | 11 |
| MLP multiplier | 2x (1024) | 3x (1536) |
| Sequence length | 1024 | 2048 |
| Batch tokens/step | 524,288 | 786,432 |
| Attention | PyTorch SDPA | FlashAttention 3 |
| RoPE | Full (all 64 dims) | Partial (16/64 dims) |
| XSA | None | Last 4 layers |
| BigramHash | None | 2048 buckets |
| SmearGate | None | Yes |
| Value Embedding | None | Shared, layers 9-10 |
| LN Scale Factor | None | 1/sqrt(layer+1) |
| QAT | None | Late int6 STE |
| EMA | None | decay=0.997 |
| SWA | None | Every 50 steps when scale<0.2 |
| Quantization | int8 + zlib | int6 GPTQ-lite + zstd-22 |
| Weight init | Zero-init only | OrthoInit + muP scaling |
| Weight decay | None | 0.04 (Muon + Adam) |
| Grad clipping | None | 0.3 |

---

## Section 2: Code Map

### 2.1 Imports and Compression Setup (Lines 1-26)

```python
try:
    import zstandard
    _COMPRESSOR = "zstd"
except ImportError:
    _COMPRESSOR = "zlib"
```

The script preferentially uses `zstandard` (zstd) for compression. Zstd at level 22 compresses the int6 quantized weights significantly better than zlib, which is critical for squeezing under the 16MB artifact limit. The `flash_attn_interface` import on line 26 pulls in FlashAttention 3, which is optimized for Hopper (H100) GPUs and is noticeably faster than PyTorch's built-in SDPA.

**Baseline difference**: The baseline uses only `zlib` and PyTorch's built-in `F.scaled_dot_product_attention`.

### 2.2 Hyperparameters (Lines 27-86)

The `Hyperparameters` class holds every tunable knob as class-level attributes, each overridable via environment variables. Key groups:

**Model shape** (lines 45-53):
- `vocab_size=1024`, `num_layers=11`, `model_dim=512`, `num_heads=8`, `num_kv_heads=4`
- `mlp_mult=3.0` -- 3x expansion means MLP hidden dim = 1536
- `tie_embeddings=True` -- the output projection reuses `tok_emb.weight`

**Training schedule** (lines 37-43):
- `iterations=20000` (but wallclock-capped at 600s, so typically ~7100 steps)
- `warmdown_iters=3500` -- the LR warmdown window
- `warmup_steps=20` -- torch.compile warmup (not LR warmup)
- `train_batch_tokens=786432` -- total tokens per gradient step across all GPUs
- `train_seq_len=2048`, `eval_seq_len=2048`

**Optimizer** (lines 54-76):
- `matrix_lr=0.025` (Muon), `tied_embed_lr=0.035` (AdamW), `scalar_lr=0.025` (AdamW)
- `muon_momentum=0.99`, warmup from 0.92 over 1500 steps
- `grad_clip_norm=0.3`
- `muon_wd=0.04`, `adam_wd=0.04` -- weight decay on all param groups

**Novel features** (lines 77-86):
- `bigram_vocab_size=2048` -- BigramHash buckets
- `xsa_last_n=4` -- XSA on last 4 layers
- `rope_dims=16` -- partial RoPE (only 16 of 64 head dims)
- `late_qat_threshold=0.15` -- enable fake quantization when LR scale drops below 0.15
- `ve_enabled=True`, `ve_dim=128`, `ve_layers="9,10"` -- Value Embedding

### 2.3 Muon Optimizer (Lines 87-151)

**Newton-Schulz orthogonalization** (lines 87-98):

```python
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    ...
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
```

This computes an approximate "zero-power" (orthogonal) version of the gradient matrix. Think of it as: instead of applying the raw gradient, Muon applies a rotation-like update that preserves the orthogonality of the weight matrix. The Newton-Schulz iteration converges to the orthogonal polar factor in 5-10 steps. The constants `a, b, c` are pre-tuned coefficients for a 5th-order iteration.

**Why this matters**: Muon (Matrix Update via Orthogonalization of Newton-schulz) has been shown to train transformers faster than Adam for matrix-shaped parameters. The orthogonalization acts as an implicit preconditioner.

**Muon optimizer step** (lines 99-151):
1. Accumulate momentum buffer: `buf = momentum * buf + grad`
2. Apply Nesterov: `g = grad + momentum * buf`
3. Orthogonalize: `g = newtonschulz(g)`
4. Scale correction: `g *= max(1, rows/cols)^0.5` (compensates for non-square matrices)
5. Distribute across ranks via `all_reduce` (each rank handles a subset of params)
6. Apply weight decay as decoupled: `p *= (1 - lr * wd)`
7. Apply update: `p -= lr * g`

**Baseline difference**: The baseline Muon has no `weight_decay` parameter. The SOTA adds decoupled weight decay (`muon_wd=0.04`).

### 2.4 BPB Evaluation Helpers (Lines 152-241)

`build_sentencepiece_luts` (lines 152-176) builds lookup tables that map token IDs to byte counts. This is needed because the competition metric is bits-per-byte (BPB), not bits-per-token. The byte count per token depends on the SentencePiece encoding of each piece.

`eval_val` (lines 186-241) runs standard validation: splits the validation set into non-overlapping sequences of length `seq_len`, runs them through the model, and computes mean cross-entropy loss and BPB.

### 2.5 Quantization Infrastructure

#### Int8 Quantization (Lines 242-353) -- used for embeddings

The int8 path is carried over from the baseline. Key points:
- `CONTROL_TENSOR_NAME_PATTERNS` (line 242) lists parameter names that should be kept in full precision (e.g., scales, gains, gates)
- Tensors with <= 65,536 elements are kept as fp16 passthrough (too small to benefit from quantization)
- 2D tensors get per-row scales; vectors/scalars get per-tensor scales
- Clip percentile is 99.99984% (line 261) to handle outliers

#### Int6 Quantization with GPTQ-lite (Lines 885-954) -- used for MLP + attention weights

This is the critical innovation. Int6 means each weight is quantized to the range [-31, 31] (6 bits of information), stored as int8 but with the restricted range.

**GPTQ-lite clip search** (lines 885-904):

```python
def quantize_int6_per_row(t, clip_range=31):
    ...
    for pct in [0.9990, 0.9995, 0.9999, 0.99999, 1.0]:
        if pct < 1.0:
            row_clip = torch.quantile(t32.abs(), pct, dim=1)
        else:
            row_clip = t32.abs().amax(dim=1)
        s = (row_clip / clip_range).clamp_min(1.0 / clip_range).to(torch.float16)
        q = torch.clamp(torch.round(t32 / s.float()[:, None]), -clip_range, clip_range).to(torch.int8)
        recon = q.float() * s.float()[:, None]
        err = (t32 - recon).pow(2).mean().item()
        if err < best_err:
            best_q, best_s, best_err = q, s, err
    return best_q, best_s
```

Instead of always clipping at the row maximum (pct=1.0), it tries 5 different clip percentiles. For each one, it quantizes the row, reconstructs, and measures MSE. It keeps whichever percentile gave the lowest reconstruction error. This is "GPTQ-lite" because it borrows the idea of optimizing the quantization parameters post-training (like GPTQ) but uses a much simpler approach.

**Why 5 candidates works**: Some rows have outlier weights that waste quantization range. Clipping at the 99.9th percentile can reduce MSE because the few clipped weights lose less than the many weights that gain better resolution from a tighter scale.

`mixed_quantize_int6` (lines 905-934) routes each parameter to either int6 (for MLP and attention weights) or int8 (for embeddings), based on the `_classify_param` helper (lines 877-884).

### 2.6 RMSNorm (Lines 408-413)

```python
class RMSNorm(nn.Module):
    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)
```

A parameterless normalization (no learnable gain/bias). RMSNorm just divides by the root-mean-square of the input, which is cheaper than LayerNorm and works well in practice.

### 2.7 CastedLinear with QAT (Lines 414-426)

```python
class CastedLinear(nn.Linear):
    _qat_enabled: bool = False   # class-level flag, toggled during training
    def forward(self, x):
        w = self.weight.to(x.dtype)
        if CastedLinear._qat_enabled and self.training and w.ndim == 2:
            with torch.no_grad():
                w32 = self.weight.float()
                row_max = w32.abs().amax(dim=1)
                scale = (row_max / 31.0).clamp_min(1.0 / 31.0)
                w_q = (torch.clamp(torch.round(w32 / scale[:, None]), -32, 31) * scale[:, None]).to(x.dtype)
            w = w + (w_q - w).detach()   # STE: straight-through estimator
        return F.linear(x, w, bias)
```

This is **Quantization-Aware Training (QAT)** with the **Straight-Through Estimator (STE)**:

1. **Forward pass**: Quantize the weight to int6 range [-32, 31], reconstruct it, and use the reconstructed weight for the forward pass.
2. **Backward pass**: The `.detach()` on `(w_q - w)` means the gradient flows through `w` as if quantization never happened. This is the STE trick -- pretend the rounding operation has gradient 1.

**Why this helps**: Without QAT, the model trains with full-precision weights and then gets quantized post-training, causing a "quantization gap" in loss. With QAT, the model learns to be robust to quantization during training itself.

**Late QAT**: QAT is not enabled from the start. It kicks in when the LR scale drops below `late_qat_threshold=0.15` (checked on line 1225). This means the model first trains normally, then adapts to quantization noise during the warmdown phase.

**Baseline difference**: The baseline `CastedLinear` has no QAT logic at all -- it just casts weights to the compute dtype.

### 2.8 Partial RoPE (Lines 432-473)

```python
class Rotary(nn.Module):
    def __init__(self, dim, base=10000.0, train_seq_len=1024, rope_dims=0):
        self.rope_dims = rope_dims if rope_dims > 0 else dim
```

With `rope_dims=16` and `head_dim=64`, only the first 16 dimensions of each head get rotary position encoding. The remaining 48 dimensions are position-free.

**NTK-aware scaling** (lines 452-456): When `seq_len > train_seq_len`, the RoPE base frequency is scaled up:
```python
scale = seq_len / self.train_seq_len
new_base = self.base * (scale ** (rd / (rd - 2)))
```
This is the NTK-aware interpolation formula that allows the model to extrapolate to longer sequences without retraining.

**`apply_rotary_emb` with partial RoPE** (lines 464-473):
```python
if rope_dims > 0 and rope_dims < x.size(-1):
    x_rope, x_pass = x[..., :rope_dims], x[..., rope_dims:]
    # apply rotation to x_rope only
    return torch.cat((x_rope_rotated, x_pass), dim=-1)
```

**Why partial RoPE**: Full RoPE on all 64 head dims forces every dimension to be position-dependent. Partial RoPE leaves most dimensions position-free, which gives the model more capacity for content-based attention while still having positional information.

**Baseline difference**: The baseline uses full RoPE on all head dimensions and has no NTK-aware scaling.

### 2.9 CausalSelfAttention (Lines 474-531)

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, ...):
        self.c_q = CastedLinear(dim, dim, bias=False)          # 512 -> 512
        self.c_k = CastedLinear(dim, kv_dim, bias=False)       # 512 -> 256 (GQA)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)       # 512 -> 256 (GQA)
        self.proj = CastedLinear(dim, dim, bias=False)         # 512 -> 512
        self.q_gain = nn.Parameter(...)                        # per-head learned scaling
        self.use_xsa = False                                   # toggled by GPT.__init__
```

**GQA (Grouped Query Attention)**: With 8 query heads and 4 KV heads, each pair of query heads shares one KV head. This halves the KV parameter count and memory while maintaining most of the quality.

**Forward pass** (lines 513-531):
1. Project Q, K, V from input
2. If `v_embed` is provided (Value Embedding), add it to V before reshaping
3. Apply QK-Norm: `q = rms_norm(q)`, `k = rms_norm(k)` -- this stabilizes attention
4. Apply partial RoPE to Q and K
5. Scale Q by learned `q_gain` (per-head)
6. Run FlashAttention 3: `y = flash_attn_3_func(q, k, v, causal=True)`
7. If XSA is enabled, apply `_xsa_efficient` to the output
8. Project back with `self.proj`

**Tensor shapes**: Note that unlike the baseline (which transposes to `[B, H, T, D]` for PyTorch SDPA), the SOTA keeps tensors in `[B, T, H, D]` format because FlashAttention 3 expects this layout.

### 2.10 XSA -- Cross-head Self-Attention Subtraction (Lines 503-512)

```python
def _xsa_efficient(self, y, v):
    """Subtract self-value projection via GQA-aware reshape."""
    B, T, H, D = y.shape
    Hkv = v.size(-2)
    group = H // Hkv                                    # = 2 (8 heads / 4 kv heads)
    y_g = y.reshape(B, T, Hkv, group, D)               # group query heads by their KV head
    vn = F.normalize(v, dim=-1).unsqueeze(-2)           # unit-norm value vector
    proj = (y_g * vn).sum(dim=-1, keepdim=True) * vn   # project attention output onto value direction
    return (y_g - proj).reshape(B, T, H, D)             # subtract that projection
```

XSA removes the component of the attention output that lies along the value direction. Intuitively: standard attention computes a weighted sum of values, so the output tends to be dominated by the "self" token's value. XSA subtracts this self-value component, forcing the attention output to capture *differences* from the current token rather than the token itself (which the residual stream already carries).

**The GQA-aware trick**: Instead of `repeat_interleave` to expand KV heads, it reshapes the query heads into groups. Each group of 2 query heads shares one KV head's value vector for the subtraction. This is a zero-allocation approach.

**Enabled on layers 7-10 only** (set in `GPT.__init__`, line 710): XSA is most useful in deeper layers where the residual stream already has strong token identity.

### 2.11 SmearGate (Lines 532-539)

```python
class SmearGate(nn.Module):
    def __init__(self, dim):
        self.gate = nn.Parameter(torch.zeros(dim))    # initialized to 0 (no smearing)
    def forward(self, x):
        g = sigmoid(self.gate)
        x_prev = cat([zeros_like(x[:, :1]), x[:, :-1]], dim=1)  # shift right by 1
        return (1 - g) * x + g * x_prev
```

SmearGate blends each position's embedding with the *previous* position's embedding. The gate is learned per-dimension. This gives the model a cheap way to incorporate one-step-back context before the transformer layers even start processing.

**Why it works**: The SentencePiece tokenizer breaks words into subword pieces. SmearGate helps the model "see" that the current token is a continuation of the previous one, without needing an attention layer.

### 2.12 BigramHashEmbedding (Lines 540-561)

```python
class BigramHashEmbedding(nn.Module):
    def __init__(self, bigram_vocab_size, bigram_dim, model_dim):
        self.embed = nn.Embedding(bigram_vocab_size, bigram_dim)  # 2048 x 128
        self.proj = CastedLinear(bigram_dim, model_dim)           # 128 -> 512
        self.scale = nn.Parameter(tensor(0.05))

    def bigram_hash(self, tokens):
        mod = self.bigram_vocab_size - 1      # = 2047
        out[..., 0] = mod                      # first position has no bigram
        out[..., 1:] = (36313 * t[..., 1:]) XOR (27191 * t[..., :-1]) % mod
        return out
```

This hashes consecutive token pairs (bigrams) into 2047 buckets (plus one special bucket for position 0). The hash is deterministic and cheap. The resulting embedding is projected to model_dim and scaled by a small learned factor.

**Why hashing**: With vocab_size=1024, there are ~1M possible bigrams. Storing a full bigram embedding table would be enormous. Hashing into 2048 buckets is a compact approximation that captures common bigram patterns.

### 2.13 ValueEmbedding (Lines 562-577)

```python
class ValueEmbedding(nn.Module):
    def __init__(self, vocab_size, ve_dim, model_dim):
        self.embed = nn.Embedding(vocab_size, ve_dim)    # 1024 x 128
        self.proj = CastedLinear(ve_dim, model_dim)      # 128 -> 256 (kv_dim)
        self.scale = nn.Parameter(tensor(0.1))
```

Value Embedding re-injects the original token identity into the attention *values* at specific layers (9 and 10). By this deep in the network, the residual stream has been heavily mixed by attention and MLP layers, so the original token identity may be diluted. The VE reminds the model "what token is actually here."

The shared embedding table is computed once and cached (`ve_cache` in the forward pass, line 743), then scaled by per-layer learned scalars.

### 2.14 MLP (Lines 578-587)

```python
class MLP(nn.Module):
    def __init__(self, dim, mlp_mult):
        hidden = int(mlp_mult * dim)      # = 1536
        self.fc = CastedLinear(dim, hidden)
        self.proj = CastedLinear(hidden, dim)
        self.proj._zero_init = True       # output projection starts at zero

    def forward(self, x):
        x = torch.relu(self.fc(x))
        return self.proj(x.square())      # ReLU-squared activation
```

**ReLU-squared**: `relu(x)^2` is a smooth, always-non-negative activation that produces sparser outputs than GELU or SiLU. The squaring emphasizes larger activations and suppresses small ones, which empirically improves language modeling at small scale.

**Zero-init on proj**: The output projection weight starts at zero, so at initialization each block's MLP contributes nothing. This is a standard trick for training deep residual networks stably.

### 2.15 Block (Lines 588-625)

```python
class Block(nn.Module):
    def __init__(self, dim, ..., layer_idx, ln_scale, dtg):
        self.attn_scale = nn.Parameter(ones(dim))     # per-dim learned residual scale
        self.mlp_scale = nn.Parameter(ones(dim))
        self.resid_mix = nn.Parameter(stack([ones(dim), zeros(dim)]))  # [2, dim]
        self.ln_scale_factor = 1.0 / sqrt(layer_idx + 1) if ln_scale else 1.0
```

Each block implements a pre-norm transformer layer with several extra features:

1. **Residual mixing** (line 617-618): `x_in = mix[0]*x + mix[1]*x0` -- the block's input is a learned blend of the running residual `x` and the original embedding `x0`. The initial values [1, 0] mean "use only the running residual" at the start of training. Over time, the model can learn to mix in the original embedding.

2. **LN scale factor** (line 609): `1/sqrt(layer_idx+1)` scales down the normalized activations in deeper layers. This is a form of depth scaling that helps training stability -- deeper layers start with smaller contributions.

3. **Per-dim residual scales** (lines 620-621): Both attention and MLP outputs are scaled by learned per-dimension vectors before being added to the residual. This gives the model fine-grained control over which dimensions are influenced by which sub-layer.

4. **DTG gate** (lines 610-615, 622-624): "Dynamic Token Gating" -- currently disabled (`dtg_enabled=False`). If enabled, it would learn a per-token gate that blends the block's output with its input, allowing the model to skip processing for some tokens.

### 2.16 GPT Model (Lines 626-781)

**Constructor** (lines 626-712):

The GPT class wires everything together:

1. Creates `tok_emb`, optionally `bigram` embedding, `smear` gate
2. Computes encoder/decoder split: `num_encoder_layers = 11 // 2 = 5`, `num_decoder_layers = 6`
3. Creates `skip_weights` for U-Net connections (5 weights, one per encoder layer)
4. Creates 11 `Block` instances
5. Sets partial RoPE on all blocks (lines 684-688)
6. Sets up shared Value Embedding for layers 9 and 10 (lines 689-698)
7. Enables XSA on the last 4 layers (lines 709-711)
8. Runs weight initialization (lines 713-725)

**Weight initialization** (lines 713-725):
- Tied embedding uses small normal init (std=0.005)
- Zero-init modules (output projections) start at zero
- All other 2D weights >= 64x64 get **orthogonal initialization** -- this is significantly better than random for Muon-trained networks
- Output projections (`.proj`) get scaled by `1/sqrt(2 * num_layers)` -- **muP-style scaling** that keeps the signal magnitude stable through the depth of the network

**Forward pass** (lines 735-781):

```python
def forward(self, input_ids, target_ids):
    x = self.tok_emb(input_ids)
    if self.bigram is not None:
        x = x + self.bigram(input_ids)        # add bigram embedding
    x = F.rms_norm(x, (x.size(-1),))          # normalize
    x = self.smear(x)                          # blend with previous position
    x0 = x                                     # save for residual mixing

    # ENCODER (layers 0-4): save skip connections
    skips = []
    for i in range(self.num_encoder_layers):
        ve = self._get_ve(i, input_ids, ve_cache)
        x = self.blocks[i](x, x0, v_embed=ve)
        skips.append(x)

    # DECODER (layers 5-10): consume skip connections
    for i in range(self.num_decoder_layers):
        bi = self.num_encoder_layers + i
        if skips:
            x = x + self.skip_weights[i] * skips.pop()   # U-Net skip
        ve = self._get_ve(bi, input_ids, ve_cache)
        x = self.blocks[bi](x, x0, v_embed=ve)

    # Output head
    x = self.final_norm(x)
    logits = F.linear(x, self.tok_emb.weight)              # tied embedding
    logits = 30.0 * tanh(logits / 30.0)                    # softcap
    return cross_entropy(logits.float(), targets)
```

**U-Net skip connections**: The encoder layers push activations onto a stack; the decoder layers pop them off. Because of the stack (LIFO), encoder layer 0's output goes to decoder layer 5's input, layer 1's to layer 4's, etc. This creates symmetric "skip connections" like in a U-Net, helping gradients flow and letting the decoder access early-layer features.

**Multi-token prediction** (lines 765-781): If `mtp_num_heads > 0` (currently 0 in defaults), additional linear heads predict tokens 2, 3, ... steps ahead. The MTP loss is weighted by `mtp_loss_weight` and added to the main loss. MTP heads are excluded from the exported artifact, so they cost zero at inference.

### 2.17 Sliding Window Evaluation (Lines 808-876)

```python
def eval_val_sliding(args, base_model, ..., stride, batch_seqs=32):
```

Standard evaluation processes the validation set as non-overlapping chunks of `seq_len` tokens. This means the first tokens in each chunk have very little context. Sliding window evaluation overlaps windows by `stride` tokens:

1. Generate window starting positions every `stride` tokens (line 825)
2. For each window, run the full model to get per-token NLL (line 851-856)
3. Only score the *last `stride` tokens* of each window (except for the first window, which scores all tokens) (line 859)

With `stride=64` and `seq_len=2048`, each token is scored with up to 2048 tokens of context. This is much better than standard evaluation where early-in-chunk tokens only have a few tokens of context. The tradeoff is ~32x more compute (2048/64 windows instead of 1).

The evaluation model is `torch.compile`d separately (line 835) for maximum inference speed.

### 2.18 Main Training Function (Lines 955-1402)

#### Setup (Lines 955-1028)

- **torch.compile** the Newton-Schulz function (line 959) for Muon speed
- Set up distributed training (NCCL backend, one process per GPU)
- `grad_accum_steps = 8 // world_size` -- with 8 GPUs, no gradient accumulation needed
- Force FlashSDP backend, disable all others (lines 980-984)
- Load tokenizer, validation tokens, and BPB lookup tables

#### Model + Optimizer Setup (Lines 1029-1142)

The model is created in bf16, then all `CastedLinear` modules are cast back to fp32 (lines 1054-1056). This means weight storage and optimizer state are fp32, but the actual matrix multiply happens in bf16 via `CastedLinear.forward()`.

**Optimizer split** (4 separate optimizers):

| Optimizer | Parameters | Algorithm | LR |
|-----------|-----------|-----------|-----|
| `optimizer_tok` | tok_emb.weight, bigram embed, VE embed | AdamW | 0.035 |
| `optimizer_muon` | All 2D weights in blocks + bigram proj + VE proj | Muon | 0.025 |
| `optimizer_scalar` | All 1D params, skip_weights, smear gate, scales | AdamW | 0.025 |
| `optimizer_head` | lm_head (if untied, not used here) | Adam | 0.008 |

#### Warmup Phase (Lines 1158-1182)

This is **not** learning rate warmup. It is **torch.compile warmup**: the model runs 20 real training steps, then resets all weights and optimizer states to their initial values. This forces torch.compile to JIT-compile all kernels before the real training clock starts, so the timed training loop is not penalized by compilation time.

After warmup, the data loader is also reset (line 1182) so training sees the same data it would have without warmup.

#### EMA State Initialization (Lines 1183-1186)

```python
ema_state = {name: t.detach().float().clone() for name, t in base_model.state_dict().items()}
ema_decay = 0.997
```

An **Exponential Moving Average** of all model parameters is maintained from step 0. At each step, the EMA is updated: `ema = 0.997 * ema + 0.003 * current_weights`. This produces a smoother version of the weights that often generalizes better than the final checkpoint.

#### Training Loop Core (Lines 1192-1281)

Each iteration:

1. **Check wallclock** and compute LR scale (lines 1223-1227)
2. **Late QAT check** (line 1225): if `scale < 0.15`, enable int6 fake quantization in all `CastedLinear` modules
3. **Micro-step loop** (lines 1230-1237): accumulate gradients over `grad_accum_steps` micro-batches
4. **Muon momentum warmup** (lines 1239-1242): linearly ramp momentum from 0.92 to 0.99 over 1500 steps
5. **LR schedule** (lines 1243-1245): multiply all base LRs by the warmdown scale
6. **Gradient clipping** (line 1247): global norm clip at 0.3
7. **Optimizer step** (lines 1248-1250)
8. **EMA update** (lines 1252-1254):
   ```python
   ema_state[name].mul_(0.997).add_(t.detach().float(), alpha=0.003)
   ```
9. **SWA collection** (lines 1257-1265): when `scale < 0.2` and step is divisible by 50, add current weights to a running sum. This is **Stochastic Weight Averaging** -- it averages multiple checkpoints from the warmdown phase.

#### LR Schedule: Wallclock-Based Warmdown (Lines 1148-1157)

```python
def lr_mul(step, elapsed_ms):
    step_ms = elapsed_ms / max(step, 1)          # average ms per step
    warmdown_ms = warmdown_iters * step_ms        # estimated warmdown duration
    remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
    return remaining_ms / warmdown_ms if remaining_ms <= warmdown_ms else 1.0
```

The LR schedule is **wallclock-based**, not step-based. It estimates when 3500 steps before the 600-second deadline will occur (based on average step time), then linearly decays the LR to 0 from that point. This is more robust than step-based scheduling because different hardware speeds are automatically handled.

#### Post-Training: EMA Application (Lines 1286-1301)

After training ends, the EMA weights replace the current model weights:
```python
avg_state = {name: t.to(dtype=current_state[name].dtype) for name, t in ema_state.items()}
base_model.load_state_dict(avg_state, strict=True)
```

Note: The SWA state (`swa_state`) is computed but the EMA state is what gets applied. The README notes that EMA was found to be better than SWA alone.

#### Export and Quantization (Lines 1302-1326)

1. MTP heads are excluded from export (line 1303)
2. `mixed_quantize_int6` is called with `{"mlp", "attn"}` as int6 categories (line 1314)
3. The quantized state dict is serialized with `torch.save`, then compressed with `zstd` level 22 (line 1318)
4. The total artifact = compressed model bytes + code bytes (line 1325)

#### Roundtrip Validation (Lines 1327-1398)

The script verifies the exported artifact by:
1. Decompressing the zstd blob
2. Dequantizing back to floating point
3. Loading into a fresh `GPT` model (with `mtp_num_heads=0`)
4. Running standard evaluation
5. Running sliding-window evaluation at `stride=64`
6. Optionally running a second sliding-window at the configured `eval_stride`

This ensures the reported BPB reflects what a judge would measure from the artifact.

---

## Section 3: Modification Guide

### 3.1 BigramHash Bucket Size

**Variable**: `bigram_vocab_size` in `Hyperparameters` class
**Line**: 77
**Current value**: `2048`

```python
bigram_vocab_size = int(os.environ.get("BIGRAM_VOCAB_SIZE", 2048))
```

To change: set the env var `BIGRAM_VOCAB_SIZE=10240` or edit the default. The embedding table on line 544 will automatically resize. Note that increasing buckets increases the embedding parameter count (currently 2048 x 128 = 262K params; at 10240 it would be 1.31M params). This adds to the artifact size, so check the 16MB limit.

**Also check**: `bigram_dim` on line 78 (currently 128). You could keep the same bucket count but change the embedding dimension.

### 3.2 Warmdown Timing

**Variable**: `warmdown_iters` in `Hyperparameters` class
**Line**: 38
**Current value**: `3500`

```python
warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 3500))
```

This controls how many steps before the wallclock deadline the LR starts decaying. Larger values = more gradual decay = less time at peak LR. The warmdown function is on lines 1148-1157. You can also override via `WARMDOWN_ITERS=4000` env var.

**Related**: `late_qat_threshold` (line 83, default 0.15) -- QAT kicks in when the LR scale drops below this. If you increase warmdown, you might want to adjust this too.

### 3.3 QK-Norm Addition

**Currently**: QK-Norm already exists on lines 521-522:
```python
q = F.rms_norm(q, (q.size(-1),))
k = F.rms_norm(k, (k.size(-1),))
```

This applies RMSNorm to Q and K *before* RoPE. If the research notes refer to a different style of QK-Norm (e.g., LayerNorm with learnable parameters, or post-RoPE normalization), you would modify lines 521-526 in `CausalSelfAttention.forward`.

To add *learnable* QK-Norm, you would:
1. Add `self.q_norm = nn.LayerNorm(self.head_dim)` and `self.k_norm = nn.LayerNorm(self.head_dim)` in `__init__` (after line 501)
2. Replace lines 521-522 with `q = self.q_norm(q)` and `k = self.k_norm(k)`
3. Add the new params to `scalar_params` in the optimizer setup (they are 1D, so they will be auto-classified)

### 3.4 Partial RoPE Fraction

**Variable**: `rope_dims` in `Hyperparameters` class
**Line**: 80
**Current value**: `16` (out of `head_dim=64`)

```python
rope_dims = int(os.environ.get("ROPE_DIMS", 16))
```

Changing this to, say, 32 would apply RoPE to half the head dimensions. Set to 0 or 64 for full RoPE. The actual application is in `apply_rotary_emb` (lines 464-473) and the `Rotary` class (lines 432-463). Override via `ROPE_DIMS=32`.

### 3.5 Adding a 12th Layer

**Variable**: `num_layers` in `Hyperparameters` class
**Line**: 46
**Current value**: `11`

Changing to 12:
1. Set `NUM_LAYERS=12` (or edit line 46)
2. The encoder/decoder split recomputes: `num_encoder_layers = 12 // 2 = 6`, `num_decoder_layers = 6`
3. `skip_weights` resizes to 6 (from 5)
4. XSA will be on layers 8-11 (last 4)
5. `ve_layers` (line 86) may need updating -- currently `"9,10"`, you might want `"10,11"` for the deepest layers

**Size impact**: Each block adds ~2.36M parameters. At int6, that is roughly 2.36M * 0.75 bytes = ~1.77 MB uncompressed. After zstd-22, probably ~1.0-1.3 MB. Check the artifact size.

**Timing impact**: More layers = more compute per step = fewer steps in 600 seconds. You may need to reduce warmdown_iters to compensate.

### 3.6 Adding Multi-Token Prediction Heads

**Variable**: `mtp_num_heads` in `Hyperparameters` class
**Line**: 69
**Current value**: `0` (disabled)

Set `MTP_NUM_HEADS=2` and `MTP_LOSS_WEIGHT=0.2` to enable. The heads are defined on lines 704-708:
```python
self.mtp_heads = nn.ModuleList(
    [CastedLinear(model_dim, vocab_size, bias=False) for _ in range(mtp_num_heads)]
)
```

Each MTP head is a simple linear projection from `model_dim` to `vocab_size`. Head `k` predicts token `k+2` from position `t` (i.e., 2-step, 3-step, etc. ahead). The loss is computed on lines 765-780 and added with weight `mtp_loss_weight`.

**Key property**: MTP heads are **excluded from the export** (line 1303: `if "mtp_heads" not in k`). They cost zero at inference. The training cost is the extra forward/backward through the linear heads plus memory for the additional parameters.

**Where they get optimized**: MTP head weights are 2D, so they get added to `matrix_params` on line 1067 and optimized by Muon.

### 3.7 LoRA TTT (Test-Time Training) at Eval Time

This would be a significant addition. The hook points are:

1. **After loading the quantized model** (line 1351): `eval_model.load_state_dict(deq_state, strict=True)`
2. **Before running sliding-window evaluation** (line 1370)

To implement LoRA TTT:
1. After line 1351, add LoRA adapters to the eval model (e.g., to the Q, K, V projections)
2. Run a TTT loop: for each window of validation tokens, fine-tune the LoRA adapters on the tokens *you have already scored* (past tokens are fair game), then score the new tokens
3. Reset or keep the LoRA state between windows depending on your strategy

**Critical constraint**: You can only train on tokens *after* they have been evaluated. The eval time limit is also 10 minutes on 8xH100.

**Where to add LoRA modules**: Modify `CausalSelfAttention.__init__` (around lines 494-497) to accept optional LoRA rank. Or, add LoRA wrappers after model construction (monkey-patching the linear layers).

---

## Section 4: Gotchas and Traps

### 4.1 Things That Look Modifiable But Are Dangerous

**Do NOT change `grad_accum_steps` directly.** It is derived from `8 // world_size` (line 968). Changing it breaks the relationship between batch size and the number of GPUs. If you want a different effective batch size, change `train_batch_tokens` instead.

**Do NOT remove `restore_low_dim_params_to_fp32` (line 1057).** This function ensures scalar parameters (scales, gains, gates, etc.) stay in fp32 even though the model body is bf16. Without it, these small parameters lose precision and training diverges.

**Do NOT change `CastedLinear` to keep weights in bf16.** The whole point of CastedLinear is that weights live in fp32 for optimizer state quality but compute in bf16. The Muon optimizer and the QAT STE both rely on fp32 weights.

**The `_qat_enabled` flag is a class variable** (line 415), not an instance variable. Setting it affects *all* CastedLinear instances simultaneously. This is intentional (QAT is all-or-nothing) but could be surprising if you try to enable QAT on specific layers only.

### 4.2 Hyperparameter Interdependencies

**warmdown_iters + late_qat_threshold**: QAT kicks in when LR scale < 0.15. The LR scale reaches 0.15 when `remaining_ms / warmdown_ms = 0.15`. If you increase warmdown_iters, QAT kicks in earlier in training. If you decrease it, QAT has less time to adapt. These two should be tuned together.

**warmdown_iters + swa_every**: SWA collects checkpoints when `scale < 0.2`. With warmdown_iters=3500 and step time ~84ms, the warmdown phase starts around step ~3600. SWA starts when scale drops to 0.2 (80% through warmdown). If warmdown is very short, SWA may never collect enough checkpoints.

**rope_dims + head_dim**: `rope_dims` must be even and <= `head_dim` (64). If you change `model_dim` or `num_heads`, `head_dim` changes and `rope_dims` may need adjustment.

**num_layers + ve_layers**: If you change `num_layers`, update `ve_layers` to reference valid layer indices. The string is parsed on line 689.

**num_layers + xsa_last_n**: XSA is enabled on layers `[num_layers - xsa_last_n, num_layers)` (line 710). If you add layers, the XSA layers shift automatically, which is probably what you want.

**bigram_vocab_size and artifact size**: The bigram embedding table is quantized as a "small tensor" (< 65536 elements at 2048 x 128 = 262,144... wait, that is > 65,536). Actually, at 2048 x 128 = 262,144 elements, it goes through the full quantization path. At 10,240 x 128 = 1,310,720 elements, this is a significant addition. Check the artifact size after any change.

### 4.3 torch.compile Implications

**The warmup phase is essential.** The 20-step warmup (lines 1158-1182) forces torch.compile to JIT-compile all code paths before the training clock starts. Without it, the first ~5 real training steps are 3-10x slower due to compilation. Since the 600-second limit is wallclock time, losing 30 seconds to compilation is a real cost.

**fullgraph=True and dynamic=False** (line 1058): This tells torch.compile to compile the entire model as a single graph with static shapes. This enables maximum optimization but means:
- You cannot change batch size or sequence length between calls without recompilation
- Python control flow inside the model must be resolvable at trace time
- The `if self.use_xsa` check (line 528) is resolved at compile time, so XSA layers and non-XSA layers are compiled as different graphs

**Dead code elimination**: torch.compile will remove code paths that are never executed. For example, if `mtp_num_heads=0`, the MTP logic in the forward pass (lines 765-780) is eliminated at compile time. If you later set `mtp_num_heads > 0`, you need to recompile.

**Compiled eval model** (line 1352): The evaluation model is compiled separately with its own graph. It uses `forward_logits` (lines 782-807) which skips loss computation. The sliding-window eval compiles this on line 835.

### 4.4 Timing Constraints and Bottlenecks

**Training is GPU-compute-bound.** The main bottleneck is the forward/backward through 11 transformer layers with 786K tokens per step. Data loading is overlapped and negligible.

**Evaluation time is separate** from training time in the competition, but the script does evaluation within the same run. The sliding-window evaluation at stride=64 is ~32x more expensive than standard evaluation. With torch.compile and batching (batch_seqs=32), it takes roughly 1-3 minutes on 8xH100.

**GPTQ-lite clip search** (lines 885-904) runs 5 quantile computations per weight row. This is done once post-training on CPU and takes a few seconds. It is not a bottleneck, but if you add many more candidates (e.g., 50 percentiles), it could slow down export.

**zstd-22 compression** (line 1318) is slow (~5-10 seconds) because level 22 is the highest compression level. This is worth it because it saves ~0.5 MB compared to lower levels, but it cannot be parallelized.

**The EMA update** (lines 1252-1254) adds a small overhead per step (one pass over all parameters to update the running average). With ~27M parameters in fp32, this is about 100MB of memory reads/writes per step. On H100 this takes <1ms per step, which is negligible compared to the ~84ms per training step.

### 4.5 Subtle Correctness Issues

**SWA is computed but may not be used.** The EMA state is what gets applied on line 1290. The SWA state is computed (lines 1257-1265) but never applied to the model in this version of the script. The README notes EMA beats SWA alone. If you want to experiment with SWA, you would need to add code to apply `swa_state` (dividing by `swa_count`) after line 1290.

**The sliding window scores tokens differently at the first window.** Line 859: `s = 0 if ws == 0 else max(wlen - stride, 0)`. The first window scores all its tokens (even the ones with little context), while subsequent windows only score the last `stride` tokens. This is standard practice but means the first few tokens have slightly worse scores.

**FlashAttention 3 vs. PyTorch SDPA.** The SOTA uses `flash_attn_3_func` directly (line 527), bypassing PyTorch's `F.scaled_dot_product_attention`. The tensor layout is `[B, T, H, D]` (not `[B, H, T, D]`). If you switch back to PyTorch SDPA, you need to add `.transpose(1, 2)` calls like the baseline does (lines 585-602 of the baseline).

**The `_zero_init` attribute** (e.g., line 498, 584, 703) is a custom flag that `_init_weights` checks. If you add new linear layers and forget to set `_zero_init = True` on output projections, they will get orthogonal init instead of zero init, which changes training dynamics.

### 4.6 Memory Considerations

The model itself is ~27M params at fp32 = ~108 MB. The EMA state doubles this to ~216 MB. Optimizer states (Adam for embeddings/scalars, Muon momentum buffers for matrices) add another ~100-200 MB. Activations for a batch of 786K tokens at seq_len=2048 = 384 sequences, times 11 layers, are the dominant memory cost. On 8xH100 (80 GB each), memory is not a concern, but if you want to run on fewer GPUs, you may need to reduce batch size.

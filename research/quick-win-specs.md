# Quick-Win Experiment Specifications

**Date**: 2026-03-24
**Base**: SOTA entry `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` (val_bpb: 1.1233, artifact: 15.55 MB)
**Script**: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

All experiments modify **only environment variables or a few lines** of the SOTA `train_gpt.py`. Each is independent and can run in parallel.

---

## Statistical Framework (applies to all experiments)

**Seeds**: Run each variant with seeds 1337, 42, and 2024 (same as SOTA baseline).

**How to compare**: After each run, look for this log line:
```
final_int6_sliding_window_s64_exact val_loss:X.XXXXXXXX val_bpb:X.XXXXXXXX
```
Record the `val_bpb` value. Compute the mean across 3 seeds.

**Success criteria**: Mean val_bpb < 1.1233 (beats SOTA mean).

**Abort criteria**: If seed 1337 alone gives val_bpb > 1.1260 (worse than SOTA by more than noise), abort the remaining seeds for that variant.

**Statistical significance** (for a submission PR): You need the improvement to be >= 0.005 nats at p < 0.01. With 3 seeds per variant:
```python
# After collecting 3-seed means for baseline and variant:
from scipy.stats import ttest_ind
baseline_bpbs = [1.1228, 1.1236, 1.1236]  # SOTA 3-seed results
variant_bpbs  = [X, Y, Z]                  # your 3-seed results
t_stat, p_value = ttest_ind(baseline_bpbs, variant_bpbs, alternative='greater')
print(f"p-value: {p_value:.4f}")  # need p < 0.01
```
For quick-wins that individually improve < 0.005 nats, you do NOT need to submit them as standalone PRs. Instead, stack multiple quick wins together (see Experiment 5 below) to cross the 0.005 threshold.

---

## Experiment 1: BigramHash Bucket Sweep

### Current Value

- **File**: `train_gpt.py`, line 77
- **Variable**: `bigram_vocab_size`
- **Current value**: `2048`
- **Line content**: `bigram_vocab_size = int(os.environ.get("BIGRAM_VOCAB_SIZE", 2048))`

### Values to Test

| Variant | Buckets | Env var override |
|---------|---------|-----------------|
| A | 4096 | `BIGRAM_VOCAB_SIZE=4096` |
| B | 8192 | `BIGRAM_VOCAB_SIZE=8192` |
| C | 10240 | `BIGRAM_VOCAB_SIZE=10240` |

### Why This Should Work

Entry #5 (`2026-03-20_10L_Int5MLP_MuonWD04_SWA50`) used 10240 buckets and showed a clear ablation progression:
- bigram=4096 (base) -> bigram=8192: -0.0012 BPB
- bigram=8192 -> bigram=10240: -0.0008 BPB

The SOTA code still uses 2048 buckets, which is the smallest tested size. Increasing it is expected to yield 0.001-0.003 BPB improvement.

### Artifact Size Calculation

The BigramHash has two weight tensors:
1. **`bigram.embed.weight`**: shape `[bigram_vocab_size, 128]` (Embedding table)
2. **`bigram.proj.weight`**: shape `[512, 128]` (projection, constant regardless of bucket count)
3. **`bigram.scale`**: 1 scalar (passthrough, negligible)

The `_classify_param` function (line 877) classifies bigram params as `"other"`. In `mixed_quantize_int6` (line 905), category `"other"` goes to **int8 quantization** (line 929-933). Each element becomes 1 byte (int8) + per-row fp16 scale (2 bytes per row).

**Raw int8 bytes for embed table (before zstd compression)**:

| Buckets | embed elements | embed int8 bytes | embed scales (fp16) | proj (constant) | Total raw bytes |
|---------|---------------|-------------------|---------------------|-----------------|----------------|
| 2048 (current) | 262,144 | 262,144 | 4,096 | 65,536 + 1,024 | 332,800 |
| 4096 | 524,288 | 524,288 | 8,192 | 65,536 + 1,024 | 599,040 |
| 8192 | 1,048,576 | 1,048,576 | 16,384 | 65,536 + 1,024 | 1,131,520 |
| 10240 | 1,310,720 | 1,310,720 | 20,480 | 65,536 + 1,024 | 1,397,760 |

**Delta raw bytes vs. current**:

| Buckets | Raw delta | Estimated zstd-22 ratio | Compressed delta | New artifact estimate |
|---------|-----------|------------------------|------------------|-----------------------|
| 4096 | +266,240 | ~1.5x | ~177 KB | ~15.73 MB |
| 8192 | +798,720 | ~1.5x | ~533 KB | ~16.08 MB |
| 10240 | +1,064,960 | ~1.5x | ~710 KB | ~16.26 MB |

**IMPORTANT**: 8192 and 10240 buckets risk exceeding the 16 MB (16,000,000 bytes) artifact limit. The zstd compression ratio for int8 bigram embeddings may be better or worse than 1.5x depending on weight distribution.

**Mitigation**: If 10240 exceeds 16 MB, try:
1. Reduce `bigram_dim` from 128 to 96 (saves 25% of embed bytes)
2. Use int6 for bigram by changing line 1314 from `{"mlp", "attn"}` to `{"mlp", "attn", "bigram"}` -- but you also need to add `"bigram"` to the `_classify_param` function

### Exact Code Change (env var only -- no code edit needed)

No code changes required. Override via environment variable:

```bash
# Example for 4096 buckets:
BIGRAM_VOCAB_SIZE=4096 SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### If Bigram Exceeds 16 MB: Int6 Bigram Fallback

If the artifact is too large, edit two sections:

**Change 1** -- line 877, add bigram classification:
```python
# OLD (line 877-884):
def _classify_param(name: str) -> str:
    if "tok_emb" in name or "lm_head" in name:
        return "embed"
    if ".mlp." in name:
        return "mlp"
    if ".attn." in name or (".proj." in name and ".mlp." not in name):
        return "attn"
    return "other"

# NEW:
def _classify_param(name: str) -> str:
    if "tok_emb" in name or "lm_head" in name:
        return "embed"
    if ".mlp." in name:
        return "mlp"
    if ".attn." in name or (".proj." in name and ".mlp." not in name):
        return "attn"
    if "bigram" in name:
        return "bigram"
    return "other"
```

**Change 2** -- line 1314, add "bigram" to int6 categories:
```python
# OLD (line 1314):
    quant_result, quant_meta = mixed_quantize_int6(sd_cpu, {"mlp", "attn"})

# NEW:
    quant_result, quant_meta = mixed_quantize_int6(sd_cpu, {"mlp", "attn", "bigram"})
```

Int6 uses 6 bits per element vs int8's 8 bits = 25% smaller. This would save ~25% of the bigram embed bytes, bringing 10240 buckets back under budget.

### Run Commands

```bash
# --- Variant A: 4096 buckets ---
BIGRAM_VOCAB_SIZE=4096 SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
BIGRAM_VOCAB_SIZE=4096 SEED=42    torchrun --standalone --nproc_per_node=8 train_gpt.py
BIGRAM_VOCAB_SIZE=4096 SEED=2024  torchrun --standalone --nproc_per_node=8 train_gpt.py

# --- Variant B: 8192 buckets ---
BIGRAM_VOCAB_SIZE=8192 SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
BIGRAM_VOCAB_SIZE=8192 SEED=42    torchrun --standalone --nproc_per_node=8 train_gpt.py
BIGRAM_VOCAB_SIZE=8192 SEED=2024  torchrun --standalone --nproc_per_node=8 train_gpt.py

# --- Variant C: 10240 buckets ---
BIGRAM_VOCAB_SIZE=10240 SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
BIGRAM_VOCAB_SIZE=10240 SEED=42    torchrun --standalone --nproc_per_node=8 train_gpt.py
BIGRAM_VOCAB_SIZE=10240 SEED=2024  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

**After each run**: Check the log output for:
1. `Total submission size int6+zstd: XXXXX bytes` -- must be < 16,000,000
2. `final_int6_sliding_window_s64_exact val_loss:X.XXXXXXXX val_bpb:X.XXXXXXXX` -- record the val_bpb

### Abort / Success Criteria

- **Abort**: If artifact > 16,000,000 bytes, stop and apply the int6 bigram fallback above.
- **Abort**: If seed 1337 val_bpb > 1.1260, skip remaining seeds for that bucket count.
- **Success**: Mean val_bpb < 1.1233 across 3 seeds.
- **Strong success**: Mean val_bpb < 1.1220 (keep for stacking).

---

## Experiment 2: Warmdown Sweep

### Current Value

- **File**: `train_gpt.py`, line 38
- **Variable**: `warmdown_iters`
- **Current value**: `3500`
- **Line content**: `warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 3500))`

### How Warmdown Works in This Code

The learning rate schedule is **wallclock-based** (not step-based). See lines 1147-1157:

```python
def lr_mul(step: int, elapsed_ms: float) -> float:
    if args.warmdown_iters <= 0:
        return 1.0
    if max_wallclock_ms is None:
        warmdown_start = max(args.iterations - args.warmdown_iters, 0)
        return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) ...
    step_ms = elapsed_ms / max(step, 1)
    warmdown_ms = args.warmdown_iters * step_ms
    remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
    return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0
```

The warmdown kicks in when `remaining_ms <= warmdown_ms`. Since `warmdown_ms = warmdown_iters * step_ms`, and each step takes roughly the same time, `warmdown_iters=3500` means "start decaying when there are ~3500 steps worth of time remaining."

With ~7100 total steps in 600 seconds, that means warmdown starts at roughly step 3600 (about halfway through training). More warmdown = more time spent decaying = smoother convergence but less time at peak LR.

### Values to Test

| Variant | warmdown_iters | Approx warmdown start step | Fraction of training at peak LR |
|---------|---------------|---------------------------|-------------------------------|
| A | 3000 | ~4100 | ~58% |
| B | 3500 (current) | ~3600 | ~51% |
| C | 4000 | ~3100 | ~44% |
| D | 4500 | ~2600 | ~37% |
| E | 5000 | ~2100 | ~30% |

### Interaction with Other Hyperparameters

**SWA (Tight SWA)**: SWA checkpoints are collected when the LR scale drops below 0.2. With a longer warmdown, the LR decays more slowly, so the scale=0.2 threshold is reached later (fewer SWA checkpoints). This could slightly affect SWA quality.

**QAT threshold**: Late QAT activates when LR scale < 0.15 (line not explicitly shown but documented in README). Longer warmdown means QAT kicks in later and runs for fewer steps.

**EMA**: EMA runs every step regardless of warmdown. No interaction.

**No other adjustments needed** -- warmdown is independent of all other hyperparameters. Just change the one value.

### Exact Code Change (env var only)

No code changes required:
```bash
WARMDOWN_ITERS=4000 SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Run Commands

```bash
# --- Variant A: 3000 ---
WARMDOWN_ITERS=3000 SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
WARMDOWN_ITERS=3000 SEED=42    torchrun --standalone --nproc_per_node=8 train_gpt.py
WARMDOWN_ITERS=3000 SEED=2024  torchrun --standalone --nproc_per_node=8 train_gpt.py

# --- Variant C: 4000 ---
WARMDOWN_ITERS=4000 SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
WARMDOWN_ITERS=4000 SEED=42    torchrun --standalone --nproc_per_node=8 train_gpt.py
WARMDOWN_ITERS=4000 SEED=2024  torchrun --standalone --nproc_per_node=8 train_gpt.py

# --- Variant D: 4500 ---
WARMDOWN_ITERS=4500 SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
WARMDOWN_ITERS=4500 SEED=42    torchrun --standalone --nproc_per_node=8 train_gpt.py
WARMDOWN_ITERS=4500 SEED=2024  torchrun --standalone --nproc_per_node=8 train_gpt.py

# --- Variant E: 5000 ---
WARMDOWN_ITERS=5000 SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
WARMDOWN_ITERS=5000 SEED=42    torchrun --standalone --nproc_per_node=8 train_gpt.py
WARMDOWN_ITERS=5000 SEED=2024  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Abort / Success Criteria

- **Abort**: If seed 1337 val_bpb > 1.1260, skip remaining seeds for that variant.
- **Success**: Mean val_bpb < 1.1233 across 3 seeds.
- **Pattern to watch**: If results improve monotonically as warmdown increases (3500 -> 4000 -> 4500 -> 5000), try 5500 and 6000. If there is a clear U-shape, the optimal is at the minimum.

---

## Experiment 3: QK-Norm Addition

### Background

**QK-Norm** applies RMSNorm to the Q and K tensors before computing the attention dot product. This prevents attention logits from growing too large, stabilizing training -- especially in deeper models.

The SOTA code **already has QK-Norm**. Look at lines 521-522 of `train_gpt.py`:

```python
q = F.rms_norm(q, (q.size(-1),))
k = F.rms_norm(k, (k.size(-1),))
```

This is exactly QK-Norm. It was already implemented in the SOTA entry.

### What This Experiment Becomes: QK-Norm Removal Test + Gain Sweep

Since QK-Norm is already present, the experiment pivots to:

**Variant A: Remove QK-Norm** (test if it is actually helping or hurting)
**Variant B: QK-Norm with different q_gain_init values**

### Where Attention Is Computed

- **File**: `train_gpt.py`, lines 513-531
- **Class**: `CausalSelfAttention.forward`

The full forward method:

```python
def forward(self, x: Tensor, v_embed: Tensor | None = None) -> Tensor:       # line 513
    bsz, seqlen, dim = x.shape                                                # line 514
    q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim)       # line 515
    k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)    # line 516
    v = self.c_v(x)                                                            # line 517
    if v_embed is not None:                                                    # line 518
        v = v + v_embed                                                        # line 519
    v = v.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)              # line 520
    q = F.rms_norm(q, (q.size(-1),))       # <-- QK-Norm on Q                 # line 521
    k = F.rms_norm(k, (k.size(-1),))       # <-- QK-Norm on K                 # line 522
    cos, sin = self.rotary(seqlen, x.device, q.dtype)                         # line 523
    q = apply_rotary_emb(q, cos, sin, self.rope_dims)                         # line 524
    k = apply_rotary_emb(k, cos, sin, self.rope_dims)                         # line 525
    q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]                # line 526
    y = flash_attn_3_func(q, k, v, causal=True)                               # line 527
    if self.use_xsa:                                                           # line 528
        y = self._xsa_efficient(y, v)                                          # line 529
    y = y.reshape(bsz, seqlen, dim)                                           # line 530
    return self.proj(y)                                                        # line 531
```

### Current q_gain_init Value

- **File**: `train_gpt.py`, line 44
- **Variable**: `qk_gain_init`
- **Current value**: `1.5`
- **Line content**: `qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))`

The `q_gain` parameter (line 499) is initialized to `qk_gain_init` per head (8 heads, each gets a learned scalar starting at 1.5). This controls the magnitude of Q after QK-Norm. Higher gain = sharper attention; lower gain = more uniform attention.

### LN Scale Env Var

- **File**: `train_gpt.py`, line 81
- **Variable**: `ln_scale`
- **Current value**: `True` (enabled)
- **Line content**: `ln_scale = bool(int(os.environ.get("LN_SCALE", "1")))`
- **To disable**: Set `LN_SCALE=0` as an env var (no code edit needed)

### Variant A: Remove QK-Norm (ablation test)

To remove QK-Norm, comment out lines 521-522:

```python
# OLD (lines 521-522):
    q = F.rms_norm(q, (q.size(-1),))
    k = F.rms_norm(k, (k.size(-1),))

# NEW:
    # q = F.rms_norm(q, (q.size(-1),))
    # k = F.rms_norm(k, (k.size(-1),))
```

This is a CODE CHANGE (not an env var). Copy the SOTA `train_gpt.py` to a new experiment directory first.

### Variant B: QK-Norm + LN Scale Interaction

The SOTA code already applies LN Scale at line 619:

```python
attn_out = self.attn(self.attn_norm(x_in) * self.ln_scale_factor, v_embed=v_embed)
```

Where `self.ln_scale_factor = 1.0 / math.sqrt(layer_idx + 1)` (line 609). This scales the input to attention by `1/sqrt(layer+1)`, which indirectly reduces Q and K magnitudes before QK-Norm renormalizes them.

**Test**: Remove LN Scale to see if QK-Norm alone is sufficient:

No code edit needed. Use the env var:

```bash
LN_SCALE=0 SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

This sets `ln_scale = bool(int("0")) = False` (line 81), which makes `ln_scale_factor = 1.0` for all layers (line 609), effectively disabling LN Scale while QK-Norm (lines 521-522) remains active.

### Variant C: q_gain_init Sweep

The `q_gain` learned parameter starts at `qk_gain_init`. Sweeping this tests whether the attention temperature initialization matters:

```bash
QK_GAIN_INIT=0.5  SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
QK_GAIN_INIT=1.0  SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
QK_GAIN_INIT=2.0  SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Interactions with XSA, Partial RoPE, Flash Attention

- **XSA**: XSA subtracts the self-value projection after attention. It operates on the attention *output*, not Q/K. QK-Norm affects attention *inputs*. No direct interaction.
- **Partial RoPE**: RoPE is applied after QK-Norm (lines 524-525 come after 521-522). The first `rope_dims=16` dimensions get rotary encoding; the remaining 48 are untouched. QK-Norm normalizes the full 64-dim vector before splitting. This is correct -- normalizing before RoPE ensures both rotary and non-rotary dimensions start at similar magnitudes.
- **Flash Attention 3**: FA3 computes standard scaled dot-product attention. QK-Norm does not affect FA3 compatibility. Note: FA3 applies its own `1/sqrt(head_dim)` scaling internally, and `q_gain` adjusts the effective temperature on top of that.

### Run Commands

```bash
# --- Variant A: Remove QK-Norm (requires code edit) ---
# First copy train_gpt.py, comment out lines 521-522, then:
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
SEED=42    torchrun --standalone --nproc_per_node=8 train_gpt.py
SEED=2024  torchrun --standalone --nproc_per_node=8 train_gpt.py

# --- Variant B: Remove LN Scale (QK-Norm only) ---
# Either set LN_SCALE=0 (if env var supported) or edit line 609
LN_SCALE=0 SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
LN_SCALE=0 SEED=42    torchrun --standalone --nproc_per_node=8 train_gpt.py
LN_SCALE=0 SEED=2024  torchrun --standalone --nproc_per_node=8 train_gpt.py

# --- Variant C: q_gain sweep (single seed first) ---
QK_GAIN_INIT=0.5 SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
QK_GAIN_INIT=1.0 SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
QK_GAIN_INIT=2.0 SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Abort / Success Criteria

- **Abort**: If removing QK-Norm (Variant A) gives val_bpb > 1.1300, it confirms QK-Norm is essential. Do not pursue removal.
- **Success for Variant B**: If LN_SCALE=0 gives val_bpb < 1.1233, QK-Norm alone is sufficient and you can remove LN Scale (saves no parameters, but simplifies the architecture and may improve training dynamics).
- **Success for Variant C**: If any q_gain_init beats 1.1233, adopt it.

---

## Experiment 4: Partial RoPE Dimension Sweep

### Current Value

- **File**: `train_gpt.py`, line 80
- **Variable**: `rope_dims`
- **Current value**: `16` (meaning 16 out of 64 head dimensions get RoPE, i.e., 25%)
- **Line content**: `rope_dims = int(os.environ.get("ROPE_DIMS", 16))`

### How Partial RoPE Works

Head dimension is 64 (model_dim=512 / num_heads=8). With `rope_dims=16`:
- First 16 dims: rotary position encoding applied
- Remaining 48 dims: no position encoding (pure content-based)

The split happens at line 465-470:
```python
if rope_dims > 0 and rope_dims < x.size(-1):
    x_rope, x_pass = x[..., :rope_dims], x[..., rope_dims:]
    half = rope_dims // 2
    x1, x2 = x_rope[..., :half], x_rope[..., half:]
    x_rope = torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)
    return torch.cat((x_rope, x_pass), dim=-1)
```

`rope_dims` must be even (it is split into two halves for the rotation).

### Values to Test

| Variant | rope_dims | Fraction | Description |
|---------|-----------|----------|-------------|
| A | 8 | 8/64 = 12.5% | Minimal position info |
| B | 16 (current) | 16/64 = 25% | Current SOTA |
| C | 24 | 24/64 = 37.5% | More positional capacity |
| D | 32 | 32/64 = 50% | Half and half |

### Exact Code Change (env var only)

No code changes required:
```bash
ROPE_DIMS=24 SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Impact on Artifact Size

Zero. RoPE dimensions only affect the forward pass computation, not the number of parameters. The Q, K weight matrices are the same size regardless of how many dims get RoPE.

### Run Commands

```bash
# --- Variant A: 8 dims ---
ROPE_DIMS=8 SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
ROPE_DIMS=8 SEED=42    torchrun --standalone --nproc_per_node=8 train_gpt.py
ROPE_DIMS=8 SEED=2024  torchrun --standalone --nproc_per_node=8 train_gpt.py

# --- Variant C: 24 dims ---
ROPE_DIMS=24 SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
ROPE_DIMS=24 SEED=42    torchrun --standalone --nproc_per_node=8 train_gpt.py
ROPE_DIMS=24 SEED=2024  torchrun --standalone --nproc_per_node=8 train_gpt.py

# --- Variant D: 32 dims ---
ROPE_DIMS=32 SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
ROPE_DIMS=32 SEED=42    torchrun --standalone --nproc_per_node=8 train_gpt.py
ROPE_DIMS=32 SEED=2024  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Abort / Success Criteria

- **Abort**: If seed 1337 val_bpb > 1.1260 for a variant, skip remaining seeds.
- **Success**: Mean val_bpb < 1.1233.
- **Pattern to watch**: There should be a sweet spot. Too few RoPE dims = insufficient positional information. Too many = wastes capacity on position when content matters more. If 24 is better than 16 and 32, the optimum is nearby. If 8 is best, try 4 and 12.

---

## Experiment 5: Combined Quick Wins

### How to Stack

After running Experiments 1-4, take the **best variant from each** and combine them in a single run. Because all four changes are orthogonal (they modify different parts of the model), they can be stacked with simple env var overrides.

### Example Combined Run

Assuming the winners were: bigram=4096, warmdown=4000, rope_dims=24 (and QK-Norm stays as-is):

```bash
BIGRAM_VOCAB_SIZE=4096 WARMDOWN_ITERS=4000 ROPE_DIMS=24 \
  SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py

BIGRAM_VOCAB_SIZE=4096 WARMDOWN_ITERS=4000 ROPE_DIMS=24 \
  SEED=42 torchrun --standalone --nproc_per_node=8 train_gpt.py

BIGRAM_VOCAB_SIZE=4096 WARMDOWN_ITERS=4000 ROPE_DIMS=24 \
  SEED=2024 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Dependencies and Order

There are **no ordering dependencies** -- all four experiments modify independent parameters:

| Experiment | What it changes | Interacts with |
|-----------|----------------|----------------|
| BigramHash buckets | Embedding table size | Artifact size budget only |
| Warmdown | LR schedule | SWA timing, QAT timing (minor) |
| QK-Norm / LN Scale | Attention normalization | None (already present) |
| RoPE dims | Position encoding split | None |

The only constraint is **artifact size**: if you increase BigramHash buckets significantly, verify the combined artifact still fits in 16 MB.

### Expected Cumulative Improvement

Individual quick wins are typically 0.0005-0.002 BPB each. Stacking 2-3 winners:
- **Conservative estimate**: 0.001-0.003 BPB total (improvements are sub-additive)
- **Optimistic estimate**: 0.003-0.005 BPB total (if improvements are additive)

To cross the 0.005 nats threshold for a new SOTA PR, you likely need the combined quick wins to reach at least **val_bpb <= 1.1183** (0.005 nats below 1.1233). This is ambitious for hyperparameter-only changes and may require stacking with a more substantial technique (multi-token prediction, hyperconnections, etc.).

### Verification Checklist for Combined Run

1. Artifact size < 16,000,000 bytes
2. Training completes within 600 seconds
3. All 3 seeds produce valid results
4. Mean val_bpb < SOTA (1.1233)
5. For a submission PR: improvement >= 0.005 nats at p < 0.01

---

## Execution Priority

Run experiments in this order (most likely to yield improvement first):

1. **BigramHash 4096** (single seed) -- highest expected gain, env-var-only change
2. **Warmdown 4000 and 4500** (single seed each) -- fast to run, no code changes
3. **RoPE 24 dims** (single seed) -- zero size cost, env-var-only
4. **QK-Norm variants** (single seed) -- mostly informational since QK-Norm already present

After single-seed screening, run 3 seeds only for variants that beat the SOTA seed-1337 result of 1.1228.

---

## Quick Reference: SOTA Baseline Numbers

| Metric | Value |
|--------|-------|
| Mean val_bpb (3 seeds) | 1.1233 |
| Seed 1337 val_bpb | 1.1228 |
| Seed 42 val_bpb | 1.1236 |
| Seed 2024 val_bpb | 1.1236 |
| Std across seeds | 0.0005 |
| Artifact size | 15,555,017 bytes |
| Budget remaining | 444,983 bytes (~435 KB) |
| Total steps | ~7100 |
| Training time | ~600 seconds |

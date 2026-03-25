# OpenAI Parameter Golf -- Leaderboard Analysis

**Generated: 2026-03-24**

This document analyzes every submission to the Parameter Golf challenge (10min/16MB track), ordered by score. The challenge asks competitors to train the best language model that fits in a 16MB artifact and trains in under 10 minutes on 8xH100 GPUs. The metric is **bits-per-byte (BPB)** -- lower is better.

---

## Summary Table (Record Track -- 10min/16MB)

| Rank | Name | val_bpb | Author | Date | Key Techniques |
|------|------|---------|--------|------|----------------|
| 1 | 11L EMA + GPTQ-lite + warmdown3500 + QAT@0.15 | **1.1228** | Tianhao Wu (signalrush) | 03-22 | GPTQ-lite clip search, EMA+SWA, Late QAT, 11L, XSA4, Partial RoPE |
| 2 | 11L Partial RoPE + LN Scale + EMA + XSA4 | **1.1248** | Jack Princz (jfprincz) | 03-21 | Partial RoPE (16/64 dims), LN Scale, EMA, XSA4 |
| 3 | 11L XSA + EMA + Int6 MLP3x + WD=0.04 | **1.1271** | Jack Princz (jfprincz) | 03-20 | XSA on last 4 layers, EMA replacing SWA |
| 4 | 11L Efficient Partial XSA + FA3 + SWA/120 | **1.1307** | vadim borisov / unnir | 03-20 | Efficient Partial XSA, FlashAttention 3, SWA/120 |
| 5 | 10L Int5-MLP + BigramHash(10240) + SWA(frac=0.4) | **1.1428** | thwu1 | 03-20 | Mixed int5/int6, BigramHash 10240, SWA frac=0.4 |
| 6 | Int6 MLP3x + SmearGate + BigramHash + MuonWD + SWA | **1.1458** | Raahil Shah (raahilshah) | 03-20 | SmearGate, BigramHash, OrthoInit, SWA, WD=0.04 |
| 7 | 11L MLP3x + WD=0.04 + Int6 QAT + zstd-22 + Sliding Window | **1.1502** | aruniyer | 03-19 | 11L, MLP 3x, QAT int6, zstd-22, WD=0.04, U-Net skips |
| 8 | SmearGate + OrthoInit + Muon WD | **1.1556** | notapplica (assumed) | 03-19 | SmearGate, BigramHash, OrthoInit, Muon WD, MLP 3x |
| 9 | 10L Int6 QAT + Zstd MLP2.6x Muon0.99 Sliding Window | **1.1598** | yahya010 | 03-19 | 10L, STE int6 QAT (zero quant gap), zstd-22, MLP 2.6x, seq 2048 |
| 10 | int6 STE QAT + MLP + bigram + U-Net | **1.1598** | (unknown) | 03-19 | Similar stack, int6 STE QAT, sliding window |
| 11 | Mixed Quant (int6+int8) + Sliding Window | **1.1630** | aquariouseworkman | 03-19 | Mixed int6/int8 quant, MLP 3x, sliding window stride=64 |
| 12 | Int6 MLP3x Sliding Window (WarmdownQuantization) | **1.1574** | samuellarson | 03-19 | Int6 post-training quant, MLP 3x, FP16 embed, Late-K passthrough |
| 13 | Sliding Window + FP16 Embed + 10L + Muon WD + Overtone Init | **1.1748** | notapplica | 03-19 | Sliding window, FP16 embed, 10L, Muon WD, Overtone spectral init |
| 14 | LoRA TTT | **1.1929** | sam (samacqua) | 03-17 | Per-document LoRA test-time training at eval |
| 15 | Sliding Window Eval (stride=64) | **1.1925** | Matthew Li (mattqlf) | 03-19 | Sliding window eval only (pure eval strategy) |
| 16 | Training Opt Seq4096 | **1.2014** | Spokane Way | 03-19 | Seq 4096, Muon momentum 0.99, lower LR, 3/4 batch |
| 17 | Long Context Seq2048 v2 | **1.2058** | Spokane Way | 03-18 | Seq 2048, tuned LR |
| 18 | 10L Mixed Precision | **1.2147** | Nan Liu (nanlliu) | 03-19 | 10 layers, mixed int6/int8 compression |
| 19 | Warmdown-Quantization | **1.2154** | samuellarson | 03-19 | Warmdown=20000, FP16 embed, NTK-RoPE eval@1408 |
| 20 | FP16 Tied Embedding + LR/Warmdown Tuning | **1.2197** | Renier Velazco (chonchiog) | 03-18 | FP16 embedding, warmdown 3600, LR 0.06 |
| 21 | Lower LR | **1.2230** | Nan Liu (nanlliu) | 03-18 | LR sweep -- MATRIX_LR=0.02 optimal |
| 22 | Naive Baseline | **1.2244** | OpenAI (Baseline) | 03-17 | 9L, 512d, int8+zlib, seq 1024 |

### Non-Record Track

| Name | val_bpb | Author | Date | Notes |
|------|---------|--------|------|-------|
| 4-Hour Quasi-10B SP1024 | 1.2074 | Will DePue (williamd) | 03-18 | 4h training (unlimited compute), same baseline arch |
| SwiGLU + Warmdown Fix + Quarter Batch (1x5090) | 1.3281 | (anon) | 03-19 | Single RTX 5090, warmdown bug fix discovery |

---

## Detailed Analysis (Ordered by Score, Best First)

---

### #1 -- 11L EMA + GPTQ-lite + warmdown3500 + QAT@0.15 (val_bpb: 1.1228)

**Author:** Tianhao Wu (@signalrush) | **Date:** 2026-03-22 | **Artifact:** 15.55 MB

This is the current SOTA. It builds on a deep stack of prior innovations (PR #374) and adds two novel post-training optimizations.

**Novel contributions:**

1. **GPTQ-lite (per-row optimal clip percentile search):** Instead of using the full row range for int6 quantization, this tries 5 different clip percentiles (0.999, 0.9995, 0.9999, 0.99999, 1.0) for each weight row and picks whichever minimizes reconstruction MSE. Think of it as finding the best "zoom level" for quantization per row -- clipping a few outliers can dramatically reduce the average quantization error. This costs zero training time since it runs only at export. Contributed **-0.0006 BPB**.

2. **EMA weight averaging (decay=0.997):** An exponential moving average of the model weights is maintained every training step. EMA smooths out the noise in SGD updates by keeping a running weighted average where recent weights count more. This stacks on top of Tight SWA (Stochastic Weight Averaging that collects checkpoints late in training). Contributed **-0.0006 BPB**.

3. **Warmdown=3500** (up from 3000): Slightly longer learning rate decay phase. Contributed **-0.0002 BPB**.

4. **Late QAT threshold=0.15** (from 0.1): Starts fake-quantization earlier during training (when LR scale drops below 0.15 instead of 0.1). Contributed **-0.0001 BPB**.

**Inherited from prior entries:**
- 11 transformer layers with U-Net skip connections
- Partial XSA on last 4 layers (efficient GQA-aware implementation)
- Partial RoPE (16/64 dims), LN Scale (1/sqrt(l+1))
- Shared Value Embedding (dim=128, layers 9,10)
- SmearGate + BigramHash (2048 buckets)
- Int6 + zstd-22 compression, FP16 tied embeddings
- Muon optimizer (lr=0.025, momentum=0.99, WD=0.04)
- FlashAttention 3
- Sliding window eval (stride=64)

**Key metrics:** 7,101 steps in 600s (84ms/step), ~5.6B train tokens.

**Estimated BPB contributions of the full stack:**
| Component | Estimated Contribution |
|-----------|----------------------|
| Baseline | 1.2244 |
| Longer seq + lower LR + optimizer tuning | -0.020 |
| Sliding window eval (stride=64) | -0.033 |
| MLP 3x + int6 compression | -0.015 |
| SmearGate + BigramHash + OrthoInit | -0.005 |
| 11th layer + U-Net | -0.003 |
| XSA on last 4 layers | -0.002 |
| EMA + SWA | -0.003 |
| Partial RoPE + LN Scale | -0.003 |
| GPTQ-lite + warmdown tuning | -0.001 |
| Weight decay 0.04 | -0.002 |
| **Final** | **~1.1228** |

---

### #2 -- 11L Partial RoPE + LN Scale + EMA + XSA4 (val_bpb: 1.1248)

**Author:** Jack Princz (@jfprincz) | **Date:** 2026-03-21 | **Artifact:** 15.6 MB

Built on PR #287 (1.1271). Adds two zero-parameter architectural improvements.

**Novel contributions:**

1. **Partial RoPE (16 of 64 dims):** Standard RoPE (Rotary Position Embeddings) applies positional information to all head dimensions. This entry applies it to only 25% of dimensions (16 out of 64). The remaining 48 dimensions attend without positional bias, allowing the model to learn position-invariant patterns (e.g., "the" always follows "of" regardless of position). This is a zero-parameter change that improves generalization.

2. **LN Scale (1/sqrt(layer_idx+1)):** RMSNorm outputs in deeper layers are scaled down by 1/sqrt(layer_index + 1). Deeper layers tend to have larger activations; this damping stabilizes training and improves convergence. Zero parameters added.

**Note on Late QAT:** The code includes a Late QAT flag, but post-submission analysis revealed that `torch.compile` constant-folds the class attribute, so the STE branch never activates. The score is purely from Partial RoPE + LN Scale.

**Mean val_bpb:** 1.1250 (3 seeds, std=0.0005).

---

### #3 -- 11L XSA + EMA + Int6 MLP3x + WD=0.04 (val_bpb: 1.1271)

**Author:** Jack Princz (@jfprincz) | **Date:** 2026-03-20 | **Artifact:** 15.5 MB

Built on PR #198 (1.1318). Introduces two key innovations.

**Novel contributions:**

1. **Exclusive Self Attention (XSA) on last 4 layers:** Based on arXiv:2603.09078. After the standard attention output, XSA subtracts the component of the output that is aligned with the token's own value vector. In other words, it forces the attention mechanism to capture only information from *other* tokens rather than reinforcing what the token already "knows." The implementation uses an efficient GQA-aware reshape (no `repeat_interleave`) that adds only ~2ms/step overhead. Zero new parameters. Contributed **~-0.002 BPB**.

2. **EMA replacing SWA:** Instead of collecting periodic SWA checkpoints (discrete snapshots), an exponential moving average shadow model updates every single step with `ema = 0.997 * ema + 0.003 * param`. This produces smoother weight distributions that generalize better and compress more efficiently. Contributed **~-0.003 BPB**.

**Mean val_bpb:** 1.1280 (3 seeds). Best seed: 1.1271.

---

### #4 -- 11L Efficient Partial XSA + FA3 + SWA/120 (val_bpb: 1.1307)

**Author:** vadim borisov / unnir (tabularis.ai) | **Date:** 2026-03-20 | **Artifact:** 15.9 MB

**Novel contributions:**

1. **Efficient GQA-Aware XSA Implementation:** The standard XSA with Grouped Query Attention requires `repeat_interleave` to expand value vectors, doubling memory per layer. This entry uses a free reshape into KV head groups + broadcasting:
```python
# Free reshape + broadcast (zero allocation)
y_grouped = y.reshape(B, T, Hkv, group_size, D)
vn = normalize(v).unsqueeze(-2)
y = (y_grouped - dot(y_grouped, vn) * vn).reshape(B, T, H, D)
```
This reduces XSA overhead from ~7ms/step to ~2ms/step.

2. **Partial Application (last 3 of 11 layers):** The XSA paper shows self-attention bias increases across layers. Applying XSA only to the deepest layers targets the highest-bias layers while minimizing compute cost. Combined with the efficient implementation: **~-0.002 BPB**.

3. **SWA every 120 steps** (13 checkpoint average during warmdown).

**Also included:** FlashAttention 3 (Hopper-optimized), SmearGate + BigramHash (2048 buckets), 11 layers, int6 + zstd-22.

---

### #5 -- 10L Int5-MLP + BigramHash(10240) + SWA(frac=0.4) (val_bpb: 1.1428)

**Author:** thwu1 | **Date:** 2026-03-20 | **Artifact:** ~15.9 MB

Built on PR #162 (SmearGate/OrthoInit baseline at 1.1485). A precision and hyperparameter refinement pass.

**Novel contributions:**

1. **Mixed Int5/Int6 Quantization:**
   - **Int5 ([-16,15], 32 levels)** for MLP weights -- these are the least precision-sensitive and compress extremely well (1.88x zstd ratio)
   - **Int6 ([-32,31], 64 levels)** for attention weights -- more precision-sensitive
   - **FP16** for tied embeddings and last-layer key projections
   - The int5 MLP savings (~1.86MB vs uniform int6) fund a 10th transformer layer. Contributed **-0.003 BPB**.

2. **BigramHash with 10240 buckets** (up from 4096): Reduces token-pair hash collisions. Contributed **-0.0008 BPB** (vs 8192 buckets).

3. **SWA with start_frac=0.4:** Only averages checkpoints from the last 40% of warmdown (the most converged portion). Quality over quantity -- fewer but better checkpoints. Contributed **-0.0006 BPB**.

**Ablation data (from README):**
| Change | val_bpb | Delta |
|--------|---------|-------|
| 9L int6 (PR162 base) | 1.1485 | baseline |
| + int5 MLP + 10th layer | 1.1453 | -0.003 |
| + WD=0.04 + warmdown=3000 | 1.1452 | -0.0001 |
| + SWA_start_frac=0.4 | 1.1446 | -0.0006 |
| + bigram=10240 | **1.1426** | -0.0020 |

---

### #6 -- Int6 MLP3x + SmearGate + BigramHash + OrthoInit + Muon WD + SWA (val_bpb: 1.1458)

**Author:** Raahil Shah (@raahilshah) | **Date:** 2026-03-20 | **Artifact:** 15.86 MB

A comprehensive 7-technique stack on the 9-layer baseline. This is the first entry to combine SmearGate, BigramHash, Orthogonal Init, and SWA together.

**Techniques (all stacked):**

1. **Per-Row Int6 Quantization + zstd-22:** Quantize MLP and attention weights to int6 with per-row scaling. zstd level 22 provides ~5% better compression than zlib-9 on int6 data. Enables fitting more parameters in 16MB.

2. **3x MLP Expansion (hidden=1536):** The biggest single contributor. Wider MLPs allow more expressive nonlinear feature transformations. Enabled by the byte savings from int6 quantization.

3. **SmearGate:** A learned per-dimension gate (~512 params) that blends each token's embedding with the previous token's: `output = gate * current + (1-gate) * prev`. Initialized near-identity (gate~0.95). Provides free bigram context at the embedding layer.

4. **BigramHash Embedding (4096 buckets, dim=128):** A hash table mapping consecutive token pairs to learned embeddings via `(prev * 31 + curr) % 4096`. Adds ~524K parameters. Complements SmearGate with additive bigram features.

5. **Orthogonal Weight Initialization:** All large weight matrices initialized with `orthogonal_(gain=1.0)`, output projections scaled by `1/sqrt(2*num_layers)`. All singular values start at 1, giving uniform gradient flow. Accelerates early convergence -- critical when you only have ~7K steps.

6. **Muon Optimizer with Weight Decay (WD=0.04):** Decoupled weight decay keeps weights small and well-distributed, directly improving int6 quantization quality. WD=0.04 was optimal in a 0.01-0.05 sweep.

7. **Stochastic Weight Averaging (SWA every 50 steps):** Averages ~30 checkpoints from the last 50% of training. Produces smoother weight distributions that quantize better.

**Mean val_bpb:** 1.1458 (3 seeds, std=0.0008). Pre-quant: 1.1616. Quantization penalty: 0.016 bpb.

---

### #7 -- 11L MLP3x + WD=0.04 + Int6 QAT + zstd-22 + Sliding Window (val_bpb: 1.1502)

**Author:** aruniyer | **Date:** 2026-03-19 | **Artifact:** ~15.4 MB

The first 11-layer entry. Stacks many improvements including QAT.

**Key techniques:**

1. **11 transformer layers** (vs 9 baseline): More depth, funded by aggressive int6 compression.
2. **3x MLP expansion (hidden=1536):** More capacity per layer.
3. **Decoupled weight decay (0.04):** On both Muon and AdamW.
4. **QAT int6 (STE fake-quantize):** Simulates int6 noise during training. The model learns to be robust to quantization. Zero quant gap.
5. **zstd-22 compression:** Saves ~1.5MB vs zlib -- critical for fitting 11L MLP3x under 16MB.
6. **FP16 tied embedding export.**
7. **Sliding window eval (stride=64):** ~0.034 BPB free improvement.
8. **Higher Muon momentum (0.99)** with warmup from 0.92 over 1500 steps.
9. **U-Net skip connections** (5 encoder + 6 decoder layers).

**Multi-seed results:** Mean val_bpb = 1.1502 (3 seeds, std=0.00043). Sliding window eval takes ~88s on 8xH100.

---

### #8 -- SmearGate + OrthoInit + Muon WD (val_bpb: 1.1556)

**Author:** (submission.json shows no explicit author field; directory associated with "notapplica" based on Overtone Init entry) | **Date:** 2026-03-19 | **Artifact:** 15.9 MB

A 9-layer model that introduces several architectural innovations that became standard in later entries.

**Key techniques:**

1. **SmearGate:** Learned per-dimension gate blending current and previous token embeddings. Initialized via sigmoid(3.0) ~ 0.95 (near-identity). The model learns per-dimension how much bigram blending helps.

2. **Bigram Hash Embedding:** 4096-bucket hash table (dim=128, projected to 512). Maps token pairs via `(prev * 92821 + cur) % 4096`.

3. **Orthogonal Weight Initialization:** Accelerates convergence by ensuring uniform gradient flow at initialization.

4. **Muon Weight Decay (0.01):** Keeps weights smaller for better quantization.

5. **MLP 3x + Int6 STE QAT + zstd-22:** Wider MLP enabled by aggressive compression.

6. **U-Net Skip Connections:** Encoder-decoder skip connections giving the decoder direct access to earlier representations.

7. **Sliding Window Eval (stride=64).**

**Metrics:** Post-quant sliding val_bpb = 1.1556. Quantization gap: ~0.0001 BPB (STE QAT virtually eliminates it). 12,047 training steps in 600s.

---

### #9/#10 -- 10L Int6 QAT + Zstd MLP2.6x (val_bpb: 1.1598)

**Authors:** yahya010 (Seq2048_FP16Emb_TunedLR) and unknown (int6_STE_QAT_MLP_bigram_U_Net) | **Date:** 2026-03-19

These two entries achieved nearly identical scores (1.1598) using similar technique stacks.

**yahya010's entry** is a 10-layer model with:
- STE int6 QAT (zero quant gap)
- MLP hidden=1344 (2.625x), wider than baseline but not full 3x
- zstd-22 compression
- FP16 tied embedding, seq 2048
- Muon momentum 0.99, gradient clipping 0.3
- Sliding window eval stride=64
- ~8,300 steps at ~72ms/step (QAT adds ~28% overhead)

**The int6_STE_QAT entry** (only has train.log, no README/submission.json) achieved sliding window val_bpb = 1.1598 with a similar stack.

---

### #11 -- Mixed Quant (int6 blocks + int8 embeddings) + Sliding Window (val_bpb: 1.1630)

**Author:** aquariouseworkman | **Date:** 2026-03-19 | **Artifact:** 15.35 MB

Four orthogonal improvements over baseline, with an excellent improvement breakdown in the README.

**Techniques and their individual contributions:**

1. **Wider MLP 3x + seq1024 + 524K batch:** Contributed **-0.0294 BPB**. The largest single improvement from having more model capacity. Seq1024 (down from 4096) gives faster steps and more total training.

2. **Mixed quantization (int6 blocks, int8 embed):** Added only **+0.0015 BPB penalty**. Key insight: all CastedLinear weights get STE int6 training, but the embedding does not. Using int8 for embedding (which has no STE protection) and int6 for blocks (which do) reduces quant penalty from +0.048 to +0.0015 -- a 32x improvement.

3. **Sliding window (stride=64):** Contributed **-0.0335 BPB additional**. Every scored token gets 960+ tokens of context.

**Total improvement: -0.0614 BPB** over baseline.

---

### #12 -- Int6 MLP3x Sliding Window / WarmdownQuantization (val_bpb: 1.1574)

**Author:** samuellarson | **Date:** 2026-03-19/20

Two entries from the same author with the same submission.json (val_bpb: 1.1574). The WarmdownQuantization README focuses on quantization-aware training strategies:

**Key insights from the WarmdownQuantization README:**

1. **Always-decaying LR Schedule (WARMDOWN_ITERS=20000):** Setting warmdown far beyond actual steps means the entire training run is in the decay phase. LR decays linearly from 61% of peak to near-zero. This produces tighter weight distributions with fewer outliers, which quantize much better. Post-quant penalty drops from 0.014 to 0.005 BPB.

2. **FP16 Tied Embeddings:** Reduces remaining post-quant penalty from 0.005 to ~0.001 BPB.

3. **NTK-RoPE Extrapolation at eval@1408:** Well-trained models develop precise positional patterns. Moderate extrapolation (1.375x) is optimal; aggressive 2x extrapolation hurts.

4. **Optimizer-Warmdown Interaction:** MUON_BACKEND_STEPS=5 outperforms 7 when combined with aggressive warmdown, because smooth weights reduce the need for per-step gradient quality.

The final Int6 MLP3x version additionally uses int6 post-training quantization, 3x MLP expansion, and sliding window eval.

---

### #13 -- Sliding Window + FP16 Embed + 10L + Muon WD + Overtone Init (val_bpb: 1.1748)

**Author:** notapplica | **Date:** 2026-03-19 | **Artifact:** ~14.7 MB

**Key techniques:**

1. **Sliding window eval (stride=64):** Standard technique by this point.
2. **FP16 tied embedding export.**
3. **10 transformer layers** (from 9).
4. **Muon weight decay (0.02).**
5. **Overtone spectral embedding init:** SVD power-law spectrum shaping (S_k ~ k^{-0.5}). An unusual initialization approach that shapes the embedding matrix's singular value spectrum to follow a power law, mimicking the distribution seen in trained embeddings.
6. **Phase-transition residual mixing:** Sigmoid-scheduled `resid_mix` initialization.

**Mean val_bpb:** 1.1748 (3 seeds, std small). Eval time: ~162s.

---

### #14 -- LoRA TTT (val_bpb: 1.1929)

**Author:** sam (@samacqua) | **Date:** 2026-03-17

A creative approach that improves the score entirely at evaluation time through test-time training.

**How LoRA TTT works:**

For each validation document:
1. Find document boundaries using BOS tokens
2. Split into overlapping 256-token chunks within 1024-token context windows
3. Score each chunk (accumulate loss for BPB), *then* train rank-8 LoRA adapters on that chunk's loss
4. Reset LoRA parameters between documents (no cross-document leakage)

LoRA (Low-Rank Adaptation) adds small trainable matrices to existing weights. Here rank-8 LoRA targets `lm_head`, `c_q`, and `c_v` projections. A single Adam step per chunk with lr=0.01.

**Ablation data (very informative):**

| Condition | val_bpb | Delta |
|-----------|---------|-------|
| Baseline (cross-doc, flat stream) | 1.2278 | -- |
| + Doc-isolated eval | 1.2168 | -0.0110 |
| + Strided eval (chunk=256) | 1.1941 | -0.0337 |
| + LoRA TTT | 1.1910 | -0.0368 |

Most improvement comes from doc isolation and strided evaluation, not TTT itself (-0.003 BPB from LoRA). Still, this entry planted an important seed for eval-time techniques.

---

### #15 -- Sliding Window Eval (val_bpb: 1.1925)

**Author:** Matthew Li (@mattqlf) | **Date:** 2026-03-19 | **Artifact:** 15.87 MB

A pure eval-strategy submission. Training is identical to baseline.

**The sliding window technique explained:**

The baseline chops validation text into non-overlapping 1024-token chunks. The first token in each chunk has zero context; on average each token gets ~512 tokens of context.

Sliding window advances by only 64 tokens per window but runs the full 1024-token forward pass. Only the rightmost 64 tokens (with 960+ tokens of context) are scored. Every token gets near-maximum context.

**Result:** -0.0319 BPB improvement with zero artifact cost. Eval time increases from ~16s to ~70s (still well within the 10-minute eval budget).

This is arguably the highest "bang for buck" technique in the entire competition.

---

### #16 -- Training Opt Seq4096 (val_bpb: 1.2014)

**Author:** Spokane Way (@spokane-way) | **Date:** 2026-03-19 | **Artifact:** 15.87 MB

Combines long-context training with aggressive optimizer tuning.

**Key changes:**
- **Seq 4096:** 4x more context per training sequence. Costs ~71ms/step (vs ~43ms) but quality improvement outweighs fewer steps.
- **Muon momentum 0.99** (vs 0.95): Stronger gradient smoothing.
- **Lower LR (0.02):** Reduces int8 quantization penalty from 0.007+ to 0.0034 BPB.
- **3/4 batch (393K tokens):** More optimizer updates per second.
- **Extended momentum warmup (1500 steps from 0.92):** Prevents early instability.
- **Longer warmdown (3000 steps).**

**Net improvement: -0.023 BPB** over baseline. 8,394 steps at 71ms/step.

---

### #17 -- Long Context Seq2048 v2 (val_bpb: 1.2058)

**Author:** Spokane Way (@spokane-way) | **Date:** 2026-03-18 | **Artifact:** 15.87 MB

A straightforward improvement from longer training sequences.

**Changes:** TRAIN_SEQ_LEN=2048 with tuned learning rates (0.040/0.032/0.032). 11,564 steps at ~52ms/step.

Statistically verified: 3 seeds, mean 1.2064, std 0.00072, p=0.0005 vs baseline.

---

### #18 -- 10L Mixed Precision (val_bpb: 1.2147)

**Author:** Nan Liu (@nanlliu) | **Date:** 2026-03-19 | **Artifact:** 15.93 MB

**Key innovation: Layer-stratified quantization precision.**

The 10-layer model at dim=512 has 18.9M params -- 1.6MB over the 16MB cap with standard int8+zlib. Solution: use int6 (64 quantization levels) for middle layers (3-6) while keeping int8 (256 levels) for early and late layers.

| Layer Group | Precision | Rationale |
|-------------|-----------|-----------|
| Layers 0-2 (early) | int8 | Critical for input processing |
| Layers 3-6 (middle) | int6 | Less sensitive, saves ~1.6MB |
| Layers 7-9 (late) | int8 | Critical for output quality |

Quant gap: only 0.0018 BPB (vs baseline's 0.0093). Combined with lower LR (0.02) and the extra layer: -0.0097 BPB over baseline.

---

### #19 -- Warmdown-Quantization (val_bpb: 1.2154)

**Author:** samuellarson | **Date:** 2026-03-19

Focused on reducing the quantization penalty through training schedule manipulation. See entry #12 above for full technical details.

**Score: val_bpb = 1.2154** (improvement: -0.009 BPB over baseline).

---

### #20 -- FP16 Tied Embedding + LR/Warmdown Tuning (val_bpb: 1.2197)

**Author:** Renier Velazco (@chonchiog) | **Date:** 2026-03-18 | **Artifact:** 15.90 MB

**Key insight: The tied embedding is the most sensitive tensor to quantize.**

Since the embedding matrix serves as both input lookup AND output head, int8 errors compound through both paths. Keeping it in FP16 drops the post-quant BPB degradation from ~0.007 to ~0.0005. The tradeoff is ~500KB extra, offset by shrinking MLP hidden from 1024 to 992.

Also tuned: warmdown 3600 (from 1200), MATRIX_LR 0.06 (up from 0.04).

**Things tried that did not work:**
- SwiGLU: Better per-step quality but 45% slower, net negative
- Depth recurrence: Needs more steps than 10 min allows
- QAT: Overhead not worth the small gap reduction at this stage
- lzma compression: Worse than zlib for int8 data
- Higher embed LR (0.08): Hurt convergence

---

### #21 -- Lower LR (val_bpb: 1.2230)

**Author:** Nan Liu (@nanlliu) | **Date:** 2026-03-18 | **Artifact:** 15.85 MB

A systematic LR sweep proving the defaults were suboptimal.

| MATRIX_LR | val_bpb | Delta |
|-----------|---------|-------|
| 0.06 | 1.2445 | +0.0159 (much worse) |
| 0.04 (default) | 1.2286 | -- |
| 0.02 (optimal) | 1.2230 | -0.0056 |
| 0.015 | 1.2234 | -0.0052 |

No architecture changes. Just MATRIX_LR=0.02, SCALAR_LR=0.02, TIED_EMBED_LR=0.03.

---

### #22 -- Naive Baseline (val_bpb: 1.2244)

**Author:** OpenAI | **Date:** 2026-03-17 | **Artifact:** 15.86 MB

The starting point for all entries.

**Architecture:** 9 transformer layers, 512 model dim, 8 attention heads, 4 KV heads (GQA), 2x MLP expansion (hidden=1024), ReLU-squared activation, tied input/output embeddings, 1024 BPE vocab, sequence length 1024.

**Training:** Muon optimizer for matrices, AdamW for scalars/embeddings. 524,288 tokens per batch. Int8 quantization + zlib compression for export.

**Metrics:** 13,780 steps at 43.54ms/step. Pre-quant: 1.2172, post-quant: 1.2244 (quant gap: 0.0072).

---

## Non-Record Entries

### 4-Hour Quasi-10B SP1024 (val_bpb: 1.2074)

**Author:** Will DePue (@williamd) | **Date:** 2026-03-18

Same baseline architecture trained for 4 hours instead of 10 minutes. 329,430 steps (vs ~13,780 in 10 min). Pre-quant reached 1.1749 BPB, but post-quant int8+zlib roundtrip scored 1.2074 -- the quantization penalty (0.032 BPB) is enormous after long training because weights develop more outliers.

**Lesson:** Longer training without quantization-aware techniques hits diminishing returns. The quantization gap grows with training.

### SwiGLU + Warmdown Fix + Quarter Batch (1x5090) (val_bpb: 1.3281)

**Author:** Anonymous | **Date:** 2026-03-19

A systematic 10-experiment exploration on a single RTX 5090. Score is limited by hardware (3,773 steps vs ~13,780 on 8xH100).

**Key discovery: Warmdown schedule bug.** The stock config sets warmdown_iters=1200, but with a 600s cap the implied warmdown window exceeds total training time. The model never trains at full LR. Fix: time-fraction approach (warmdown_frac=0.2). Worth **-0.006 BPB**.

Other findings: SwiGLU -0.004, quarter batch -0.016, gradient accumulation -0.002. Layer recurrence was the worst result (+0.051).

---

## Key Takeaways

### 1. The Three Pillars of Improvement

The ~0.10 BPB gap from baseline (1.2244) to SOTA (1.1228) comes from three roughly equal pillars:

| Pillar | Estimated Contribution | Key Techniques |
|--------|----------------------|----------------|
| **Eval strategy** | ~0.033 BPB | Sliding window (stride=64) |
| **Model capacity** | ~0.035 BPB | MLP 3x, 11 layers, int6/int5 compression, zstd-22 |
| **Training optimization** | ~0.032 BPB | Lower LR, Muon WD, EMA/SWA, OrthoInit, SmearGate, XSA, Partial RoPE |

### 2. Sliding Window Eval Is the Single Biggest Free Win

The sliding window evaluation technique (stride=64) provides ~0.033 BPB improvement at zero artifact cost and zero training cost. It just requires more eval compute (~70-160s vs ~16s). Every competitive entry uses it.

### 3. Quantization Is the Hidden Bottleneck

The baseline loses 0.007 BPB to int8 quantization. Naive approaches to adding capacity (more layers, wider MLP) fail because they push the artifact over 16MB. The breakthrough was using *lower-precision* int6 quantization with **quantization-aware training (QAT)** to eliminate the quantization gap entirely, while the lower bits-per-weight enable cramming more parameters into 16MB. Key enabling techniques:

- **STE QAT (Straight-Through Estimator):** Fake-quantize weights in the forward pass during training; gradients flow through rounding as identity. The model learns to be robust to quantization.
- **FP16 tied embeddings:** The embedding is uniquely sensitive (dual input/output role). Keep it in FP16.
- **GPTQ-lite clip search:** Per-row optimal clipping at export time.
- **Weight decay:** Keeps weight distributions tight, improving quantization.
- **zstd-22 over zlib:** Better compression for int6 data, saving ~1.5MB.

### 4. Compression Enables Capacity

The progression from int8+zlib to int6+zstd-22 freed roughly 3-4MB in the artifact budget. This enabled:
- MLP expansion from 2x to 3x (biggest single quality gain)
- Adding 2 more transformer layers (9 -> 11)
- Adding BigramHash embedding tables

### 5. Small Architectural Innovations Stack

Many entries contribute small but additive improvements:
- **SmearGate** (~0.002 BPB): Cheap bigram context at embedding layer
- **BigramHash** (~0.002 BPB): Token-pair features via hash embeddings
- **Orthogonal Init** (~0.001 BPB): Better convergence with limited steps
- **XSA** (~0.002 BPB): Forces attention to learn from context, not self-reference
- **Partial RoPE** (~0.001 BPB): Position-invariant patterns in most dims
- **LN Scale** (~0.001 BPB): Stabilizes deep models
- **EMA/SWA** (~0.003 BPB): Smoother weights for better generalization and quantization

### 6. Weight Averaging Is Crucial

Both SWA (Stochastic Weight Averaging) and EMA (Exponential Moving Average) appear in all top-5 entries. They serve dual purposes:
- **Better generalization** (the averaging itself acts as regularization)
- **Better quantization** (smoother weight distributions quantize with less error)

### 7. The Optimizer Matters More Than Expected

The Muon optimizer with tuned hyperparameters makes a large difference:
- **Momentum 0.99** (with warmup from 0.92): Better gradient smoothing
- **Weight decay 0.04**: Regularizes for quantization
- **Lower LR (0.02-0.025):** The defaults were ~2x too high
- **Warmdown tuning (3000-3500 iters):** Proper LR scheduling critical for short runs

### 8. Building on Others' Work Is the Dominant Strategy

The top entries form clear lineages:
- **jfprincz lineage:** PR#70 -> PR#164 -> PR#198 -> PR#287 -> PR#374
- **thwu1** built on unnir's SmearGate/OrthoInit (PR#162)
- **signalrush** (current SOTA) built on jfprincz's PR#374

Each entry typically adds 1-2 novel contributions worth 0.001-0.005 BPB while inheriting 0.05-0.09 BPB of improvements from predecessors.

### 9. Negative Results Are Valuable

Several entries documented what did NOT work:
- **SwiGLU:** Better per-step quality but too slow for the 10-min budget
- **Layer recurrence (looping):** Halves effective steps, net negative
- **Depth recurrence:** Needs more steps than available
- **lzma compression:** Worse than zlib for int8 data
- **Higher embed LR (0.08):** Hurts convergence
- **Aggressive NTK-RoPE extrapolation:** Helps undertrained models, hurts well-trained ones

### 10. The Competition Is Approaching Diminishing Returns

The improvement rate is slowing: the first week saw -0.065 BPB improvement (1.2244 -> 1.1598), while the second week added only -0.037 BPB more (1.1598 -> 1.1228). The easiest gains (sliding window, lower LR, int6+MLP3x) are picked. Remaining improvements come from stacking many small innovations, each worth 0.001-0.003 BPB.

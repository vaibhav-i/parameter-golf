# Parameter Golf: April 5, 2026 Research Brief

*Synthesized from 4 parallel agent scans: PR #1327+ scan, wPoE+ATS deep-dive, arXiv search, Nacrith+SSE implementation details. Supersedes 2026-04-04 brief.*

---

## 1. Frontier Update

### New Best Scores (as of 2026-04-05)

| PR | Score | Technique | Legal? |
|---|---|---|---|
| #1350 | **1.0046 BPB** | L-BFGS Causal SLOT (context-only positions) | Yes |
| #1351 | **1.0807 BPB** | Discriminative TTT (per-block adaptive LR) | Yes |
| #1329 | 0.63614 BPB | Per-sample scored-position SLOT | Almost certainly illegal |
| #1218 | 1.09785 BPB | SP4096 base (Vocab4096 + MLP4x + WD=0.090) | Yes |

**Key shift**: The legal no-SLOT frontier is now ~1.08 BPB (PR #1351). The legal SLOT frontier is ~1.00 BPB (PR #1350). These are both substantially better than the April 4 estimates.

### Official SOTA
Still 1.1147 BPB (PR #1019, abaybektursun, 2026-03-25). No new merged record yet.

### Technique Waves (PRs #1327–#1360)

**Depth Recurrence**: 7+ PRs in the April 3–5 window. Layers 3/4/5 or 4/5 share MLP weights (virtual extra layers). Requires WD=0.095, MLR=0.022 for stability on WD=0.090 base stack. Confirmed gain: −0.005 to −0.010 BPB.

**MuonEq-R**: Row-normalize gradient before Newton-Schulz: `G = G / (G.norm(dim=-1, keepdim=True) + 1e-8)`. Now effectively standard in all competitive PRs. Universal ~−0.001 BPB.

**Polar Express Newton-Schulz** (PR #1344, #1298): 4 minimax-optimal NS steps ≈ 5 fixed-coefficient steps. Saves ~2ms/step → ~0.014 BPB. Combined with GPTQ damp=0.005: total ~−0.006 BPB.

**No PRs yet attempting**: Multi-layer delta SLOT, streaming log-bias (Nacrith), wPoE, SSE, BAQ quantization, or Adaptive Temperature Scaling. These remain unexplored white space.

---

## 2. Immediate Implementation Targets

### Tier 1 — Drop-in, Confirmed, Highest Priority

#### T1-A: Rebase on SP4096 Base Stack
- **Mechanism**: Vocab4096 + MLP 4x (instead of 3x) + WD=0.090 from PR #1218 (clarkkev). Synergy: larger vocab = richer token stats; MLP4x = more capacity per byte; WD=0.090 = weights compress more uniformly under GPTQ.
- **BPB Gain**: −0.013 vs old SOTA base (1.09785 vs 1.1147 BPB baseline)
- **Difficulty**: 2/5
- **Legal**: Yes
- **Code sketch**:
  ```bash
  VOCAB_SIZE=4096 MLP_MULT=4 WEIGHT_DECAY=0.090 torchrun --standalone --nproc_per_node=8 train_gpt.py
  ```

#### T1-B: MuonEq-R (Row-Normalize Gradient)
- **Mechanism**: Before Newton-Schulz in Muon optimizer, row-normalize: `G = G / (G.norm(dim=-1, keepdim=True) + 1e-8)`. Ensures each row has equal influence; prevents large-norm rows dominating the orthogonalization.
- **BPB Gain**: −0.001 BPB (universal, stacks with everything)
- **Difficulty**: 1/5
- **Legal**: Yes
- **Code sketch** (in Muon step, before NS call):
  ```python
  G = grad.clone()
  G = G / (G.norm(dim=-1, keepdim=True) + 1e-8)
  # then proceed with Newton-Schulz as before
  ```

#### T1-C: Polar Express Newton-Schulz + GPTQ damp=0.005
- **Mechanism**: Replace 5-step fixed-coefficient NS with 4-step minimax-optimal coefficients (PR #1344). Also lower GPTQ damping from default 0.01 to 0.005 for sharper Hessian estimates.
- **BPB Gain**: ~−0.006 BPB combined
- **Difficulty**: 2/5
- **Legal**: Yes
- **Note**: Saves ~2ms/step of training overhead → more training steps in 10 minutes.

#### T1-D: Pre-Quant TTT with Discriminative Per-Block LR (PR #1351)
- **Mechanism**: Before GPTQ quantization, run AdamW on bf16 EMA weights for 10 epochs on self-generated calibration data. Key: per-block adaptive learning rate. Early blocks (index 0–3): 0.3× base LR. Later blocks (index 7–10): 1.0× base LR. Linear interpolation for middle blocks. Rationale: early layers are more general/stable; later layers are more task-specific and need stronger adaptation.
- **BPB Gain**: −0.010 vs flat-LR TTT; net ~−0.022 BPB vs no TTT (from April 4 data)
- **Difficulty**: 3/5
- **Legal**: Yes (pre-quant, no GPTQ weight destruction issue)
- **Code sketch**:
  ```python
  def get_block_lr(block_idx, n_blocks, base_lr):
      t = block_idx / (n_blocks - 1)  # 0.0 to 1.0
      scale = 0.3 + 0.7 * t           # 0.3 to 1.0
      return base_lr * scale

  # Before GPTQ quantization, after EMA weights are finalized:
  ema_state_dict = get_ema_weights(model)
  ttt_model = load_weights(ema_state_dict)
  for block_idx, block in enumerate(ttt_model.transformer.h):
      lr = get_block_lr(block_idx, n_blocks=11, base_lr=1e-5)
      optimizer = torch.optim.AdamW(block.parameters(), lr=lr)
      # run 10 epochs of self-gen calibration
      for epoch in range(10):
          calib_tokens = generate_calib_data(ttt_model, n_tokens=512)
          loss = forward_loss(ttt_model, calib_tokens)
          loss.backward(); optimizer.step(); optimizer.zero_grad()
  quantize_with_gptq(ttt_model)
  ```

#### T1-E: Causal SLOT with L-BFGS (PR #1350)
- **Mechanism**: Optimize a delta vector applied to context-only (non-scored) positions. L-BFGS optimizer (max_iter=25, history_size=20) instead of AdamW. Focal loss weighting on last 128 tokens/window. Warm-start deltas across windows. Delta clamp ±5.
- **BPB Gain**: 1.0046 BPB total (−0.087 BPB from SLOT component alone relative to ~1.09 base)
- **Difficulty**: 4/5
- **Legal**: Yes (context-only positions, no scored-position causality violation)
- **Note**: L-BFGS dramatically outperforms AdamW for this optimization landscape due to second-order curvature information.
- **Code sketch**:
  ```python
  # In eval loop, per window:
  delta = torch.zeros(ctx_len, d_model, requires_grad=True, device=device)
  if warm_start and prev_delta is not None:
      delta.data = prev_delta.data.clone()

  def closure():
      optimizer.zero_grad()
      h = context_hidden + delta  # apply delta to context positions only
      logits = model.forward_from_hidden(h, scored_positions)
      # focal loss weighting on last 128 tokens
      weights = torch.ones(scored_len)
      weights[-128:] *= 2.0
      loss = (F.cross_entropy(logits, targets, reduction='none') * weights).mean()
      loss.backward()
      return loss

  optimizer = torch.optim.LBFGS([delta], max_iter=25, history_size=20, lr=1.0)
  for _ in range(3):  # outer iterations
      optimizer.step(closure)
      delta.data.clamp_(-5.0, 5.0)

  prev_delta = delta.detach().clone()
  ```

---

### Tier 2 — Moderate Difficulty, Strong Payoff

#### T2-A: Streaming Log-Bias (Nacrith, arXiv:2602.19626)
- **Mechanism**: Maintain bias vector b ∈ R^V (vocab size), initialized to zeros. After each token position t with true token t*:
  ```
  b_t -= α · (p̃(t) − one_hot(t*))
  ```
  where `p̃(t) = softmax(log_p_llm + b)`. Equivalent to: `b += α · (one_hot(x_t) - softmax(log_p_llm + b))`. Applied as: `p̃ = softmax(log_p_llm + b)`. Reset b to zeros at each document boundary.
- **Exact params**: α = 0.001, float64 required (float32 loses symmetry), reset per document
- **BPB Gain**: −0.015 BPB (from Nacrith paper Table 7 ablation)
- **Difficulty**: 3/5 (requires refactoring eval to sequential per-token mode)
- **Legal**: Yes (causal — b_t is updated only from tokens before position t)
- **Implementation blocker**: Current SOTA eval loop (`eval_val_sliding`) is batched across windows, NOT sequential per-token. Needs architectural refactoring to call `forward_logits` one position at a time.
- **Code sketch**:
  ```python
  def eval_with_log_bias(model, tokens, alpha=0.001):
      b = torch.zeros(vocab_size, dtype=torch.float64, device=device)
      total_loss = 0.0
      for t in range(1, len(tokens)):
          logits = model.forward_single(tokens[:t])[-1]  # last position
          log_p = F.log_softmax(logits.double(), dim=-1)
          p_tilde = F.softmax(log_p + b, dim=-1)
          loss = -(log_p + b)[tokens[t]] + torch.log(p_tilde.sum())
          total_loss += loss.item()
          # update bias
          one_hot = torch.zeros_like(b); one_hot[tokens[t]] = 1.0
          b -= alpha * (p_tilde - one_hot)
          # reset at doc boundary if needed
      return total_loss / (len(tokens) - 1)
  ```
- **Concern**: O(N) forward passes (one per token) vs. O(N/window_size) in batched mode. Will be much slower. May need to batch with KV cache.

#### T2-B: Depth Recurrence (Virtual Layers)
- **Mechanism**: Layers 3,4,5 share MLP weights (3 real layers, each used twice = 6 virtual MLP applications). Attention weights remain distinct. Required hyperparameter adjustments: WD=0.095 (up from 0.090), MLR=0.022 (tune carefully).
- **BPB Gain**: −0.005 to −0.010 BPB
- **Difficulty**: 3/5
- **Legal**: Yes
- **Note**: Failed on old stack (pre-WD=0.090). The WD=0.090 base stack enables this. 7+ PRs exploring this in April 3–5 alone.
- **Code sketch**:
  ```python
  # In Block forward pass — share MLP between specific layer indices:
  self.recurrent_layers = [3, 4, 5]
  # During training, layer 3's MLP is reused for layers 4 and 5:
  if layer_idx in recurrent_layers[1:]:
      mlp_output = recurrent_layers[0].mlp(x)  # reuse layer 3's MLP
  else:
      mlp_output = self.mlp(x)
  ```

#### T2-C: BAQ Quantization (arXiv:2506.05664)
- **Mechanism**: Bit Allocation Quantization — convex optimization for layer-wise bitwidth allocation using Hessian-derived sensitivity scores. Allows some layers to use int5, others int7, optimizing total bits under a budget constraint. "Up to 56× lower perplexity at same bitwidth vs GPTQ" on 125M models.
- **BPB Gain**: Unknown (never tried in golf context). Conservative estimate: −0.005 to −0.015 BPB if some layers can go to int5 and fund extra architecture capacity.
- **Difficulty**: 4/5
- **Legal**: Yes
- **Note**: Tested only on 125M parameter models. Our model is ~17M. Scaling behavior unknown. The 56× perplexity claim is extraordinary — likely regime-specific. Real gain probably much smaller.

#### T2-D: N-Gram Agreement via Exponential Tilting (PR #1302)
- **Mechanism**: Compute a simple n-gram model over the current context window. Use exponential tilting to interpolate: `p_final ∝ p_llm(x) · q_ngram(x)^λ`. Lambda tuned per-evaluation via line search.
- **BPB Gain**: −0.003 BPB (from PR #1302 results)
- **Difficulty**: 3/5
- **Legal**: Yes

---

### Tier 3 — Exploratory, Higher Uncertainty

#### T3-A: wPoE (Weighted Product of Experts, arXiv:2511.10660)
- **Mechanism**: Neural LM × classical component, with learned mixture weight α: `π_α(a) ∝ q(a)^α · p_θ(a)^{1-α}`. Classical component is NOT PPM — it's **Naive Bayes unigram counting**: `q(a|X_{<n}) = (count(a in X_{<n}) + 1) / (n - 1 + D)` (Laplace-smoothed frequency). Alpha optimized via L-BFGS (10 iterations) on held-out context.
- **BPB Gain**: ~0.1% on enwik-style text. For strong models on FineWeb, probably 0.001–0.003 BPB. May be negligible.
- **Difficulty**: 2/5 (simple to implement)
- **Legal**: Yes
- **Critical caveat**: The paper's classical component provides weak signal for strong neural LMs. The gain shrinks as the neural LM improves. At 1.08 BPB quality, wPoE classical signal is nearly redundant. Consider only if implementation is trivial and gains stack additively.
- **Code sketch**:
  ```python
  def wPoE_logits(logits, token_history, alpha):
      # Compute unigram frequencies from context
      counts = torch.zeros(vocab_size, device=device)
      for t in token_history:
          counts[t] += 1
      q = (counts + 1) / (len(token_history) + vocab_size)  # Laplace smoothing
      log_q = torch.log(q)
      log_p = F.log_softmax(logits, dim=-1)
      # Combine: log π ∝ α*log_q + (1-α)*log_p
      combined = alpha * log_q + (1 - alpha) * log_p
      return combined  # renormalize at eval time
  ```

#### T3-B: Adaptive Temperature Scaling (arXiv:2409.19817)
- **Mechanism**: Post-hoc causal transformer: a small linear head that takes the last hidden state and predicts per-token temperature τ. Applied as `calibrated_logits = logits / exp(τ)`. Trained on held-out data to minimize ECE (Expected Calibration Error).
- **BPB Gain**: Uncertain. Paper targets ECE, not NLL/BPB. If calibration is imperfect at eval time, correcting it helps BPB. But highly-tuned models are already well-calibrated. Conservative estimate: 0–0.002 BPB.
- **Difficulty**: 3/5
- **Legal**: Yes
- **Note**: Inference overhead +8ms for 7B model. Negligible for 17M model. The head adds ~0.003 MB to artifact size (trivial). Main risk is zero BPB gain.

#### T3-C: Multi-Layer Delta SLOT (Unexplored)
- **Mechanism**: Extension of causal SLOT: apply delta vectors at ALL 11 layer hidden states (not just input). Creates 11× more optimization parameters. Potentially 11× stronger adaptation signal.
- **BPB Gain**: Estimate −0.05 to −0.15 BPB (very uncertain — no prior art). If each layer provides independent signal, gains could be additive. If correlated, gains saturate quickly.
- **Difficulty**: 5/5
- **Legal**: Yes (if restricted to context-only positions at all layers)
- **Risk**: High implementation complexity. May cause optimization instability with L-BFGS across 11 layers simultaneously.
- **Note**: Zero PRs attempting this in the April 3–5 sweep. First mover advantage.

#### T3-D: IDTS / Entropy-Adaptive Temperature
- **Mechanism**: During eval, compute entropy of the model's predictive distribution. High-entropy positions → increase temperature slightly (model is uncertain, scale down confidence). Low-entropy positions → decrease temperature (model is confident, sharpen predictions). 3-line change, zero extra parameters.
- **BPB Gain**: Estimate 0.001–0.005 BPB (very uncertain)
- **Difficulty**: 1/5
- **Legal**: Yes
- **Code sketch**:
  ```python
  entropy = -(p * log_p).sum(dim=-1, keepdim=True)  # [bsz, seq_len, 1]
  temp = 1.0 + 0.1 * (entropy - entropy.mean()) / entropy.std()
  calibrated_logits = logits / temp
  ```

---

## 3. New Technique Details

### 3.1 Discriminative TTT / Per-Block Adaptive LR (PR #1351)

**What's new vs. prior TTT**: Previous pre-quant TTT used flat learning rate across all blocks. PR #1351 discovered that early blocks (embedding + initial attention) need low LR for stability, while later blocks (final attention + MLP) benefit from aggressive adaptation to calibration tokens.

**Full parameter set** (from PR #1351):
- Epochs: 10
- Base LR: 1e-5
- Block LR schedule: `lr_i = base_lr × (0.3 + 0.7 × i / (n_blocks - 1))`
- All blocks trainable (including embedding, LN parameters)
- Calibration data: self-generated from EMA model (AR sampling, no external data)
- Calibration length: 512 tokens × 8 sequences = 4096 total tokens

**Result**: 1.0807 BPB, best legal no-SLOT score as of April 5.

**Why it works**: GPTQ's Hessian-based weight selection is most accurate when weights reflect the task distribution. Later blocks, being more task-specific, diverge more from random initialization and benefit most from task-specific pre-adaptation. Early blocks are more universal (positional encodings, basic syntactic features) and are destabilized by aggressive updates.

**Interaction with GPTQ**: Pre-quant TTT is safe because GPTQ runs AFTER the TTT update. The compensatory weight structure that GPTQ creates is built on the TTT-adapted weights. SGD TTT post-GPTQ is catastrophic (PR #1341: +0.030 BPB) because gradient updates destroy GPTQ's carefully constructed error-compensation chains.

---

### 3.2 Streaming Log-Bias (Nacrith, arXiv:2602.19626)

**Full algorithm** (Algorithm 1 from paper):

```
Input: LLM p_θ, token sequence X = [x_1, ..., x_N], α=0.001
Output: per-token losses

Initialize: b ← 0^V  (V = vocab size, float64)
For each document boundary: reset b ← 0^V

For t = 1 to N-1:
    log_p ← log p_θ(·|x_{1:t})           # LLM logits, log-prob
    log_p_tilde ← log_p + b               # apply bias
    p_tilde ← softmax(log_p_tilde)        # renormalize
    loss_t ← -log_p_tilde[x_{t+1}] + log(Σ exp(log_p_tilde))
           = -log p_tilde[x_{t+1}]        # cross-entropy under tilted dist
    b ← b - α · (p_tilde - e_{x_{t+1}})  # SGD update toward true token
```

**Key properties**:
- b is a running correction that tilts the LLM toward tokens that actually appeared
- Update `b[k] -= α*(p_tilde[k] - I[k==x_t])`: increases b for true token, decreases for all others
- Over time, b converges to correct systematic biases in the LLM
- Reset at document boundaries prevents cross-contamination
- float64 required: the update is antisymmetric (sum of updates = 0) — float32 accumulates rounding error that breaks this property

**Ablation results** (Table 7 from paper):
- Baseline LLM alone: reference
- + streaming log-bias: −0.015 BPB
- Different α values tested: α=0.001 optimal; α=0.01 slightly worse; α=0.0001 too slow to converge

**Implementation blocker for golf**: The SOTA eval loop (`eval_val_sliding` in the SOTA `train_gpt.py`) processes batched windows in parallel. To use streaming log-bias, we need sequential per-token processing. Options:
1. Rewrite eval to sequential mode with KV cache (slow but correct)
2. Maintain per-sequence bias vectors and apply batch-wise (approximation — loses exact causality within a window)
3. Apply only between windows, not within (coarser approximation but vectorizable)

Recommended approach: Start with option 3 (between-window updates only), measure gain, then implement option 1 if justified.

---

### 3.3 wPoE (Weighted Product of Experts, arXiv:2511.10660)

**What we learned from the paper** (not assumptions from April 4 brief):

The classical component is NOT PPM (Prediction by Partial Matching) — it's simple unigram counting with Laplace smoothing:

```
q(a | X_{<n}) = (Σ_{k<n} I(X_k = a) + 1) / (n - 1 + |V|)
```

This is just the empirical frequency of token a in the current document prefix, smoothed with a uniform Dirichlet prior. No context-conditioning, no suffix matching, no trie structure.

**Mixture formula**:
```
π_α(a | X_{<n}) ∝ q(a | X_{<n})^α · p_θ(a | X_{<n})^{1-α}
```

Alpha is a scalar optimized per-evaluation window via L-BFGS (10 iterations). Alpha typically converges to small values (0.05–0.15 for strong models).

**Why this is weaker than expected**:
- Unigram q(a) contains no context information beyond "token a appeared N times in this document"
- For strong LMs already using context, q provides minimal additional signal
- The PPM assumption in the April 4 brief was incorrect

**Revised BPB estimate**: 0.001–0.003 BPB for a model already at 1.08 BPB quality. Not worth significant implementation effort unless trivial to add.

**When wPoE helps more**: On repetitive text (code, structured documents, text with repeated entities) where unigram frequency genuinely correlates with future occurrence probability.

---

### 3.4 Adaptive Temperature Scaling (arXiv:2409.19817)

**Full mechanism**:
1. Train a small per-token temperature head: `h_temp = Linear(d_model, 1)` (2KB parameters)
2. During eval: `τ_t = h_temp(last_hidden_state_t)` — scalar per token
3. Apply: `calibrated_logits = logits / exp(τ_t)` (equiv. to softmax with adaptive temperature)
4. Head trained on held-out calibration set to minimize ECE (Expected Calibration Error)

**Paper results**:
- Primary metric: ECE reduction (calibration quality), not NLL/BPB
- Secondary finding: slight NLL improvement on calibration set (~0.5%)
- Inference overhead: +8ms for 7B model → negligible for 17M model
- Artifact cost: ~0.003 MB (trivial)

**Uncertainty for golf**:
- Paper shows ECE improvement, which does not directly imply BPB improvement
- For BPB to improve, the model must be systematically overconfident or underconfident in ways that the temperature head can detect
- GPTQ quantization may introduce new miscalibration that ATS can correct post-hoc
- Recommended: only try after exhausting confirmed gains

---

### 3.5 SSE (Secondary Symbol Estimation)

**Background**: SSE is a technique from PAQ-family lossless compressors (cmix, PAQ8, ZPAQ). After a mixer combines multiple model predictions, SSE provides a final non-linear calibration layer.

**Mechanism from cmix source**:
```python
p_final = sse.Predict(p_mixed)
```
Where `sse.Predict()` is a table-based lookup: for each probability value p (discretized to ~1024 buckets) and a small context hash, the table stores the empirically observed frequency of the event. The table is updated online as tokens are consumed.

**Conceptual equivalent**:
```python
# Initialize table: P_table[bucket][context_hash] = 0.5
# For each token:
bucket = int(p_model * 1024)
context = hash(last_few_tokens) % num_contexts
p_corrected = P_table[bucket][context]  # lookup
# Update table:
P_table[bucket][context] += lr * (I[event_occurred] - P_table[bucket][context])
```

**Why this works**: The neural LM may have systematic biases (e.g., consistently overestimates probability of punctuation at certain context types). SSE learns these biases from the streaming data and corrects them online.

**Implementation caveat**: PAQ source code is not publicly available (mattmahoney.net URLs returning 404 as of April 5). cmix is on GitHub but the SSE implementation parameters (table size, context hash width, update rate) require empirical tuning.

**Recommended approach**: Implement a simplified version with 512 buckets × 64 context hashes, online update rate 0.01. Measure BPB gain. If positive, expand table.

---

### 3.6 Multi-Layer Delta SLOT (Unexplored, Novel)

**Concept**: In standard causal SLOT (PR #1350), a delta vector is applied to context positions in the input embedding space. Multi-layer SLOT would apply separate delta vectors at ALL 11 layer hidden states simultaneously.

**Motivation**: Each layer represents progressively more abstract features. A delta at layer 0 (token embeddings) can only influence predictions through the entire forward pass. A delta at layer 10 (just before LM head) can make very precise, targeted corrections. By optimizing all 11 layers jointly, the optimizer has 11× more degrees of freedom to correct systematic biases.

**Expected architecture**:
```python
deltas = [torch.zeros(ctx_len, d_model, requires_grad=True) for _ in range(n_layers)]

def forward_with_multi_delta(tokens, deltas):
    x = embed(tokens)
    x += deltas[0]  # apply delta at embedding level
    for i, block in enumerate(transformer_blocks):
        x = block(x)
        if i < n_layers - 1:
            x += deltas[i+1]  # apply delta after each block
    return lm_head(x)
```

**L-BFGS joint optimization**: All 11 delta tensors in a single L-BFGS call. Parameter count = 11 × ctx_len × d_model = 11 × 512 × 512 = ~2.9M scalars per window. L-BFGS handles this in 25 iterations.

**Risk assessment**:
- High memory overhead per window
- Optimization landscape may be ill-conditioned across 11 layers
- No validated prior art in this specific form
- Could be the biggest untried technique in the competition

---

## 4. Updated Dead Ends

### Confirmed Dead (do not re-attempt)

| Technique | Why Dead | Source |
|---|---|---|
| Post-quant SGD/AdamW TTT | Destroys GPTQ compensatory weight structure, +0.030 BPB | PR #1341 |
| Standard scored-position SLOT | 100% causal violation rate | PR #1240; SOTA holder self-removed |
| kNN-LM | Hurts strong models, worse BPB | PR #1259 |
| Mixed int4/int5/int8 quantization | All worse than uniform int6 | Multiple PRs |
| QAT (Quantization-Aware Training) | Dead code — torch.compile constant-folds the flag | PR #1032 |
| N-gram blending unnormalized | Was measurement artifact | PR #1272 |
| Depth recurrence on OLD stack | Works on WD=0.090 stack only | Multiple PRs |
| VRL | Disabled as "risky" by SOTA author | PR #1019 notes |
| KD (Knowledge Distillation) | 11ms/step overhead kills it | PR #1029 |
| Full-model SGD TTT (v9 experiment) | Neutral to negative on SOTA stack | 25 attempts |

### New Dead Ends (discovered April 3–5)

| Technique | Why Dead | Source |
|---|---|---|
| Sub-1.0 BPB scored-position SLOT claims (0.63–0.94 BPB) | Almost certainly illegal; causal violation via scored positions; may also be below Shannon floor | PRs #1329, multiple sub-1.0 claims |
| LaCT (Learning as Context) | Requires long context that blows artifact budget | arXiv search; confirmed not applicable |
| FOEM (First-Order Error Compensation) | Designed for 3-bit quantization; we use int6 where GPTQ already good | arXiv:2507.11017 |

### Uncertain / Needs Investigation

| Technique | Uncertainty |
|---|---|
| BAQ quantization | 125M model results; unclear if applicable to 17M golf model |
| wPoE with PPM (not unigram) | PPM classical component would be stronger; unigram version has small gains |
| ATS (Adaptive Temperature Scaling) | ECE calibration benefit, not guaranteed BPB improvement |
| SSE | No verified source code; implementation requires empirical tuning |

---

## 5. Recommended Experiment Order

### Phase 1: Foundation (Days 1–2, Low Risk)

1. **Rebase on SP4096** (T1-A): Vocab4096 + MLP4x + WD=0.090. Immediate free gain, required for all subsequent experiments.
2. **MuonEq-R** (T1-B): 1-line change, stacks with everything, universal gain.
3. **Polar Express NS + GPTQ damp=0.005** (T1-C): 2-line change in NS function, +flag for GPTQ.
4. **Depth Recurrence** (T2-B): Reuse MLP layers 3/4/5, tune WD=0.095, MLR=0.022.

Combined expected gain from Phase 1: ~−0.020 to −0.025 BPB from 1.1147 baseline → ~1.09 BPB.

### Phase 2: High-Value Additions (Days 3–5, Medium Effort)

5. **Pre-quant Discriminative TTT** (T1-D): Per-block adaptive LR, 10 epochs. Biggest single technique gain.
6. **Causal SLOT with L-BFGS** (T1-E): Implement context-only SLOT, L-BFGS optimizer, focal loss, warm-start.
7. **N-gram agreement** (T2-D): Exponential tilting with self-computed n-gram model.

Combined expected gain from Phase 2 on top of Phase 1: ~−0.030 to −0.040 BPB → ~1.05 BPB.

### Phase 3: Novel / Exploratory (Days 6–10, Higher Risk)

8. **Streaming log-bias** (T2-A): Requires eval loop refactoring. Start with between-window version.
9. **Multi-layer delta SLOT** (T3-C): Novel architecture, no prior art. High ceiling, high variance.
10. **BAQ quantization** (T2-C): Test on SP4096 base, measure quality vs. uniform int6.
11. **IDTS** (T3-D): 3-line change, extremely low effort, worth a quick test.
12. **wPoE** (T3-A): Implement unigram version, measure if gain is real.

Expected from Phase 3: uncertain, potentially −0.010 to −0.050 additional BPB.

### Triage Rule
- Any technique costing >1ms/step average overhead is automatically deprioritized (1ms = ~0.007 BPB penalty from fewer steps in 10 minutes).
- All eval-time techniques (SLOT, streaming log-bias, wPoE, ATS, SSE) have zero training overhead — they only affect eval speed, not training BPB.

---

## 6. Conservative BPB Path

Starting from official SOTA: **1.1147 BPB**

| Step | Technique | Expected BPB | Delta |
|---|---|---|---|
| Baseline SOTA | PR #1019 | 1.1147 | — |
| + SP4096 base rebase | Vocab4096+MLP4x+WD=0.090 | 1.0980 | −0.017 |
| + MuonEq-R | Row-normalize gradient | 1.0970 | −0.001 |
| + Polar Express NS + GPTQ damp | 4-step NS + damp=0.005 | 1.0910 | −0.006 |
| + Depth Recurrence | MLP layer sharing (3/4/5) | 1.0860 | −0.005 |
| + Pre-quant Discriminative TTT | Per-block adaptive LR, 10 epochs | 1.0640 | −0.022 |
| + N-gram agreement | Exponential tilting | 1.0610 | −0.003 |
| + Causal SLOT (L-BFGS) | Context-only positions, 25 iter | 1.0040 | −0.057 |

**Conservative total: ~1.004 BPB** — well clear of the 1.1147 SOTA, and approaching (but not quite reaching) 1.0 BPB without any novel techniques.

**With streaming log-bias added**: ~0.989 BPB (if implementation succeeds and −0.015 BPB gain holds on our model/data).

**With multi-layer SLOT or other novel techniques**: Unknown, potentially sub-0.95 BPB.

### Risk-Adjusted Estimate
- **80% confidence floor**: 1.02 BPB (if causal SLOT works but some techniques underperform)
- **50% confidence target**: 1.004 BPB (all confirmed techniques work as expected)
- **20% upside**: 0.97–0.99 BPB (streaming log-bias + multi-layer SLOT both work)

### BPB Gap to Targets
- Beat current SOTA (1.1147) by ≥0.005 nats: **easily achievable with just Phase 1**
- Break 1.10 barrier: **Phase 1 alone should suffice**
- Break 1.05 barrier: **Phase 1 + 2 required**
- Break 1.00 barrier: **Phase 1 + 2 + Causal SLOT required**
- Break 0.99 barrier: **Phase 1 + 2 + Causal SLOT + Streaming Log-Bias required**

---

## Appendix: Key PR Reference Table

| PR | Author | Score | Key Technique | Status |
|---|---|---|---|---|
| #1350 | — | 1.0046 BPB | L-BFGS Causal SLOT | Legal |
| #1351 | — | 1.0807 BPB | Discriminative TTT (per-block LR) | Legal |
| #1329 | — | 0.63614 BPB | Per-sample SLOT (scored positions) | Illegal |
| #1218 | clarkkev | 1.09785 BPB | SP4096 base (Vocab4096+MLP4x+WD=0.090) | Legal |
| #1344 | — | — | Polar Express NS | Legal |
| #1302 | — | — | N-gram exponential tilting | Legal |
| #1341 | — | — | GPTQ+TTT incompatibility analysis | Analysis |
| #1240 | — | — | SLOT causality violation proof | Analysis |
| #1019 | abaybektursun | 1.1147 BPB | Full SOTA stack (current record) | Legal |

---

*Next update: After Phase 1 experiments complete on RunPod H100.*

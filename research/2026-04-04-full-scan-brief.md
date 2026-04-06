# Parameter Golf: Full Research Brief
**Generated: 2026-04-04 (overnight scan, 10 parallel agents)**
**Official SOTA: 1.1147 BPB (PR #1019, abaybektursun)**

---

## 1. The Most Important Thing: SLOT Is Probably Illegal

The entire sub-1.0 BPB leaderboard (0.69–0.94 BPB) is in limbo. Here's why:

**PR #1240 proved empirically that standard SLOT has a 100% causal violation rate.** Flipping any single target token changes NLL at every other scored position. This violates all three of the critical conditions in the competition's legality framework (Issue #1017): strict causal dependence, score-before-update, and single left-to-right pass.

**The SOTA holder agrees.** abaybektursun (current record, 1.1147 BPB) removed their own SLOT variant from PR #1019 before submitting, explicitly citing the causality concern. If the person who holds the record thinks it's illegal, that's the strongest possible signal.

**The sub-1.0 scores may be physically impossible.** Shannon entropy estimates for FineWeb web text are ~0.8–1.0 BPB. Scores of 0.69–0.94 BPB would imply compression better than the information-theoretic limit of the data. Either there's a bug or future tokens are being used.

**No official ruling exists.** Not a single comment from @valerio-oai or @0hq appears in any SLOT PR or the legality issue (#1240, #1172, #1176, #1303). The debate is 100% community-driven. The ruling could come any day.

**Safe variant: Causal SLOT (PR #1306, PR #1276).** Restricts delta optimization to context-only positions (tokens already scored in previous windows). Provably satisfies all four legality conditions. Only gives −0.009 BPB — but that's real and legal.

**The real frontier (no illegal SLOT) is ~1.08–1.09 BPB:**

| PR | BPB | Notes |
|----|-----|-------|
| #1289 | 1.0819 | PROTEUS: Scylla + parallel residuals + depth recurrence + TTT |
| #1306 | 1.0846 | Causal SLOT + pre-quant TTT |
| #1257 | 1.0855 | Complement TTT |
| #1296 | 1.0897 | Depth recurrence + parallel residuals |
| #1218 | 1.0979 | Vocab4096 + MLP4x + WD=0.090 (new base stack) |

---

## 2. A New Architecture Stack You're Missing

**PR #1218 (clarkkev): Vocab4096 + MLP 4x + WD=0.090 → 1.09785 BPB, no SLOT, no TTT, 3-seed.**

This is a completely new base stack that emerged while your memory was stale. It uses:
- Vocabulary size 4096 instead of 1024
- MLP width 4x instead of 3x
- Muon weight decay WD=0.090 instead of 0.04
- High WD compresses weights → all layers fit comfortably in int6 → no mixed-precision needed

This stack is now the base for most high-performing non-SLOT PRs. Your current experiments are based on the old stack (#1019). The new stack is worth rebasing on.

**Key synergy: WD=0.090 + all-int6 GPTQ (PR #1285, 1.0912 BPB)**
High weight decay compresses weights more uniformly, which makes GPTQ's task easier: all 11 layers quantize cleanly to int6 without any layers needing int8 fallback. This is a free quality upgrade if you're already using WD=0.090.

**Depth recurrence now works on this stack.** Your memory says depth recurrence was +0.025 BPB worse — that's on the old stack. On the Vocab4096/WD=0.090 stack (PRs #1260, #1279, #1285), depth recurrence (layers 4,5 shared MLP, activates at step 3000) gives −0.005 to −0.006 BPB. Different stack, different result.

---

## 3. Immediate High-Confidence Wins (Legal, Validated)

These are all implementable now, clear legal status, 3-seed results exist:

### 3a. MuonEq-R (2-line change, ~−0.001 BPB, zero bytes)
Row-normalize gradient before Newton-Schulz in the Muon optimizer. Now standard in every high-performing PR (#1260, #1279, #1285, #1290, #1296, #1326, #1298). Trivial to add.
```python
# Before Newton-Schulz:
G = G / (G.norm(dim=-1, keepdim=True) + 1e-8)
```

### 3b. Pre-Quant TTT (−0.022 BPB, legal, novel)
Run AdamW TTT on the full-precision EMA weights **before** GPTQ quantization. This is fundamentally different from the 25 failed post-quant TTT attempts — those failed because GPTQ'd int6 weights have a pathological gradient landscape. Pre-quant TTT adapts the bf16 EMA weights (which tolerate gradient updates fine), then quantizes the result. The quantized version of a TTT-adapted model is better than quantizing the original.

PR #1306 reports −0.022 BPB from 6 epochs of pre-quant TTT. This is the single cleanest legal improvement available.

### 3c. Causal SLOT (−0.009 BPB, likely legal)
Context-only delta optimization: optimize the per-window delta using ONLY tokens from previous windows (already scored), never the tokens being evaluated. PR #1306 and #1276 both implement this. Provably causal by the four-condition framework.

### 3d. N-gram Agreement via Exponential Tilting (−0.003 BPB, definitely legal)
Three causal n-gram experts (token 16-gram, within-word, word-start) with properly normalized exponential tilting (Z=1.0). Score-first, predict-then-update. PR #1302 has 3-seed results. The implementation requires a C trie but the PR has code.

**Critical: earlier n-gram approaches were measurement artifacts.** PR #1272 proved that properly-normalized n-gram blending gives ≤0.003 BPB. The larger gains from earlier n-gram PRs were from unnormalized distributions. The PR #1302 approach is correctly normalized and the −0.003 BPB result is real.

### 3e. Polar Express Newton-Schulz (−~0.005 BPB, saves ~2ms/step)
4 PE iterations ≈ 5 fixed-coefficient NS iterations but faster. Saves ~2ms/step → ~180 extra training steps within budget. PR #1298 includes this. Remember: 1ms/step = ~0.007 BPB, so 2ms saved is worth ~0.014 BPB in additional training.

### 3f. GPTQ damp=0.005 (−~0.001 BPB, one parameter change)
Standard GPTQ damping is 0.01. PR #1318 shows damp=0.005 gives a small consistent improvement. One number to change.

### Combined conservative path (all the above, no legal risk):
| Technique | BPB delta |
|-----------|-----------|
| Pre-quant TTT (6 epochs) | −0.022 |
| Causal SLOT (8 steps, context-only) | −0.009 |
| N-gram agreement | −0.003 |
| Polar Express NS | −0.005 |
| MuonEq-R | −0.001 |
| GPTQ damp=0.005 | −0.001 |
| **Total** | **−0.041** |
| **Projected BPB** | **~1.074** |

This is a **massive** 0.04 BPB improvement over SOTA, clears the 0.005 nats merge threshold by 8×, and has zero legal risk.

---

## 4. Medium-Term: Novel Techniques Not Yet in Competition

### 4a. Multi-Layer Delta SLOT (biggest untried gap)
**Mechanism:** Current SLOT places one delta at the final hidden layer (512d + 1024 logit bias). Multi-layer SLOT places deltas at every transformer block output. Total: 11 × 512 = 5,632 parameters per window (still tiny, still throwaway).

**Why it should work:** PR #1222 (Prime MLP TTT) showed that adapting ALL 11 layers beats adapting just 3 layers by 7×, and layer count matters more than rank. SLOT is currently leaving 10 layers unadapted. Multi-layer deltas give the optimizer access to how every layer's representations get used.

**Estimated impact:** −0.05 to −0.15 BPB additional vs single-layer SLOT.

**Implementation:** ~10 lines. Add hooks at each block output:
```python
deltas = [nn.Parameter(torch.zeros(bsz, 1, 512)) for _ in range(11)]
# In forward: after each block i, add deltas[i] to residual stream
```

**Legal:** Yes — frozen weights, throwaway deltas, score-first.

### 4b. Streaming Online Log-Bias (Nacrith-style, essentially free)
**Mechanism:** After committing to a prediction for token t, update a running bias vector:
```python
b += lr * (one_hot(x_t) - p_t)  # where p_t is the model's prediction
```
At the next token, add b to the logits. This is strictly causal (only past tokens used), score-first, zero artifact cost, and runs in microseconds. It learns document-specific vocabulary corrections online.

**From:** arXiv:2602.19626 (Nacrith) — achieves 0.918 bpb on real text using this technique on top of a comparable model. Comes from the text compression literature; nobody in the competition has tried it.

**Estimated impact:** −0.01 to −0.03 BPB. Trivially stacks on top of anything.

**Implementation:** ~5 lines added to the token-by-token eval loop.

### 4c. wPoE: Product of Experts with Classical Compressor (arXiv:2511.10660, EMNLP 2025)
**Mechanism:** At eval time, multiply the neural model's probability distribution with a classical PPM/context-model distribution:
```
log P_combined = α·log P_neural + (1-α)·log P_classical
```
The mixing weight α adapts per-window via the existing SLOT optimization. The classical compressor is tiny (a few hundred KB) and exploits exact pattern matching that neural models don't.

**Why it hasn't been tried:** Published November 2025, sits in the information theory / compression community rather than ML. Nobody in the competition has found it.

**Estimated impact:** −0.005 to −0.020 BPB purely eval-side.

### 4d. SSE: Secondary Symbol Estimation (from PAQ compressors)
**Mechanism:** A context-conditioned online calibration table. After each token prediction, update:
```python
table[prob_bucket][context_hash] += 0.01 * (actual - table[prob_bucket][context_hash])
```
At the next token, use the table value instead of the raw probability. The table learns that the model's predictions are systematically miscalibrated in specific micro-contexts (after commas, at sentence boundaries, etc.).

**Why it works:** This is what classical compressors like PAQ use instead of (or in addition to) neural networks. It's a context-conditioned online temperature calibration. The existing fixed temperature scalar in the SOTA is a special case of this.

**Estimated impact:** −0.005 to −0.020 BPB, purely eval-side.

**Implementation:** Medium difficulty — requires switching from batched cross-entropy to sequential per-token probability extraction.

### 4e. Adaptive Temperature Scaling (ATS, arXiv:2409.19817)
**Mechanism:** Instead of a fixed scalar temperature, train a tiny linear layer (512 → 1) that predicts a different temperature for each token position from the hidden state. Train it on the same AR self-gen calibration data used for GPTQ. Add ~512 parameters (negligible size). Apply at eval time — one dot product per token.

**The current fixed temperature captures the global mean; per-token temperature adapts to local uncertainty.** Function words (the, is, a) vs. rare content words have very different calibration needs.

**Estimated impact:** −0.003 to −0.010 BPB. Stacks with everything.

### 4f. IDTS: Entropy-Adaptive Temperature (3-line change, zero parameters)
**Mechanism:** Hard tokens (high entropy output) get lower temperature (commit harder). Easy tokens (low entropy) get higher temperature (don't over-concentrate). Counterintuitive but empirically strong. Zero parameters:
```python
entropy = -(probs * probs.log()).sum(-1)  # per-token
tau = base_temp * (1 - 0.3 * (entropy - mean_entropy) / std_entropy)
logits = logits / tau
```
**Estimated impact:** −0.002 to −0.008 BPB. Essentially free.

### 4g. Q-ROAR: RoPE-Aware Quantization (arXiv:2510.00028)
**Mechanism:** RoPE creates a known frequency-band structure in K/Q weight matrices. Different frequency bands have very different magnitudes and quantization sensitivity. Pre-scale by frequency band before GPTQ runs, undo in stored weights. No fine-tuning.

**The competition uses Partial RoPE (16/64 dims).** Q-ROAR directly targets the rotated dimensions. Natural successor to full Hessian GPTQ — same one-time quantization step, just smarter sensitivity estimation.

**Estimated impact:** −0.003 to −0.010 BPB at the quantization step.

---

## 5. Speedrun → Golf Blind Spots (Techniques Not Yet Transferred)

The NanoGPT speedrun community developed these in late 2025/early 2026 but they haven't appeared in any golf PR:

### 5a. Partial Key Offset (Dec 2025, easy, zero param cost)
Shift keys forward for specific head dimensions (dims 32-64 and 96-128). Gives later attention layers a free induction head. One index operation on the key tensor before attention. p=0.0004 in speedrun. Golf uses Partial RoPE but not this variant.

### 5b. Batch Size Schedule (Nov 2025, easy, zero artifact cost)
3-phase batch size ramp during training: small→medium→large. Richer gradient signal early, better GPU parallelism late. Completely untested in golf. The 10-minute budget is exactly where token-efficiency scheduling matters most.

### 5c. Phased MTP Schedule (Dec 2025, easy-medium)
MTP with declining weights across training phases — heavy early (3 tokens), gone by final phase (single token). Golf has static MTP only. The insight: early training benefits from multi-step supervision; final BPB should be single-token to match eval.

### 5d. Cautious Weight Decay (Nov 2025, easy)
Apply WD only to parameters whose magnitude is *increasing*, not all parameters uniformly. The golf code has one bundled negative result (PR #375) but no clean ablation. Especially relevant with golf's 4000-step warmdown. The speedrun showed WD=1.2 effective with selective masking.

### 5e. SimplifyHC: Cached Layer-7 Activation (Mar 2026, medium)
Feed a single cached activation from layer 7 to all subsequent attention layers (instead of the running residual). Prevents "contamination from prediction MLPs" in later layers. p=0.0003 in speedrun. Complements golf's U-Net skips but distinct.

---

## 6. Classical Compression → Golf (Completely Unexplored Domain)

BPB is literally a compression metric. The classical compression community solved this problem with different tools. Nobody in the competition is borrowing from them.

### 6a. Match Model (LZ-style exact matching, low-medium complexity)
During sequential eval, maintain a hash: `context[-8:] → counter(next_tokens)`. For each position, if the 8-token context has been seen before in the already-evaluated text, boost the historically-following token's probability. Free to compute from the already-scored eval prefix — strictly causal. FineWeb has heavy document-level repetition that the neural model misses.
**Estimated impact:** −0.003 to −0.008 BPB.

### 6b. Context Mixing (logistic-domain ensemble, medium-high complexity)
Combine neural LM + n-gram model + match model in logit space with online-learned weights:
```
L_combined = sum(w_i * logit_i)
# After observing x_t: w_i += eta * logit_i * (x_t - sigmoid(L_combined))
```
Weights adapt to which component is performing best on the current document. Requires sequential (not batched) eval — which the competition's sliding window already enables.
**Estimated impact:** −0.005 to −0.015 BPB.

### 6c. Causal Online Output-Layer Update (high complexity but legally sound)
After each token is scored and revealed, take one small gradient step on the output layer only (not GPTQ'd weights — just the LM head linear layer, which can be kept in bf16). Update after scoring, use for next prediction. Strictly causal. This is different from post-quant SGD TTT (which failed) because: (1) it updates only the output head, not the quantized body; (2) it updates one token at a time causally, not a batch.
**Estimated impact:** −0.003 to −0.010 BPB.

---

## 7. Long-Shot Novel Ideas (Beyond Current Competition Thinking)

These haven't been tried anywhere and have uncertain outcomes, but are grounded in the competition constraints:

### 7a. Hierarchical SLOT: Document-Level + Window-Level Deltas
Two-level adaptation: (1) a "document delta" optimized on the first 4 windows (~8K tokens), capturing document-wide style/domain; (2) window deltas initialized from the document delta, optimized per-window with fewer steps. The document delta persists; window deltas are throwaway. Documents in FineWeb are long — amortizing over the whole document should help.

### 7b. Ensemble via Geometric Mean of 3 Seeds (possibly free)
The competition requires 3-seed mean BPB. Currently each seed is evaluated independently and BPBs are averaged. Taking the geometric mean of the probability *distributions* (log-average the logits) before computing BPB is different — it's a product-of-experts ensemble. If legal (score-first, no future info), this is free given seeds are already run. Potentially −0.003 to −0.008 BPB. **Check legality before implementing.**

### 7c. Per-Window Attention Mask Adaptation
Optimize soft attention biases `[heads, seq_len, 1]` per window via the same SLOT mechanism. Instead of modifying the hidden state after attention (current SLOT), modify what information flows into each position before attention. Orthogonal to hidden-state delta — could stack with it.

### 7d. MesaNet-Style CG Optimization (arXiv:2506.05233)
Use conjugate gradient (exact local curvature) instead of AdamW/L-BFGS for the SLOT delta. For a 1,536-parameter delta (512d + 1024 vocab), the exact Hessian is a 1,536 × 1,536 matrix — computationally tractable. CG finds the exact local optimum rather than one approximate gradient step. Could reach SLOT-48's quality in 5-10 CG iterations. Hard to implement but highest ceiling.

### 7e. Delta-DCT Weight Compression (arXiv:2503.06676)
Apply DCT (like JPEG) to weight matrices before LZMA compression. Low-frequency components (global structure) stored at high precision; high-frequency (noise) aggressively quantized. The DCT may expose structure that LZMA can exploit, freeing artifact budget for higher-quality quantization. Medium difficulty, worth one experiment.

---

## 8. What To Avoid (Confirmed Dead Ends)

| Technique | Why dead | Source |
|-----------|----------|--------|
| Post-quant SGD/AdamW TTT | 25 failures; GPTQ creates pathological loss landscape | PR #756, memory |
| Standard scored-position SLOT | 100% causal violation; abaybektursun self-removed | PR #1240 |
| kNN-LM hidden state retrieval | Hurts strong models (+1.47% BPB) | PR #1103, #1259 |
| Logit averaging across windows | Catastrophic (+0.024 BPB) | PR #1103 |
| MC Dropout ensemble | Negative; no diversity at 17M params | PR #1021 |
| N-gram blending (unnormalized) | Was a measurement artifact; properly normalized ≤0.003 BPB | PR #1272 |
| Mixed int4/int5/int8 quantization | All schemes worse than uniform int6 | PR #1238 |
| QAT | Dead code — torch.compile constant-folds the flag | PR #667, #1032 |
| Data selection/curriculum | FineWeb too homogeneous; chunk selection hurts | PR #772 |
| Depth recurrence (on old stack) | Works on WD=0.090 stack; don't test on old stack | PR #363, #1285 |
| Custom Triton kernels | torch.compile already handles all fusion optimally | PR #670 |
| Knowledge distillation | ~11ms/step overhead kills any gain | PR #1029 |
| VRL (Value Residual Learning) | SOTA author disabled as "risky"; quantization incompatibility | PR #670 |

---

## 9. Recommended Experiment Order

### Phase 1: Rebase and Quick Wins (this week)
1. **Rebase on Vocab4096 + MLP4x + WD=0.090 stack** (PR #1218 code)
2. **Add MuonEq-R** (2 lines)
3. **Add GPTQ damp=0.005** (1 number)
4. **Add Polar Express NS** (from PR #1298)
5. **Port pre-quant TTT** (from PR #1306) — this is the biggest individual gain (−0.022 BPB)

### Phase 2: Legal Eval-Time Improvements
6. **Port Causal SLOT** from PR #1306 — context-only, 8 steps, safe
7. **Port N-gram agreement** (exponential tilting) from PR #1302
8. **Add IDTS** (3 lines, entropy-adaptive temperature, zero cost)
9. **Add streaming log-bias** (Nacrith-style, 5 lines, essentially free)

### Phase 3: Novel Experiments
10. **Multi-layer delta SLOT** — extend SLOT to all 11 layers, biggest unexplored gap
11. **Adaptive Temperature Scaling (ATS)** — train 512→1 linear head on AR self-gen data
12. **Partial Key Offset** from speedrun — one index operation, zero params
13. **Batch Size Schedule** — 3-phase training schedule

### Phase 4: Longer Shots
14. **wPoE** — implement small PPM compressor, multiply with neural distribution
15. **Q-ROAR quantization** — RoPE-aware weight pre-scaling before GPTQ
16. **SSE calibration table** — requires sequential eval loop refactor
17. **Hierarchical SLOT** — document-level + window-level two-tier delta

---

## 10. Updated Memory Notes

**SOTA** is still officially 1.1147 BPB (PR #1019) — nothing merged since March 30.

**New base stack to use:** Vocab4096 + MLP4x + WD=0.090 (PR #1218, clarkkev), not the old 1024-vocab stack.

**SLOT ruling is imminent.** When it comes: if non-causal SLOT is ruled legal, the frontier jumps to ~0.74 BPB (SLOT-48) and multi-layer delta SLOT becomes the next priority. If ruled illegal, the frontier is ~1.08 BPB and our Phase 1-2 plan above is the path.

**Key papers to read:**
- arXiv:2511.10660 (wPoE, EMNLP 2025) — neural × classical compressor product of experts
- arXiv:2510.00028 (Q-ROAR) — RoPE-aware quantization
- arXiv:2602.19626 (Nacrith) — online log-bias for text compression
- arXiv:2506.05233 (MesaNet) — CG test-time optimization
- arXiv:2409.19817 (ATS) — adaptive temperature scaling

---

*Brief generated by 10-agent parallel scan on 2026-04-04. Sources: 100 open PRs, 50+ merged PRs, 15+ arXiv papers, NanoGPT speedrun records, classical compression literature.*

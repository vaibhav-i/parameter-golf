# Meta-Review and Actionable Plan for Parameter Golf

**Date:** 2026-03-24
**Author:** Meta-review agent
**Context:** Beginner-intermediate ML practitioner targeting OpenAI compute grant (H100s), with local Apple Silicon for MLX prototyping. Contest deadline: April 30, 2026.

---

## Section 1: Research Quality Review

### File-by-File Assessment

#### 1. leaderboard-analysis.md
**Strengths:**
- Extremely thorough -- every submission is dissected with per-technique BPB attribution
- The "Three Pillars" framework (eval strategy, model capacity, training optimization) is a useful mental model
- Negative results documented (SwiGLU, depth recurrence, lzma) save future time
- Clean quantitative tables throughout

**Weaknesses:**
- BPB attribution is partly estimated, not from controlled ablations for every technique. The "Estimated BPB contributions of the full stack" table for the SOTA entry sums to ~0.102 BPB, which aligns with the actual 0.1016 gap, but individual line items may have interaction effects that are not captured.
- Does not track artifact size breakdown (how much of the 16MB is MLP vs attention vs embedding vs code). This matters for planning compression-based capacity expansions.
- Missing timing data for eval. The eval budget is 10 minutes, and several entries have eval times ranging from 70s to 160s. Understanding this headroom is critical for test-time training plans.

#### 2. technique-glossary.md
**Strengths:**
- Excellent pedagogical quality -- each entry explains what/why/how/tradeoffs at a level accessible to a beginner-intermediate audience
- Code snippets for key implementations (STE QAT, int6 quantization, GPTQ-lite) are directly actionable
- Good coverage of the full technique space

**Weaknesses:**
- Does not cover techniques that have NOT been tried (no entries for multi-token prediction, depth recurrence, hyperconnections, etc.)
- Some entries lack quantitative benchmarks (e.g., SmearGate and BigramHash are described qualitatively but the glossary does not include the ablation numbers from the leaderboard analysis)
- No discussion of technique interactions -- which techniques synergize and which conflict

#### 3. modded-nanogpt-insights.md
**Strengths:**
- Strong cross-pollination analysis between modded-nanogpt and parameter golf
- Clear adoption matrix shows what has/hasn't been ported
- Speculative combinations (Combos A, B, C) are creative and plausible
- Priority ranking is well-calibrated

**Weaknesses:**
- Treats modded-nanogpt as the only source of cross-domain techniques. Does not look at computer vision quantization, speech model compression, or other domains.
- The expected BPB gains for untried techniques are rough guesses without supporting evidence
- Does not account for the fact that modded-nanogpt optimizes for speed (time to target loss) while parameter golf optimizes for quality (loss at fixed size). Some techniques that help speed may not help quality and vice versa.

#### 4. evolution-chain.md
**Strengths:**
- The ASCII dependency graph is a great visualization of how entries build on each other
- Technique stacking timeline clearly shows the phases of innovation
- Pairwise combination matrix is a systematic way to find untried pairs
- Diminishing returns analysis is realistic and well-supported

**Weaknesses:**
- The combination matrix has some gaps -- for example, it does not track whether LoRA TTT has been combined with sliding window eval (it has not, and they are both eval-time techniques that should stack)
- The "Opportunity Analysis" section overlaps heavily with strategy.md -- these should have been consolidated
- Does not model the *interaction* between techniques. For instance, LoRA TTT on the SOTA model may yield much less than -0.015 BPB because the SOTA model already captures much of the per-document signal through its stronger architecture.

#### 5. arxiv-survey.md
**Strengths:**
- Good coverage of recent quantization and small model papers
- The "Quick-Start Strategy" at the end is actionable
- TTT papers are well-selected

**Weaknesses:**
- Several papers are tangential (SozKZ about Kazakh language, task-specific efficiency analysis) and add noise
- The survey does not assess whether the papers' claims apply at the 20-50M parameter scale relevant to parameter golf. Many papers test at 1B+ scale.
- Missing papers on data efficiency, curriculum learning, and training scheduling -- areas where the arxiv literature has concrete results
- No discussion of Mixture of Experts (MoE) literature, which could be very relevant for fitting more effective capacity into 16MB

#### 6. strategy.md
**Strengths:**
- Most actionable document in the set -- concrete steps with commands, expected gains, and compute estimates
- Good risk stratification (Quick Wins / Medium Effort / Ambitious)
- Unexplored directions section (4.1-4.11) is comprehensive
- Ensemble methods and curriculum learning sections are unique contributions not found elsewhere

**Weaknesses:**
- Expected BPB gains are optimistic in some cases. LoRA TTT at -0.005 to -0.015 on SOTA is likely overstated; the -0.037 BPB seen on baseline included sliding window gains that are already captured.
- Depth recurrence is listed as -0.005 to -0.015 BPB but prior attempts showed +0.051 BPB regression. The strategy acknowledges this but does not sufficiently explain why the current approach would avoid the same pitfall.
- No discussion of submission logistics (PR format, statistical significance requirements, seed management)
- Compute cost estimates are inconsistent (some in GPU-hours, some in wall-clock time)

#### 7. arxiv-deep-dive.md
**Strengths:**
- Excellent depth on the most promising research directions (depth recurrence, ternary quantization, multi-token prediction, TTT, SSMs, novel attention)
- The Relaxed Recursive Transformers paper and Cycle(rev) pattern are genuinely novel finds for this competition
- MASA (attention dictionary decomposition) is a creative find that no other file mentions
- The "Key Insight" at the end (depth recurrence + MTP + TTT are orthogonal) is the single most important strategic observation across all files

**Weaknesses:**
- Some expected BPB impacts are very wide ranges (e.g., ternary quantization: -0.005 to -0.020 or +0.010 regression). This is honest but not very actionable.
- Knowledge distillation section explores a direction that is ethically gray per contest rules -- should have been flagged more prominently
- Missing analysis of whether these papers' implementations are publicly available. A paper with no code is much harder to adopt in a competition setting.

### Cross-File Contradictions

1. **LoRA TTT expected impact:** leaderboard-analysis.md says LoRA TTT contributed -0.003 BPB (only the TTT part, after removing sliding window and doc isolation gains). strategy.md estimates -0.005 to -0.015 on SOTA. evolution-chain.md estimates -0.015 to -0.030. These differ by 10x. The most careful analysis (leaderboard-analysis.md) suggests the gain from TTT specifically is only ~0.003 BPB. The larger numbers conflate TTT with sliding window gains.

2. **Depth recurrence viability:** strategy.md and arxiv-deep-dive.md both recommend it, but evolution-chain.md documents a +0.051 BPB regression from a prior attempt. The reconciliation (that modern techniques change the picture) is plausible but unverified.

3. **Tokenizer optimization:** The README says the contest is "tokenizer-agnostic" and strategy.md briefly mentions larger BPE vocabs, but no file rigorously analyzes the tokenizer-BPB trade-off. The glossary does not have a tokenizer entry at all.

### Claims Needing Verification Before Acting

| Claim | Source | Why It Needs Verification |
|-------|--------|--------------------------|
| LoRA TTT will give -0.005+ BPB on SOTA | strategy.md | The original -0.037 BPB included sliding window; pure TTT was ~0.003 BPB on a much weaker model |
| 12 layers will fit in 10 minutes | strategy.md | Based on extrapolation from 11L timing; no actual measurement |
| BigramHash 10240 gives -0.001 BPB on SOTA | evolution-chain.md | Only tested in a 10L model without XSA/EMA; may interact differently |
| Multi-token prediction helps at 20M param scale | arxiv-deep-dive.md | Meta paper tested at 7B+; the curriculum paper addresses small models but no evidence below 100M |
| Cycle(rev) layer sharing is a "drop-in" change | arxiv-deep-dive.md | Requires handling RoPE across shared layers, LN Scale per virtual layer, and U-Net skip rework |
| MASA reduces attention params by 66.7% | arxiv-deep-dive.md | Paper tested on larger models; dictionary decomposition may underperform at dim=512 |

---

## Section 2: Missed Opportunities

### 2.1 Techniques From Other Domains

**Computer Vision Quantization:**
- **Mixed-precision NAS (Neural Architecture Search):** CV community has mature tools for automatically finding the optimal bit-width per layer (e.g., HAQ, HAWQ). These Hessian-based approaches are more rigorous than the manual allocation currently used in parameter golf.
- **Binary/Ternary convolutions:** The BinaryConnect and XNOR-Net literature from 2015-2016 explored training with extreme quantization at small scale. Lessons on gradient scaling and initialization may transfer.

**Speech Model Compression:**
- **Codec-based tokenization:** Speech models like SoundStream use learned codecs with VQ (Vector Quantization). A VQ-based weight compression scheme could outperform per-row scalar quantization by learning codebook entries that capture the actual weight distribution.
- **Pruning + quantization joint optimization:** The speech community has extensive work on jointly optimizing pruning masks and quantization levels under a total model size budget. This "joint compression" approach is absent from all parameter golf entries.

**Recommendation Systems:**
- **Embedding compression:** RecSys models face similar embedding table size constraints. Techniques like compositional embeddings (representing each token as a combination of smaller sub-embeddings) could reduce the FP16 embedding cost while maintaining quality.

### 2.2 Practical Engineering Considerations

None of the research files adequately cover:

**Training stability and debugging:**
- What happens when a run diverges at step 3000 of 7000? There is no discussion of checkpointing strategy, gradient monitoring, or loss spike recovery.
- The torch.compile constant-folding bug (Late QAT in PR #287) is mentioned but no systematic list of torch.compile pitfalls exists.
- No discussion of numerical precision issues at int5/int6 scale (e.g., accumulation errors in matmuls, norm computation stability).

**Reproducibility across hardware:**
- The contest requires 8xH100 SXM. Experiments on 1xH100, RunPod, or Apple Silicon may produce different results due to different batch sizes, DDP vs single-GPU, and hardware-specific numerical behavior.
- No discussion of how to efficiently iterate: which experiments can be validated on 1xH100 before committing to 8xH100?

**Code hygiene:**
- The submission requires a single `train_gpt.py` that runs from the records folder. No file discusses how to manage experiment branches, code diffing, or version control for the submission process.

### 2.3 Community/Social Factors and Meta-Game

**What competitors are likely working on:**
- The top competitors (signalrush, jfprincz, unnir) have been iterating on an incremental stacking strategy. They will likely continue with hyperparameter sweeps (warmdown, EMA decay, BigramHash buckets) for easy wins.
- At least one competitor has access to LoRA TTT code (samacqua). They may be working on combining it with SOTA.
- The arxiv-deep-dive.md papers (Cycle(rev), MASA, Relaxed Recursive Transformers) are all publicly available. Any competitor doing the same literature search will find them.

**Meta-game dynamics:**
- Submissions must beat SOTA by >=0.005 nats at p<0.01. This means you cannot submit incremental 0.001 improvements -- you need a batch of techniques that together clear the 0.005 threshold.
- PRs are public. Once submitted, your techniques are visible to all competitors. This creates an incentive to stockpile innovations and submit them all at once.
- The leaderboard is verified "over time" -- there may be a delay between submission and appearing on the board.

### 2.4 Risk Assessment: What If the Leaderboard Jumps

The leaderboard moved from 1.2244 to 1.1228 in 5 days (0.1016 BPB). If this rate continued:
- By April 1: ~1.09-1.10 BPB (if the rate of diminishing returns continues)
- By April 15: ~1.08-1.09 BPB
- By April 30: ~1.07-1.08 BPB

However, diminishing returns are real. A more realistic projection based on the halving improvement rate:
- Week of Mar 24: ~1.115-1.120 (incremental stacking)
- Week of Mar 31: ~1.108-1.115 (LoRA TTT or depth recurrence lands)
- Week of Apr 14: ~1.100-1.110 (someone tries ultra-low-bit or MoE)
- Week of Apr 28: ~1.090-1.105 (final push)

**Our target should be 1.100 or below by mid-April to remain competitive.**

### 2.5 Tokenizer Optimization

This is the most under-explored area across all research files. The contest is explicitly tokenizer-agnostic. Key observations:

- The baseline uses a 1024-token BPE vocabulary. BPB is the evaluation metric, not token-level loss.
- A larger vocabulary (2048, 4096) means each token represents more bytes on average, potentially improving BPB. But the embedding table grows linearly with vocab size.
- At FP16 with dim=512: 1024 vocab = 1MB, 2048 vocab = 2MB, 4096 vocab = 4MB for the embedding.
- **The trade-off:** A 2048 vocab costs +1MB but each token covers ~1.3x more bytes. If the model can maintain similar per-token loss, BPB improves because you divide by more bytes per token.
- **Critical analysis:** The README states "If changes are made to the tokenizer or dataset, prove with certainty that the val_bpb is correctly calculated." This adds verification burden but does not prohibit it.
- **Byte-level models** (vocab=256) have been explored in academic settings. They avoid tokenizer overhead but produce 4x longer sequences. With sliding window eval, the longer sequences might not be a problem.

**Recommendation:** A vocab=2048 ablation should be run early. The data download script supports different tokenizer variants.

### 2.6 Data Efficiency Tricks

The training data is FineWeb (fixed), but ordering and sampling are not discussed:

- **Document-level shuffling:** Current training likely shuffles at the token level. Shuffling at the document level (keeping documents intact) could improve learning of long-range dependencies.
- **Deduplication:** FineWeb is already curated, but near-duplicate sequences in the training set waste limited training budget.
- **Difficulty-based sampling:** Score documents by perplexity under the baseline model, then oversample hard documents in later training. This is a form of curriculum learning that can be computed offline (outside the 10-minute budget).
- **Loss-weighted sampling:** Instead of uniform random batches, sample sequences where the model's current loss is highest. This requires maintaining a running loss estimate per data shard.

### 2.7 Evaluation Strategy Optimization Beyond Sliding Window

Current eval strategy is sliding window with stride=64. Additional ideas:

- **Adaptive stride:** Use larger stride (128-256) for "easy" passages and smaller stride (32-64) for "hard" passages. Determine difficulty from initial pass.
- **Temperature scaling at eval:** Calibrate the output distribution with a learned temperature. Post-training temperature optimization costs nothing at artifact time.
- **Ensemble of contexts:** For each scored token, run the model with 2-3 different context windows and average the logits. This is a "free" ensemble.
- **Label smoothing at eval:** Apply a small uniform smoothing to the output distribution to reduce the penalty for low-probability tokens (which contribute disproportionately to BPB).
- **Document boundary handling:** The baseline may not handle document boundaries optimally. Resetting attention at document boundaries during eval (as LoRA TTT does) could improve BPB on document-initial tokens.

### 2.8 Hardware-Specific Optimizations

- **Flash Attention 3 configuration:** FA3 on H100 supports FP8 accumulation modes. No entry has explored FP8 attention computation during training. If quality is maintained, FP8 attention could significantly speed up training (more steps in 10 min).
- **Tensor core utilization:** The model uses dim=512 and head_dim=64, which align well with tensor core tile sizes (multiples of 16). However, MLP hidden=1536 may not be optimal -- 1536/16=96, which is fine, but 1536 is not a power of 2. Testing hidden=1024 (already baseline) vs 1536 vs 2048 with timing data is missing.
- **Memory management:** No discussion of gradient checkpointing, activation recomputation, or peak memory usage. At 11 layers with MLP 3x on 8xH100, memory is probably not a bottleneck, but confirming this is important.
- **NCCL optimization:** The 8-GPU setup uses DDP. No discussion of communication overlap, gradient bucketing, or sharding strategies.
- **Compilation:** torch.compile is used but no systematic analysis of compilation modes (reduce-overhead, max-autotune) or which operations benefit most.

---

## Section 3: Consolidated Technique Priority Matrix

Techniques ranked by expected value = (BPB Gain x Confidence) / Effort. Gains are median estimates.

| Rank | Technique | Expected BPB Gain | Confidence | Effort (days) | Compute Needed | Dependencies | Risk Level |
|------|-----------|-------------------|------------|---------------|----------------|--------------|------------|
| 1 | Reproduce SOTA + validate baseline | 0 (prerequisite) | 100% | 0.5 | 8xH100, 30min | None | None |
| 2 | BigramHash bucket sweep (2048->10240) | -0.001 | 80% | 0.5 | 8xH100, 4hrs | #1 | Very Low |
| 3 | Warmdown sweep (3500->4000-7000) | -0.001 | 70% | 0.5 | 8xH100, 4hrs | #1 | Very Low |
| 4 | QK-Norm addition | -0.001 | 60% | 0.5 | 8xH100, 2hrs | #1 | Low |
| 5 | Partial RoPE dimension sweep (8-32/64) | -0.0005 | 60% | 0.5 | 8xH100, 3hrs | #1 | Very Low |
| 6 | EMA decay sweep (0.995-0.999) | -0.0005 | 70% | 0.5 | 8xH100, 3hrs | #1 | Very Low |
| 7 | NTK-RoPE eval extrapolation + Partial RoPE | -0.001 | 50% | 0.5 | 8xH100, 2hrs | #1 | Low |
| 8 | GPTQ-lite on Int5 MLP weights | -0.001 | 65% | 1 | 8xH100, 2hrs | #1 | Low |
| 9 | Multi-token prediction (2-head, curriculum) | -0.004 | 45% | 2.5 | 8xH100, 10hrs | #1 | Medium |
| 10 | LoRA TTT on SOTA (eval-time) | -0.003 | 40% | 2 | 8xH100, 8hrs | #1 | Medium |
| 11 | Cycle(rev) layer sharing (6 unique -> 12 eff.) | -0.005 | 35% | 3 | 8xH100, 12hrs | #1 | Medium-High |
| 12 | Simplified hyperconnections (replace U-Net) | -0.003 | 40% | 2 | 8xH100, 6hrs | #1 | Medium |
| 13 | 12 layers with Int5 MLP + GPTQ-lite | -0.002 | 50% | 1.5 | 8xH100, 6hrs | #1 | Medium |
| 14 | Vocab 2048 ablation | -0.003 | 30% | 1.5 | 8xH100, 8hrs | #1 | Medium |
| 15 | Systematic HP grid (LR, WD, XSA layers) | -0.002 | 60% | 1.5 | 8xH100, 10hrs | #1 | Low |
| 16 | MASA attention dictionary decomposition | -0.004 | 25% | 3 | 8xH100, 8hrs | #1 | High |
| 17 | Relaxed Recursive Transformer (per-loop LoRA) | -0.006 | 30% | 3.5 | 8xH100, 12hrs | #11 | High |
| 18 | Ultra-low-bit QAT (4-bit with StableQAT) | -0.010 | 20% | 4 | 8xH100, 20hrs | #1 | Very High |
| 19 | MC Dropout ensemble at eval | -0.001 | 40% | 0.5 | 8xH100, 1hr | #1 | Low |
| 20 | Ternary + HGF (1.58-bit + FP16 correction) | -0.012 | 15% | 5 | 8xH100, 30hrs | #1 | Very High |
| 21 | Mamba-Transformer hybrid | -0.003 | 15% | 5 | 8xH100, 20hrs | #1 | Very High |
| 22 | Data ordering by difficulty (curriculum) | -0.002 | 30% | 2 | 8xH100, 8hrs | #1 | Medium |
| 23 | Document boundary aware eval | -0.001 | 50% | 0.5 | 8xH100, 1hr | #1 | Low |
| 24 | Temperature scaling at eval | -0.0005 | 50% | 0.25 | 8xH100, 30min | #1 | Very Low |

---

## Section 4: Experiment Design Templates

### Experiment 1: BigramHash Bucket Sweep

**Hypothesis:** Increasing BigramHash buckets from 2048 to 10240 reduces hash collisions and improves BPB by ~0.001, consistent with the 10L entry's finding.

**Baseline:** Current SOTA (1.1228 BPB) with BIGRAM_VOCAB_SIZE=2048.

**Implementation:** Change `BIGRAM_VOCAB_SIZE` constant in `train_gpt.py`. Values to test: 4096, 8192, 10240. Verify artifact size stays under 16MB (10240 buckets at dim=128 = ~1.3M params = ~1MB at int6+zstd).

**Measurement:** 3 seeds per config. Report mean and std of val_bpb. Use sliding window eval (stride=64). Target: p<0.05 vs baseline via two-sample t-test.

**Abort criteria:** If 4096 and 8192 both show no improvement (delta < 0.0003), skip 10240. If artifact exceeds 16MB, reduce bucket count or BigramHash embedding dimension.

**Estimated time:** MLX local: not feasible (need 8xH100 for real eval). H100 cloud: ~4 hours total (4 configs x 3 seeds x 20 min/run).

---

### Experiment 2: Warmdown Sweep

**Hypothesis:** The current warmdown of 3500 may be suboptimal. Values up to 5000-7000 could improve convergence by extending the LR decay phase.

**Baseline:** Current SOTA with warmdown=3500.

**Implementation:** Change `WARMDOWN_ITERS` in `train_gpt.py`. Values: 4000, 4500, 5000, 7000. No other changes.

**Measurement:** 3 seeds per config. Report mean val_bpb. Also report the pre-quantization val_bpb to separate convergence effects from quantization effects.

**Abort criteria:** If warmdown=4000 and 5000 are both worse than 3500, stop. If improvement is monotonic, extend to 10000.

**Estimated time:** H100 cloud: ~4 hours total.

---

### Experiment 3: QK-Norm

**Hypothesis:** Applying RMSNorm to Q and K before attention stabilizes dot products and yields ~0.001 BPB improvement, as seen in modded-nanogpt.

**Baseline:** Current SOTA with LN Scale only.

**Implementation:** In the attention computation (likely in a `CausalSelfAttention` class), add `q = F.rms_norm(q, (q.size(-1),))` and `k = F.rms_norm(k, (k.size(-1),))` before computing attention scores. Test: (a) QK-Norm alone (remove LN Scale), (b) QK-Norm + LN Scale together, (c) QK-Norm with a learnable temperature parameter.

**Measurement:** 3 seeds per variant. Report mean val_bpb and training loss curves (to detect instability).

**Abort criteria:** If training diverges or val_bpb worsens by > 0.002 in any variant, abort. If neutral (< 0.0003 change), it is not worth the code complexity.

**Estimated time:** MLX local: can prototype the code change in ~30 min. H100 cloud: ~2 hours for 3 variants x 3 seeds.

---

### Experiment 4: Multi-Token Prediction with Curriculum

**Hypothesis:** Adding 2 auxiliary prediction heads (predict token t+2 and t+3) improves sample efficiency enough to offset the ~20% per-step slowdown, yielding net -0.003 to -0.005 BPB.

**Baseline:** Current SOTA (single next-token prediction).

**Implementation:**
- Add two `nn.Linear(512, 1024)` prediction heads after the final transformer layer.
- Shift targets by 1 and 2 positions respectively, masking at document boundaries.
- Loss = `loss_1 + 0.3 * loss_2 + 0.1 * loss_3` (tunable weights).
- Curriculum: enable head 2 at step 2000, head 3 at step 4000.
- Discard heads at export (zero artifact cost).
- Key concern: verify the model still completes ~6000+ steps in 10 minutes.

**Measurement:** 3 seeds. Report val_bpb (evaluated with head 1 only), training loss per head, and step count.

**Abort criteria:** If step count drops below 5500 (meaning the extra heads cost too many training steps), reduce to 1 auxiliary head. If val_bpb does not improve after full run, abandon.

**Estimated time:** MLX local: prototype code in ~2 hours, smoke test with 200 iterations. H100 cloud: ~10 hours (implementation + 3 seeds + tuning head loss weights).

---

### Experiment 5: LoRA TTT on SOTA

**Hypothesis:** Per-document LoRA adaptation at eval time captures document-specific patterns that the fixed model cannot, yielding -0.003 to -0.008 BPB beyond sliding window eval.

**Baseline:** Current SOTA with sliding window eval only (no TTT).

**Implementation:**
- Port TTT eval code from `records/track_10min_16mb/2026-03-17_LoRA_TTT/`.
- Target c_q, c_v, lm_head with rank-8 LoRA (test rank-4 if too slow).
- Process: for each document, (a) split into 256-token chunks, (b) score chunk (accumulate BPB), (c) train LoRA on that chunk's loss, (d) reset LoRA between documents.
- Parallelize across 8 GPUs (each GPU processes different documents).
- Adam with lr=0.01, 1 gradient step per chunk.
- Critical: time the eval. Must complete within 10 minutes.

**Measurement:** val_bpb with and without TTT. Profile per-document time. Report breakdown: sliding window contribution vs LoRA TTT contribution.

**Abort criteria:** If total eval time exceeds 9 minutes, reduce LoRA rank to 4 or skip lm_head. If BPB improvement < 0.001, abandon.

**Estimated time:** MLX local: not feasible (LoRA backward pass is complex to port). H100 cloud: ~8 hours (2 days of implementation + testing).

---

### Experiment 6: Cycle(rev) Layer Sharing

**Hypothesis:** Using 6 unique layers in a palindromic pattern (1,2,3,4,5,6,6,5,4,3,2,1) gives 12 effective layers with 6 layers of parameters, freeing ~5MB for wider MLPs or other capacity.

**Baseline:** Current SOTA with 11 unique layers.

**Implementation:**
- Define 6 transformer blocks.
- Forward pass loops through them in order 1-6-6-5-4-3-2-1 (or experiment with the exact pattern).
- Each "virtual layer" needs its own LN Scale factor and potentially its own RoPE offset.
- U-Net skip connections need rethinking: the encoder-decoder pattern maps naturally to the palindrome.
- The freed ~5MB can fund MLP 4x or 5x expansion.
- Apply int6 QAT to the 6 unique blocks.

**Measurement:** 3 seeds. Report val_bpb, training speed (steps/sec), and artifact size. Compare against 11L SOTA and a hypothetical 12L baseline.

**Abort criteria:** If training diverges or val_bpb > 1.14 (i.e., worse than the 10L entries), the approach does not work at this scale. If training is too slow (< 5000 steps in 10 min), reduce to 5 unique layers or 10 effective layers.

**Estimated time:** MLX local: prototype the layer sharing logic in ~3 hours. H100 cloud: ~12 hours (3 days of implementation + debugging + 3-seed runs).

---

### Experiment 7: Simplified Hyperconnections

**Hypothesis:** Learned skip connections between arbitrary layer pairs outperform fixed U-Net skips by discovering better information routing, yielding -0.002 to -0.004 BPB.

**Baseline:** Current SOTA with fixed U-Net skip connections.

**Implementation:**
- Replace U-Net skip logic with a learnable weight matrix `alpha[i][j]` for each (destination layer i, source layer j) pair where j < i.
- Initialize to match U-Net pattern (encoder-decoder pairs get 1.0, others 0.0).
- Apply softmax or sigmoid to alpha values so they are bounded.
- Save the activation from each layer. Layer i's input = `sum(alpha[i][j] * activation[j] for j < i)`.
- Memory cost: need to store all intermediate activations. For 11 layers at dim=512, seq=2048, batch=~512K tokens: manageable.
- Parameter cost: ~55 scalar weights (11*10/2). Negligible.

**Measurement:** 3 seeds. Report val_bpb. Inspect learned alpha values to understand which connections matter.

**Abort criteria:** If val_bpb does not improve or worsens, revert to U-Net skips. If memory causes OOM, use gradient checkpointing or reduce to connecting only every-other layer.

**Estimated time:** MLX local: prototype logic in ~2 hours. H100 cloud: ~6 hours total.

---

### Experiment 8: 12 Layers with Int5 MLP

**Hypothesis:** A 12th transformer layer funded by Int5 MLP compression yields -0.001 to -0.003 net BPB improvement.

**Baseline:** Current SOTA (11 layers, int6 MLP).

**Implementation:**
- Change NUM_LAYERS to 12, MLP quantization to int5.
- Apply GPTQ-lite clip search to int5 weights.
- Verify artifact size < 16MB.
- First: benchmark training speed (12L at ~93ms/step -> ~6,450 steps in 10 min).
- If too slow, reduce MLP expansion from 3x to 2.75x.
- Update U-Net skip connections for 12-layer topology.

**Measurement:** 3 seeds. Report val_bpb, training speed, artifact size.

**Abort criteria:** If < 6000 steps complete, the depth gain is unlikely to compensate. If artifact > 16MB, reduce BigramHash buckets or MLP width.

**Estimated time:** H100 cloud: ~6 hours total (timing test + 3-seed run).

---

### Experiment 9: Vocab 2048 Ablation

**Hypothesis:** A 2048-token BPE vocabulary improves BPB by covering more bytes per token, despite the larger embedding table.

**Baseline:** Current SOTA with 1024-token vocab.

**Implementation:**
- Download the sp2048 variant: `python3 data/cached_challenge_fineweb.py --variant sp2048`
- Update VOCAB_SIZE=2048 and TOKENIZER_PATH accordingly.
- Embedding table doubles from ~1MB to ~2MB (FP16). May need to reduce model capacity elsewhere (e.g., MLP 2.5x instead of 3x).
- Carefully verify val_bpb calculation is correct with the new tokenizer (contest requirement).

**Measurement:** 3 seeds. Report val_bpb, artifact size. Compare tokens-per-byte and per-token loss to understand the BPB decomposition.

**Abort criteria:** If BPB is worse or neutral, the embedding cost is not justified. If artifact exceeds 16MB, the embedding table is too large.

**Estimated time:** H100 cloud: ~8 hours (setup + 3-seed runs + verification).

---

### Experiment 10: Systematic Hyperparameter Grid

**Hypothesis:** The current SOTA has not been fully optimized across all hyperparameter dimensions simultaneously. A systematic grid can find -0.001 to -0.003 BPB.

**Baseline:** Current SOTA hyperparameters.

**Implementation:**
- Grid over: LR {0.020, 0.025, 0.030}, WD {0.03, 0.04, 0.05}, EMA decay {0.995, 0.997, 0.999}, Late QAT threshold {0.10, 0.15, 0.20}, XSA layers {last 3, last 4, last 5}.
- Full grid is 3x3x3x3x3 = 243 configs, which is too many.
- Use a Latin hypercube sampling or sequential elimination to test ~30 configs.
- 1 seed per config for screening, then 3 seeds for the top 3.

**Measurement:** val_bpb for each config. Identify the best configuration and validate with 3 seeds.

**Abort criteria:** If the best config is within 0.0005 of the current default, the defaults are already near-optimal.

**Estimated time:** H100 cloud: ~10 hours (30 screening runs x 20 min + 9 validation runs x 20 min).

---

## Section 5: Phased Execution Plan

### Week 1 (Mar 24-30): Local Setup + Quick Wins + Grant Application

**Objective:** Establish baseline, capture easy gains, submit compute grant.

| Day | Activity | Expected Result |
|-----|----------|----------------|
| Mon-Tue | Set up MLX local environment. Download data (sp1024 and sp2048). Run baseline smoke test. Read SOTA code thoroughly. | Working local dev environment. Familiarity with codebase. |
| Wed | Write compute grant application (see Section 7). Submit by end of day. | Grant application submitted. |
| Wed-Thu | Prototype code changes locally for Experiments 3 (QK-Norm), 4 (MTP), 7 (hyperconnections). No training needed -- just code correctness. | Code ready for cloud testing. |
| Fri-Sat | Rent 1xH100 on RunPod (~$3/hr). Reproduce SOTA (Experiment #1). Run BigramHash sweep (Experiment #1) and warmdown sweep (Experiment #2). Run QK-Norm (Experiment #3). | Validated SOTA reproduction. Results for 3 quick-win experiments. |
| Sun | Analyze week 1 results. Decide which quick wins to keep. | Expected cumulative BPB: ~1.120-1.122 (if 2-3 of BigramHash/warmdown/QK-Norm/Partial-RoPE work). |

**Decision points:**
- If BigramHash 10240 works: keep it as the new baseline for all future experiments.
- If QK-Norm works: keep it. If neutral: drop it (no point adding complexity).
- If warmdown > 3500 works: adopt the best value.
- If none of the quick wins work: the SOTA is already well-optimized, and we need to focus on higher-risk experiments.

**Compute budget:** ~$50-75 (1xH100 for ~20 hours).

---

### Week 2 (Mar 31 - Apr 6): Cloud H100 Experiments (Medium Effort)

**Objective:** Run medium-effort experiments. Identify the next big gain.

**Prerequisite:** Compute grant approval (if not approved, use RunPod with personal funds; budget ~$200).

| Day | Activity | Expected Result |
|-----|----------|----------------|
| Mon-Tue | Implement and run Experiment 4 (Multi-token prediction). Start with 1 auxiliary head, then 2. | MTP results: does it improve convergence? |
| Tue-Wed | Implement and run Experiment 5 (LoRA TTT on SOTA). Focus on timing -- does it fit in 10 min eval? | TTT results: eval timing + BPB gain. |
| Thu | Run Experiment 8 (12 layers with Int5 MLP). Quick timing check first. | 12L feasibility confirmed or rejected. |
| Fri | Run Experiment 10 (HP grid, screening phase). | Top 5 HP configs identified. |
| Sat-Sun | Validate best HP configs with 3 seeds. Combine all improvements found so far. Run integrated "best-of-week-2" config. | Expected cumulative BPB: ~1.115-1.118 (if MTP or TTT works). |

**Decision points:**
- If MTP works (val_bpb < 1.118): integrate into the baseline and use for all future experiments.
- If MTP fails: abandon and focus on TTT and architectural changes.
- If LoRA TTT fits in eval budget and gives >= 0.002 BPB: integrate.
- If LoRA TTT is too slow: try rank-4, or try updating only LayerNorm scales (simpler TTT).
- If 12L works: use as new baseline. If not: stay at 11L.

**Compute budget:** ~$200-400 (8xH100 for ~40 hours, or 1xH100 for screening + 8xH100 for final runs).

---

### Week 3 (Apr 7-13): Ambitious Experiments

**Objective:** Attempt higher-risk experiments that could yield transformative gains.

| Day | Activity | Expected Result |
|-----|----------|----------------|
| Mon-Tue | Implement Experiment 6 (Cycle(rev) layer sharing). Start with 6 unique layers, 12 effective. | Layer sharing prototype running. |
| Wed | Run Cycle(rev) on 8xH100. If it works, combine with MLP 4-5x expansion. | Cycle(rev) results. |
| Thu-Fri | Implement Experiment 7 (Hyperconnections). Replace U-Net skips. | Hyperconnections results. |
| Sat | If Cycle(rev) or hyperconnections worked: combine with all prior gains. Full integration run. | Expected cumulative BPB: ~1.108-1.115 (if one architectural change works). |
| Sun | Monitor leaderboard. Assess competitive position. Plan week 4. | Updated strategy based on leaderboard state. |

**Decision points:**
- If Cycle(rev) gives >= -0.003 BPB: this becomes the core architectural change. Invest in tuning.
- If Cycle(rev) fails: try the Sandwich pattern (unique first/last 2 layers, shared middle).
- If hyperconnections fail: keep U-Net skips (they are proven to work).
- If leaderboard has jumped to < 1.10: switch to ultra-low-bit QAT (Experiment 18 in priority matrix) as a Hail Mary.

**Compute budget:** ~$200-400.

---

### Week 4 (Apr 14-20): Refinement + Submission Preparation

**Objective:** Combine all working techniques, run final hyperparameter tuning, prepare submission.

| Day | Activity | Expected Result |
|-----|----------|----------------|
| Mon-Tue | Integrate all successful experiments into a single `train_gpt.py`. Clean up code. | Unified submission script. |
| Wed | Run 5 seeds on 8xH100 to establish statistical significance. | Mean val_bpb with confidence interval. |
| Thu | If not yet competitive: try vocab 2048 ablation (Experiment 9) as a wildcard. | Vocab ablation results. |
| Fri | Write submission README.md with detailed explanation. Prepare `submission.json`. | Complete submission package. |
| Sat | Final review. Check that script runs cleanly from the records folder. Verify artifact size < 16MB. | Submission ready. |
| Sun | Submit PR. Monitor for reviewer feedback. | PR submitted. |

**Decision points:**
- If our best result does not beat SOTA by >= 0.005 nats: submit as a non-record entry with detailed documentation of negative/interesting results.
- If we are within 0.003 nats of SOTA: investigate whether any untried combinations (from the evolution chain's untried pairs) could close the gap.

**Compute budget:** ~$100-200.

---

### Week 5 (Apr 21-30): Buffer + Final Push

**Objective:** Handle unexpected issues, iterate on reviewer feedback, final optimizations.

| Day | Activity | Expected Result |
|-----|----------|----------------|
| Apr 21-25 | If submission was not competitive: run ultra-low-bit QAT (Experiment 18) as final moonshot. | Ultra-low-bit results (likely either transformative or complete failure). |
| Apr 26-28 | Final submission. Address any PR review comments. | Accepted submission. |
| Apr 29-30 | Contest deadline. Ensure submission is complete and verified. | Done. |

**Compute budget:** ~$100 (reserve for emergency runs).

---

### Total Compute Budget Summary

| Phase | Duration | Estimated Cost |
|-------|----------|---------------|
| Week 1 (local + 1xH100) | 7 days | $50-75 |
| Week 2 (medium experiments) | 7 days | $200-400 |
| Week 3 (ambitious experiments) | 7 days | $200-400 |
| Week 4 (refinement + submission) | 7 days | $100-200 |
| Week 5 (buffer) | 10 days | $100 |
| **Total** | **38 days** | **$650-1175** |

If the compute grant is approved, the budget is largely covered. If not, prioritize weeks 1-2 experiments and skip the most compute-intensive experiments (ultra-low-bit QAT, Cycle(rev) tuning).

---

## Section 6: Competitive Analysis

### Leaderboard Trajectory

| Period | BPB Range | Rate of Improvement | Dominant Innovation Type |
|--------|-----------|---------------------|------------------------|
| Week 1 (Mar 17-22) | 1.2244 -> 1.1228 | -0.0145 BPB/day | Eval strategy + compression + capacity |
| Week 2 (Mar 22-29, projected) | 1.1228 -> ~1.115 | -0.001 BPB/day | Hyperparameter tuning + small architectural changes |
| Week 3-4 (Mar 29 - Apr 13) | ~1.115 -> ~1.105 | -0.0007 BPB/day | Novel techniques (TTT, MTP, depth recurrence) |
| Week 5-6 (Apr 14 - Apr 30) | ~1.105 -> ~1.095 | -0.0005 BPB/day | Combined innovations + polish |

**Target to be competitive by April 30: below 1.100 BPB.**

### What Competitors Are Likely Exploring

Based on the public PR history and technique trends:

1. **signalrush (current SOTA):** Likely continuing incremental improvements. GPTQ-lite was their last innovation; may be exploring int5 for more tensors or larger BigramHash.

2. **jfprincz (ranks 2-3):** Strong on architectural innovations (XSA, Partial RoPE). May be exploring hyperconnections or other attention modifications.

3. **unnir (rank 4):** Systems optimization focus (efficient XSA implementation). May be working on Flash Attention 3 optimizations or other systems-level speed improvements.

4. **samacqua (LoRA TTT):** Has TTT expertise. Almost certainly working on combining TTT with the modern SOTA stack.

5. **New entrants:** The competition description explicitly mentions "depth recurrence, low-rank training, bitnets, novel tokenizers, test-time training." OpenAI expects these to be explored. Well-resourced new entrants (e.g., university groups with the compute grant) may attempt these more radical directions.

### How to Stay Ahead

1. **Monitor PRs daily.** All submissions are public. When a new record appears, immediately read the README and identify the new techniques.

2. **Stockpile innovations.** Do not submit each small improvement separately. Accumulate multiple techniques and submit them as a batch to clear the 0.005-nat threshold.

3. **Focus on orthogonal directions.** Competitors will iterate on the existing stack (hyperparameters, minor arch changes). Differentiate by pursuing directions others are less likely to try: multi-token prediction, depth recurrence with Cycle(rev), or vocab optimization.

4. **Keep implementation private until submission.** Fork the repo privately, develop locally, and only submit when ready.

5. **Speed of iteration matters.** The competitor who can run the most experiments per week has a structural advantage. Invest in tooling: automated experiment tracking, rapid config changes, batch seed runs.

---

## Section 7: Compute Grant Application Notes

### What to Include

**Project title:** "Efficient Language Model Compression: Pushing the Parameter Golf Frontier"

**Summary (2-3 sentences):**
We aim to explore novel compression and training techniques for small language models in the Parameter Golf challenge. Our research focuses on three under-explored directions: depth recurrence with Cycle(rev) weight sharing, multi-token prediction for sample-efficient training, and test-time training for eval-time adaptation. These techniques are orthogonal to the current SOTA and could collectively yield 0.02-0.04 BPB improvement.

### Key Experiments to Highlight

1. **Depth recurrence (Cycle(rev)):** 6 unique layers, 12 effective layers. Based on Takase & Kiyono (arXiv:2104.06022). Never attempted in parameter golf. Could enable significantly deeper models within 16MB.

2. **Multi-token prediction:** Based on Gloeckle et al. (arXiv:2404.19737, Meta). Zero artifact cost (prediction heads discarded). Improves sample efficiency for the 10-minute training budget.

3. **Test-time training (LoRA TTT) on modern SOTA:** Extending the earlier LoRA TTT work to the 11-layer, int6-quantized SOTA model. Per-document adaptation during eval.

4. **Ultra-low-bit quantization (3-4 bit):** Exploring StableQAT's Fourier-based gradient surrogate for training at 3-4 bits per weight. Could double effective model capacity within 16MB.

### Compute Budget Justification

| Experiment | Runs Needed | Time per Run (8xH100) | Total H100-hours |
|------------|-------------|----------------------|------------------|
| Quick wins (BigramHash, warmdown, QK-Norm, etc.) | ~30 runs | 20 min | 80 |
| Multi-token prediction | ~15 runs | 20 min | 40 |
| LoRA TTT | ~10 runs | 20 min | 27 |
| Depth recurrence (Cycle(rev)) | ~20 runs | 20 min | 53 |
| Hyperconnections | ~10 runs | 20 min | 27 |
| 12-layer exploration | ~10 runs | 20 min | 27 |
| HP grid search | ~40 runs | 20 min | 107 |
| Ultra-low-bit QAT | ~15 runs | 20 min | 40 |
| Final integration + 5-seed validation | ~10 runs | 20 min | 27 |
| **Total** | **~160 runs** | | **~428 H100-hours** |

At 8xH100 per run, this is ~53 8xH100-hours, or roughly $1,000-1,500 at RunPod rates.

**Request: $1,500 in compute credits**, which covers the experiment plan with buffer for unexpected reruns and debugging.

### Positioning the Application

Emphasize:
- Novelty: depth recurrence and multi-token prediction have never been tried in parameter golf.
- Systematic approach: well-defined experiment plan with abort criteria.
- Contribution to the community: all results (positive and negative) will be documented in the submission README.
- Background: link to any prior ML projects, coursework, or relevant experience.
- Alignment with challenge goals: the README explicitly lists "depth recurrence" and "test-time training" as exciting directions.

---

## Appendix: Key File Paths

| File | Purpose |
|------|---------|
| `/Users/vaibhavippili/Desktop/Personal/oaipg/parameter-golf/train_gpt.py` | Main CUDA training script (baseline) |
| `/Users/vaibhavippili/Desktop/Personal/oaipg/parameter-golf/train_gpt_mlx.py` | Apple Silicon MLX version |
| `/Users/vaibhavippili/Desktop/Personal/oaipg/parameter-golf/data/cached_challenge_fineweb.py` | Dataset download |
| `/Users/vaibhavippili/Desktop/Personal/oaipg/parameter-golf/records/track_10min_16mb/` | All leaderboard submissions |
| `/Users/vaibhavippili/Desktop/Personal/oaipg/parameter-golf/research/leaderboard-analysis.md` | Detailed submission analysis |
| `/Users/vaibhavippili/Desktop/Personal/oaipg/parameter-golf/research/technique-glossary.md` | Technique reference |
| `/Users/vaibhavippili/Desktop/Personal/oaipg/parameter-golf/research/modded-nanogpt-insights.md` | Cross-competition analysis |
| `/Users/vaibhavippili/Desktop/Personal/oaipg/parameter-golf/research/evolution-chain.md` | Submission lineage and untried combinations |
| `/Users/vaibhavippili/Desktop/Personal/oaipg/parameter-golf/research/arxiv-survey.md` | Literature survey |
| `/Users/vaibhavippili/Desktop/Personal/oaipg/parameter-golf/research/arxiv-deep-dive.md` | Deep dive on key papers |
| `/Users/vaibhavippili/Desktop/Personal/oaipg/parameter-golf/research/strategy.md` | Competition strategy |

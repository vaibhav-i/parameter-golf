# Parameter Golf: Solution Evolution Chain

## 1. Dependency Graph

The graph below shows how each submission built on prior work. The **main trunk** is the
bold vertical line on the left. Branches represent independent explorations that fed
techniques back into the trunk.

```
DATE        ENTRY (BPB)                                    AUTHOR LINEAGE
==========  =============================================  ========================

03-17       NaiveBaseline (1.2244)
            |
            |---> LoRA_TTT (1.195)                         [BRANCH A: Test-Time Training]
            |     (adds LoRA TTT at eval; training unchanged)
            |     Dead-end: never merged back into trunk
            |
            |---> LowerLR (1.2230)                         [BRANCH B: LR Tuning]
            |     (LR sweep only, no arch changes)
            |     Findings absorbed into trunk
            |
            +---> FP16Embed_WD3600 (1.2197)                [TRUNK continues]
            |     (fp16 embed, warmdown 3600, LR 0.06)
            |
            |---> WarmdownQuantization (1.2154)            [BRANCH C: Schedule Tuning]
            |     (warmdown=20000, NTK-RoPE eval@1408)
            |     Warmdown insight partially absorbed
            |
            +---> LongContextSeq2048 (1.2058)              [TRUNK]
            |     (seq_len 2048, tuned LR)
            |
            +---> TrainingOptSeq4096 (1.2014)              [TRUNK]
            |     (seq_len 4096, Muon 0.99, lower LR)
            |
            +---> Seq2048_FP16Emb_TunedLR (1.1598)        [TRUNK - major jump]
            |     (10L, int6 QAT STE, zstd, MLP 2.6x,
            |      fp16 embed, seq 2048, sliding eval)
            |
            +---> SlidingWindowEval (1.1925)               [BRANCH D: Eval-only]
            |     (baseline model + sliding window stride=64)
            |     Technique absorbed into every later entry
            |
            |---> 10L_MixedPrecision (1.2147)              [BRANCH E: Depth + Mixed Quant]
            |     (10L, int6 on middle layers, int8 on edges)
            |     Superseded by full int6 QAT approach
            |
            +---> MixedQuant_Int6Int8_SlidingWindow (1.1630)  [TRUNK variant]
            |     (9L, MLP 3x, STE int6 QAT, mixed quant,
            |      sliding eval stride=64)
            |
03-19       +---> MLP3x_QAT_Int6_SlidingWindow (1.1502)   [TRUNK - signalrush line]
            |     (11L, MLP 3x, int6 QAT, zstd-22,
            |      WD=0.04, Muon 0.99, U-Net skips,
            |      sliding eval stride=64)
            |
            |---> smeargate_orthoinit_muonwd (1.1556)      [BRANCH F: Embedding Innovations]
            |     (9L, SmearGate, BigramHash 4096,
            |      OrthoInit, Muon WD=0.01, MLP 3x,
            |      int6 QAT, sliding eval)
            |
            +---> Int6_MLP3x_SmearGate_BigramHash_         [TRUNK - merge point]
            |     MuonWD_SWA (1.1458)
            |     (9L, SmearGate, BigramHash 4096,
            |      OrthoInit, WD=0.04, SWA every 50,
            |      MLP 3x, int6+zstd, sliding eval)
            |
            |---> SlidingWindow_FP16Emb_10L_MuonWD_        [BRANCH G: 10L + Overtone]
            |     OvertoneInit (1.1748)
            |     (10L, Overtone spectral init,
            |      phase-transition residual mixing)
            |     Overtone init NOT adopted downstream
            |
03-20       +---> 10L_Int5MLP_MuonWD04_SWA50 (1.1428)     [TRUNK - depth + int5]
            |     (10L, int5 MLP / int6 attn,
            |      BigramHash 10240, SWA start_frac=0.4,
            |      SmearGate, WD=0.04)
            |     Built on PR #162 (smeargate_orthoinit)
            |
            +---> 11L_EfficientPartialXSA_FA3_             [TRUNK - XSA introduced]
            |     SWA120 (1.1307)
            |     (11L, Partial XSA last 3 layers,
            |      FlashAttention 3, SWA every 120,
            |      SmearGate, BigramHash 2048)
            |
            +---> 11L_XSA4_EMA_Int6_MLP3x_                 [TRUNK]
            |     WD04_1.1271 (1.1271)
            |     (11L, XSA last 4, EMA replaces SWA,
            |      decay=0.997)
            |     PR #287
            |
03-21       +---> 11L_XSA4_EMA_PartialRoPE_               [TRUNK]
            |     LateQAT_1.1248 (1.1248)
            |     (Partial RoPE 16/64 dims,
            |      LN Scale 1/sqrt(layer+1))
            |     Note: Late QAT was dead code
            |     PR #374
            |
03-22       +---> 11L_EMA_GPTQ-lite_warmdown3500_          [TRUNK - current SOTA]
                  QAT015_1.1233 (1.1233)
                  (GPTQ-lite clip search,
                   EMA + tight SWA stacked,
                   warmdown 3500, QAT@0.15)
                  PR #374+
```

### Lineage Summary

**Main Trunk** (bold path from baseline to SOTA):
```
Baseline -> FP16Embed -> Seq2048 -> Seq4096 -> Seq2048+Int6QAT+10L
  -> 11L+MLP3x+QAT -> SmearGate+BigramHash merged in -> 10L+Int5
  -> 11L+XSA+SWA -> 11L+XSA+EMA -> PartialRoPE+LNScale -> GPTQ-lite
```

**Dead-end branches** (techniques not carried forward):
- **LoRA TTT** (Branch A) -- never combined with anything beyond baseline
- **WarmdownQuantization** (Branch C) -- extreme warmdown=20000 abandoned
- **10L_MixedPrecision** (Branch E) -- superseded by full int6 QAT
- **Overtone Init** (Branch G) -- spectral init not adopted

**Successfully merged branches:**
- **SlidingWindowEval** (Branch D) -- adopted universally after 03-19
- **smeargate_orthoinit** (Branch F) -- SmearGate + BigramHash + OrthoInit adopted into trunk
- **LowerLR** (Branch B) -- lower LR findings absorbed early

---

## 2. Technique Stacking Timeline

Each stage below shows what the **dominant strategy** was and what new techniques accumulated.

### Stage 1: Baseline (03-17) -- BPB ~1.224

| Technique | Status |
|-----------|--------|
| 9 layers, 512 dim, MLP 2x | baseline arch |
| int8 + zlib quantization | baseline compression |
| seq_len 1024 | baseline context |
| Muon + AdamW optimizers | baseline training |
| Tied embeddings | baseline |

**Key bottleneck:** quantization gap (~0.007 BPB) and suboptimal LR.

### Stage 2: FP16 Embed + Longer Context (03-18) -- BPB ~1.200

| New Technique | Impact |
|---------------|--------|
| FP16 tied embedding (skip int8 quant on embed) | -0.006 BPB quant gap |
| Longer warmdown (1200 -> 3600) | better convergence |
| seq_len 2048 or 4096 | more context per token |
| Lower LR (0.04 -> 0.02) | -0.005 BPB |
| Higher Muon momentum (0.95 -> 0.99 with warmup) | smoother optimization |

**Key insight:** The embedding is the most quantization-sensitive tensor because it
pulls double duty as input lookup AND output projection.

### Stage 3: Int6 Quantization + Sliding Window Eval (03-19 early) -- BPB ~1.155

| New Technique | Impact |
|---------------|--------|
| STE int6 QAT (fake quantize during training) | eliminates quant gap entirely |
| Sliding window eval (stride=64) | -0.032 BPB (free at eval time) |
| MLP 3x expansion (funded by int6 savings) | -0.029 BPB |
| zstd-22 compression (replaces zlib) | saves ~1.5 MB -> funds more params |
| U-Net skip connections | modest improvement |
| Weight decay for Muon (0.01-0.04) | regularizes + helps quantization |

**Key insight:** Int6 QAT is the biggest single unlock. By training the model to expect
quantization noise, you can use fewer bits AND a wider model. The size savings from int6
fund going from MLP 2x to 3x, which is the single largest quality contributor.

### Stage 4: Architecture + Embedding Innovations (03-19 late to 03-20) -- BPB ~1.143

| New Technique | Impact |
|---------------|--------|
| 10-11 transformer layers (from 9) | -0.003 to -0.010 BPB |
| SmearGate (bigram blending at embedding layer) | -0.002 BPB |
| BigramHash embedding (hash table for token pairs) | -0.003 BPB |
| Orthogonal weight init | faster convergence |
| SWA (stochastic weight averaging in warmdown) | -0.001 BPB, smoother weights |
| Int5 for MLP weights (int6 for attention) | saves space -> funds 10th layer |
| Larger BigramHash (4096 -> 10240 buckets) | -0.001 BPB |

**Key insight:** SmearGate and BigramHash give the model "free" bigram context before
the first attention layer. This is cheap in parameters but valuable because attention
is the bottleneck in a small model. Going to 10-11 layers pushes depth as far as the
size budget allows.

### Stage 5: XSA + EMA + Advanced Quantization (03-20 to 03-22) -- BPB ~1.123

| New Technique | Impact |
|---------------|--------|
| Exclusive Self Attention (XSA, last 3-4 layers) | -0.002 BPB |
| EMA weight averaging (replaces SWA) | -0.001 BPB, smoother |
| FlashAttention 3 (Hopper-optimized) | more steps in 10 min |
| Partial RoPE (16/64 dims) | -0.002 BPB |
| LN Scale (1/sqrt(layer+1)) | stabilizes deep models |
| GPTQ-lite (per-row optimal clip percentile) | -0.0006 BPB |
| EMA + Tight SWA stacked | -0.0006 BPB combined |
| Late QAT (STE in final 15% of training) | -0.0001 BPB |
| Warmdown 3500 (tuned from 3000) | -0.0002 BPB |

**Key insight:** Returns are diminishing. XSA is the last "structural" win. Most gains
now come from post-training quantization tricks (GPTQ-lite) and weight averaging
refinements.

---

## 3. Diminishing Returns Analysis

### BPB Improvement Over Time (Conceptual)

```
BPB
1.225 |*  Baseline
      |
1.220 |  *  FP16 Embed
      |
1.205 |     *  Seq2048
      |
1.200 |      *  Seq4096
      |                              <-- BIG JUMP: int6 QAT + sliding window
1.165 |            *  MixedQuant+Slide
      |
1.155 |             *  SmearGate 9L
      |
1.150 |              *  11L MLP3x QAT
      |
1.145 |               *  SmearGate+BigramHash merged
      |
1.143 |                *  10L Int5 MLP
      |
1.131 |                  *  11L XSA + SWA
      |
1.127 |                   *  11L XSA + EMA
      |
1.125 |                    *  Partial RoPE
      |
1.123 |                    *  GPTQ-lite        <-- current SOTA
      |_______________________________________________
      03-17    03-18    03-19    03-20    03-21  03-22
```

### Gap Analysis

| Transition | BPB Drop | Category |
|------------|----------|----------|
| Baseline -> Int6 QAT + Sliding + MLP3x | -0.061 | **Enormous** (quant + eval + capacity) |
| + 11L + SmearGate + BigramHash | -0.017 | **Large** (arch + embeddings) |
| + XSA + EMA | -0.016 | **Medium** (attention + averaging) |
| + Partial RoPE + LN Scale | -0.002 | **Small** (positional + normalization) |
| + GPTQ-lite + warmdown tune | -0.002 | **Tiny** (post-train polish) |

**Where we are now:** Deep in the diminishing-returns tail. Each incremental BPB
improvement costs more engineering effort. The "easy" wins (int6 QAT, sliding window,
wider MLP) were found early. Remaining gains likely require fundamentally different
approaches or clever recombination of untried technique pairs.

---

## 4. Combination Matrix: Untried Technique Pairs

The table below maps which technique pairs from DIFFERENT evolutionary branches have
and have not been combined. "Y" = tried together, blank = never combined.

### Technique Origin Map

| Technique | Origin Branch | First Appeared |
|-----------|---------------|----------------|
| LoRA TTT | Branch A | 03-17 |
| Sliding Window Eval | Branch D | 03-19 |
| SmearGate | Branch F | 03-19 |
| BigramHash | Branch F | 03-19 |
| OrthoInit | Branch F | 03-19 |
| XSA | Trunk (03-20) | 03-20 |
| EMA | Trunk (03-20) | 03-20 |
| Overtone Init | Branch G | 03-19 |
| Extreme Warmdown (20k) | Branch C | 03-19 |
| NTK-RoPE Eval Extrapolation | Branch C | 03-19 |
| Partial RoPE | Trunk (03-21) | 03-21 |
| GPTQ-lite | Trunk (03-22) | 03-22 |
| Int5 MLP | Trunk (03-20) | 03-20 |
| Depth Recurrence (looped layers) | mentioned, never tried | -- |
| SwiGLU | tried and abandoned | 03-18 |
| LoRA (at train time) | not tried | -- |

### Pairwise Combination Matrix

```
                  LoRA  Slide SmearG Bigram XSA   EMA   Overtone ExtWD  NTK-  Partial GPTQ  Int5
                  TTT   Win          Hash                        (20k)  RoPE  RoPE    lite  MLP
LoRA TTT           --
Slide Win               --    Y      Y      Y     Y              .      .     Y       Y     Y
SmearGate                     --     Y      Y     Y              .      .     Y       Y     Y
BigramHash                           --     Y     Y              .      .     Y       Y     Y
XSA                                         --    Y                           Y       Y
EMA                                               --                          Y       Y
Overtone Init                                           --
Ext Warmdown(20k)                                                --
NTK-RoPE Eval                                                          --
Partial RoPE                                                                  --      Y
GPTQ-lite                                                                             --
Int5 MLP                                                                                    --
```

### Key UNTRIED Combinations

| Combination | Why It Matters | Tried? |
|-------------|----------------|--------|
| **LoRA TTT + anything beyond baseline** | TTT gave -0.037 BPB on baseline; never tested with int6, XSA, SmearGate, 11L, etc. | NO |
| **LoRA TTT + Sliding Window** | Both are eval-time-only techniques. Could they stack? | NO |
| **LoRA TTT + XSA + 11L** | TTT adapts per-document; XSA improves attention quality. Orthogonal wins? | NO |
| **XSA + Int5 MLP** | XSA was only tried with int6. Int5 MLP saves space -> could fund 12th layer with XSA. | NO |
| **Overtone Init + XSA + 11L** | Spectral init was only tried at 10L without XSA. | NO |
| **Overtone Init + SmearGate + BigramHash** | Two different embedding enhancement strategies, never combined. | NO |
| **Extreme Warmdown (20k) + int6 QAT** | WD=20k showed great quant-friendliness with int8; might be even better with int6 QAT. | NO |
| **NTK-RoPE Eval Extrapolation + Partial RoPE** | NTK eval scaling at 1408 was only tried at 9L baseline; Partial RoPE changes the game. | NO |
| **GPTQ-lite + Int5 MLP** | GPTQ-lite clip search was only tried with int6; int5 has more quantization error to optimize. | NO |
| **BigramHash (10240 buckets) + XSA + 11L** | The 10240-bucket version was only in the 10L entry; 11L+XSA used 2048 buckets. | NO |
| **SmearGate + XSA + GPTQ-lite + Int5** | Full combination of embedding + attention + quant innovations from different branches. | NO |
| **Depth Recurrence (looped layers)** | Mentioned as "promising but needs more steps" in FP16Embed entry; never actually tried. | NO |
| **SWA + EMA stacked + Int5 MLP** | Stacked averaging was only tried with int6, not int5. | NO |
| **12 layers** (funded by int5 + aggressive compression) | Nobody tried going beyond 11 layers. | NO |

---

## 5. Opportunity Analysis: Ranked by Expected Impact and Difficulty

### Tier 1: High Expected Impact, Moderate Difficulty

| Rank | Combination | Expected Impact | Difficulty | Rationale |
|------|------------|-----------------|------------|-----------|
| 1 | **LoRA TTT on current SOTA (11L+XSA+EMA)** | -0.015 to -0.030 BPB | Medium | LoRA TTT gave -0.037 on the weak baseline. Even if the stronger model captures some of the same signal, per-document adaptation should still help. The eval budget (10 min) is tight -- need to optimize TTT speed. The original author used only 1/10th of eval budget, so there is headroom. |
| 2 | **BigramHash 10240 + XSA + 11L + GPTQ-lite** | -0.003 to -0.008 BPB | Low | The 10240-bucket BigramHash gave -0.001 BPB over 4096 in a weaker model. Current SOTA uses only 2048 buckets. Scaling up is a simple config change with known positive signal. Combined with existing techniques, nearly free. |
| 3 | **Int5 MLP + 12L (funded by size savings)** | -0.003 to -0.005 BPB | Medium | Int5 MLP saves ~1.86 MB over int6. That could fund a 12th layer. Nobody has tried 12L. The question is whether 12L fits in 10 min training. At ~85ms/step for 11L, 12L would be ~93ms/step -> ~6,450 steps. Might work. |

### Tier 2: Medium Expected Impact, Higher Difficulty

| Rank | Combination | Expected Impact | Difficulty | Rationale |
|------|------------|-----------------|------------|-----------|
| 4 | **Extreme Warmdown + Int6 QAT** | -0.001 to -0.003 BPB | Low | WD=20000 showed the entire training run in decay phase produces the best quantization-friendly weights with int8. With int6 QAT, the model already learns to tolerate quantization, so the gain is smaller, but WD has never been pushed beyond 3500 in the int6 QAT regime. Worth a quick sweep. |
| 5 | **NTK-RoPE Eval + Partial RoPE + XSA** | -0.001 to -0.003 BPB | Low | Partial RoPE leaves 48/64 dims position-free. NTK extrapolation at eval time (e.g., eval_seq_len=1408) might interact differently with partial RoPE since fewer dims carry positional info. Simple to try. |
| 6 | **Depth Recurrence (looped layers)** | -0.005 to -0.015 BPB | High | Reuse a block of layers multiple times (e.g., loop 3 layers 4 times = 12 effective layers but only 3 layers of parameters). This was noted as "promising but needs more steps" in March 18 entry. With int5/int6 compression and 10-min budget, it might now be viable. But it requires significant code changes and careful tuning. |
| 7 | **GPTQ-lite on Int5 MLP weights** | -0.001 to -0.002 BPB | Low | GPTQ-lite clip search should help int5 even more than int6 (more quantization error to optimize). Straightforward extension. |

### Tier 3: Speculative / High Risk

| Rank | Combination | Expected Impact | Difficulty | Rationale |
|------|------------|-----------------|------------|-----------|
| 8 | **LoRA at train time (not just TTT)** | Unknown | High | Using LoRA during training to create a base model + adapter pair, where the adapter is applied at eval. Could allow a smaller base model with per-task adaptation. Untested territory in this competition. |
| 9 | **Overtone Init + OrthoInit comparison at 11L** | -0.001 to -0.002 BPB | Low | Overtone spectral init was only tried at 10L in a weaker config. At 11L with all current tricks, it might outperform ortho init. Quick experiment. |
| 10 | **SwiGLU + Int5 compression** | Unknown | Medium | SwiGLU was abandoned because it was 45% slower per step. But with int5 compression the model is smaller -> SwiGLU might be faster now. Worth re-benchmarking. |
| 11 | **Mixture of Experts (MoE)** | Potentially large | Very High | Never attempted. Even a simple 2-expert MoE with top-1 routing could double MLP capacity at ~1.5x compute. Fits the size budget if experts are int5-quantized. Very significant code changes required. |

### Summary of Recommendations

For someone at a **beginner-intermediate ML level**, the best next steps are:

1. **Try BigramHash 10240 in the current SOTA config** -- this is a one-line config change
   (BIGRAM_VOCAB_SIZE=10240) that has demonstrated positive signal. Lowest effort, clear upside.

2. **Sweep warmdown values** (3500 -> 4000, 5000, 7000) in the current SOTA -- another
   pure hyperparameter change, no code modifications needed.

3. **Try LoRA TTT on the current SOTA** -- requires understanding the eval code, but the
   LoRA TTT implementation already exists from the 03-17 entry. The challenge is fitting
   it into the 10-minute eval budget with the heavier 11L model.

4. **Try 12 layers with Int5 MLP** -- requires checking if training fits in 10 minutes.
   If it does, this stacks depth (proven to help) with aggressive compression (proven to work).

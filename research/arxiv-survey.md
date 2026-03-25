# Arxiv Survey: Techniques Relevant to Parameter Golf

**Date:** 2026-03-24
**Search methodology:** Queried the arxiv API using title and abstract searches across four topic areas: quantization-aware training, small language models, model compression, and test-time training. Results were filtered to the ~15 most relevant papers for the Parameter Golf challenge constraints (16MB model, 10-minute training budget, compression, quantization, small transformers).

---

## Quantization and Compression

### 1. pQuant: Towards Effective Low-Bit Language Models via Decoupled Linear Quantization-Aware Training
- **Authors:** Wenzheng Zhang, Bingzheng Liu, Yang Hu, et al.
- **Date:** 2026-02-26 | **ID:** [2602.22592](https://arxiv.org/abs/2602.22592)
- **Summary:** Addresses extreme low-bit LLM quantization by splitting linear layers into a dominant 1-bit branch and a compact high-precision branch. Uses tailored feature scaling and extends the architecture into sparsely-activated experts for efficient capacity scaling.
- **Relevance:** Directly applicable -- a 1-bit dominant branch with a small high-precision branch could fit a strong model into 16MB. The sparse expert extension could add capacity without proportional parameter cost.

### 2. Attn-QAT: 4-Bit Attention With Quantization-Aware Training
- **Authors:** Peiyuan Zhang, Matthew Noto, Wenxuan Tan, et al.
- **Date:** 2026-02-09 | **ID:** [2603.00040](https://arxiv.org/abs/2603.00040)
- **Summary:** Systematic study of 4-bit QAT for attention mechanisms on FP4-capable GPUs. Identifies key principles for stable FP4 attention and implements fused Triton kernels with up to 1.5x speedup.
- **Relevance:** 4-bit attention weights could cut model size by 4-8x vs FP16/FP32. The stability principles (matching low-precision recomputation in backward pass) are critical for training a quantized model from scratch within the time budget.

### 3. HESTIA: Hessian-Guided Differentiable QAT for Extremely Low-Bit LLMs
- **Authors:** Guoan Wang, Feiyu Wang, Zongwei Lv, et al.
- **Date:** 2026-01-28 | **ID:** [2601.20745](https://arxiv.org/abs/2601.20745)
- **Summary:** Replaces rigid step functions with temperature-controlled softmax relaxation guided by Hessian trace metrics for sensitivity-aware quantization. Shows 5.39% zero-shot improvements for ternary (1.58-bit) quantized 1B models.
- **Relevance:** Ternary quantization (1.58 bits/weight) is extremely aggressive -- a model with ~67M "logical" parameters would only need ~13MB of storage. The Hessian-guided approach helps allocate precision where it matters most.

### 4. StableQAT: Stable Quantization-Aware Training at Ultra-Low Bitwidths
- **Authors:** Tianyi Chen, Sihan Chen, Xiaoyi Qu, et al.
- **Date:** 2026-01-27 | **ID:** [2601.19320](https://arxiv.org/abs/2601.19320)
- **Summary:** Unified QAT framework that stabilizes ultra low-bit training via a discrete Fourier analysis surrogate for backpropagation, strictly generalizing the Straight-Through Estimator (STE) with smooth, bounded gradients. Works at 2-4 bit regimes with negligible overhead.
- **Relevance:** Training stability at 2-4 bits is a key challenge for parameter golf. This paper's Fourier-based gradient surrogate could replace the standard STE and avoid the training instability that often ruins low-bit runs.

### 5. Sparse-BitNet: 1.58-bit LLMs are Naturally Friendly to Semi-Structured Sparsity
- **Authors:** Di Zhang, Xun Wu, Shaohan Huang, et al.
- **Date:** 2026-03-05 | **ID:** [2603.05168](https://arxiv.org/abs/2603.05168)
- **Summary:** Shows that 1.58-bit BitNet models are naturally compatible with N:M structured sparsity, achieving up to 1.30x speedup in both training and inference by combining ternary quantization with sparsity.
- **Relevance:** Combining 1.58-bit quantization with structured sparsity is a double compression win. Could pack even more effective parameters into 16MB while also speeding up the 10-minute training window.

### 6. PoT-QAT: Power-of-Two Quantization-Aware Training for LLMs
- **Authors:** Mahmoud Elgenedy
- **Date:** 2026-01-05 | **ID:** [2601.02298](https://arxiv.org/abs/2601.02298)
- **Summary:** Investigates power-of-two weight quantization where only exponents are stored and multiplications become bit shifts. Reports 87.5% memory savings and 3-10x faster inference on GPT-2 124M after QAT.
- **Relevance:** GPT-2 124M is a relevant model scale. PoT quantization with 87.5% memory savings would shrink a 124M-param model from ~250MB (FP16) to ~31MB -- close to the 16MB target. The bit-shift inference is also very fast.

### 7. UniComp: Unified Evaluation of LLM Compression via Pruning, Quantization and Distillation
- **Authors:** Jonathan von Rad, Yong Cao, Andreas Geiger
- **Date:** 2026-02-09 | **ID:** [2602.09130](https://arxiv.org/abs/2602.09130)
- **Summary:** Comprehensive benchmark finding that quantization provides the best overall trade-off between performance, reliability, and efficiency among compression techniques. Compression exhibits consistent knowledge bias patterns.
- **Relevance:** Validates that quantization (not pruning or distillation) should be the primary compression strategy for parameter golf. Good reference for choosing which compression approach to prioritize.

---

## Small Model Training

### 8. SozKZ: Training Efficient Small Language Models for Kazakh from Scratch
- **Authors:** Saken Tukenov
- **Date:** 2026-03-21 | **ID:** [2603.20854](https://arxiv.org/abs/2603.20854)
- **Summary:** Trains Llama-architecture models from 50M to 600M parameters from scratch on 9B tokens with a dedicated 50K BPE tokenizer. The 600M model approaches 2x larger multilingual baselines on domain benchmarks.
- **Relevance:** Demonstrates the practical recipe for training small (50M-600M) Llama models from scratch. The 50M model is directly in parameter-golf territory. Shows that a dedicated tokenizer and architecture tuning matter more than raw scale.

### 9. Task-Specific Efficiency Analysis: When Small Language Models Outperform Large Language Models
- **Authors:** Jinghan Cao, Yu Ma, Xinjin Li, et al.
- **Date:** 2026-03-22 | **ID:** [2603.21389](https://arxiv.org/abs/2603.21389)
- **Summary:** Introduces a Performance-Efficiency Ratio (PER) metric integrating accuracy, throughput, memory, and latency. Finds that 0.5-3B parameter models achieve superior PER scores across all tested NLP tasks.
- **Relevance:** Provides quantitative evidence that small models win on efficiency metrics. For parameter golf, this supports focusing on the 0.5-3B sweet spot and then aggressively quantizing down to 16MB.

### 10. An Empirical Study of SFT-DPO Interaction in Small Language Models
- **Authors:** Yuming Feng, Christy Yang
- **Date:** 2026-03-20 | **ID:** [2603.20100](https://arxiv.org/abs/2603.20100)
- **Summary:** Compares training strategies (SFT, DPO, LoRA, full fine-tuning) on GPT-2-scale decoders. Finds that full fine-tuning consistently outperforms LoRA at this scale, and LoRA does not save wall-clock time on tested hardware.
- **Relevance:** Important negative result for parameter golf: at small model scales, LoRA may not help. Full-parameter training is the better strategy when the model already fits in memory, which a 16MB model certainly does.

### 11. FlashHead: Efficient Drop-In Replacement for Classification Head in LM Inference
- **Authors:** Wilhelm Tranheden, Shahnawaz Ahmed, Devdatt Dubhashi, et al.
- **Date:** 2026-03-15 | **ID:** [2603.14591](https://arxiv.org/abs/2603.14591)
- **Summary:** Replaces the standard output classification head (which maps hidden dim to vocab size) with an efficient alternative, delivering up to 1.75x inference speedup while maintaining accuracy. Targets small architectures on consumer devices.
- **Relevance:** The output embedding/unembedding layer is often the largest single component in a small LM. An efficient replacement could free up parameter budget for more expressive hidden layers.

---

## Test-Time Training

### 12. Beyond Test-Time Training: Learning to Reason via Hardware-Efficient Optimal Control
- **Authors:** Peihao Wang, Shan Yang, Xijun Wang, et al.
- **Date:** 2026-03-10 | **ID:** [2603.09221](https://arxiv.org/abs/2603.09221)
- **Summary:** Introduces Test-Time Control (TTC) layers that perform finite-horizon LQR planning over latent states at inference time. Improves math reasoning by up to +27.8% on MATH-500 and 2-3x Pass@8 on competition math.
- **Relevance:** If parameter golf allows inference-time compute, TTC layers could boost a small model's effective reasoning ability without increasing stored parameters. The "hardware-efficient" framing suggests low overhead.

### 13. Test-Time Training with KV Binding Is Secretly Linear Attention
- **Authors:** Junchen Liu, Sven Elflein, Or Litany, et al.
- **Date:** 2026-02-24 | **ID:** [2602.21204](https://arxiv.org/abs/2602.21204)
- **Summary:** Shows that TTT architectures can be reframed as learned linear attention operators rather than memorization mechanisms. Provides architectural simplifications and fully parallel formulations that preserve performance.
- **Relevance:** Linear attention is more parameter-efficient than standard softmax attention. If TTT layers can be simplified to linear attention, this could be a compact architecture choice that still adapts at test time.

### 14. SR-TTT: Surprisal-Aware Residual Test-Time Training
- **Authors:** Swamynathan V P
- **Date:** 2026-02-26 | **ID:** [2603.06642](https://arxiv.org/abs/2603.06642)
- **Summary:** Augments TTT models with loss-gated sparse memory, maintaining O(1) memory for low-entropy context while using exact attention only for critical (high-surprisal) tokens.
- **Relevance:** A small model could use this approach to selectively attend to the hardest parts of the input, effectively increasing capacity where it matters without parameter overhead.

---

## Architecture and Other

### 15. NanoNet: Parameter-Efficient Learning with Label-Scarce Supervision for Lightweight Text Mining
- **Authors:** Qianren Mao, Yashuo Luo, Ziqi Qin, et al.
- **Date:** 2026-02-05 | **ID:** [2602.06093](https://arxiv.org/abs/2602.06093)
- **Summary:** Uses online knowledge distillation to create multiple small models that learn from each other via mutual learning regularization, minimizing both training cost and supervision requirements.
- **Relevance:** The mutual-learning approach between small models could be adapted for parameter golf -- train multiple tiny models that teach each other, then keep the best one. Works well with limited data/compute.

---

## Most Promising Papers for Parameter Golf

These five papers are the most directly actionable for building a competitive 16MB model trained in 10 minutes:

| Rank | Paper | Key Technique | Why It Matters |
|------|-------|--------------|----------------|
| 1 | **Sparse-BitNet** (2603.05168) | 1.58-bit weights + N:M sparsity | Double compression: ternary quantization plus structured sparsity could pack ~80M effective parameters into 16MB with training speedups. |
| 2 | **StableQAT** (2601.19320) | Fourier-based gradient surrogate for ultra-low-bit QAT | Solves the training stability problem that makes low-bit QAT hard. Essential for reliably training a 2-4 bit model in only 10 minutes. |
| 3 | **pQuant** (2602.22592) | Decoupled 1-bit + high-precision branches | Clever architecture: most weights at 1 bit with a small high-precision branch for sensitive parameters. Maximizes capacity per byte. |
| 4 | **SozKZ** (2603.20854) | 50M-600M Llama from scratch | Practical recipe for training tiny Llama models. The 50M variant is directly parameter-golf-sized and shows dedicated tokenizers matter. |
| 5 | **Beyond TTT / TTC** (2603.09221) | Test-time optimal control layers | If inference compute is allowed, TTC layers add reasoning ability (+27.8% on math) without adding stored parameters -- a free lunch for parameter golf scoring. |

### Quick-Start Strategy

Based on this survey, a strong parameter golf approach would be:

1. **Architecture:** Start with a small Llama/GPT-2 style transformer (~50-80M parameters at full precision).
2. **Quantization:** Apply 1.58-bit (ternary) or 2-bit QAT using StableQAT's Fourier gradient surrogate for training stability.
3. **Sparsity:** Layer on N:M structured sparsity (Sparse-BitNet approach) for additional compression.
4. **Training:** Use full-parameter training (not LoRA -- per the SFT-DPO study, it does not help at small scale).
5. **Inference boost:** If allowed, add TTC layers or TTT mechanisms for test-time adaptation.

This combination could yield an effective ~60-80M parameter model compressed into 16MB with competitive task performance.

# v8: Kitchen Sink (All Non-Conflicting Improvements)

## Hypothesis
Combine ALL improvements that don't conflict with each other:
- Int5 MLP quantization (saves space)
- 12 layers (uses saved space)
- MTP 2 heads (better training signal, free at inference)
- BigramHash 4096 (better embeddings)
- Warmdown 4500 (better convergence)
- NTK-RoPE eval at 2816 (longer eval context)
- Partial RoPE 24 dims (more positional info)

## Changes
**Code change** -- uses v4's train_gpt.py (12 layers + int5 MLP) with all env vars stacked.

## Run Command

```bash
# 8xH100 full run
MTP_NUM_HEADS=2 MTP_LOSS_WEIGHT=0.2 \
BIGRAM_VOCAB_SIZE=4096 \
WARMDOWN_ITERS=4500 \
EVAL_SEQ_LEN=2816 \
ROPE_DIMS=24 \
RUN_ID=v8_kitchen_sink SEED=1337 \
torchrun --standalone --nproc_per_node=8 experiments/v4_12layers_int5/train_gpt.py
```

## Expected Gain (optimistic)
- 12 layers + int5: -0.005 to -0.015 BPB
- MTP: -0.003 to -0.005 BPB
- BigramHash 4096: -0.001 BPB
- Warmdown 4500: -0.001 BPB
- NTK eval: -0.001 to -0.003 BPB
- RoPE 24: -0.0005 BPB
- **Combined: -0.011 to -0.025 BPB** (if additive)
- Target: sub-1.100 BPB

## Risk
- Highest risk experiment -- many moving parts
- Training will be slowest (12 layers + MTP)
- If individual experiments don't show gains, this won't either
- Run individual experiments first to validate components

## Ablation Plan
If this works, systematically remove one component at a time to measure marginal
contributions. This helps identify which components are essential.

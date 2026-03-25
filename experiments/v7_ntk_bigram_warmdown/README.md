# v7: NTK-RoPE Eval + BigramHash 4096 + Warmdown 4500

## Hypothesis
Stack three quick-win improvements that target different axes:
- NTK-RoPE eval (longer context at eval)
- BigramHash 4096 (better input embeddings)
- Warmdown 4500 (longer warmdown for better convergence)

## Changes
**None** -- env vars only.

## Run Command

```bash
# 8xH100 run
EVAL_SEQ_LEN=2816 BIGRAM_VOCAB_SIZE=4096 WARMDOWN_ITERS=4500 \
RUN_ID=v7_ntk_bigram_warmdown SEED=1337 \
torchrun --standalone --nproc_per_node=8 experiments/baseline_from_sota/train_gpt.py
```

## Expected Gain
- NTK eval 2816: -0.001 to -0.003 BPB
- BigramHash 4096: -0.001 BPB
- Warmdown 4500: -0.001 BPB
- Combined: -0.003 to -0.005 BPB (if additive)

## Risk
- Longer warmdown means less time at full learning rate
- NTK eval at 2816 may slow evaluation phase
- Effects may not be fully additive

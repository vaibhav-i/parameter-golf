# v6: Int5 MLP + 12 Layers + MTP 2 heads

## Hypothesis
Triple stack: int5 MLP frees space for a 12th layer, and MTP provides better
training signal to help the deeper model converge in the limited training budget.

## Changes
**Code change** -- uses v4's train_gpt.py (12 layers + int5 MLP) with MTP env vars.

## Run Command

```bash
# 8xH100 run
MTP_NUM_HEADS=2 MTP_LOSS_WEIGHT=0.2 \
RUN_ID=v6_int5_12layers_mtp SEED=1337 \
torchrun --standalone --nproc_per_node=8 experiments/v4_12layers_int5/train_gpt.py
```

## Expected Gain
- 12th layer: -0.005 to -0.015 BPB
- Int5 MLP: ~0 to -0.001 BPB (may slightly hurt)
- MTP: -0.003 to -0.005 BPB
- Combined: -0.008 to -0.021 BPB (optimistic)

## Risk
- Training significantly slower per step (12 layers + MTP overhead)
- May not complete enough steps in 10 minutes
- Int5 + deeper model compounds quantization noise

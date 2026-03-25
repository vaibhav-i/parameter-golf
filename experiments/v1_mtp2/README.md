# v1: Multi-Token Prediction (2 heads)

## Hypothesis
MTP auxiliary training heads improve representation quality during training.
Heads are automatically excluded from the exported model (line 1303), so there's
zero artifact size cost. Expected BPB gain: -0.003 to -0.005.

## Changes
**None** -- env vars only. MTP is already implemented in the SOTA code
(lines 640-780) but disabled by default (`MTP_NUM_HEADS=0`).

## How It Works
- 2 extra linear heads predict tokens at positions +2 and +3
- Auxiliary loss weighted at 0.2x is added to the main cross-entropy loss
- At export time, `mtp_heads` params are filtered out (line 1303)
- Training cost: ~2 extra forward passes through the heads per step
- Inference cost: zero (heads are discarded)

## Run Command

```bash
# Single GPU smoke test
MTP_NUM_HEADS=2 MTP_LOSS_WEIGHT=0.2 \
RUN_ID=v1_mtp2 \
torchrun --standalone --nproc_per_node=1 experiments/baseline_from_sota/train_gpt.py

# Full 8xH100 run
MTP_NUM_HEADS=2 MTP_LOSS_WEIGHT=0.2 \
RUN_ID=v1_mtp2 SEED=1337 \
torchrun --standalone --nproc_per_node=8 experiments/baseline_from_sota/train_gpt.py
```

## Sweep Variants
- `MTP_NUM_HEADS=1 MTP_LOSS_WEIGHT=0.1` -- minimal overhead
- `MTP_NUM_HEADS=2 MTP_LOSS_WEIGHT=0.2` -- recommended starting point
- `MTP_NUM_HEADS=3 MTP_LOSS_WEIGHT=0.3` -- aggressive

## Risk
- Training may slow down ~5-10% from extra forward passes
- MTP weight too high could hurt main loss convergence
- Need to verify heads are properly excluded from artifact size

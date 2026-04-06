# exp_polar_express: Polar Express Newton-Schulz

Base: PR #1394 (clarkkev SP8192 + SDClip + GPTQ embeddings, 1.08563 BPB)

## Change
Replaces standard fixed-coefficient Newton-Schulz (5 steps, fixed a/b/c) with
Polar Express per-step minimax-optimal coefficients (arXiv:2505.16932).
Uses 4 steps instead of 10 — same orthogonalization quality, ~2ms/step faster.

## Expected gain
~−0.002 BPB via time savings → more training steps. PR #1344 confirmed 1.0923 BPB.

## Run
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py

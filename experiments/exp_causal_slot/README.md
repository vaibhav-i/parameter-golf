# exp_causal_slot: Causal SLOT Eval Adaptation

Base: PR #1394 (clarkkev SP8192 + SDClip, 1.08563 BPB)

## Change
Adds causal SLOT eval-time adaptation (PR #1333 approach, context-only delta optimization).
Per-window: optimize delta [1,1,dim] on context tokens (AdamW, 16 steps), score stride tokens with delta.
Weights frozen. Delta re-initialized per window. Single left-to-right pass.

## Expected gain
−0.013 BPB (confirmed on SP4096 stack, PR #1333). May differ on SP8192 base.

## Run
SLOT_ENABLED=1 SLOT_STEPS=16 SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
# Without SLOT (baseline):
SLOT_ENABLED=0 SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py

# v3: NTK-RoPE Extended Context Evaluation

## Hypothesis
Evaluating with a longer context window (via NTK-RoPE frequency scaling) lets the
model use more context per prediction, reducing BPB. The Rotary class already
implements NTK scaling automatically when `seq_len > train_seq_len` (lines 452-455).

## Changes
**None** -- env vars only. NTK-RoPE is already built into the Rotary class.
The Rotary's `train_seq_len=1024`, so when eval_seq_len > 1024, NTK scaling
is automatically applied: `new_base = base * (scale ^ (dim/(dim-2)))`.

Training already runs at seq_len=2048 with NTK applied (2048 > 1024).
This experiment tests evaluating at even longer contexts (2816 = 1.375x).

## Key Insight
The WarmdownQuantization entry found optimal eval at 1.375x training context.
Longer context = more information per prediction = lower BPB, if the model
can extrapolate. NTK-RoPE enables this extrapolation.

## Run Command

```bash
# Evaluate at 2816 tokens (1.375x training context of 2048)
EVAL_SEQ_LEN=2816 \
RUN_ID=v3_ntk_eval_2816 \
torchrun --standalone --nproc_per_node=8 experiments/baseline_from_sota/train_gpt.py

# Evaluate at 3072 tokens (1.5x)
EVAL_SEQ_LEN=3072 \
RUN_ID=v3_ntk_eval_3072 \
torchrun --standalone --nproc_per_node=8 experiments/baseline_from_sota/train_gpt.py

# Evaluate at 4096 tokens (2x)
EVAL_SEQ_LEN=4096 \
RUN_ID=v3_ntk_eval_4096 \
torchrun --standalone --nproc_per_node=8 experiments/baseline_from_sota/train_gpt.py
```

## Sweep Variants
| EVAL_SEQ_LEN | Scale | Notes |
|---|---|---|
| 2048 | 1.0x | Baseline (default) |
| 2816 | 1.375x | Optimal per WarmdownQuantization |
| 3072 | 1.5x | Moderate extrapolation |
| 4096 | 2.0x | Aggressive extrapolation |

## Risk
- Longer eval context means slower evaluation (more windows to process)
- Very long contexts may degrade quality if NTK extrapolation breaks down
- Must stay within 10-minute eval time budget
- Sliding window eval (stride=64) already provides good context overlap

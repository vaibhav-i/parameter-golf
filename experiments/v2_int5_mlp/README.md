# v2: Int5 MLP Quantization

## Hypothesis
MLP weights are more robust to quantization than attention weights. Using int5
(clip_range=15) for MLP and int6 (clip_range=31) for attention saves ~1.86MB of
artifact space while preserving model quality.

## Changes
**Code change** in `mixed_quantize_int6()` (line 924-928):
- MLP weights quantized with `clip_range=15` (int5, 5-bit effective)
- Attention weights remain at `clip_range=31` (int6, 6-bit effective)
- Meta records `int5` or `int6` type for each parameter

Reference: `records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/train_gpt.py` line 366

## Size Savings
- MLP weights are ~60% of total model parameters
- Int5 saves 1 bit per value for MLP: ~1.86MB savings
- This headroom could fund a 12th layer (see v4)

## Run Command

```bash
# 8xH100 run
RUN_ID=v2_int5_mlp SEED=1337 \
torchrun --standalone --nproc_per_node=8 experiments/v2_int5_mlp/train_gpt.py
```

## Risk
- Int5 quantization error may degrade MLP representations
- GPTQ-lite clip search (lines 889-899) may need re-tuning for int5 range
- Need to verify artifact stays under 16MB

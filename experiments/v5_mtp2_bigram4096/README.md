# v5: MTP 2 heads + BigramHash 4096

## Hypothesis
Stack two "free" improvements: MTP improves training signal (heads discarded at
export), larger BigramHash improves input embeddings. Both should be additive
since they target different parts of the pipeline.

## Changes
**None** -- env vars only.

## Run Command

```bash
# 8xH100 run
MTP_NUM_HEADS=2 MTP_LOSS_WEIGHT=0.2 BIGRAM_VOCAB_SIZE=4096 \
RUN_ID=v5_mtp2_bigram4096 SEED=1337 \
torchrun --standalone --nproc_per_node=8 experiments/baseline_from_sota/train_gpt.py
```

## Expected Gain
- MTP: -0.003 to -0.005 BPB
- BigramHash 4096: -0.001 BPB
- Combined: -0.004 to -0.006 BPB (if additive)

## Risk
- BigramHash 4096 increases embedding table size slightly
- Both techniques add minor training overhead

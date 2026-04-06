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

## Running with SP1024 (available now)

SP8192 is not yet in the upstream manifest — only SP1024 is available. With SP1024
results will be ~0.03–0.05 BPB worse than SP8192 frontier numbers, but techniques
transfer directly and the relative gain from Polar Express NS is still measurable.

```bash
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

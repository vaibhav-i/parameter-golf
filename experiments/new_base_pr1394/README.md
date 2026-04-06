# new_base_pr1394: New Base Stack (PR #1394, clarkkev)

Base: PR #1394 (clarkkev) — Vocab4096 + SDClip + GPTQ embeddings, 1.08563 BPB

## What This Is

This is the new recommended base stack for all experiments as of 2026-04-04.
PR #1394 builds on PR #1218 (Vocab4096 + MLP 4x + WD=0.090) and adds:
- Spectral/delta clipping (SDClip) on weight updates
- GPTQ quantization applied to embedding layers as well as transformer weights
- Result: 1.08563 BPB, no SLOT, no TTT

Why rebase here: the WD=0.090 + all-int6 GPTQ synergy means weights compress more
uniformly, and depth recurrence (which failed on the old stack) works on this stack.

## Run (SP8192, competition target)

```bash
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Running with SP1024 (available now)

SP8192 is not yet in the upstream manifest — only SP1024 is available. With SP1024
results will be ~0.03–0.05 BPB worse than SP8192 frontier numbers. Use this to
validate the base stack and iterate on technique ideas before moving to SP8192.

```bash
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Expected baseline on SP1024: approximately 1.11–1.13 BPB (vs 1.08563 BPB on SP8192).

## Techniques to stack on this base

Priority order (from research/2026-04-04-full-scan-brief.md):
1. Pre-quant TTT: AdamW on EMA weights before GPTQ (−0.022 BPB, PR #1306)
2. MuonEq-R: 2-line gradient row-normalization (−0.001 BPB)
3. Causal SLOT: context-only, 8 steps (−0.009 BPB, PR #1306/1276) — see exp_causal_slot/
4. N-gram agreement via exponential tilting (−0.003 BPB, PR #1302)
5. Polar Express NS + GPTQ damp=0.005 (−0.006 BPB) — see exp_polar_express/
6. Streaming log-bias (−0.015 BPB estimate) — see exp_log_bias/

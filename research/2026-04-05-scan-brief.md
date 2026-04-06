# April 5–6 Competition Scan Brief

**Date**: 2026-04-05 (scan covers events through ~04-06)
**Clean frontier**: 1.08563 BPB (clarkkev, PR #1394)

---

## Major Rule Change: Pre-Quant TTT RULED ILLEGAL

PRs #1350 and #1351 closed April 5 by @resouer. Any `ttt_adapt_adamw(model, val_tokens)` that touches val tokens before scoring is illegal. This was the "biggest legal gain" in the old priority list — now dead.

---

## SLOT Situation (Updated)

- **Standard SLOT**: Illegal (PR #1240)
- **Batched causal SLOT**: Leaky — overlapping windows sharing one delta leaks gradients
- **Per-window SLOT (our fix)**: Fresh delta per window, processed sequentially — likely safe
- **Status**: No official @valerio-oai ruling yet on causal SLOT variants

---

## New Techniques Discovered (April 5–6)

### Optimizer Improvements

| Technique | Paper | Status |
|-----------|-------|--------|
| Polar Express NS | arXiv:2505.16932 | Implemented in exp_polar_express |
| Muon+ | arXiv:2602.21545 | Not yet implemented |
| Newton-Muon | arXiv:2604.01472 | Not yet implemented |
| MUD optimizer | arXiv:2603.17970 | Not yet implemented |

### Quantization

| Technique | Source | Status |
|-----------|--------|--------|
| MR-GPTQ (Hadamard rotation) | PR #1400, arXiv:2603.29078 | Implemented in exp_mr_gptq |
| Q-ROAR | arXiv:2510.00028 | Not yet implemented |

### Eval-Side

| Technique | Source | Gain | Status |
|-----------|--------|------|--------|
| Streaming log-bias (ETLB) | PR #1399 | −0.002 | In exp_log_bias |
| N-gram exponential tilting | PR #1302 | −0.003 | Not yet implemented |
| wPoE | arXiv:2511.10660 | −0.005–0.020 | Not yet implemented |
| dTTT | PRs #1406/1408 | ~−0.006? | Legality unresolved |

---

## Ranked Experiment Queue for Tomorrow

### Tier 1 (run first, high confidence)
1. `new_base_pr1394` — PR #1394 architecture (depth recurrence, SDClip, MuonEq-R)
2. `exp_polar_express` — Polar Express NS coefficients on new_base
3. `exp_log_bias` — streaming log-bias at eval, causal and legal

### Tier 2 (run after Tier 1)
4. `exp_mr_gptq` — Hadamard rotation before GPTQ, zero training overhead
5. `exp_causal_slot` — per-window SLOT (leakage fixed)

### Tier 3 (8×H100 multi-seed only)
- Run Tier 1+2 winners with 3 seeds for statistical significance

---

## SP1024 Run Commands

```bash
# Pull latest
cd /workspace/parameter-golf && git pull origin experiments/research-and-variants

# 1. new_base_pr1394
RUN_ID=new_base_pr1394_sp1024 DATA_PATH=./data/datasets/fineweb10B_sp1024/ TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model VOCAB_SIZE=1024 torchrun --standalone --nproc_per_node=1 experiments/new_base_pr1394/train_gpt.py

# 2. exp_polar_express
RUN_ID=exp_polar_express_sp1024 DATA_PATH=./data/datasets/fineweb10B_sp1024/ TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model VOCAB_SIZE=1024 torchrun --standalone --nproc_per_node=1 experiments/exp_polar_express/train_gpt.py

# 3. exp_log_bias
RUN_ID=exp_log_bias_sp1024 DATA_PATH=./data/datasets/fineweb10B_sp1024/ TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model VOCAB_SIZE=1024 LOG_BIAS_ENABLED=1 torchrun --standalone --nproc_per_node=1 experiments/exp_log_bias/train_gpt.py

# 4. exp_mr_gptq
RUN_ID=exp_mr_gptq_sp1024 DATA_PATH=./data/datasets/fineweb10B_sp1024/ TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model VOCAB_SIZE=1024 MR_GPTQ_ENABLED=1 torchrun --standalone --nproc_per_node=1 experiments/exp_mr_gptq/train_gpt.py

# 5. exp_causal_slot
RUN_ID=exp_causal_slot_sp1024 DATA_PATH=./data/datasets/fineweb10B_sp1024/ TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model VOCAB_SIZE=1024 SLOT_ENABLED=1 torchrun --standalone --nproc_per_node=1 experiments/exp_causal_slot/train_gpt.py
```

---

## Key Things to Watch

- **new_base vs baseline**: If < 1.36, PR #1394 architecture improvements work on SP1024
- **exp_polar_express**: Should be ~0.001 below new_base — if larger, coefficients are very impactful
- **exp_mr_gptq**: Zero training overhead, gain shows at eval only. If ≥0.003, include in all future experiments
- **exp_causal_slot**: May be slow (16 optim steps per eval window). Watch for timeout

## Next Steps After Runs

1. Rebase on whichever base wins
2. Implement Muon+ (trivial, 5 lines)
3. Consider N-gram exponential tilting (−0.003, no legality issues)
4. Wait for @valerio-oai ruling on dTTT before implementing

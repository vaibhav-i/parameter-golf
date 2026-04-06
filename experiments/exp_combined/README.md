# exp_combined

Combines three eval-only techniques on top of the `exp_muon_plus` base stack
(SP8192, MLP 4x, WD=0.090, Polar Express NS, Muon+).

**Training is identical to exp_muon_plus. Only the post-quantization evaluation changes.**

## Techniques stacked at eval

### 1. Streaming Log-Bias (ETLB / Nacrith arXiv:2602.19626)
A bias vector `b ∈ R^vocab` is maintained across the validation set.
After scoring each token `x_t`:
```
b[x_t] += lr
b -= lr * softmax(logits_t + b)
```
Applied strictly causally — b is updated only after scoring. No model weight changes.

Hyperparameters:
- `LOG_BIAS_ENABLED` (default 1)
- `LOG_BIAS_LR` (default 0.001)
- `LOG_BIAS_RESET` (default 0) — if 1, resets b to zero at each window start

### 2. Causal SLOT (per-window hidden-state delta)
Each sliding window gets a fresh `delta ∈ R^{1,1,model_dim}` optimized via AdamW
on the window's context tokens only (positions before the scored stride). The delta
is added to the hidden states: `hidden_adapted = forward_hidden(x) + delta`.
Scoring uses the frozen optimized delta.

Windows are processed one at a time (no cross-window batching) to prevent gradient
leakage via overlapping windows, per clarkkev's review of the original batched design.

Hyperparameters:
- `SLOT_ENABLED` (default 1)
- `SLOT_STEPS` (default 16) — AdamW steps to optimize delta on context
- `SLOT_LR` (default 0.005)

## Eval mode selection logic

| `SLOT_ENABLED` | `LOG_BIAS_ENABLED` | eval function used |
|---|---|---|
| 1 | 1 | `eval_val_sliding_causal_slot_with_log_bias` (combined) |
| 1 | 0 | `eval_val_sliding_causal_slot` |
| 0 | 1 | `eval_val_sliding_log_bias` |
| 0 | 0 | `eval_val_sliding` (regular sliding window, if `SLIDING_WINDOW_ENABLED=1`) |

## Architecture changes vs exp_muon_plus

- `GPT` gains `forward_hidden()` and `hidden_to_logits()` split methods (required by SLOT)
- `forward_logits()` delegates to these two methods (no change in output)

## Expected gains

- Log-bias alone: ~−0.002 BPB (from exp_log_bias baseline)
- Causal SLOT alone: −0.013 BPB claimed (PR #1333), but per-window (batch=1) reduces gains significantly
- Combined: additive if effects are orthogonal; log-bias corrects marginal distribution,
  SLOT adapts hidden-state shift — these are complementary in principle

## Run command

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Disable individual techniques:
```bash
SLOT_ENABLED=0 LOG_BIAS_ENABLED=0 torchrun ...   # regular sliding window
SLOT_ENABLED=0 torchrun ...                       # log_bias only
LOG_BIAS_ENABLED=0 torchrun ...                   # causal SLOT only
```

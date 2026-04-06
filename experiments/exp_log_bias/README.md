# exp_log_bias: Streaming Online Log-Bias (Nacrith)

Base: PR #1394 (clarkkev SP8192 + SDClip, 1.08563 BPB)

## Change
Adds streaming online log-bias correction at eval time (arXiv:2602.19626, Tacconelli 2026).
Zero artifact cost. Strictly causal. Single pass.

Mechanism: maintain b ∈ R^vocab. Before each token: logits += b.
After each token: b += lr * (one_hot(x_t) - softmax(logits+b)).
lr=0.001, no momentum, no reset across windows.

## Expected gain
−0.015 BPB (confirmed on enwik8 in Nacrith paper). Untested in competition.

## Run
LOG_BIAS_ENABLED=1 LOG_BIAS_LR=0.001 SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
# Without log-bias (baseline):
LOG_BIAS_ENABLED=0 SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py

## Ablations to try
LOG_BIAS_LR=0.0001   # slower adaptation
LOG_BIAS_LR=0.01     # faster adaptation (may overshoot)
LOG_BIAS_RESET=1     # reset b per window (weaker but safer)

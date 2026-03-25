# v9: LoRA Test-Time Training on SOTA Base

## Hypothesis
LoRA TTT adapts the model per-document at evaluation time. The original LoRA TTT
submission achieved 1.1928 BPB on a weak 9-layer baseline. On the much stronger
SOTA base (1.1228), document-isolated eval + LoRA adaptation should yield further
gains. Expected: -0.002 to -0.008 BPB.

## Changes
**Major code changes** -- this is the most complex experiment:

1. **TTT hyperparameters** added to Hyperparameters class (lines 87-92)
2. **CausalSelfAttention.forward** extended with optional `q_delta`/`v_delta` args
3. **Block.forward** extended with optional `q_delta_fn`/`v_delta_fn` args
4. **GPT.forward_ttt** new method: forward pass with LoRA, returns per-token loss
5. **LoRA classes**: `BatchedLinearLoRA`, `BatchedTTTLoRA`
6. **TTT eval function**: `eval_val_ttt_lora` with document isolation, chunked eval
7. **TTT eval call** at the end of main(), after standard eval

Training is completely unchanged -- LoRA TTT is eval-time only.

## How It Works
1. Find document boundaries (BOS token = ID 1) in validation data
2. Split each document into 256-token chunks within 2048-token context windows
3. For each chunk: SCORE first (record BPB), then TRAIN (one Adam step on LoRA)
4. Reset LoRA parameters between documents (no cross-document leakage)

## Key Hyperparameters
| Env Var | Default | Description |
|---|---|---|
| TTT_LORA_RANK | 8 | LoRA adapter rank |
| TTT_LORA_LR | 0.01 | Adam LR for LoRA during TTT |
| TTT_CHUNK_SIZE | 256 | New tokens scored per chunk |
| TTT_EVAL_SEQ_LEN | 2048 | Context window for TTT eval |
| TTT_BATCH_SIZE | 64 | Documents processed in parallel |
| TTT_ENABLED | 1 | Enable/disable TTT eval |

## Run Command

```bash
# 8xH100 run (full training + TTT eval)
RUN_ID=v9_lora_ttt SEED=1337 \
torchrun --standalone --nproc_per_node=8 experiments/v9_lora_ttt/train_gpt.py

# Disable TTT to compare standard eval
TTT_ENABLED=0 RUN_ID=v9_no_ttt SEED=1337 \
torchrun --standalone --nproc_per_node=8 experiments/v9_lora_ttt/train_gpt.py
```

## Expected Performance
- TTT eval time: ~60-120 seconds on 8xH100 (well within 600s budget)
- Memory: ~3-5GB extra for LoRA params + activations
- BPB gain: -0.002 to -0.008 (primarily from document isolation)

## Porting Notes
- FlashAttention 3 uses (B,T,H,D) layout -- LoRA deltas added before reshape, compatible
- ValueEmbedding and XSA are orthogonal to LoRA -- they compose cleanly
- The eval uses `eval_model` (quantized + dequantized) not the training model

## Risk
- TTT eval may be slower than expected with 11 layers + FlashAttn 3
- LoRA backward through FlashAttn 3 may need fallback to standard attention
- Document isolation benefit may overlap with existing sliding-window eval
- Most complex experiment -- highest chance of bugs

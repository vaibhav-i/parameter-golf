# v4: 12 Layers + Int5 MLP Quantization

## Hypothesis
Int5 MLP quantization saves ~1.86MB, enough to fit a 12th transformer layer
within the 16MB artifact limit. More depth = better representations = lower BPB.

## Changes
1. **Default NUM_LAYERS changed to 12** (line 46): `NUM_LAYERS=12`
2. **Int5 MLP quantization** (line 924-928): same as v2

## Architecture
- 12 layers / 512 dim / 4 KV heads (GQA) / MLP 3x
- Encoder: 6 layers, Decoder: 6 layers (with U-Net skip connections)
- XSA on last 4 layers (layers 8-11)

## Size Budget
- 11-layer SOTA: ~15.55MB compressed
- Int5 MLP savings: ~1.86MB
- 12th layer cost: ~1.5MB (512*512*4 attention + 512*1536*2 MLP, quantized)
- Expected total: ~15.2MB (under 16MB)

## Run Command

```bash
# 8xH100 run
RUN_ID=v4_12layers_int5 SEED=1337 \
torchrun --standalone --nproc_per_node=8 experiments/v4_12layers_int5/train_gpt.py

# Override back to 11 layers for comparison
NUM_LAYERS=11 RUN_ID=v4_11layers_int5 SEED=1337 \
torchrun --standalone --nproc_per_node=8 experiments/v4_12layers_int5/train_gpt.py
```

## Risk
- Training may be ~10% slower per step (12 vs 11 layers)
- May not reach enough training steps in 10 minutes (~6000+ steps needed)
- 12th layer adds ~1.5MB; verify total artifact stays under 16MB
- More layers + int5 may compound quantization error

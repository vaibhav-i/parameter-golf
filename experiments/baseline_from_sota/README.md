# Experiment Baseline: SOTA (1.1228 bpb)

This is the current SOTA `train_gpt.py` from signalrush (2026-03-22), used as a starting point for experiments.

## What's already in this code (combined from all top entries)

### From baseline (1.2244)
- 512-dim transformer, 8 attention heads, 4 KV heads (GQA)
- Tied embeddings (input/output share weights)
- Muon optimizer for matrix params, AdamW for embeddings

### From FP16 Embed entry (1.2197)
- FP16 tied embeddings (skip int8 quant on embed layer)

### From Seq2048 entries (1.206)
- 2048 training sequence length (up from 512)

### From Int6 QAT entries (1.158-1.150)
- Int6 per-row quantization with STE (straight-through estimator)
- MLP 3x expansion (hidden=1536) — enabled by int6 space savings
- zstd-22 compression (saves ~1.5MB vs zlib)
- Sliding window eval (stride=64)

### From SmearGate/BigramHash entries (1.155-1.145)
- SmearGate: learned gate blending current + previous token embeddings
- BigramHash: hash-based bigram embedding table (2048 buckets, dim=128)
- Orthogonal initialization

### From 11L XSA entries (1.130-1.127)
- 11 transformer layers with U-Net skip connections
- XSA (Exclusive Self Attention) on last 4 layers
- EMA weight averaging (decay=0.997)

### From Partial RoPE entry (1.1248)
- Partial RoPE: positional encoding on only 16/64 head dims
- LN Scale factor: 1/sqrt(layer_idx+1) for stability

### From SOTA entry (1.1228)
- GPTQ-lite: per-row clip percentile search (5 candidates) at export
- Extended warmdown (3500 iterations)
- Late QAT threshold at 0.15

## Key env vars for quick experiments

```bash
# BigramHash bucket sweep
BIGRAM_VOCAB_SIZE=4096   # default: 2048, try: 4096, 8192, 10240

# Warmdown sweep
WARMDOWN_ITERS=4000      # default: 3500, try: 4000, 4500, 5000

# Partial RoPE sweep
ROPE_DIMS=24             # default: 16, try: 8, 24, 32

# QK-Norm gain
QK_GAIN_INIT=1.0         # default: 1.5, try: 1.0, 2.0, 0.0 (disable)

# LN Scale toggle
LN_SCALE=0               # default: 1 (enabled), 0 to disable

# Layer count
NUM_LAYERS=12            # default: 11, try: 12 (check artifact size!)
```

## Run locally (Apple Silicon smoke test)
Note: This script requires CUDA/FlashAttention3. For local testing, use `train_gpt_mlx.py` instead.

## Run on RunPod (1xH100)
```bash
cd /workspace/parameter-golf
python3 data/cached_challenge_fineweb.py --variant sp1024
RUN_ID=experiment_name SEED=1337 \
torchrun --standalone --nproc_per_node=1 experiments/baseline_from_sota/train_gpt.py
```

## Run on RunPod (8xH100, for submission timing)
```bash
RUN_ID=experiment_name SEED=1337 \
torchrun --standalone --nproc_per_node=8 experiments/baseline_from_sota/train_gpt.py
```

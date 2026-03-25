#!/bin/bash
# Quick-win experiment sweep runner
# Usage: bash experiments/run_sweep.sh <experiment_name>
# Run from the parameter-golf root directory

set -euo pipefail

EXPERIMENT=${1:-"all"}
SEEDS="1337 42 2024"
SCRIPT="experiments/baseline_from_sota/train_gpt.py"
NPROC=${NPROC:-1}  # Set NPROC=8 for 8xH100 submission runs

run_experiment() {
    local name=$1
    shift
    for seed in $SEEDS; do
        echo "=== Running $name seed=$seed ==="
        RUN_ID="${name}_s${seed}" SEED=$seed "$@" \
            torchrun --standalone --nproc_per_node=$NPROC $SCRIPT \
            2>&1 | tee "experiments/logs/${name}_s${seed}.log"
    done
}

mkdir -p experiments/logs

case "$EXPERIMENT" in
    "bigram_4096")
        run_experiment bigram4096 BIGRAM_VOCAB_SIZE=4096
        ;;
    "bigram_8192")
        run_experiment bigram8192 BIGRAM_VOCAB_SIZE=8192
        ;;
    "bigram_10240")
        run_experiment bigram10240 BIGRAM_VOCAB_SIZE=10240
        ;;
    "warmdown_4000")
        run_experiment wd4000 WARMDOWN_ITERS=4000
        ;;
    "warmdown_4500")
        run_experiment wd4500 WARMDOWN_ITERS=4500
        ;;
    "warmdown_5000")
        run_experiment wd5000 WARMDOWN_ITERS=5000
        ;;
    "rope_8")
        run_experiment rope8 ROPE_DIMS=8
        ;;
    "rope_24")
        run_experiment rope24 ROPE_DIMS=24
        ;;
    "rope_32")
        run_experiment rope32 ROPE_DIMS=32
        ;;
    "no_ln_scale")
        run_experiment no_lnscale LN_SCALE=0
        ;;
    "qk_gain_1.0")
        run_experiment qkgain1 QK_GAIN_INIT=1.0
        ;;
    "baseline")
        run_experiment baseline_reproduce
        ;;
    "all_quick")
        echo "Running all quick-win experiments..."
        bash "$0" baseline
        bash "$0" bigram_4096
        bash "$0" bigram_8192
        bash "$0" warmdown_4000
        bash "$0" warmdown_4500
        bash "$0" rope_8
        bash "$0" rope_24
        bash "$0" rope_32
        bash "$0" qk_gain_1.0
        bash "$0" no_ln_scale
        ;;
    *)
        echo "Available experiments:"
        echo "  baseline        - Reproduce SOTA (1.1228)"
        echo "  bigram_4096     - BigramHash 4096 buckets"
        echo "  bigram_8192     - BigramHash 8192 buckets"
        echo "  bigram_10240    - BigramHash 10240 buckets (may exceed 16MB!)"
        echo "  warmdown_4000   - Warmdown at 4000 iters"
        echo "  warmdown_4500   - Warmdown at 4500 iters"
        echo "  warmdown_5000   - Warmdown at 5000 iters"
        echo "  rope_8          - Partial RoPE 8/64 dims"
        echo "  rope_24         - Partial RoPE 24/64 dims"
        echo "  rope_32         - Partial RoPE 32/64 dims"
        echo "  no_ln_scale     - Disable LN Scale"
        echo "  qk_gain_1.0     - QK gain = 1.0 (default 1.5)"
        echo "  all_quick       - Run all quick-win experiments"
        echo ""
        echo "Set NPROC=8 for 8xH100 runs"
        ;;
esac

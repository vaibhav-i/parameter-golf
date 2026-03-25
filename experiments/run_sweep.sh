#!/bin/bash
# Quick-win experiment sweep runner
# Usage: bash experiments/run_sweep.sh <experiment_name>
# Run from the parameter-golf root directory

set -euo pipefail

EXPERIMENT=${1:-"all"}
SEEDS="1337 42 2024"
SCRIPT_BASELINE="experiments/baseline_from_sota/train_gpt.py"
SCRIPT_INT5="experiments/v2_int5_mlp/train_gpt.py"
SCRIPT_12L_INT5="experiments/v4_12layers_int5/train_gpt.py"
SCRIPT_TTT="experiments/v9_lora_ttt/train_gpt.py"
NPROC=${NPROC:-1}  # Set NPROC=8 for 8xH100 submission runs

run_experiment() {
    local name=$1
    local script=${2:-$SCRIPT_BASELINE}
    shift 2 2>/dev/null || shift 1
    for seed in $SEEDS; do
        echo "=== Running $name seed=$seed ==="
        RUN_ID="${name}_s${seed}" SEED=$seed "$@" \
            torchrun --standalone --nproc_per_node=$NPROC $script \
            2>&1 | tee "experiments/logs/${name}_s${seed}.log"
    done
}

mkdir -p experiments/logs

case "$EXPERIMENT" in
    # --- Group A: Env-var-only quick wins (baseline script) ---
    "baseline")
        run_experiment baseline_reproduce "$SCRIPT_BASELINE"
        ;;
    "bigram_4096")
        run_experiment bigram4096 "$SCRIPT_BASELINE" BIGRAM_VOCAB_SIZE=4096
        ;;
    "bigram_8192")
        run_experiment bigram8192 "$SCRIPT_BASELINE" BIGRAM_VOCAB_SIZE=8192
        ;;
    "bigram_10240")
        run_experiment bigram10240 "$SCRIPT_BASELINE" BIGRAM_VOCAB_SIZE=10240
        ;;
    "warmdown_4000")
        run_experiment wd4000 "$SCRIPT_BASELINE" WARMDOWN_ITERS=4000
        ;;
    "warmdown_4500")
        run_experiment wd4500 "$SCRIPT_BASELINE" WARMDOWN_ITERS=4500
        ;;
    "warmdown_5000")
        run_experiment wd5000 "$SCRIPT_BASELINE" WARMDOWN_ITERS=5000
        ;;
    "rope_8")
        run_experiment rope8 "$SCRIPT_BASELINE" ROPE_DIMS=8
        ;;
    "rope_24")
        run_experiment rope24 "$SCRIPT_BASELINE" ROPE_DIMS=24
        ;;
    "rope_32")
        run_experiment rope32 "$SCRIPT_BASELINE" ROPE_DIMS=32
        ;;
    "no_ln_scale")
        run_experiment no_lnscale "$SCRIPT_BASELINE" LN_SCALE=0
        ;;
    "qk_gain_1.0")
        run_experiment qkgain1 "$SCRIPT_BASELINE" QK_GAIN_INIT=1.0
        ;;

    # --- v1: MTP (env var only) ---
    "v1_mtp2")
        run_experiment v1_mtp2 "$SCRIPT_BASELINE" MTP_NUM_HEADS=2 MTP_LOSS_WEIGHT=0.2
        ;;

    # --- v2: Int5 MLP (code change) ---
    "v2_int5_mlp")
        run_experiment v2_int5_mlp "$SCRIPT_INT5"
        ;;

    # --- v3: NTK-RoPE eval (env var only) ---
    "v3_ntk_2816")
        run_experiment v3_ntk_2816 "$SCRIPT_BASELINE" EVAL_SEQ_LEN=2816
        ;;
    "v3_ntk_3072")
        run_experiment v3_ntk_3072 "$SCRIPT_BASELINE" EVAL_SEQ_LEN=3072
        ;;
    "v3_ntk_4096")
        run_experiment v3_ntk_4096 "$SCRIPT_BASELINE" EVAL_SEQ_LEN=4096
        ;;

    # --- v4: 12 Layers + Int5 MLP (code change) ---
    "v4_12layers_int5")
        run_experiment v4_12layers_int5 "$SCRIPT_12L_INT5"
        ;;

    # --- v5: MTP + BigramHash 4096 (env var only) ---
    "v5_mtp2_bigram4096")
        run_experiment v5_mtp2_bigram4096 "$SCRIPT_BASELINE" MTP_NUM_HEADS=2 MTP_LOSS_WEIGHT=0.2 BIGRAM_VOCAB_SIZE=4096
        ;;

    # --- v6: Int5 + 12 Layers + MTP (code + env var) ---
    "v6_int5_12layers_mtp")
        run_experiment v6_int5_12layers_mtp "$SCRIPT_12L_INT5" MTP_NUM_HEADS=2 MTP_LOSS_WEIGHT=0.2
        ;;

    # --- v7: NTK + BigramHash + Warmdown (env var only) ---
    "v7_ntk_bigram_warmdown")
        run_experiment v7_ntk_bigram_warmdown "$SCRIPT_BASELINE" EVAL_SEQ_LEN=2816 BIGRAM_VOCAB_SIZE=4096 WARMDOWN_ITERS=4500
        ;;

    # --- v8: Kitchen Sink (code + env var) ---
    "v8_kitchen_sink")
        run_experiment v8_kitchen_sink "$SCRIPT_12L_INT5" MTP_NUM_HEADS=2 MTP_LOSS_WEIGHT=0.2 BIGRAM_VOCAB_SIZE=4096 WARMDOWN_ITERS=4500 EVAL_SEQ_LEN=2816 ROPE_DIMS=24
        ;;

    # --- v9: LoRA TTT (code change) ---
    "v9_lora_ttt")
        run_experiment v9_lora_ttt "$SCRIPT_TTT"
        ;;

    # --- Batch runners ---
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
    "all_variants")
        echo "Running all experiment variants..."
        bash "$0" v1_mtp2
        bash "$0" v2_int5_mlp
        bash "$0" v3_ntk_2816
        bash "$0" v4_12layers_int5
        bash "$0" v5_mtp2_bigram4096
        bash "$0" v6_int5_12layers_mtp
        bash "$0" v7_ntk_bigram_warmdown
        bash "$0" v8_kitchen_sink
        bash "$0" v9_lora_ttt
        ;;
    *)
        echo "Available experiments:"
        echo ""
        echo "  Quick wins (env var only, baseline script):"
        echo "  baseline          - Reproduce SOTA (1.1228)"
        echo "  bigram_4096       - BigramHash 4096 buckets"
        echo "  bigram_8192       - BigramHash 8192 buckets"
        echo "  bigram_10240      - BigramHash 10240 buckets (may exceed 16MB!)"
        echo "  warmdown_4000     - Warmdown at 4000 iters"
        echo "  warmdown_4500     - Warmdown at 4500 iters"
        echo "  warmdown_5000     - Warmdown at 5000 iters"
        echo "  rope_8            - Partial RoPE 8/64 dims"
        echo "  rope_24           - Partial RoPE 24/64 dims"
        echo "  rope_32           - Partial RoPE 32/64 dims"
        echo "  no_ln_scale       - Disable LN Scale"
        echo "  qk_gain_1.0       - QK gain = 1.0 (default 1.5)"
        echo ""
        echo "  Experiment variants:"
        echo "  v1_mtp2            - Multi-token prediction (2 heads)"
        echo "  v2_int5_mlp        - Int5 quantization for MLP weights"
        echo "  v3_ntk_2816        - NTK-RoPE eval at 2816 tokens"
        echo "  v3_ntk_3072        - NTK-RoPE eval at 3072 tokens"
        echo "  v3_ntk_4096        - NTK-RoPE eval at 4096 tokens"
        echo "  v4_12layers_int5   - 12 layers + int5 MLP"
        echo ""
        echo "  Combinations:"
        echo "  v5_mtp2_bigram4096     - MTP + BigramHash 4096"
        echo "  v6_int5_12layers_mtp   - Int5 + 12 layers + MTP"
        echo "  v7_ntk_bigram_warmdown - NTK eval + BigramHash + warmdown"
        echo "  v8_kitchen_sink        - All improvements combined"
        echo "  v9_lora_ttt            - LoRA test-time training"
        echo ""
        echo "  Batch runners:"
        echo "  all_quick          - Run all quick-win experiments"
        echo "  all_variants       - Run all experiment variants"
        echo ""
        echo "Set NPROC=8 for 8xH100 runs"
        ;;
esac

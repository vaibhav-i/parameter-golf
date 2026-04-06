# exp_mr_gptq: Hadamard-Rotated GPTQ (MR-GPTQ)

Base: PR #1394 (clarkkev SP8192 + SDClip, 1.08563 BPB)

## Change
Applies random Hadamard rotation to weight matrices before GPTQ quantization.
Spreads outliers evenly across columns → dramatically lower quantization MSE.
PR #1400 claims 68× lower MSE at zero artifact overhead.

## Expected gain
~−0.005 to −0.010 BPB (estimated, no direct ablation yet in competition)

## Run
MR_GPTQ_ENABLED=1 VOCAB_SIZE=1024 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
SEED=1337 torchrun --standalone --nproc_per_node=1 train_gpt.py

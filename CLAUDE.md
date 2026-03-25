# Parameter Golf - Claude Code Notes

## What This Repo Is

OpenAI's **Model Craft Challenge: Parameter Golf** — compete to train the best language model that fits in a **16MB artifact** and trains in **under 10 minutes on 8xH100s**, evaluated by bits-per-byte compression on the FineWeb validation set.

Contest runs: March 18 – April 30, 2026.
Current SOTA: **1.1228 bpb** (signalrush, 2026-03-22).

---

## Hard Constraints

| Constraint | Limit |
|---|---|
| Artifact size | **16,000,000 bytes** (decimal MB, not MiB) |
| Training time | **≤ 10 min** on 8xH100 SXM |
| Evaluation time | **≤ 10 min** on 8xH100 (separate from training) |
| Beat SOTA by | **≥ 0.005 nats** at **p < 0.01** (need multiple run logs) |

**Artifact = code bytes (`train_gpt.py`) + compressed model bytes.** No external downloads, network calls, or training data access during evaluation.

---

## Submission Checklist

Submissions are **PRs that add a single new folder** to:
```
records/track_10min_16mb/YYYY-MM-DD_<ShortName>/
```

Required files in that folder:

- [ ] **`README.md`** — explain the approach in reasonable detail
- [ ] **`submission.json`** — metadata (see format below)
- [ ] **Train logs** — enough runs to show statistical significance (typically 3 runs averaged)
- [ ] **`train_gpt.py`** — complete, runnable script; must compile and run from within the records folder

Optional:
- `requirements.txt` if using extra packages (note them in README too)

---

## submission.json Format

```json
{
  "author": "Your Name",
  "github_id": "your_github_handle",
  "name": "Short descriptive run name",
  "blurb": "1–2 sentence description of what this run does.",
  "date": "2026-03-XX T00:00:00Z",
  "val_loss": 2.07269931,
  "val_bpb": 1.2243657,
  "bytes_total": 15863489,
  "bytes_code": 47642
}
```

---

## SOTA Record Requirements

1. Beat existing SOTA by **≥ 0.005 nats**, demonstrated at **p < 0.01** via multiple run logs.
   - Waived for pure systems optimizations (no ML changes).
2. If tokenizer or dataset is changed, prove `val_bpb` is correctly calculated.
3. Must reproducibly run in **≤ 10 min on 8xH100s**.

---

## Non-Record Submissions

OK to submit even if you don't beat SOTA, as long as:
- Satisfies the 16MB artifact limit
- Includes full justification for the approach
- Interesting / novel / negative results are welcome
- Unlimited-compute runs are accepted if clearly labeled in README

Non-record submissions go in the same folder structure.

---

## Key Rules / FAQ

- **No validation data during training** — cannot compress it into the 16MB as a "paid prefix"
- **Test-time training on val tokens** is only allowed *after* those tokens have been evaluated
- **External compute**: Hyperparameter tuning across runs is fine; brute-forcing seeds is not
- **Packages**: Any package is fine (FlashAttention, etc.) — just add `requirements.txt`. Cannot sneak in extra compute/capabilities via custom libraries
- **Broken scripts** will not be accepted — test that `train_gpt.py` runs from within the records folder

---

## Getting Started Locally (Apple Silicon)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install mlx numpy sentencepiece huggingface-hub datasets tqdm

# Download FineWeb (1 shard for smoke test)
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1

# Run MLX smoke test
RUN_ID=mlx_smoke ITERATIONS=200 TRAIN_BATCH_TOKENS=8192 \
VAL_LOSS_EVERY=0 VAL_BATCH_SIZE=8192 python3 train_gpt_mlx.py
```

## Getting Started on RunPod (CUDA)

```bash
# Use the official template: https://console.runpod.io/deploy?template=y5cejece4j&ref=nl2r56th
# All Python deps are pre-installed in the image

cd /workspace && git clone https://github.com/openai/parameter-golf.git && cd parameter-golf
python3 data/cached_challenge_fineweb.py --variant sp1024

# Single H100 baseline run
RUN_ID=baseline_sp1024 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

Expected baseline: ~1.2244 bpb, model under 16MB.

---

## Key Files

| File | Purpose |
|---|---|
| `train_gpt.py` | Main CUDA training script (baseline) |
| `train_gpt_mlx.py` | Apple Silicon / MLX version |
| `data/cached_challenge_fineweb.py` | Downloads FineWeb dataset |
| `data/tokenizer_specs.json` | Tokenizer configs |
| `records/track_10min_16mb/` | All leaderboard submissions |
| `records/track_non_record_16mb/` | Non-record submissions |

---

## Research Corpus (March 24, 2026)

Comprehensive research in `research/` — see `research/README.md` for full index.

### SOTA Stack (1.1228 bpb)
11 layers, 512 dim, 4 KV heads (GQA), MLP 3x expansion, int6 QAT (STE), GPTQ-lite clip search, EMA, FP16 embeddings, tied embeddings, Partial RoPE (16/64 dims), XSA on last 4 layers, sliding window eval (stride 64), Muon optimizer (WD=0.04), zstd-22 compression, warmdown at step 3500.

### Top Untried Opportunities (ranked by expected value)
1. **BigramHash 10240 buckets** in SOTA — config change, ~0.001-0.003 BPB
2. **Warmdown sweep** beyond 3500 — hyperparameter tuning, ~0.001-0.002 BPB
3. **QK-Norm ablation** — already in SOTA code (lines 521-522), test removing/tuning q_gain_init
4. **Multi-token prediction** — auxiliary heads discarded at inference (zero size cost), ~0.003-0.008 BPB
5. **LoRA TTT on SOTA base** — only tested on baseline (got 1.1928), huge room on modern base, ~0.005-0.015 BPB
6. **Depth recurrence** (Cycle(rev) weight sharing) — 12 effective layers at 6-layer cost, ~0.005-0.015 BPB
7. **Simplified hyperconnections** — generalizes U-Net skip pattern, ~0.002-0.004 BPB

### Key Research Files
- `research/strategy.md` — actionable playbook with technique rankings + next steps
- `research/meta-review-and-plan.md` — week-by-week execution plan + experiment templates + compute grant notes
- `research/technique-glossary.md` — reference for all techniques
- `research/evolution-chain.md` — solution dependency graph + untried combination matrix
- `research/arxiv-deep-dive.md` — targeted papers on depth recurrence, ternary quant, MTP, TTT

### Implementation Guides
- `research/sota-code-walkthrough.md` — annotated SOTA train_gpt.py (architecture, line-by-line, modification points, gotchas)
- `research/quick-win-specs.md` — copy-paste experiment commands (all via env vars, no code edits needed)
- `research/lora-ttt-port-guide.md` — LoRA TTT porting to SOTA: code diffs, timing estimates, optimization strategies

### SOTA Code Quick Reference
- `bigram_vocab_size`: line 77, env `BIGRAM_VOCAB_SIZE` (default 2048)
- `warmdown_iters`: line 38, env `WARMDOWN_ITERS` (default 3500)
- `rope_dims`: line 80, env `ROPE_DIMS` (default 16)
- `qk_gain_init`: line 44 (default 1.5)
- `ln_scale`: line 81, env `LN_SCALE` (default 1, set 0 to disable)
- Artifact headroom: ~444 KB free (15.55 MB used of 16 MB)

---

## Experiment Variants (March 25, 2026)

Ready-to-run experiment code in `experiments/`. Use `bash experiments/run_sweep.sh` for the full menu.

### Env-var-only experiments (use baseline script)
| Experiment | Command | Expected Gain |
|---|---|---|
| v1: MTP 2 heads | `MTP_NUM_HEADS=2 MTP_LOSS_WEIGHT=0.2` | -0.003 to -0.005 |
| v3: NTK eval 2816 | `EVAL_SEQ_LEN=2816` | -0.001 to -0.003 |
| v5: MTP + BigramHash | `MTP_NUM_HEADS=2 MTP_LOSS_WEIGHT=0.2 BIGRAM_VOCAB_SIZE=4096` | -0.004 to -0.006 |
| v7: NTK + Bigram + WD | `EVAL_SEQ_LEN=2816 BIGRAM_VOCAB_SIZE=4096 WARMDOWN_ITERS=4500` | -0.003 to -0.005 |

### Code-change experiments (modified train_gpt.py)
| Experiment | Script | Key Change |
|---|---|---|
| v2: Int5 MLP | `experiments/v2_int5_mlp/train_gpt.py` | int5 quant for MLP, int6 for attention |
| v4: 12L + Int5 | `experiments/v4_12layers_int5/train_gpt.py` | 12 layers + int5 MLP |
| v8: Kitchen Sink | Uses v4 script + env vars | All improvements stacked |
| v9: LoRA TTT | `experiments/v9_lora_ttt/train_gpt.py` | Test-time LoRA adaptation |

### Experiment Priorities (run in this order)
1. **v1_mtp2** — free gain, zero risk
2. **v3_ntk_2816** — free gain from longer eval context
3. **v2_int5_mlp** — validate int5 quality
4. **v5_mtp2_bigram4096** — stack free wins
5. **v4_12layers_int5** — if int5 works, add 12th layer
6. **v9_lora_ttt** — highest EV but most complex

---

## Support

OpenAI Discord: `#parameter-golf-discussions` and `#parameter-golf-announcements`
Compute grants: https://openai.com/index/parameter-golf/#credit-form
Participant form: https://jobs.ashbyhq.com/openai/form/open-ai-challenge-parameter-golf

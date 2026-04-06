# Parameter Golf - Claude Code Notes

## What This Repo Is

OpenAI's **Model Craft Challenge: Parameter Golf** — compete to train the best language model that fits in a **16MB artifact** and trains in **under 10 minutes on 8xH100s**, evaluated by bits-per-byte compression on the FineWeb validation set.

Contest runs: March 18 – April 30, 2026.
Current SOTA: **1.1147 bpb** (abaybektursun, 2026-03-25).

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

### SOTA Stack (1.1147 bpb — 2026-03-25, abaybektursun)
11 layers, 512 dim, 8 GQA heads / 4 KV heads, MLP 3x with LeakyReLU(0.5)^2, Full Hessian GPTQ int6 with AR self-gen calibration (model generates own calib data, no external data during quant), XSA on ALL 11 layers, BigramHash 3072×112, Partial RoPE (16/64 dims), LN Scale, VE128 layers 9-10, SmearGate, U-Net skips, EMA(0.997) + Tight SWA(every 50), Parallel Muon + Parameter Banking, warmdown 4000, LZMA preset=9 compression, selective ±1 pruning, sliding eval stride=16, temperature scaling. **No TTT** (dropped after 25 failed attempts).

### Top Opportunities (ranked by expected value, updated 2026-03-30)
1. **Multi-token prediction** — auxiliary heads discarded at inference (zero artifact cost), ~0.003-0.008 BPB
2. **Port Full Hessian GPTQ + AR self-gen calibration** to our experiments — biggest single technique gain
3. **LZMA compression** — replace zstd, better compression ratio
4. **Int5 MLP + 12th layer** — save bytes via int5 on MLP, fund an extra layer
5. **QK-Norm ablation** — test removing/tuning q_gain_init
6. ~~Depth recurrence~~ — **CONFIRMED DEAD** (+0.025 BPB worse, PR #363)
7. ~~TTT~~ — **NEUTRAL/NEGATIVE** on current stack (25 failed attempts by SOTA author)

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
### New SOTA (2026-03-25) Quick Reference
- Script: `records/track_10min_16mb/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072/train_gpt.py`
- Run: `BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112 WARMDOWN_ITERS=4000 TARGET_MB=15.9 SEED=314 torchrun --standalone --nproc_per_node=8 train_gpt.py`
- Key: `xsa_last_n=11`, `bigram_vocab_size=3072`, `bigram_dim=112`, `warmdown_iters=4000`
- Compression: LZMA preset=9 (not zstd)
- GPTQ: Full Hessian with AR self-generated calibration data
- Artifact: ~15.91 MB (tight — ~90KB headroom)

---

## Experiment Variants (updated 2026-03-30)

Ready-to-run experiment code in `experiments/`. Use `bash experiments/run_sweep.sh` for the full menu.

**IMPORTANT**: Our experiment baseline is the 2026-03-22 signalrush code. The new SOTA (2026-03-25) adds Full GPTQ + AR self-gen + XSA-all + BigramHash 3072 + LZMA. We should rebase on the new SOTA code for maximum competitiveness.

### Quick wins on NEW SOTA (env vars only, use new SOTA script)
| Experiment | Command | Expected Gain |
|---|---|---|
| MTP 2 heads | `MTP_NUM_HEADS=2 MTP_LOSS_WEIGHT=0.2` | -0.003 to -0.005 |
| XSA-all (already default in new SOTA) | `XSA_LAST_N=11` | already included |
| BigramHash 3072 (already default in new SOTA) | `BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112` | already included |
| Warmdown 4000 (already default in new SOTA) | `WARMDOWN_ITERS=4000` | already included |

### Code-change experiments (modified train_gpt.py, based on old baseline)
| Experiment | Script | Key Change |
|---|---|---|
| v2: Int5 MLP | `experiments/v2_int5_mlp/train_gpt.py` | int5 quant for MLP + LeakyReLU + stride-16 + temp scaling |
| v4: 12L + Int5 | `experiments/v4_12layers_int5/train_gpt.py` | 12 layers + int5 MLP + LeakyReLU + stride-16 + temp scaling |
| v9: Full-model TTT | `experiments/v9_lora_ttt/train_gpt.py` | SGD TTT (likely dead — neutral on SOTA stack) |

### Experiment Priorities (updated 2026-03-30)
1. **Rebase on new SOTA** (2026-03-25 code) — get Full GPTQ + AR self-gen + LZMA for free
2. **MTP 2 heads on new SOTA** — free gain, zero artifact cost
3. **Int5 MLP on new SOTA** — validate int5 quality, potentially fund 12th layer
4. **12 layers + int5 on new SOTA** — if int5 works, biggest architectural gain
5. ~~v9 TTT~~ — **deprioritized** (SOTA author found TTT neutral/negative after 25 attempts)

---

## Support

OpenAI Discord: `#parameter-golf-discussions` and `#parameter-golf-announcements`
Compute grants: https://openai.com/index/parameter-golf/#credit-form
Participant form: https://jobs.ashbyhq.com/openai/form/open-ai-challenge-parameter-golf

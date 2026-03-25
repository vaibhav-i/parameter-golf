# Parameter Golf Research Corpus

Deep-dive research into the OpenAI Parameter Golf challenge, produced March 24, 2026.

**Current SOTA:** 1.1228 bpb (signalrush, 2026-03-22)
**Contest deadline:** April 30, 2026

## Research Files

### Core Analysis
| File | Description |
|------|-------------|
| [leaderboard-analysis.md](leaderboard-analysis.md) | All 23 entries analyzed — scores, techniques, ablations, code snippets |
| [technique-glossary.md](technique-glossary.md) | Reference guide for every technique (quantization, attention, embeddings, optimizers, etc.) |
| [evolution-chain.md](evolution-chain.md) | Dependency graph of solutions + untried combination matrix |

### Cross-References
| File | Description |
|------|-------------|
| [modded-nanogpt-insights.md](modded-nanogpt-insights.md) | Modded-NanoGPT techniques + what transfers to parameter golf |
| [arxiv-survey.md](arxiv-survey.md) | Initial arxiv survey (~15 papers across QAT, small models, TTT) |
| [arxiv-deep-dive.md](arxiv-deep-dive.md) | Targeted arxiv deep dive — depth recurrence, ternary quant, multi-token prediction, TTT, SSMs, attention efficiency |

### Strategy
| File | Description |
|------|-------------|
| [strategy.md](strategy.md) | Synthesis — technique rankings, integration opportunities, concrete next steps |
| [meta-review-and-plan.md](meta-review-and-plan.md) | Meta-review of all research + phased execution plan + experiment templates + compute grant notes |

### Literature Reviews
| File | Description |
|------|-------------|
| [citation-graph-review.md](citation-graph-review.md) | Citation-graph research from modded-nanogpt + slowrun references via Semantic Scholar |
| [advanced-literature-review.md](advanced-literature-review.md) | Keyword-based literature review across 10 topic areas |

### Implementation Guides
| File | Description |
|------|-------------|
| [sota-code-walkthrough.md](sota-code-walkthrough.md) | Annotated line-by-line walkthrough of SOTA train_gpt.py — architecture, code map, modification points, gotchas |
| [lora-ttt-port-guide.md](lora-ttt-port-guide.md) | Concrete porting guide: LoRA TTT eval on SOTA 11L architecture — code diffs, timing, memory, optimization |
| [quick-win-specs.md](quick-win-specs.md) | Copy-paste ready experiment specs for BigramHash sweep, warmdown sweep, QK-Norm ablation, Partial RoPE sweep |

## Quick Start

1. Read **strategy.md** for the actionable playbook
2. Read **quick-win-specs.md** for copy-paste experiment commands (all run via env vars, no code edits)
3. Read **sota-code-walkthrough.md** to understand the SOTA codebase before modifying it
4. Read **lora-ttt-port-guide.md** when ready for the highest-EV medium-effort experiment
5. Reference **technique-glossary.md** when you need to understand a specific technique
6. Check **evolution-chain.md** for untried combinations to explore

## Key Discovery
**QK-Norm is already in the SOTA code** (lines 521-522). The quick-win spec pivots this to an ablation test instead.

## Key Numbers

| Metric | Value |
|--------|-------|
| Baseline BPB | 1.2244 |
| Current SOTA | 1.1228 |
| Total improvement so far | -0.1016 BPB |
| Biggest single technique | Sliding window eval (-0.033 BPB) |
| Target for competition | < 1.100 BPB |
| Estimated compute needed | ~428 H100-hours |

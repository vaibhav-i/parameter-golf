# exp_muon_plus

## Hypothesis

The standard Muon post-Newton-Schulz scaling heuristic:

```python
g *= max(1, g.size(0) / g.size(1)) ** 0.5
```

scales the update by the square root of the row/column ratio of the parameter matrix. This is a shape-ratio heuristic that produces inconsistent effective step sizes across matrices with different aspect ratios (e.g., attention projections vs. MLP up/down projections).

**Muon+** replaces this with RMS normalization targeting RMS=1 per element:

```python
g = g * (g.numel() ** 0.5) / (g.norm() + 1e-7)
```

This normalizes the update so that the RMS of the gradient update is exactly 1.0 across all parameter matrices, regardless of their shape. The resulting effective step size depends only on the learning rate, not on the arbitrary shape of each matrix.

## Expected Effect

- More consistent effective step size across all matrix types (c_q, c_k, c_v, proj, fc, etc.)
- No shape-ratio bias: square matrices and rectangular matrices get the same per-element update magnitude
- May improve convergence stability or allow a higher learning rate

## Controls

| Env var | Default | Description |
|---|---|---|
| `MUON_PLUS` | `1` | Set to `0` to use the original shape-ratio scaling |

## Based On

`exp_polar_express` — Polar Express (4-step minimax NS coefficients, PR #1344) base stack.

## Ablation Plan

1. Run with `MUON_PLUS=1` (default) — Muon+ RMS normalization
2. Run with `MUON_PLUS=0` — baseline shape-ratio scaling, same random seed
3. Compare val_bpb at same wallclock budget

## Implementation Notes

- Zero parameter overhead (no new tensors stored)
- Zero compute overhead vs. original (same number of operations, just different formula)
- The `1e-7` epsilon in the denominator prevents division by zero on degenerate gradient matrices

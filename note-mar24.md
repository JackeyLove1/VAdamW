# Batch `mar24`

Date: 2026-03-24
Branch: `autoresearch/mar24`

## Summary

- Best validated run so far: `945c757`
- Best `val_bpb`: `1.396539`
- Main finding: this setup was severely under-scaled at `DEPTH=1`; increasing model capacity to `DEPTH=4` produced a large gain with only a small VRAM increase.
- Idea status: gradient-variation damping did not help in lightweight AdamW-only form, and the naive Muon-side version was too slow for the fixed-budget setting.

## Experiment Log

### `1c27729` baseline

- Result: `val_bpb=1.628552`, `memory_gb=0.6`
- Status: keep
- Notes: original `train.py` was heavily underutilizing the 5060. The model had only `3.3M` parameters and extremely low effective throughput.

### `50edbb7` gradvar AdamW damping on Adam groups

- Result: `val_bpb=1.628637`, `memory_gb=0.6`
- Status: discard
- Notes: adding gradient-variation damping only to AdamW parameter groups did not move the metric. This is consistent with the code structure: most important matrix parameters are optimized by Muon, so the idea had too little leverage.

### `945c757` depth 4 baseline

- Result: `val_bpb=1.396539`, `memory_gb=0.8`
- Status: keep
- Notes: increasing `DEPTH` from `1` to `4` gave the first significant improvement. This indicates model capacity dominates optimizer micro-tweaks in the current five-minute regime.

### `e01ba24` depth 8

- Result: no valid summary
- Status: crash
- Notes: exceeded the 10 minute wall-clock budget once final evaluation was included. The run was rejected even though training itself was progressing.

### `fcd3a39` Muon gradvar damping

- Result: no valid summary
- Status: crash
- Notes: extending gradient-variation damping directly into the Muon path made each step far too expensive. The implementation needs a much cheaper approximation to be viable in this codebase.

### `7211718` tuned embedding AdamW with grad variation

- Result: interrupted before final summary
- Status: incomplete
- Notes: this run combined a lower embedding LR with lightweight AdamW-side gradvar on top of the stronger `DEPTH=4` baseline. It was interrupted externally before `val_bpb` was printed, so it is not considered a validated result.

### `fb8e9c9` depth 6

- Result: no valid summary
- Status: crash
- Notes: `DEPTH=6` completed the training budget but final evaluation pushed total wall-clock past the 10 minute limit. This confirms that `DEPTH=4` is near the practical upper bound unless evaluation cost is reduced elsewhere.

### `2f3ac24` wider depth 4 (`ASPECT_RATIO=96`)

- Result: no valid summary
- Status: crash
- Notes: widening the four-layer model also exceeded the total runtime budget once evaluation was included. The fixed evaluation cost on this GPU leaves very little headroom beyond the current `DEPTH=4`, `ASPECT_RATIO=64` baseline.

### `eafbe84` larger device batch

- Result: no valid summary
- Status: crash
- Notes: increasing `DEVICE_BATCH_SIZE` to `16` and `TOTAL_BATCH_SIZE` to `8192` did not solve the wall-clock issue. The run reached the end of training but still failed to produce a final summary within the allowed runtime, so it cannot replace the existing baseline.

### `3aef2f4` full attention at depth 4

- Result: no valid summary
- Status: crash
- Notes: replacing `SSSL` with `LLLL` increased runtime enough that the run again missed the overall wall-clock constraint. For this GPU budget, partial local attention appears necessary.

### `5e9992e` scalar-gradvar optimizer variant

- Result: `val_bpb=1.748301`, `memory_gb=0.9`
- Status: discard
- Notes: the optimizer idea ran end-to-end, but the first Muon-side implementation still imposed too much overhead. Training speed collapsed to roughly `1.9M` tokens in the five-minute budget, so the worse metric is confounded by severe throughput loss rather than a clean optimizer comparison.

### `0abe953` RMS-diff gradvar damping for Muon

- Result: no valid summary
- Status: crash
- Notes: replacing full gradient history with per-matrix RMS-difference statistics fixed the throughput collapse, but total wall-clock still crossed the 10 minute limit before a final summary was printed. This approximation is much closer to viable, but still not acceptable under the current runtime rule.

### `1ed83f4` AdamW gradvar damping only

- Result: no valid summary
- Status: crash
- Notes: even restricting gradient-variation damping to the AdamW parameter groups still slowed the run enough to miss the overall runtime target. The current fused step implementation is not a good host for this idea without a different systems-level formulation.

### `b19f340` group-level gradvar LR gating

- Result: `val_bpb=1.750228`, `memory_gb=0.8`
- Status: discard
- Notes: moving the idea fully outside the fused kernels solved the runtime problem, but the optimizer behavior degraded badly. The group-level gate is cheap enough to test quickly, yet at this strength it over-damps learning and does not preserve the useful dynamics of the baseline.

### `15abf40` mild Muon-only gradvar gate

- Result: no valid summary
- Status: crash
- Notes: restricting the cheap group-level gate to Muon only and dropping the damping strength still failed the wall-clock requirement. Training throughput looked acceptable, but the run again stalled long enough in the final stage that it could not be counted as a valid experiment.

### `b16bf22` rewritten gated GradVar-AdamW

- Result: `val_bpb=1.759879`, `memory_gb=0.9`
- Status: discard
- Notes: fully rewriting the AdamW branch and only activating gradient-variation damping in the second half of training still hurt both throughput and final quality. This suggests the current non-Muon parameter groups are too important to perturb with per-parameter history at this budget.

### `5fcdc09` selective GradVar-AdamW groups

- Result: `val_bpb=1.749533`, `memory_gb=0.8`
- Status: discard
- Notes: restricting GradVar-AdamW to the `lm_head` and scalar parameter groups preserved runtime, but still badly underperformed the baseline. The negative effect is therefore not just an embedding-state overhead issue; the update rule itself is misaligned with this setup.

## Current Direction

- Keep `DEPTH=4` as the active baseline.
- If revisiting the idea, prefer low-overhead variants that do not touch Muon's hot path elementwise.
- Future experiments should continue to separate "capacity wins" from "optimizer wins" so the note stays interpretable.

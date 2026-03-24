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

## Current Direction

- Keep `DEPTH=4` as the active baseline.
- If revisiting the idea, prefer low-overhead variants that do not touch Muon's hot path elementwise.
- Future experiments should continue to separate "capacity wins" from "optimizer wins" so the note stays interpretable.

# Run diagnostics (pose height + trajectory webbing)

## What was ruled out by code comparison (2026-06-12)

A line-by-line comparison against PanoLASER (the reference that produces
correct heights) found the two pipelines **functionally identical** in every
place that could explain the height overshoot:

- `panovggt_model.py` is byte-identical between the repos; PRISM's
  `local_points` = `directions * exp(log_d)` uses the exact same ERP
  convention (Y-down, same azimuth) as PanoLASER's
  `unproject_equirectangular_to_points(depth)`, so the point inputs to the
  floor fit are equivalent.
- `camera_poses` are cam-to-world by construction (the model computes
  `world_points = camera_poses @ local_points`), matching how the engine
  uses them.
- `estimate_metric_scale_from_floor`, `align_cam_pts_irls`, and the nvblox
  cubemap TSDF are functionally identical (whitespace/comment diffs only).
- Engine pose math (scale → canonical rebase → Kabsch anchor → global) is
  structurally identical. Remaining deltas: PRISM's Kabsch variant (centered
  positions + free axis vectors vs PanoLASER's position-anchored triads),
  trajectory decimation (PRISM decimates, PanoLASER appends every frame), and
  TSDF cadence (PRISM integrates every frame, PanoLASER keyframes only).

Conclusion: neither bug is explainable statically — they are emergent, so the
engine now records data every run.

## How to use

Diagnostics are **on by default** (`StreamingWindowEngine(debug_dump_dir=...)`,
set to `None` to disable). Each `process_sequence` call writes
`debug_dumps/run_<timestamp>/submap_*.npz` + `sequence.npz` (poses, scales,
floor planes, trajectory — a few KB per submap), and prints per submap:

- `[Diag] cam height above fitted floor (global frame)` — the same quantity
  you eyeball in the viewer (dot vs. floor), computed exactly.
- anchor rotation/translation, min `R[1,1]`, flip count, max trajectory step.

After (or during) a run:

```
python apps/diagnose_run.py            # latest run
python apps/diagnose_run.py debug_dumps/run_YYYYMMDD_HHMMSS
```

It also writes `trajectory_debug.html` (trajectory colored by submap, abnormal
jumps in red) into the run folder.

## How to read the result (decision tree)

1. **Section 2 heights ≈ 1.7 everywhere, but the viewer shows ~2.1:**
   the *mesh floor* is misplaced, not the trajectory. Suspects: TSDF
   integration (pose flips firing — see section 4; cubemap sampling), or
   floor-level drift layering multiple floors (section 3 spread).
2. **Section 2 heights ≈ 2.1:** the floor fit / scale chain is wrong —
   RANSAC is latching onto a plane above the real floor (check `raw|d|`
   column and `floor_conf`), or the scale blend drifts (section 1 jumps).
3. **Section 5 abnormal jumps at submap boundaries:** webbing comes from
   Kabsch anchoring discontinuities; jumps *inside* submaps point at the
   model's raw poses; no jumps at all means the webbing is a viewer-side
   artifact.

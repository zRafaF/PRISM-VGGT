# prism-v2: alignment and guard fixes

Branch `prism-v2`. Engine-only changes in `prism_vggt/engine.py` and
`prism_vggt/utils/geometry.py`; no API change (`process_sequence`, `get_poses`,
`get_current_cloud` behave as before). New: `engine.get_alignment_report()`.

## What was wrong

1. **Levelling only looked at one frame.** The world was levelled from the floor of the
   first window's *middle* frame only. When that frame's floor confidence was below 0.4
   the world stayed in camera 0's frame for the whole run (Z = camera forward). On the
   2026-08-10 bundle that is 30 of 60 PRISM runs (camera tilt 90 deg in `poses.tum`).
2. **Every Z-up guard then acted on a horizontal axis.** The Z re-level nudged the map
   sideways, and the cubemap upside-down guard flipped frames 180 deg at random (the
   down axis' Z component is noise around 0 in an unlevelled world), for the TSDF only.
   This is the "glitches out and messes up the transform" failure.
3. **The alignment study was not group-vs-group.** Sim(3)/SE(3) were fit from the 4
   overlap camera centres (+ axes); SL(4) from ~3000 dense overlap points. And after
   the warm-up the "Sim(3)" arm froze its scale (`LOCK_SCALE_AFTER_FIRST=1`), so it was
   SE(3) in practice, while SL(4) kept a free per-window local scale.
4. **SL(4) flip-guard pivot.** The flip correction rotated the homography about the
   world origin, moving the submap by |R - I| x its distance from the origin.
5. **Scale-lock jump.** At the lock, the buffered world was rescaled but the current
   window's overlap targets were not, so the lock window was placed (k-1)|t| off and the
   overlap pin snapped it back.

## What changed

| Area | Change |
|---|---|
| Levelling | Most confident floor among 5 frames of window 0; else a **gravity prior** (mean camera up axis, camera placed at its height); refined once to the first confident floor while the pre-lock buffer still holds the map (`_transform_world_rigid`). `engine.level_source` records which. |
| Guards | Z re-level only in a levelled world (and uses the SL(4) local scale). Upside-down guard no longer rotates frames: a levelled camera whose down axis points clearly up is *not integrated* and counted. Flip guard pivots on the overlap centroid. |
| Alignment | All groups fit from the same dense, pixel-corresponded overlap points (stride grid, masked, range-gated, near points weighted). Sim(3)/SE(3): robust IRLS-Huber weighted Umeyama. SL(4): trimmed DLT, rejected (dense Sim(3) used) above `PRISM_SL4_MAX_NONSIM` % distortion. Camera-centre fit kept as fallback and as the legacy `*_cam` modes. |
| Scale | Warm-up commits the agreement of **world-to-metric ratios** (floor scale / placement scale), which is identical to the old behaviour for a fixed placement scale and correct for a free one. After commit, `sim3` keeps the dense per-window scale (true 7-DoF) unless `PRISM_SCALE_MODE=lock`; `se3` is always locked. |
| Diagnostics | Per-window `align_log` (fit source, dense rmse / inliers, scale, overlap rotation, guard actions) and `get_alignment_report()`. |
| Perception dump | `PRISM_PERC_DUMP=<dir>` saves each window's PanoVGGT output (float16, `PRISM_PERC_DUMP_STRIDE` grid) to replay alignment variants on CPU. |

## Environment flags

| Flag | Default | Meaning |
|---|---|---|
| `PRISM_ALIGN` | `sl4` | `sim3`, `se3`, `sl4` (dense) or `sim3_cam`, `se3_cam` (original camera-centre fits) |
| `PRISM_SCALE_MODE` | `free` | `free` / `lock` scale after commit (dense `sim3` only) |
| `PRISM_ALIGN_STRIDE` | `4` | pixel stride of the dense overlap correspondences |
| `PRISM_DENSE_MIN_INLIERS` | `0.35` | below this inlier fraction the camera-centre fit is used |
| `PRISM_SL4_MAX_NONSIM` | `25` | SL(4) distortion (%) above which the window uses dense Sim(3) |
| `PRISM_LEVEL_PRIOR` | `1` | gravity-prior levelling when window 0 has no confident floor |
| `PRISM_LEVEL_REFINE` | `1` | refine the prior to the first confident floor (pre-integration) |
| `PRISM_UPSIDE_GUARD` | `skip` | `skip` (do not integrate inverted cameras), `flip` (old), `off` |
| `PRISM_PERC_DUMP` / `_STRIDE` | unset / `4` | dump perception outputs |
| `LOCK_SCALE_AFTER_FIRST` | `1` | now only read by the `*_cam` modes |

All the earlier flags (`PRISM_FLIP_GUARD`, `PRISM_Z_RELEVEL*`, `SCALE_*`, ...) are unchanged.

## CPU simulation (not a benchmark)

A synthetic box room, 110 panoramas on a two-lap loop, windows 16/4, a fake PanoVGGT with
6 % per-window native-scale variation, 0.4 deg pose noise, 1 % depth noise and 5 % depth
outliers; nvblox stubbed. Mean of 3 seeds, ATE after Sim(3) alignment.

| Scenario | old sim3 | old sl4 | new sim3 | new sim3 lock | new se3 | new sl4 |
|---|---|---|---|---|---|---|
| normal | 24.7 cm | 6.7 cm | **5.3 cm** | 20.4 cm | 20.4 cm | 5.4 cm |
| no floor in window 0 | 89.4 cm, tilt 90 deg | 6.6 cm, tilt 90 deg, scale 26 % | **5.3 cm**, tilt 0.5 deg | 21.2 cm | 21.2 cm | 5.2 cm |
| bad overlap orientation (25 deg) | 16.4 cm | 11.5 cm | **6.0 cm** | 14.8 cm | 14.8 cm | 4.8 cm |

The simulation reproduces both field symptoms (the 90 deg tilt and SL(4) "looking much
better" than Sim(3)) and shows the gap closing once both use the same data and Sim(3)
keeps its scale free. How much of the lock-vs-free gap is real depends on how much
PanoVGGT's native scale varies between windows; the benchmark has to settle that
(`sim3` vs `sim3` with `PRISM_SCALE_MODE=lock`).

## Still to do

- The README's alignment-study numbers (SL(4) 31 % vs Sim(3) 20 % scale error on loops)
  come from the old, unequal study and must be regenerated.
- vat-monorepo pins `PRISM_ALIGN=sim3`: after bumping the submodule the live system gets
  the dense free-scale Sim(3).

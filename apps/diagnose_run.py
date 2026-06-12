"""Offline analyzer for StreamingWindowEngine debug dumps.

Usage:
    python apps/diagnose_run.py [path/to/debug_dumps/run_YYYYMMDD_HHMMSS]

If no path is given, the most recent run under ./debug_dumps is used.

Reads the per-submap .npz files written by the engine and answers, with real
numbers instead of eyeballing:

  1. HEIGHT BUG: camera height above the RANSAC-fitted floor plane, measured in
     the global frame, per submap. If these sit at ~target height but the viewer
     shows the dots higher above the mesh floor, the mesh floor is misplaced
     (TSDF integration side / pose flips). If these are too high, the floor fit
     or the scale chain is wrong.
  2. WEBBING BUG: consecutive trajectory deltas, largest jumps (attributed to
     submaps), and backtracking segments. Also writes trajectory_debug.html with
     the trajectory colored by submap and suspect segments in red.
  3. CHAIN HEALTH: per-submap anchor rotation/translation, min R[1,1] (pose
     flips), scale evolution, and global floor-level drift across submaps.
"""

import os
import sys
import glob

import numpy as np


def find_latest_run(base="debug_dumps"):
    runs = sorted(glob.glob(os.path.join(base, "run_*")))
    if not runs:
        sys.exit(f"No runs found under ./{base}. Run a sequence first.")
    return runs[-1]


def load_run(run_dir):
    submaps = []
    for path in sorted(glob.glob(os.path.join(run_dir, "submap_*.npz"))):
        submaps.append(dict(np.load(path)))
    seq = None
    seq_path = os.path.join(run_dir, "sequence.npz")
    if os.path.exists(seq_path):
        seq = dict(np.load(seq_path))
    if not submaps:
        sys.exit(f"No submap_*.npz files in {run_dir}")
    return submaps, seq


def fmt(x, p=3):
    return "  nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{p}f}"


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main():
    run_dir = sys.argv[1] if len(sys.argv) > 1 else find_latest_run()
    submaps, seq = load_run(run_dir)
    print(f"Analyzing: {run_dir}  ({len(submaps)} submaps)")
    target = float(submaps[0]["target_camera_height"])

    # ------------------------------------------------------------------ scale
    section("1. SCALE EVOLUTION")
    print(f"{'submap':>6} {'floor_scale':>12} {'floor_conf':>11} {'metric_scale':>13}  flags")
    prev = None
    for k, sm in enumerate(submaps):
        ms = float(sm["metric_scale"])
        flags = []
        if prev is not None and abs(ms - prev) / prev > 0.02:
            flags.append(f"JUMP {100 * (ms - prev) / prev:+.1f}%")
        print(f"{k:>6} {fmt(float(sm['floor_scale'])):>12} {fmt(float(sm['floor_conf'])):>11} "
              f"{ms:>13.4f}  {' '.join(flags)}")
        prev = ms

    # ----------------------------------------------------------------- height
    section(f"2. CAMERA HEIGHT ABOVE FITTED FLOOR (global frame, target={target}m)")
    print(f"{'submap':>6} {'mid':>8} {'min':>8} {'max':>8} {'raw|d|':>8}  flags")
    for k, sm in enumerate(submaps):
        h = sm["cam_heights"]
        plane = sm["floor_plane"]
        if np.all(np.isnan(h)):
            print(f"{k:>6} {'-':>8} {'-':>8} {'-':>8} {'-':>8}  no floor fit")
            continue
        mid = int(sm["mid_idx"])
        flags = []
        if abs(h[mid] - target) > 0.15:
            flags.append("HEIGHT OFF TARGET")
        if (h.max() - h.min()) > 0.3:
            flags.append("TILTED/INCONSISTENT WINDOW")
        print(f"{k:>6} {h[mid]:>8.3f} {h.min():>8.3f} {h.max():>8.3f} "
              f"{abs(float(plane[3])):>8.3f}  {' '.join(flags)}")
    print("\nInterpretation: mid≈target everywhere but viewer shows higher dots")
    print("  -> mesh floor is misplaced (TSDF side). mid>target -> floor fit/scale chain.")

    # ------------------------------------------------------- global floor drift
    section("3. GLOBAL FLOOR LEVEL PER SUBMAP (floor drift => layered mesh floor)")
    print(f"{'submap':>6} {'floor_Y_global':>15} {'normal_Y':>9}  (Y is down; bigger = lower)")
    levels = []
    for k, sm in enumerate(submaps):
        pg = sm["floor_plane_global"]
        if np.any(np.isnan(pg)):
            print(f"{k:>6} {'-':>15} {'-':>9}")
            continue
        n_g, d_g = pg[:3], pg[3]
        if abs(n_g[1]) < 1e-6:
            print(f"{k:>6} {'vertical?!':>15} {n_g[1]:>9.3f}")
            continue
        # Floor Y at the mid camera's (X, Z): n·x + d = 0 -> y = -(nx*x + nz*z + d)/ny
        t_mid = sm["global_poses"][int(sm["mid_idx"])][:3, 3]
        y_floor = -(n_g[0] * t_mid[0] + n_g[2] * t_mid[2] + d_g) / n_g[1]
        levels.append(y_floor)
        print(f"{k:>6} {y_floor:>15.3f} {n_g[1]:>9.3f}")
    if len(levels) > 1:
        print(f"\nFloor level spread across submaps: {max(levels) - min(levels):.3f} m "
              f"({'OK' if max(levels) - min(levels) < 0.10 else 'DRIFTING -> layered floors in mesh'})")

    # ------------------------------------------------------------ chain health
    section("4. CHAIN HEALTH (anchor alignment, pose flips)")
    print(f"{'submap':>6} {'anchor_rot_deg':>14} {'anchor_t_m':>11} {'minR11':>8} {'flips':>6}  flags")
    for k, sm in enumerate(submaps):
        A = sm["anchor_pose"]
        rot = np.degrees(np.arccos(np.clip((np.trace(A[:3, :3]) - 1) / 2, -1, 1)))
        t = np.linalg.norm(A[:3, 3])
        r11 = min(T[1, 1] for T in sm["global_poses"])
        nf = len(sm["flip_frames"])
        flags = []
        if rot > 10:
            flags.append("BIG ANCHOR ROTATION")
        if r11 < 0.5:
            flags.append("POSE NEAR-FLIPPED")
        if nf:
            flags.append("TSDF FLIP FIRED")
        print(f"{k:>6} {rot:>14.2f} {t:>11.3f} {r11:>8.3f} {nf:>6}  {' '.join(flags)}")

    # --------------------------------------------------------------- webbing
    section("5. TRAJECTORY / WEBBING")
    if seq is None or len(seq["trajectory"]) < 3:
        print("No sequence.npz / trajectory too short.")
        return
    traj = seq["trajectory"]
    deltas = np.linalg.norm(np.diff(traj, axis=0), axis=1)
    med = np.median(deltas)
    thresh = max(3 * med, 0.30)
    print(f"{len(traj)} trajectory points. step: median={med:.3f}m p95={np.percentile(deltas, 95):.3f}m "
          f"max={deltas.max():.3f}m | jump threshold={thresh:.3f}m")

    # Attribute trajectory indices to submaps via kept_positions counts
    counts = [len(sm["kept_positions"]) for sm in submaps]
    bounds = np.cumsum([0] + counts)

    def which_submap(idx):
        return int(np.searchsorted(bounds, idx, side="right") - 1)

    jumps = np.where(deltas > thresh)[0]
    if len(jumps) == 0:
        print("No abnormal jumps -> webbing is not from out-of-order/jumping points;")
        print("check the viewer side or marker density instead.")
    else:
        print(f"\n{len(jumps)} abnormal jump(s):")
        for idx in jumps[:20]:
            sm_a, sm_b = which_submap(idx), which_submap(idx + 1)
            boundary = " <-- SUBMAP BOUNDARY" if sm_a != sm_b else ""
            print(f"  traj[{idx}] -> traj[{idx + 1}]: {deltas[idx]:.3f}m "
                  f"(submap {sm_a} -> {sm_b}){boundary}")

    # Backtracking: consecutive segments that reverse direction with large steps
    v = np.diff(traj, axis=0)
    nrm = np.linalg.norm(v, axis=1)
    valid = (nrm[:-1] > 2 * med) & (nrm[1:] > 2 * med)
    cosang = np.einsum("ij,ij->i", v[:-1], v[1:]) / np.maximum(nrm[:-1] * nrm[1:], 1e-9)
    backtracks = np.where(valid & (cosang < -0.5))[0]
    if len(backtracks):
        print(f"\n{len(backtracks)} backtracking point(s) (sharp reversals with big steps):")
        for idx in backtracks[:20]:
            print(f"  at traj[{idx + 1}] (submap {which_submap(idx + 1)}) cos={cosang[idx]:.2f}")

    # ------------------------------------------------------------- html viz
    try:
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=traj[:, 0], y=traj[:, 2], z=-traj[:, 1],
            mode="lines+markers", name="trajectory",
            line=dict(color="cyan", width=3),
            marker=dict(size=4, color=[which_submap(i) for i in range(len(traj))],
                        colorscale="Turbo", showscale=True,
                        colorbar=dict(title="submap")),
            text=[f"traj[{i}] submap {which_submap(i)}" for i in range(len(traj))],
        ))
        for idx in jumps:
            seg = traj[idx:idx + 2]
            fig.add_trace(go.Scatter3d(
                x=seg[:, 0], y=seg[:, 2], z=-seg[:, 1], mode="lines",
                line=dict(color="red", width=8), name=f"jump {idx}", showlegend=False))
        fig.update_layout(scene=dict(aspectmode="data"), title=os.path.basename(run_dir))
        out = os.path.join(run_dir, "trajectory_debug.html")
        fig.write_html(out)
        print(f"\nWrote {out} (trajectory colored by submap, jumps in red)")
    except ImportError:
        print("\nplotly not available; skipped trajectory_debug.html")


if __name__ == "__main__":
    main()

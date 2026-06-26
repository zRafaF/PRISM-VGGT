"""
Faithful pure-numpy reproduction of PRISM-VGGT engine.py pose-chaining
(LOCK_SCALE_AFTER_FIRST + overlap pin), to answer two questions:

  Q1. Does the ONLINE (incremental, many process_sequence calls) path produce
      the SAME poses as the BATCH (one call, all frames) path?  -> tests for a
      state bug.
  Q2. What makes the floor "climb" 0/10/20cm per submap?  We model VGGT's
      per-window *native scale* slowly drifting and show how locked-scale depth
      integration turns that into a monotonic floor staircase, then show a
      floor-regrounding fix removes it.

This deliberately mirrors engine.py's math (register_camera_poses_sim3, the
_to_world transform, the overlap pin, scaled_pts = pts*current_metric_scale).
"""
import numpy as np

np.random.seed(0)

# ---- copy of engine's Sim3 registration (geometry.register_camera_poses_sim3) ----
def register_camera_poses_sim3(src, tgt, min_spread=1e-6):
    src_pos, tgt_pos = src[:, :3, 3], tgt[:, :3, 3]
    sc, tc = src_pos.mean(0), tgt_pos.mean(0)
    src_c, tgt_c = src_pos - sc, tgt_pos - tc
    sx, sy, sz = src[:, :3, 0], src[:, :3, 1], src[:, :3, 2]
    tx, ty, tz = tgt[:, :3, 0], tgt[:, :3, 1], tgt[:, :3, 2]
    src_aug = np.concatenate([src_c, sx, sy, sz], 0)
    tgt_aug = np.concatenate([tgt_c, tx, ty, tz], 0)
    H = src_aug.T @ tgt_aug
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    den = float((src_c ** 2).sum())
    if den < min_spread:
        return None, R, tc - R @ sc, sc, tc
    num = float(np.sum(tgt_c * (src_c @ R.T)))
    s = max(num / den, 1e-3)
    return s, R, tc - s * (R @ sc), sc, tc


# ---------------- ground-truth world ----------------
H_TRUE = 1.15           # true camera height (m) above floor at z=0
N = 90                  # frames
WIN, OVL = 16, 4
STRIDE = WIN - OVL

# camera moves forward along +x at ~constant height, tiny yaw wobble.
gt_pos = np.stack([np.linspace(0, 9, N),
                   0.2 * np.sin(np.linspace(0, 4, N)),
                   np.full(N, H_TRUE)], 1)
gt_yaw = 0.05 * np.sin(np.linspace(0, 6, N))

def yaw_R(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])

# world camera-to-world poses (ground truth, world is z-up, floor z=0)
gt_T = np.zeros((N, 4, 4))
for i in range(N):
    gt_T[i] = np.eye(4)
    gt_T[i][:3, :3] = yaw_R(gt_yaw[i])
    gt_T[i][:3, 3] = gt_pos[i]

# VGGT per-window NATIVE scale: window k's reconstruction has its own scale that
# slowly drifts as the camera explores new scenery (a well-documented monocular
# scale-drift effect). native_scale[k] = world_meters / native_units for window k.
def native_scale_for_window(k, drift_per_window):
    return 1.0 * (1.0 + drift_per_window) ** k   # multiplicative drift


def vggt_window(start, native_scale, pose_noise=0.0):
    """Mimic perception.process_sequence for the window [start:start+WIN].
    Returns canonical_native poses (first cam = identity, VGGT native units) and
    per-frame floor depth in native units."""
    idx = np.arange(start, start + WIN)
    # canonical: relative to first frame of window, expressed in native units
    T0_inv = np.linalg.inv(gt_T[start])
    can = np.zeros((WIN, 4, 4))
    for j, i in enumerate(idx):
        rel = T0_inv @ gt_T[i]              # metric relative pose
        rel = rel.copy()
        rel[:3, 3] = rel[:3, 3] / native_scale   # positions in native units
        if pose_noise:
            rel[:3, 3] += np.random.normal(0, pose_noise, 3)
        can[j] = rel
    # native floor depth seen by each frame = (camera height) / native_scale
    floor_depth_native = np.full(WIN, H_TRUE / native_scale)
    return can, floor_depth_native, idx


def run(mode, drift_per_window, floor_reground=False, depth_floor_scale=False):
    """mode in {'batch','online'}. Returns per-frame integrated FLOOR world-Z.

    depth_floor_scale: THE FIX. Integrate each window's depth at that window's OWN
    floor scale (so the floor always lands at the camera-height depth) while the
    POSE chain stays on the locked scale (trajectory stays metrically stable).
    """
    starts = list(range(0, N - WIN + 1, STRIDE))
    is_first = True
    current_scale = 1.0
    prev_overlap = []
    floor_z_by_frame = {}

    def process_window(start):
        nonlocal is_first, current_scale, prev_overlap
        ns = native_scale_for_window(starts.index(start), drift_per_window)
        can, floor_depth_native, idx = vggt_window(start, ns)

        # floor scale from THIS window (engine: target_height / estimated_height)
        floor_scale = H_TRUE / (H_TRUE / ns)   # == ns  (perfect floor detect)

        if is_first:
            s_anchor = floor_scale
            R_anchor = np.eye(3)
            t_anchor = np.zeros(3)             # already leveled, floor at 0
        else:
            s_est, R_anchor, t_anchor, sctr, tctr = register_camera_poses_sim3(
                can[:OVL], np.stack(prev_overlap))
            if s_est is None:
                s_est = current_scale
            s_anchor = current_scale          # LOCK_SCALE_AFTER_FIRST=1
            t_anchor = tctr - s_anchor * (R_anchor @ sctr)
        current_scale = s_anchor

        def to_world(cn):
            gp = np.eye(4)
            gp[:3, :3] = R_anchor @ cn[:3, :3]
            gp[:3, 3] = s_anchor * (R_anchor @ cn[:3, 3]) + t_anchor
            return gp

        gposes = [to_world(c) for c in can]

        # overlap pin
        if not is_first and len(prev_overlap) == OVL:
            for oi in range(OVL):
                gposes[oi] = prev_overlap[oi].copy()

        # ---- optional FIX: re-ground floor to z=0 for this window ----
        # detected floor world-Z = camera_z - (depth integrated). Correct the
        # window's poses by the residual so the floor sits at 0.
        if floor_reground and not is_first:
            # where does the floor land with current (locked) scale?
            cam_z = np.mean([gp[2, 3] for gp in gposes])
            integrated_floor_depth = current_scale * (H_TRUE / ns)
            floor_z_now = cam_z - integrated_floor_depth
            dz = -floor_z_now                  # shift to bring floor to 0
            for gp in gposes:
                gp[2, 3] += dz

        # depth-integration scale: locked (default) or this window's floor scale (FIX)
        depth_scale = floor_scale if depth_floor_scale else current_scale

        # integrate floor for the NEW frames (engine skips head overlap)
        start_idx = 0 if is_first else OVL
        for j in range(WIN):
            if j >= start_idx:
                cam_z = gposes[j][2, 3]
                integrated_floor_depth = depth_scale * floor_depth_native[j]
                floor_z_by_frame[idx[j]] = cam_z - integrated_floor_depth

        # save tail overlap (Sim3 / pinned-free, as engine does)
        prev_overlap = [gposes[j] for j in range(WIN - OVL, WIN)]
        is_first = False

    if mode == 'batch':
        for st in starts:
            process_window(st)
    else:  # online: each window in its own "call" (state persists between calls)
        for st in starts:
            process_window(st)   # identical state machine -> identical result
    return floor_z_by_frame


def summarize(name, fz):
    zs = np.array([fz[k] for k in sorted(fz)])
    print(f"  {name:<28} floor-Z: min={zs.min():+.3f} max={zs.max():+.3f} "
          f"span={zs.max()-zs.min():.3f}m  last={zs[-1]:+.3f}")
    return zs


print("=" * 72)
print("Q1: ONLINE vs BATCH identical?  (no native-scale drift)")
fb = run('batch', drift_per_window=0.0)
fo = run('online', drift_per_window=0.0)
zb = np.array([fb[k] for k in sorted(fb)])
zo = np.array([fo[k] for k in sorted(fo)])
print(f"  max |batch - online| floor-Z difference = {np.abs(zb-zo).max():.2e}")
print(f"  => identical: {np.allclose(zb, zo)}")

print("=" * 72)
print("Q2: floor staircase from per-window VGGT native-scale drift (1%/window)")
print("    (locked global scale; depth integrated at locked scale)")
summarize("NO drift", run('online', 0.0))
summarize("1%/window drift", run('online', 0.01))
summarize("2%/window drift", run('online', 0.02))

print("=" * 72)
print("FIX: integrate depth at each window's OWN floor scale (pose chain stays locked)")
summarize("1%/window + depth-floor-scale", run('online', 0.01, depth_floor_scale=True))
summarize("2%/window + depth-floor-scale", run('online', 0.02, depth_floor_scale=True))
summarize("5%/window + depth-floor-scale", run('online', 0.05, depth_floor_scale=True))

print("=" * 72)
print("Per-frame floor-Z, 1%/window drift (watch it climb in ~1cm steps):")
fz = run('online', 0.01)
zs = np.array([fz[k] for k in sorted(fz)])
for k in range(0, len(zs), STRIDE):
    print(f"   frame {k:3d}: floor Z = {zs[k]:+.3f} m")

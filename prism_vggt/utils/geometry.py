import torch
import numpy as np

def unproject_equirectangular_to_points(depth_map: np.ndarray) -> np.ndarray:
    """Converts an equirectangular radial depth map into 3D Cartesian coordinates."""
    H, W = depth_map.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    
    theta = (u / W - 0.5) * 2 * np.pi
    phi = (v / H - 0.5) * np.pi
    
    X = depth_map * np.cos(phi) * np.sin(theta)
    Y = depth_map * np.sin(phi)
    Z = depth_map * np.cos(phi) * np.cos(theta)
    
    return np.stack([X, Y, Z], axis=-1)

def rotation_aligning_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Returns a 3x3 rotation matrix R such that R @ a is parallel to b.

    Uses the Rodrigues formula for the rotation about the axis (a x b). Handles the
    degenerate parallel / anti-parallel cases. Inputs need not be normalized.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)

    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = float(np.dot(a, b))

    if s < 1e-8:
        # Already aligned, or exactly opposite.
        if c > 0:
            return np.eye(3)
        # 180 degrees: rotate about any axis orthogonal to a.
        ortho = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            ortho = np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, ortho)
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        return np.eye(3) + 2.0 * (K @ K)

    K = np.array([[0, -v[2], v[1]],
                  [v[2], 0, -v[0]],
                  [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1.0 - c) / (s ** 2))


def homogenize_points(points):
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)

def homogenize_points_np(points):
    return np.concatenate([points, np.ones_like(points[..., :1])], axis=-1)

def register_camera_poses_kabsch(src_cam_poses: np.ndarray, tgt_cam_poses: np.ndarray, scale=1.0):
    """
    Aligns two sets of camera poses using Kabsch algorithm.
    Fixed centroid bug: Rotation unit vectors are now decoupled from the translational centroid.
    """
    assert src_cam_poses.shape == tgt_cam_poses.shape
    
    src_pos = src_cam_poses[:, :3, 3] * scale
    tgt_pos = tgt_cam_poses[:, :3, 3]

    # Calculate centroids ONLY from the actual camera positions
    src_centroid = np.mean(src_pos, axis=0)
    tgt_centroid = np.mean(tgt_pos, axis=0)
    
    src_centered = src_pos - src_centroid
    tgt_centered = tgt_pos - tgt_centroid

    # Extract rotation axes to lock orientation
    src_x = src_cam_poses[:, :3, :3] @ np.array([1., 0., 0.])
    src_y = src_cam_poses[:, :3, :3] @ np.array([0., 1., 0.])
    src_z = src_cam_poses[:, :3, :3] @ np.array([0., 0., 1.])

    tgt_x = tgt_cam_poses[:, :3, :3] @ np.array([1., 0., 0.])
    tgt_y = tgt_cam_poses[:, :3, :3] @ np.array([0., 1., 0.])
    tgt_z = tgt_cam_poses[:, :3, :3] @ np.array([0., 0., 1.])

    # Append the rotation vectors directly to the centered positions
    src_pts = np.concatenate([src_centered, src_x, src_y, src_z], axis=0)
    tgt_pts = np.concatenate([tgt_centered, tgt_x, tgt_y, tgt_z], axis=0)

    # Standard Kabsch SVD
    H = src_pts.T @ tgt_pts
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Fix reflection
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Translation offset
    t = tgt_centroid - R @ src_centroid

    return R, t


def register_camera_poses_sim3(src_cam_poses: np.ndarray, tgt_cam_poses: np.ndarray, min_spread: float = 1e-6):
    """Estimate a similarity transform (Sim3) aligning src camera poses to tgt.

    Maps src -> tgt for the camera centers as ``p_tgt ~= s * R @ p_src + t`` (the
    camera orientation is additionally rotated by R). Unlike a decoupled SE(3) anchor
    plus a separate global scale heuristic, this jointly estimates rotation,
    translation AND the relative scale directly from the overlapping cameras.

    - ``s`` is the Umeyama least-squares optimal scale given R (positions only).
    - ``R`` is estimated from centered positions AUGMENTED with the camera
      orientation axes, so it stays well-conditioned even for short, near-collinear
      overlap snippets (same trick as the Kabsch variant).

    Returns ``(s, R, t, src_centroid, tgt_centroid)``. ``s`` is ``None`` when the
    camera baseline is too small to observe scale (caller should fall back).
    """
    assert src_cam_poses.shape == tgt_cam_poses.shape

    src_pos = src_cam_poses[:, :3, 3]
    tgt_pos = tgt_cam_poses[:, :3, 3]

    src_centroid = src_pos.mean(axis=0)
    tgt_centroid = tgt_pos.mean(axis=0)
    src_c = src_pos - src_centroid
    tgt_c = tgt_pos - tgt_centroid

    # Orientation axes (unit -> scale invariant) stabilise the rotation estimate.
    src_x, src_y, src_z = src_cam_poses[:, :3, 0], src_cam_poses[:, :3, 1], src_cam_poses[:, :3, 2]
    tgt_x, tgt_y, tgt_z = tgt_cam_poses[:, :3, 0], tgt_cam_poses[:, :3, 1], tgt_cam_poses[:, :3, 2]

    src_aug = np.concatenate([src_c, src_x, src_y, src_z], axis=0)
    tgt_aug = np.concatenate([tgt_c, tgt_x, tgt_y, tgt_z], axis=0)

    H = src_aug.T @ tgt_aug
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Umeyama optimal scale from positions only: sum<tgt_c, R src_c> / sum||src_c||^2.
    den = float((src_c ** 2).sum())
    if den < min_spread:
        return None, R, tgt_centroid - R @ src_centroid, src_centroid, tgt_centroid

    num = float(np.sum(tgt_c * (src_c @ R.T)))
    s = max(num / den, 1e-3)
    t = tgt_centroid - s * (R @ src_centroid)

    return s, R, t, src_centroid, tgt_centroid


# ======================================================================
# SL(4) / PGL(4) projective alignment  — the 15-DoF ABLATION arm
# ----------------------------------------------------------------------
# PRISM aligns submaps with a 7-DoF Sim(3) (rotation + translation + one
# global scale). VGGT-SLAM instead aligns them with a 15-DoF SL(4)/PGL(4)
# projective transform, which on top of rotation/translation/scale also
# permits ANISOTROPIC scaling, SHEAR and PERSPECTIVE warp. These helpers
# fit and apply such a transform so the alignment group can be ablated in
# isolation (everything else in the engine held fixed). The extra 8 DoF fit
# the overlap better but distort metric geometry — which is exactly what we
# measure. See sl4_nonsimilarity_report for the distortion diagnostic.
# ======================================================================

def _hartley_normalize_3d(pts: np.ndarray):
    """Isotropic (Hartley) normalization: centre at origin, mean distance sqrt(3).
    Returns the 4x4 normalizing transform T and the normalized points."""
    c = pts.mean(axis=0)
    d = pts - c
    mean_dist = float(np.sqrt((d ** 2).sum(axis=1)).mean())
    s = (np.sqrt(3.0) / mean_dist) if mean_dist > 1e-12 else 1.0
    T = np.eye(4)
    T[:3, :3] *= s
    T[:3, 3] = -s * c
    return T, d * s


def register_points_sl4(src: np.ndarray, tgt: np.ndarray, max_pts: int = 3000,
                        seed: int = 0):
    """Fit a 3D projective homography H (4x4, |det|=1) mapping ``src`` -> ``tgt``
    point-wise via the Direct Linear Transform — the 15-DoF SL(4)/PGL(4) analog of
    VGGT-SLAM's submap alignment.

    ``src``/``tgt`` are (N,3) *corresponded* points (same physical surface, one from
    each overlapping submap). Hartley-normalized for conditioning; returns H (native
    -> world), or ``None`` if the correspondence set is degenerate.
    """
    src = np.asarray(src, dtype=np.float64)
    tgt = np.asarray(tgt, dtype=np.float64)
    if src.shape != tgt.shape or src.ndim != 2 or src.shape[1] != 3 or len(src) < 5:
        return None
    fin = np.isfinite(src).all(1) & np.isfinite(tgt).all(1)
    src, tgt = src[fin], tgt[fin]
    if len(src) < 5:
        return None
    if len(src) > max_pts:
        idx = np.random.default_rng(seed).choice(len(src), max_pts, replace=False)
        src, tgt = src[idx], tgt[idx]

    Ts, sn = _hartley_normalize_3d(src)
    Tt, tn = _hartley_normalize_3d(tgt)
    N = len(sn)
    Xh = np.concatenate([sn, np.ones((N, 1))], axis=1)          # (N,4)

    # DLT: for each point, 3 rows enforcing (row_a . X) - x'_a (row_3 . X) = 0.
    A = np.zeros((3 * N, 16))
    r = 0
    for p in range(N):
        X = Xh[p]
        xp = tn[p]                       # (3,), homogeneous w' = 1
        for a in range(3):
            row = np.zeros(16)
            row[a * 4:(a + 1) * 4] = X            # (row_a . X) * w'
            row[12:16] -= xp[a] * X               # - x'_a (row_3 . X)
            A[r] = row
            r += 1

    try:
        _, _, Vt = np.linalg.svd(A, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    Hn = Vt[-1].reshape(4, 4)
    H = np.linalg.inv(Tt) @ Hn @ Ts                # denormalize
    d = np.linalg.det(H)
    if not np.isfinite(d) or abs(d) < 1e-30:
        return None
    return H / (abs(d) ** 0.25)                    # |det| -> 1


def apply_sl4(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 4x4 projective transform to 3D point(s) with the perspective divide."""
    pts = np.asarray(pts, dtype=np.float64)
    single = pts.ndim == 1
    P = pts.reshape(-1, 3)
    Ph = np.concatenate([P, np.ones((len(P), 1))], axis=1)
    Y = Ph @ H.T
    w = Y[:, 3:4]
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    out = Y[:, :3] / w
    return out[0] if single else out


def sl4_local_sim3(H: np.ndarray, p: np.ndarray):
    """Best-fit local similarity (scale s, rotation R) of the projective map ``H`` at
    point ``p``, via the polar decomposition of its Jacobian. Used to integrate a
    submap into the (rigid) TSDF: nvblox cannot represent shear/perspective, so we
    integrate at the local rigid+scale approximation and report the discarded
    non-rigid part separately (sl4_nonsimilarity_report)."""
    p = np.asarray(p, dtype=np.float64)
    xh = np.append(p, 1.0)
    num = H[:3, :] @ xh                              # (3,)
    w = float(H[3, :] @ xh)
    if abs(w) < 1e-12:
        w = 1e-12
    J = (H[:3, :3] * w - np.outer(num, H[3, :3])) / (w * w)
    U, S, Vt = np.linalg.svd(J)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U = U.copy(); U[:, -1] *= -1.0
        R = U @ Vt
    s = float(np.cbrt(max(S[0] * S[1] * S[2], 1e-30)))
    return s, R


def sl4_nonsimilarity_report(H: np.ndarray, at: np.ndarray = None) -> dict:
    """Quantify how far ``H`` departs from a pure similarity at point ``at`` — the
    exact freedom Sim(3) forbids and VGGT-SLAM's SL(4) permits.

    Returns anisotropy (max/min local stretch - 1; 0 for a similarity), a shear
    proxy (dispersion of the Jacobian's singular values), the perspective row
    magnitude, and ``nonsimilarity_pct`` = 100 * anisotropy for reporting.
    """
    at = np.zeros(3) if at is None else np.asarray(at, dtype=np.float64)
    xh = np.append(at, 1.0)
    num = H[:3, :] @ xh
    w = float(H[3, :] @ xh)
    if abs(w) < 1e-12:
        w = 1e-12
    J = (H[:3, :3] * w - np.outer(num, H[3, :3])) / (w * w)
    S = np.linalg.svd(J, compute_uv=False)
    s = float(np.cbrt(max(S[0] * S[1] * S[2], 1e-30)))
    anisotropy = float(S[0] / max(S[2], 1e-12) - 1.0)
    shear = float(np.std(S) / (np.mean(S) + 1e-12))
    perspective = float(np.linalg.norm(H[3, :3]) / (abs(H[3, 3]) + 1e-12))
    return {"local_scale": s, "anisotropy": anisotropy, "shear": shear,
            "perspective": perspective, "nonsimilarity_pct": 100.0 * anisotropy}
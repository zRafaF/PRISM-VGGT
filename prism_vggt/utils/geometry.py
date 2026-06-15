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
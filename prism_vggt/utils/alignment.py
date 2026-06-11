import torch
import numpy as np
from typing import Tuple

def align_cam_pts_irls(
    src_pts: torch.Tensor,  
    tgt_pts: torch.Tensor,  
    mask: torch.Tensor = None,  
    iters: int = 10,
    eps: float = 1e-8,
    stop_tol: float = 0.05,
    clamp_min: float = 1e-6
) -> float:
    """
    Calculates the global scale difference between two overlapping 3D point clouds 
    using Iteratively Reweighted Least Squares (IRLS).
    """
    if mask is not None:
        if src_pts.dim() == 4:
            mask_expanded = mask.unsqueeze(0).unsqueeze(-1).expand_as(src_pts)
            src_valid = src_pts[mask_expanded.bool()].view(-1, 3)
            tgt_valid = tgt_pts[mask_expanded.bool()].view(-1, 3)
        else:
            src_valid = src_pts[mask.bool()]
            tgt_valid = tgt_pts[mask.bool()]
    else:
        src_valid = src_pts.reshape(-1, 3)
        tgt_valid = tgt_pts.reshape(-1, 3)

    src_np = src_valid.cpu().numpy()
    tgt_np = tgt_valid.cpu().numpy()
    
    num = np.nanmean(np.linalg.norm(tgt_np, axis=-1))
    den = np.nanmean(np.linalg.norm(src_np, axis=-1))
    s_d = np.maximum(num / den, clamp_min)

    for _ in range(iters):
        d_res = s_d * src_np - tgt_np
        res = np.linalg.norm(d_res, axis=-1) + eps
        w = 1.0 / res
        
        num = (w * np.linalg.norm(src_np, axis=-1) * np.linalg.norm(tgt_np, axis=-1)).sum()
        den = (w * np.linalg.norm(src_np, axis=-1) ** 2).sum()
        s_d_new = np.maximum(num / den, clamp_min)
        
        if abs(s_d_new - s_d) < stop_tol:
            break
        s_d = s_d_new
        
    return float(s_d)

def register_camera_poses_kabsch(src_cam_poses: np.ndarray, tgt_cam_poses: np.ndarray, scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aligns two sets of camera poses using Kabsch algorithm.
    Enhanced with full 3D coordinate frame locking to prevent collinear degeneracy.
    """
    assert src_cam_poses.shape == tgt_cam_poses.shape
    
    src_pos = src_cam_poses[:, :3, 3] * scale
    tgt_pos = tgt_cam_poses[:, :3, 3]

    # Extract full coordinate frame (X, Y, Z axes) for orientation locking
    src_x = src_cam_poses[:, :3, :3] @ np.array([1., 0., 0.])
    src_y = src_cam_poses[:, :3, :3] @ np.array([0., 1., 0.])
    src_z = src_cam_poses[:, :3, :3] @ np.array([0., 0., 1.])

    tgt_x = tgt_cam_poses[:, :3, :3] @ np.array([1., 0., 0.])
    tgt_y = tgt_cam_poses[:, :3, :3] @ np.array([0., 1., 0.])
    tgt_z = tgt_cam_poses[:, :3, :3] @ np.array([0., 0., 1.])

    src_pts = np.concatenate([src_pos, src_pos + src_x, src_pos + src_y, src_pos + src_z], axis=0)
    tgt_pts = np.concatenate([tgt_pos, tgt_pos + tgt_x, tgt_pos + tgt_y, tgt_pos + tgt_z], axis=0)

    src_centroid = np.mean(src_pts, axis=0)
    tgt_centroid = np.mean(tgt_pts, axis=0)

    src_pts_centered = src_pts - src_centroid
    tgt_pts_centered = tgt_pts - tgt_centroid

    H = src_pts_centered.T @ tgt_pts_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Fix improper rotation (reflection)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = tgt_centroid - R @ src_centroid
    
    return R, t
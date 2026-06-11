import numpy as np
import torch
import cv2

def get_radial_shells(depth: np.ndarray, mask: np.ndarray, num_shells: int = 6) -> np.ndarray:
    """Segments the depth map into K concentric shells using adaptive quantiles."""
    valid_depths = depth[mask]
    if len(valid_depths) == 0:
        return np.zeros_like(depth, dtype=np.int32)
    
    # Calculate adaptive shell boundaries based on actual scene geometry distribution
    quantiles = np.linspace(0, 1, num_shells + 1)
    bins = np.quantile(valid_depths, quantiles)
    bins[-1] += 1e-5  # Ensure the absolute max value is included safely
    
    # Assign pixels to their respective depth shells
    labels = np.digitize(depth, bins) - 1
    labels = np.clip(labels, 0, num_shells - 1)
    
    # Invalidate pixels outside our polar exclusion mask
    labels[~mask] = -1
    return labels

def align_radial_shells(src_depth: np.ndarray, tgt_depth: np.ndarray, src_labels: np.ndarray, tgt_labels: np.ndarray, num_shells: int) -> np.ndarray:
    """Calculates the local scale factor for each discrete concentric shell."""
    scale_map = np.ones_like(tgt_depth)
    
    for k in range(num_shells):
        src_k_mask = (src_labels == k)
        tgt_k_mask = (tgt_labels == k)
        
        if not np.any(src_k_mask) or not np.any(tgt_k_mask):
            continue
            
        # Median is critical: robust against dynamic objects entering the shell
        src_median = np.median(src_depth[src_k_mask])
        tgt_median = np.median(tgt_depth[tgt_k_mask])
        
        if tgt_median > 1e-3:
            scale_map[tgt_k_mask] = src_median / tgt_median
            
    return scale_map

def refine_depth_segments(src_pcd: torch.Tensor, tgt_pcd: torch.Tensor, mask: torch.Tensor, num_shells: int = 6, smooth_kernel: int = 31) -> torch.Tensor:
    """
    Aligns concentric radial shells to fix Monocular Radial Ballooning.
    """
    src_pcd_np = src_pcd.cpu().numpy()
    tgt_pcd_np = tgt_pcd.cpu().numpy()
    mask_np = mask.cpu().numpy().astype(bool)
    
    # 1. Extract raw radial depth (distance from camera)
    src_depth = np.linalg.norm(src_pcd_np, axis=-1)
    tgt_depth = np.linalg.norm(tgt_pcd_np, axis=-1)
    
    # 2. Segment geometry into Concentric Shells
    src_labels = get_radial_shells(src_depth, mask_np, num_shells)
    tgt_labels = get_radial_shells(tgt_depth, mask_np, num_shells)
    
    # 3. Calculate Scale Correction Map per Shell
    raw_scale_map = align_radial_shells(src_depth, tgt_depth, src_labels, tgt_labels, num_shells)
    
    # 4. Smooth the Scale Map to create a continuous deformation field
    smoothed_scale_map = cv2.GaussianBlur(raw_scale_map, (smooth_kernel, smooth_kernel), 0)
    
    # Revert invalid areas to a safe 1.0 scale
    smoothed_scale_map[~mask_np] = 1.0
    
    return torch.from_numpy(smoothed_scale_map[..., None]).to(src_pcd.device)
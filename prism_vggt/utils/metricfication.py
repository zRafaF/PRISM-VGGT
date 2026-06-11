import numpy as np
import open3d as o3d
from typing import Tuple, Optional

def estimate_metric_scale_from_floor(
    local_pts: np.ndarray, 
    target_camera_height: float = 1.7, 
    normal_tolerance_deg: float = 15.0, 
    inlier_threshold: float = 0.3
) -> Tuple[Optional[float], float]:
    """
    Finds the floor plane in the local camera point cloud to calculate absolute metric scale.
    Assumes OpenCV camera coordinate system (Y is DOWN).
    
    Returns:
        (scale_factor, confidence_score)
    """
    local_pts = local_pts.reshape(-1, 3)
    
    # Drop points originating from masked origin (0,0,0)
    valid_mask = np.linalg.norm(local_pts, axis=-1) > 0.1
    local_pts = local_pts[valid_mask]
    
    # Isolate lower hemisphere (Y > 0.1 in OpenCV space)
    lower_pts = local_pts[local_pts[:, 1] > 0.1]
    
    if len(lower_pts) < 100:
        return None, 0.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(lower_pts)
    
    try:
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=0.05,
            ransac_n=3,
            num_iterations=200
        )
    except Exception:
        return None, 0.0
        
    a, b, c, d = plane_model
    normal = np.array([a, b, c])
    normal = normal / np.linalg.norm(normal)
    
    # Check if plane is horizontal (aligned with Y-axis)
    y_axis = np.array([0, 1, 0])
    dot_product = np.clip(np.abs(np.dot(normal, y_axis)), 0.0, 1.0)
    angle = np.degrees(np.arccos(dot_product))
    
    confidence = len(inliers) / float(len(lower_pts))
    
    if angle <= normal_tolerance_deg and confidence >= inlier_threshold:
        estimated_height = abs(d)
        if estimated_height < 0.1:
            return None, confidence
            
        scale_factor = target_camera_height / estimated_height
        return scale_factor, confidence
        
    return None, confidence
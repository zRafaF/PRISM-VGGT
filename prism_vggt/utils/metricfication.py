import numpy as np
import open3d as o3d
from typing import Tuple, Optional, Dict


def estimate_metric_scale_from_floor(
    local_pts: np.ndarray,
    target_camera_height: float = 1.7,
    normal_tolerance_deg: float = 15.0,
    inlier_threshold: float = 0.3
) -> Tuple[Optional[float], float, Optional[Dict]]:
    """
    Finds the floor plane in the local camera point cloud to calculate absolute metric scale.
    Assumes OpenCV camera coordinate system (Y is DOWN).

    Returns:
        (scale_factor, confidence_score, plane_info)

    ``plane_info`` is ``None`` when no horizontal floor is found. Otherwise it is a
    dict describing the detected plane *in the same (metric, local-camera) frame as
    ``local_pts``*::

        {
            "normal":   (3,) unit normal, oriented to point "up" (towards the camera),
            "centroid": (3,) centroid of the inlier points,
            "extent":   float, half-size of the inlier footprint (for drawing a patch),
            "d":        signed plane offset (a*x + b*y + c*z + d = 0),
        }
    """
    local_pts = local_pts.reshape(-1, 3)

    # Drop points originating from masked origin (0,0,0)
    valid_mask = np.linalg.norm(local_pts, axis=-1) > 0.1
    local_pts = local_pts[valid_mask]

    # Isolate lower hemisphere (Y > 0.1 in OpenCV space)
    lower_pts = local_pts[local_pts[:, 1] > 0.1]

    if len(lower_pts) < 100:
        return None, 0.0, None

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(lower_pts)

    try:
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=0.05,
            ransac_n=3,
            num_iterations=200
        )
    except Exception:
        return None, 0.0, None

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
            return None, confidence, None

        # Build a description of the detected plane for leveling + visualization.
        # Orient the normal to point "up" (opposite to the floor, i.e. towards the
        # camera which sits above it -> negative Y in OpenCV convention).
        oriented_normal = normal if normal[1] < 0 else -normal
        inlier_pts = lower_pts[inliers]
        centroid = inlier_pts.mean(axis=0)
        # In-plane footprint radius (used only to size the rendered patch).
        extent = float(np.percentile(np.linalg.norm(inlier_pts - centroid, axis=1), 95))

        plane_info = {
            "normal": oriented_normal,
            "centroid": centroid,
            "extent": extent,
            "d": float(d),
        }

        scale_factor = target_camera_height / estimated_height
        return scale_factor, confidence, plane_info

    return None, confidence, None
import numpy as np

def unproject_equirectangular_to_points(depth_map: np.ndarray) -> np.ndarray:
    """
    Converts an equirectangular radial depth map into 3D Cartesian coordinates.
    Assumes an OpenCV coordinate system (X: right, Y: down, Z: forward).
    """
    H, W = depth_map.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    
    # Convert pixels to spherical angles
    theta = (u / W - 0.5) * 2 * np.pi
    phi = (v / H - 0.5) * np.pi
    
    # Spherical to Cartesian
    X = depth_map * np.cos(phi) * np.sin(theta)
    Y = depth_map * np.sin(phi)
    Z = depth_map * np.cos(phi) * np.cos(theta)
    
    return np.stack([X, Y, Z], axis=-1)

def homogenize_points_np(points: np.ndarray) -> np.ndarray:
    """Convert batched points (xyz) to homogenous coordinates (xyz1)."""
    return np.concatenate([points, np.ones_like(points[..., :1])], axis=-1)
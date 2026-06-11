import cv2
import numpy as np

def visualize_polar_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Dims and tints the excluded regions of the image to visualize the mask."""
    vis_image = image.copy()
    
    darkened = (vis_image * 0.4).astype(np.uint8)
    red_tint = np.zeros_like(vis_image)
    red_tint[:, :, 0] = 80  # Add to Red channel
    
    darkened = cv2.add(darkened, red_tint)
    
    invalid_mask = ~mask
    vis_image[invalid_mask] = darkened[invalid_mask]
    
    return vis_image

def visualize_depth(depth_map: np.ndarray) -> np.ndarray:
    """Converts a 1-channel depth map into a colorized PLASMA heatmap."""
    depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_uint8 = depth_norm.astype(np.uint8)
    
    colorized = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_PLASMA)
    return cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB)
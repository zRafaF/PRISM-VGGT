from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, List

class BasePerceptionExtractor(ABC):
    """
    Abstract interface for 360° perception models.
    The PRISM-VGGT engine relies ONLY on this interface, allowing any
    panoramic depth/pose network to be hot-swapped into the backend.
    """
    
    @abstractmethod
    def process_frame(self, rgb_image: np.ndarray) -> Dict[str, Any]:
        """
        Processes a single RGB image.
        
        Args:
            rgb_image (np.ndarray): HxWxC RGB image array [0, 255].
            
        Returns:
            Dict containing at minimum:
                - 'depth': np.ndarray of shape (H, W) containing depth in meters.
        """
        pass

    @abstractmethod
    def process_sequence(self, rgb_images: List[np.ndarray]) -> Dict[str, Any]:
        """
        Processes a sliding window sequence of frames.
        
        Args:
            rgb_images: List of HxWxC RGB image arrays [0, 255].
            
        Returns:
            Dict containing at minimum:
                - 'depths': list of (H, W) depth arrays or torch Tensors.
                - 'poses': list of (4, 4) local pose matrices.
                - 'points': list of (N, 3) local point clouds.
        """
        pass
import numpy as np
import torch

def get_spherical_valid_mask(H: int, W: int, zenith_deg: float = 75.0, nadir_deg: float = -65.0) -> np.ndarray:
    """
    Creates a boolean mask for an equirectangular image, invalidating poles.
    Latitude ranges from +90 (top/zenith) to -90 (bottom/nadir).
    
    Returns:
        np.ndarray: A boolean mask of shape (H, W) where True means VALID.
    """
    latitudes = np.linspace(90.0, -90.0, H)
    valid_rows = (latitudes <= zenith_deg) & (latitudes >= nadir_deg)
    mask = np.broadcast_to(valid_rows[:, None], (H, W))
    return mask

def apply_mask_to_tensor(tensor: torch.Tensor, mask: np.ndarray, fill_value=0.0) -> torch.Tensor:
    """
    Applies a numpy boolean mask to a PyTorch tensor, maintaining device placement.
    """
    mask_tensor = torch.from_numpy(mask).to(tensor.device)
    
    while mask_tensor.dim() < tensor.dim():
        mask_tensor = mask_tensor.unsqueeze(0)
        
    return torch.where(
        mask_tensor, 
        tensor, 
        torch.tensor(fill_value, dtype=tensor.dtype, device=tensor.device)
    )
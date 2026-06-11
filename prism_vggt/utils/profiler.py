import torch

class VRAMProfiler:
    """Tracks PyTorch and overall system VRAM usage (useful for observing Nvblox C++ allocations)."""
    
    def __init__(self, device: int = 0):
        self.device = device
        
    def start(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
        
    def stop(self) -> tuple[float, float, float, float]:
        """Returns: (PyTorch_Allocated_GB, PyTorch_Reserved_GB, System_Used_GB, System_Total_GB)"""
        if not torch.cuda.is_available():
            return 0.0, 0.0, 0.0, 0.0
            
        pt_alloc = torch.cuda.max_memory_allocated(self.device) / (1024**3)
        pt_res = torch.cuda.max_memory_reserved(self.device) / (1024**3)
        
        free, total = torch.cuda.mem_get_info(self.device)
        sys_used = (total - free) / (1024**3)
        sys_total = total / (1024**3)
        
        return pt_alloc, pt_res, sys_used, sys_total
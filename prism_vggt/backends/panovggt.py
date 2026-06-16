import os
import sys
import torch
import numpy as np
from omegaconf import OmegaConf
from typing import Dict, Any, List

from prism_vggt.perception_base import BasePerceptionExtractor

# Dynamically add the third-party submodule to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
third_party_dir = os.path.abspath(os.path.join(current_dir, "../../third_party/PanoVGGT"))
if third_party_dir not in sys.path:
    sys.path.append(third_party_dir)

from panovggt.models.panovggt_model import PanoVGGTModel

class PanoVGGTBackend(BasePerceptionExtractor):
    def __init__(
        self, 
        config_path="third_party/PanoVGGT/training/config/default.yaml", 
        weights_path="checkpoints/model.pt", 
        device="cuda"
    ):
        self.device = device
        print(f"[PRISM] Loading PanoVGGT Backend onto {self.device}...")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Could not find config at {config_path}")

        if not os.path.exists(weights_path):
            from prism_vggt.weights import missing_weights_message
            raise FileNotFoundError(missing_weights_message(weights_path))

        cfg = OmegaConf.load(config_path)
        OmegaConf.resolve(cfg)
        mc = cfg.model
        
        self.model = PanoVGGTModel(
            img_size=cfg.img_size,
            patch_size=cfg.patch_size,
            embed_dim=cfg.embed_dim,
            enable_camera=mc.enable_camera,
            enable_depth=mc.enable_depth,
            enable_point=mc.enable_point,
            aggregator=OmegaConf.to_container(mc.aggregator, resolve=True),
        ).to(self.device)
        
        ckpt = torch.load(weights_path, map_location=self.device, weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt.get("model", ckpt.get("state_dict", ckpt)))
        sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
        
        self.model.load_state_dict(sd, strict=False)
        self.model.eval()
        print("[PRISM] ✅ PanoVGGT Backend ready.")

    @torch.no_grad()
    def process_frame(self, rgb_image: np.ndarray) -> Dict[str, Any]:
        """Processes a single frame for debugging or single-shot extraction."""
        img_tensor = torch.from_numpy(rgb_image).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).unsqueeze(0).to(self.device)
        
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.amp.autocast("cuda", dtype=dtype):
            preds = self.model(img_tensor)
            
        depth_out = preds.get("depth")
        if depth_out is None:
            depth_out = torch.norm(preds["local_points"], dim=-1)
            
        return {"depth": depth_out.squeeze().cpu().float().numpy()}

    @torch.no_grad()
    def process_sequence(self, rgb_images: List[np.ndarray]) -> Dict[str, Any]:
        """Processes a sliding window sequence, outputting batched spatial geometry."""
        # Preprocess: [B, H, W, C] -> [1, B, C, H, W]
        img_tensors = [torch.from_numpy(img).float() / 255.0 for img in rgb_images]
        batch_tensor = torch.stack(img_tensors).permute(0, 3, 1, 2)
        batch_tensor = batch_tensor.unsqueeze(0).to(self.device)  
        
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.amp.autocast("cuda", dtype=dtype):
            preds = self.model(batch_tensor)
            
        # 1. Robust Pose Extraction
        if "camera_poses" in preds:
            raw_poses = preds["camera_poses"]
        elif "poses" in preds:
            raw_poses = preds["poses"]
        elif "pose" in preds:
            raw_poses = preds["pose"]
        elif "camera" in preds:
            raw_poses = preds["camera"]
        else:
            print(f"⚠️ [WARNING] Could not find pose key. Available keys: {list(preds.keys())}")
            raw_poses = torch.eye(4, device=self.device).unsqueeze(0).unsqueeze(0).repeat(1, batch_tensor.shape[1], 1, 1)

        # 2. Extract Points
        raw_points = preds.get("local_points")
        if raw_points is None:
            raise KeyError(f"Missing 'local_points' in model output. Available keys: {list(preds.keys())}")
            
        # 3. Extract Depths
        raw_depths = preds.get("depth")
        if raw_depths is None:
            raw_depths = torch.norm(raw_points, dim=-1)
            
        return {
            "depths": [d.squeeze().cpu().float().numpy() for d in raw_depths.squeeze(0)],
            "poses": [p.cpu().numpy() for p in raw_poses.squeeze(0)],
            "points": [pts.cpu().numpy() for pts in raw_points.squeeze(0)]
        }
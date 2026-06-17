"""Public input/config types for PRISM-VGGT.

Kept dependency-light (numpy + stdlib typing) so it can be imported without pulling
in torch/nvblox.
"""
from dataclasses import dataclass
from typing import Literal, Union

import numpy as np

# --- Literal config aliases (let callers/IDE catch typos) ------------------------
WorldFrame = Literal["floor", "camera"]
"""World origin convention (always Z-up, right-handed): 'floor' puts Z=0 on the
detected floor; 'camera' puts the first camera at the origin."""

ProcessingMode = Literal["parallel", "sequential"]
"""'parallel' overlaps GPU inference with mapping (A/B double buffer); 'sequential'
runs them one after another (lower peak VRAM, slower)."""

Timestamp = Union[float, int]


@dataclass
class FrameInput:
    """One input frame for :meth:`StreamingWindowEngine.process_sequence`.

    Attributes:
        image: (H, W, 3) uint8 RGB equirectangular panorama.
        mask: (H, W) validity mask (nonzero = valid pixel).
        camera_height: instantaneous camera height above the floor (meters) measured
            at this frame's capture time; used for that window's metric scaling.
        timestamp: capture time or frame id, attached to the output camera pose so a
            downstream client/SLAM can time-reference each pose.
    """
    image: np.ndarray
    mask: np.ndarray
    camera_height: float
    timestamp: Timestamp

"""PRISM-VGGT: streaming panoramic SLAM + dense TSDF reconstruction.

Public API (two-object design):

    from prism_vggt import PanoVGGTBackend, StreamingWindowEngine, download_weights

    download_weights("checkpoints/model.pt")            # explicit, one-time
    perception = PanoVGGTBackend(weights_path="checkpoints/model.pt")
    engine = StreamingWindowEngine(perception, voxel_size=0.02, max_depth=4.5)
    for mesh, pcd, trajectory, floor in engine.process_sequence(frames, masks):
        ...

The heavy backends (torch / nvblox) are imported lazily, so ``import prism_vggt``
and ``download_weights`` stay cheap and dependency-light.
"""
from .weights import download_weights, PANOVGGT_WEIGHTS_URL, DEFAULT_WEIGHTS_PATH
from .perception_base import BasePerceptionExtractor

__all__ = [
    "download_weights",
    "PANOVGGT_WEIGHTS_URL",
    "DEFAULT_WEIGHTS_PATH",
    "BasePerceptionExtractor",
    "StreamingWindowEngine",
    "PanoVGGTBackend",
]


def __getattr__(name):
    # Lazy heavy imports (PEP 562) so torch/nvblox load only when actually used.
    if name == "StreamingWindowEngine":
        from .engine import StreamingWindowEngine
        return StreamingWindowEngine
    if name == "PanoVGGTBackend":
        from .backends.panovggt import PanoVGGTBackend
        return PanoVGGTBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

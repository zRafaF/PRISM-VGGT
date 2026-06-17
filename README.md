<div align="center">
<h1>PRISM-VGGT</h1>
<h3>Panoramic Reconstruction with Incremental SLAM and Dense Modeling from 360° Video</h3>
</div>

**PRISM-VGGT** is a streaming panoramic SLAM and dense reconstruction system for semi-realtime 360° video. 

Built for robotics, VR streaming, and spatial computing, PRISM-VGGT adapts sliding-window architectures to process equirectangular panoramic streams. By combining neural depth/pose estimation with highly optimized GPU-based Voxel Block Hashing (via `nvblox`), this pipeline addresses monocular scale drift, aligns concentric spherical shells, and outputs globally consistent, queryable 3D environments.

Originally starting as a spinoff of [LASER](https://github.com/neu-vi/LASER), we completely redesigned the architecture to integrate concepts from [VGGT-SLAM](https://github.com/MIT-SPARK/VGGT-SLAM). Using [PanoVGGT](https://github.com/YijingGuo-June/PanoVGGT) as our geometry engine, PRISM-VGGT processes frames in sliding-window batches to address monocular scale drift, utilizing NVIDIA's `nvblox` to fuse multi-view geometry into a globally consistent, dense TSDF mesh.

## Requirements
* **OS:** Linux (Ubuntu 24.04 / Debian)
* **GPU:** NVIDIA GPU with CUDA 12.8 support (compute capability >= 7.5). Tested on Blackwell (RTX PRO 6000, sm_120/sm_122).
* **CUDA:** 12.8 (the container image `nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04` is known-good)
* **Python:** 3.12

> **Why CUDA 12.8?** Blackwell GPUs (RTX 50-series, RTX PRO 6000) need `sm_120`/`sm_122` kernels, which first ship in PyTorch's `cu128` build (`torch==2.8`). Older `cu124` wheels fail with *"no kernel image is available for execution on the device"*.

## Quick Install (RunPod / Ubuntu)

Because this project relies on a submodule, ensure you clone recursively. Then, run our automated setup script:

```bash
git clone --recurse-submodules https://github.com/zRafaF/PRISM-VGGT
cd PRISM-VGGT
chmod +x setup.sh
./setup.sh
```

### Manual Installation Steps

If you prefer to install manually:

1. Ensure [`uv`](https://docs.astral.sh/uv/) is installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`).
2. Sync the environment: `uv sync`.
* If you want to install the dependencies for the Gradio UI, run `uv sync --extra apps`.
* To install benchmarking tools, run `uv sync --extra benchmarks`.


3. Download the weights. The library never downloads anything on its own; fetch the
   checkpoint explicitly with the bundled helper:

```bash
uv run python -c "from prism_vggt import download_weights; download_weights('checkpoints/model.pt')"
```

   (Or manually: `wget -qnc https://huggingface.co/YijingGuo/PanoVGGT/resolve/main/model.pt -O checkpoints/model.pt`.)
   If the checkpoint is missing when you construct `PanoVGGTBackend`, it raises a clear
   error telling you to run the line above.

### Handling Custom Environments (Different OS / CUDA versions)

By default, `uv sync` installs the official `nvblox-torch` wheel pre-compiled for **Ubuntu 24.04 and CUDA 12.8** (`nvblox_torch-0.0.10+cu12ubuntu24-...`).

> **`uv` note:** that wheel uses a non-manylinux filename, so `uv` will reject it unless you set `UV_SKIP_WHEEL_FILENAME_CHECK=1` in the environment (already exported by `setup.sh` and in the container's `docker-compose`). Without it, `uv sync` fails with a wheel-filename validation error.

If you are on a different operating system or require a different CUDA version, you do not need to manually edit configuration files. You can dynamically swap the wheel by running the `uv add` command with the URL of the wheel you need.

For example, to swap to a different wheel URL, run:

```bash
uv add "nvblox-torch @ <YOUR_CUSTOM_WHEEL_URL>"
```

This will automatically update the project configuration and install the correct wheel. *If you need to build `nvblox` from source, please see `docs/ADVANCED_NVBLOX.md` and the [official nvblox documentation](https://nvidia-isaac.github.io/nvblox/public/pages/installation.html).*

## Usage

Whenever executing scripts, prepend your command with `uv run` to ensure it runs inside the strictly locked dependency network.

### Launching the Interactive Sandbox

```bash
uv run apps/gradio_ui.py
```

### Using PRISM-VGGT as a Python Library

The public API is two objects: a perception backend (any `BasePerceptionExtractor`)
and the `StreamingWindowEngine` that fuses it into a dense map.

```python
from prism_vggt import PanoVGGTBackend, StreamingWindowEngine, FrameInput, download_weights

download_weights("checkpoints/model.pt")            # explicit, one-time
perception = PanoVGGTBackend(weights_path="checkpoints/model.pt")
engine = StreamingWindowEngine(perception, voxel_size=0.02, max_depth=4.5)
engine.processing_mode = "parallel"   # or "sequential" (lower peak VRAM)

# Each input frame carries its image, validity mask, an instantaneous camera-height
# measurement, and a timestamp/id (attached to the output pose for downstream SLAM).
frames = [FrameInput(image=img, mask=msk, camera_height=h, timestamp=t)
          for img, msk, h, t in my_capture]

for mesh, pointcloud, trajectory, floor in engine.process_sequence(
    frames, window_size=16, overlap=4, generate_esdf=True
):
    # `pointcloud` is the dense local-viz cloud for this submap.
    print(f"Submap {engine.submap_count}: map version {engine.get_map_version()}")

timestamps, poses = engine.get_poses()   # (N,), (N,4,4) -- timestamped camera poses
```

#### Streaming the point cloud to a client (deltas + snapshot)

The map is exposed as **one colored point per ~5&nbsp;cm block**, with a monotonic
version and per-block content hashes, so a client can stream only what changed and
resync robustly after a dropped/garbled update:

```python
# Fast path: only blocks that changed since the client's last version.
delta = engine.get_point_cloud_delta(since_version=client_version)
#   -> {"points", "colors", "keys", "block_hashes", "version", "from_version"}

# Resync path: full cloud + a whole-map hash to detect drift.
snap = engine.get_point_cloud_snapshot()
#   -> {"points", "colors", "keys", "block_hashes", "version", "map_hash"}
```

The library does **not** do the networking; it gives you stable block ids (`keys`),
per-block hashes, and a map version so your transport layer can do delta or partial/full
refetch. (Block *removal*, e.g. via TSDF decay, is not tracked yet.)

#### Collision distances (ESDF)

```python
engine.compute_esdf = True                  # recompute the ESDF each submap (off by default)
# ... after running process_sequence ...
sl = engine.get_esdf_slice(height=1.0)      # horizontal (constant-Z) slice, Z-up, floor at 0
# sl -> {"xs", "ys", "distance" (Hy x Wx, meters), "z", "valid"}
```

The world frame is **Z-up, right-handed** (ROS REP-103 / nvblox): the floor is at
`Z = 0` and an upright camera sits at `Z ~= camera_height`.

#### Performance / parallelism

Perception inference (GPU) and mapping (TSDF + mesh + color) run concurrently as a
one-deep **A/B double buffer**: while window *k* is being mapped, window *k+1* is
already being inferred. Windows are consumed strictly in order and none is skipped.
Set `engine.processing_mode = "sequential"` to run them one after another. Other knobs:
`engine.point_cloud_only` (skip the per-submap triangle mesh), `engine.mesh_extract_every`
(amortize mesh rebuilds), and `engine.face_size` (cubemap resolution).
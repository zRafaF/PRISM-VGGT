<div align="center">
<h1>PRISM-VGGT</h1>
<h3>Panoramic Reconstruction with Incremental SLAM and Dense Modeling from 360° Video</h3>
</div>

**PRISM-VGGT** is a streaming panoramic SLAM and dense reconstruction system for semi-realtime 360° video. 

Built for robotics, VR streaming, and spatial computing, PRISM-VGGT adapts sliding-window architectures to process equirectangular panoramic streams. By combining neural depth/pose estimation with highly optimized GPU-based Voxel Block Hashing (via `nvblox`), this pipeline addresses monocular scale drift, aligns concentric spherical shells, and outputs globally consistent, queryable 3D environments.

Originally starting as a spinoff of [LASER](https://github.com/neu-vi/LASER), we completely redesigned the architecture to integrate concepts from [VGGT-SLAM](https://github.com/MIT-SPARK/VGGT-SLAM). Using [PanoVGGT](https://github.com/YijingGuo-June/PanoVGGT) as our geometry engine, PRISM-VGGT processes frames in sliding-window batches to address monocular scale drift, utilizing NVIDIA's `nvblox` to fuse multi-view geometry into a globally consistent, dense TSDF mesh.

## Requirements
* **OS:** Linux (Ubuntu 22.04 / Debian)
* **GPU:** NVIDIA GPU with CUDA 12.4 support
* **Python:** 3.11

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


3. Download the weights:

```bash
mkdir -p checkpoints
wget -qnc [https://huggingface.co/YijingGuo/PanoVGGT/resolve/main/model.pt](https://huggingface.co/YijingGuo/PanoVGGT/resolve/main/model.pt) -O checkpoints/model.pt
```

### Handling Custom Environments (Different OS / CUDA versions)

By default, `uv sync` installs an `nvblox-torch` wheel pre-compiled for **Ubuntu 22.04 and CUDA 12.4**.

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

```python
from prism_vggt.backends.panovggt import PanoVGGTBackend
from prism_vggt.engine import StreamingWindowEngine

# 1. Initialize perception
perception = PanoVGGTBackend(weights_path="checkpoints/model.pt")

# 2. Initialize the TSDF engine
engine = StreamingWindowEngine(perception, voxel_size=0.02, max_depth=4.5)

# 3. Process a sequence of images in overlapping batches
for mesh, pointcloud, trajectory, edges in engine.process_sequence(
    frames=my_rgb_list, 
    masks=my_mask_list,
    window_size=16, 
    overlap=4
):
    print(f"Submap processed. Current map contains {len(mesh.vertices)} vertices.")
```
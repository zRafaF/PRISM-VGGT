# Building Custom `nvblox` Wheels

`nvblox` is a strict C++ library heavily dependent on PyTorch's internal CMake paths and your specific CUDA version. If the official or our provided wheel (`.whl`) does not work for your hardware, you must compile it from source.

### 1. Prerequisites
Ensure you have the required build tools and Git LFS:
```bash
apt-get update && apt-get install -y git-lfs cmake build-essential python3-dev
git lfs install
```

### 2. Clone the Target Branch

We use a specific locked version of `nvblox` (`v0.0.10`).

```bash
git clone -b v0.0.10 https://github.com/zRafaF/nvblox.git
cd nvblox
```
> Depending on your case you may want to use the official repo https://github.com/nvidia-isaac/nvblox, ours just increases the max set size for the TSDF volume. We are also using the fork for longevity reasons.

### 3. Compile the C++ Core

You **must** activate your Python virtual environment so CMake can locate PyTorch's C++ libraries. Do NOT pass `-DBUILD_PYTORCH_WRAPPER=0`.

```bash
# Assuming you are using 'uv', source the environment:
source ../.venv/bin/activate

mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH="$(python3 -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')" -DBUILD_RENDERER=0
make -j$(nproc)
```

### 4. Build the Python Wheel

Once the C++ core is compiled, navigate to the PyTorch bindings folder to package the wheel:

```bash
cd ../nvblox_torch

# Create the wheel, pretending a specific version tag to avoid local version metadata issues
SETUPTOOLS_SCM_PRETEND_VERSION="0.0.10+cu128ubuntu24" uv build --wheel --no-config \
  --config-setting="--build-option=--plat-name" \
  --config-setting="--build-option=linux_x86_64"
```

The resulting `.whl` file will be generated in `nvblox_torch/dist/`. You can upload this to GitHub Releases and install it via `uv pip install <URL>`, or install it locally:

```bash
uv pip install dist/nvblox_torch-0.0.10+cu128ubuntu24-py3-none-linux_x86_64.whl
```

> **Tip (Blackwell / RTX PRO 6000):** A from-source build compiles `nvblox` only for the **local** GPU's compute capability, so it always matches your card. This is the most reliable fix if the pre-built wheel ever errors at runtime with *"no kernel image is available for execution on the device"* on `sm_120`/`sm_122`.

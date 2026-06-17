# Building Custom `nvblox` Wheels

`nvblox` is a strict C++ library heavily dependent on PyTorch's internal CMake paths and your specific CUDA version. If the official or our provided wheel (`.whl`) does not work for your hardware, you must compile it from source.

> **When do you need this?** The prebuilt wheels are linked against a specific libtorch.
> On **torch ≥ 2.8 + cu128** (new C++ ABI) and **Blackwell GPUs** (RTX PRO 6000, `sm_120`),
> `import nvblox_torch` **segfaults at load time** — inside `ctypes`/`dlopen`, before any
> kernel runs, with *no Python traceback*. That is an ABI/arch mismatch, and the fix is a
> source build against your active `torch`. See [Troubleshooting](#troubleshooting) below.

## Quick Path (Recommended): `build_nvblox.sh`

With the project venv active, just run the helper:

```bash
./scripts/build_nvblox.sh
```

It auto-detects your GPU compute capability, CUDA version, and PyTorch C++ ABI from the
**active** `torch`, builds the C++ core and the Python wheel with matching flags, names the
wheel after your config (e.g. `nvblox_torch-0.0.10+cu128ubuntu24-...`), installs it, and runs
an import smoke test. Override `NVBLOX_REPO`, `NVBLOX_REF`, `CUDA_ARCH`, etc. via env vars
(see the header of the script). The manual steps below document what it does.

## Manual Build

### 1. Prerequisites
Ensure you have the required build tools and Git LFS:
```bash
apt-get update && apt-get install -y git-lfs cmake build-essential python3-dev
git lfs install
```

You also need the CUDA compiler (`nvcc`) on your PATH. The `nvidia/cuda:*-devel-*`
images ship it but don't always export it, which makes CMake fail with
*"No CMAKE_CUDA_COMPILER could be found."* Fix it for the session (and add to
`~/.bashrc` / the container's `ENV`):
```bash
export PATH=/usr/local/cuda/bin:$PATH
export CUDACXX=/usr/local/cuda/bin/nvcc
nvcc --version   # confirm release 12.8
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

# Two flags are critical, or the resulting .so segfaults at `import nvblox_torch`:
#
#  1. CMAKE_CUDA_ARCHITECTURES — set it to your GPU's compute capability. Auto-
#     detection returns empty on new GPUs (CMake then errors: "must be non-empty
#     if set"). RTX PRO 6000 Blackwell = sm_120 -> 120. Check with:
#        python -c "import torch; print(torch.cuda.get_device_capability(0))"
#
#  2. PRE_CXX11_ABI_LINKABLE — must match PyTorch's C++ ABI. torch 2.8 cu128 uses
#     the new ABI, so pass OFF. Verify with:
#        python -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"
#     True  -> add -DPRE_CXX11_ABI_LINKABLE=OFF   (cu128 wheels)
#     False -> omit the flag (default pre-cxx11)
cmake .. \
  -DCMAKE_PREFIX_PATH="$(python3 -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')" \
  -DBUILD_RENDERER=0 \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DPRE_CXX11_ABI_LINKABLE=OFF
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

The resulting `.whl` file will be generated in `nvblox_torch/dist/`. You can upload this to GitHub Releases and install it via `uv pip install <URL>`, or install it locally (the `linux_x86_64` tag is non-manylinux, so the skip flag is required, and `--no-deps` keeps your existing torch):

```bash
UV_SKIP_WHEEL_FILENAME_CHECK=1 uv pip install --no-deps \
  dist/nvblox_torch-0.0.10+cu128ubuntu24-py3-none-linux_x86_64.whl
```

> **Stop `uv sync` from overwriting your build.** The project's `pyproject.toml` lists
> `nvblox-torch` as a direct-URL dependency, so a plain `uv sync` will reinstall the
> (broken) prebuilt wheel over your source build. Either always launch with
> `uv run --no-sync`, or point the dependency at your local build:
>
> ```toml
> [tool.uv.sources]
> nvblox-torch = { path = "nvblox/nvblox_torch", editable = true }
> ```

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `import nvblox_torch` **segfaults** (Fatal Python error: Segmentation fault inside `ctypes`/`load_library`), no traceback | Prebuilt wheel's C++ ABI ≠ your torch's ABI (torch 2.8 cu128 = post-cxx11) | Source build with `-DPRE_CXX11_ABI_LINKABLE=OFF` |
| `CMAKE_CUDA_ARCHITECTURES must be non-empty if set` | Arch auto-detect returns empty on a new GPU | Pass `-DCMAKE_CUDA_ARCHITECTURES=120` (your card's capability) |
| `No CMAKE_CUDA_COMPILER could be found` | `nvcc` not on `PATH` | `export PATH=/usr/local/cuda/bin:$PATH` + `CUDACXX` (see Prerequisites) |
| Runtime: `no kernel image is available for execution on the device` | Library compiled for the wrong arch | Rebuild with the correct `CMAKE_CUDA_ARCHITECTURES` |
| Wheel named `...dev1+cuubuntu24...` (missing CUDA version) | No pinned version tag, so `setuptools_scm` guessed | Set `SETUPTOOLS_SCM_PRETEND_VERSION="0.0.10+cu128ubuntu24"` (the helper script does this) |

> **Tip (Blackwell / RTX PRO 6000):** A from-source build compiles `nvblox` only for the **local** GPU's compute capability, so it always matches your card. This is the most reliable fix if the pre-built wheel ever errors at runtime with *"no kernel image is available for execution on the device"* on `sm_120`/`sm_122`.

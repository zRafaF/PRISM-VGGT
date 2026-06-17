#!/usr/bin/env bash
#
# build_nvblox.sh — Build & install nvblox_torch from source, matched to THIS
# machine's GPU architecture, CUDA version, and PyTorch C++ ABI.
#
# Why this exists
# ---------------
# The prebuilt nvblox_torch wheels (PyPI / GitHub releases) are compiled against a
# specific libtorch build. On torch>=2.8 + cu128 (which uses the NEW C++ ABI,
# _GLIBCXX_USE_CXX11_ABI=1) and on brand-new Blackwell GPUs (RTX PRO 6000, sm_120),
# loading those wheels SEGFAULTS at `import nvblox_torch` — the crash happens inside
# ctypes/dlopen, before any CUDA kernel runs, with no Python traceback. Building from
# source against the active venv's torch fixes both the ABI and the GPU-arch mismatch.
#
# Usage
# -----
#   # From the repo root, with the project venv ACTIVE (or via `uv run`):
#   ./scripts/build_nvblox.sh
#
# Everything is auto-detected from the active `torch`. Override any of these via env:
#   NVBLOX_REPO   git remote        (default: https://github.com/zRafaF/nvblox.git)
#   NVBLOX_REF    git tag/branch    (default: v0.0.10)
#   NVBLOX_DIR    checkout location (default: <repo>/nvblox)
#   CUDA_HOME     CUDA toolkit root (default: /usr/local/cuda)
#   CUDA_ARCH     compute capability w/o dot, e.g. 120 (default: detected)
#   JOBS          parallel make jobs (default: nproc)
#   DO_INSTALL    1 = install the built wheel into the active venv (default: 1)
#
set -euo pipefail

# ---- Config (override via env) -------------------------------------------------
NVBLOX_REPO="${NVBLOX_REPO:-https://github.com/zRafaF/nvblox.git}"
NVBLOX_REF="${NVBLOX_REF:-v0.0.10}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NVBLOX_DIR="${NVBLOX_DIR:-${REPO_ROOT}/nvblox}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
JOBS="${JOBS:-$(nproc)}"
DO_INSTALL="${DO_INSTALL:-1}"

log() { printf '\033[1;36m[build-nvblox]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[build-nvblox] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# ---- Locate the CUDA compiler --------------------------------------------------
export PATH="${CUDA_HOME}/bin:${PATH}"
export CUDACXX="${CUDA_HOME}/bin/nvcc"
command -v nvcc >/dev/null 2>&1 || die "nvcc not found at ${CUDACXX}. Set CUDA_HOME to your CUDA toolkit (the *-devel container image ships it under /usr/local/cuda)."

# ---- Detect machine config from the ACTIVE torch -------------------------------
command -v python3 >/dev/null 2>&1 || die "python3 not found. Activate the project venv first (e.g. 'source .venv/bin/activate')."
DETECT="$(python3 - <<'PY' 2>/dev/null || true
import torch
try:
    cc = torch.cuda.get_device_capability(0); arch = f"{cc[0]}{cc[1]}"
except Exception:
    arch = ""
print(arch, int(torch._C._GLIBCXX_USE_CXX11_ABI), torch.version.cuda or "")
PY
)"
[ -n "${DETECT}" ] || die "Could not import torch. Run this with the project venv active, or 'uv run ./scripts/build_nvblox.sh'."

DET_ARCH="$(echo "${DETECT}" | awk '{print $1}')"
DET_ABI="$(echo "${DETECT}" | awk '{print $2}')"
DET_CUDA="$(echo "${DETECT}" | awk '{print $3}')"

CUDA_ARCH="${CUDA_ARCH:-${DET_ARCH}}"
ABI_ON="${ABI_ON:-${DET_ABI}}"
CUDA_VER="${CUDA_VER:-${DET_CUDA}}"

[ -n "${CUDA_ARCH}" ] || die "Could not detect GPU compute capability (no visible GPU?). Set CUDA_ARCH=120 (RTX PRO 6000) and re-run."
[ -n "${CUDA_VER}" ]  || die "Could not read torch.version.cuda. Is this a CUDA build of torch?"

# torch cu128 uses the NEW abi -> nvblox must be built post-cxx11 (linkable OFF).
if [ "${ABI_ON}" = "1" ]; then PRE_CXX11="OFF"; else PRE_CXX11="ON"; fi

# Build the version tag that goes into the wheel filename, e.g. 0.0.10+cu128ubuntu24
CUDA_SHORT="cu${CUDA_VER//./}"                                   # 12.8  -> cu128
# shellcheck disable=SC1091
UBUNTU_SHORT="ubuntu$( . /etc/os-release 2>/dev/null; echo "${VERSION_ID%%.*}" )"  # 24.04 -> ubuntu24
VERSION_TAG="${NVBLOX_REF#v}+${CUDA_SHORT}${UBUNTU_SHORT}"       # 0.0.10+cu128ubuntu24

log "GPU sm_${CUDA_ARCH} | CUDA ${CUDA_VER} (${CUDA_SHORT}) | ${UBUNTU_SHORT} | cxx11_abi=${ABI_ON} -> PRE_CXX11_ABI_LINKABLE=${PRE_CXX11}"
log "Wheel version tag: ${VERSION_TAG}"
log "Source: ${NVBLOX_REPO} @ ${NVBLOX_REF} -> ${NVBLOX_DIR}"

# ---- Fetch source --------------------------------------------------------------
if [ ! -d "${NVBLOX_DIR}/.git" ]; then
    log "Cloning nvblox..."
    git clone --branch "${NVBLOX_REF}" "${NVBLOX_REPO}" "${NVBLOX_DIR}"
fi
git lfs install >/dev/null 2>&1 || log "WARN: git-lfs not installed; tests/assets may be missing (apt-get install git-lfs)."

# ---- Build the C++ core --------------------------------------------------------
log "Configuring + building C++ core..."
rm -rf "${NVBLOX_DIR}/build"
mkdir -p "${NVBLOX_DIR}/build"
cd "${NVBLOX_DIR}/build"
cmake .. \
    -DCMAKE_PREFIX_PATH="$(python3 -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')" \
    -DBUILD_RENDERER=0 \
    -DCMAKE_CUDA_COMPILER="${CUDACXX}" \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
    -DPRE_CXX11_ABI_LINKABLE="${PRE_CXX11}"
make -j"${JOBS}"

# ---- Build the Python wheel ----------------------------------------------------
log "Building nvblox_torch wheel (${VERSION_TAG})..."
cd "${NVBLOX_DIR}/nvblox_torch"
export CUDAARCHS="${CUDA_ARCH}"   # so the bindings' own cmake pass targets the same arch
rm -rf dist
SETUPTOOLS_SCM_PRETEND_VERSION="${VERSION_TAG}" uv build --wheel --no-config \
    --config-setting="--build-option=--plat-name" \
    --config-setting="--build-option=linux_x86_64"

WHEEL="$(ls -1 "${NVBLOX_DIR}/nvblox_torch/dist/"nvblox_torch-*.whl | head -n1)"
[ -n "${WHEEL}" ] || die "Wheel build produced no .whl in dist/."
log "Built: ${WHEEL}"

# ---- Install + smoke test ------------------------------------------------------
if [ "${DO_INSTALL}" = "1" ]; then
    log "Installing into the active environment..."
    UV_SKIP_WHEEL_FILENAME_CHECK=1 uv pip install --force-reinstall --no-deps "${WHEEL}"
    log "Smoke test: import nvblox_torch ..."
    UV_SKIP_WHEEL_FILENAME_CHECK=1 python3 -c "import nvblox_torch; from nvblox_torch.mapper import Mapper; print('nvblox import OK')"
fi

log "Done. Wheel at: ${WHEEL}"
log "Tip: point pyproject.toml at this wheel so 'uv sync' won't overwrite it:"
log "      nvblox-torch @ file://${WHEEL}"

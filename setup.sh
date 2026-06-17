#!/bin/bash
set -e

echo "=============================================="
echo "    🚀 PRISM-VGGT: Environment Setup Script    "
echo "=============================================="

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# --- nvblox install mode --------------------------------------------------------
# How nvblox_torch gets installed. Resolution order:
#   1. First CLI argument:   ./setup.sh source
#   2. Env var:              NVBLOX_MODE=source ./setup.sh
#   3. Interactive menu (only when attached to a terminal)
#   4. Default = prebuilt    (so unattended / piped runs just work)
#
# Modes:
#   prebuilt  -> default wheel pinned in pyproject.toml (fastest)
#   source    -> compile via scripts/build_nvblox.sh (matches GPU/CUDA/ABI)
#   url       -> a specific wheel URL (set NVBLOX_WHEEL_URL or you will be asked)
NVBLOX_MODE="${1:-${NVBLOX_MODE:-}}"
NVBLOX_WHEEL_URL="${NVBLOX_WHEEL_URL:-}"
DEFAULT_NVBLOX_URL="https://github.com/nvidia-isaac/nvblox/releases/download/v0.0.10/nvblox_torch-0.0.10+cu12ubuntu24-py3-none-linux_x86_64.whl"

choose_nvblox_mode() {
    [ -n "$NVBLOX_MODE" ] && return            # already chosen via arg/env
    if [ ! -t 0 ]; then NVBLOX_MODE="prebuilt"; return; fi   # non-interactive default
    echo ""
    echo "How should nvblox_torch be installed?"
    echo "  1) Pre-built wheel   - default URL, fastest (may segfault on Blackwell + cu128)"
    echo "  2) Build from source - matches your GPU/CUDA/ABI (recommended for RTX PRO 6000)"
    echo "  3) Pre-built wheel   - from a custom URL you provide"
    local choice=""
    read -r -t 60 -p "Select [1/2/3] (default 1, auto in 60s): " choice || choice=""
    case "$choice" in
        2) NVBLOX_MODE="source" ;;
        3) NVBLOX_MODE="url" ;;
        *) NVBLOX_MODE="prebuilt" ;;
    esac
}

choose_nvblox_mode
echo "[*] nvblox install mode: $NVBLOX_MODE"
case "$NVBLOX_MODE" in
    prebuilt|source|url) : ;;
    *) echo "[!] Unknown nvblox mode '$NVBLOX_MODE'; using 'prebuilt'."; NVBLOX_MODE="prebuilt" ;;
esac

# 1. Initialize UV
if ! command -v uv &> /dev/null; then
    echo "[!] 'uv' not found. Installing astral/uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Make uv available for the remainder of this script
export PATH="$HOME/.local/bin:$PATH"

# Persist uv to the user's shell profile so it survives after this script exits
SHELL_RC="$HOME/.bashrc"
[ -f "$HOME/.zshrc" ] && SHELL_RC="$HOME/.zshrc"

if ! grep -q 'local/bin' "$SHELL_RC" 2>/dev/null; then
    echo "" >> "$SHELL_RC"
    echo '# Added by PRISM-VGGT setup' >> "$SHELL_RC"
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$SHELL_RC"
    echo "[*] Added \$HOME/.local/bin to PATH in $SHELL_RC"
fi

# 2. Ensure the PanoVGGT submodule is present (the package imports from it)
if [ ! -f third_party/PanoVGGT/inference.py ]; then
    echo "[*] Initializing git submodules (PanoVGGT)..."
    git submodule update --init --recursive
fi

# 3. Environment knobs required for this stack
#  - nvblox_torch ships a non-manylinux wheel ('+cu12ubuntu24 ... linux_x86_64')
#    whose filename trips uv's wheel-name validation. Skip that check.
#  - Use the system Python 3.12 (installed in the container) instead of letting
#    uv download a managed interpreter, which fails on hosts without GitHub access.
export UV_SKIP_WHEEL_FILENAME_CHECK=1
export UV_PYTHON_PREFERENCE=system

# 4. If a custom wheel URL was chosen, pin it BEFORE syncing so uv installs it.
if [ "$NVBLOX_MODE" = "url" ]; then
    if [ -z "$NVBLOX_WHEEL_URL" ] && [ -t 0 ]; then
        read -r -p "Enter nvblox wheel URL: " NVBLOX_WHEEL_URL || NVBLOX_WHEEL_URL=""
    fi
    if [ -z "$NVBLOX_WHEEL_URL" ]; then
        echo "[!] No URL provided; falling back to the default pre-built wheel."
        NVBLOX_MODE="prebuilt"
    else
        echo "[*] Pinning custom nvblox wheel: $NVBLOX_WHEEL_URL"
        uv add "nvblox-torch @ $NVBLOX_WHEEL_URL"
    fi
fi

# 5. Sync standard dependencies (creates .venv, installs torch 2.8 / cu128 + deps)
echo "[*] Syncing Python environment using uv (CUDA 12.8 / torch 2.8)..."
uv sync

# 6. Source build (after sync, so the venv's torch is available to compile against).
if [ "$NVBLOX_MODE" = "source" ]; then
    echo "[*] Building nvblox from source to match this machine..."

    # Make sure the C++ build toolchain is present (best effort; container runs as root).
    SUDO=""; [ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null 2>&1 && SUDO="sudo"
    NEED=()
    command -v cmake    >/dev/null 2>&1 || NEED+=(cmake)
    command -v git-lfs  >/dev/null 2>&1 || NEED+=(git-lfs)
    command -v c++      >/dev/null 2>&1 || NEED+=(build-essential)
    if [ "${#NEED[@]}" -gt 0 ] && command -v apt-get >/dev/null 2>&1; then
        echo "[*] Installing build tools: ${NEED[*]}"
        $SUDO apt-get update -qq && $SUDO apt-get install -y "${NEED[@]}" python3-dev || \
            echo "[!] Could not auto-install build tools; install them manually if the build fails."
    fi

    # Run the builder inside the project venv so it compiles against the right torch.
    # build_nvblox.sh auto-detects GPU arch, CUDA version, and C++ ABI.
    source "$REPO_ROOT/.venv/bin/activate"
    "$REPO_ROOT/scripts/build_nvblox.sh"
fi

# 7. Download weights
echo "[*] Downloading PanoVGGT backbone weights..."
mkdir -p checkpoints
WEIGHTS_URL="https://huggingface.co/YijingGuo/PanoVGGT/resolve/main/model.pt"
if [ ! -f checkpoints/model.pt ]; then
    if command -v wget >/dev/null 2>&1; then
        wget -q --show-progress -O checkpoints/model.pt "$WEIGHTS_URL"
    else
        curl -L --fail --progress-bar -o checkpoints/model.pt "$WEIGHTS_URL"
    fi
    echo "[*] Weights downloaded."
else
    echo "[*] Weights already present, skipping."
fi

echo "=============================================="
echo "✅ Setup Complete! To launch the UI, run:"
echo "   uv sync --extra apps"
echo "   uv run apps/gradio_ui.py"
echo ""
if [ "$NVBLOX_MODE" = "source" ]; then
echo "💡 You built nvblox from source. To stop 'uv sync' from reinstalling the"
echo "   pre-built wheel over it, either launch with 'uv run --no-sync', or add to"
echo "   pyproject.toml:"
echo "     [tool.uv.sources]"
echo "     nvblox-torch = { path = \"nvblox/nvblox_torch\", editable = true }"
echo ""
fi
echo "💡 If this is your first time running setup, reload"
echo "   your shell first:  source $SHELL_RC"
echo "=============================================="

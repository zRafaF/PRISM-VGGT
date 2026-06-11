#!/bin/bash
set -e

echo "=============================================="
echo "    🚀 PRISM-VGGT: Environment Setup Script    "
echo "=============================================="

# 1. Initialize UV
if ! command -v uv &> /dev/null; then
    echo "[!] 'uv' not found. Installing astral/uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env
fi

# 2. Sync standard dependencies
echo "[*] Syncing Python environment using uv..."
uv sync

# 3. Download weights
echo "[*] Downloading PanoVGGT backbone weights..."
mkdir -p checkpoints
wget -qnc https://huggingface.co/YijingGuo/PanoVGGT/resolve/main/model.pt -O checkpoints/model.pt
echo "[*] Weights downloaded."

# 4. Install specific NVBLOX wheel
echo "[*] Installing custom Nvblox tensor bindings..."
UV_SKIP_WHEEL_FILENAME_CHECK=1 uv pip install https://github.com/zRafaF/nvblox/releases/download/v0.0.10/nvblox_torch-0.0.10.dev1+cuubuntu22-py3-none-linux_x86_64.whl

echo "=============================================="
echo "✅ Setup Complete! To launch the UI, run:"
echo "   uv sync --extra apps"
echo "   uv run apps/gradio_ui.py"
echo "=============================================="
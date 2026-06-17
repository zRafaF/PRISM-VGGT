#!/bin/bash
set -e

echo "=============================================="
echo "    🚀 PRISM-VGGT: Environment Setup Script    "
echo "=============================================="

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

# 4. Sync standard dependencies
echo "[*] Syncing Python environment using uv (CUDA 12.8 / torch 2.8)..."
uv sync

# 5. Download weights
echo "[*] Downloading PanoVGGT backbone weights..."
mkdir -p checkpoints
# -q suppresses headers/noise, --show-progress forces the progress bar
wget -q --show-progress -nc \
    https://huggingface.co/YijingGuo/PanoVGGT/resolve/main/model.pt \
    -O checkpoints/model.pt
echo "[*] Weights downloaded."

echo "=============================================="
echo "✅ Setup Complete! To launch the UI, run:"
echo "   uv sync --extra apps"
echo "   uv run apps/gradio_ui.py"
echo ""
echo "💡 If this is your first time running setup, reload"
echo "   your shell first:  source $SHELL_RC"
echo "=============================================="
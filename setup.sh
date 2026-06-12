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

# 2. Sync standard dependencies
echo "[*] Syncing Python environment using uv..."
uv sync

# 3. Download weights
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
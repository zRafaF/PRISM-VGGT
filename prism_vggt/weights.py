"""PanoVGGT checkpoint management.

The library never reaches out to the network on its own. Call ``download_weights``
explicitly (e.g. from your install/setup step) to fetch the checkpoint before
constructing :class:`~prism_vggt.backends.panovggt.PanoVGGTBackend`.
"""
import os
import sys
import urllib.request

PANOVGGT_WEIGHTS_URL = "https://huggingface.co/YijingGuo/PanoVGGT/resolve/main/model.pt"
DEFAULT_WEIGHTS_PATH = "checkpoints/model.pt"


def _progress(block_num: int, block_size: int, total_size: int) -> None:
    if total_size <= 0:
        return
    done = min(block_num * block_size, total_size)
    pct = 100.0 * done / total_size
    filled = int(pct // 2)
    sys.stdout.write(
        f"\r[PRISM] Downloading weights |{'#' * filled}{'.' * (50 - filled)}| "
        f"{pct:5.1f}% ({done / 1e6:6.1f} / {total_size / 1e6:6.1f} MB)"
    )
    sys.stdout.flush()


def download_weights(dest: str = DEFAULT_WEIGHTS_PATH,
                     url: str = PANOVGGT_WEIGHTS_URL,
                     force: bool = False) -> str:
    """Download the PanoVGGT checkpoint to ``dest`` and return its path.

    Args:
        dest: Destination file path (parent dirs are created as needed).
        url: Source URL. Defaults to the official Hugging Face release.
        force: Re-download even if the file already exists.

    Returns:
        The absolute path to the downloaded checkpoint.
    """
    dest = os.path.abspath(dest)
    if os.path.exists(dest) and not force:
        print(f"[PRISM] Weights already present at {dest}")
        return dest

    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    tmp = dest + ".part"
    print(f"[PRISM] Fetching PanoVGGT weights from {url}")
    try:
        urllib.request.urlretrieve(url, tmp, _progress)
        sys.stdout.write("\n")
        os.replace(tmp, dest)
    except Exception as e:  # pragma: no cover - network dependent
        if os.path.exists(tmp):
            os.remove(tmp)
        raise RuntimeError(f"Failed to download PanoVGGT weights from {url}: {e}") from e

    print(f"[PRISM] Weights saved to {dest}")
    return dest


def missing_weights_message(weights_path: str) -> str:
    """Human-readable instructions shown when the checkpoint is absent."""
    return (
        f"PanoVGGT checkpoint not found at '{weights_path}'.\n"
        f"Fetch it explicitly before loading the backend, e.g.:\n"
        f"    python -c \"from prism_vggt import download_weights; "
        f"download_weights('{weights_path}')\"\n"
        f"or download it manually from {PANOVGGT_WEIGHTS_URL}"
    )

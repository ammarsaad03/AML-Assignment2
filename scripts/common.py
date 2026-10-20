"""Shared helpers for all model wrappers: identical image I/O, logging and timing for every model."""
import os

# Must be set before numpy is imported: OpenBLAS (numpy's BLAS in aml2) otherwise reserves buffers for all 12 CPU
# threads (~1.9 GB committed per process vs ~0.5 GB), which exhausts the Windows commit limit on this 8 GB laptop
# once data-loader workers are spawned. Model computation runs in torch and is not affected.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import hashlib  # noqa: E402
import logging  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

ROOT =Path(__file__).resolve().parents[1]
DATASETS = {
    "LOL-v1": {"low": ROOT / "data" / "LOL-v1" / "eval15" / "low", "high": ROOT / "data" / "LOL-v1" / "eval15" / "high"},
    "LSRW": {"low": ROOT / "data" / "LSRW" / "eval" / "low", "high": ROOT / "data" / "LSRW" / "eval" / "high"},
}
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp"}


def list_images(folder):
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMG_EXT)


def load_rgb01(path, device):
    """Read an image as RGB float32 in [0, 1], full resolution, shape (1, 3, H, W) on `device`."""
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def save_rgb01(tensor, path):
    """Clip a (1, 3, H, W) tensor to [0, 1] and save it as an 8-bit RGB PNG (rounded to nearest)."""
    arr = tensor[0].detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    Image.fromarray(np.round(arr * 255.0).astype(np.uint8)).save(path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get_logger(log_path, mode="w"):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(log_path.stem)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    for h in (logging.FileHandler(log_path, mode=mode, encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


def log_environment(logger):
    logger.info(f"python {platform.python_version()} | torch {torch.__version__} | CUDA build {torch.version.cuda}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")


class CudaTimer:
    """Wall-clock time of a GPU forward pass (synchronised before and after)."""

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.seconds = time.perf_counter() - self.t0

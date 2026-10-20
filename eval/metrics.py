"""Shared evaluation code, identical for all models.

- PSNR and SSIM on RGB images in [0, 1], full resolution, no GT-mean correction
  (skimage.metrics, data_range=1, channel_axis=-1; SSIM uses skimage defaults: 7x7 uniform window).
- LPIPS with the AlexNet backbone (lpips package, v0.1); inputs scaled from [0, 1] to [-1, 1].

Predictions are matched to ground truth by file stem (e.g. pred "x.png" <-> GT "x.jpg").

Usage:
    python eval/metrics.py --pred results/SCI/LOL-v1 --gt data/LOL-v1/eval15/high --csv results/SCI_LOL-v1_metrics.csv
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp"}
_lpips_model = None


def read_rgb01(path):
    """Read an image as RGB float64 in [0, 1], shape (H, W, 3)."""
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0


def get_lpips(device):
    global _lpips_model
    if _lpips_model is None:
        import lpips
        _lpips_model = lpips.LPIPS(net="alex", version="0.1", verbose=False).to(device).eval()
    return _lpips_model


def psnr(pred, gt):
    return peak_signal_noise_ratio(gt, pred, data_range=1)


def ssim(pred, gt):
    return structural_similarity(gt, pred, data_range=1, channel_axis=-1)


@torch.no_grad()
def lpips_score(pred, gt, device):
    to_t = lambda x: torch.from_numpy(x).permute(2, 0, 1)[None].float().to(device) * 2 - 1
    return get_lpips(device)(to_t(pred), to_t(gt)).item()


def list_images(folder):
    return {p.stem: p for p in sorted(Path(folder).iterdir()) if p.suffix.lower() in IMG_EXT}


def evaluate_folder(pred_dir, gt_dir, csv_path=None, device=None):
    """Score every image in gt_dir against the same-stem image in pred_dir.

    Returns (rows, means): rows is a list of dicts {name, psnr, ssim, lpips}; means is a dict of the averages.
    Raises if a prediction is missing or its size differs from the ground truth.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    preds, gts = list_images(pred_dir), list_images(gt_dir)
    missing = sorted(set(gts) - set(preds))
    if missing:
        raise FileNotFoundError(f"{len(missing)} predictions missing in {pred_dir}: {missing[:5]}")

    rows = []
    for stem, gt_path in gts.items():
        pred, gt = read_rgb01(preds[stem]), read_rgb01(gt_path)
        if pred.shape != gt.shape:
            raise ValueError(f"{stem}: prediction {pred.shape} vs ground truth {gt.shape}")
        rows.append({"name": gt_path.name, "psnr": psnr(pred, gt), "ssim": ssim(pred, gt),
                     "lpips": lpips_score(pred, gt, device)})

    means = {k: float(np.mean([r[k] for r in rows])) for k in ("psnr", "ssim", "lpips")}
    if csv_path:
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["name", "psnr", "ssim", "lpips"])
            w.writeheader()
            w.writerows({"name": r["name"], **{k: f"{r[k]:.6f}" for k in ("psnr", "ssim", "lpips")}} for r in rows)
            w.writerow({"name": "MEAN", **{k: f"{v:.6f}" for k, v in means.items()}})
    return rows, means


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="PSNR / SSIM / LPIPS of a results folder against ground truth")
    ap.add_argument("--pred", required=True, help="folder with enhanced images")
    ap.add_argument("--gt", required=True, help="folder with ground-truth (normal-light) images")
    ap.add_argument("--csv", help="where to save the per-image CSV")
    args = ap.parse_args()
    rows, means = evaluate_folder(args.pred, args.gt, args.csv)
    print(f"{len(rows)} images | PSNR {means['psnr']:.3f} dB | SSIM {means['ssim']:.4f} | LPIPS {means['lpips']:.4f}")

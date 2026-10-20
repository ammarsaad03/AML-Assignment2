"""Analysis for the supplementary RUAS-long run (50,000 iterations) vs the Protocol B RUAS run (800 iterations).

For both final checkpoints: validation PSNR and mean output brightness on the 35 held-out pairs (computed with the
test wrapper), and on both test sets the mean brightness and the share of blown-out output pixels (>= 0.98, read
from the saved outputs). Writes results/analysis_ruas_long.csv (one row per run and image set).

Usage: python scripts/analysis_ruas_long.py
"""
import csv

import numpy as np
import torch
from PIL import Image

from common import DATASETS, ROOT, list_images
from run_ruas import build_model, enhance
from train_common import ValSet, to_tensor01

RUNS = {"RUAS_B": ROOT / "checkpoints" / "RUAS" / "final.pt",
        "RUAS_B_long": ROOT / "checkpoints" / "RUAS_long" / "final.pt"}


def main():
    val = ValSet()
    gt_val = float(np.mean([h.mean() for h in val.high]))
    rows = []
    for name, path in RUNS.items():
        model, _ = build_model(path, "cuda")
        with torch.no_grad():
            outs = [enhance(model, to_tensor01(lo).cuda()).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy() for lo in val.low]
            vpsnr = val.mean_psnr(lambda x: enhance(model, x))
        rows.append({"run": name, "image_set": "validation (35 pairs)", "psnr": vpsnr,
                     "mean_brightness": float(np.mean([o.mean() for o in outs])), "gt_mean_brightness": gt_val,
                     "blown_out_share": float(np.mean([(o >= 0.98).mean() for o in outs]))})
        for ds, d in DATASETS.items():
            gts = list_images(d["high"])
            ims = [np.asarray(Image.open(ROOT / "results" / name / ds / f"{p.stem}.png").convert("RGB"), dtype=np.float32) / 255
                   for p in gts]
            gt_b = float(np.mean([np.asarray(Image.open(p).convert("RGB"), dtype=np.float32).mean() / 255 for p in gts]))
            rows.append({"run": name, "image_set": ds, "psnr": "", "mean_brightness": float(np.mean([i.mean() for i in ims])),
                         "gt_mean_brightness": gt_b, "blown_out_share": float(np.mean([(i >= 0.98).mean() for i in ims]))})
    path = ROOT / "results" / "analysis_ruas_long.csv"
    fields = ["run", "image_set", "psnr", "mean_brightness", "gt_mean_brightness", "blown_out_share"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
            print(" | ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in r.items()))
    print("written:", path.relative_to(ROOT))


if __name__ == "__main__":
    main()

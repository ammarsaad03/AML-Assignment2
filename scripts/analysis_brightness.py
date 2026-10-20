"""Analysis: mean brightness (mean of RGB in [0, 1]) of the input, the ground truth and every model's outputs.

Used in the report to explain why some retrained models score lower (e.g. retrained SCI brightens less).
Writes results/analysis_brightness.csv (dataset, source, mean_brightness).

Usage: python scripts/analysis_brightness.py
"""
import csv

import numpy as np
from PIL import Image

from common import DATASETS, ROOT, list_images

FOLDERS = ["SCI", "PairLIE", "URetinex-Net", "Retinexformer", "RUAS", "SCI_B", "PairLIE_B", "Retinexformer_B", "RUAS_B"]


def mean_brightness(paths):
    return float(np.mean([np.asarray(Image.open(p).convert("RGB"), dtype=np.float32).mean() / 255.0 for p in paths]))


def main():
    rows = []
    for ds, d in DATASETS.items():
        gts = list_images(d["high"])
        rows.append({"dataset": ds, "source": "input (low)", "mean_brightness": mean_brightness(list_images(d["low"]))})
        rows.append({"dataset": ds, "source": "ground truth", "mean_brightness": mean_brightness(gts)})
        for f in FOLDERS:
            outs = [ROOT / "results" / f / ds / f"{p.stem}.png" for p in gts]
            if all(o.exists() for o in outs):
                rows.append({"dataset": ds, "source": f, "mean_brightness": mean_brightness(outs)})
    path = ROOT / "results" / "analysis_brightness.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["dataset", "source", "mean_brightness"])
        w.writeheader()
        w.writerows({**r, "mean_brightness": f"{r['mean_brightness']:.4f}"} for r in rows)
    for r in rows:
        print(f"{r['dataset']:7s} {r['source']:16s} {r['mean_brightness']:.3f}")
    print("written:", path.relative_to(ROOT))


if __name__ == "__main__":
    main()

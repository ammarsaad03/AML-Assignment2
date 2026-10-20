"""Score every model on every dataset with eval/metrics.py.

Writes results/<model>_<dataset>_metrics.csv (per image + MEAN) and results/metrics_summary.csv
(mean per model, dataset and subset; LSRW subsets = filename prefix huawei_ / nikon_).

Usage: python scripts/evaluate_all.py [--datasets LOL-v1 LSRW]
"""
import argparse
import csv
import sys

import numpy as np

from common import DATASETS, ROOT

sys.path.insert(0, str(ROOT))
from eval.metrics import evaluate_folder  # noqa: E402

MODELS = ["SCI", "PairLIE", "URetinex-Net", "Retinexformer", "RUAS"]  # Protocol A folders (default)
METRICS = ("psnr", "ssim", "lpips")


def subsets(dataset, rows):
    groups = {"all": rows}
    if dataset == "LSRW":
        for prefix in ("huawei", "nikon"):
            groups[prefix] = [r for r in rows if r["name"].startswith(prefix + "_")]
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    ap.add_argument("--models", nargs="+", default=MODELS,
                    help="result folder names, e.g. SCI PairLIE_B (Protocol B folders end in _B)")
    args = ap.parse_args()

    summary = []
    for dataset in args.datasets:
        for model in args.models:
            rows, _ = evaluate_folder(ROOT / "results" / model / dataset, DATASETS[dataset]["high"],
                                      ROOT / "results" / f"{model}_{dataset}_metrics.csv")
            for subset, sub_rows in subsets(dataset, rows).items():
                means = {k: float(np.mean([r[k] for r in sub_rows])) for k in METRICS}
                summary.append({"model": model, "dataset": dataset, "subset": subset, "n": len(sub_rows), **means})
                print(f"{dataset:7s} {subset:6s} {model:14s} n={len(sub_rows):2d} | PSNR {means['psnr']:.3f} | "
                      f"SSIM {means['ssim']:.4f} | LPIPS {means['lpips']:.4f}")

    # merge with an existing summary so running one dataset keeps the other
    path = ROOT / "results" / "metrics_summary.csv"
    keep = []
    if path.exists():
        with open(path, encoding="utf-8") as f:
            keep = [r for r in csv.DictReader(f) if not (r["dataset"] in args.datasets and r["model"] in args.models)]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "dataset", "subset", "n", *METRICS])
        w.writeheader()
        w.writerows(keep)
        w.writerows({**r, **{k: f"{r[k]:.6f}" for k in METRICS}} for r in summary)


if __name__ == "__main__":
    main()

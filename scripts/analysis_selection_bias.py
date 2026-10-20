"""Analysis (NOT a reported result): how much does choosing a checkpoint ON THE TEST SET inflate the score?

The official Retinexformer option file validates on LOL-v1 Test and keeps the best checkpoint by that score.
This script scores every checkpoint saved during our Protocol B run (one per 1000 iterations) on LOL-v1 eval15 with
exactly our test pipeline (official padding, clamp, 8-bit rounding, eval/metrics.py PSNR/SSIM), and compares
  - our honest selection: best validation PSNR on the 35 held-out training pairs (iteration from best_psnr_*.pth), with
  - "best on test": the checkpoint with the highest LOL-v1 eval15 PSNR (what validating on the test set would pick).
Writes results/analysis_retinexformer_selection.csv (iter, val_psnr, test_psnr, test_ssim).

Usage: python scripts/analysis_selection_bias.py
"""
import csv
import re

import numpy as np
import torch

from common import DATASETS, ROOT, list_images, load_rgb01
from eval.metrics import psnr, read_rgb01, ssim
from run_retinexformer import PROTOCOL_B_DIR, build_model, enhance, load_arch  # noqa: F401


def main():
    device = "cuda"
    lows = [load_rgb01(p, device) for p in list_images(DATASETS["LOL-v1"]["low"])]
    gts = [read_rgb01(p) for p in list_images(DATASETS["LOL-v1"]["high"])]

    val = {}
    for line in (PROTOCOL_B_DIR / "metric.csv").read_text(encoding="utf-8").splitlines():
        m = re.fullmatch(r"\s*(\d+),([0-9.]+)\s*", line)
        if m:
            val[int(m.group(1))] = float(m.group(2))
    selected = int(re.search(r"_(\d+)\.pth$", next(PROTOCOL_B_DIR.glob("best_psnr_*.pth")).name).group(1))

    ckpts = sorted(PROTOCOL_B_DIR.glob("models/net_g_[0-9]*.pth"), key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)))
    model, _, _ = build_model(device, ckpts[0])
    rows = []
    with torch.no_grad():
        for p in ckpts:
            it = int(re.search(r"(\d+)", p.stem).group(1))
            model.load_state_dict(torch.load(p, map_location="cpu", weights_only=False)["params"], strict=True)
            ps, ss = [], []
            for x, gt in zip(lows, gts):
                out = enhance(model, x).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
                out = np.round(out * 255.0) / 255.0  # same 8-bit rounding as the saved test outputs
                ps.append(psnr(out, gt)); ss.append(ssim(out, gt))
            rows.append({"iter": it, "val_psnr": val.get(it), "test_psnr": float(np.mean(ps)), "test_ssim": float(np.mean(ss))})
            if it % 10000 == 0:
                print(f"iter {it:>6}: val {val.get(it, float('nan')):.3f} | LOL-v1 test PSNR {rows[-1]['test_psnr']:.3f}", flush=True)

    out = ROOT / "results" / "analysis_retinexformer_selection.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["iter", "val_psnr", "test_psnr", "test_ssim"])
        w.writeheader()
        w.writerows(rows)

    sel = next(r for r in rows if r["iter"] == selected)
    best = max(rows, key=lambda r: r["test_psnr"])
    tp = np.array([r["test_psnr"] for r in rows]); vp = np.array([r["val_psnr"] for r in rows])
    print(f"\nselected on validation : iter {selected:>6} | LOL-v1 test PSNR {sel['test_psnr']:.3f} | SSIM {sel['test_ssim']:.4f}")
    print(f"best on test (leak)    : iter {best['iter']:>6} | LOL-v1 test PSNR {best['test_psnr']:.3f} | SSIM {best['test_ssim']:.4f}")
    print(f"inflation from selecting on the test set: +{best['test_psnr'] - sel['test_psnr']:.3f} dB")
    print(f"test PSNR over all {len(rows)} checkpoints: mean {tp.mean():.3f}, std {tp.std():.3f}, min {tp.min():.3f}, max {tp.max():.3f}")
    print(f"last 50 checkpoints (100k-150k): mean {tp[-50:].mean():.3f}, max {tp[-50:].max():.3f}")
    print(f"Pearson correlation validation vs test PSNR: {np.corrcoef(vp, tp)[0, 1]:.3f}")
    print("written:", out.relative_to(ROOT))


if __name__ == "__main__":
    main()

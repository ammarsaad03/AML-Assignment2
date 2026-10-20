"""Plot a Protocol B training curve: training loss (top) and validation PSNR on the 35 held-out pairs (bottom).

- SCI, PairLIE: read logs/train_<model>.csv (epoch, iters, train_loss, val_psnr, lr, ...); x-axis = epoch;
  tested checkpoint = final.
- Retinexformer: read basicsr's output - loss from the progress lines in logs/train_Retinexformer.log
  ("iter: 1,000, lr:(...)] ... l_pix: 8.1e-02"), validation PSNR from <experiment>/metric.csv ("iter,psnr" lines,
  appended on resume -> the last value per iteration is kept); x-axis = iteration; tested checkpoint = the iteration in
  the best_psnr_<psnr>_<iter>.pth file name.
Writes figures/train_curve_<model>.png (300 dpi). Learning-rate drops / restarts are faint dashed vertical lines.

Usage: python scripts/plot_train_curve.py --model PairLIE | SCI | Retinexformer
"""
import argparse
import csv
import re

from common import ROOT  # first: sets OPENBLAS_NUM_THREADS before numpy/matplotlib load

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

INK, INK_2, GRID, SURFACE, SERIES = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#2a78d6"
RF_EXP = ROOT / "checkpoints" / "Retinexformer" / "RetinexFormer_LOL_v1_protocolB"
RF_LOG = ROOT / "logs" / "train_Retinexformer.log"


def read_csv_curves(model):
    """Our own training CSV: SCI/PairLIE per epoch (epoch, train_loss, lr), RUAS per 50 iterations
    (iter, enhance_loss = IEM loss, lr_iem). Validation PSNR where logged; lr drops marked."""
    with open(ROOT / "logs" / f"train_{model}.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    num = lambda v: float(v) if v not in (None, "") else None
    per_epoch = "epoch" in rows[0]
    xcol, losscol, lrcol = ("epoch", "train_loss", "lr") if per_epoch else ("iter", "enhance_loss", "lr_iem")
    pts = [(num(r[xcol]), num(r[losscol])) for r in rows if num(r[losscol]) is not None]
    val = [(num(r[xcol]), num(r["val_psnr"])) for r in rows if num(r["val_psnr"]) is not None]
    lrs = [(num(r[xcol]), num(r[lrcol])) for r in rows if num(r[lrcol]) is not None]
    drops = [lrs[i][0] for i in range(1, len(lrs)) if lrs[i][1] < lrs[i - 1][1]]
    return {"x": [p[0] for p in pts], "loss": [p[1] for p in pts], "val": val, "marks": drops, "mark_label": "lr halved",
            "selected": num(rows[-1][xcol]), "xlabel": "Epoch" if per_epoch else "Iteration",
            "loss_label": "Training loss (log scale)" if per_epoch else "IEM training loss (log scale)", "log_loss": True}


def read_retinexformer_curves(bin_size=1000):
    """basicsr log + metric.csv (Retinexformer)."""
    pat = re.compile(r"iter:\s*([\d,]+), lr:.*?l_pix: ([0-9.eE+-]+)")
    per_iter = {}
    for line in RF_LOG.read_text(encoding="utf-8", errors="replace").splitlines():
        m = pat.search(line)
        if m:
            per_iter[int(m.group(1).replace(",", ""))] = float(m.group(2))  # resumed runs: last value wins
    its = np.array(sorted(per_iter))
    ls = np.array([per_iter[i] for i in its])
    bins = (its - 1) // bin_size  # mean of the printed losses in each block of 1000 iterations
    x = [float((b + 1) * bin_size) for b in np.unique(bins)]
    loss = [float(ls[bins == b].mean()) for b in np.unique(bins)]

    val_by_iter = {}
    for line in (RF_EXP / "metric.csv").read_text(encoding="utf-8").splitlines():
        m = re.fullmatch(r"\s*(\d+),([0-9.]+)\s*", line)
        if m:
            val_by_iter[int(m.group(1))] = float(m.group(2))
    val = sorted((float(k), v) for k, v in val_by_iter.items())

    best = sorted(RF_EXP.glob("best_psnr_*.pth"))
    selected = float(re.search(r"_(\d+)\.pth$", best[0].name).group(1)) if len(best) == 1 else None
    return {"x": x, "loss": loss, "val": val, "marks": [46000.0], "mark_label": "LR schedule restart (46k)",
            "selected": selected, "xlabel": "Iteration", "loss_label": f"L1 training loss\n(mean per {bin_size} it.)",
            "log_loss": False}


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["PairLIE", "SCI", "Retinexformer", "RUAS", "RUAS_long"])
    args = ap.parse_args()

    c = read_retinexformer_curves() if args.model == "Retinexformer" else read_csv_curves(args.model)
    rule = "final checkpoint" if args.model != "Retinexformer" else "best validation PSNR"

    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": INK_2})
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.0, 5.6), sharex=True, gridspec_kw={"hspace": 0.18})
    fig.patch.set_facecolor(SURFACE)
    for ax in (ax1, ax2):
        style(ax)
        for m in c["marks"]:
            ax.axvline(m, color=GRID, linewidth=1.0, linestyle="--", zorder=0)

    ax1.plot(c["x"], c["loss"], color=SERIES, linewidth=2)
    if c["log_loss"]:
        ax1.set_yscale("log")
        # label intermediate ticks too, so narrow ranges (e.g. SCI 0.4-4) are readable
        ax1.yaxis.set_minor_formatter(matplotlib.ticker.LogFormatter(labelOnlyBase=False, minor_thresholds=(2, 0.5)))
        ax1.tick_params(axis="y", which="minor", labelsize=8, colors=INK_2)
    ax1.set_ylabel(c["loss_label"], fontsize=10)
    ax1.set_title(f"{args.model} - Protocol B training (LOL-v1 train, seed 42)", fontsize=12, color=INK, loc="left", pad=10)

    vx, vy = zip(*c["val"])
    ax2.plot(vx, vy, color=SERIES, linewidth=2, marker="o", markersize=4 if len(vx) <= 60 else 0,
             markeredgecolor=SURFACE, markeredgewidth=1)
    if c["selected"] is not None:
        sel = min(c["val"], key=lambda p: abs(p[0] - c["selected"]))
        ax2.plot([sel[0]], [sel[1]], marker="o", markersize=9, color=SERIES, markeredgecolor=INK, markeredgewidth=1.5)
        ax2.annotate(f"tested checkpoint ({rule})\n{c['xlabel'].lower()} {int(sel[0]):,}: {sel[1]:.2f} dB", xy=sel,
                     xytext=(0.97, 0.08), textcoords="axes fraction", ha="right", va="bottom", fontsize=9,
                     color=INK_2, arrowprops={"arrowstyle": "-", "color": INK_2, "linewidth": 0.8})
    best = max(c["val"], key=lambda p: p[1])
    if c["selected"] is None or abs(best[0] - c["selected"]) > 1e-9:
        ax2.annotate(f"max {best[1]:.2f} dB ({c['xlabel'].lower()} {int(best[0]):,})", xy=best, xytext=(0, 10),
                     textcoords="offset points", ha="center", fontsize=9, color=INK_2)
    ax2.set_ylabel("Validation PSNR (dB)\n35 held-out pairs", fontsize=10)
    ax2.set_xlabel(c["xlabel"], fontsize=10)
    if c["xlabel"] == "Iteration" and max(c["x"]) >= 10000:
        ax2.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
    if c["marks"]:
        ax2.text(c["marks"][0], ax2.get_ylim()[0], " " + c["mark_label"], fontsize=8, color=INK_2, va="bottom")

    out = ROOT / "figures" / f"train_curve_{args.model}.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=SURFACE)
    print("written:", out.relative_to(ROOT))


if __name__ == "__main__":
    main()

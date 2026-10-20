"""Report figures (PNG, 300 dpi).

1. figures/qualitative_<dataset>.png  - rows = 3 test images, columns = Input | SCI | PairLIE | URetinex-Net |
   Retinexformer | Ground truth, with PSNR / LPIPS under each result. Images are chosen automatically as those with
   the LARGEST PSNR spread between the four models (clear differences); for LSRW the best Huawei and the best Nikon
   image are always included. Protocol A outputs by default (--protocol B for the retrained models).
2. figures/psnr_ssim_bar.png          - grouped bars: rows = PSNR, SSIM; columns = Protocol A, Protocol B;
   groups = datasets; bars = models (fixed colours). Bars start at 0.
3. figures/quality_vs_speed.png       - LSRW PSNR vs runtime (ms/image on LSRW, log scale); marker area grows with
   the parameter count; filled = Protocol A, ring = Protocol B.

Inputs: per-image CSVs results/<folder>_<dataset>_metrics.csv, results/metrics_summary.csv, results/complexity.csv.

Usage: python scripts/make_figures.py [--only qualitative bars scatter] [--protocol A|B]
"""
import argparse
import csv

from common import DATASETS, ROOT  # first: sets OPENBLAS_NUM_THREADS before numpy/matplotlib load

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

MODELS = ["SCI", "PairLIE", "URetinex-Net", "Retinexformer", "RUAS"]
PROTOCOL_A_ONLY = {"URetinex-Net"}  # no official training code
COLORS = {"SCI": "#2a78d6", "PairLIE": "#eb6834", "URetinex-Net": "#1baf7a", "Retinexformer": "#eda100",
          "RUAS": "#e87ba4"}  # categorical slots 1-5 in fixed order; colour follows the model, never its rank
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
FIG = ROOT / "figures"


def read_csv(path):
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def folder(model, protocol):
    return model if protocol == "A" else f"{model}_B"


def models_for(protocol):
    return [m for m in MODELS if protocol == "A" or m not in PROTOCOL_A_ONLY]


def style(ax, grid_axis="y"):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- 1. qualitative grids
def per_image(dataset, protocol):
    """{image name: {model: {psnr, ssim, lpips}}} from the per-image CSVs."""
    table = {}
    for m in models_for(protocol):
        for r in read_csv(ROOT / "results" / f"{folder(m, protocol)}_{dataset}_metrics.csv"):
            if r["name"] != "MEAN":
                table.setdefault(r["name"], {})[m] = {k: float(r[k]) for k in ("psnr", "ssim", "lpips")}
    return {n: v for n, v in table.items() if len(v) == len(models_for(protocol))}


def pick_images(dataset, scores, n=3):
    spread = {name: max(v["psnr"] for v in s.values()) - min(v["psnr"] for v in s.values()) for name, s in scores.items()}
    ranked = sorted(spread, key=spread.get, reverse=True)
    if dataset != "LSRW":
        return ranked[:n]
    chosen = [next(x for x in ranked if x.startswith("huawei_")), next(x for x in ranked if x.startswith("nikon_"))]
    return chosen + [x for x in ranked if x not in chosen][: n - len(chosen)]


def qualitative(dataset, protocol):
    scores = per_image(dataset, protocol)
    if not scores:
        print(f"skip qualitative {dataset}: no complete per-image results for protocol {protocol}")
        return
    names = pick_images(dataset, scores)
    models = models_for(protocol)
    cols = ["Input"] + models + ["Ground truth"]
    low_dir, gt_dir = DATASETS[dataset]["low"], DATASETS[dataset]["high"]
    fig, axes = plt.subplots(len(names), len(cols), figsize=(2.3 * len(cols), 1.95 * len(names) + 0.4))
    fig.patch.set_facecolor(SURFACE)
    for i, name in enumerate(names):
        stem = name.rsplit(".", 1)[0]
        paths = [low_dir / name] + [ROOT / "results" / folder(m, protocol) / dataset / f"{stem}.png" for m in models] + [gt_dir / name]
        for j, (col, p) in enumerate(zip(cols, paths)):
            ax = axes[i, j]
            img = Image.open(p).convert("RGB")
            img.thumbnail((480, 480))
            ax.imshow(np.asarray(img))
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if i == 0:
                ax.set_title(col + (" (B)" if protocol == "B" and col in models else ""), fontsize=10, color=INK)
            if col in models:
                s = scores[name][col]
                ax.set_xlabel(f"{s['psnr']:.2f} dB / LPIPS {s['lpips']:.3f}", fontsize=8, color=INK_2, labelpad=2)
            elif j == 0:
                ax.set_ylabel(stem, fontsize=8, color=INK_2)
    fig.suptitle(f"{dataset} - Protocol {protocol}: images with the largest PSNR spread between models "
                 f"(PSNR / LPIPS under each result)", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = FIG / (f"qualitative_{dataset}.png" if protocol == "A" else f"qualitative_{dataset}_B.png")
    fig.savefig(out, dpi=300, facecolor=SURFACE)
    plt.close(fig)
    print("written:", out.relative_to(ROOT), "| images:", ", ".join(names))


# ---------------------------------------------------------------- 2. grouped bars
def bars():
    summary = {(r["model"], r["dataset"]): r for r in read_csv(ROOT / "results" / "metrics_summary.csv") if r["subset"] == "all"}
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), sharey="row")
    fig.patch.set_facecolor(SURFACE)
    for c, protocol in enumerate(("A", "B")):
        models = models_for(protocol)
        width = 0.8 / len(models)
        for r, (metric, label) in enumerate((("psnr", "PSNR (dB) ↑"), ("ssim", "SSIM ↑"))):
            ax = axes[r, c]
            style(ax)
            for g, ds in enumerate(DATASETS):
                for k, m in enumerate(models):
                    x = g + (k - (len(models) - 1) / 2) * width
                    row = summary.get((folder(m, protocol), ds))
                    if row is None:
                        ax.text(x, 0, "pending", rotation=90, fontsize=7, color=INK_2, ha="center", va="bottom")
                        continue
                    ax.bar(x, float(row[metric]), width, color=COLORS[m], edgecolor=SURFACE, linewidth=2)
            ax.set_xticks(range(len(DATASETS)))
            ax.set_xticklabels(DATASETS, fontsize=10, color=INK)
            ax.set_ylim(bottom=0)
            if c == 0:
                ax.set_ylabel(label, fontsize=10)
            if r == 0:
                ax.set_title("Protocol A - official weights" if protocol == "A" else "Protocol B - retrained (LOL-v1 train)",
                             fontsize=11, color=INK, loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[m]) for m in MODELS]
    fig.legend(handles, MODELS, loc="upper center", ncol=len(MODELS), frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.02))
    fig.text(0.01, -0.01, "URetinex-Net is evaluated in Protocol A only (no official training code).",
             fontsize=8, color=INK_2)
    fig.tight_layout(rect=(0, 0.01, 1, 0.96))
    out = FIG / "psnr_ssim_bar.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("written:", out.relative_to(ROOT))


# ---------------------------------------------------------------- 3. quality vs speed
def scatter():
    summary = {(r["model"], r["dataset"]): r for r in read_csv(ROOT / "results" / "metrics_summary.csv") if r["subset"] == "all"}
    complexity = {r["model"]: r for r in read_csv(ROOT / "results" / "complexity.csv")}
    if not complexity:
        print("skip quality_vs_speed: results/complexity.csv missing (run scripts/measure_complexity.py)")
        return
    max_p = max(int(r["params"]) for r in complexity.values())
    area = lambda p: 40 + 900 * np.sqrt(p / max_p)  # area grows with sqrt(params) so the 258-parameter SCI stays visible
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    fig.patch.set_facecolor(SURFACE)
    style(ax, grid_axis="both")
    for m in MODELS:
        c = complexity[m]
        t, p = float(c["runtime_ms_LSRW"]), int(c["params"])
        a = summary.get((m, "LSRW"))
        if a:
            ax.scatter(t, float(a["psnr"]), s=area(p), color=COLORS[m], edgecolor=SURFACE, linewidth=2, zorder=3)
            ptxt = f"{p:,} params" if p < 10000 else f"{p / 1e6:.2f} M params"
            ax.annotate(f"{m}\n{ptxt}", (t, float(a["psnr"])), xytext=(12, 6), textcoords="offset points",
                        fontsize=9, color=INK)
        b = summary.get((f"{m}_B", "LSRW")) if m in models_for("B") else None
        if b:
            ax.scatter(t, float(b["psnr"]), s=area(p), facecolor="none", edgecolor=COLORS[m], linewidth=2, zorder=3)
    ax.set_xscale("log")
    ax.set_xticks([5, 10, 20, 50, 100, 200, 400])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("Runtime per image on LSRW (ms, log scale) - RTX 3060, batch 1", fontsize=10)
    ax.set_ylabel("LSRW PSNR (dB) ↑", fontsize=10)
    ax.set_title("Quality vs speed on LSRW (marker area grows with parameter count)", fontsize=11, color=INK, loc="left")
    ax.scatter([], [], s=120, color=INK_2, label="Protocol A (official)")
    ax.scatter([], [], s=120, facecolor="none", edgecolor=INK_2, linewidth=2, label="Protocol B (retrained)")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.margins(x=0.25, y=0.2)
    out = FIG / "quality_vs_speed.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("written:", out.relative_to(ROOT))


# ---------------------------------------------------------------- 4. training curves, 2x2
def curves_panel():
    """Combine the four Protocol B training curves (made by scripts/plot_train_curve.py) into one 2x2 image."""
    names = ["SCI", "PairLIE", "Retinexformer", "RUAS"]
    paths = [FIG / f"train_curve_{m}.png" for m in names]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        print("skip train_curves_protocolB: missing", missing)
        return
    ims = [Image.open(p).convert("RGB") for p in paths]
    w, h = max(i.width for i in ims), max(i.height for i in ims)
    pad = 40
    canvas = Image.new("RGB", (2 * w + 3 * pad, 2 * h + 3 * pad), SURFACE)
    for k, im in enumerate(ims):
        r, c = divmod(k, 2)
        canvas.paste(im, (pad + c * (w + pad), pad + r * (h + pad)))
    out = FIG / "train_curves_protocolB.png"
    canvas.save(out, dpi=(300, 300))
    print("written:", out.relative_to(ROOT), f"({canvas.width}x{canvas.height})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", default=["qualitative", "bars", "scatter", "curves"],
                    choices=["qualitative", "bars", "scatter", "curves"])
    ap.add_argument("--protocol", default="A", choices=["A", "B"], help="protocol for the qualitative grids")
    args = ap.parse_args()
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": INK_2})
    FIG.mkdir(exist_ok=True)
    if "qualitative" in args.only:
        for ds in DATASETS:
            qualitative(ds, args.protocol)
    if "bars" in args.only:
        bars()
    if "scatter" in args.only:
        scatter()
    if "curves" in args.only:
        curves_panel()


if __name__ == "__main__":
    main()

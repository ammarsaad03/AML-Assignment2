"""Final results table: Protocol A and Protocol B, both datasets, plus complexity.

Inputs : results/metrics_summary.csv (from scripts/evaluate_all.py; result folders <model> = Protocol A,
         <model>_B = Protocol B) and results/complexity.csv (from scripts/measure_complexity.py).
Outputs: results/results_table.csv (all numbers) and results/results_table.md (one table per protocol, best value of
         each column in bold; plus an LSRW Huawei/Nikon table).
URetinex-Net has no official training code, so it appears in Protocol A only (5 models in A, 4 in B).
"PSNR drop" = PSNR(LOL-v1) - PSNR(LSRW): positive = worse on the unseen dataset; smaller is better.

Usage: python scripts/make_results_table.py
"""
import csv

from common import ROOT

MODELS = ["SCI", "PairLIE", "URetinex-Net", "Retinexformer", "RUAS"]
PROTOCOL_A_ONLY = {"URetinex-Net"}  # no official training code
DATASETS = ["LOL-v1", "LSRW"]
METRICS = ["psnr", "ssim", "lpips"]
# column -> (header, format, better)
COLS = {
    "LOL-v1_psnr": ("LOL-v1 PSNR ↑", "{:.2f}", "max"), "LOL-v1_ssim": ("LOL-v1 SSIM ↑", "{:.4f}", "max"),
    "LOL-v1_lpips": ("LOL-v1 LPIPS ↓", "{:.4f}", "min"), "LSRW_psnr": ("LSRW PSNR ↑", "{:.2f}", "max"),
    "LSRW_ssim": ("LSRW SSIM ↑", "{:.4f}", "max"), "LSRW_lpips": ("LSRW LPIPS ↓", "{:.4f}", "min"),
    "params_M": ("Params (M) ↓", "{:.4f}", "min"), "gmacs_256": ("FLOPs@256 (G) ↓", "{:.3f}", "min"),
    "runtime_ms_LOL-v1": ("Runtime LOL-v1 (ms) ↓", "{:.1f}", "min"), "runtime_ms_LSRW": ("Runtime LSRW (ms) ↓", "{:.1f}", "min"),
    "psnr_drop": ("PSNR drop LOL-v1→LSRW (dB) ↓", "{:+.2f}", "min"),
}


def read_csv(path):
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    summary = read_csv(ROOT / "results" / "metrics_summary.csv")
    metrics = {(r["model"], r["dataset"], r["subset"]): r for r in summary}
    complexity = {r["model"]: r for r in read_csv(ROOT / "results" / "complexity.csv")}

    rows = []
    for protocol in ("A", "B"):
        for model in MODELS:
            if protocol == "B" and model in PROTOCOL_A_ONLY:
                continue
            folder = model if protocol == "A" else f"{model}_B"
            weights = "official" if protocol == "A" else "retrained (ours)"
            row ={"protocol": protocol, "model": model, "weights": weights, "result_folder": folder}
            for ds in DATASETS:
                m = metrics.get((folder, ds, "all"))
                for k in METRICS:
                    row[f"{ds}_{k}"] = float(m[k]) if m else None
                for sub in ("huawei", "nikon"):
                    s = metrics.get((folder, "LSRW", sub))
                    row[f"LSRW_{sub}_psnr"] = float(s["psnr"]) if s else None
            c = complexity.get(model)  # same architecture in both protocols
            row["params_M"] = int(c["params"]) / 1e6 if c else None
            row["gmacs_256"] = float(c["gmacs_256"]) if c else None
            for ds in DATASETS:
                row[f"runtime_ms_{ds}"] = float(c[f"runtime_ms_{ds}"]) if c else None
            lol, lsrw = row["LOL-v1_psnr"], row["LSRW_psnr"]
            row["psnr_drop"] = lol - lsrw if lol is not None and lsrw is not None else None
            rows.append(row)

    # CSV (all numbers, no formatting)
    fields = ["protocol", "model", "weights", "result_folder"] + list(COLS) + ["LSRW_huawei_psnr", "LSRW_nikon_psnr"]
    with open(ROOT / "results" / "results_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r[k]) for k in fields})

    # Markdown: one table per protocol, best value per column in bold (within that protocol)
    md = ["# Results", "",
          "Test sets: LOL-v1 eval15 (15 pairs), LSRW Eval (50 pairs). Same preprocessing and metric code for every model",
          "(`eval/metrics.py`; PSNR/SSIM on RGB in [0, 1], LPIPS AlexNet); no GT at inference, no GT-mean, no test-set tuning.",
          "Best value per column (within each protocol) in **bold**. FLOPs: fvcore, 1x3x256x256, multiply-adds.",
          "Runtime: RTX 3060, batch 1, mean ms per image after warm-up. PSNR drop = PSNR(LOL-v1) − PSNR(LSRW).", ""]
    for protocol, title in (("A", "Protocol A - official weights"), ("B", "Protocol B - retrained on LOL-v1 train (450 pairs), one protocol")):
        part = [r for r in rows if r["protocol"] == protocol]
        best = {}
        for k, (_, _, better) in COLS.items():
            vals = [r[k] for r in part if r[k] is not None]
            best[k] = (max(vals) if better == "max" else min(vals)) if vals else None
        md += [f"## {title}", "", "| Model | Weights | " + " | ".join(h for h, _, _ in COLS.values()) + " |",
               "|---|---|" + "---|" * len(COLS)]
        for r in part:
            cells = []
            for k, (_, fmt, _) in COLS.items():
                if r[k] is None:
                    cells.append("pending")
                else:
                    s = fmt.format(r[k])
                    cells.append(f"**{s}**" if best[k] is not None and fmt.format(best[k]) == s else s)
            md.append(f"| {r['model']} | {r['weights']} | " + " | ".join(cells) + " |")
        md.append("")
    md += ["## LSRW by camera (PSNR, dB)", "", "| Model | A: Huawei (30) | A: Nikon (20) | B: Huawei (30) | B: Nikon (20) |",
           "|---|---|---|---|---|"]
    fmt = lambda v: "pending" if v is None else f"{v:.2f}"
    for model in MODELS:
        a = next(r for r in rows if r["protocol"] == "A" and r["model"] == model)
        b = next((r for r in rows if r["protocol"] == "B" and r["model"] == model), None)
        bh, bn = ("n/a", "n/a") if b is None else (fmt(b["LSRW_huawei_psnr"]), fmt(b["LSRW_nikon_psnr"]))
        md.append(f"| {model} | {fmt(a['LSRW_huawei_psnr'])} | {fmt(a['LSRW_nikon_psnr'])} | {bh} | {bn} |")
    md += ["", "URetinex-Net: no official training code, so it is evaluated in Protocol A only (see docs/protocol.md)."]
    (ROOT / "results" / "results_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("written: results/results_table.csv, results/results_table.md")
    print("\n".join(md))


if __name__ == "__main__":
    main()

"""Verify the test datasets: counts, low/high pairing, image sizes, brightness sanity check.

Usage: python scripts/check_data.py
"""
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "LOL-v1": ROOT / "data" / "LOL-v1" / "eval15",
    "LSRW": ROOT / "data" / "LSRW" / "eval",
}
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp"}


def list_images(folder):
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXT)


def check(name, root):
    low, high = list_images(root / "low"), list_images(root / "high")
    print(f"=== {name} ({root.relative_to(ROOT)})")
    print(f"  low : {len(low)} images | high: {len(high)} images")

    low_names, high_names = {p.name for p in low}, {p.name for p in high}
    only_low, only_high = sorted(low_names - high_names), sorted(high_names - low_names)
    print(f"  pairs matched: {len(low_names & high_names)} | only in low: {only_low} | only in high: {only_high}")

    sizes, problems, darker = Counter(), [], 0
    for lp in low:
        hp = root / "high" / lp.name
        if not hp.exists():
            continue
        li, hi = Image.open(lp), Image.open(hp)
        if li.size != hi.size:
            problems.append(f"{lp.name}: low {li.size} vs high {hi.size}")
        if li.mode != "RGB" or hi.mode != "RGB":
            problems.append(f"{lp.name}: modes low={li.mode} high={hi.mode}")
        sizes[li.size] += 1
        if np.asarray(li.convert("RGB")).mean() < np.asarray(hi.convert("RGB")).mean():
            darker += 1

    print("  sizes (W x H): " + ", ".join(f"{w}x{h} ({n})" for (w, h), n in sizes.most_common()))
    print(f"  low darker than high: {darker}/{len(low)}")
    print(f"  problems: {problems if problems else 'none'}")
    if name == "LSRW":
        print(f"  subsets: {dict(Counter(p.name.split('_')[0] for p in low))}")
    return not (only_low or only_high or problems)


if __name__ == "__main__":
    ok = all(check(n, r) for n, r in DATASETS.items())
    print("\nRESULT:", "ALL OK" if ok else "PROBLEMS FOUND")

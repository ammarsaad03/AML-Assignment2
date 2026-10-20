"""Protocol B data preparation.

1. Copy the LOL-v1 training set (485 pairs) from the extracted archive to data/LOL-v1/train/{low,high}.
2. Hold out a fixed random validation subset (35 pairs, seed 42) -> configs/val_split.txt; the rest -> configs/train_split.txt.
3. Also write basicsr meta-info versions (name + shape per line) for Retinexformer: configs/protocol_B/*_meta.txt.
4. Check by SHA-256 of decoded pixels that no training image equals a test image (LOL-v1 eval15, LSRW eval).

Usage: python scripts/prepare_protocol_b.py
"""
import hashlib
import random
import shutil

import numpy as np
from PIL import Image

from common import DATASETS, ROOT, list_images

SRC = ROOT / "data" / "_downloads" / "LOLv1_extracted" / "LOLv1" / "Train"
DST = ROOT / "data" / "LOL-v1" / "train"
N_VAL, SEED = 35, 42


def pixel_hash(path):
    return hashlib.sha256(np.asarray(Image.open(path).convert("RGB")).tobytes()).hexdigest()


def main():
    for src_sub, dst_sub in (("input", "low"), ("target", "high")):
        (DST / dst_sub).mkdir(parents=True, exist_ok=True)
        for p in list_images(SRC / src_sub):
            shutil.copy2(p, DST / dst_sub / p.name)
    low, high = list_images(DST / "low"), list_images(DST / "high")
    names = sorted(p.name for p in low)
    assert names == sorted(p.name for p in high) and len(names) == 485, (len(names), len(high))

    val = sorted(random.Random(SEED).sample(names, N_VAL), key=lambda n: int(n.split(".")[0]))
    train = sorted(set(names) - set(val), key=lambda n: int(n.split(".")[0]))
    cfg = ROOT / "configs"
    (cfg / "val_split.txt").write_text("\n".join(val) + "\n", encoding="utf-8")
    (cfg / "train_split.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
    (cfg / "protocol_B").mkdir(exist_ok=True)
    for split, lst in (("train", train), ("val", val)):
        lines = []
        for n in lst:
            w, h = Image.open(DST / "high" / n).size
            lines.append(f"{n} ({h},{w},3)")
        (cfg / "protocol_B" / f"{split}_meta.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"train: {len(train)} | val: {len(val)} (seed {SEED}) | val = {', '.join(val)}")

    train_hashes = {pixel_hash(p): p.name for p in low + high}
    for ds, d in DATASETS.items():
        clash = [(p.name, train_hashes[h]) for p in list_images(d["low"]) + list_images(d["high"])
                 if (h := pixel_hash(p)) in train_hashes]
        print(f"overlap LOL-v1 train vs {ds} test: {clash if clash else 'none'}")


if __name__ == "__main__":
    main()

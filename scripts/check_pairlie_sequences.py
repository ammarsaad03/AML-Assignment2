"""Identify LOL-derived sequences in the PairLIE authors' training set and check overlap with our test sets.

Each image is identified by SHA-256 of its decoded RGB pixels. A sequence is "LOL-derived" if every image in it is
found among the LOL-v1 training LOW images. Any sequence image that matches a test image (LOL-v1 eval15 or LSRW eval,
low or high) is reported and must be removed.

Writes configs/protocol_B/pairlie_lol_sequences.txt (one line per kept sequence: seq_id: lol_names...).

Usage: python scripts/check_pairlie_sequences.py
"""
import hashlib
from collections import Counter

import numpy as np
from PIL import Image

from common import DATASETS, ROOT, list_images

SEQ_DIR = ROOT / "data" / "_downloads" / "PairLIE" / "extracted" / "PairLIE" / "PairLIE-training-dataset"
LOL_TRAIN = ROOT / "data" / "LOL-v1" / "train"


def pixel_hash(path):
    return hashlib.sha256(np.asarray(Image.open(path).convert("RGB")).tobytes()).hexdigest()


def main():
    lol_low = {pixel_hash(p): p.name for p in list_images(LOL_TRAIN / "low")}
    lol_high = {pixel_hash(p): p.name for p in list_images(LOL_TRAIN / "high")}
    test = {}
    for ds, d in DATASETS.items():
        for kind in ("low", "high"):
            for p in list_images(d[kind]):
                test[pixel_hash(p)] = f"{ds}/{kind}/{p.name}"

    kinds, kept, test_hits, partial = Counter(), [], [], []
    for seq in sorted(SEQ_DIR.iterdir(), key=lambda d: int(d.name)):
        imgs = list_images(seq)
        hashes = [pixel_hash(p) for p in imgs]
        test_hits += [(seq.name, p.name, test[h]) for p, h in zip(imgs, hashes) if h in test]
        in_low = [h in lol_low for h in hashes]
        if any(h in lol_high for h in hashes):
            kinds["contains LOL normal-light image"] += 1
        if all(in_low):
            kinds["all images = LOL-v1 train low"] += 1
            kept.append((seq.name, [lol_low[h] for h in hashes]))
        elif any(in_low):
            kinds["mixed"] += 1
            partial.append(seq.name)
        else:
            kinds["no LOL image (e.g. SICE)"] += 1

    print("sequence types:", dict(kinds))
    print(f"LOL-derived sequences: {len(kept)} | images: {sum(len(n) for _, n in kept)} | "
          f"sizes: {dict(sorted(Counter(len(n) for _, n in kept).items()))}")
    print("mixed sequences:", partial if partial else "none")
    print("matches with test images:", test_hits if test_hits else "none")
    lol_names = [n for _, ns in kept for n in ns]
    print(f"distinct LOL-v1 train low images used: {len(set(lol_names))} of {len(lol_low)}")

    out = ROOT / "configs" / "protocol_B" / "pairlie_lol_sequences.txt"
    out.write_text("".join(f"{s}: {' '.join(ns)}\n" for s, ns in kept), encoding="utf-8")
    print("written:", out.relative_to(ROOT))


if __name__ == "__main__":
    main()

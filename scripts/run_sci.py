"""Run the official SCI (CVPR 2022) pretrained model on a test set.

Uses the official model code from models/SCI/CVPR (Finetunemodel: one illumination-estimation block,
output r = clamp(input / illumination, 0, 1)). No repo files are modified.

Usage:
    python scripts/run_sci.py --dataset LOL-v1
    python scripts/run_sci.py --dataset LSRW --weights medium
    python scripts/run_sci.py --dataset LOL-v1 --protocol B     # retrained checkpoint (checkpoints/SCI/final.pt)
"""
import argparse
import sys

import torch

from common import DATASETS, ROOT, CudaTimer, get_logger, list_images, load_rgb01, log_environment, save_rgb01, sha256

SCI_DIR = ROOT / "models" / "SCI" / "CVPR"
sys.path.insert(0, str(SCI_DIR))
from model import Finetunemodel  # noqa: E402  (official SCI code)


def build_model(weights_path, device):
    """Official Finetunemodel, plus a strict check: SCI's own loader silently skips keys it cannot match."""
    model = Finetunemodel(str(weights_path))
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    expected = {k[len("enhance."):]: v for k, v in ckpt.items() if k.startswith("enhance.")}
    loaded = model.enhance.state_dict()
    if expected.keys() != loaded.keys() or not all(torch.equal(expected[k].cpu(), loaded[k].cpu()) for k in loaded):
        raise RuntimeError(f"SCI weights not fully loaded from {weights_path}")
    return model.to(device).eval(), len(expected), len(ckpt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--weights", default="medium", choices=["easy", "medium", "difficult"])
    ap.add_argument("--protocol", default="A", choices=["A", "B"],
                    help="A: official weights (--weights), B: our retrained checkpoints/SCI/final.pt")
    args = ap.parse_args()

    device = "cuda"
    if args.protocol == "A":
        weights_path, name = SCI_DIR / "weights" / f"{args.weights}.pt", "SCI"
    else:
        weights_path, name = ROOT / "checkpoints" / "SCI" / "final.pt", "SCI_B"
    out_dir = ROOT / "results" / name / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    log = get_logger(ROOT / "logs" / f"{name}_{args.dataset}.log")

    log_environment(log)
    model, n_loaded, n_ckpt = build_model(weights_path, device)
    n_params = sum(p.numel() for p in model.enhance.parameters())
    log.info(f"weights: {weights_path.relative_to(ROOT)} | sha256 {sha256(weights_path)}")
    log.info(f"loaded {n_loaded}/{n_ckpt} checkpoint tensors (enhance.* only, strict) | params {n_params}")

    images = list_images(DATASETS[args.dataset]["low"])
    log.info(f"dataset {args.dataset}: {len(images)} images -> {out_dir.relative_to(ROOT)}")

    times = []
    with torch.no_grad():
        for p in images:
            x = load_rgb01(p, device)
            with CudaTimer() as t:
                _, r = model(x)
            times.append(t.seconds)
            save_rgb01(r, out_dir / f"{p.stem}.png")
            log.info(f"{p.name}: {x.shape[3]}x{x.shape[2]} | {t.seconds * 1000:.1f} ms")

    steady = times[1:] if len(times) > 1 else times  # first image includes CUDA warm-up
    log.info(f"done: {len(images)} images | mean time {1000 * sum(steady) / len(steady):.2f} ms/img (excluding first)")


if __name__ == "__main__":
    main()

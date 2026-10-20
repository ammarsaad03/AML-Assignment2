"""Run the official RUAS (CVPR 2021) model on a test set.

Uses the official Network from models/RUAS/model.py (3 illumination-estimation stages + noise-removal module).
For the LOL model the official test.py saves u_list[-1] (the output after noise removal); we do the same.
Network() loads ./model/denoise.pt relative to the working directory, so the model is built with the repo as the
working directory; the checkpoint is then loaded strictly. No repo files are modified.

Usage:
    python scripts/run_ruas.py --dataset LOL-v1                  # Protocol A (official ckpt/lol.pt)
    python scripts/run_ruas.py --dataset LOL-v1 --protocol B     # Protocol B (checkpoints/RUAS/final.pt)
    python scripts/run_ruas.py --dataset LOL-v1 --protocol B_long  # supplementary 50k-iteration run
"""
import argparse
import contextlib
import os
import sys

import torch

from common import DATASETS, ROOT, CudaTimer, get_logger, list_images, load_rgb01, log_environment, save_rgb01, sha256

RUAS_DIR = ROOT / "models" / "RUAS"
sys.path.insert(0, str(RUAS_DIR))
from model import Network  # noqa: E402  (official RUAS code)


@contextlib.contextmanager
def in_repo_dir():
    old = os.getcwd()
    os.chdir(RUAS_DIR)
    try:
        yield
    finally:
        os.chdir(old)


def build_model(weights_path, device):
    with in_repo_dir():
        model = Network()  # also creates the official optimizers and loads ./model/denoise.pt (overwritten below)
    state = torch.load(weights_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=True)  # raises if any key is missing/unexpected
    return model.to(device).eval(), len(state)


def enhance(model, x):
    u_list, _ = model(x)
    return u_list[-1]  # official test.py output for the LOL model (after noise removal)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--protocol", default="A", choices=["A", "B", "B_long"],
                    help="A: official ckpt/lol.pt, B: checkpoints/RUAS/final.pt, B_long: checkpoints/RUAS_long/final.pt")
    args = ap.parse_args()

    device = "cuda"
    if args.protocol == "A":
        weights_path, name = RUAS_DIR / "ckpt" / "lol.pt", "RUAS"
    elif args.protocol == "B":
        weights_path, name = ROOT / "checkpoints" / "RUAS" / "final.pt", "RUAS_B"
    else:  # supplementary 50,000-iteration run
        weights_path, name = ROOT / "checkpoints" / "RUAS_long" / "final.pt", "RUAS_B_long"
    out_dir = ROOT / "results" / name / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    log = get_logger(ROOT / "logs" / f"{name}_{args.dataset}.log")

    log_environment(log)
    model, n_loaded = build_model(weights_path, device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"weights: {weights_path.relative_to(ROOT)} | sha256 {sha256(weights_path)}")
    log.info(f"loaded {n_loaded} tensors (strict) | params {n_params} | output u_list[-1] (after noise removal)")

    images = list_images(DATASETS[args.dataset]["low"])
    log.info(f"dataset {args.dataset}: {len(images)} images -> {out_dir.relative_to(ROOT)}")

    times = []
    with torch.no_grad():
        for p in images:
            x = load_rgb01(p, device)
            with CudaTimer() as t:
                out = enhance(model, x)
            times.append(t.seconds)
            save_rgb01(out, out_dir / f"{p.stem}.png")
            log.info(f"{p.name}: {x.shape[3]}x{x.shape[2]} | {t.seconds * 1000:.1f} ms")

    steady = times[1:] if len(times) > 1 else times  # first image includes CUDA warm-up
    log.info(f"done: {len(images)} images | mean time {1000 * sum(steady) / len(steady):.2f} ms/img (excluding first)")


if __name__ == "__main__":
    main()

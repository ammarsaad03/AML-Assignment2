"""Run the official PairLIE (CVPR 2023) pretrained model on a test set.

Uses the official network from models/PairLIE/net/net.py and the official test-time formula from eval.py:
    L, R, X = net(input);  I = L ** 0.2 * R
The exponent stays at the default 0.2 for every dataset (no tuning on test data; see docs/protocol.md).
No repo files are modified.

Usage:
    python scripts/run_pairlie.py --dataset LOL-v1                       # Protocol A (official weights)
    python scripts/run_pairlie.py --dataset LOL-v1 --protocol B          # Protocol B (checkpoints/PairLIE/final.pth)
"""
import argparse
import sys

import torch

from common import DATASETS, ROOT, CudaTimer, get_logger, list_images, load_rgb01, log_environment, save_rgb01, sha256

PAIRLIE_DIR = ROOT / "models" / "PairLIE"
sys.path.insert(0, str(PAIRLIE_DIR))
from net.net import net  # noqa: E402  (official PairLIE code)

EXPONENT = 0.2  # official default in eval.py


def build_model(weights_path, device):
    model = net()
    state = torch.load(weights_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=True)  # raises if any key is missing/unexpected
    return model.to(device).eval(), len(state)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--protocol", default="A", choices=["A", "B"], help="A: official weights, B: our retrained final.pth")
    args = ap.parse_args()

    device = "cuda"
    if args.protocol == "A":
        weights_path, name = PAIRLIE_DIR / "weights" / "PairLIE.pth", "PairLIE"
    else:
        weights_path, name = ROOT / "checkpoints" / "PairLIE" / "final.pth", "PairLIE_B"
    out_dir = ROOT / "results" / name / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    log = get_logger(ROOT / "logs" / f"{name}_{args.dataset}.log")

    log_environment(log)
    model, n_loaded = build_model(weights_path, device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"weights: {weights_path.relative_to(ROOT)} | sha256 {sha256(weights_path)}")
    log.info(f"loaded {n_loaded} tensors (strict) | params {n_params} | exponent {EXPONENT}")

    images = list_images(DATASETS[args.dataset]["low"])
    log.info(f"dataset {args.dataset}: {len(images)} images -> {out_dir.relative_to(ROOT)}")

    times = []
    with torch.no_grad():
        for p in images:
            x = load_rgb01(p, device)
            with CudaTimer() as t:
                L, R, _ = model(x)
                out = torch.pow(L, EXPONENT) * R
            times.append(t.seconds)
            save_rgb01(out, out_dir / f"{p.stem}.png")
            log.info(f"{p.name}: {x.shape[3]}x{x.shape[2]} | {t.seconds * 1000:.1f} ms")

    steady = times[1:] if len(times) > 1 else times  # first image includes CUDA warm-up
    log.info(f"done: {len(images)} images | mean time {1000 * sum(steady) / len(steady):.2f} ms/img (excluding first)")


if __name__ == "__main__":
    main()

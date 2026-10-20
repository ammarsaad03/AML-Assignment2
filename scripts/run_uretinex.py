"""Run the official URetinex-Net (CVPR 2022) pretrained model on a test set.

Uses the official `Inference` class from models/URetinex-Net/test.py (decomposition -> unfolding -> illumination
adjustment with the fixed default ratio 5). The official evaluate.py is NOT used, because it computes the ratio from
the ground-truth image (see docs/protocol.md). No repo files are modified.

The checkpoints store an argparse.Namespace ("opts"), which PyTorch >= 2.6 refuses to load with its default
weights_only=True, so torch.load is called with weights_only=False while the model is built.

Usage:
    python scripts/run_uretinex.py --dataset LOL-v1
"""
import argparse
import contextlib
import functools
import sys
from types import SimpleNamespace

import torch

from common import DATASETS, ROOT, CudaTimer, get_logger, list_images, load_rgb01, log_environment, save_rgb01, sha256

URET_DIR = ROOT / "models" / "URetinex-Net"
sys.path.insert(0, str(URET_DIR))
from test import Inference  # noqa: E402  (official URetinex-Net code)

RATIO = 5  # official default in test.py
CKPT = {
    "Decom_model_low_path": URET_DIR / "ckpt" / "init_low.pth",
    "unfolding_model_path": URET_DIR / "ckpt" / "unfolding.pth",
    "adjust_model_path": URET_DIR / "ckpt" / "L_adjust.pth",
}


@contextlib.contextmanager
def torch_load_weights_only_false():
    original = torch.load
    torch.load = functools.partial(original, weights_only=False)
    try:
        yield
    finally:
        torch.load = original


def build_model(device):
    opts = SimpleNamespace(ratio=RATIO, **{k: str(v) for k, v in CKPT.items()})
    with torch_load_weights_only_false(), contextlib.redirect_stdout(None):  # Inference prints the full networks
        model = Inference(opts)
    return model.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    args = ap.parse_args()

    device = "cuda"
    out_dir = ROOT / "results" / "URetinex-Net" / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    log = get_logger(ROOT / "logs" / f"URetinex-Net_{args.dataset}.log")

    log_environment(log)
    model = build_model(device)
    for name, path in CKPT.items():
        log.info(f"weights ({name}): {path.relative_to(ROOT)} | sha256 {sha256(path)}")
    parts = {"decom_low": model.model_Decom_low, "unfolding_R": model.model_R, "unfolding_L": model.model_L,
             "adjust": model.adjust_model}
    n_params = {k: sum(p.numel() for p in m.parameters()) for k, m in parts.items()}
    log.info(f"params used at test: {n_params} | total {sum(n_params.values())}")
    u = model.unfolding_opts
    log.info(f"unfolding opts from checkpoint: round={u.round} gamma={u.gamma} lamda={u.lamda} "
             f"Roffset={u.Roffset} Loffset={u.Loffset} | ratio={RATIO} (fixed)")

    images = list_images(DATASETS[args.dataset]["low"])
    log.info(f"dataset {args.dataset}: {len(images)} images -> {out_dir.relative_to(ROOT)}")

    times = []
    with torch.no_grad():
        for p in images:
            x = load_rgb01(p, device)
            with CudaTimer() as t:
                out, _ = model(x)
            times.append(t.seconds)
            save_rgb01(out, out_dir / f"{p.stem}.png")
            log.info(f"{p.name}: {x.shape[3]}x{x.shape[2]} | {t.seconds * 1000:.1f} ms")

    steady = times[1:] if len(times) > 1 else times  # first image includes CUDA warm-up
    log.info(f"done: {len(images)} images | mean time {1000 * sum(steady) / len(steady):.2f} ms/img (excluding first)")


if __name__ == "__main__":
    main()

"""Run the official Retinexformer (ICCV 2023) pretrained model on a test set.

Option A (docs/protocol.md): no basicsr install. The official architecture file
models/Retinexformer/basicsr/models/archs/RetinexFormer_arch.py is loaded directly by path, with the network
arguments read from the official option file Options/RetinexFormer_LOL_v1.yml. Test-time steps follow
Enhancement/test_from_dataset.py: reflect-pad to a multiple of 4, forward, crop back, clamp to [0, 1].
No --GT_mean, no --self_ensemble. No repo files are modified.

Usage:
    python scripts/run_retinexformer.py --dataset LOL-v1                 # Protocol A (official LOL_v1.pth)
    python scripts/run_retinexformer.py --dataset LOL-v1 --protocol B    # Protocol B (best validation checkpoint)
"""
import argparse
import importlib.util

import torch
import torch.nn.functional as F
import yaml

from common import DATASETS, ROOT, CudaTimer, get_logger, list_images, load_rgb01, log_environment, save_rgb01, sha256

RF_DIR = ROOT / "models" / "Retinexformer"
ARCH_FILE = RF_DIR / "basicsr" / "models" / "archs" / "RetinexFormer_arch.py"
OPT_FILE = RF_DIR / "Options" / "RetinexFormer_LOL_v1.yml"
WEIGHTS = ROOT / "weights" / "Retinexformer" / "LOL_v1.pth"
FACTOR = 4  # official padding factor in test_from_dataset.py


def load_arch():
    spec = importlib.util.spec_from_file_location("RetinexFormer_arch", ARCH_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.RetinexFormer


PROTOCOL_B_DIR = ROOT / "checkpoints" / "Retinexformer" / "RetinexFormer_LOL_v1_protocolB"


def protocol_b_weights():
    """The best-validation-PSNR checkpoint saved by basicsr (best_psnr_<psnr>_<iter>.pth; basicsr keeps only one)."""
    found = sorted(PROTOCOL_B_DIR.glob("best_psnr_*.pth"))
    if len(found) != 1:
        raise FileNotFoundError(f"expected exactly one best_psnr_*.pth in {PROTOCOL_B_DIR}, found {len(found)}")
    return found[0]


def build_model(device, weights=WEIGHTS):
    net_opts = dict(yaml.safe_load(open(OPT_FILE, encoding="utf-8"))["network_g"])
    assert net_opts.pop("type") == "RetinexFormer"
    model = load_arch()(**net_opts)
    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["params"], strict=True)  # raises if any key is missing/unexpected
    return model.to(device).eval(), net_opts, len(ckpt["params"])


def enhance(model, x):
    """Official test-time steps: reflect-pad to a multiple of FACTOR, forward, crop back."""
    h, w = x.shape[2], x.shape[3]
    H, W = ((h + FACTOR) // FACTOR) * FACTOR, ((w + FACTOR) // FACTOR) * FACTOR
    padh = H - h if h % FACTOR != 0 else 0
    padw = W - w if w % FACTOR != 0 else 0
    x = F.pad(x, (0, padw, 0, padh), "reflect")
    return model(x)[:, :, :h, :w]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--protocol", default="A", choices=["A", "B"],
                    help="A: official LOL_v1.pth, B: our best-validation checkpoint (best_psnr_*.pth)")
    args = ap.parse_args()

    device = "cuda"
    weights, name = (WEIGHTS, "Retinexformer") if args.protocol == "A" else (protocol_b_weights(), "Retinexformer_B")
    out_dir = ROOT / "results" / name / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    log = get_logger(ROOT / "logs" / f"{name}_{args.dataset}.log")

    log_environment(log)
    model, net_opts, n_loaded = build_model(device, weights)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"weights: {weights.relative_to(ROOT)} | sha256 {sha256(weights)}")
    log.info(f"network (from {OPT_FILE.relative_to(ROOT)}): {net_opts} | loaded {n_loaded} tensors (strict) | params {n_params}")

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

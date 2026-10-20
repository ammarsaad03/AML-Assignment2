"""Protocol B: train RUAS (CVPR 2021) on the LOL-v1 training split (low-light images only).

Follows the official models/RUAS/train.py + Network.optimizer(): authors' searched architecture (genotypes.py),
800 iterations, batch 1, random 320x320 crop, IEM updated every step (SGD lr 0.015, momentum 0.9, wd 3e-4, clip 5),
NRM initialised from the authors' model/denoise.pt and updated every 50th step (SGD lr 0.001), official losses.
Deviations (docs/protocol.md): seed 42; our 450 training images; a RANDOM image at every step (the official script
always used the first image - data-loading bug); the NRM step uses a detached IEM output (PyTorch >= 1.5 in-place
error; the NRM gradients are identical); validation PSNR on the 35 held-out pairs every 50 iterations (curve only;
the FINAL checkpoint is used, RUAS is unsupervised).

Usage:
    python scripts/train_ruas.py                 # full training (800 iterations, a few minutes)
    python scripts/train_ruas.py --resume        # continue from checkpoints/RUAS/last.pt
    python scripts/train_ruas.py --debug 60      # short debug run -> checkpoints/RUAS_debug, logs/train_RUAS_debug.*
    python scripts/train_ruas.py --long          # supplementary run: 50,000 iterations -> checkpoints/RUAS_long
    python scripts/train_ruas.py --long --resume # continue the long run
"""
import argparse
import random
import time

import numpy as np
import torch
import torch.nn as nn

from common import ROOT
from run_ruas import Network, in_repo_dir  # official RUAS Network (repo on sys.path)
from train_common import LOL_TRAIN, SEED, Timer, TrainLog, ValSet, load_uint8, read_split, seed_everything, to_tensor01

ITERS, CROP, CLIP, NRM_EVERY = 800, 320, 5, 50  # official train.py / config.py / model.py
VAL_EVERY, CKPT_EVERY = 50, 100
# Supplementary "RUAS-long" run (decided 2026-10-20, before any result): same settings, 50,000 iterations, final checkpoint
LONG_ITERS, LONG_VAL_EVERY, LONG_CKPT_EVERY = 50000, 500, 1000


def random_crop(img):
    """Official MemoryFriendlyLoader crop (including its '- 1' in the offset range)."""
    h, w = img.shape[:2]
    top = random.randint(0, max(0, h - CROP - 1))
    left = random.randint(0, max(0, w - CROP - 1))
    return img[top:top + CROP, left:left + CROP]


def train_step(model, x, step):
    """Official Network.optimizer(), with the NRM update on a detached IEM output (see module docstring)."""
    u_list, t_list = model(x)
    model.enhancement_optimizer.zero_grad()
    enh_loss = model._criterion(x, u_list, t_list)
    enh_loss.backward()
    nn.utils.clip_grad_norm_(model.enhance_net.parameters(), CLIP)
    model.enhancement_optimizer.step()

    den_loss = None
    if step % NRM_EVERY == 0:
        u_in = u_list[-2].detach()  # IEM output, computed before the IEM step (same value as in the official code)
        model.denoise_optimizer.zero_grad()
        u_d, _ = model.denoise_net(u_in)
        den_loss = model._denoise_criterion(u_d, u_in)
        den_loss.backward()
        nn.utils.clip_grad_norm_(model.denoise_net.parameters(), CLIP)
        model.denoise_optimizer.step()
    return enh_loss.item(), (den_loss.item() if den_loss is not None else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--debug", type=int, default=0, help="run only N iterations into separate debug folders")
    ap.add_argument("--long", action="store_true",
                    help=f"supplementary run with {LONG_ITERS} iterations (fixed in advance) -> checkpoints/RUAS_long")
    args = ap.parse_args()
    global CKPT_EVERY
    if args.debug:
        tag, n_iters, val_every = ("RUAS_long_debug" if args.long else "RUAS_debug"), args.debug, 25
    elif args.long:
        tag, n_iters, val_every, CKPT_EVERY = "RUAS_long", LONG_ITERS, LONG_VAL_EVERY, LONG_CKPT_EVERY
    else:
        tag, n_iters, val_every = "RUAS", ITERS, VAL_EVERY
    ckpt_dir = ROOT / "checkpoints" / tag

    seed_everything(SEED)
    torch.backends.cudnn.benchmark = True  # as in official train.py
    images = [load_uint8(LOL_TRAIN / "low" / n) for n in read_split("train")]
    with in_repo_dir():
        model = Network().cuda()  # official init: NRM <- ./model/denoise.pt; creates both official SGD optimizers
    log = TrainLog(tag, ["iter", "enhance_loss", "denoise_loss", "val_psnr", "lr_iem", "lr_nrm", "elapsed_s"],
                   resume=args.resume)
    log.info(f"RUAS Protocol B{' (supplementary long run)' if args.long else ''} | {len(images)} training images (random image + {CROP}x{CROP} crop per step) | "
             f"{n_iters} iterations | IEM SGD lr 0.015, NRM SGD lr 0.001 every {NRM_EVERY} steps | seed {SEED}")

    step, elapsed0 = 0, 0.0
    if args.resume:
        s = torch.load(ckpt_dir / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(s["model"])
        model.enhancement_optimizer.load_state_dict(s["opt_iem"]); model.denoise_optimizer.load_state_dict(s["opt_nrm"])
        random.setstate(s["py_rng"]); np.random.set_state(s["np_rng"]); torch.set_rng_state(s["torch_rng"])
        step, elapsed0 = s["step"], s["elapsed_s"]
        log.info(f"resumed after iteration {step}")

    val = ValSet()

    def enhance_fn(x):
        u_list, _ = model(x)
        return u_list[-1]

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    t0, enh_window, last_den = time.perf_counter(), [], None
    if step == 0:  # validation of the initial model (IEM random, NRM = denoise.pt)
        model.eval(); v0 = val.mean_psnr(enhance_fn); model.train()
        log.row(iter=0, val_psnr=v0, lr_iem=0.015, lr_nrm=0.001, elapsed_s=0)
        log.info(f"iter 0/{n_iters} | val PSNR {v0:.3f} dB (initial model)")
    while step < n_iters:
        step += 1
        model.train()
        x = to_tensor01(random_crop(random.choice(images))).cuda()
        with Timer():
            enh, den = train_step(model, x, step)
        enh_window.append(enh)
        if den is not None:
            last_den = den
        if step % val_every == 0 or step == n_iters:
            model.eval()
            v = val.mean_psnr(enhance_fn)
            elapsed = elapsed0 + time.perf_counter() - t0
            row = {"iter": step, "enhance_loss": float(np.mean(enh_window)), "denoise_loss": last_den, "val_psnr": v,
                   "lr_iem": model.enhancement_optimizer.param_groups[0]["lr"],
                   "lr_nrm": model.denoise_optimizer.param_groups[0]["lr"], "elapsed_s": round(elapsed, 1)}
            log.row(**row)
            s_it = elapsed / step
            log.info(f"iter {step}/{n_iters} | IEM loss {row['enhance_loss']:.5f} | NRM loss "
                     f"{'-' if last_den is None else f'{last_den:.5f}'} | val PSNR {v:.3f} dB (curve only) | "
                     f"{s_it:.3f} s/it | elapsed {elapsed:.0f} s | ETA {s_it * (n_iters - step):.0f} s")
            enh_window = []
        if step % CKPT_EVERY == 0 or step == n_iters:
            torch.save({"model": model.state_dict(), "opt_iem": model.enhancement_optimizer.state_dict(),
                        "opt_nrm": model.denoise_optimizer.state_dict(), "step": step,
                        "elapsed_s": elapsed0 + time.perf_counter() - t0, "py_rng": random.getstate(),
                        "np_rng": np.random.get_state(), "torch_rng": torch.get_rng_state()}, ckpt_dir / "last.pt")
            log.info(f"checkpoint saved: {ckpt_dir / 'last.pt'} (iteration {step})")
    torch.save(model.state_dict(), ckpt_dir / "final.pt")  # same format as the official ckpt/lol.pt
    log.info(f"TRAINING FINISHED: final checkpoint {ckpt_dir / 'final.pt'}")


if __name__ == "__main__":
    main()

"""Protocol B: train SCI (CVPR 2022) from scratch on the LOL-v1 training split (low-light images only).

Follows the official models/SCI/CVPR/train.py: Network(stage=3) with the official weight init and unsupervised loss,
Adam(lr 3e-4, betas (0.9, 0.999), weight_decay 3e-4), grad-norm clip 5, batch 1, full-resolution images, no
augmentation. Differences: 200 epochs instead of the official 1000 (~34 h on an RTX 3060; see docs/protocol.md),
seed 42, our 450-image training split, images cached in RAM, validation PSNR on the 35 held-out pairs every 5 epochs
(logged for curves only; the FINAL checkpoint is used, as SCI is unsupervised).

Usage:
    python scripts/train_sci.py                    # full training (200 epochs)
    python scripts/train_sci.py --resume           # continue from checkpoints/SCI/last.pt
    python scripts/train_sci.py --speed_test 300   # time 300 iterations, save nothing
    python scripts/train_sci.py --debug 2          # 2-epoch debug run -> checkpoints/SCI_debug, logs/train_SCI_debug.*
"""
import argparse
import random
import sys

import torch
import torch.nn as nn

from common import ROOT
from train_common import (LOL_TRAIN, SEED, Progress, Timer, TrainLog, ValSet, load_uint8, read_split, seed_everything,
                          speed_report, to_tensor01)

sys.path.insert(0, str(ROOT / "models" / "SCI" / "CVPR"))
from model import Network  # noqa: E402  (official SCI code)

OFFICIAL_EPOCHS = 1000
EPOCHS, LR, WD, CLIP, STAGE = 200, 3e-4, 3e-4, 5, 3  # official train.py values, except epochs (reduced, Protocol B)
VAL_EVERY, CKPT_EVERY, PRINT_EVERY = 5, 5, 150


def build():
    model = Network(stage=STAGE)
    for m in (model.enhance.in_conv, model.enhance.conv, model.enhance.out_conv,
              model.calibrate.in_conv, model.calibrate.convs, model.calibrate.out_conv):
        m.apply(model.weights_init)  # official init, as in train.py
    model = model.cuda()
    opt = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=WD)
    return model, opt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--speed_test", type=int, default=0, help="run N iterations, report speed, save nothing")
    ap.add_argument("--debug", type=int, default=0, help="run only N epochs into separate debug folders")
    args = ap.parse_args()
    global VAL_EVERY, CKPT_EVERY, PRINT_EVERY
    tag = "SCI_speedtest" if args.speed_test else "SCI_debug" if args.debug else "SCI"
    ckpt_dir = ROOT / "checkpoints" / tag
    n_epochs = args.debug or EPOCHS
    if args.debug:
        VAL_EVERY, CKPT_EVERY, PRINT_EVERY = 1, 1, 50

    seed_everything(SEED)
    torch.backends.cudnn.benchmark = True  # as in official train.py
    names = read_split("train")
    images = [load_uint8(LOL_TRAIN / "low" / n) for n in names]
    model, opt = build()
    log = TrainLog(tag, ["epoch", "iters", "train_loss", "val_psnr", "lr", "epoch_seconds"], resume=args.resume)
    log.info(f"SCI Protocol B | {len(images)} training images | epochs {n_epochs} (official {OFFICIAL_EPOCHS}) | "
             f"Adam lr {LR} wd {WD} | clip {CLIP} | batch 1 | seed {SEED}")

    start_epoch, iters = 0, 0
    if args.resume:
        state = torch.load(ckpt_dir / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"]); opt.load_state_dict(state["optimizer"])
        random.setstate(state["py_rng"]); torch.set_rng_state(state["torch_rng"])
        start_epoch, iters = state["epoch"], state["iters"]
        log.info(f"resumed after epoch {start_epoch}")

    val = ValSet()
    enhance_fn = lambda x: torch.clamp(x / model.enhance(x), 0, 1)  # = official test-time Finetunemodel
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    times = []
    progress = Progress(log, n_epochs * len(images), iters, PRINT_EVERY)
    for epoch in range(start_epoch, n_epochs):
        model.train()
        order = list(range(len(images)))
        random.shuffle(order)
        losses = []
        with Timer() as t_epoch:
            for idx in order:
                x = to_tensor01(images[idx]).cuda()
                with Timer() as t:
                    opt.zero_grad()
                    loss = model._loss(x)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), CLIP)
                    opt.step()
                times.append(t.seconds)
                losses.append(loss.item())
                iters += 1
                if args.speed_test and iters >= args.speed_test:
                    speed_report(log, "SCI", times, EPOCHS * len(images), 1)
                    return
                progress.step(epoch + 1, n_epochs, iters, losses[-1])
        row = {"epoch": epoch + 1, "iters": iters, "train_loss": sum(losses) / len(losses),
               "lr": opt.param_groups[0]["lr"], "epoch_seconds": round(t_epoch.seconds, 1)}
        if (epoch + 1) % VAL_EVERY == 0 or epoch == 0:
            model.eval()
            row["val_psnr"] = val.mean_psnr(enhance_fn)
            log.info(f"epoch {epoch + 1}/{n_epochs} | mean train loss {row['train_loss']:.5f} | val PSNR "
                     f"{row['val_psnr']:.3f} dB (35 held-out pairs, curve only)")
        log.row(**row)
        if (epoch + 1) % CKPT_EVERY == 0 or epoch + 1 == n_epochs:
            state = {"model": model.state_dict(), "optimizer": opt.state_dict(), "epoch": epoch + 1, "iters": iters,
                     "py_rng": random.getstate(), "torch_rng": torch.get_rng_state()}
            torch.save(state, ckpt_dir / "last.pt")
            log.info(f"checkpoint saved: {ckpt_dir / 'last.pt'} (epoch {epoch + 1})")
    torch.save(model.state_dict(), ckpt_dir / "final.pt")  # same format as the official weights/*.pt
    log.info(f"TRAINING FINISHED: final checkpoint {ckpt_dir / 'final.pt'}")


if __name__ == "__main__":
    main()

"""Protocol B: train PairLIE (CVPR 2023) from scratch on LOL-v1-only low-light sequences.

Training data: the LOL-derived sequences of the authors' training set (configs/protocol_B/pairlie_train_sequences.txt:
128 sequences, 318 LOL-v1 training low images, validation images removed). Follows the official models/PairLIE/main.py:
net(), loss = 1 * C_loss + 1 * R_loss + 500 * P_loss, Adam(lr 1e-4, betas (0.9, 0.999), eps 1e-8), MultiStepLR
(x0.5 every 100 epochs), batch 1, 400 epochs; each step samples two different images of one sequence and applies the
SAME random 128x128 crop to both (official data.py transform1 + shared seed). Differences: seed 42, images cached in
RAM, validation PSNR (I = L^0.2 * R) on the 35 held-out pairs every 10 epochs, for curves only; the FINAL checkpoint
is used (PairLIE is unsupervised).

Usage:
    python scripts/train_pairlie.py                # full training (400 epochs)
    python scripts/train_pairlie.py --resume       # continue from checkpoints/PairLIE/last.pt
    python scripts/train_pairlie.py --speed_test 300
    python scripts/train_pairlie.py --debug 2      # 2-epoch debug run -> checkpoints/PairLIE_debug, logs/train_PairLIE_debug.*
"""
import argparse
import random
import sys

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs

from common import ROOT
from train_common import (LOL_TRAIN, SEED, Progress, Timer, TrainLog, ValSet, load_uint8, seed_everything, speed_report,
                          to_tensor01)

sys.path.insert(0, str(ROOT / "models" / "PairLIE"))
from net.net import net  # noqa: E402  (official PairLIE code)
from utils import C_loss, P_loss, R_loss  # noqa: E402  (official PairLIE losses)

EPOCHS, LR, DECAY, GAMMA, CROP, EXPONENT = 400, 1e-4, 100, 0.5, 128, 0.2  # official main.py / data.py / eval.py
VAL_EVERY, CKPT_EVERY, PRINT_EVERY = 10, 20, 640  # print every 5 epochs (128 iterations per epoch)
SEQ_FILE = ROOT / "configs" / "protocol_B" / "pairlie_train_sequences.txt"


def load_sequences():
    seqs = []
    for line in SEQ_FILE.read_text(encoding="utf-8").splitlines():
        names = line.split(":")[1].split()
        seqs.append([load_uint8(LOL_TRAIN / "low" / n) for n in names])
    return seqs


def sample_pair(seq):
    """Two different images of one sequence, same random 128x128 crop (official DatasetFromFolder + RandomCrop)."""
    i1, i2 = random.sample(range(len(seq)), 2)
    h, w = seq[i1].shape[:2]
    top, left = random.randint(0, h - CROP), random.randint(0, w - CROP)
    crop = lambda a: a[top:top + CROP, left:left + CROP]
    return to_tensor01(crop(seq[i1])).cuda(), to_tensor01(crop(seq[i2])).cuda()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--speed_test", type=int, default=0)
    ap.add_argument("--debug", type=int, default=0, help="run only N epochs into separate debug folders")
    args = ap.parse_args()
    global EPOCHS, VAL_EVERY, CKPT_EVERY, PRINT_EVERY
    tag = "PairLIE_speedtest" if args.speed_test else "PairLIE_debug" if args.debug else "PairLIE"
    ckpt_dir = ROOT / "checkpoints" / tag
    n_epochs = args.debug or EPOCHS  # debug runs keep the 400-epoch LR schedule, they just stop early
    if args.debug:
        VAL_EVERY, CKPT_EVERY, PRINT_EVERY = 1, 1, 64

    seed_everything(SEED)
    torch.backends.cudnn.benchmark = True  # as in official main.py
    seqs = load_sequences()
    model = net().cuda()
    opt = optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8)
    sched = lrs.MultiStepLR(opt, [e for e in range(1, EPOCHS + 1) if e % DECAY == 0], GAMMA)
    log = TrainLog(tag, ["epoch", "iters", "train_loss", "val_psnr", "lr", "epoch_seconds"], resume=args.resume)
    log.info(f"PairLIE Protocol B | {len(seqs)} sequences, {sum(map(len, seqs))} images | epochs {EPOCHS} | "
             f"Adam lr {LR}, x{GAMMA} every {DECAY} epochs | crop {CROP} | seed {SEED}")

    start_epoch, iters = 1, 0
    if args.resume:
        state = torch.load(ckpt_dir / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"]); opt.load_state_dict(state["optimizer"]); sched.load_state_dict(state["scheduler"])
        random.setstate(state["py_rng"]); torch.set_rng_state(state["torch_rng"])
        start_epoch, iters = state["epoch"] + 1, state["iters"]
        log.info(f"resumed after epoch {state['epoch']}")

    val = ValSet()

    def enhance_fn(x):
        L, R, _ = model(x)
        return torch.pow(L, EXPONENT) * R

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    times = []
    progress = Progress(log, n_epochs * len(seqs), iters, PRINT_EVERY)
    for epoch in range(start_epoch, n_epochs + 1):  # official: epochs 1..nEpochs, scheduler.step() after each
        model.train()
        order = list(range(len(seqs)))
        random.shuffle(order)
        losses = []
        with Timer() as t_epoch:
            for idx in order:
                im1, im2 = sample_pair(seqs[idx])
                with Timer() as t:
                    L1, R1, X1 = model(im1)
                    L2, R2, X2 = model(im2)
                    loss = C_loss(R1, R2) * 1 + R_loss(L1, R1, im1, X1) * 1 + P_loss(im1, X1) * 500
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                times.append(t.seconds)
                losses.append(loss.item())
                iters += 1
                if args.speed_test and iters >= args.speed_test:
                    speed_report(log, "PairLIE", times, EPOCHS * len(seqs), 1)
                    return
                progress.step(epoch, n_epochs, iters, losses[-1])
        sched.step()
        row = {"epoch": epoch, "iters": iters, "train_loss": sum(losses) / len(losses),
               "lr": opt.param_groups[0]["lr"], "epoch_seconds": round(t_epoch.seconds, 1)}
        if epoch % VAL_EVERY == 0 or epoch == 1:
            model.eval()
            row["val_psnr"] = val.mean_psnr(enhance_fn)
        log.row(**row)
        if "val_psnr" in row:
            log.info(f"epoch {epoch}/{n_epochs} | mean train loss {row['train_loss']:.5f} | val PSNR {row['val_psnr']:.3f} dB "
                     f"(35 held-out pairs, curve only) | lr {row['lr']:.2e}")
        if epoch % CKPT_EVERY == 0 or epoch == n_epochs:
            state = {"model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                     "epoch": epoch, "iters": iters, "py_rng": random.getstate(), "torch_rng": torch.get_rng_state()}
            torch.save(state, ckpt_dir / "last.pt")
            log.info(f"checkpoint saved: {ckpt_dir / 'last.pt'} (epoch {epoch})")
    torch.save(model.state_dict(), ckpt_dir / "final.pth")  # same format as the official weights/PairLIE.pth
    log.info(f"TRAINING FINISHED: final checkpoint {ckpt_dir / 'final.pth'}")


if __name__ == "__main__":
    main()

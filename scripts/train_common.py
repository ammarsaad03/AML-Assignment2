"""Shared helpers for Protocol B training scripts: seeding, data lists, CSV/text logging, validation PSNR, timing."""
import csv
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from common import ROOT, get_logger

sys.path.insert(0, str(ROOT))
from eval.metrics import psnr  # noqa: E402

SEED = 42
LOL_TRAIN = ROOT / "data" / "LOL-v1" / "train"


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def read_split(name):
    """Names from configs/train_split.txt or configs/val_split.txt."""
    return (ROOT / "configs" / f"{name}_split.txt").read_text(encoding="utf-8").split()


def load_uint8(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def to_tensor01(arr):
    """uint8 HxWx3 -> float32 1x3xHxW in [0, 1] (same as torchvision ToTensor)."""
    return torch.from_numpy(arr).permute(2, 0, 1).float().div(255.0).unsqueeze(0)


class ValSet:
    """The 35 held-out LOL-v1 training pairs (configs/val_split.txt), kept in memory as uint8."""

    def __init__(self):
        self.names = read_split("val")
        self.low = [load_uint8(LOL_TRAIN / "low" / n) for n in self.names]
        self.high = [load_uint8(LOL_TRAIN / "high" / n).astype(np.float64) / 255.0 for n in self.names]

    @torch.no_grad()
    def mean_psnr(self, enhance_fn, device="cuda"):
        """enhance_fn: 1x3xHxW tensor in [0, 1] -> 1x3xHxW tensor. PSNR on the 8-bit rounded output, as in testing."""
        scores = []
        for low, high in zip(self.low, self.high):
            out = enhance_fn(to_tensor01(low).to(device)).clamp(0, 1)
            out = np.round(out[0].permute(1, 2, 0).cpu().numpy() * 255.0) / 255.0
            scores.append(psnr(out, high))
        return float(np.mean(scores))


class TrainLog:
    """Text log (logs/train_<model>.log) + CSV (logs/train_<model>.csv), appended to when resuming."""

    def __init__(self, model_name, fields, resume=False):
        self.text = get_logger(ROOT / "logs" / f"train_{model_name}.log", mode="a" if resume else "w")
        self.csv_path = ROOT / "logs" / f"train_{model_name}.csv"
        self.fields = fields
        if not resume or not self.csv_path.exists():
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(fields)

    def row(self, **values):
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([values.get(k, "") for k in self.fields])

    def info(self, msg):
        self.text.info(msg)


class Progress:
    """Prints 'epoch | iter i/N | loss | s/it | elapsed | ETA' every `every` iterations (wall-clock, incl. data)."""

    def __init__(self, log, total_iters, start_iter, every):
        self.log, self.total, self.every = log, total_iters, every
        self.t0, self.i0, self.losses = time.perf_counter(), start_iter, []

    def step(self, epoch, n_epochs, it, loss):
        self.losses.append(loss)
        if it % self.every == 0 or it == self.total:
            elapsed = time.perf_counter() - self.t0
            s_it = elapsed / max(it - self.i0, 1)
            fmt = lambda s: time.strftime("%H:%M:%S", time.gmtime(s)) if s < 86400 else f"{s / 3600:.1f}h"
            self.log.info(f"epoch {epoch}/{n_epochs} | iter {it}/{self.total} | loss {np.mean(self.losses):.5f} | "
                          f"{s_it:.3f} s/it | elapsed {fmt(elapsed)} | ETA {fmt(s_it * (self.total - it))}")
            self.losses = []


def speed_report(log, model_name, times, iters_total, batch_size):
    """Log s/iter (excluding the first 10 warm-up iterations), peak GPU memory and the full-training estimate."""
    steady = times[10:] if len(times) > 20 else times
    s_per_iter = float(np.mean(steady))
    mem_gb = torch.cuda.max_memory_allocated() / 1024 ** 3
    reserved_gb = torch.cuda.max_memory_reserved() / 1024 ** 3
    est_h = s_per_iter * iters_total / 3600
    log.info(f"SPEED {model_name}: {len(times)} iters measured (batch {batch_size}) | {s_per_iter:.4f} s/iter | "
             f"peak GPU mem allocated {mem_gb:.2f} GB (reserved {reserved_gb:.2f} GB) | "
             f"full training = {iters_total} iters -> estimated {est_h:.2f} h")
    return s_per_iter, mem_gb, est_h


class Timer:
    def __enter__(self):
        torch.cuda.synchronize()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        torch.cuda.synchronize()
        self.seconds = time.perf_counter() - self.t0

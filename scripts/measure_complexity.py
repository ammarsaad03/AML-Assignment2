"""Parameters, FLOPs at one fixed input size, and GPU runtime for the four models (official Protocol A weights;
Protocol B models have identical architectures, so the same numbers apply).

- Params: parameters of the network actually used at test time (SCI: the single illumination block).
- FLOPs: fvcore FlopCountAnalysis on a 1x3x256x256 input (the size used in the Retinexformer paper's table).
  fvcore counts one multiply-add as one "flop" (i.e. MACs); element-wise ops it does not support are skipped and
  listed in the CSV column `fvcore_skipped_ops`.
- Runtime: per dataset, all test images pre-loaded on the GPU; 10 warm-up passes, then every image is timed 3 times
  (CUDA-synchronised) and averaged; reported = mean over images (ms/image), batch 1, same RTX 3060, no training running.
  Includes the test-time steps of each wrapper (e.g. Retinexformer padding, PairLIE L^0.2 * R), not file I/O.

Each model runs in its own subprocess (sequentially): separate GPU memory and no clashes between the repos' modules
(SCI/URetinex/PairLIE all have top-level files with the same names).

Usage: python scripts/measure_complexity.py            # all models -> results/complexity.csv
       python scripts/measure_complexity.py --one SCI  # (internal) one model -> JSON on stdout
"""
import argparse
import csv
import json
import subprocess
import sys
import time

from common import DATASETS, ROOT, list_images, load_rgb01

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

MODELS = ["SCI", "PairLIE", "URetinex-Net", "Retinexformer", "RUAS"]
SIZE = 256
WARMUP, REPEATS = 10, 3


class TestTime(nn.Module):
    """Wraps a model's full test-time computation as one module with a single tensor output (needed by fvcore)."""

    def __init__(self, net, fn):
        super().__init__()
        self.net, self.fn = net, fn

    def forward(self, x):
        return self.fn(self.net, x)


def build(name, device):
    if name == "SCI":
        import run_sci
        model, _, _ = run_sci.build_model(run_sci.SCI_DIR / "weights" / "medium.pt", device)
        return TestTime(model, lambda m, x: m(x)[1])
    if name == "PairLIE":
        import run_pairlie
        model, _ = run_pairlie.build_model(run_pairlie.PAIRLIE_DIR / "weights" / "PairLIE.pth", device)

        def pairlie(m, x):
            L, R, _ = m(x)
            return torch.pow(L, run_pairlie.EXPONENT) * R
        return TestTime(model, pairlie)
    if name == "URetinex-Net":
        import run_uretinex
        return TestTime(run_uretinex.build_model(device), lambda m, x: m(x)[0])  # Inference returns (image, time)
    if name == "Retinexformer":
        import run_retinexformer
        model, _, _ = run_retinexformer.build_model(device)
        return TestTime(model, run_retinexformer.enhance)
    if name == "RUAS":
        import run_ruas
        model, _ = run_ruas.build_model(run_ruas.RUAS_DIR / "ckpt" / "lol.pt", device)
        return TestTime(model, run_ruas.enhance)
    raise ValueError(name)


@torch.inference_mode()
def runtime_ms(model, images):
    for _ in range(WARMUP):
        model(images[0])
    per_image = []
    for x in images:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(REPEATS):
            model(x)
        torch.cuda.synchronize()
        per_image.append((time.perf_counter() - t0) / REPEATS * 1000)
    return sum(per_image) / len(per_image)


def measure_one(name):
    from fvcore.nn import FlopCountAnalysis

    device = "cuda"
    model = build(name, device).eval()
    params = sum(p.numel() for p in model.parameters())
    with torch.no_grad():
        flops = FlopCountAnalysis(model, (torch.rand(1, 3, SIZE, SIZE, device=device),))
        flops.unsupported_ops_warnings(False).uncalled_modules_warnings(False)
        total = flops.total()
        skipped = sorted(flops.unsupported_ops().keys())
    result = {"model": name, "params": params, "gmacs_256": total / 1e9, "fvcore_skipped_ops": " ".join(skipped)}
    for ds, d in DATASETS.items():
        images = [load_rgb01(p, device) for p in list_images(d["low"])]
        result[f"runtime_ms_{ds}"] = runtime_ms(model, images)
    result["gpu"] = torch.cuda.get_device_name(0)
    result["torch"] = torch.__version__
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--one", choices=MODELS, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.one:
        print("RESULT_JSON " + json.dumps(measure_one(args.one)))
        return

    rows = []
    for name in MODELS:  # sequential subprocesses
        print(f"measuring {name} ...", flush=True)
        out = subprocess.run([sys.executable, __file__, "--one", name], capture_output=True, text=True, cwd=ROOT)
        line = next((l for l in out.stdout.splitlines() if l.startswith("RESULT_JSON ")), None)
        if out.returncode != 0 or line is None:
            print(out.stdout[-2000:], out.stderr[-3000:], sep="\n")
            raise SystemExit(f"{name}: measurement failed (exit {out.returncode})")
        r = json.loads(line[len("RESULT_JSON "):])
        rows.append(r)
        print(f"  params {r['params']:,} | {r['gmacs_256']:.3f} GMACs @256 (fvcore) | "
              f"{r['runtime_ms_LOL-v1']:.2f} ms/img LOL-v1 | {r['runtime_ms_LSRW']:.2f} ms/img LSRW"
              + (f" | fvcore skipped: {r['fvcore_skipped_ops']}" if r["fvcore_skipped_ops"] else ""))

    path = ROOT / "results" / "complexity.csv"
    fields = ["model", "params", "gmacs_256", "runtime_ms_LOL-v1", "runtime_ms_LSRW", "fvcore_skipped_ops", "gpu", "torch"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({**r, "gmacs_256": f"{r['gmacs_256']:.4f}", "runtime_ms_LOL-v1": f"{r['runtime_ms_LOL-v1']:.3f}",
                        "runtime_ms_LSRW": f"{r['runtime_ms_LSRW']:.3f}"})
    print("written:", path.relative_to(ROOT))


if __name__ == "__main__":
    main()

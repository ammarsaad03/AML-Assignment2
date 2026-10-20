"""Protocol B: train Retinexformer with the OFFICIAL basicsr/train.py and our option file.

Runs models/Retinexformer/basicsr/train.py (cwd = repo, repo on PYTHONPATH, basicsr not installed) with
configs/protocol_B/RetinexFormer_LOL_v1_protocolB.yml, and copies its console output to logs/train_Retinexformer.log.
basicsr auto-resumes from the newest training state in checkpoints/Retinexformer/<name>/training_states/, so the same
command continues a stopped run. TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 lets PyTorch >= 2.6 load those state files.

Usage:
    python scripts/train_retinexformer.py            # full training (150k iterations), or resume if states exist
    python scripts/train_retinexformer.py --debug 200  # 200-iteration trial run (val + checkpoint every 100)
"""
import argparse
import os
import subprocess
import sys

from common import ROOT

RF_DIR = ROOT / "models" / "Retinexformer"
CFG_DIR = ROOT / "configs" / "protocol_B"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", type=int, default=0, help="short trial run with N iterations (val + checkpoint every N/2)")
    args = ap.parse_args()

    if args.debug:
        subprocess.run([sys.executable, str(ROOT / "scripts" / "make_retinexformer_cfg.py"), "--debug", str(args.debug)], check=True)
    cfg = CFG_DIR / ("RetinexFormer_trial.yml" if args.debug else "RetinexFormer_LOL_v1_protocolB.yml")
    log_path = ROOT / "logs" / ("train_Retinexformer_debug.log" if args.debug else "train_Retinexformer.log")

    exp = RF_DIR / "experiments"
    if not exp.exists():  # basicsr writes to <repo>/experiments; point it at checkpoints/Retinexformer
        (ROOT / "checkpoints" / "Retinexformer").mkdir(parents=True, exist_ok=True)
        subprocess.run(["cmd", "/c", "mklink", "/J", str(exp), str(ROOT / "checkpoints" / "Retinexformer")], check=True)

    # OPENBLAS_NUM_THREADS=1: inherited by basicsr and every data-loader worker (memory limit on 8 GB RAM; see README)
    env = dict(os.environ, PYTHONPATH=str(RF_DIR), TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1", PYTHONUNBUFFERED="1",
               OPENBLAS_NUM_THREADS="1")
    cmd = [sys.executable, "basicsr/train.py", "--opt", str(cfg)]
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"\n===== {' '.join(cmd)} =====\n")
        proc = subprocess.Popen(cmd, cwd=RF_DIR, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace", bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
            log.flush()
        code = proc.wait()
    print(f"\nbasicsr/train.py exited with code {code} | log: {log_path}")
    sys.exit(code)


if __name__ == "__main__":
    main()

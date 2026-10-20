"""Write the Protocol B option file for Retinexformer from the official Options/RetinexFormer_LOL_v1.yml.

Only these fields change (everything else, incl. 150k iterations, batch 8, patch 128, Adam 2e-4, cosine schedule,
L1 loss, flip/rotation + mixup, grad clipping, val every 1e3 iterations, is kept):
  - name                    -> RetinexFormer_LOL_v1_protocolB
  - manual_seed             -> 42 (official 100)
  - train data              -> data/LOL-v1/train, 450 pairs listed in configs/protocol_B/train_meta.txt
  - val data                -> 35 held-out training pairs, configs/protocol_B/val_meta.txt (official: LOL-v1 Test = test set)
  - num_worker_per_gpu      -> 2 (official 8): Windows + 8 GB RAM. Each worker is a full Python process (torch + numpy);
                               with 4 workers the peak commit reached 20.6 of 21.4 GB (96%). See README "Known issues".
  - dataset_enlarge_ratio   -> 50 (official 1): data-loader setting only. With 450 images and batch 8 an epoch is just 56
                               iterations, and Windows re-spawns the loader workers every epoch (0.50 vs 0.14 s/iter).
                               Total iterations, sampling distribution and training are unchanged.
  - logger.use_tb_logger    -> false (tensorboard not installed); wandb project -> null
  - logger.print_freq       -> 100 (official 500): progress printed more often, logging only

Usage: python scripts/make_retinexformer_cfg.py [--debug 200]
"""
import argparse

import yaml

from common import ROOT

OFFICIAL = ROOT / "models" / "Retinexformer" / "Options" / "RetinexFormer_LOL_v1.yml"
OUT_DIR = ROOT / "configs" / "protocol_B"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", type=int, default=0, help="short debug config with N iterations")
    args = ap.parse_args()

    opt = yaml.safe_load(open(OFFICIAL, encoding="utf-8"))
    train_dir = (ROOT / "data" / "LOL-v1" / "train").as_posix()
    opt["name"] = "RetinexFormer_LOL_v1_protocolB"
    opt["manual_seed"] = 42
    for split in ("train", "val"):
        d = opt["datasets"][split]
        d["dataroot_gt"], d["dataroot_lq"] = f"{train_dir}/high", f"{train_dir}/low"
        d["meta_info_file"] = (OUT_DIR / f"{split}_meta.txt").as_posix()
    opt["datasets"]["train"]["num_worker_per_gpu"] = 2
    opt["datasets"]["train"]["dataset_enlarge_ratio"] = 50
    opt["logger"]["use_tb_logger"] = False
    opt["logger"]["wandb"]["project"] = None
    opt["logger"]["print_freq"] = 100

    out = OUT_DIR / "RetinexFormer_LOL_v1_protocolB.yml"
    if args.debug:  # short run that exercises validation, best-model saving and resuming
        opt["name"] += "_trial"  # not "_debug": basicsr overrides frequencies for names containing "debug"
        opt["train"]["total_iter"] = args.debug
        opt["val"]["val_freq"] = args.debug // 2
        opt["logger"]["save_checkpoint_freq"] = args.debug // 2
        opt["logger"]["print_freq"] = 20
        out = OUT_DIR / "RetinexFormer_trial.yml"
    out.write_text(yaml.safe_dump(opt, sort_keys=False), encoding="utf-8")
    print("written:", out.relative_to(ROOT))


if __name__ == "__main__":
    main()

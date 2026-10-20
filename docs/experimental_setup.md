# Experimental setup

Two protocols. **Protocol A** tests the official pretrained weights (training rows = the authors' settings, not
re-trained by me). **Protocol B** is my own retraining under one protocol on the LOL-v1 training split. Sources:
`configs/<model>.yaml`, `docs/protocol.md` (all deviations), training logs in `logs/`.

## Common to all models and both protocols

| Field | Setting |
|---|---|
| Test datasets | LOL-v1 eval15 (15 pairs, 600x400); LSRW Eval (50 pairs: Huawei 30 at 960x720, Nikon 20 at 960x640) |
| Data source | LOL-v1: Google Drive link in the Retinexformer README (official page https://daooshee.github.io/BMVC2018website/); LSRW: Kaggle mirror of the official release (https://github.com/JianghaiSCU/R2RNet) |
| Test preprocessing | RGB, uint8 / 255 -> [0, 1], full resolution, no resizing; output clipped to [0, 1], saved as 8-bit PNG (rounded); same wrapper code for every model (`scripts/common.py`) |
| Test-time rules | no ground truth at inference, no GT-mean correction, no self-ensemble, no tuning on test sets (URetinex-Net fixed ratio 5, PairLIE exponent 0.2) |
| Test input size / batch | full resolution, batch 1 |
| Metrics | PSNR, SSIM (scikit-image, RGB, `data_range=1`), LPIPS (AlexNet); Params, FLOPs at 1x3x256x256 (fvcore), GPU runtime (warm-up, batch 1) |
| Hardware / software | NVIDIA RTX 3060 12 GB, 8 GB RAM, Windows 11; Python 3.10.21, PyTorch 2.7.1 + CUDA 11.8, cuDNN 9.1 (`environment.txt`) |

## Protocol A - official weights (training = author's setting, not re-trained)

| Field | SCI | PairLIE | URetinex-Net | Retinexformer | RUAS |
|---|---|---|---|---|---|
| Pretrained weights | `CVPR/weights/medium.pt` | `weights/PairLIE.pth` | `ckpt/init_low.pth`, `unfolding.pth`, `L_adjust.pth` | `LOL_v1.pth` (authors' Google Drive) | `ckpt/lol.pt` (+ `model/denoise.pt` init) |
| Trained on (authors) | mainly LOL + LSRW (README; not the paper checkpoint) | SICE part 2 + LOL train, 324 sequences | LOL train | LOL-v1 train, 485 pairs (validated on LOL-v1 Test) | LOL, 689 pairs (paper; LOL-v2-real size) |
| Supervision | unsupervised (low only) | unsupervised (low-light pairs) | supervised | supervised | unsupervised (low only) |
| Train input size | full image | 128x128 crop | 48x48 patches (unfolding) | 128x128 crop | 320x320 crop |
| Batch size | 8 (paper); 1 (code) | 1 | 4 | 8 | 1 |
| Epochs / iterations | 1000 epochs | 400 epochs | not reported (LR step at 30K it.) | 150k it. (yml); 250k (paper) | 800 it. |
| Optimizer | Adam (code: + wd 3e-4) | Adam | Adam | Adam, grad clip | SGD, momentum 0.9, wd 3e-4 |
| Learning rate | 1e-4 (paper); 3e-4 (code) | 1e-4, x0.5 every 100 ep. | 1e-4, /10 after 30K it. | 2e-4, cosine to 1e-6 | IEM 0.015, NRM 0.001 |
| Loss | fidelity + smoothness | consistency + Retinex + 500 x projection | per module; adjustment: grad L1 + MSE + (1 - SSIM) | L1 | IEM fidelity + smoothness; NRM fidelity + TV |
| Augmentation | none | random crop | not reported | flip/rotation + mixup | random crop |
| Authors' hardware | TITAN X | RTX 2080Ti | Tesla V100 | not reported | TITAN X |

## Protocol B - retrained by me (LOL-v1 train 450 pairs, validation 35 pairs, seed 42)

| Field | SCI | PairLIE | Retinexformer | RUAS |
|---|---|---|---|---|
| Training data | 450 LOL-v1 low images | 128 LOL-derived sequences of the authors' set (318 LOL-v1 train low images) | 450 LOL-v1 pairs | 450 LOL-v1 low images (random image per step) |
| Validation (35 held-out pairs) | PSNR every 5 epochs, curve only | PSNR every 10 epochs, curve only | PSNR every 1000 it., selects checkpoint | PSNR every 50 it., curve only |
| Initialisation | random (official init) | random | random | IEM random; NRM from authors' `denoise.pt` |
| Train input size | full 600x400 | 128x128 crop | 128x128 crop | 320x320 crop |
| Batch size | 1 | 1 | 8 | 1 |
| Epochs / iterations | 200 epochs = 90,000 it. (official 1000 epochs) | 400 epochs = 51,200 it. | 150,000 it. | 800 it. |
| Optimizer | Adam, wd 3e-4, clip 5 | Adam | Adam, grad clip | SGD, momentum 0.9, wd 3e-4, clip 5 |
| Learning rate | 3e-4 constant | 1e-4, x0.5 every 100 epochs | 2e-4, cosine schedule (official yml) | IEM 0.015, NRM 0.001 (every 50th step) |
| Loss | official | official | official (L1) | official |
| Augmentation | none | random crop | flip/rotation + mixup | random crop |
| Checkpoint tested | final (epoch 200) | final (epoch 400) | best validation (it. 103,000, 22.72 dB) | final (it. 800) |
| Training time (RTX 3060) | 6 h 39 min | 18 min 43 s | 6 h 31 min | 60 s |

URetinex-Net is not in Protocol B: the official repository has no training code (see `docs/protocol.md`).

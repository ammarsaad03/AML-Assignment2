# Results

Test sets: LOL-v1 eval15 (15 pairs), LSRW Eval (50 pairs). Same preprocessing and metric code for every model
(`eval/metrics.py`; PSNR/SSIM on RGB in [0, 1], LPIPS AlexNet); no GT at inference, no GT-mean, no test-set tuning.
Best value per column (within each protocol) in **bold**. FLOPs: fvcore, 1x3x256x256, multiply-adds.
Runtime: RTX 3060, batch 1, mean ms per image after warm-up. PSNR drop = PSNR(LOL-v1) − PSNR(LSRW).

## Protocol A - official weights

| Model | Weights | LOL-v1 PSNR ↑ | LOL-v1 SSIM ↑ | LOL-v1 LPIPS ↓ | LSRW PSNR ↑ | LSRW SSIM ↑ | LSRW LPIPS ↓ | Params (M) ↓ | FLOPs@256 (G) ↓ | Runtime LOL-v1 (ms) ↓ | Runtime LSRW (ms) ↓ | PSNR drop LOL-v1→LSRW (dB) ↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SCI | official | 14.85 | 0.5274 | 0.3389 | 15.28 | 0.4219 | 0.3221 | **0.0003** | **0.016** | **5.9** | **15.7** | **-0.42** |
| PairLIE | official | 18.50 | 0.7430 | 0.2434 | 17.61 | 0.5077 | 0.3310 | 0.3418 | 22.347 | 35.4 | 96.2 | +0.89 |
| URetinex-Net | official | 19.86 | 0.8239 | **0.1282** | **18.25** | **0.5188** | **0.2994** | 0.3401 | 56.934 | 81.7 | 222.5 | +1.61 |
| Retinexformer | official | **25.15** | **0.8434** | 0.1314 | 17.78 | 0.5085 | 0.3196 | 1.6057 | 16.716 | 103.6 | 286.1 | +7.37 |
| RUAS | official | 16.45 | 0.5053 | 0.2698 | 14.29 | 0.4633 | 0.4648 | 0.0034 | 0.214 | 37.1 | 101.0 | +2.15 |

## Protocol B - retrained on LOL-v1 train (450 pairs), one protocol

| Model | Weights | LOL-v1 PSNR ↑ | LOL-v1 SSIM ↑ | LOL-v1 LPIPS ↓ | LSRW PSNR ↑ | LSRW SSIM ↑ | LSRW LPIPS ↓ | Params (M) ↓ | FLOPs@256 (G) ↓ | Runtime LOL-v1 (ms) ↓ | Runtime LSRW (ms) ↓ | PSNR drop LOL-v1→LSRW (dB) ↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SCI | retrained (ours) | 12.97 | 0.4964 | 0.3493 | 14.31 | 0.3986 | **0.3297** | **0.0003** | **0.016** | **5.9** | **15.7** | **-1.34** |
| PairLIE | retrained (ours) | 18.22 | 0.7217 | 0.3955 | 17.52 | **0.5175** | 0.4793 | 0.3418 | 22.347 | 35.4 | 96.2 | +0.69 |
| Retinexformer | retrained (ours) | **22.11** | **0.8237** | **0.1508** | **17.61** | 0.5095 | 0.3310 | 1.6057 | 16.716 | 103.6 | 286.1 | +4.50 |
| RUAS | retrained (ours) | 16.17 | 0.5001 | 0.2651 | 14.05 | 0.4603 | 0.4606 | 0.0034 | 0.214 | 37.1 | 101.0 | +2.12 |

## LSRW by camera (PSNR, dB)

| Model | A: Huawei (30) | A: Nikon (20) | B: Huawei (30) | B: Nikon (20) |
|---|---|---|---|---|
| SCI | 15.76 | 14.55 | 14.00 | 14.79 |
| PairLIE | 19.02 | 15.50 | 19.05 | 15.23 |
| URetinex-Net | 19.30 | 16.66 | n/a | n/a |
| Retinexformer | 19.09 | 15.81 | 19.19 | 15.25 |
| RUAS | 15.73 | 12.14 | 15.53 | 11.82 |

URetinex-Net: no official training code, so it is evaluated in Protocol A only (see docs/protocol.md).

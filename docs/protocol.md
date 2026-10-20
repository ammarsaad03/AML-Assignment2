# Evaluation protocol

Two protocols are reported:
- **Protocol A (official weights):** each model's official pretrained checkpoint, tested as described below.
- **Protocol B (retrained by us under one protocol):** each model trained from scratch on the LOL-v1 training set
  (450 pairs; 35 held out for validation, seed 42) with its own official architecture, loss, optimizer and learning
  rate, then tested exactly like Protocol A. Deviations from the official training setups are listed at the end.

Test-only comparison of four low-light image enhancement models with their official pretrained weights:
SCI, PairLIE, URetinex-Net, Retinexformer, on LOL-v1 (eval15) and LSRW (official Eval split, Huawei + Nikon).

## Fairness decisions

General rules (apply to every model and dataset):
- **No ground truth at inference.** The normal-light image is only read by `eval/metrics.py`, never by a model script.
- **No GT-mean correction.** Outputs are scored as produced (no rescaling of the output to the GT mean brightness).
- **No tuning on test sets.** Every hyper-parameter is fixed to the official default and is the same for both datasets.
- **Same preprocessing for all models.** Each low image is read as RGB, converted to float in [0, 1], full resolution,
  no resizing or cropping. Outputs are clipped to [0, 1] and saved as 8-bit PNG, **rounded** to the nearest level
  (`scripts/common.py`). Some official scripts truncate instead (e.g. SCI's `astype('uint8')`); this changes pixels by
  at most 1/255 but shifts PSNR slightly (SCI on LOL-v1: 14.851 dB rounded vs 14.781 dB truncated), so one shared
  saving function is used for every model.
- **Same metrics code for all models** (`eval/metrics.py`): PSNR and SSIM on RGB in [0, 1] (skimage, `data_range=1`,
  `channel_axis=-1`), LPIPS with AlexNet backbone.

Per-model decisions:

| Model | Decision | Reason |
|---|---|---|
| SCI | Use the weights shipped in the repo (`CVPR/weights/`). | The SCI README states these are *not* the exact checkpoints behind the paper's tables; our numbers may therefore differ from the paper. |
| SCI | Use `medium.pt` for both datasets. | It is the official default in `test.py` and, per the README, was trained mainly on LOL + LSRW, i.e. the matching domain. Caveat: the README does not say which LOL/LSRW images were used (SCI is unsupervised and needs only low-light images), so overlap with the test images cannot be ruled out. `easy.pt` (MIT) and `difficult.pt` (DARK FACE) target other domains. |
| SCI | Test-time network = the official `Finetunemodel` (one illumination block, `r = clamp(x / i, 0, 1)`); our wrapper additionally checks that all 18 `enhance.*` tensors were loaded, because the official loader silently skips unmatched keys. | Our output matches the official `test.py` output to within 1/255 per pixel (difference = rounding vs truncation only). |
| SCI | Do not use SCI's own LSRW link. | It is the authors' random 500/50 split, not the official LSRW Eval set. |
| URetinex-Net | Use the `test.py` pipeline with the fixed default illumination ratio **5**. | The official `evaluate.py` computes the ratio from the ground-truth image, which leaks GT information. |
| PairLIE | Keep the default exponent **0.2** (`I = L^0.2 * R`) for both datasets. | The code comment suggests 0.14 for LOL; choosing it would be tuning on the test set. |
| PairLIE | Official network `net/net.py`, weights `weights/PairLIE.pth` loaded strictly. | Our output matches the official `eval.py` output to within 1/255 per pixel (rounding vs truncation). For that cross-check only, `eval.py` was run through a launcher that changes its hard-coded `CUDA_VISIBLE_DEVICES='3'` to `'0'` in memory; the repo file was not edited. |
| URetinex-Net | Official `Inference` class from `test.py` (ratio 5), built with `torch.load(..., weights_only=False)` because the checkpoints contain an `argparse.Namespace`. `ckpt/init_high.pth` is not used (it decomposes the GT image). | Our output matches the official `test.py` to within 1/255 per pixel (rounding vs truncation). For that cross-check only, `test.py` was run with `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`; no repo file was edited. |
| RUAS | Official `ckpt/lol.pt` (README: "trained from LOL dataset"; paper: 689 LOL pairs for search/training, which matches the size of LOL-v2-real, not LOL-v1 - overlap with our test images cannot be checked). Output = after the noise-removal module (`u_list[-1]`), as `test.py` does for `lol`. | Official test setting for the LOL model; no tuning. |
| Retinexformer | Import the model file directly from our wrapper (no `basicsr` install); load with `torch.load(..., weights_only=False)`; no `--GT_mean`, no `--self_ensemble`. | Same code and weights as official, nothing extra installed; `weights_only=False` is needed for old checkpoints under PyTorch >= 2.6. |
| Retinexformer | Cross-check: official `Enhancement/test_from_dataset.py` run once with the repo on `PYTHONPATH` (basicsr still not installed; only its import dependencies `natsort lmdb addict yapf` were pip-installed into aml2) and a scratch copy of `Options/RetinexFormer_LOL_v1.yml` whose `val` dataroots point to our data. | Our output is pixel-identical to the official output (0 differing pixels). The official script prints SSIM 0.8455 vs our 0.8434 only because it uses its own SSIM implementation; we score all models with `eval/metrics.py`. |

## Protocol B deviations

Common to all retrained models:
- **Seed 42** for all models (official: SCI 2, PairLIE 123456789, Retinexformer 100).
- **Training data:** LOL-v1 training set only, 450 pairs (`configs/train_split.txt`).
- **Validation:** 35 pairs held out from the LOL-v1 training set (`configs/val_split.txt`, random, seed 42). LOL-v1 eval15
  and LSRW are never used during training or for checkpoint selection. Pixel-hash check: no training or validation
  image equals a test image.
- **Checkpoint selection:** final checkpoint for the unsupervised methods (SCI, PairLIE, RUAS; validation PSNR is only
  logged for the curves), best validation PSNR for the supervised methods (Retinexformer).
- **Testing:** same wrappers, preprocessing, saving and `eval/metrics.py` as Protocol A.

| Model | Deviation from the official training setup | Reason |
|---|---|---|
| SCI | Trained by `scripts/train_sci.py`, which imports the official `Network` (incl. loss) and repeats the official `train.py` loop (same init, Adam 3e-4 / wd 3e-4, grad clip 5, batch 1, full images). | Official `train.py` hard-codes the data path as the string `'Your train dataset'`; no repo edit. |
| SCI | **200 epochs (90,000 iterations) instead of the official 1000.** | Reduced from the official 1000 epochs (~34 h on RTX 3060) to 200 epochs due to hardware limits; validation PSNR curve logged to show convergence. |
| SCI | Training images cached in RAM; validation PSNR every 5 epochs. | Speed / curves only. |
| SCI | Code values used (batch 1, lr 3e-4, wd 3e-4), not the paper's (batch 8, lr 1e-4). | Decision: follow the official training code. |
| PairLIE | Training data = only the LOL-derived sequences of the authors' training set (128 sequences, 318 LOL-v1 training low images; SICE sequences and our validation images removed). | Protocol B is LOL-only; PairLIE needs >= 2 low-light shots of the same scene, which plain LOL-v1 pairs do not provide. |
| PairLIE | Trained by `scripts/train_pairlie.py` with the official `net`, losses, optimizer and schedule; one shared random 128x128 crop for both images of a pair (the official code achieves this by re-seeding before each transform). | Official `dataset.py` hard-codes 324 sequences in folders `1..N`; no repo edit. |
| PairLIE | Official 400 epochs kept, i.e. 400 x 128 sequences = 51,200 iterations (official: 400 x 324 = 129,600). | The fewer iterations follow from using only the LOL-derived sequences (128 instead of 324 per epoch). |
| Retinexformer | Official `basicsr/train.py`, run with the repo on `PYTHONPATH` and our option file `configs/protocol_B/RetinexFormer_LOL_v1_protocolB.yml` (generated by `scripts/make_retinexformer_cfg.py`). Changed fields only: data paths + meta-info lists, seed 42, **validation = our 35 held-out pairs (official: LOL-v1 Test, i.e. the test set)**, `num_worker_per_gpu` 2 (official 8; with 4 the peak memory commit reached 96% of the 21.4 GB limit on this 8 GB RAM laptop, speed is the same with 2: ~0.144 s/iter), `OPENBLAS_NUM_THREADS=1` for all processes, `use_tb_logger` false, `dataset_enlarge_ratio` 50 (official 1). | Fixes the official validation-on-test leak; Windows data-loader speed (epoch of 56 iterations re-spawned workers: 0.50 vs 0.14 s/iter); tensorboard not installed. Iterations (150k), batch 8, patch 128, optimizer, schedule, loss and augmentation unchanged. |
| Retinexformer | `models/Retinexformer/experiments` is a directory junction to `checkpoints/Retinexformer`. | basicsr writes to `<repo>/experiments`; keeps checkpoints in our folder without a code change. |
| Retinexformer | Run through `scripts/train_retinexformer.py`, which starts the official `basicsr/train.py` and copies its output to `logs/train_Retinexformer.log`; `print_freq` 100 (official 500). | Logging only. |
| URetinex-Net | **Protocol A only** (decision 2026-10-19, replacing the earlier U2 choice of repeating its official weights in Protocol B). RUAS (CVPR 2021) was added to both protocols as the fourth retrainable model, so Protocol A has 5 models and Protocol B has 4. | The official repo has no training code. The authors' URetinex-Net-PLUS repo (inspected, see `docs/sources.md`) has identical network definitions for all modules (strict weight loading both ways), but its adjustment-stage training (`adjust_L_training.py`) always trains an extra fusion network and uses a different loss (MSE + gradient + spatial consistency instead of the paper's gradient + Frobenius + SSIM terms), so it cannot train URetinex-Net's adjustment module without code changes. |
| RUAS | **Used the authors' searched architecture** (`genotypes.py`: IEM and NRM genotypes); `train_search.py` (architecture search) was not run. | The README notes that a re-search on other data gives a different architecture; searching (paper: ~5 h) is outside the scope of retraining. |
| RUAS | **Fixed data-loading bug in official `train.py` (always used one image):** it calls `next(iter(train_queue))` on a non-shuffled DataLoader every step, which always returns the first image. We draw a random training image (from our 450 LOL-v1 train low images) at every step, with the official random 320x320 crop. | Bug fix; the paper does not describe training on a single image. |
| RUAS | **Denoiser not trained from scratch (official behaviour kept):** `Network.__init__` initialises the noise-removal module (NRM) from the authors' `model/denoise.pt`; it is then updated only every 50th step (SGD lr 0.001), i.e. 16 updates in 800 iterations. The training data of `denoise.pt` is **not reported** (README, paper and code are silent). The illumination module (IEM) is trained from scratch. | Official code; a randomly initialised NRM would barely change in 16 small updates. Limitation: part of the Protocol B RUAS model comes from pretrained weights of unknown origin. |
| RUAS | Official 800 iterations, batch 1, SGD (IEM lr 0.015, NRM lr 0.001), momentum 0.9, weight decay 3e-4, grad clip 5, official losses; seed 42. Trained by `scripts/train_ruas.py` with the official `Network`. | Same as the official code ("3 minutes to train" in the paper). |
| RUAS | PyTorch 2.x fix in our script: the NRM update uses a detached copy of the IEM output. The official step crashes in PyTorch >= 1.5 ("modified by an inplace operation"), because the NRM backward pass runs through IEM weights already updated in place by the IEM optimizer step. The NRM gradients are identical; the extra gradients the official code sent into the IEM were cleared by the next `zero_grad()` before any use. Also: paths set by our script (official ones are hard-coded), `./model/denoise.pt` loaded with the repo as working directory. | Compatibility only; no repo file edited. |

## Supplementary experiment: RUAS-long (decided 2026-10-20, before any result)

The Protocol B RUAS run uses the official 800 iterations, and its validation PSNR was still rising at the end
(Figure 5). To test whether the short official schedule limits RUAS, one extra run was **fixed in advance**:
identical settings to Protocol B RUAS (data, seed 42, losses, SGD learning rates, NRM every 50th step, 320x320 crops),
only **50,000 iterations** instead of 800; validation every 500 iterations (curve only); the **final** checkpoint is
tested (no checkpoint picking, as for all unsupervised models). Command: `scripts/train_ruas.py --long`, output
`checkpoints/RUAS_long/`, results folder `results/RUAS_B_long/`. This is reported as a supplementary result; the main
Protocol B table keeps the official 800-iteration RUAS, because every model in Protocol B uses its own official
training length (only SCI was shortened, for hardware reasons).

**Result** (training 51 min 21 s; `logs/train_RUAS_long.*`, `figures/train_curve_RUAS_long.png`):

| RUAS | LOL-v1 PSNR / SSIM / LPIPS | LSRW PSNR / SSIM / LPIPS | Mean brightness LOL-v1 (GT 0.451) |
|---|---|---|---|
| Official weights (Protocol A) | 16.45 / 0.5053 / 0.2698 | 14.29 / 0.4633 / 0.4648 | 0.371 |
| 800 iterations (Protocol B) | 16.17 / 0.5001 / 0.2651 | 14.05 / 0.4603 / 0.4606 | 0.397 |
| 50,000 iterations (supplementary) | 10.14 / 0.4246 / 0.4712 | 8.80 / 0.3946 / 0.6047 | 0.755 |

Longer training made RUAS much worse: it over-brightens (44% of LOL-v1 output pixels >= 0.98). The validation PSNR
peaked at 12.99 dB around iteration 1,000 and then stayed at about 12.2-12.5 dB, so it did not reveal the drift:
on the validation pairs the 800-iteration model is too dark (mean 0.302 vs GT 0.458) and the 50k model too bright
(0.654), two opposite errors with a similar PSNR. Both values were re-checked with the test wrapper (12.897 and
12.309 dB, equal to the logged values). Conclusion: the short official schedule is not what limits RUAS; with a
constant learning rate and an unsupervised loss, longer training drifts towards over-exposure.

## Analyses (not reported as results)

**Retinexformer: effect of selecting the checkpoint on the test set** (`scripts/analysis_selection_bias.py`,
`results/analysis_retinexformer_selection.csv`). All 150 checkpoints of our Protocol B run (one per 1000 iterations)
were scored on LOL-v1 eval15 with our test pipeline, only to quantify the official practice of validating on LOL-v1
Test:

| Checkpoint choice | Iteration | LOL-v1 eval15 PSNR | SSIM |
|---|---|---|---|
| Ours: best validation PSNR on the 35 held-out training pairs (reported) | 103,000 | 22.110 | 0.8237 |
| "Best on test" (what validating on LOL-v1 Test would select) | 93,000 | 23.628 | 0.8255 |

Selecting on the test set adds +1.52 dB. Over all 150 checkpoints the test PSNR has mean 22.14, std 0.82, range
19.83-23.63 dB; validation and test PSNR correlate with r = 0.70. About half of the 3.04 dB gap between the official
checkpoint (25.15 dB) and our retrained one is therefore explained by test-set selection alone; the rest by other
differences (485 vs 450 training pairs, seed, run-to-run variance).

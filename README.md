# AML Assignment 2 - Retinex-based low-light image enhancement: SCI, PairLIE, URetinex-Net, Retinexformer, RUAS

Comparative evaluation of five Retinex-based low-light image enhancement (LLIE) models on **LOL-v1** (eval15,
15 pairs) and **LSRW** (official Eval split, 50 pairs: Huawei 30 + Nikon 20). Assignment setting: 1, Dataset-Based
Evaluation. Author: Ammar Elsonbaty, 22646204. Repository: https://github.com/ammarsaad03/AML-Assignment2

- **Protocol A - official weights:** each model's official pretrained checkpoint, test only (5 models).
- **Protocol B - retrained under one protocol:** SCI, PairLIE, Retinexformer and RUAS trained on the LOL-v1 training
  set (450 pairs, 35 more held out for validation, seed 42), each with its own official architecture, loss, optimizer
  and learning rate. URetinex-Net has no official training code and is evaluated in Protocol A only.
- **Supplementary:** RUAS trained for 50,000 instead of 800 iterations (length fixed before any result).

All models are tested with the same preprocessing and the same metric code (PSNR, SSIM, LPIPS), with no ground truth
at inference, no GT-mean correction and no tuning on the test sets.

## Contents

| Item | Content |
|---|---|
| `report/AML_Assignment2_Report.docx` | Technical report |
| `scripts/`, `eval/`, `setup_repos.ps1` | Complete source code (wrappers, training, evaluation, tables, figures, analyses) |
| `configs/` | Per-model configuration (`<model>.yaml`: authors' settings + `protocol_B`), train/val split, Protocol B option files and lists |
| `results/` | Result tables (`results_table.md/.csv`), per-image and summary metrics, complexity, analyses |
| `figures/` | Qualitative grids, PSNR/SSIM bars, quality vs speed, training curves |
| `logs/` | All test logs (`<model>_<dataset>.log`) and training logs/curves (`train_<model>.log/.csv`) |
| `docs/protocol.md` | Fairness rules, per-model decisions, Protocol B deviations, supplementary RUAS run, analyses |
| `docs/sources.md` | Repositories + commits, weight files, dataset sources, data checks, DOIs |
| `docs/references.md` | Papers, repositories, data and weight links |
| `docs/experimental_setup.md` | Settings tables (both protocols) |
| `docs/code_changes.md` | No official repository file was modified; list of compatibility fixes and where they are |
| `environment.txt` | Exact software environment (versions, `pip freeze`, `conda list`, notes on fixes) |

Datasets, official repositories, weights and checkpoints are **not** included (redistribution not permitted / too
large); sections 2 and 3 give the official links and the steps to obtain them.

## 1. Environment

Tested on Windows 11, NVIDIA RTX 3060 12 GB (driver 610.88), 8 GB RAM, conda (Miniconda).

1. Create the environment (Python 3.10, PyTorch 2.7.1 + CUDA 11.8). I cloned an existing env
   (`conda create --name aml2 --clone AICV_PTCH`); from scratch, the equivalent is:
   ```powershell
   conda create -n aml2 python=3.10 -y
   D:\Programs\MiniConda\envs\aml2\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
   ```
2. **Install Pillow with pip, not conda.** A conda Pillow clashes with torchvision's bundled image DLLs on Windows
   ("DLL load failed while importing _imaging" after `import torchvision`).
3. Libraries (pip): `numpy scipy matplotlib pandas tqdm pyyaml opencv-python scikit-image lpips thop fvcore gdown
   einops timm natsort lmdb addict yapf`. `natsort lmdb addict yapf` are only needed so the official
   Retinexformer scripts can import `basicsr`; basicsr itself is **not** installed (the repo is put on `PYTHONPATH`).
4. **`OPENBLAS_NUM_THREADS=1`** is set automatically by `scripts/common.py` and `scripts/train_retinexformer.py`
   (see Known issues).
5. Exact versions: `environment.txt`.

`conda activate` may not work in PowerShell if script execution is disabled; every command below therefore calls the
environment's Python directly. Define it once per terminal (paths are those of my machine; adjust them):
```powershell
git clone https://github.com/ammarsaad03/AML-Assignment2 D:\AML_Assignment2
cd D:\AML_Assignment2
$py = "D:\Programs\MiniConda\envs\aml2\python.exe"
```

## 2. Code and weights

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_repos.ps1
```
Clones the five official repositories into `models/` at the exact commits, downloads the Retinexformer weights
(`weights/Retinexformer/LOL_v1.pth`, official Google Drive folder) and creates the `experiments` junction for basicsr.
SCI (`CVPR/weights/medium.pt`), PairLIE (`weights/PairLIE.pth`), URetinex-Net (`ckpt/*.pth`) and RUAS
(`ckpt/lol.pt`, `model/denoise.pt`) ship their weights inside the repositories. No repository file is modified.

## 3. Datasets (official links)

Download into `data/_downloads/` (details and checks in `docs/sources.md`):

| Dataset | Source | Archive |
|---|---|---|
| LOL-v1 (Wei et al., BMVC 2018) | Official page https://daooshee.github.io/BMVC2018website/ ; archive used: Google Drive link in the Retinexformer README, https://drive.google.com/file/d/1L-kqSQyrmMueBh_ziWoPFhfsAh50h20H | `LOLv1.zip` |
| LSRW (Hai et al., R2RNet, JVCIR 2023) | Official release https://github.com/JianghaiSCU/R2RNet ; archive used: Kaggle mirror https://www.kaggle.com/datasets/bjzh123/lsrw123 (structure and counts match the official release: Huawei 2480 / Nikon 3170 pairs) | `archive.zip` |
| PairLIE training set (Protocol B only) | PairLIE README: https://drive.google.com/file/d/1gM3QeNDOCzx0m1gpOoQD1TnGv1BELy08 | `PairLIE.zip` |

Extract and copy the test splits (PowerShell, from the project root):
```powershell
$d = "data\_downloads"
Expand-Archive "$d\LOLv1.zip" "$d\LOLv1_extracted"; Expand-Archive "$d\archive.zip" "$d\LSRW_extracted"
New-Item -ItemType Directory -Force data\LOL-v1\eval15\low, data\LOL-v1\eval15\high, data\LSRW\eval\low, data\LSRW\eval\high | Out-Null
Copy-Item "$d\LOLv1_extracted\LOLv1\Test\input\*"  data\LOL-v1\eval15\low
Copy-Item "$d\LOLv1_extracted\LOLv1\Test\target\*" data\LOL-v1\eval15\high
foreach ($cam in "Huawei", "Nikon") { foreach ($q in "low", "high") {
    Get-ChildItem "$d\LSRW_extracted\LSRW\Eval\Eval\$cam\$q" -File | ForEach-Object {
        Copy-Item $_.FullName ("data\LSRW\eval\$q\" + $cam.ToLower() + "_" + $_.Name) } } }
& $py scripts\check_data.py            # counts, pairing, sizes, brightness check
```
Protocol B data (LOL-v1 training set, 450/35 split with seed 42, pixel-hash overlap check against the test sets):
```powershell
& $py scripts\prepare_protocol_b.py
Expand-Archive "data\_downloads\PairLIE\PairLIE.zip" "data\_downloads\PairLIE\extracted"
& $py scripts\check_pairlie_sequences.py   # finds the LOL-derived PairLIE sequences, checks test overlap
```
The split lists (`configs/train_split.txt`, `configs/val_split.txt`) and the PairLIE sequence list
(`configs/protocol_B/pairlie_train_sequences.txt`, 128 sequences) are included.

## 4. Protocol A - test official weights

```powershell
foreach ($ds in "LOL-v1", "LSRW") {
    & $py scripts\run_sci.py --dataset $ds
    & $py scripts\run_pairlie.py --dataset $ds
    & $py scripts\run_uretinex.py --dataset $ds          # fixed ratio 5 (official evaluate.py reads the GT)
    & $py scripts\run_retinexformer.py --dataset $ds     # no GT-mean, no self-ensemble
    & $py scripts\run_ruas.py --dataset $ds              # official ckpt/lol.pt, output after noise removal
}
& $py scripts\evaluate_all.py --models SCI PairLIE URetinex-Net Retinexformer RUAS
```
Outputs: `results/<model>/<dataset>/*.png`, logs `logs/<model>_<dataset>.log`, per-image metrics
`results/<model>_<dataset>_metrics.csv`, summary `results/metrics_summary.csv` (incl. Huawei/Nikon subsets).

## 5. Protocol B - train and test

Run each training in its own terminal, one at a time (memory). All are resumable.

| Model | Command | Time (RTX 3060) | Checkpoint tested |
|---|---|---|---|
| PairLIE | `& $py scripts\train_pairlie.py` (resume: `--resume`) | 18 min 43 s | `checkpoints/PairLIE/final.pth` (final; unsupervised) |
| Retinexformer | `& $py scripts\train_retinexformer.py` (resume: same command) | 6 h 31 min | `checkpoints/Retinexformer/RetinexFormer_LOL_v1_protocolB/best_psnr_*.pth` (best validation PSNR) |
| SCI | `& $py scripts\train_sci.py` (resume: `--resume`) | 6 h 39 min (200 epochs) | `checkpoints/SCI/final.pt` (final; unsupervised) |
| RUAS | `& $py scripts\train_ruas.py` (resume: `--resume`) | 60 s (800 iterations) | `checkpoints/RUAS/final.pt` (final; unsupervised) |
| RUAS, supplementary | `& $py scripts\train_ruas.py --long` (resume: `--long --resume`) | 51 min (50,000 iterations) | `checkpoints/RUAS_long/final.pt` (final) |
| URetinex-Net | Protocol A only (no official training code) | - | - |

Training logs: `logs/train_<model>.log` and `logs/train_<model>.csv` (Retinexformer: basicsr's console log in
`logs/train_Retinexformer.log` and `metric.csv` in the experiment folder). The Retinexformer option file is generated
by `scripts/make_retinexformer_cfg.py` (only data paths, validation split, seed, workers and logging differ from the
official yml).

Test the Protocol B checkpoints with the same wrappers:
```powershell
foreach ($ds in "LOL-v1", "LSRW") {
    & $py scripts\run_sci.py --dataset $ds --protocol B
    & $py scripts\run_pairlie.py --dataset $ds --protocol B
    & $py scripts\run_retinexformer.py --dataset $ds --protocol B
    & $py scripts\run_ruas.py --dataset $ds --protocol B
    & $py scripts\run_ruas.py --dataset $ds --protocol B_long     # supplementary run
}
& $py scripts\evaluate_all.py --models SCI_B PairLIE_B Retinexformer_B RUAS_B RUAS_B_long
```

## 6. Tables, figures and analyses

```powershell
& $py scripts\measure_complexity.py       # Params, FLOPs at 256x256 (fvcore), runtime with warm-up -> results/complexity.csv
& $py scripts\make_results_table.py       # results/results_table.csv and .md (both protocols, best value marked)
foreach ($m in "PairLIE", "SCI", "Retinexformer", "RUAS", "RUAS_long") { & $py scripts\plot_train_curve.py --model $m }
& $py scripts\make_figures.py             # qualitative grids, PSNR/SSIM bars, quality vs speed, 2x2 training curves
& $py scripts\analysis_selection_bias.py  # Retinexformer: all checkpoints on LOL-v1 (effect of test-set selection)
& $py scripts\analysis_brightness.py      # mean output brightness of all models
& $py scripts\analysis_ruas_long.py       # RUAS 800 vs 50,000 iterations: validation PSNR, brightness, blown-out pixels
foreach ($d in @(@("LOL-v1","data\LOL-v1\eval15"), @("LSRW","data\LSRW\eval"))) {
    & $py eval\metrics.py --pred "$($d[1])\low" --gt "$($d[1])\high" --csv "results\input_baseline_$($d[0])_metrics.csv" }
```
Run these one at a time, with no training running.

## 7. Report

The report is `report/AML_Assignment2_Report.docx`. Every table and result number in it comes from the files in
`results/` and `logs/`, and every figure from `figures/`.

## Known issues (Windows, 8 GB RAM)

- **`OPENBLAS_NUM_THREADS=1`** is set by `scripts/common.py` and `scripts/train_retinexformer.py`. numpy in `aml2` uses
  OpenBLAS, which otherwise reserves buffers for all 12 CPU threads (~1.9 GB committed per Python process instead of
  ~0.5 GB); together with data-loader workers this exhausted the Windows commit limit (WinError 1455 "paging file is
  too small"). Results are unaffected (models run in torch; metrics re-checked identical).
- **Retinexformer data-loader workers = 2** (official 8; `num_worker_per_gpu` in
  `configs/protocol_B/RetinexFormer_LOL_v1_protocolB.yml`). Speed is unchanged (~0.144 s/iter). Even so, peak memory
  commit reached 89% of the limit with VS Code and a browser open: run long trainings from a standalone PowerShell
  window and close large programs.
- **numpy BLAS switched from MKL to OpenBLAS**: with MKL, any numpy matrix multiplication crashed Python (0xC06D007F),
  which made matplotlib unusable (see `environment.txt`).
- **basicsr "debug" names:** basicsr overrides print/validation/save frequencies for experiment names containing
  "debug", so short test runs are called `_trial`.
- **PyTorch >= 2.6 `torch.load`:** old checkpoints (URetinex-Net, basicsr training states) need `weights_only=False`;
  the wrappers pass it, and `train_retinexformer.py` sets `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` for basicsr.
- `conda activate` may fail in PowerShell when script execution is disabled; call the env's `python.exe` directly.

## References

1. L. Ma, T. Ma, R. Liu, X. Fan, Z. Luo, "Toward Fast, Flexible, and Robust Low-Light Image Enhancement," CVPR 2022.
   DOI: 10.1109/CVPR52688.2022.00555
2. Z. Fu, Y. Yang, X. Tu, Y. Huang, X. Ding, K.-K. Ma, "Learning a Simple Low-Light Image Enhancer from Paired
   Low-Light Instances," CVPR 2023. DOI: 10.1109/CVPR52729.2023.02131
3. W. Wu, J. Weng, P. Zhang, X. Wang, W. Yang, J. Jiang, "URetinex-Net: Retinex-based Deep Unfolding Network for
   Low-light Image Enhancement," CVPR 2022. DOI: 10.1109/CVPR52688.2022.00581
4. Y. Cai, H. Bian, J. Lin, H. Wang, R. Timofte, Y. Zhang, "Retinexformer: One-stage Retinex-based Transformer for
   Low-light Image Enhancement," ICCV 2023. DOI: 10.1109/ICCV51070.2023.01149
5. R. Liu, L. Ma, J. Zhang, X. Fan, Z. Luo, "Retinex-inspired Unrolling with Cooperative Prior Architecture Search for
   Low-light Image Enhancement," CVPR 2021. DOI: 10.1109/CVPR46437.2021.01042
6. C. Wei, W. Wang, W. Yang, J. Liu, "Deep Retinex Decomposition for Low-Light Enhancement," BMVC 2018.
   https://daooshee.github.io/BMVC2018website/
7. J. Hai, Z. Xuan, R. Yang, Y. Hao, F. Zou, F. Lin, S. Han, "R2RNet: Low-light Image Enhancement via Real-low to
   Real-normal Network," Journal of Visual Communication and Image Representation, vol. 90, 103712, 2023.
   DOI: 10.1016/j.jvcir.2022.103712
8. J. Cai, S. Gu, L. Zhang, "Learning a Deep Single Image Contrast Enhancer from Multi-Exposure Images," IEEE TIP 2018.
   DOI: 10.1109/TIP.2018.2794218
9. Z. Wang, A. C. Bovik, H. R. Sheikh, E. P. Simoncelli, "Image Quality Assessment: From Error Visibility to Structural
   Similarity," IEEE TIP 2004. DOI: 10.1109/TIP.2003.819861
10. R. Zhang, P. Isola, A. A. Efros, E. Shechtman, O. Wang, "The Unreasonable Effectiveness of Deep Features as a
    Perceptual Metric," CVPR 2018. DOI: 10.1109/CVPR.2018.00068

Full entries with page numbers: `docs/references.md`.

## Acknowledgements

**Model code and weights** (official repositories, used unmodified):
- SCI - https://github.com/vis-opt-group/SCI @ `f6f88fd73cd614dbeee17d61a0dbde3678b7e183`
- PairLIE - https://github.com/zhenqifu/PairLIE @ `3fd5c1586aa5f8f0736382de36d5bf46a4e208c2`
- URetinex-Net - https://github.com/AndersonYong/URetinex-Net @ `7ee2196e093b6a9ae5c20242e48f9d61021b6364`
- Retinexformer (incl. its BasicSR-based training code) - https://github.com/caiyuanhao1998/Retinexformer @ `1e9a0efce4b306b6701b824768370ff26066c32a`
- RUAS - https://github.com/KarelZhang/RUAS @ `f621bb19ceffbe27b2d9625bb806bc7cc60aea7b`
- Inspected only (not used for results): URetinex-Net-PLUS - https://github.com/AndersonYong/URetinex-Net-PLUS @ `94926c252a1d077504b12160ef7fb66ea1c10476`

**Datasets:** LOL (Wei et al., BMVC 2018); LSRW (Hai et al., JVCIR 2023; obtained via the Kaggle mirror
https://www.kaggle.com/datasets/bjzh123/lsrw123); the PairLIE training sequences (Fu et al., CVPR 2023), of which only
the LOL-derived ones were used.

**Libraries:** PyTorch, torchvision, scikit-image, lpips, fvcore, NumPy, matplotlib, Pillow, OpenCV, gdown, python-docx.

**AI tools:** Claude Code (Anthropic; model Claude Opus 5.5) was used as a coding assistant for writing the wrapper,
training, evaluation and plotting scripts and the documentation, and for checking the official code. All decisions
(protocols, fairness rules, deviations) were made and checked by the author; all full training runs were started by
the author, and all results were produced by running the scripts in this repository.

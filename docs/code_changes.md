# Code changes

**No official repository file was modified.** All five repositories are used exactly at the commits in
`docs/sources.md` (`git status` in each repo is clean apart from generated `__pycache__/` folders). Every
compatibility fix and every protocol setting is implemented in my own scripts in `scripts/`. Therefore there is no
`patches/` folder.

Official test scripts were run unmodified to cross-check each wrapper (outputs equal within 1/255 per pixel;
Retinexformer pixel-identical). Two of them needed a run-time setting only: PairLIE's `eval.py` hard-codes
`CUDA_VISIBLE_DEVICES='3'` (it was started through a small launcher that sets `'0'` in memory), and URetinex-Net's
`test.py` was run with the environment variable `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`.

## Compatibility fixes (PyTorch 2.7, Windows) and where they are

| Model | Problem in the official code | Fix (in my script) | Where |
|---|---|---|---|
| SCI | Official loader `Finetunemodel` silently skips checkpoint keys it cannot match (a wrong file would run with random weights). | Strict check that all 18 `enhance.*` tensors were loaded. | `scripts/run_sci.py`, `build_model()` |
| SCI | `train.py` hard-codes the data path as the string `'Your train dataset'`. | Own loop with the official `Network`, init, loss, optimizer and grad clipping; our data split. | `scripts/train_sci.py` |
| PairLIE | `eval.py` hard-codes `CUDA_VISIBLE_DEVICES='3'` and data paths. | Own test wrapper with the official `net` and the official `I = L^0.2 * R`. | `scripts/run_pairlie.py` |
| PairLIE | `dataset.py` hard-codes 324 sequences in folders `1..N`. | Own sampler over the LOL-derived sequences; same shared random 128x128 crop for both images. | `scripts/train_pairlie.py`, `sample_pair()` |
| URetinex-Net | Checkpoints store an `argparse.Namespace`; `torch.load` fails under PyTorch >= 2.6 (`weights_only=True`). | `torch.load(..., weights_only=False)` while the official `Inference` model is built. | `scripts/run_uretinex.py`, `torch_load_weights_only_false()` |
| URetinex-Net | `evaluate.py` computes the illumination ratio from the ground-truth image. | Not used; the official `test.py` pipeline with the fixed ratio 5. | `scripts/run_uretinex.py` |
| Retinexformer | Importing `basicsr.models` needs basicsr installed (`setup.py develop`). | The architecture file is loaded directly by path (no install). | `scripts/run_retinexformer.py`, `load_arch()` |
| Retinexformer | Old checkpoints / training states and `torch.load` defaults in PyTorch >= 2.6. | `weights_only=False` in the wrapper; `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` for basicsr training. | `scripts/run_retinexformer.py`, `scripts/train_retinexformer.py` |
| Retinexformer | Official option file validates on LOL-v1 Test; Windows data-loader too slow / too much memory. | Generated option file: our validation split, seed 42, 2 workers, `dataset_enlarge_ratio` 50, no tensorboard. | `scripts/make_retinexformer_cfg.py` -> `configs/protocol_B/RetinexFormer_LOL_v1_protocolB.yml` |
| Retinexformer | basicsr writes to `<repo>/experiments`. | Directory junction `models/Retinexformer/experiments` -> `checkpoints/Retinexformer`. | `scripts/train_retinexformer.py`, `setup_repos.ps1` |
| RUAS | `Network()` loads `./model/denoise.pt` relative to the working directory. | Model built with the repo as working directory. | `scripts/run_ruas.py`, `in_repo_dir()` |
| RUAS | `train.py` calls `next(iter(train_queue))` on a non-shuffled DataLoader: it always trains on the first image. | A random training image at every step, official 320x320 crop. | `scripts/train_ruas.py`, `random_crop()` / main loop |
| RUAS | `Network.optimizer()` crashes in PyTorch >= 1.5 ("modified by an inplace operation") at the first NRM update. | NRM update on a detached IEM output (identical NRM gradients). | `scripts/train_ruas.py`, `train_step()` |
| RUAS | `train.py` / `test.py` hard-code data paths, only read `*.png`, use Windows path splitting. | Own test wrapper and training loop with our paths and image I/O. | `scripts/run_ruas.py`, `scripts/train_ruas.py` |
| All | Some official scripts truncate when saving 8-bit images, others round. | One shared saving function (round) for every model. | `scripts/common.py`, `save_rgb01()` |
| All | numpy (OpenBLAS) reserves buffers for 12 threads per process (~1.9 GB), exhausting memory with data-loader workers on 8 GB RAM. | `OPENBLAS_NUM_THREADS=1`. | `scripts/common.py`, `scripts/train_retinexformer.py` |

Environment changes (not code) are listed at the top of `environment.txt` (Pillow from pip, numpy BLAS switched from
MKL to OpenBLAS).

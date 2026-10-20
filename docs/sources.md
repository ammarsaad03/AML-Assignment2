# Sources

All official repositories were cloned on 2026-10-17 into `models/` (not tracked by this repo's git).
To reproduce: `git clone <url> models/<name>` then `git -C models/<name> checkout <commit>`.

## Model repositories

| Model | Paper | Repository | Branch | Commit | Commit date |
|---|---|---|---|---|---|
| SCI | Ma et al., CVPR 2022, DOI 10.1109/CVPR52688.2022.00555 | https://github.com/vis-opt-group/SCI | main | `f6f88fd73cd614dbeee17d61a0dbde3678b7e183` | 2025-10-09 |
| PairLIE | Fu et al., CVPR 2023, DOI 10.1109/CVPR52729.2023.02131 | https://github.com/zhenqifu/PairLIE | main | `3fd5c1586aa5f8f0736382de36d5bf46a4e208c2` | 2023-07-31 |
| URetinex-Net | Wu et al., CVPR 2022, DOI 10.1109/CVPR52688.2022.00581 | https://github.com/AndersonYong/URetinex-Net | main | `7ee2196e093b6a9ae5c20242e48f9d61021b6364` | 2025-01-22 |
| Retinexformer | Cai et al., ICCV 2023, DOI 10.1109/ICCV51070.2023.01149 | https://github.com/caiyuanhao1998/Retinexformer | master | `1e9a0efce4b306b6701b824768370ff26066c32a` | 2026-05-23 |
| RUAS (added 2026-10-19) | Liu et al., CVPR 2021, DOI 10.1109/CVPR46437.2021.01042 | https://github.com/KarelZhang/RUAS | main | `f621bb19ceffbe27b2d9625bb806bc7cc60aea7b` | 2021-03-17 |

**DOIs.** The five model DOIs were provided by the report author and are **verified via IEEE/CVF bibliographic record**
(cross-checked on 2026-10-20 against the Crossref record: title, venue and authors match). Note that the IEEE
proceedings page numbers (attached to the DOI) differ from the CVF open-access page numbers in the authors' READMEs:
SCI 5627-5636 (CVF 5637-5646), PairLIE 22252-22261 (same), URetinex-Net 5891-5900 (CVF 5901-5910), Retinexformer
12470-12479 (CVF 12504-12513), RUAS 10556-10565.

Other DOIs used in `docs/references.md`, each verified against its Crossref record on 2026-10-20:
LSRW / R2RNet 10.1016/j.jvcir.2022.103712 (JVCIR vol. 90, article 103712, 2023); SSIM 10.1109/TIP.2003.819861;
LPIPS 10.1109/CVPR.2018.00068; SICE 10.1109/TIP.2018.2794218. LOL-v1 (BMVC 2018) has no DOI; it is cited with its
official URL https://daooshee.github.io/BMVC2018website/.

Inspected resource (Protocol B feasibility check only; cloned 2026-10-18, read-only, not used for any result):

| Repo | Repository | Branch | Commit | Commit date | Purpose |
|---|---|---|---|---|---|
| URetinex-Net-PLUS | https://github.com/AndersonYong/URetinex-Net-PLUS | main | `94926c252a1d077504b12160ef7fb66ea1c10476` | 2025-01-22 | Checked whether its training scripts can retrain URetinex-Net (they cannot for the adjustment stage; see `docs/protocol.md`). Not cited as a paper. |

## Pretrained weights

| Model | File(s) | Where | Trained on (per README / paper) |
|---|---|---|---|
| SCI | `CVPR/weights/easy.pt`, `medium.pt`, `difficult.pt` (43 KB each) | in repo | easy: mainly MIT-FiveK; medium: mainly LOL + LSRW; difficult: mainly DARK FACE. README notes these are *not* the exact models used for the paper's tables. |
| SCI++ (TPAMI 2025) | `TPAMI/weights/weights_1_3500.pt` (102 KB) | in repo | not documented |
| PairLIE | `weights/PairLIE.pth` (1.3 MB) | in repo | single model; README does not state training set (to verify against the paper) |
| URetinex-Net | `ckpt/init_low.pth`, `init_high.pth`, `unfolding.pth`, `L_adjust.pth` | in repo | LOL-v1 training set (paper) |
| Retinexformer | `LOL_v1.pth` (+ others) | Google Drive folder: https://drive.google.com/drive/folders/1ynK5hfQachzc8y96ZumhkPPDXzHJwaQV ; Baidu: https://pan.baidu.com/s/13zNqyKuxvLBiQunIxG_VhQ?pwd=cyh2 | LOL-v1 training set (485 pairs) |

## Datasets

Archives were downloaded manually by the user into `data/_downloads/` and extracted there (2026-10-17).
Only the test/eval splits were copied into `data/`. Verified with `scripts/check_data.py`.

| Dataset | Reference | Source used | Archive | Eval split | Copied to | Pairs | Image size (W x H) |
|---|---|---|---|---|---|---|---|
| LOL-v1 | Wei et al., "Deep Retinex Decomposition for Low-Light Enhancement", BMVC 2018 | Google Drive link from the Retinexformer README: https://drive.google.com/file/d/1L-kqSQyrmMueBh_ziWoPFhfsAh50h20H/view (official site: https://daooshee.github.io/BMVC2018website/) | `LOLv1.zip` | `LOLv1/Test/input` (low) and `LOLv1/Test/target` (high) = the standard eval15 | `data/LOL-v1/eval15/{low,high}` | 15 | 600x400 (all 15) |
| LSRW | Hai et al., "R2RNet: Low-light image enhancement via Real-low to Real-normal Network", JVCIR 2023, DOI 10.1016/j.jvcir.2022.103712 | Kaggle mirror of the official release: https://www.kaggle.com/datasets/bjzh123/lsrw123 (official repo: https://github.com/JianghaiSCU/R2RNet) | `archive.zip` | `LSRW/Eval/Eval/{Huawei,Nikon}/{low,high}` | `data/LSRW/eval/{low,high}`, files prefixed `huawei_` / `nikon_` | 50 (Huawei 30 + Nikon 20) | Huawei 960x720 (30), Nikon 960x640 (20) |

Notes:
- LSRW was obtained from a Kaggle mirror, not directly from the authors' link. Its structure matches the official
  release: training split Huawei 2450 + Nikon 3150 pairs, Eval split Huawei 30 + Nikon 20 pairs, i.e. 2480 Huawei and
  3170 Nikon pairs in total, as reported for LSRW.
- Checks passed for both datasets: equal low/high counts, every low image has a same-named high image, same size per
  pair, all RGB, and every low image is darker than its high image.
- SCI's own LSRW link (random 500/50 split) was **not** used.

### Protocol B training data

| Data | Source | Used |
|---|---|---|
| LOL-v1 training set (485 pairs) | `LOLv1.zip` above, `LOLv1/Train/{input,target}` -> `data/LOL-v1/train/{low,high}` | 450 pairs for training (`configs/train_split.txt`), 35 held out for validation (`configs/val_split.txt`, random, seed 42). Pixel-hash check: no image equals any LOL-v1 eval15 or LSRW eval image. |
| PairLIE authors' training set | `PairLIE.zip` (194 MB), Google Drive link in the PairLIE README: https://drive.google.com/file/d/1gM3QeNDOCzx0m1gpOoQD1TnGv1BELy08 (downloaded with gdown, 2026-10-18) -> `data/_downloads/PairLIE/` | Only the LOL-derived sequences: 145 of 324 sequences consist entirely of LOL-v1 training low images (pixel-hash match; the other 179 are SICE). After removing our 35 validation images: 128 sequences, 318 images (`configs/protocol_B/pairlie_train_sequences.txt`). No image matches any test image. Checked with `scripts/check_pairlie_sequences.py`. |

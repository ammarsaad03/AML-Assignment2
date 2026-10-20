# References

DOI policy: every DOI below is recorded in `docs/sources.md`. The five model DOIs were provided by the report author
(verified via IEEE/CVF bibliographic record); all DOIs were cross-checked against their Crossref records on 2026-10-20
(title, venue, authors). Page numbers are those of the DOI (IEEE) record; the CVF open-access versions use different
page numbers (see `docs/sources.md`). LOL-v1 (BMVC 2018) has no DOI and is cited with its official URL.

## Models (papers)

1. **SCI** - L. Ma, T. Ma, R. Liu, X. Fan, Z. Luo. "Toward Fast, Flexible, and Robust Low-Light Image Enhancement."
   *2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 5627-5636, 2022.
   DOI: 10.1109/CVPR52688.2022.00555

2. **PairLIE** - Z. Fu, Y. Yang, X. Tu, Y. Huang, X. Ding, K.-K. Ma. "Learning a Simple Low-Light Image Enhancer from
   Paired Low-Light Instances." *2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*,
   pp. 22252-22261, 2023. DOI: 10.1109/CVPR52729.2023.02131

3. **URetinex-Net** - W. Wu, J. Weng, P. Zhang, X. Wang, W. Yang, J. Jiang. "URetinex-Net: Retinex-based Deep Unfolding
   Network for Low-light Image Enhancement." *2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition
   (CVPR)*, pp. 5891-5900, 2022. DOI: 10.1109/CVPR52688.2022.00581

4. **Retinexformer** - Y. Cai, H. Bian, J. Lin, H. Wang, R. Timofte, Y. Zhang. "Retinexformer: One-stage Retinex-based
   Transformer for Low-light Image Enhancement." *2023 IEEE/CVF International Conference on Computer Vision (ICCV)*,
   pp. 12470-12479, 2023. DOI: 10.1109/ICCV51070.2023.01149

5. **RUAS** - R. Liu, L. Ma, J. Zhang, X. Fan, Z. Luo. "Retinex-inspired Unrolling with Cooperative Prior Architecture
   Search for Low-light Image Enhancement." *2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition
   (CVPR)*, pp. 10556-10565, 2021. DOI: 10.1109/CVPR46437.2021.01042

## Datasets

6. **LOL (LOL-v1)** - C. Wei, W. Wang, W. Yang, J. Liu. "Deep Retinex Decomposition for Low-Light Enhancement."
   *British Machine Vision Conference (BMVC)*, 2018. No DOI; official page: https://daooshee.github.io/BMVC2018website/

7. **LSRW** - J. Hai, Z. Xuan, R. Yang, Y. Hao, F. Zou, F. Lin, S. Han. "R2RNet: Low-light Image Enhancement via
   Real-low to Real-normal Network." *Journal of Visual Communication and Image Representation*, vol. 90, 103712, 2023.
   DOI: 10.1016/j.jvcir.2022.103712. Official code and data: https://github.com/JianghaiSCU/R2RNet; data obtained via
   the Kaggle mirror https://www.kaggle.com/datasets/bjzh123/lsrw123.

8. **SICE** (source of part of the official PairLIE training set; only the LOL-derived sequences were used in
   Protocol B) - J. Cai, S. Gu, L. Zhang. "Learning a Deep Single Image Contrast Enhancer from Multi-Exposure Images."
   *IEEE Transactions on Image Processing*, 27(4):2049-2062, 2018. DOI: 10.1109/TIP.2018.2794218

## Metrics

9. **SSIM** - Z. Wang, A. C. Bovik, H. R. Sheikh, E. P. Simoncelli. "Image Quality Assessment: From Error Visibility to
   Structural Similarity." *IEEE Transactions on Image Processing*, 13(4):600-612, 2004. DOI: 10.1109/TIP.2003.819861

10. **LPIPS** - R. Zhang, P. Isola, A. A. Efros, E. Shechtman, O. Wang. "The Unreasonable Effectiveness of Deep Features
    as a Perceptual Metric." *2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 586-595,
    2018. DOI: 10.1109/CVPR.2018.00068. Code: https://github.com/richzhang/PerceptualSimilarity (pip `lpips`, v0.1,
    AlexNet).

## Code repositories (with the commits used)

| Repository | URL | Commit |
|---|---|---|
| SCI (official) | https://github.com/vis-opt-group/SCI | `f6f88fd73cd614dbeee17d61a0dbde3678b7e183` |
| PairLIE (official) | https://github.com/zhenqifu/PairLIE | `3fd5c1586aa5f8f0736382de36d5bf46a4e208c2` |
| URetinex-Net (official) | https://github.com/AndersonYong/URetinex-Net | `7ee2196e093b6a9ae5c20242e48f9d61021b6364` |
| Retinexformer (official, incl. BasicSR-based training) | https://github.com/caiyuanhao1998/Retinexformer | `1e9a0efce4b306b6701b824768370ff26066c32a` |
| RUAS (official) | https://github.com/KarelZhang/RUAS | `f621bb19ceffbe27b2d9625bb806bc7cc60aea7b` |
| URetinex-Net-PLUS (inspected only, not used for results) | https://github.com/AndersonYong/URetinex-Net-PLUS | `94926c252a1d077504b12160ef7fb66ea1c10476` |
| fvcore (FLOP counting) | https://github.com/facebookresearch/fvcore | pip `fvcore 0.1.5.post20221221` |

## Data and weight links

| Item | Link |
|---|---|
| LOL-v1 archive used | https://drive.google.com/file/d/1L-kqSQyrmMueBh_ziWoPFhfsAh50h20H (from the Retinexformer README) |
| LSRW archive used | https://www.kaggle.com/datasets/bjzh123/lsrw123 (mirror of the official release) |
| PairLIE training set | https://drive.google.com/file/d/1gM3QeNDOCzx0m1gpOoQD1TnGv1BELy08 (from the PairLIE README) |
| Retinexformer pretrained weights | https://drive.google.com/drive/folders/1ynK5hfQachzc8y96ZumhkPPDXzHJwaQV (from the Retinexformer README) |
| SCI / PairLIE / URetinex-Net / RUAS weights | inside the official repositories (`CVPR/weights/`, `weights/`, `ckpt/`, `ckpt/` + `model/`) |

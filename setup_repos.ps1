# Clone the official model repositories at the exact commits used in this project (see docs/sources.md),
# and download the Retinexformer pretrained weights (Google Drive, via gdown).
#
# Usage (from the project root, in Windows PowerShell):
#     powershell -ExecutionPolicy Bypass -File .\setup_repos.ps1
# Requires: git on PATH, and the conda env "aml2" (for gdown).

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
$Python = "D:\Programs\MiniConda\envs\aml2\python.exe"   # change if your env lives elsewhere

$Repos = @(
    @{ Name = "SCI";           Url = "https://github.com/vis-opt-group/SCI";            Commit = "f6f88fd73cd614dbeee17d61a0dbde3678b7e183" },
    @{ Name = "PairLIE";       Url = "https://github.com/zhenqifu/PairLIE";             Commit = "3fd5c1586aa5f8f0736382de36d5bf46a4e208c2" },
    @{ Name = "URetinex-Net";  Url = "https://github.com/AndersonYong/URetinex-Net";    Commit = "7ee2196e093b6a9ae5c20242e48f9d61021b6364" },
    @{ Name = "Retinexformer"; Url = "https://github.com/caiyuanhao1998/Retinexformer"; Commit = "1e9a0efce4b306b6701b824768370ff26066c32a" },
    @{ Name = "RUAS";          Url = "https://github.com/KarelZhang/RUAS";              Commit = "f621bb19ceffbe27b2d9625bb806bc7cc60aea7b" }
)

New-Item -ItemType Directory -Force "$Root\models" | Out-Null
foreach ($r in $Repos) {
    $dir = Join-Path "$Root\models" $r.Name
    if (-not (Test-Path $dir)) {
        Write-Host "Cloning $($r.Name) ..."
        git clone $r.Url $dir
    }
    git -C $dir checkout --quiet $r.Commit
    Write-Host ("{0,-14} at {1}" -f $r.Name, (git -C $dir rev-parse HEAD))
}

# Retinexformer pretrained weights (official Google Drive folder from the Retinexformer README).
# Only LOL_v1.pth is used; gdown downloads the whole folder (~67 MB).
$w = "$Root\weights\Retinexformer"
if (-not (Test-Path "$w\LOL_v1.pth")) {
    New-Item -ItemType Directory -Force $w | Out-Null
    & $Python -m gdown --folder "https://drive.google.com/drive/folders/1ynK5hfQachzc8y96ZumhkPPDXzHJwaQV" -O $w
}
Write-Host "Retinexformer LOL_v1.pth present: $(Test-Path "$w\LOL_v1.pth")"

# basicsr (Retinexformer) writes training output to <repo>/experiments; point it at checkpoints/Retinexformer.
$exp = "$Root\models\Retinexformer\experiments"
if (-not (Test-Path $exp)) {
    New-Item -ItemType Directory -Force "$Root\checkpoints\Retinexformer" | Out-Null
    New-Item -ItemType Junction -Path $exp -Target "$Root\checkpoints\Retinexformer" | Out-Null
}
Write-Host "Done. SCI, PairLIE and URetinex-Net weights are inside their repos."

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $env:USERPROFILE ".conda\envs\minimind\python.exe"

Set-Location $Root

if (-not (Test-Path $Python)) {
    throw "Cannot find minimind Python: $Python"
}

& $Python -m PyInstaller --version | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller is not installed in minimind. Install it first: conda activate minimind; pip install pyinstaller"
}

& $Python -m PyInstaller `
    --noconfirm `
    --clean `
    --onedir `
    --name BAIT_GUI `
    --exclude-module gradio `
    --exclude-module fastapi `
    --exclude-module uvicorn `
    --exclude-module starlette `
    --exclude-module datasets `
    --exclude-module transformers `
    --exclude-module tokenizers `
    --exclude-module torchvision `
    --exclude-module torchaudio `
    --exclude-module cv2 `
    --exclude-module sklearn `
    --exclude-module pandas `
    --exclude-module pyarrow `
    --exclude-module IPython `
    --exclude-module jupyter `
    --exclude-module notebook `
    --exclude-module pydub `
    --exclude-module librosa `
    --exclude-module soundfile `
    --exclude-module tensorflow `
    --add-data "checkpoints_multi;checkpoints_multi" `
    --add-data "example_cases;example_cases" `
    --add-data "DATA_INTERFACE.md;." `
    GUI.py

$DistRoot = Join-Path $Root "dist\BAIT_GUI"
Copy-Item -Path (Join-Path $Root "example_cases") -Destination (Join-Path $DistRoot "example_cases") -Recurse -Force
Copy-Item -Path (Join-Path $Root "DATA_INTERFACE.md") -Destination (Join-Path $DistRoot "DATA_INTERFACE.md") -Force

$ZipPath = Join-Path $Root "dist\BAIT_GUI.zip"
if (Test-Path $ZipPath) {
    Remove-Item $ZipPath -Force
}
Compress-Archive -Path $DistRoot -DestinationPath $ZipPath -Force

Write-Host "Packaged to: $Root\dist\BAIT_GUI"
Write-Host "Zip created: $ZipPath"

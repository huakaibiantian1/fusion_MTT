$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$OutDir = Join-Path $Root "release\BAIT_GUI_source"
$ZipPath = Join-Path $Root "release\BAIT_GUI_source.zip"

Set-Location $Root

if (Test-Path $OutDir) {
    Remove-Item $OutDir -Recurse -Force
}
if (Test-Path $ZipPath) {
    Remove-Item $ZipPath -Force
}

New-Item -ItemType Directory -Path $OutDir | Out-Null

$Files = @(
    "GUI.py",
    "bait_data_io.py",
    "run_bait_file.py",
    "bait_model.py",
    "evaluate_3d.py",
    "kf_bait_tracker.py",
    "data_generation.py",
    "data_generation_with_crossing.py",
    "data_generation_multi_scenario.py",
    "metrics.py",
    "requirements_GUI.txt",
    "DATA_INTERFACE.md",
    "run_GUI.bat"
)

foreach ($File in $Files) {
    Copy-Item -Path (Join-Path $Root $File) -Destination (Join-Path $OutDir $File) -Force
}

New-Item -ItemType Directory -Path (Join-Path $OutDir "checkpoints_multi") | Out-Null
Copy-Item `
    -Path (Join-Path $Root "checkpoints_multi\best_model.pth") `
    -Destination (Join-Path $OutDir "checkpoints_multi\best_model.pth") `
    -Force
Copy-Item -Path (Join-Path $Root "example_cases") -Destination (Join-Path $OutDir "example_cases") -Recurse -Force

Compress-Archive -Path $OutDir -DestinationPath $ZipPath -Force

Write-Host "Source package created: $ZipPath"

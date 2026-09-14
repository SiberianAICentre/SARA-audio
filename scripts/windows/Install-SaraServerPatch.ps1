[CmdletBinding()]
param(
    [string]$ProjectRoot = (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)),
    [string]$PatchRoot = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path $ProjectRoot).Path
if (-not $PatchRoot) {
    $PatchRoot = Split-Path -Parent $PSScriptRoot
    $PatchRoot = Split-Path -Parent $PatchRoot
}
$PatchRoot = (Resolve-Path $PatchRoot).Path

$required = @(
    "pyproject.toml",
    "requirements-cuda11.txt",
    "requirements-cuda12.txt",
    "requirements-cuda12-native.txt",
    "config\rtx4090_cuda11_8.yaml",
    "config\rtx4090_cuda12.yaml",
    "config\native_cpu.yaml",
    "json_string_filter.json",
    "models\methodology\filter.json",
    "models\methodology\participation_modal_n.json",
    "models\methodology\greeting_flg.json",
    "models\methodology\farewell_flg.json",
    "models\methodology\etiquette_present_flg.json",
    "models\methodology\etiquette_absent_flg.json",
    "models\methodology\participation_dominant_flg.json",
    "models\methodology\distancing_dominant_flg.json",
    "models\methodology\clarifying_absent_flg.json",
    "models\methodology\agent_sex.json",
    "models\methodology\burnout.json",
    "scripts\windows\Invoke-SaraServerPipeline.ps1",
    "scripts\windows\Install-SaraCuda12Native.ps1",
    "scripts\windows\Trace-SaraPipeline.ps1",
    "scripts\windows\New-SaraWheelhouse.ps1"
)

foreach ($relativePath in $required) {
    $source = Join-Path $PatchRoot $relativePath
    if (-not (Test-Path $source)) {
        throw "Patch file is missing: $relativePath"
    }
}
$sourcePackage = Join-Path $PatchRoot "src\sara_audio"
if (-not (Test-Path $sourcePackage)) {
    throw "Patch source package is missing: src\sara_audio"
}

Write-Host "Installing SARA server patch into $ProjectRoot" -ForegroundColor Cyan
foreach ($relativePath in $required) {
    $source = Join-Path $PatchRoot $relativePath
    $destination = Join-Path $ProjectRoot $relativePath
    $parent = Split-Path -Parent $destination
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
    Copy-Item -LiteralPath $source -Destination $destination -Force
}
$destinationSourceRoot = Join-Path $ProjectRoot "src"
New-Item -ItemType Directory -Force -Path $destinationSourceRoot | Out-Null
Copy-Item -LiteralPath $sourcePackage -Destination $destinationSourceRoot -Recurse -Force
$dockerSource = Join-Path $PatchRoot "docker"
if (Test-Path $dockerSource) {
    Copy-Item -LiteralPath $dockerSource -Destination $ProjectRoot -Recurse -Force
}

Write-Host "Patch installed." -ForegroundColor Green
Write-Host "Existing Python and CUDA dependencies were not changed." -ForegroundColor Cyan
Write-Host "Restart sara-web so it loads the updated interface and prediction code." -ForegroundColor Cyan

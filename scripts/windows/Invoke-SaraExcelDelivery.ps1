[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$SourcePath,
    [string]$DestinationPath
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = Join-Path $ProjectRoot ".server-venv\Scripts\python.exe"
$Wheelhouse = Join-Path $ProjectRoot "wheelhouse-cuda118"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Server environment not found: $Python"
}
if (-not (Test-Path -LiteralPath $Wheelhouse)) {
    throw "CUDA 11.8 wheelhouse not found: $Wheelhouse"
}

$SourcePath = (Resolve-Path -LiteralPath $SourcePath).Path
if (-not $DestinationPath) {
    $DestinationPath = Join-Path (Split-Path -Parent $SourcePath) "$(Split-Path -Leaf $SourcePath)-delivery-xlsx"
}

Write-Host "[1/2] Installing the Excel delivery update..." -ForegroundColor Cyan
& $Python -m pip install --force-reinstall --no-index --find-links $Wheelhouse --no-deps "sara-audio==0.1.0"
if ($LASTEXITCODE -ne 0) {
    throw "Could not install the bundled SARA delivery update."
}

Write-Host "[2/2] Creating XLSX delivery from existing results (no audio processing)..." -ForegroundColor Cyan
& $Python -m sara_audio.server_delivery $SourcePath $DestinationPath
if ($LASTEXITCODE -ne 0) {
    throw "Excel delivery creation failed."
}

Write-Host "Finished. XLSX delivery: $DestinationPath" -ForegroundColor Green

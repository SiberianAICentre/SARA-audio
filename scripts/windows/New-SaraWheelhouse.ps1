[CmdletBinding()]
param(
    [string]$ProjectRoot = (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)),
    [string]$Python = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path $ProjectRoot).Path
if (-not $Python) {
    $candidate = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
    if (Test-Path $candidate) {
        $Python = $candidate
    } elseif (Get-Command py -ErrorAction SilentlyContinue) {
        $Python = "py"
    } else {
        throw "Use -Python with a Python 3.11 executable."
    }
}

$wheelhouse = Join-Path $ProjectRoot "wheelhouse"
New-Item -ItemType Directory -Force -Path $wheelhouse | Out-Null
$pythonArguments = if ($Python -eq "py") { @("-3.11") } else { @() }

Write-Host "[1/2] Building the SARA project wheel..." -ForegroundColor Cyan
& $Python @pythonArguments -m pip wheel --wheel-dir $wheelhouse --no-deps $ProjectRoot
if ($LASTEXITCODE -ne 0) {
    throw "Could not build the SARA project wheel."
}

Write-Host "[2/2] Building Windows Python 3.11 dependency wheels..." -ForegroundColor Cyan
& $Python @pythonArguments -m pip wheel --wheel-dir $wheelhouse `
    "hatchling>=1.25" `
    "imageio-ffmpeg>=0.5,<1" `
    "numpy>=1.26,<3" `
    "opensmile>=2.5,<3" `
    "pandas>=2.2,<3" `
    "PyYAML>=6.0,<7" `
    "faster-whisper==1.2.1" `
    "ctranslate2==4.8.2" `
    "tokenizers==0.22.2" `
    "natasha>=1.6,<2" `
    "pyarrow>=16,<24" `
    "torch>=2,<3" `
    "transformers>=4.50,<5" `
    "gradio>=5,<6"
if ($LASTEXITCODE -ne 0) {
    throw "Could not download all dependency wheels."
}
Write-Host "Finished. Copy this directory to the server: $wheelhouse" -ForegroundColor Green

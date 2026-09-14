[CmdletBinding()]
param(
    [string]$ProjectRoot = (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)),
    [string]$VenvPath = "",
    [string]$SmokeInput = "",
    [switch]$LaunchWeb
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path $ProjectRoot).Path
if (-not $VenvPath) {
    $VenvPath = Join-Path $ProjectRoot ".cuda12-venv-final"
}
$VenvPath = [System.IO.Path]::GetFullPath($VenvPath)
$python = Join-Path $VenvPath "Scripts\python.exe"
$requirements = Join-Path $ProjectRoot "requirements-cuda12-native.txt"
$config = Join-Path $ProjectRoot "config\rtx4090_cuda12.yaml"

if (-not (Test-Path $requirements)) {
    throw "Requirements file was not found: $requirements"
}
if (-not (Test-Path $config)) {
    throw "CUDA 12 config was not found: $config"
}
if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "Python launcher 'py' was not found. Install Python 3.11 first."
}
if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    throw "nvidia-smi was not found. Install the NVIDIA driver first."
}
if (Test-Path $python) {
    throw "The target environment already exists: $VenvPath. Choose a new -VenvPath."
}

Write-Host "Creating clean native CUDA 12 environment: $VenvPath" -ForegroundColor Cyan
& py -3.11 -m venv $VenvPath
if ($LASTEXITCODE -ne 0) {
    throw "Could not create virtual environment: $VenvPath"
}

Write-Host "Installing pinned Python dependencies..." -ForegroundColor Cyan
& $python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "Could not upgrade pip."
}
& $python -m pip install --requirement $requirements
if ($LASTEXITCODE -ne 0) {
    throw "Pinned dependency installation failed."
}

# CTranslate2 4.4.0 speech models use cuDNN 8 on CUDA 12.x. These packages
# are installed without dependency resolution to avoid an unrelated upgrade.
$cudaRuntimePackages = @(
    "nvidia-cublas-cu12==12.4.2.65",
    "nvidia-cuda-nvrtc-cu12==12.4.127",
    "nvidia-cudnn-cu12==8.9.7.29"
)
Write-Host "Installing pinned CUDA 12.4/cuDNN 8 runtime..." -ForegroundColor Cyan
& $python -m pip install --force-reinstall --no-deps @cudaRuntimePackages
if ($LASTEXITCODE -ne 0) {
    throw "Could not install the pinned CUDA runtime."
}

# Text embeddings remain enabled, but run on CPU. Whisper ASR remains on CUDA.
Write-Host "Installing CPU-only PyTorch for text embeddings..." -ForegroundColor Cyan
& $python -m pip install --force-reinstall --no-deps `
    --index-url "https://download.pytorch.org/whl/cpu" "torch==2.5.1"
if ($LASTEXITCODE -ne 0) {
    throw "Could not install CPU-only PyTorch."
}

# The CPU wheel is intentionally installed without dependencies so pip cannot
# replace the pinned CUDA runtime. Install its required symbolic-math runtime
# explicitly and verify the model class used by ru-en-RoSBERTa.
& $python -m pip install "sympy==1.13.1"
if ($LASTEXITCODE -ne 0) {
    throw "Could not install the PyTorch symbolic-math dependency."
}
& $python -c "import sympy; from transformers.models.roberta.modeling_roberta import RobertaModel; print('RoBERTa runtime OK')"
if ($LASTEXITCODE -ne 0) {
    throw "RoBERTa text-embedding runtime validation failed."
}

$sitePackages = Join-Path $VenvPath "Lib\site-packages"
$runtimeDirectories = @(
    (Join-Path $sitePackages "nvidia\cublas\bin"),
    (Join-Path $sitePackages "nvidia\cudnn\bin"),
    (Join-Path $sitePackages "nvidia\cuda_nvrtc\bin")
) | Where-Object { Test-Path $_ }
$env:PATH = (($runtimeDirectories + @($env:PATH)) -join [System.IO.Path]::PathSeparator)

Write-Host "Running CUDA ASR load diagnostic..." -ForegroundColor Cyan
& (Join-Path $VenvPath "Scripts\sara-diagnose-asr.exe") --config $config --load-model
if ($LASTEXITCODE -ne 0) {
    throw "CUDA ASR model-load diagnostic failed."
}

if ($SmokeInput) {
    $smokeInput = (Resolve-Path $SmokeInput).Path
    $output = Join-Path $ProjectRoot ("results\cuda12-smoke-" + (Get-Date -Format "yyyyMMdd-HHmmss"))
    Write-Host "Running one-file end-to-end smoke test..." -ForegroundColor Cyan
    & (Join-Path $VenvPath "Scripts\sara-extract.exe") $smokeInput $output --config $config
    $extractExitCode = $LASTEXITCODE
    if ($extractExitCode -ne 0) {
        throw "End-to-end smoke test failed with exit code $extractExitCode."
    }
}

if ($LaunchWeb) {
    & (Join-Path $VenvPath "Scripts\sara-web.exe") `
        --config $config `
        --output-root (Join-Path $ProjectRoot "results\web") `
        --host "0.0.0.0" `
        --port 7860
}

Write-Host "Native CUDA 12 environment is ready: $VenvPath" -ForegroundColor Green

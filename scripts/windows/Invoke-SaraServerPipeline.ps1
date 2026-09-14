[CmdletBinding()]
param(
    [string]$InputPath = "",

    [string]$SmokeInputPath = "",

    [string]$OutputPath = "",

    [string]$ProjectRoot = (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)),

    [string]$CudaLibraryPath = "",

    [string]$WheelhousePath = "",

    [ValidateSet("cuda11_8", "cuda12")]
    [string]$CudaConfig = "cuda11_8",

    [switch]$Offline
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path $ProjectRoot).Path
$configPath = Join-Path $ProjectRoot "config\rtx4090_$CudaConfig.yaml"
if (-not (Test-Path $configPath)) {
    throw "Server config was not found: $configPath"
}
$predictionModelPath = Join-Path $ProjectRoot "json_string_filter.json"
if (-not (Test-Path $predictionModelPath)) {
    Write-Warning "json_string_filter.json is missing. Extraction can run, but prediction artifacts will be unavailable."
}
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
Write-Host "[1/5] Resolving input and CUDA environment..." -ForegroundColor Cyan
if (-not $InputPath) {
    $newCorpusRoot = Join-Path $ProjectRoot "data\new_raw_data\2_file"
    if (Test-Path (Join-Path $newCorpusRoot "test.zip")) {
        $InputPath = $newCorpusRoot
    } else {
        $InputPath = Join-Path $ProjectRoot "data\raw"
    }
}
if (-not $OutputPath) {
    $OutputPath = Join-Path $ProjectRoot "results\server-$timestamp"
}
$InputPath = (Resolve-Path $InputPath).Path

if (-not $CudaLibraryPath -and (Test-Path (Join-Path $ProjectRoot "cuda-libs"))) {
    $CudaLibraryPath = Join-Path $ProjectRoot "cuda-libs"
}
if ($CudaLibraryPath) {
    if (-not (Test-Path $CudaLibraryPath)) {
        throw "Specified CUDA DLL directory does not exist: $CudaLibraryPath"
    }
    $env:SARA_CUDA_LIB_DIR = (Resolve-Path $CudaLibraryPath).Path
    $env:PATH = "$env:SARA_CUDA_LIB_DIR;$env:PATH"
}
if (-not $env:CUDA_PATH) {
    $cudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    $cudaToolkit = Get-ChildItem -Path $cudaRoot -Directory -ErrorAction SilentlyContinue |
        Sort-Object Name -Descending |
        Select-Object -First 1
    if ($cudaToolkit -and (Test-Path (Join-Path $cudaToolkit.FullName "bin"))) {
        $env:CUDA_PATH = $cudaToolkit.FullName
        $env:PATH = "$(Join-Path $cudaToolkit.FullName 'bin');$env:PATH"
    }
}
if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    throw "nvidia-smi is unavailable. Install the NVIDIA driver before running SARA."
}

if (-not $SmokeInputPath) {
    $candidate = Get-ChildItem -Path $InputPath -Recurse -File |
        Where-Object { $_.Extension.ToLowerInvariant() -in ".wav", ".m4a", ".mp3", ".flac", ".ogg" } |
        Sort-Object FullName |
        Select-Object -First 1
    if (-not $candidate) {
        throw "No supported audio file found under $InputPath."
    }
    $SmokeInputPath = Join-Path $ProjectRoot ".server-smoke-input-$timestamp"
    New-Item -ItemType Directory -Path $SmokeInputPath | Out-Null
    Copy-Item -LiteralPath $candidate.FullName -Destination $SmokeInputPath
}
$SmokeInputPath = (Resolve-Path $SmokeInputPath).Path
Write-Host "[2/5] Creating the isolated server Python environment..." -ForegroundColor Cyan
$serverVenv = [System.IO.Path]::Combine($ProjectRoot, ".server-venv")
$venvPython = [System.IO.Path]::Combine($serverVenv, "Scripts", "python.exe")
if (-not (Test-Path $venvPython)) {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3.11 -m venv $serverVenv
        if ($LASTEXITCODE -ne 0) {
            throw "Could not create the server Python environment: $serverVenv"
        }
    } else {
        throw "Python 3.11 launcher 'py' is required to create the server environment."
    }
}

$env:PIP_DISABLE_PIP_VERSION_CHECK = "1"
$env:PIP_DEFAULT_TIMEOUT = "600"
Write-Host "[3/5] Installing project dependencies (first run may take several minutes)..." -ForegroundColor Cyan
if ($Offline) {
    $wheelhouse = if ($WheelhousePath) { $WheelhousePath } else { Join-Path $ProjectRoot "wheelhouse" }
    if (-not (Test-Path $wheelhouse)) {
        throw "Offline mode requires $wheelhouse."
    }
    $wheelhouse = (Resolve-Path $wheelhouse).Path
    & $venvPython -m pip install --force-reinstall --no-index --find-links $wheelhouse --no-deps "sara-audio==0.1.0"
    if ($LASTEXITCODE -ne 0) {
        throw "Could not install the bundled SARA package."
    }
    $profilePackages = if ($CudaConfig -eq "cuda11_8") {
        @(
            "faster-whisper==0.10.1",
            "ctranslate2==3.24.0",
            "tokenizers==0.15.2",
            "transformers==4.38.2",
            "huggingface-hub>=0.13,<1",
            "nvidia-cublas-cu11==11.11.3.6",
            "nvidia-cuda-nvrtc-cu11==11.8.89",
            "nvidia-cudnn-cu11==8.9.5.29"
        )
    } else {
        @(
            "faster-whisper==1.2.1",
            "ctranslate2==4.8.2",
            "tokenizers==0.22.2",
            "transformers>=4.50,<5",
            "huggingface-hub>=0.34,<1"
        )
    }
    & $venvPython -m pip install --no-index --find-links $wheelhouse --no-deps @profilePackages
    if ($LASTEXITCODE -ne 0) {
        throw "Could not install the selected server ASR profile: $CudaConfig"
    }
    & $venvPython -m pip install --no-index --find-links $wheelhouse `
        "imageio-ffmpeg>=0.5,<1" "numpy>=1.26,<3" "opensmile>=2.5,<3" `
        "pandas>=2.2,<3" "PyYAML>=6.0,<7" "av>=11" "onnxruntime<2,>=1.14" `
        "requests>=2.31" "tqdm" "natasha>=1.6,<2" "pyarrow>=16,<24" `
        "torch>=2,<3" "gradio>=5,<6"
} else {
    if ($CudaConfig -eq "cuda11_8") {
        & $venvPython -m pip install --timeout 600 --retries 3 -e "${ProjectRoot}[nlp,parquet,web]"
        & $venvPython -m pip install --timeout 600 --retries 3 --upgrade --no-deps `
            "faster-whisper==0.10.1" "ctranslate2==3.24.0"
        & $venvPython -m pip install --timeout 600 --retries 3 --upgrade `
            "av>=11" "onnxruntime<2,>=1.14" "tokenizers==0.15.2" `
            "huggingface-hub>=0.13,<1" "transformers==4.38.2" `
            "torch>=2,<3" "nvidia-cublas-cu11==11.11.3.6" `
            "nvidia-cuda-nvrtc-cu11==11.8.89" "nvidia-cudnn-cu11==8.9.5.29"
        if ($LASTEXITCODE -ne 0) {
            throw "Could not install the server ASR package set."
        }
    } else {
        & $venvPython -m pip install --timeout 600 --retries 3 -e "${ProjectRoot}[asr,nlp,parquet,embeddings,web]"
    }
}
if ($LASTEXITCODE -ne 0) {
    throw "Dependency installation failed. Check PyPI connectivity, or prepare wheelhouse and rerun with -Offline."
}

if ((Test-Path (Join-Path $InputPath "test.zip")) -and (Test-Path (Join-Path $InputPath "train"))) {
    Write-Host "Preparing train + test corpus..." -ForegroundColor Cyan
    $prepareCorpusExe = [System.IO.Path]::Combine($serverVenv, "Scripts", "sara-prepare-corpus.exe")
    if (-not (Test-Path $prepareCorpusExe)) {
        throw "Installed console script was not found: $prepareCorpusExe"
    }
    & $prepareCorpusExe $InputPath
    if ($LASTEXITCODE -ne 0) {
        throw "Corpus preparation failed."
    }
}

$smokeOutput = "$OutputPath-smoke"
if (Test-Path $smokeOutput) {
    throw "Smoke output already exists: $smokeOutput"
}
if (Test-Path $OutputPath) {
    throw "Output already exists: $OutputPath"
}

Write-Host "[4/5] Running CUDA smoke test on $SmokeInputPath..." -ForegroundColor Cyan
$serverSmokeExe = [System.IO.Path]::Combine($serverVenv, "Scripts", "sara-server-smoke.exe")
$extractExe = [System.IO.Path]::Combine($serverVenv, "Scripts", "sara-extract.exe")
if (-not (Test-Path $serverSmokeExe) -or -not (Test-Path $extractExe)) {
    throw "Installed SARA console scripts were not found under $serverVenv\Scripts"
}
& $serverSmokeExe $SmokeInputPath $smokeOutput --config $configPath
if ($LASTEXITCODE -ne 0) {
    throw "Server smoke test failed; full extraction was not started."
}
Write-Host "[5/5] Processing all recordings from $InputPath..." -ForegroundColor Cyan
& $extractExe $InputPath $OutputPath --config $configPath
$extractExitCode = $LASTEXITCODE
$wideFeaturesPath = Join-Path $OutputPath "features_wide.csv"
if ($extractExitCode -ne 0 -and -not (Test-Path $wideFeaturesPath)) {
    throw "Full extraction failed."
}
if ($extractExitCode -ne 0) {
    Write-Warning "The extractor returned exit code $extractExitCode after writing $wideFeaturesPath. Keeping completed artifacts."
}
Write-Host "Finished. Results: $OutputPath" -ForegroundColor Green

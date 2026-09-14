[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$InputPath,

    [string]$ConfigPath = "",

    [string]$OutputPath = "",

    [string]$VenvPath = "",

    [string]$LogPath = "",

    [string]$ProjectRoot = (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path $ProjectRoot).Path
$InputPath = (Resolve-Path $InputPath).Path

if (-not $ConfigPath) {
    $ConfigPath = Join-Path $ProjectRoot "config\rtx4090_cuda12.yaml"
} elseif (-not [System.IO.Path]::IsPathRooted($ConfigPath)) {
    $ConfigPath = Join-Path $ProjectRoot $ConfigPath
}
$ConfigPath = (Resolve-Path $ConfigPath).Path

if (-not $VenvPath) {
    $VenvPath = Join-Path $ProjectRoot ".cuda12-venv-final"
} elseif (-not [System.IO.Path]::IsPathRooted($VenvPath)) {
    $VenvPath = Join-Path $ProjectRoot $VenvPath
}
$VenvPath = [System.IO.Path]::GetFullPath($VenvPath)
$Python = Join-Path $VenvPath "Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Python executable was not found: $Python"
}

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
if (-not $OutputPath) {
    $OutputPath = Join-Path $ProjectRoot "results\pipeline-trace-$timestamp"
}
if (Test-Path $OutputPath) {
    throw "Output directory already exists: $OutputPath"
}

if (-not $LogPath) {
    $LogPath = Join-Path $ProjectRoot "logs\pipeline-trace-$timestamp.log"
}
$logParent = Split-Path -Parent $LogPath
New-Item -ItemType Directory -Force -Path $logParent | Out-Null

Push-Location $ProjectRoot
try {
    Write-Host "Tracing full SARA pipeline. Log: $LogPath" -ForegroundColor Cyan
    $previousErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $Python -u -X faulthandler -m sara_audio.web_worker `
        --config $ConfigPath `
        --input $InputPath `
        --output $OutputPath 2>&1 |
        Tee-Object -FilePath $LogPath
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousErrorAction
} finally {
    Pop-Location
}

Write-Host "Pipeline exit code: $exitCode" -ForegroundColor Yellow
Write-Host "Trace log: $LogPath" -ForegroundColor Yellow
exit $exitCode

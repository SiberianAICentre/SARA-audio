[CmdletBinding()]
param(
    [ValidateSet("cuda11_8", "cuda12")]
    [string]$CudaProfile = "cuda12",
    [string]$Image = "sara-audio:rtx4090-$CudaProfile"
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$dockerInfo = docker info 2>&1
if ($LASTEXITCODE -ne 0) {
    throw "Docker Linux engine is unavailable. Start Docker Desktop, select the Linux engine, and rerun this command.`n$dockerInfo"
}
$dockerfile = if ($CudaProfile -eq "cuda11_8") {
    Join-Path $PSScriptRoot "Dockerfile.cuda11_8"
} else {
    Join-Path $PSScriptRoot "Dockerfile"
}

Write-Host "Building $Image using $CudaProfile profile..." -ForegroundColor Cyan
docker build --file $dockerfile --tag $Image $projectRoot
if ($LASTEXITCODE -ne 0) {
    throw "Docker build failed for profile $CudaProfile."
}
Write-Host "Built $Image" -ForegroundColor Green

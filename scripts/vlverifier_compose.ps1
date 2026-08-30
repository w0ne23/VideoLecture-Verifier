# Select the portable CPU stack or NVIDIA stack from Windows PowerShell.
# Docker Desktop must use the WSL 2 backend for NVIDIA GPU containers.
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ComposeArgs
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$Mode = if ($env:VLVERIFIER_MODE) { $env:VLVERIFIER_MODE.ToLowerInvariant() } else { "auto" }
if ($Mode -notin @("auto", "cpu", "gpu")) {
    throw "Invalid VLVERIFIER_MODE=$Mode; use auto, cpu, or gpu."
}

$GpuReady = $false
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    try {
        & nvidia-smi -L *> $null
        if ($LASTEXITCODE -eq 0) {
            $Runtimes = (& docker info --format '{{json .Runtimes}}' 2>$null) -join ""
            if ($Runtimes -match "nvidia") {
                $GpuReady = $true
            }
        }
    } catch {
        $GpuReady = $false
    }
}

if ($Mode -eq "gpu" -and -not $GpuReady) {
    throw "VLVERIFIER_MODE=gpu was requested, but NVIDIA Docker GPU support was not detected. Check Docker Desktop WSL 2 GPU integration."
}
if ($Mode -eq "auto") {
    $Mode = if ($GpuReady) { "gpu" } else { "cpu" }
}

$Files = @("-f", "docker-compose.yml")
if ($Mode -eq "gpu") {
    $Files += @("-f", "docker-compose.gpu.yml", "--profile", "ocr")
    Write-Host "VLVerifier mode: GPU (CUDA decode, TensorRT, Nemotron OCR)"
} else {
    $Files += @("--profile", "rapidocr")
    Write-Host "VLVerifier mode: CPU (OpenCV decode, CPU YOLO, RapidOCR PP-OCRv5 Korean)"
}

& docker compose @Files @ComposeArgs
exit $LASTEXITCODE

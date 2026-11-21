<#
.SYNOPSIS
    Run the remaining GPU experiments in sequence.
.DESCRIPTION
    Runs the U-Net structure-preserving loss ablation, the robustness grid, and
    the inference efficiency benchmark. Native Python output stays attached to
    the terminal so tqdm progress bars can update on one line.
#>
$ErrorActionPreference = "Stop"
$projectRoot = $PSScriptRoot
$pythonExe = Join-Path $projectRoot ".venv-gpu\Scripts\python.exe"
$logDir = Join-Path $projectRoot "logs"
if (-not (Test-Path $pythonExe)) {
    throw "GPU Python environment not found: $pythonExe"
}
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
}
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$runLog = Join-Path $logDir "remaining_experiments_$timestamp.log"

function Write-RunLog {
    param([string]$Message, [string]$Level = "INFO")
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] [$Level] $Message"
    Write-Host $line
    [System.IO.File]::AppendAllText($runLog, "$line`r`n", [System.Text.Encoding]::UTF8)
}

function Invoke-ExperimentStage {
    param([string]$Name, [string[]]$Arguments)
    $stageStart = Get-Date
    Write-RunLog "============================================================"
    Write-RunLog "STARTING: $Name"
    Write-RunLog "Command: $pythonExe $($Arguments -join ' ')"

    # Keep native stdout attached to the terminal for single-line tqdm updates.
    & $pythonExe @Arguments
    $exitCode = $LASTEXITCODE
    $duration = (Get-Date) - $stageStart
    $durationText = "{0:D2}h:{1:D2}m:{2:D2}s" -f [int]$duration.TotalHours, $duration.Minutes, $duration.Seconds
    if ($exitCode -ne 0) {
        Write-RunLog "FAILED: $Name (exit code $exitCode, duration $durationText)" "ERROR"
        throw "Experiment stage '$Name' failed. Earlier outputs and checkpoints are preserved. See $runLog"
    }
    Write-RunLog "COMPLETED: $Name (duration $durationText)" "SUCCESS"
}

Write-RunLog "Starting remaining GPU experiments."
Write-RunLog "Python environment: $pythonExe"
Write-RunLog "Run log: $runLog"

$cudaStatus = & $pythonExe -c "import torch; print(torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0 -or "$cudaStatus".Trim() -ne "True") {
    Write-RunLog "CUDA is unavailable in .venv-gpu; no experiments were started." "ERROR"
    exit 1
}
Write-RunLog "CUDA is available."

try {
    Invoke-ExperimentStage "U-Net loss ablation" @(
        "-u", "-m", "experiments.run_loss_ablation",
        "--model", "unet", "--epochs", "50", "--batch-size", "16", "--num-workers", "2", "--resume"
    )
    Invoke-ExperimentStage "Test-set noise robustness grid" @(
        "-u", "-m", "experiments.run_robustness",
        "--batch-size", "8", "--num-workers", "2", "--resume"
    )
    Invoke-ExperimentStage "Inference efficiency benchmark" @(
        "-u", "-m", "evaluation.run_efficiency",
        "--batch-sizes", "1", "8", "--resume"
    )
} catch {
    Write-RunLog $_.Exception.Message "ERROR"
    exit 1
}

Write-RunLog "All remaining GPU experiments completed successfully." "SUCCESS"
Write-Host "Results: $projectRoot\results"
Write-Host "Run log: $runLog"

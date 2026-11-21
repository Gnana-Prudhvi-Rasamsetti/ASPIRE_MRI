<#
.SYNOPSIS
    Automated sequential training pipeline for ASPIRE-MRI on RTX 4060.
.DESCRIPTION
    Runs stages sequentially: dncnn -> unet -> restormer -> noise_detector -> adaptive (-> fine_tune optional).
    Stops immediately on error, auto-resumes interrupted stages, streams output in real-time,
    and writes a timestamped log to logs\pipeline_YYYYMMDD_HHMMSS.log.
#>
param(
    [switch]$IncludeFineTune = $false
)

$ErrorActionPreference = "Continue"
$projectRoot = $PSScriptRoot
if (-not $projectRoot) { $projectRoot = Get-Location }

# Configure UTF-8 encoding consistently across PowerShell 5.1 and Python
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"
try {
    chcp 65001 > $null
} catch {}

# 1. Verify GPU environment
$pythonExe = Join-Path $projectRoot ".venv-gpu\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    Write-Error "Python executable not found in .venv-gpu: $pythonExe"
    exit 1
}

# 2. Setup logging directory and file
$logDir = Join-Path $projectRoot "logs"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$pipelineLog = Join-Path $logDir "pipeline_$timestamp.log"
$env:PIPELINE_LOG_FILE = $pipelineLog

function Log-Message {
    [Diagnostics.CodeAnalysis.SuppressMessageAttribute("PSUseApprovedVerbs", "")]
    param(
        [string]$Message,
        [string]$Level = "INFO"
    )
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] [$Level] $Message"
    Write-Host $line
    [System.IO.File]::AppendAllText($pipelineLog, "$line`r`n", [System.Text.Encoding]::UTF8)
}

Log-Message "Starting ASPIRE-MRI automated training pipeline."
Log-Message "Python: $pythonExe"
Log-Message "Master Log File: $pipelineLog"

# Define stage pipeline configuration
$stages = @(
    @{ Name = "dncnn";          Epochs = 50; BatchSize = 16; NumWorkers = 2; LatestCkpt = "checkpoints\dncnn\latest.pt";          BestCkpt = "checkpoints\dncnn\best.pt" },
    @{ Name = "unet";           Epochs = 50; BatchSize = 16; NumWorkers = 2; LatestCkpt = "checkpoints\unet\latest.pt";           BestCkpt = "checkpoints\unet\best.pt" },
    @{ Name = "restormer";      Epochs = 50; BatchSize = 8;  NumWorkers = 2; LatestCkpt = "checkpoints\restormer\latest.pt";      BestCkpt = "checkpoints\restormer\best.pt" },
    @{ Name = "noise_detector"; Epochs = 50; BatchSize = 16; NumWorkers = 2; LatestCkpt = "checkpoints\noise_detector\latest.pt"; BestCkpt = "checkpoints\noise_detector\best.pt" },
    @{ Name = "adaptive";       Epochs = 50; BatchSize = 16; NumWorkers = 2; LatestCkpt = "checkpoints\adaptive\latest.pt";       BestCkpt = "checkpoints\adaptive\best.pt" }
)

if ($IncludeFineTune) {
    $stages += @{
        Name         = "fine_tune"
        Epochs       = 10
        BatchSize    = 16
        NumWorkers   = 2
        LearningRate = 0.00001
        LatestCkpt   = "checkpoints\adaptive\fine_tuned_latest.pt"
        BestCkpt     = "checkpoints\adaptive\fine_tuned.pt"
    }
}

$stageSummaries = @()
$pipelineStartTime = Get-Date

foreach ($st in $stages) {
    $stageName = $st.Name
    $stageStart = Get-Date
    Log-Message "============================================================"
    Log-Message "STARTING STAGE: $stageName"
    
    # Check for resume support
    $latestPath = Join-Path $projectRoot $st.LatestCkpt
    $argsList = @("-u", "-m", "training.run_stage", $stageName, "--epochs", $st.Epochs, "--batch-size", $st.BatchSize, "--num-workers", $st.NumWorkers)
    
    if ($st.ContainsKey("LearningRate")) {
        $argsList += @("--learning-rate", $st.LearningRate)
    }

    if (Test-Path $latestPath) {
        Log-Message "Found existing checkpoint at $latestPath. Resuming..."
        $argsList += "--resume"
    } else {
        Log-Message "No existing checkpoint found. Starting fresh from epoch 1."
    }

    Log-Message "Command: python $($argsList -join ' ')"
    
    # Run the stage directly attached to terminal to preserve TTY and single-line tqdm updates
    & $pythonExe @argsList
    $exitCode = $LASTEXITCODE
    $stageDuration = (Get-Date) - $stageStart
    $durationStr = "{0:D2}h:{1:D2}m:{2:D2}s" -f [int]$stageDuration.TotalHours, $stageDuration.Minutes, $stageDuration.Seconds

    if ($exitCode -ne 0) {
        Log-Message "STAGE FAILED: $stageName with exit code $exitCode (Duration: $durationStr)" "ERROR"
        Log-Message "Halting pipeline execution immediately. Previous checkpoints preserved." "ERROR"
        
        Write-Host "`n=================== PIPELINE EXECUTION SUMMARY ===================" -ForegroundColor Red
        Write-Host "Status: FAILED at stage '$stageName'" -ForegroundColor Red
        Write-Host "Log file: $pipelineLog" -ForegroundColor Yellow
        exit $exitCode
    }

    $bestPath = Join-Path $projectRoot $st.BestCkpt
    Log-Message "STAGE COMPLETED: $stageName successfully (Duration: $durationStr)" "SUCCESS"
    if (Test-Path $bestPath) {
        Log-Message "Best checkpoint verified: $bestPath"
    }

    $stageSummaries += [PSCustomObject]@{
        Stage      = $stageName
        Status     = "COMPLETED"
        Duration   = $durationStr
        Checkpoint = $st.BestCkpt
    }
}

$totalDuration = (Get-Date) - $pipelineStartTime
$totalStr = "{0:D2}h:{1:D2}m:{2:D2}s" -f [int]$totalDuration.TotalHours, $totalDuration.Minutes, $totalDuration.Seconds

Log-Message "============================================================"
Log-Message "ALL STAGES COMPLETED SUCCESSFULLY! Total time: $totalStr" "SUCCESS"

Write-Host "`n=================== PIPELINE EXECUTION SUMMARY ===================" -ForegroundColor Green
$stageSummaries | Format-Table -AutoSize
Write-Host "Total Execution Time: $totalStr" -ForegroundColor Cyan
Write-Host "Checkpoints Location: $projectRoot\checkpoints" -ForegroundColor Cyan
Write-Host "Complete Log File:    $pipelineLog" -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Green


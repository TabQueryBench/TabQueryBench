[CmdletBinding()]
param(
    [string]$ProjectRoot = "D:\dpan\Uni\Project\HKUNAISS\SQLagent",
    [string]$Model = "gpt-5.4",
    [int]$AiCliTimeoutSeconds = 180,
    [int]$AiCliRetries = 2,
    [int]$StartGapSeconds = 8,
    [int]$BootstrapDelaySeconds = 6,
    [string]$RunStamp = "",
    [switch]$DryRun,
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"

if (-not $RunStamp) {
    $RunStamp = Get-Date -Format "yyyyMMdd_HHmmss"
}

$scriptPath = Join-Path $ProjectRoot "scripts\run_subitem_workload_v2.py"
$launchDir = Join-Path $ProjectRoot "logs\subitem_workload_v2\launches"
New-Item -ItemType Directory -Force -Path $launchDir | Out-Null
$workerDir = Join-Path $launchDir "workers"
New-Item -ItemType Directory -Force -Path $workerDir | Out-Null

function Resolve-PythonPath {
    param([string]$Requested)

    if ($Requested -and (Test-Path $Requested)) {
        return (Resolve-Path $Requested).Path
    }

    if ($env:CONDA_PYTHON_EXE -and (Test-Path $env:CONDA_PYTHON_EXE)) {
        return (Resolve-Path $env:CONDA_PYTHON_EXE).Path
    }

    if ($env:CONDA_PREFIX) {
        $condaPython = Join-Path $env:CONDA_PREFIX "python.exe"
        if (Test-Path $condaPython) {
            return (Resolve-Path $condaPython).Path
        }
    }

    return (Get-Command $Requested -ErrorAction Stop).Source
}

function Quote-SingleQuoted {
    param([string]$Text)
    return "'" + ($Text -replace "'", "''") + "'"
}

function Parse-DatasetList {
    param([string]$DatasetIds)
    return @($DatasetIds -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ })
}

$pythonPath = Resolve-PythonPath -Requested $PythonExe
$powershellPath = (Get-Command powershell.exe -ErrorAction Stop).Source

$shards = @(
    @{
        suffix = "a"
        datasets = "n8,c10,m1,m9,n9,c4,n20,n12"
    },
    @{
        suffix = "b"
        datasets = "n15,m10,m6,c8,c15,c7,c17,n11"
    },
    @{
        suffix = "c"
        datasets = "m12,n19,n5,m7,n2,c11,c9,n16"
    },
    @{
        suffix = "d"
        datasets = "c16,n17,c19,c14,c18,n10,n1,m4"
    },
    @{
        suffix = "e"
        datasets = "m5,n4,c12,c20,c13,c5,n7,c6,c3"
    },
    @{
        suffix = "f"
        datasets = "m2,m11,n18,m8,n14,c2,n6,n3"
    }
)

$allDatasets = @()
foreach ($shard in $shards) {
    $allDatasets += Parse-DatasetList -DatasetIds $shard.datasets
}

$duplicateDatasets = $allDatasets | Group-Object | Where-Object { $_.Count -gt 1 } | Select-Object -ExpandProperty Name
if ($duplicateDatasets) {
    throw "Duplicate datasets detected in six-way shard plan: $($duplicateDatasets -join ', ')"
}

if ($allDatasets.Count -ne 49) {
    throw "Expected 49 dataset assignments in six-way shard plan, found $($allDatasets.Count)."
}

$launchRows = @()
foreach ($shard in $shards) {
    $runId = "v2_cli_${RunStamp}_$($shard.suffix)"
    $argumentList = @(
        $scriptPath,
        "--dataset-ids",
        $shard.datasets,
        "--engine",
        "cli",
        "--run-id",
        $runId,
        "--ai-cli-preset",
        "codex",
        "--model",
        $Model,
        "--ai-cli-timeout-seconds",
        "$AiCliTimeoutSeconds",
        "--ai-cli-retries",
        "$AiCliRetries"
    )

    $row = [ordered]@{
        run_id = $runId
        suffix = $shard.suffix
        dataset_ids = $shard.datasets
        dataset_count = (Parse-DatasetList -DatasetIds $shard.datasets).Count
        inner_command = @($pythonPath) + $argumentList -join " "
        status = if ($DryRun) { "planned" } else { "launched" }
    }

    $workerScriptPath = Join-Path $workerDir "worker_${RunStamp}_$($shard.suffix).ps1"
    $workerLogPath = Join-Path $workerDir "worker_${RunStamp}_$($shard.suffix).log"
    $workerCommand = @(
        "& " + (Quote-SingleQuoted $pythonPath),
        (Quote-SingleQuoted $scriptPath),
        "--dataset-ids",
        (Quote-SingleQuoted $shard.datasets),
        "--engine",
        "cli",
        "--run-id",
        (Quote-SingleQuoted $runId),
        "--ai-cli-preset",
        "codex",
        "--model",
        (Quote-SingleQuoted $Model),
        "--ai-cli-timeout-seconds",
        "$AiCliTimeoutSeconds",
        "--ai-cli-retries",
        "$AiCliRetries",
        "2>&1 | Tee-Object -FilePath " + (Quote-SingleQuoted $workerLogPath) + " -Append"
    ) -join " "
    @(
        '$ErrorActionPreference = "Stop"',
        ("Set-Location " + (Quote-SingleQuoted $ProjectRoot)),
        $workerCommand
    ) | Set-Content -Path $workerScriptPath -Encoding UTF8

    $row.worker_script_path = $workerScriptPath
    $row.log_path = $workerLogPath
    $row.command = @($powershellPath, "-NoExit", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $workerScriptPath) -join " "

    if (-not $DryRun) {
        $proc = Start-Process -FilePath $powershellPath -ArgumentList @(
            "-NoExit",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            $workerScriptPath
        ) -WorkingDirectory $ProjectRoot -PassThru
        $row.process_id = $proc.Id
        $row.started_at = (Get-Date).ToString("o")
        if ($StartGapSeconds -gt 0) {
            Start-Sleep -Seconds $StartGapSeconds
        }
    }

    $launchRows += [pscustomobject]$row
}

if (-not $DryRun -and $BootstrapDelaySeconds -gt 0) {
    Start-Sleep -Seconds $BootstrapDelaySeconds
}

$manifestPath = Join-Path $launchDir "launch_${RunStamp}.json"
$launchRows | ConvertTo-Json -Depth 4 | Set-Content -Path $manifestPath -Encoding UTF8

Write-Host "[v2-launch] manifest=$manifestPath"
$launchRows | Format-Table -AutoSize

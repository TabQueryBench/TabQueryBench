[CmdletBinding()]
param(
    [string]$ProjectRoot = ".",
    [ValidateSet("v2", "v3", "v4")]
    [string]$LineVersion = "v2",
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
$launchDir = Join-Path $ProjectRoot ("logs\subitem_workload_{0}\launches" -f $LineVersion)
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

$pythonPath = Resolve-PythonPath -Requested $PythonExe
$powershellPath = (Get-Command powershell.exe -ErrorAction Stop).Source

function Quote-SingleQuoted {
    param([string]$Text)
    return "'" + ($Text -replace "'", "''") + "'"
}

$shards = switch ($LineVersion) {
    "v3" {
        @(
            @{ suffix = "a"; datasets = "c14,n11" },
            @{ suffix = "b"; datasets = "m6,n6" },
            @{ suffix = "c"; datasets = "m8,n3" },
            @{ suffix = "d"; datasets = "c2,c7,m4" }
        )
    }
    "v4" {
        @(
            @{ suffix = "a"; datasets = "c14,n11" },
            @{ suffix = "b"; datasets = "m6,n6" },
            @{ suffix = "c"; datasets = "m8,n3" },
            @{ suffix = "d"; datasets = "c2,c7,m4" }
        )
    }
    default {
        @(
            @{ suffix = "a"; datasets = "c2,c6,c10,c14,c18,m1,m6,m10,n1,n5,n9,n14,n18" },
            @{ suffix = "b"; datasets = "c3,c7,c11,c15,c19,m2,m7,m11,n2,n6,n10,n15,n19" },
            @{ suffix = "c"; datasets = "c4,c8,c12,c16,c1,m4,m8,m12,n3,n7,n11,n16,n20" },
            @{ suffix = "d"; datasets = "c5,c9,c13,c17,m5,m9,n4,n8,n12,n17" }
        )
    }
}

$launchRows = @()
foreach ($shard in $shards) {
    $runId = "${LineVersion}_cli_${RunStamp}_$($shard.suffix)"
    $argumentList = @(
        $scriptPath,
        "--line-version",
        $LineVersion,
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
        inner_command = @($pythonPath) + $argumentList -join " "
        status = if ($DryRun) { "planned" } else { "launched" }
    }

    $workerScriptPath = Join-Path $workerDir "worker_${RunStamp}_$($shard.suffix).ps1"
    $workerLogPath = Join-Path $workerDir "worker_${RunStamp}_$($shard.suffix).log"
    $workerCommand = @(
        "& " + (Quote-SingleQuoted $pythonPath),
        (Quote-SingleQuoted $scriptPath),
        "--line-version",
        (Quote-SingleQuoted $LineVersion),
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

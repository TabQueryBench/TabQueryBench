param(
    [string]$PythonExe = "C:\Users\16943\anaconda3\python.exe",
    [string]$LatexEngine = "D:\dpan\Uni\Project\HKUNAISS\SQLagent\tools\tectonic\tectonic-0.16.9\tectonic.exe",
    [int]$AnalysisMaxWorkers = 4,
    [switch]$SkipWait
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$EvalRoot = Join-Path $RepoRoot "Evaluation"
$RunTagBase = "eval_v3_v4_layers_20260505_0100_zurich"
$RunRoot = Join-Path (Join-Path $EvalRoot "orchestration") $RunTagBase
$TaskScriptRoot = Join-Path $RunRoot "task_scripts"
$TaskLogRoot = Join-Path $RunRoot "task_logs"
$TaskResultRoot = Join-Path $RunRoot "task_results"
$StartupStatusPath = Join-Path $RunRoot "startup_status.json"
$ManifestPath = Join-Path $RunRoot "orchestration_manifest.json"

New-Item -ItemType Directory -Force -Path $TaskScriptRoot | Out-Null
New-Item -ItemType Directory -Force -Path $TaskLogRoot | Out-Null
New-Item -ItemType Directory -Force -Path $TaskResultRoot | Out-Null

$AnalysisV3RunTag = "${RunTagBase}_analysis_v3"
$AnalysisV4RunTag = "${RunTagBase}_analysis_v4"
$AnalysisV3RunDir = Join-Path $EvalRoot "analysis\runs\$AnalysisV3RunTag"
$AnalysisV4RunDir = Join-Path $EvalRoot "analysis\runs\$AnalysisV4RunTag"

function Resolve-SwissTimeZone {
    $candidateIds = @(
        "Europe/Zurich",
        "W. Europe Standard Time",
        "Central European Standard Time"
    )
    foreach ($candidate in $candidateIds) {
        try {
            return [System.TimeZoneInfo]::FindSystemTimeZoneById($candidate)
        } catch {
            continue
        }
    }
    throw "Could not resolve a Swiss/Zurich timezone on this machine."
}

function Resolve-PythonExecutable {
    param([string]$Requested)
    $command = Get-Command $Requested -ErrorAction SilentlyContinue
    if ($command -and $command.Source) {
        return $command.Source
    }
    if (Test-Path $Requested) {
        return (Resolve-Path $Requested).Path
    }
    throw "Could not resolve Python executable from '$Requested'."
}

function Resolve-LatestTaskRunDir {
    param([string]$TaskName)
    $latestPath = Join-Path $EvalRoot "$TaskName\LATEST_RUN.json"
    if (-not (Test-Path $latestPath)) {
        throw "Missing latest run pointer for task '$TaskName': $latestPath"
    }
    $payload = Get-Content -Path $latestPath -Raw | ConvertFrom-Json
    $runDir = [string]$payload.run_dir
    if (-not [string]::IsNullOrWhiteSpace($runDir)) {
        if ([System.IO.Path]::IsPathRooted($runDir)) {
            $candidate = $runDir
        } else {
            $candidate = Join-Path $RepoRoot $runDir
        }
        if (Test-Path $candidate) {
            return (Resolve-Path $candidate).Path
        }
    }
    $runTag = [string]$payload.run_tag
    if (-not [string]::IsNullOrWhiteSpace($runTag)) {
        $fallback = Join-Path $EvalRoot "$TaskName\runs\$runTag"
        if (Test-Path $fallback) {
            return (Resolve-Path $fallback).Path
        }
    }
    throw "Could not resolve an existing run directory for task '$TaskName'."
}

function Wait-UntilScheduledSwissStart {
    if ($SkipWait) {
        Write-Host "[schedule] SkipWait is set. Starting immediately." -ForegroundColor Yellow
        return
    }

    $swissTz = Resolve-SwissTimeZone
    $targetSwissText = "2026-05-05 01:00:00"
    $targetSwiss = [datetime]::ParseExact(
        $targetSwissText,
        "yyyy-MM-dd HH:mm:ss",
        [System.Globalization.CultureInfo]::InvariantCulture
    )
    $targetSwiss = [datetime]::SpecifyKind($targetSwiss, [System.DateTimeKind]::Unspecified)
    $targetUtc = [System.TimeZoneInfo]::ConvertTimeToUtc($targetSwiss, $swissTz)
    $nowUtc = [datetime]::UtcNow

    if ($nowUtc -ge $targetUtc) {
        Write-Host "[schedule] Current UTC time is already past Swiss start time 2026-05-05 01:00. Starting immediately." -ForegroundColor Yellow
        return
    }

    $waitSeconds = [math]::Ceiling(($targetUtc - $nowUtc).TotalSeconds)
    Write-Host "[schedule] Waiting until Swiss time 2026-05-05 01:00:00 ($($swissTz.Id)); sleeping $waitSeconds second(s)." -ForegroundColor Cyan
    Start-Sleep -Seconds $waitSeconds
}

function New-PythonTaskFile {
    param(
        [string]$TaskName,
        [string]$PythonBody
    )
    $path = Join-Path $TaskScriptRoot "$TaskName.py"
    Set-Content -Path $path -Value $PythonBody -Encoding UTF8
    return $path
}

function New-ResultPrelude {
    param([string]$TaskName)
    $resultPath = Join-Path $TaskResultRoot "$TaskName.json"
    return @"
import json
import sys
from pathlib import Path
from multiprocessing import freeze_support

repo_root = Path(r"$RepoRoot")
result_path = Path(r"$resultPath")
task_name = "$TaskName"
sys.path.insert(0, str(repo_root))

def write_task_payload(result_obj):
    if isinstance(result_obj, dict) and "manifest" in result_obj:
        run_dir = result_obj.get("run_dir")
        manifest = result_obj.get("manifest") or {}
    else:
        run_dir = result_obj.get("run_dir") if isinstance(result_obj, dict) else None
        manifest = result_obj if isinstance(result_obj, dict) else {}
    payload = {
        "task": task_name,
        "run_dir": str(Path(run_dir).resolve()) if run_dir else str(Path(manifest.get("run_dir", "")).resolve()) if manifest.get("run_dir") else "",
        "manifest": manifest,
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
"@
}

function Write-AnalysisWrapper {
    param(
        [string]$TaskName,
        [string]$RunTag,
        [string]$SqlSourceVersion
    )
    $prelude = New-ResultPrelude -TaskName $TaskName
    $body = @"
$prelude

from src.eval.analysis.runner import run_sql_analysis

def main():
    result = run_sql_analysis(
        run_tag="$RunTag",
        datasets=None,
        latest_only=True,
        engines=("cli",),
        sql_source_version="$SqlSourceVersion",
        include_all_sql_statements=True,
        max_sql_per_dataset=0,
        query_row_limit=0,
        max_workers=$AnalysisMaxWorkers,
        latex_engine=r"$LatexEngine",
    )
    write_task_payload(result)

if __name__ == "__main__":
    freeze_support()
    main()
"@
    New-PythonTaskFile -TaskName $TaskName -PythonBody $body | Out-Null
}

function Write-RunPyWrapper {
    param(
        [string]$TaskName,
        [string]$TargetScriptPath,
        [string]$SqlSourceVersion
    )
    $prelude = New-ResultPrelude -TaskName $TaskName
    $body = @"
$prelude

import os
import runpy

def main():
    os.environ["EVAL_SQL_SOURCE_VERSION"] = "$SqlSourceVersion"
    runpy.run_path(r"$TargetScriptPath", run_name="__main__")
    write_task_payload({"run_dir": "", "manifest": {"sql_source_version": "$SqlSourceVersion"}})

if __name__ == "__main__":
    freeze_support()
    main()
"@
    New-PythonTaskFile -TaskName $TaskName -PythonBody $body | Out-Null
}

function Write-DirectWrapper {
    param(
        [string]$TaskName,
        [string]$ImportLine,
        [string]$CallExpression,
        [string]$SqlSourceVersion = ""
    )
    $prelude = New-ResultPrelude -TaskName $TaskName
    $envBlock = if ($SqlSourceVersion) { "import os`nos.environ[`"EVAL_SQL_SOURCE_VERSION`"] = `"$SqlSourceVersion`"" } else { "" }
    $body = @"
$prelude

$envBlock
$ImportLine

def main():
    result = $CallExpression
    write_task_payload(result)

if __name__ == "__main__":
    freeze_support()
    main()
"@
    New-PythonTaskFile -TaskName $TaskName -PythonBody $body | Out-Null
}

function Start-Task {
    param([string]$TaskName)
    $scriptPath = Join-Path $TaskScriptRoot "$TaskName.py"
    $stdoutPath = Join-Path $TaskLogRoot "$TaskName.stdout.log"
    $stderrPath = Join-Path $TaskLogRoot "$TaskName.stderr.log"
    Remove-Item -LiteralPath $stdoutPath -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $stderrPath -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath (Join-Path $TaskResultRoot "$TaskName.json") -ErrorAction SilentlyContinue
    return Start-Process -FilePath $ResolvedPythonExe -ArgumentList @($scriptPath) -WorkingDirectory $RepoRoot -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
}

function Wait-TaskGroup {
    param(
        [string]$GroupName,
        [hashtable]$Tasks
    )
    Write-Host "[$GroupName] waiting for $($Tasks.Count) task(s)..." -ForegroundColor Cyan
    foreach ($process in $Tasks.Values) {
        $null = $process.WaitForExit()
    }
    $failed = @()
    foreach ($entry in $Tasks.GetEnumerator()) {
        if ($entry.Value.ExitCode -ne 0) {
            $failed += $entry.Key
        }
    }
    if ($failed.Count -gt 0) {
        throw "Task group '$GroupName' failed: $($failed -join ', ')"
    }
}

$ResolvedPythonExe = Resolve-PythonExecutable -Requested $PythonExe
$DistanceRunDir = Resolve-LatestTaskRunDir -TaskName "distance"
$ValidationRunDir = Resolve-LatestTaskRunDir -TaskName "validation"

$startupStatus = [ordered]@{
    run_tag = $RunTagBase
    repo_root = $RepoRoot
    run_root = $RunRoot
    script_started_utc = [datetime]::UtcNow.ToString("o")
    python_executable = $ResolvedPythonExe
    distance_run_dir = $DistanceRunDir
    validation_run_dir = $ValidationRunDir
    analysis_max_workers = $AnalysisMaxWorkers
    scheduled_start_swiss = "2026-05-05 01:00:00"
    scheduled_timezone_label = "Switzerland / Europe-Zurich"
}
$startupStatus | ConvertTo-Json -Depth 4 | Set-Content -Path $StartupStatusPath -Encoding UTF8

$manifest = [ordered]@{
    run_tag = $RunTagBase
    repo_root = $RepoRoot
    run_root = $RunRoot
    scheduled_start_swiss = "2026-05-05 01:00:00"
    scheduled_timezone_label = "Switzerland / Europe-Zurich"
    distance_run_dir = $DistanceRunDir
    validation_run_dir = $ValidationRunDir
    analysis_max_workers = $AnalysisMaxWorkers
    layers = @(
        @{ name = "layer1_analysis"; tasks = @("analysis_v3", "analysis_v4") },
        @{ name = "layer2_sql_downstream"; tasks = @("sql_eval_v3", "dataset_counts_v3", "distance_query_scatter_v3", "sqlvisualize_v3", "sql_eval_v4", "dataset_counts_v4", "distance_query_scatter_v4", "sqlvisualize_v4") },
        @{ name = "layer3_breakdowns"; tasks = @("subgroup_v3", "conditional_v3", "missingness_v3", "tail_v3", "cardinality_v3", "subgroup_v4", "conditional_v4", "missingness_v4", "tail_v4", "cardinality_v4") },
        @{ name = "layer4_model_radar"; tasks = @("model_radar_v3", "model_radar_v4") }
    )
}
$manifest | ConvertTo-Json -Depth 6 | Set-Content -Path $ManifestPath -Encoding UTF8

Write-AnalysisWrapper -TaskName "analysis_v3" -RunTag $AnalysisV3RunTag -SqlSourceVersion "v3"
Write-AnalysisWrapper -TaskName "analysis_v4" -RunTag $AnalysisV4RunTag -SqlSourceVersion "v4"

$breakdownScripts = @{
    subgroup = Join-Path $RepoRoot "src\eval\query_fivepart_breakdown\subgroup_breakdown\runner.py"
    conditional = Join-Path $RepoRoot "src\eval\query_fivepart_breakdown\conditional_breakdown\runner.py"
    missingness = Join-Path $RepoRoot "src\eval\query_fivepart_breakdown\missingness_breakdown\runner.py"
    tail = Join-Path $RepoRoot "src\eval\query_fivepart_breakdown\tail_breakdown\runner.py"
    cardinality = Join-Path $RepoRoot "src\eval\query_fivepart_breakdown\cardinality\runner.py"
}

foreach ($version in @("v3", "v4")) {
    foreach ($entry in $breakdownScripts.GetEnumerator()) {
        Write-RunPyWrapper -TaskName ("{0}_{1}" -f $entry.Key, $version) -TargetScriptPath $entry.Value -SqlSourceVersion $version
    }
}

Write-DirectWrapper -TaskName "sql_eval_v3" `
    -ImportLine "from pathlib import Path`nfrom src.eval.sql_eval.runner import run_sql_rank_stability" `
    -CallExpression "run_sql_rank_stability(run_tag='${RunTagBase}_sql_eval_v3', analysis_run_dir=Path(r'$AnalysisV3RunDir'), top_k=3, latex_engine=r'$LatexEngine', sql_source_version_override='v3')" `
    -SqlSourceVersion "v3"
Write-DirectWrapper -TaskName "sql_eval_v4" `
    -ImportLine "from pathlib import Path`nfrom src.eval.sql_eval.runner import run_sql_rank_stability" `
    -CallExpression "run_sql_rank_stability(run_tag='${RunTagBase}_sql_eval_v4', analysis_run_dir=Path(r'$AnalysisV4RunDir'), top_k=3, latex_engine=r'$LatexEngine', sql_source_version_override='v4')" `
    -SqlSourceVersion "v4"

Write-DirectWrapper -TaskName "dataset_counts_v3" `
    -ImportLine "from src.eval.dataset_subitem_sql_counts.runner import run_dataset_subitem_sql_counts" `
    -CallExpression "run_dataset_subitem_sql_counts(run_tag='${RunTagBase}_dataset_counts_v3', datasets=None, engines=('cli',), sql_source_version='v3', latex_engine=r'$LatexEngine')" `
    -SqlSourceVersion "v3"
Write-DirectWrapper -TaskName "dataset_counts_v4" `
    -ImportLine "from src.eval.dataset_subitem_sql_counts.runner import run_dataset_subitem_sql_counts" `
    -CallExpression "run_dataset_subitem_sql_counts(run_tag='${RunTagBase}_dataset_counts_v4', datasets=None, engines=('cli',), sql_source_version='v4', latex_engine=r'$LatexEngine')" `
    -SqlSourceVersion "v4"

Write-DirectWrapper -TaskName "distance_query_scatter_v3" `
    -ImportLine "from pathlib import Path`nfrom src.eval.distance_query_scatter.runner import run_distance_query_scatter" `
    -CallExpression "run_distance_query_scatter(run_tag='${RunTagBase}_distance_query_scatter_v3', analysis_run_dir=Path(r'$AnalysisV3RunDir'), distance_run_dir=Path(r'$DistanceRunDir'), compile_pdf=True, latex_engine=r'$LatexEngine')" `
    -SqlSourceVersion "v3"
Write-DirectWrapper -TaskName "distance_query_scatter_v4" `
    -ImportLine "from pathlib import Path`nfrom src.eval.distance_query_scatter.runner import run_distance_query_scatter" `
    -CallExpression "run_distance_query_scatter(run_tag='${RunTagBase}_distance_query_scatter_v4', analysis_run_dir=Path(r'$AnalysisV4RunDir'), distance_run_dir=Path(r'$DistanceRunDir'), compile_pdf=True, latex_engine=r'$LatexEngine')" `
    -SqlSourceVersion "v4"

Write-DirectWrapper -TaskName "sqlvisualize_v3" `
    -ImportLine "from pathlib import Path`nfrom src.eval.SQLvisualize.runner import run_sqlvisualize" `
    -CallExpression "run_sqlvisualize(run_tag='${RunTagBase}_sqlvisualize_v3', analysis_run_dir=Path(r'$AnalysisV3RunDir'), validation_run_dir=Path(r'$ValidationRunDir'), allow_analysis_rebuild=False, force_analysis_rebuild=False, analysis_engines=('cli',), analysis_sql_source_version='v3', analysis_latest_only=True, analysis_max_workers=$AnalysisMaxWorkers, latex_engine=r'$LatexEngine')" `
    -SqlSourceVersion "v3"
Write-DirectWrapper -TaskName "sqlvisualize_v4" `
    -ImportLine "from pathlib import Path`nfrom src.eval.SQLvisualize.runner import run_sqlvisualize" `
    -CallExpression "run_sqlvisualize(run_tag='${RunTagBase}_sqlvisualize_v4', analysis_run_dir=Path(r'$AnalysisV4RunDir'), validation_run_dir=Path(r'$ValidationRunDir'), allow_analysis_rebuild=False, force_analysis_rebuild=False, analysis_engines=('cli',), analysis_sql_source_version='v4', analysis_latest_only=True, analysis_max_workers=$AnalysisMaxWorkers, latex_engine=r'$LatexEngine')" `
    -SqlSourceVersion "v4"

Write-RunPyWrapper -TaskName "model_radar_v3" -TargetScriptPath (Join-Path $RepoRoot "src\eval\model_radar\runner.py") -SqlSourceVersion "v3"
Write-RunPyWrapper -TaskName "model_radar_v4" -TargetScriptPath (Join-Path $RepoRoot "src\eval\model_radar\runner.py") -SqlSourceVersion "v4"

Wait-UntilScheduledSwissStart

$layer1 = @{
    analysis_v3 = (Start-Task -TaskName "analysis_v3")
    analysis_v4 = (Start-Task -TaskName "analysis_v4")
}
Wait-TaskGroup -GroupName "layer1_analysis" -Tasks $layer1

$layer2 = @{}
foreach ($name in @(
    "sql_eval_v3", "dataset_counts_v3", "distance_query_scatter_v3", "sqlvisualize_v3",
    "sql_eval_v4", "dataset_counts_v4", "distance_query_scatter_v4", "sqlvisualize_v4"
)) {
    $layer2[$name] = Start-Task -TaskName $name
}
Wait-TaskGroup -GroupName "layer2_sql_downstream" -Tasks $layer2

$layer3 = @{}
foreach ($name in @(
    "subgroup_v3", "conditional_v3", "missingness_v3", "tail_v3", "cardinality_v3",
    "subgroup_v4", "conditional_v4", "missingness_v4", "tail_v4", "cardinality_v4"
)) {
    $layer3[$name] = Start-Task -TaskName $name
}
Wait-TaskGroup -GroupName "layer3_breakdowns" -Tasks $layer3

$layer4 = @{
    model_radar_v3 = (Start-Task -TaskName "model_radar_v3")
    model_radar_v4 = (Start-Task -TaskName "model_radar_v4")
}
Wait-TaskGroup -GroupName "layer4_model_radar" -Tasks $layer4

@{
    status = "ok"
    run_tag = $RunTagBase
    run_root = $RunRoot
    analysis_v3_run_dir = $AnalysisV3RunDir
    analysis_v4_run_dir = $AnalysisV4RunDir
    distance_run_dir = $DistanceRunDir
    validation_run_dir = $ValidationRunDir
} | ConvertTo-Json -Depth 4 | Set-Content -Path (Join-Path $RunRoot "run_summary.json") -Encoding UTF8

Write-Host "[done] v3/v4 layered evaluation refresh completed" -ForegroundColor Green

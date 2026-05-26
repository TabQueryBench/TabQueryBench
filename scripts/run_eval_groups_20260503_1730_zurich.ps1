param(
    [string]$PythonExe = "python",
    [string]$RunTag = "eval_refresh_20260503_1730_zurich",
    [int]$DistanceMaxWorkers = 4,
    [int]$ValidationMaxWorkers = 4,
    [int]$AnalysisMaxWorkers = 4,
    [string]$LatexEngine = "",
    [string]$PaperDir = "",
    [switch]$SkipPdf,
    [switch]$KeepAllAssets,
    [switch]$SkipWait
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$EvalRoot = Join-Path $RepoRoot "Evaluation"
$OrchestrationRoot = Join-Path $EvalRoot "orchestration"
$RunRoot = Join-Path $OrchestrationRoot $RunTag
$TaskScriptRoot = Join-Path $RunRoot "task_scripts"
$TaskResultRoot = Join-Path $RunRoot "task_results"
$TaskLogRoot = Join-Path $RunRoot "task_logs"
$StartupStatusPath = Join-Path $RunRoot "startup_status.json"

New-Item -ItemType Directory -Force -Path $TaskScriptRoot | Out-Null
New-Item -ItemType Directory -Force -Path $TaskResultRoot | Out-Null
New-Item -ItemType Directory -Force -Path $TaskLogRoot | Out-Null

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

function Convert-ToPythonLiteral {
    param([AllowNull()][object]$Value)
    return ($Value | ConvertTo-Json -Compress)
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

function New-PythonTaskFile {
    param(
        [string]$TaskName,
        [string]$PythonBody
    )
    $path = Join-Path $TaskScriptRoot "$TaskName.py"
    Set-Content -Path $path -Value $PythonBody -Encoding UTF8
    return $path
}

function Start-PythonTask {
    param(
        [string]$TaskName,
        [string]$ScriptPath
    )
    $stdoutPath = Join-Path $TaskLogRoot "$TaskName.stdout.log"
    $stderrPath = Join-Path $TaskLogRoot "$TaskName.stderr.log"
    $process = Start-Process `
        -FilePath $ResolvedPythonExe `
        -ArgumentList @($ScriptPath) `
        -WorkingDirectory $RepoRoot `
        -RedirectStandardOutput $stdoutPath `
        -RedirectStandardError $stderrPath `
        -PassThru
    return [pscustomobject]@{
        TaskName = $TaskName
        Process = $process
        StdoutPath = $stdoutPath
        StderrPath = $stderrPath
        ScriptPath = $ScriptPath
        ResultPath = (Join-Path $TaskResultRoot "$TaskName.json")
    }
}

function Wait-TaskGroup {
    param(
        [string]$GroupName,
        [array]$Tasks
    )
    Write-Host "[$GroupName] waiting for $($Tasks.Count) task(s)..." -ForegroundColor Cyan
    foreach ($task in $Tasks) {
        $null = $task.Process.WaitForExit()
    }

    $failed = @($Tasks | Where-Object { $_.Process.ExitCode -ne 0 })
    if ($failed.Count -gt 0) {
        Write-Host "[$GroupName] failed tasks:" -ForegroundColor Red
        foreach ($task in $failed) {
            Write-Host "  - $($task.TaskName) exit=$($task.Process.ExitCode)" -ForegroundColor Red
            Write-Host "    stdout: $($task.StdoutPath)"
            Write-Host "    stderr: $($task.StderrPath)"
        }
        throw "Task group '$GroupName' failed."
    }

    Write-Host "[$GroupName] all tasks completed successfully." -ForegroundColor Green
}

function Read-TaskResult {
    param([string]$TaskName)
    $path = Join-Path $TaskResultRoot "$TaskName.json"
    if (-not (Test-Path $path)) {
        throw "Result file not found for task '$TaskName': $path"
    }
    return Get-Content -Path $path -Raw | ConvertFrom-Json
}

function Wait-UntilScheduledSwissStart {
    if ($SkipWait) {
        Write-Host "[schedule] SkipWait is set. Starting immediately." -ForegroundColor Yellow
        return
    }

    $swissTz = Resolve-SwissTimeZone
    $targetSwissText = "2026-05-03 17:30:00"
    $targetSwiss = [datetime]::ParseExact(
        $targetSwissText,
        "yyyy-MM-dd HH:mm:ss",
        [System.Globalization.CultureInfo]::InvariantCulture
    )
    $targetSwiss = [datetime]::SpecifyKind($targetSwiss, [System.DateTimeKind]::Unspecified)
    $targetUtc = [System.TimeZoneInfo]::ConvertTimeToUtc($targetSwiss, $swissTz)
    $nowUtc = [datetime]::UtcNow

    if ($nowUtc -ge $targetUtc) {
        Write-Host "[schedule] Current UTC time is already past Swiss start time 2026-05-03 17:30. Starting immediately." -ForegroundColor Yellow
        return
    }

    $waitSeconds = [math]::Ceiling(($targetUtc - $nowUtc).TotalSeconds)
    Write-Host "[schedule] Waiting until Swiss time 2026-05-03 17:30:00 ($($swissTz.Id)); sleeping $waitSeconds second(s)." -ForegroundColor Cyan
    Start-Sleep -Seconds $waitSeconds
}

$RepoRootPy = Convert-ToPythonLiteral $RepoRoot
$RunTagPy = Convert-ToPythonLiteral $RunTag
$LatexEnginePy = Convert-ToPythonLiteral $LatexEngine
$PaperDirPy = Convert-ToPythonLiteral $PaperDir
$SkipPdfPy = if ($SkipPdf) { "True" } else { "False" }
$LatestOnlyPy = if ($KeepAllAssets) { "False" } else { "True" }

function New-ResultWriterPrelude {
    param(
        [string]$TaskName,
        [string]$ResultPath
    )
    $taskPy = Convert-ToPythonLiteral $TaskName
    $resultPy = Convert-ToPythonLiteral $ResultPath
    return @"
import json
import sys
from pathlib import Path
from multiprocessing import freeze_support

repo_root = Path($RepoRootPy)
result_path = Path($resultPy)
task_name = $taskPy
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

function Build-TaskPython {
    param(
        [string]$TaskName,
        [string]$ImportLine,
        [string]$CallExpression
    )
    $resultPath = Join-Path $TaskResultRoot "$TaskName.json"
    $prelude = New-ResultWriterPrelude -TaskName $TaskName -ResultPath $resultPath
    return @"
$prelude

$ImportLine

def main():
    result = $CallExpression
    write_task_payload(result)

if __name__ == "__main__":
    freeze_support()
    main()
"@
}

Wait-UntilScheduledSwissStart

$ResolvedPythonExe = Resolve-PythonExecutable -Requested $PythonExe
$startupStatus = [ordered]@{
    run_tag = $RunTag
    repo_root = $RepoRoot
    run_root = $RunRoot
    script_started_utc = [datetime]::UtcNow.ToString("o")
    python_executable = $ResolvedPythonExe
    latest_only = (-not $KeepAllAssets)
    skip_pdf = [bool]$SkipPdf
    skip_wait = [bool]$SkipWait
}
$startupStatus | ConvertTo-Json -Depth 4 | Set-Content -Path $StartupStatusPath -Encoding UTF8

$manifest = [ordered]@{
    run_tag = $RunTag
    repo_root = $RepoRoot
    scheduled_start_swiss = "2026-05-03 17:30:00"
    scheduled_timezone_label = "Switzerland / Europe-Zurich"
    latest_only = (-not $KeepAllAssets)
    skip_pdf = [bool]$SkipPdf
    latex_engine = $LatexEngine
    paper_dir = $PaperDir
    distance_max_workers = $DistanceMaxWorkers
    validation_max_workers = $ValidationMaxWorkers
    analysis_max_workers = $AnalysisMaxWorkers
    groups = @{
        group1 = @("analysis_v2")
        group2 = @("distance", "validation", "sql_eval", "dataset_subitem_sql_counts_v2")
        group3 = @("distance_query_scatter", "SQLvisualize", "appendix_tables")
    }
}
$manifestPath = Join-Path $RunRoot "orchestration_manifest.json"
$manifest | ConvertTo-Json -Depth 6 | Set-Content -Path $manifestPath -Encoding UTF8

Write-Host "[orchestrator] run root: $RunRoot" -ForegroundColor Cyan
Write-Host "[orchestrator] shared run_tag: $RunTag" -ForegroundColor Cyan

$group1Tasks = @()
$analysisScript = Build-TaskPython `
    -TaskName "analysis_v2" `
    -ImportLine "from src.eval.analysis.runner import run_sql_analysis" `
    -CallExpression "run_sql_analysis(run_tag=$RunTagPy, datasets=None, latest_only=$LatestOnlyPy, engines=('cli',), sql_source_version='v2', include_all_sql_statements=True, max_sql_per_dataset=0, query_row_limit=0, max_workers=$AnalysisMaxWorkers, latex_engine=(None if $LatexEnginePy == '' else $LatexEnginePy))"
$group1Tasks += Start-PythonTask -TaskName "analysis_v2" -ScriptPath (New-PythonTaskFile -TaskName "analysis_v2" -PythonBody $analysisScript)

Wait-TaskGroup -GroupName "group1" -Tasks $group1Tasks

$analysisResult = Read-TaskResult -TaskName "analysis_v2"
$AnalysisRunDirPy = Convert-ToPythonLiteral $analysisResult.run_dir

$group2Tasks = @()
$distanceScript = Build-TaskPython `
    -TaskName "distance" `
    -ImportLine "from src.eval.distance.runner import run_distance_evaluation" `
    -CallExpression "run_distance_evaluation(run_tag=$RunTagPy, datasets=None, latest_only=$LatestOnlyPy, max_workers=$DistanceMaxWorkers, latex_engine=(None if $LatexEnginePy == '' else $LatexEnginePy))"
$group2Tasks += Start-PythonTask -TaskName "distance" -ScriptPath (New-PythonTaskFile -TaskName "distance" -PythonBody $distanceScript)

$validationScript = Build-TaskPython `
    -TaskName "validation" `
    -ImportLine "from src.eval.validation.runner import run_validation_evaluation" `
    -CallExpression "run_validation_evaluation(run_tag=$RunTagPy, datasets=None, latest_only=$LatestOnlyPy, max_workers=$ValidationMaxWorkers)"
$group2Tasks += Start-PythonTask -TaskName "validation" -ScriptPath (New-PythonTaskFile -TaskName "validation" -PythonBody $validationScript)

$sqlEvalScript = Build-TaskPython `
    -TaskName "sql_eval" `
    -ImportLine "from pathlib import Path`nfrom src.eval.sql_eval.runner import run_sql_rank_stability" `
    -CallExpression "run_sql_rank_stability(run_tag=$RunTagPy, analysis_run_dir=Path($AnalysisRunDirPy), top_k=3, latex_engine=(None if $LatexEnginePy == '' else $LatexEnginePy), sql_source_version_override='v2')"
$group2Tasks += Start-PythonTask -TaskName "sql_eval" -ScriptPath (New-PythonTaskFile -TaskName "sql_eval" -PythonBody $sqlEvalScript)

$sqlCountsScript = Build-TaskPython `
    -TaskName "dataset_subitem_sql_counts_v2" `
    -ImportLine "from src.eval.dataset_subitem_sql_counts.runner import run_dataset_subitem_sql_counts" `
    -CallExpression "run_dataset_subitem_sql_counts(run_tag=$RunTagPy, datasets=None, engines=('cli',), sql_source_version='v2', latex_engine=(None if $LatexEnginePy == '' else $LatexEnginePy))"
$group2Tasks += Start-PythonTask -TaskName "dataset_subitem_sql_counts_v2" -ScriptPath (New-PythonTaskFile -TaskName "dataset_subitem_sql_counts_v2" -PythonBody $sqlCountsScript)

Wait-TaskGroup -GroupName "group2" -Tasks $group2Tasks

$distanceResult = Read-TaskResult -TaskName "distance"
$validationResult = Read-TaskResult -TaskName "validation"
$DistanceRunDirPy = Convert-ToPythonLiteral $distanceResult.run_dir
$ValidationRunDirPy = Convert-ToPythonLiteral $validationResult.run_dir

$group3Tasks = @()

$distanceScatterScript = Build-TaskPython `
    -TaskName "distance_query_scatter" `
    -ImportLine "from pathlib import Path`nfrom src.eval.distance_query_scatter.runner import run_distance_query_scatter" `
    -CallExpression "run_distance_query_scatter(run_tag=$RunTagPy, analysis_run_dir=Path($AnalysisRunDirPy), distance_run_dir=Path($DistanceRunDirPy), compile_pdf=(not $SkipPdfPy), latex_engine=(None if $LatexEnginePy == '' else $LatexEnginePy))"
$group3Tasks += Start-PythonTask -TaskName "distance_query_scatter" -ScriptPath (New-PythonTaskFile -TaskName "distance_query_scatter" -PythonBody $distanceScatterScript)

$sqlVisualizeScript = Build-TaskPython `
    -TaskName "SQLvisualize" `
    -ImportLine "from pathlib import Path`nfrom src.eval.SQLvisualize.runner import run_sqlvisualize" `
    -CallExpression "run_sqlvisualize(run_tag=$RunTagPy, analysis_run_dir=Path($AnalysisRunDirPy), validation_run_dir=Path($ValidationRunDirPy), allow_analysis_rebuild=False, force_analysis_rebuild=False, analysis_engines=('cli',), analysis_sql_source_version='v2', analysis_latest_only=$LatestOnlyPy, analysis_max_workers=$AnalysisMaxWorkers, latex_engine=(None if $LatexEnginePy == '' else $LatexEnginePy))"
$group3Tasks += Start-PythonTask -TaskName "SQLvisualize" -ScriptPath (New-PythonTaskFile -TaskName "SQLvisualize" -PythonBody $sqlVisualizeScript)

$appendixScript = Build-TaskPython `
    -TaskName "appendix_tables" `
    -ImportLine "from pathlib import Path`nfrom src.eval.appendix_tables.runner import run_appendix_table_bundle" `
    -CallExpression "run_appendix_table_bundle(run_tag=$RunTagPy, analysis_run_dir=Path($AnalysisRunDirPy), validation_run_dir=Path($ValidationRunDirPy), paper_dir=(Path($PaperDirPy) if $PaperDirPy else None), compile_pdf=(not $SkipPdfPy), latex_engine=(None if $LatexEnginePy == '' else $LatexEnginePy), runtime_audit_csv=None, rebuild_runtime_audit=False)"
$group3Tasks += Start-PythonTask -TaskName "appendix_tables" -ScriptPath (New-PythonTaskFile -TaskName "appendix_tables" -PythonBody $appendixScript)

Wait-TaskGroup -GroupName "group3" -Tasks $group3Tasks

$finalSummary = [ordered]@{
    run_tag = $RunTag
    run_root = $RunRoot
    group1 = @{
        analysis_v2 = $analysisResult
    }
    group2 = @{
        distance = $distanceResult
        validation = $validationResult
        sql_eval = (Read-TaskResult -TaskName "sql_eval")
        dataset_subitem_sql_counts_v2 = (Read-TaskResult -TaskName "dataset_subitem_sql_counts_v2")
    }
    group3 = @{
        distance_query_scatter = (Read-TaskResult -TaskName "distance_query_scatter")
        SQLvisualize = (Read-TaskResult -TaskName "SQLvisualize")
        appendix_tables = (Read-TaskResult -TaskName "appendix_tables")
    }
}

$finalSummaryPath = Join-Path $RunRoot "run_summary.json"
$finalSummary | ConvertTo-Json -Depth 8 | Set-Content -Path $finalSummaryPath -Encoding UTF8

Write-Host "[orchestrator] complete. Summary written to $finalSummaryPath" -ForegroundColor Green

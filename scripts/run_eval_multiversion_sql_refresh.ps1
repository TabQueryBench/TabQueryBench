param(
    [string]$PythonExe = "C:\Users\16943\anaconda3\python.exe",
    [string]$LatexEngine = "D:\dpan\Uni\Project\HKUNAISS\SQLagent\tools\tectonic\tectonic-0.16.9\tectonic.exe",
    [int]$AnalysisMaxWorkers = 4
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$EvalRoot = Join-Path $RepoRoot "Evaluation"
$RunTagBase = "multiversion_sql_refresh_{0}" -f (Get-Date -Format "yyyyMMdd_HHmmss")
$RunRoot = Join-Path (Join-Path $EvalRoot "orchestration") $RunTagBase
$TaskScriptRoot = Join-Path $RunRoot "task_scripts"
$TaskLogRoot = Join-Path $RunRoot "task_logs"
$TaskResultRoot = Join-Path $RunRoot "task_results"

New-Item -ItemType Directory -Force -Path $TaskScriptRoot | Out-Null
New-Item -ItemType Directory -Force -Path $TaskLogRoot | Out-Null
New-Item -ItemType Directory -Force -Path $TaskResultRoot | Out-Null

$AnalysisV3RunTag = "${RunTagBase}_analysis_v3"
$AnalysisV4RunTag = "${RunTagBase}_analysis_v4"
$AnalysisV3RunDir = Join-Path $EvalRoot "analysis\runs\$AnalysisV3RunTag"
$AnalysisV4RunDir = Join-Path $EvalRoot "analysis\runs\$AnalysisV4RunTag"
$DistanceRunDir = Join-Path $EvalRoot "distance\runs\eval_refresh_20260503_1730_zurich"
$ValidationRunDir = Join-Path $EvalRoot "validation\runs\eval_refresh_20260503_1730_zurich"

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
    return Start-Process -FilePath $PythonExe -ArgumentList @($scriptPath) -WorkingDirectory $RepoRoot -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
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

Write-AnalysisWrapper -TaskName "analysis_v3" -RunTag $AnalysisV3RunTag -SqlSourceVersion "v3"
Write-AnalysisWrapper -TaskName "analysis_v4" -RunTag $AnalysisV4RunTag -SqlSourceVersion "v4"

$breakdownScripts = @{
    subgroup = Join-Path $RepoRoot "src\eval\query_fivepart_breakdown\subgroup_breakdown\runner.py"
    conditional = Join-Path $RepoRoot "src\eval\query_fivepart_breakdown\conditional_breakdown\runner.py"
    missingness = Join-Path $RepoRoot "src\eval\query_fivepart_breakdown\missingness_breakdown\runner.py"
    tail = Join-Path $RepoRoot "src\eval\query_fivepart_breakdown\tail_breakdown\runner.py"
    cardinality = Join-Path $RepoRoot "src\eval\query_fivepart_breakdown\cardinality\runner.py"
}

foreach ($version in @("v2", "v3", "v4")) {
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

Write-DirectWrapper -TaskName "appendix_tables_v4" `
    -ImportLine "from pathlib import Path`nfrom src.eval.appendix_tables.runner import run_appendix_table_bundle" `
    -CallExpression "run_appendix_table_bundle(run_tag='${RunTagBase}_appendix_v4', analysis_run_dir=Path(r'$AnalysisV4RunDir'), validation_run_dir=Path(r'$ValidationRunDir'), paper_dir=None, compile_pdf=True, latex_engine=r'$LatexEngine', runtime_audit_csv=None, rebuild_runtime_audit=False, task_name='appendix_tables')" `
    -SqlSourceVersion "v4"

Write-RunPyWrapper -TaskName "model_radar_v4" -TargetScriptPath (Join-Path $RepoRoot "src\eval\model_radar\runner.py") -SqlSourceVersion "v4"

$phase1Breakdowns = @{}
foreach ($name in @("subgroup_v2", "conditional_v2", "missingness_v2", "tail_v2", "cardinality_v2")) {
    $phase1Breakdowns[$name] = Start-Task -TaskName $name
}
$phase1Analyses = @{
    analysis_v3 = (Start-Task -TaskName "analysis_v3")
    analysis_v4 = (Start-Task -TaskName "analysis_v4")
}

Wait-TaskGroup -GroupName "phase1_breakdowns_v2" -Tasks $phase1Breakdowns
Wait-TaskGroup -GroupName "phase1_analysis_v3_v4" -Tasks $phase1Analyses

$phase2 = @{}
foreach ($name in @(
    "sql_eval_v3", "dataset_counts_v3", "distance_query_scatter_v3", "sqlvisualize_v3",
    "subgroup_v3", "conditional_v3", "missingness_v3", "tail_v3", "cardinality_v3",
    "sql_eval_v4", "dataset_counts_v4", "distance_query_scatter_v4", "sqlvisualize_v4",
    "subgroup_v4", "conditional_v4", "missingness_v4", "tail_v4", "cardinality_v4",
    "appendix_tables_v4"
)) {
    $phase2[$name] = Start-Task -TaskName $name
}

Wait-TaskGroup -GroupName "phase2_v3_v4_downstream" -Tasks $phase2

$phase3 = @{
    model_radar_v4 = (Start-Task -TaskName "model_radar_v4")
}
Wait-TaskGroup -GroupName "phase3_model_radar_v4" -Tasks $phase3

@{
    status = "ok"
    run_tag = $RunTagBase
    run_root = $RunRoot
    analysis_v3_run_dir = $AnalysisV3RunDir
    analysis_v4_run_dir = $AnalysisV4RunDir
} | ConvertTo-Json -Depth 4 | Set-Content -Path (Join-Path $RunRoot "run_summary.json") -Encoding UTF8

Write-Host "[done] multiversion SQL refresh completed" -ForegroundColor Green

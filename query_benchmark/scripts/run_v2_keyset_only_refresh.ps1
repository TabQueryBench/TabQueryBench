param(
    [string]$PythonExe = "C:\Users\16943\anaconda3\python.exe",
    [int]$AnalysisWorkers = 4
)

$ErrorActionPreference = "Stop"

$RepoRoot = "."
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$RunTag = "v2_keyset_only_refresh_$Timestamp"
$OrchDir = Join-Path $RepoRoot "Evaluation\orchestration\$RunTag"
$LogDir = Join-Path $OrchDir "task_logs"
$ResultDir = Join-Path $OrchDir "task_results"

New-Item -ItemType Directory -Force -Path $OrchDir, $LogDir, $ResultDir | Out-Null

function New-Task {
    param(
        [string]$Name,
        [string]$Command
    )
    return [pscustomobject]@{
        Name = $Name
        Command = $Command
    }
}

function Start-Task {
    param([pscustomobject]$Task)
    $stdout = Join-Path $LogDir "$($Task.Name).stdout.log"
    $stderr = Join-Path $LogDir "$($Task.Name).stderr.log"
    $wrapped = "set EVAL_SQL_SOURCE_VERSION=v2 && cd /d `"$RepoRoot`" && $($Task.Command)"
    $proc = Start-Process -FilePath "cmd.exe" -ArgumentList "/c $wrapped" -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
    return [pscustomobject]@{
        Name = $Task.Name
        Process = $proc
        Stdout = $stdout
        Stderr = $stderr
        Command = $Task.Command
    }
}

function Wait-TaskGroup {
    param(
        [string]$GroupName,
        [array]$Tasks
    )
    $running = @()
    foreach ($task in $Tasks) {
        $running += Start-Task -Task $task
    }
    $statuses = @()
    foreach ($item in $running) {
        Wait-Process -Id $item.Process.Id
        $item.Process.Refresh()
        $exitCode = $item.Process.ExitCode
        $status = [pscustomobject]@{
            group = $GroupName
            name = $item.Name
            pid = $item.Process.Id
            exit_code = $exitCode
            stdout = $item.Stdout
            stderr = $item.Stderr
            command = $item.Command
        }
        $statuses += $status
        $status | ConvertTo-Json -Depth 4 | Set-Content -Path (Join-Path $ResultDir "$($item.Name).json") -Encoding UTF8
    }
    $failed = $statuses | Where-Object { $_.exit_code -ne 0 }
    if ($failed) {
        $failed | ConvertTo-Json -Depth 4 | Set-Content -Path (Join-Path $OrchDir "failed_group_$GroupName.json") -Encoding UTF8
        throw "Task group '$GroupName' failed. See logs under $LogDir"
    }
}

$manifest = [ordered]@{
    run_tag = $RunTag
    started_at = (Get-Date).ToString("o")
    sql_source_version = "v2"
    scoring_mode = "key_set_score_only"
    analysis_workers = $AnalysisWorkers
    repo_root = $RepoRoot
    python_exe = $PythonExe
    orchestration_dir = $OrchDir
}
$manifest | ConvertTo-Json -Depth 5 | Set-Content -Path (Join-Path $OrchDir "orchestration_manifest.json") -Encoding UTF8

$analysisCmd = "`"$PythonExe`" -m src.eval.analysis.runner --run-tag $RunTag --sql-source-version v2 --engines cli --max-workers $AnalysisWorkers"
Wait-TaskGroup -GroupName "analysis" -Tasks @(
    (New-Task -Name "analysis_v2_keyset" -Command $analysisCmd)
)

$group2 = @(
    (New-Task -Name "sql_eval_v2_keyset" -Command "`"$PythonExe`" -m src.eval.sql_eval.runner --run-tag $RunTag"),
    (New-Task -Name "distance_query_scatter_v2_keyset" -Command "`"$PythonExe`" -m src.eval.distance_query_scatter.runner --run-tag $RunTag"),
    (New-Task -Name "sqlvisualize_v2_keyset" -Command "`"$PythonExe`" -m src.eval.SQLvisualize.runner --run-tag $RunTag --no-rebuild-analysis"),
    (New-Task -Name "appendix_tables_v2_keyset" -Command "`"$PythonExe`" -m src.eval.appendix_tables.runner --run-tag $RunTag"),
    (New-Task -Name "subgroup_breakdown_v2_keyset" -Command "`"$PythonExe`" -m src.eval.query_fivepart_breakdown.subgroup_breakdown.runner"),
    (New-Task -Name "conditional_breakdown_v2_keyset" -Command "`"$PythonExe`" -m src.eval.query_fivepart_breakdown.conditional_breakdown.runner"),
    (New-Task -Name "missingness_breakdown_v2_keyset" -Command "`"$PythonExe`" -m src.eval.query_fivepart_breakdown.missingness_breakdown.runner"),
    (New-Task -Name "tail_breakdown_v2_keyset" -Command "`"$PythonExe`" -m src.eval.query_fivepart_breakdown.tail_breakdown.runner")
)
Wait-TaskGroup -GroupName "core_downstream" -Tasks $group2

$group3 = @(
    (New-Task -Name "strength_diagnostic_v2_keyset" -Command "`"$PythonExe`" -m src.eval.query_fivepart_breakdown.missingness_breakdown.strength_diagnostic.runner"),
    (New-Task -Name "strict_pairwise_diagnostic_v2_keyset" -Command "`"$PythonExe`" -m src.eval.query_fivepart_breakdown.missingness_breakdown.strict_pairwise_diagnostic.runner"),
    (New-Task -Name "model_radar_v2_keyset" -Command "`"$PythonExe`" -m src.eval.model_radar.runner"),
    (New-Task -Name "benchmark_overall_table_v2_keyset" -Command "`"$PythonExe`" Evaluation\benchmark_overall_table\build_overall_benchmark_table.py"),
    (New-Task -Name "benchmark_query_category_table_v2_keyset" -Command "`"$PythonExe`" Evaluation\benchmark_query_category_table\build_benchmark_query_category_table.py"),
    (New-Task -Name "query_fivepart_overview_v2_keyset" -Command "`"$PythonExe`" -m src.eval.query_fivepart_breakdown.runner")
)
Wait-TaskGroup -GroupName "paper_derivatives" -Tasks $group3

Wait-TaskGroup -GroupName "overview_regenerated" -Tasks @(
    (New-Task -Name "overview_regenerated_v2_keyset" -Command "`"$PythonExe`" -m src.eval.overview_regenerated.runner")
)

$summary = [ordered]@{
    run_tag = $RunTag
    finished_at = (Get-Date).ToString("o")
    status = "completed"
}
$summary | ConvertTo-Json -Depth 4 | Set-Content -Path (Join-Path $OrchDir "run_summary.json") -Encoding UTF8
Write-Output "Completed $RunTag"

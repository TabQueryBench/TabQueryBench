$ErrorActionPreference = 'Stop'

$RepoRoot = '.'
$PythonExe = 'C:\Users\16943\anaconda3\python.exe'
$RunTag = 'v2_trainonly_refresh_new_assets_20260505_2355'
$OrchDir = Join-Path $RepoRoot "Evaluation\orchestration\$RunTag"
$TaskLogsDir = Join-Path $OrchDir 'task_logs'
$TaskResultsDir = Join-Path $OrchDir 'task_results'
$CacheRoot = "F:\SQLagentAnalysisCache\$RunTag"

New-Item -ItemType Directory -Force -Path $TaskLogsDir | Out-Null
New-Item -ItemType Directory -Force -Path $TaskResultsDir | Out-Null
New-Item -ItemType Directory -Force -Path $CacheRoot | Out-Null

function Start-TaskProcess {
    param(
        [string]$TaskName,
        [string]$Module,
        [string[]]$Arguments
    )
    $stdout = Join-Path $TaskLogsDir "$TaskName.stdout.log"
    $stderr = Join-Path $TaskLogsDir "$TaskName.stderr.log"
    $argList = @('-m', $Module) + $Arguments
    $proc = Start-Process -FilePath $PythonExe -ArgumentList $argList -WorkingDirectory $RepoRoot -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
    return [pscustomobject]@{
        TaskName = $TaskName
        Process = $proc
        Stdout = $stdout
        Stderr = $stderr
    }
}

function Wait-TaskProcess {
    param(
        [Parameter(Mandatory=$true)]$Task
    )
    Wait-Process -Id $Task.Process.Id
    $Task.Process.Refresh()
    $exitCode = $Task.Process.ExitCode
    $payload = [ordered]@{
        task = $Task.TaskName
        pid = $Task.Process.Id
        exit_code = $exitCode
        stdout = $Task.Stdout
        stderr = $Task.Stderr
        completed_utc = [DateTime]::UtcNow.ToString('o')
    }
    $payload | ConvertTo-Json -Depth 4 | Set-Content -Path (Join-Path $TaskResultsDir "$($Task.TaskName).json") -Encoding UTF8
    if ($exitCode -ne 0) {
        throw "Task $($Task.TaskName) failed with exit code $exitCode"
    }
}

$manifest = [ordered]@{
    task = 'trainonly_refresh_new_assets'
    run_tag = $RunTag
    created_utc = [DateTime]::UtcNow.ToString('o')
    repo_root = $RepoRoot
    python = $PythonExe
    synthetic_root = 'Benchmark-trainonly-v1'
    phases = @(
        [ordered]@{
            phase = 'analysis'
            tasks = @('analysis_trainonly')
        },
        [ordered]@{
            phase = 'downstream'
            tasks = @('sql_eval_trainonly', 'subgroup_trainonly', 'conditional_trainonly')
        }
    )
}
$manifest | ConvertTo-Json -Depth 6 | Set-Content -Path (Join-Path $OrchDir 'orchestration_manifest.json') -Encoding UTF8

$analysisArgs = @(
    '--run-tag', $RunTag,
    '--sql-source-version', 'v2',
    '--engines', 'cli',
    '--latest-only',
    '--max-workers', '8',
    '--cache-root', $CacheRoot,
    '--root-names', 'Benchmark-trainonly-v1',
    '--skip-final-publish'
)
$analysisTask = Start-TaskProcess -TaskName 'analysis_trainonly' -Module 'src.eval.analysis.runner' -Arguments $analysisArgs
Wait-TaskProcess -Task $analysisTask

$analysisRunDir = Join-Path $RepoRoot "Evaluation\analysis\runs\$RunTag"

$downstreamTasks = @(
    (Start-TaskProcess -TaskName 'sql_eval_trainonly' -Module 'src.eval.sql_eval.runner' -Arguments @('--run-tag', "${RunTag}_sql_eval", '--analysis-run-dir', $analysisRunDir, '--skip-final-publish')),
    (Start-TaskProcess -TaskName 'subgroup_trainonly' -Module 'src.eval.query_fivepart_breakdown.subgroup_breakdown.runner' -Arguments @('--analysis-run-dir', $analysisRunDir, '--skip-final-publish')),
    (Start-TaskProcess -TaskName 'conditional_trainonly' -Module 'src.eval.query_fivepart_breakdown.conditional_breakdown.runner' -Arguments @('--analysis-run-dir', $analysisRunDir, '--skip-final-publish'))
)

foreach ($task in $downstreamTasks) {
    Wait-TaskProcess -Task $task
}

$summary = [ordered]@{
    task = 'trainonly_refresh_new_assets'
    run_tag = $RunTag
    completed_utc = [DateTime]::UtcNow.ToString('o')
    analysis_run_dir = $analysisRunDir
    task_results_dir = $TaskResultsDir
}
$summary | ConvertTo-Json -Depth 4 | Set-Content -Path (Join-Path $OrchDir 'run_summary.json') -Encoding UTF8

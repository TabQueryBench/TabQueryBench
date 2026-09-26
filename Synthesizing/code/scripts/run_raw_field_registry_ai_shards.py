#!/usr/bin/env python3
"""Run raw field-registry Claude review in dataset-level parallel shards."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from urllib import request


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = REPO_ROOT / "raw_data" / "tabular_datasets"
BUILD_SCRIPT = REPO_ROOT / "code" / "scripts" / "build_raw_field_registry_with_ai_review.py"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def count_columns(dataset_dir: Path) -> int:
    train_path = dataset_dir / f"{dataset_dir.name}-train.csv"
    if not train_path.exists():
        return 0
    with train_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        return len(next(csv.reader(handle)))


def dataset_review_counts(dataset_dir: Path) -> dict[str, object]:
    reviews_path = dataset_dir / "metadata" / "field_ai_reviews.jsonl"
    success = 0
    errors = 0
    cached = 0
    last_column = ""
    if not reviews_path.exists():
        return {"success": success, "errors": errors, "cached": cached, "last_column": last_column}
    with reviews_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("ai_provider") != "claude":
                continue
            last_column = str(record.get("column") or last_column)
            if record.get("ai_cached"):
                cached += 1
            if record.get("ai_error"):
                errors += 1
            elif isinstance(record.get("ai_review"), dict):
                success += 1
    return {"success": success, "errors": errors, "cached": cached, "last_column": last_column}


def list_datasets(data_root: Path, dataset_ids: str, order: str) -> list[tuple[str, int]]:
    if dataset_ids:
        ids = [item.strip() for item in dataset_ids.split(",") if item.strip()]
    else:
        ids = [path.name for path in sorted(data_root.iterdir()) if path.is_dir() and not path.name.startswith(".")]
    datasets = [(dataset_id, count_columns(data_root / dataset_id)) for dataset_id in ids]
    if order == "largest-first":
        datasets.sort(key=lambda item: (-item[1], item[0]))
    elif order == "smallest-first":
        datasets.sort(key=lambda item: (item[1], item[0]))
    else:
        datasets.sort(key=lambda item: item[0])
    return datasets


def load_status(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {str(item["dataset_id"]): item for item in payload.get("datasets", []) if "dataset_id" in item}


def write_status(path: Path, status: dict[str, dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": now_iso(),
        "datasets": [status[key] for key in sorted(status)],
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def notify_email(notify_url: str, to_addr: str, note: str, *, phase: str | None = None, job_finished: bool = False, exit_code: int | None = None) -> None:
    if not to_addr:
        return
    payload: dict[str, object] = {
        "pid": os.getpid(),
        "to": to_addr,
        "note": note,
        "invoke_child": " ".join(sys.argv),
    }
    if phase:
        payload["phase"] = phase
    if job_finished:
        payload["job_finished"] = True
        payload["exit_code"] = 0 if exit_code is None else exit_code
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(
        notify_url,
        data=data,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with request.urlopen(req, timeout=30) as resp:
        resp.read()


def start_dataset(
    dataset_id: str,
    *,
    data_root: Path,
    log_dir: Path,
    report_dir: Path,
    ai_timeout_sec: int,
    claude_home: Path | None,
) -> tuple[subprocess.Popen[bytes], object, Path]:
    log_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{dataset_id}.log"
    log_handle = log_path.open("ab", buffering=0)
    log_handle.write(f"\n===== {now_iso()} start {dataset_id} =====\n".encode())
    env = os.environ.copy()
    if claude_home is not None:
        env["HOME"] = str(claude_home)
        env.setdefault("CLAUDE_CONFIG_DIR", str(claude_home / ".claude"))
    cmd = [
        sys.executable,
        "-u",
        str(BUILD_SCRIPT),
        "--data-root",
        str(data_root),
        "--dataset-ids",
        dataset_id,
        "--ai-provider",
        "claude",
        "--ai-timeout-sec",
        str(ai_timeout_sec),
        "--report-path",
        str(report_dir / f"{dataset_id}.json"),
    ]
    proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), stdout=log_handle, stderr=subprocess.STDOUT, env=env)
    return proc, log_handle, log_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Claude field-registry reviews across datasets in parallel.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--dataset-ids", default="", help="Comma-separated dataset ids. Defaults to all datasets.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--ai-timeout-sec", type=int, default=120)
    parser.add_argument("--order", choices=["name", "largest-first", "smallest-first"], default="largest-first")
    parser.add_argument("--claude-home", type=Path, default=None, help="Private HOME containing Claude auth/config.")
    parser.add_argument("--log-dir", type=Path, default=REPO_ROOT / "tmp" / "logs" / "raw_field_registry_ai_shards")
    parser.add_argument("--report-dir", type=Path, default=REPO_ROOT / "tmp" / "raw_field_registry_ai_shard_reports")
    parser.add_argument("--status-path", type=Path, default=REPO_ROOT / "tmp" / "raw_field_registry_ai_shards_status.json")
    parser.add_argument("--retry-failed", action="store_true", help="Retry datasets previously marked failed.")
    parser.add_argument("--notify-email", default="", help="Email run start and per-dataset completion notifications.")
    parser.add_argument("--notify-url", default="http://127.0.0.1:18765/notify")
    args = parser.parse_args()

    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    status = load_status(args.status_path)
    queue = deque()
    for dataset_id, field_count in list_datasets(args.data_root, args.dataset_ids, args.order):
        prior = status.get(dataset_id, {})
        if prior.get("status") == "complete":
            continue
        if prior.get("status") == "failed" and not args.retry_failed:
            continue
        queue.append((dataset_id, field_count))
        status[dataset_id] = {
            "dataset_id": dataset_id,
            "field_count": field_count,
            "status": "queued",
            "updated_at": now_iso(),
        }
    write_status(args.status_path, status)
    run_start_note = (
        "TabQueryBench preprocessing AI review START\n"
        f"workers={args.workers}\n"
        f"queued_datasets={len(queue)}\n"
        f"ai_timeout_sec={args.ai_timeout_sec}\n"
        f"status_path={args.status_path}\n"
        f"log_dir={args.log_dir}"
    )
    try:
        notify_email(args.notify_url, args.notify_email, run_start_note, phase="start")
    except Exception as exc:  # noqa: BLE001
        print(f"[ai-shards] WARNING: start email failed: {exc}", flush=True)

    running: dict[str, tuple[subprocess.Popen[bytes], object, Path, float]] = {}
    final_rc = 0
    while queue or running:
        while queue and len(running) < args.workers:
            dataset_id, field_count = queue.popleft()
            proc, log_handle, log_path = start_dataset(
                dataset_id,
                data_root=args.data_root,
                log_dir=args.log_dir,
                report_dir=args.report_dir,
                ai_timeout_sec=args.ai_timeout_sec,
                claude_home=args.claude_home,
            )
            running[dataset_id] = (proc, log_handle, log_path, time.time())
            status[dataset_id] = {
                "dataset_id": dataset_id,
                "field_count": field_count,
                "status": "running",
                "pid": proc.pid,
                "log_path": str(log_path),
                "started_at": now_iso(),
                "updated_at": now_iso(),
            }
            write_status(args.status_path, status)
            print(f"[ai-shards] started {dataset_id} pid={proc.pid} fields={field_count} log={log_path}", flush=True)

        time.sleep(5)
        for dataset_id, (proc, log_handle, log_path, started_at) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            log_handle.write(f"===== {now_iso()} exit {dataset_id} rc={rc} =====\n".encode())
            log_handle.close()
            elapsed_sec = round(time.time() - started_at, 1)
            status[dataset_id].update(
                {
                    "status": "complete" if rc == 0 else "failed",
                    "returncode": rc,
                    "elapsed_sec": elapsed_sec,
                    "updated_at": now_iso(),
                }
            )
            review_counts = dataset_review_counts(args.data_root / dataset_id)
            status[dataset_id].update(
                {
                    "ai_success": review_counts["success"],
                    "ai_errors": review_counts["errors"],
                    "ai_cached": review_counts["cached"],
                    "last_column": review_counts["last_column"],
                }
            )
            write_status(args.status_path, status)
            del running[dataset_id]
            print(f"[ai-shards] finished {dataset_id} rc={rc} elapsed={elapsed_sec}s", flush=True)
            if rc != 0:
                final_rc = 1
            note = (
                "TabQueryBench preprocessing AI review dataset finished\n"
                f"dataset={dataset_id}\n"
                f"status={'complete' if rc == 0 else 'failed'}\n"
                f"returncode={rc}\n"
                f"elapsed_sec={elapsed_sec}\n"
                f"fields={status[dataset_id].get('field_count')}\n"
                f"ai_success={review_counts['success']}\n"
                f"ai_errors={review_counts['errors']}\n"
                f"ai_cached={review_counts['cached']}\n"
                f"last_column={review_counts['last_column']}\n"
                f"log_path={log_path}\n"
                f"status_path={args.status_path}"
            )
            try:
                notify_email(args.notify_url, args.notify_email, note, job_finished=True, exit_code=rc)
            except Exception as exc:  # noqa: BLE001
                print(f"[ai-shards] WARNING: dataset email failed for {dataset_id}: {exc}", flush=True)

    try:
        notify_email(
            args.notify_url,
            args.notify_email,
            f"TabQueryBench preprocessing AI review ALL DONE\nstatus_path={args.status_path}",
            job_finished=True,
            exit_code=final_rc,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[ai-shards] WARNING: final email failed: {exc}", flush=True)
    return final_rc


if __name__ == "__main__":
    raise SystemExit(main())

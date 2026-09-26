#!/usr/bin/env python3
"""Ground several Claude models over the V11 dataset set with one shared worker pool.

Round-based batching (run N datasets, wait for all N, start the next N) idles on the
slowest dataset of every round. This keeps a fixed number of datasets in flight across
all requested models instead, so a worker picks up the next dataset the moment it is free.

Concurrency is capped because the Claude CLI degrades past ~16 parallel calls: measured
on claude-haiku-4-5 with a real binding prompt, 16 parallel calls all succeeded (~17x a
single call) while 32 returned empty output for 13 of 32 and 49 for 28 of 49. Those are
silent failures, not rate-limit errors, so the cap is the useful limit rather than a
politeness setting.

The weekly subscription quota is read between dispatches; once it reaches the stop
threshold no further datasets start and the in-flight ones finish.
"""

from __future__ import annotations

import argparse
import json
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tqb_query.subitem_workload_v2.paths import default_dataset_ids_for_line_version  # noqa: E402
from tqb_query.workload_grounding.v10_versions import format_grounding_version, resolve_model_version  # noqa: E402

QUOTA_REFRESH_SECONDS = 300
# The 5-hour session window refills on its own, so a session limit pauses the pool
# instead of failing the datasets that happened to be in flight.
SESSION_PAUSE_SECONDS = 900
SESSION_LIMIT = "__session_limit__"


def log(message: str) -> None:
    print(f"[{datetime.now(timezone.utc).astimezone():%H:%M:%S}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True, help="Grounding models, in the order to finish them.")
    parser.add_argument("--workers", type=int, default=16, help="Datasets in flight across all models.")
    parser.add_argument("--run", type=int, default=1, help="Run number; 2 grounds v11.x.y-2_<model>, and so on.")
    parser.add_argument("--week-stop-percent", type=int, default=90)
    parser.add_argument("--max-attempts", type=int, default=3, help="Attempts per dataset before giving up.")
    parser.add_argument(
        "--data-root",
        type=Path,
        # PROJECT_ROOT is Query/code, so the datasets sit two levels up; resolve it here
        # because the child process runs from a different working directory.
        default=(PROJECT_ROOT.parent.parent / "Synthesizing/raw_data/tabular_datasets").resolve(),
    )
    parser.add_argument("--log-root", type=Path, required=True)
    return parser.parse_args()


class QuotaGuard:
    """One weekly-usage reading shared by every worker, refreshed on a timer."""

    def __init__(self, log_root: Path, stop_percent: int) -> None:
        self._log_root = log_root
        self._stop_percent = stop_percent
        self._lock = threading.Lock()
        self._checked_at = 0.0
        self._used = -1
        self._paused_until = 0.0

    def note_session_limit(self) -> None:
        """A worker hit the 5-hour session limit: hold every worker until it refills."""
        with self._lock:
            if time.monotonic() < self._paused_until:
                return
            self._paused_until = time.monotonic() + SESSION_PAUSE_SECONDS
            log(f"[session-limit] pausing all workers for {SESSION_PAUSE_SECONDS // 60}m")

    def wait_if_paused(self) -> None:
        while True:
            with self._lock:
                remaining = self._paused_until - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(remaining, 30))

    def _read(self) -> int:
        out_dir = self._log_root / "usage" / datetime.now().strftime("%H%M%S")
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(
                [sys.executable, str(PROJECT_ROOT / "scripts/capture_claude_usage.py"), "--output-dir", str(out_dir)],
                capture_output=True,
                timeout=180,
                check=True,
            )
            files = sorted(out_dir.glob("*summary.json"))
            payload = json.loads(files[-1].read_text(encoding="utf-8"))
            return int((payload.get("week_all_models") or {}).get("used_percent") or -1)
        except Exception:
            return -1

    def should_stop(self) -> tuple[bool, int]:
        with self._lock:
            if time.monotonic() - self._checked_at > QUOTA_REFRESH_SECONDS:
                used = self._read()
                self._checked_at = time.monotonic()
                if used >= 0:
                    self._used = used
                    log(f"[quota] weekly used={used}% (stop at {self._stop_percent}%)")
                elif self._used < 0:
                    log("[quota] usage unreadable and never read; stopping to stay safe")
                    return True, -1
            return self._used >= self._stop_percent, self._used


def ground_dataset(model: str, dataset_id: str, version: str, args: argparse.Namespace) -> tuple[bool, float, str]:
    log_path = args.log_root / version / f"{dataset_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/build_subitem_workload_v2_inventory.py"),
        "--line-version", "v11",
        "--planner-kind", "agent-select-bind",
        "--planner-model", model,
        "--grounding-version", version,
        "--dataset-ids", dataset_id,
        "--data-root", str(args.data_root),
        "--agent-bind-problems-per-template", "1",
        "--resume",
    ]
    with log_path.open("a", encoding="utf-8") as handle:
        result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, cwd=PROJECT_ROOT.parent)
    elapsed = time.monotonic() - started
    if result.returncode == 0:
        return True, elapsed, ""
    text = log_path.read_text(encoding="utf-8", errors="replace")
    if "session limit" in text.lower() or "weekly limit" in text.lower():
        return False, elapsed, SESSION_LIMIT
    tail = ""
    for line in text.splitlines()[::-1]:
        if line.strip().startswith(("Error", "Traceback")) or "Error:" in line:
            tail = line.strip()[:160]
            break
    return False, elapsed, tail or f"rc={result.returncode}"


def pending_datasets(version: str) -> list[str]:
    inventory_dir = PROJECT_ROOT / "data/workload_grounding_v11/variants" / version / "inventories"
    done = {path.name.split("_inventory_")[0] for path in inventory_dir.glob(f"*_inventory_{version}.json")}
    return [ds for ds in default_dataset_ids_for_line_version("v11") if ds not in done]


def main() -> None:
    args = parse_args()
    args.log_root.mkdir(parents=True, exist_ok=True)
    guard = QuotaGuard(args.log_root, args.week_stop_percent)

    tasks: queue.Queue = queue.Queue()
    versions: dict[str, str] = {}
    total = 0
    for model in args.models:
        version = resolve_model_version(
            model, line_family="v11", grounding_version=format_grounding_version("v11", model, args.run)
        ).grounding_version
        versions[model] = version
        todo = pending_datasets(version)
        log(f"[plan] {model} -> {version}: {len(todo)} datasets pending")
        for dataset_id in todo:
            tasks.put((model, dataset_id, 1))
            total += 1
    if not total:
        log("[done] nothing pending")
        return
    log(f"[plan] {total} dataset-model tasks, {args.workers} in flight")

    state = {"ok": 0, "failed": 0, "stopped": False, "in_flight": 0}
    state_lock = threading.Lock()
    started_at = time.monotonic()

    def worker(worker_id: int) -> None:
        while True:
            try:
                model, dataset_id, attempt = tasks.get_nowait()
            except queue.Empty:
                # A paused worker will requeue its dataset, so an empty queue only means
                # "done" once nothing is still running.
                with state_lock:
                    idle = state["in_flight"] == 0
                if idle:
                    return
                time.sleep(5)
                continue
            with state_lock:
                state["in_flight"] += 1
            guard.wait_if_paused()
            stop, used = guard.should_stop()
            if stop:
                with state_lock:
                    state["in_flight"] -= 1
                    if not state["stopped"]:
                        log(f"[stop] weekly quota at {used}%; not starting more datasets")
                        state["stopped"] = True
                tasks.task_done()
                continue
            version = versions[model]
            ok, elapsed, error = ground_dataset(model, dataset_id, version, args)
            if not ok and error == SESSION_LIMIT:
                # Not the dataset's fault: requeue it unchanged and let the window refill.
                guard.note_session_limit()
                tasks.put((model, dataset_id, attempt))
                with state_lock:
                    state["in_flight"] -= 1
                tasks.task_done()
                continue
            with state_lock:
                state["in_flight"] -= 1
                if ok:
                    state["ok"] += 1
                else:
                    if attempt < args.max_attempts:
                        tasks.put((model, dataset_id, attempt + 1))
                        log(f"[retry] {version} {dataset_id} attempt {attempt} failed in {elapsed:.0f}s: {error}")
                    else:
                        state["failed"] += 1
                        log(f"[failed] {version} {dataset_id} after {attempt} attempts: {error}")
                finished = state["ok"] + state["failed"]
                if ok:
                    rate = (time.monotonic() - started_at) / max(finished, 1)
                    eta = rate * (total - finished) / 60
                    log(
                        f"[ok] {version} {dataset_id} in {elapsed:.0f}s "
                        f"({finished}/{total} done, ~{eta:.0f}m left)"
                    )
            tasks.task_done()

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(args.workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    elapsed = (time.monotonic() - started_at) / 60
    log(f"[done] ok={state['ok']} failed={state['failed']} in {elapsed:.0f}m")
    for model, version in versions.items():
        remaining = pending_datasets(version)
        log(f"[final] {model} {version}: {49 - len(remaining)}/49 complete" + (f", pending {remaining}" if remaining else ""))


if __name__ == "__main__":
    main()

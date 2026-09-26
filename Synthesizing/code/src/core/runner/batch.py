"""Run many (dataset, model) jobs through core.runner.runner with GPU/CPU slot scheduling.

  PYTHONPATH=Synthesizing/code/src python -m core.runner.batch --datasets all --models all --preset smoke --gpus 0,1
  PYTHONPATH=Synthesizing/code/src python -m core.runner.batch --datasets all --models all \
      --gpu-plan "1=*,0=ctgan,tvae,tabpfgen" --jobs-per-gpu 2 --cpu-jobs 3

Each job runs in its own subprocess (so one failure never stops the batch), writes a
log under <output>/_batch/<batch_id>/logs/, and the batch keeps batch_status.csv /
batch_summary.json up to date while it runs.

Scheduling:
- `--gpu-plan` pins which models may run on which GPU, e.g. keep a shared/preemptible GPU
  for short jobs only: `--gpu-plan "1=*,0=ctgan,tvae,tabpfgen"`.
- `--order longest-first` (default) starts the most expensive jobs first, which shortens the
  total wall-clock time of the batch (estimates come from row counts and per-model weights).
- Failed jobs can simply be re-run later with `--skip-done`, which keeps successful runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .config import SUPPORTED_MODELS, get_new_tabular_dataset_root, get_output_base

# (fixed seconds, seconds per 1000 training rows, seconds per 1000 generated rows).
# Calibrated on 2026-09-16 against measured runs of the WorkloadTest datasets with the
# "fast" preset; only used to order jobs and print an ETA, so wrong values cost ordering
# quality and nothing else. Re-calibrate with: python -m core.runner.calibrate (or by hand
# from batch_status.csv, comparing duration_sec against est_sec).
MODEL_COST_WEIGHTS: Dict[str, tuple] = {
    "arf": (60.0, 2.0, 0.5),
    "bayesnet": (50.0, 0.3, 0.3),
    "ctgan": (260.0, 7.0, 1.0),
    "forestdiffusion": (120.0, 15.0, 3.0),
    "realtabformer": (120.0, 18.0, 20.0),
    "tabbyflow": (200.0, 16.0, 1.5),
    "tabddpm": (60.0, 22.0, 3.0),
    "tabdiff": (200.0, 16.0, 1.5),
    "tabpfgen": (40.0, 0.5, 1.2),
    "tabsyn": (60.0, 26.0, 1.0),
    "tvae": (65.0, 4.2, 1.0),
}
DOCKER_OVERHEAD_SEC = 20.0


@dataclass
class Job:
    dataset: str
    model: str
    needs_gpu: bool
    est_sec: float = 0.0
    status: str = "pending"
    device: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    duration_sec: Optional[float] = None
    exit_code: Optional[int] = None
    run_dir: Optional[str] = None
    synthetic_csv: Optional[str] = None
    reason: Optional[str] = None
    quality_warnings: int = 0
    attempts: int = 0
    log: Optional[str] = None
    extra: Dict[str, str] = field(default_factory=dict)


def list_prepared_datasets(root: Path) -> List[str]:
    out = []
    if not root.is_dir():
        return out
    for d in sorted(root.iterdir()):
        if d.is_dir() and (d / f"{d.name}-train.csv").is_file() and (d / "metadata_core" / "field_registry.json").is_file():
            out.append(d.name)
    return out


def dataset_sizes(root: Path, dataset: str) -> tuple:
    """(training rows actually used, rows to generate) from the prepared registry."""
    path = root / dataset / "metadata_core" / "field_registry.json"
    try:
        reg = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 10_000, 10_000
    train_rows = int((reg.get("row_counts") or {}).get("train") or 0)
    generation = reg.get("generation") or {}
    cap = generation.get("max_train_rows")
    used = min(train_rows, int(cap)) if cap else train_rows
    num_rows = generation.get("num_rows")
    gen_rows = int(num_rows) if isinstance(num_rows, int) and not isinstance(num_rows, bool) else train_rows
    return max(used, 1), max(gen_rows, 1)


def estimate_seconds(model: str, train_rows: int, gen_rows: int) -> float:
    fixed, w_train, w_gen = MODEL_COST_WEIGHTS.get(model, (60.0, 10.0, 2.0))
    return DOCKER_OVERHEAD_SEC + fixed + train_rows / 1000.0 * w_train + gen_rows / 1000.0 * w_gen


def model_needs_gpu(model: str) -> bool:
    from .adapters import get_adapter

    try:
        return bool(get_adapter(model)._needs_gpu)
    except Exception:
        return True


def _latest_run_dir(output_base: Path, dataset: str, model: str, since: float) -> Optional[Path]:
    model_dir = output_base / dataset / model
    if not model_dir.is_dir():
        return None
    runs = [d for d in model_dir.iterdir() if d.is_dir() and d.stat().st_mtime >= since - 1]
    return max(runs, key=lambda d: d.stat().st_mtime) if runs else None


def _has_successful_run(output_base: Path, dataset: str, model: str, preset: Optional[str]) -> Optional[Path]:
    model_dir = output_base / dataset / model
    if not model_dir.is_dir():
        return None
    for run in sorted(model_dir.iterdir(), key=lambda d: d.stat().st_mtime, reverse=True):
        rr, rc = run / "runtime_result.json", run / "run_config.json"
        if not rr.is_file():
            continue
        try:
            result = json.loads(rr.read_text(encoding="utf-8"))
            config = json.loads(rc.read_text(encoding="utf-8")) if rc.is_file() else {}
        except ValueError:
            continue
        if result.get("generate_status") != "success":
            continue
        if (config.get("cli_args") or {}).get("preset") == preset:
            return run
    return None


@dataclass
class DeviceRule:
    """Which jobs a GPU accepts, and how many at a time."""

    models: set
    max_est_sec: Optional[float] = None
    slots: int = 1

    def accepts(self, model: str, est_sec: float) -> bool:
        if model not in self.models:
            return False
        return self.max_est_sec is None or est_sec <= self.max_est_sec


def parse_gpu_plan(plan: str, gpus: str, models: Sequence[str], default_slots: int) -> Dict[str, DeviceRule]:
    """Parse ``--gpu-plan``.

    Entry forms (comma-separated), each optionally suffixed with ``:SLOTS``:
      ``1=*``              every model, no limit
      ``0=ctgan+tvae``     only these models
      ``0=<=12m``          only jobs estimated at 12 minutes or less (also ``<=600s``)
      ``0=<=12m:3``        ... running 3 at a time
    Without a plan, every listed --gpus device accepts every model.
    """
    rules: Dict[str, DeviceRule] = {}
    if plan.strip():
        for part in plan.split(","):
            if "=" not in part:
                raise ValueError(f"invalid --gpu-plan entry {part!r}; use DEVICE=* | DEVICE=m1+m2 | DEVICE=<=12m[:slots]")
            device, spec = part.split("=", 1)
            device, spec = device.strip(), spec.strip()
            slots = default_slots
            if ":" in spec:
                spec, slot_text = spec.rsplit(":", 1)
                slots = int(slot_text)
            spec = spec.strip()
            max_est = None
            if spec.startswith("<="):
                value = spec[2:].strip().lower()
                unit = value[-1]
                max_est = float(value[:-1]) * (60 if unit == "m" else 3600 if unit == "h" else 1) if unit in "msh" else float(value)
                allowed_models = set(models)
            elif spec == "*":
                allowed_models = set(models)
            else:
                allowed_models = {n.strip() for n in spec.split("+") if n.strip()}
            rules[device] = DeviceRule(models=allowed_models, max_est_sec=max_est, slots=slots)
    else:
        for device in [g.strip() for g in gpus.split(",") if g.strip()]:
            rules[device] = DeviceRule(models=set(models), slots=default_slots)
    return rules


def describe_rule(rule: DeviceRule, all_models: Sequence[str]) -> str:
    if rule.max_est_sec is not None:
        what = f"jobs estimated <= {rule.max_est_sec / 60:.0f} min"
    elif set(rule.models) == set(all_models):
        what = "all models"
    else:
        what = ",".join(sorted(rule.models))
    return f"{what} ({rule.slots} concurrent)"


class DevicePool:
    """GPU devices with their own capacity; a job may only use devices whose rule accepts it."""

    def __init__(self, rules: Dict[str, DeviceRule]):
        self.rules = rules
        self.free = {device: rule.slots for device, rule in rules.items()}
        self.cv = threading.Condition()
        self.stopped = False

    def eligible(self, model: str, est_sec: float) -> List[str]:
        return [d for d, rule in self.rules.items() if rule.accepts(model, est_sec)]

    def acquire(self, model: str, est_sec: float) -> Optional[str]:
        devices = self.eligible(model, est_sec)
        if not devices:
            return None
        with self.cv:
            while not self.stopped:
                # Prefer the device with the most free capacity to spread load.
                candidates = [d for d in devices if self.free[d] > 0]
                if candidates:
                    device = max(candidates, key=lambda d: self.free[d])
                    self.free[device] -= 1
                    return device
                self.cv.wait(timeout=5)
            return None

    def release(self, device: str) -> None:
        with self.cv:
            self.free[device] += 1
            self.cv.notify_all()

    def stop(self) -> None:
        with self.cv:
            self.stopped = True
            self.cv.notify_all()


class BatchRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.output_base = Path(args.output_dir).expanduser().resolve() if args.output_dir else get_output_base()
        self.batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.batch_dir = self.output_base / "_batch" / self.batch_id
        self.log_dir = self.batch_dir / "logs"
        self.lock = threading.Lock()
        self.procs: Dict[int, subprocess.Popen] = {}
        self.stopping = False
        self.jobs: List[Job] = []
        self.pool: Optional[DevicePool] = None

    def build_jobs(self, datasets: List[str], models: List[str], root: Path) -> None:
        gpu_cache = {m: model_needs_gpu(m) for m in models}
        for d in datasets:
            train_rows, gen_rows = dataset_sizes(root, d)
            if self.args.max_train_rows:
                train_rows = min(train_rows, self.args.max_train_rows)
            if self.args.num_rows:
                gen_rows = self.args.num_rows
            for m in models:
                self.jobs.append(
                    Job(dataset=d, model=m, needs_gpu=gpu_cache[m], est_sec=round(estimate_seconds(m, train_rows, gen_rows), 1))
                )
        if self.args.order == "longest-first":
            self.jobs.sort(key=lambda j: j.est_sec, reverse=True)

    def _cmd(self, job: Job) -> List[str]:
        a = self.args
        cmd = [sys.executable, "-m", "core.runner.runner", "--model", job.model, "--dataset", job.dataset,
               "--train", "--generate", "--output-dir", str(self.output_base), "--no-stats"]
        if a.prepared_root:
            cmd += ["--prepared-root", str(Path(a.prepared_root).expanduser().resolve())]
        if a.preset:
            cmd += ["--preset", a.preset]
        for flag, value in (("--max-train-rows", a.max_train_rows), ("--num-rows", a.num_rows), ("--epochs", a.epochs)):
            if value is not None:
                cmd += [flag, str(value)]
        return cmd

    def write_status(self) -> None:
        with self.lock:
            rows = [asdict(j) for j in self.jobs]
        self.batch_dir.mkdir(parents=True, exist_ok=True)
        with open(self.batch_dir / "batch_status.csv", "w", newline="", encoding="utf-8") as f:
            cols = ["dataset", "model", "status", "device", "est_sec", "duration_sec", "quality_warnings", "attempts",
                    "exit_code", "reason", "synthetic_csv", "run_dir", "log", "started_at", "finished_at"]
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        counts: Dict[str, int] = {}
        for r in rows:
            counts[r["status"]] = counts.get(r["status"], 0) + 1
        summary = {
            "batch_id": self.batch_id,
            "output_base": str(self.output_base),
            "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(self.args).items()},
            "counts": counts,
            "jobs": rows,
        }
        (self.batch_dir / "batch_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def run_job(self, job: Job, device: Optional[str]) -> None:
        a = self.args
        if a.skip_done:
            done = _has_successful_run(self.output_base, job.dataset, job.model, a.preset)
            if done is not None:
                with self.lock:
                    job.status, job.run_dir, job.reason = "skipped_done", str(done), "successful run with same preset exists"
                self.write_status()
                return
        attempts = 1 + max(0, a.retries)
        for attempt in range(1, attempts + 1):
            self._run_attempt(job, device, attempt)
            if job.status != "fail" or attempt == attempts or self.stopping:
                return
            print(f"[batch] retry  {job.dataset} × {job.model} (attempt {attempt + 1}/{attempts}): {(job.reason or '')[:80]}", flush=True)
            time.sleep(a.retry_wait)

    def _run_attempt(self, job: Job, device: Optional[str], attempt: int) -> None:
        a = self.args
        env = os.environ.copy()
        src_dir = str(Path(__file__).resolve().parents[2])
        env["PYTHONPATH"] = src_dir + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        if device is not None:
            env[f"BENCHMARK_{job.model.upper()}_GPUS"] = f"device={device}"
        suffix = "" if attempt == 1 else f".attempt{attempt}"
        log_path = self.log_dir / f"{job.dataset}__{job.model}{suffix}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        start = time.time()
        with self.lock:
            job.status, job.device, job.log, job.attempts = "running", device, str(log_path), attempt
            job.started_at = datetime.now().isoformat(timespec="seconds")
        self.write_status()
        print(f"[batch] start  {job.dataset} × {job.model} (device={device or 'cpu'}, est {job.est_sec / 60:.1f} min)", flush=True)
        timeout = a.timeout_hours * 3600 if a.timeout_hours else None
        with open(log_path, "w", encoding="utf-8") as log:
            log.write("$ " + " ".join(self._cmd(job)) + "\n")
            log.flush()
            proc = subprocess.Popen(self._cmd(job), stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            with self.lock:
                self.procs[proc.pid] = proc
            try:
                code = proc.wait(timeout=timeout)
                timed_out = False
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    code = proc.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    code = proc.wait()
                timed_out = True
            finally:
                with self.lock:
                    self.procs.pop(proc.pid, None)

        run_dir = _latest_run_dir(self.output_base, job.dataset, job.model, start)
        result = {}
        if run_dir and (run_dir / "runtime_result.json").is_file():
            try:
                result = json.loads((run_dir / "runtime_result.json").read_text(encoding="utf-8"))
            except ValueError:
                result = {}
        with self.lock:
            job.exit_code = code
            job.duration_sec = round(time.time() - start, 1)
            job.finished_at = datetime.now().isoformat(timespec="seconds")
            job.run_dir = str(run_dir) if run_dir else None
            job.synthetic_csv = (result.get("artifacts") or {}).get("synthetic_csv")
            job.quality_warnings = len(result.get("quality_warnings") or [])
            if timed_out:
                job.status, job.reason = "timeout", f"exceeded {a.timeout_hours} h"
            elif self.stopping:
                job.status, job.reason = "cancelled", "batch interrupted"
            elif code == 0 and result.get("generate_status") == "success":
                job.status = "success"
            else:
                job.status = "fail"
                detail = result.get("reason_detail") or ""
                if not detail:
                    try:
                        tail = log_path.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-3:]
                        detail = " | ".join(tail)
                    except OSError:
                        detail = ""
                job.reason = (result.get("reason_code") or "exit_code_%s" % code) + (": " + detail[:500] if detail else "")
        self.write_status()
        print(f"[batch] {job.status:7s} {job.dataset} × {job.model} in {job.duration_sec}s", flush=True)

    def run(self) -> int:
        a = self.args
        models = sorted({j.model for j in self.jobs})
        gpu_models = [m for m in models if any(j.model == m and j.needs_gpu for j in self.jobs)]
        rules = parse_gpu_plan(a.gpu_plan, a.gpus, gpu_models, max(1, a.jobs_per_gpu))
        if not rules and any(j.needs_gpu for j in self.jobs):
            print("[batch] no --gpus/--gpu-plan given; GPU jobs will run one at a time with --gpus all", flush=True)
            rules = {"all": DeviceRule(models=set(gpu_models), slots=max(1, a.jobs_per_gpu))}
        unschedulable = [j for j in self.jobs if j.needs_gpu and not any(r.accepts(j.model, j.est_sec) for r in rules.values())]
        if unschedulable:
            raise SystemExit("[batch] no GPU accepts: " + ", ".join(f"{j.dataset}×{j.model}" for j in unschedulable[:10]))
        self.pool = DevicePool(rules)
        cpu_slots = threading.Semaphore(max(1, a.cpu_jobs))

        self.write_status()
        total_est = sum(j.est_sec for j in self.jobs)
        print(f"[batch] {len(self.jobs)} jobs -> {self.batch_dir}", flush=True)
        for device, rule in sorted(rules.items()):
            print(f"[batch] GPU {device}: {describe_rule(rule, gpu_models)}", flush=True)
        print(f"[batch] CPU: {a.cpu_jobs} concurrent | estimated total work {total_est / 3600:.1f} h "
              f"(ordering: {a.order})", flush=True)

        def handle_sigint(signum, frame):
            self.stopping = True
            if self.pool:
                self.pool.stop()
            print("\n[batch] interrupt: stopping running jobs...", flush=True)
            with self.lock:
                for proc in list(self.procs.values()):
                    try:
                        os.killpg(proc.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass

        signal.signal(signal.SIGINT, handle_sigint)
        signal.signal(signal.SIGTERM, handle_sigint)

        def worker(job: Job) -> None:
            if self.stopping:
                return
            if job.needs_gpu:
                device = self.pool.acquire(job.model, job.est_sec) if self.pool else None
                if device is None:
                    with self.lock:
                        job.status = "cancelled" if self.stopping else "unschedulable"
                        job.reason = None if self.stopping else f"no GPU in --gpu-plan accepts model {job.model}"
                    self.write_status()
                    return
                try:
                    if not self.stopping:
                        self.run_job(job, None if device == "all" else device)
                finally:
                    self.pool.release(device)
            else:
                with cpu_slots:
                    if not self.stopping:
                        self.run_job(job, None)

        threads = [threading.Thread(target=worker, args=(j,), daemon=True) for j in self.jobs]
        started = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for j in self.jobs:
            if j.status == "pending":
                j.status = "cancelled"
        self.write_status()

        width = max([len(j.dataset) for j in self.jobs] + [7])
        print("\n" + f"{'dataset':{width}}  {'model':16} {'status':12} {'sec':>8} {'est':>8} {'qwarn':>5}  reason")
        for j in sorted(self.jobs, key=lambda x: (x.dataset, x.model)):
            print(f"{j.dataset:{width}}  {j.model:16} {j.status:12} {str(j.duration_sec or ''):>8} "
                  f"{j.est_sec:>8.0f} {j.quality_warnings:>5}  {(j.reason or '')[:110]}")
        elapsed = timedelta(seconds=int(time.time() - started))
        ok = sum(1 for j in self.jobs if j.status in {"success", "skipped_done"})
        print(f"\n{ok}/{len(self.jobs)} ok in {elapsed}; status: {self.batch_dir / 'batch_status.csv'}")
        return 0 if ok == len(self.jobs) else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m core.runner.batch", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", default="all", help="comma-separated dataset ids, or 'all' prepared datasets")
    p.add_argument("--models", default="all", help=f"comma-separated models, or 'all' ({','.join(SUPPORTED_MODELS)})")
    p.add_argument("--prepared-root", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--preset", default=None, help="e.g. smoke or default (see model_presets.json)")
    p.add_argument("--gpus", default="", help="GPU device ids for GPU models, e.g. '0,1' (all models allowed on each)")
    p.add_argument("--gpu-plan", default="", help="per-device model whitelist, e.g. \"1=*,0=ctgan+tvae+tabpfgen\"")
    p.add_argument("--jobs-per-gpu", type=int, default=1)
    p.add_argument("--cpu-jobs", type=int, default=2, help="concurrent CPU-only model jobs (arf, bayesnet, forestdiffusion)")
    p.add_argument("--order", choices=["longest-first", "file"], default="longest-first", help="job order; longest-first shortens total wall clock")
    p.add_argument("--max-train-rows", type=int, default=None)
    p.add_argument("--num-rows", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--timeout-hours", type=float, default=None)
    p.add_argument("--retries", type=int, default=1, help="re-run a failed job (covers a preempted GPU or a transient CUDA OOM)")
    p.add_argument("--retry-wait", type=float, default=30.0, help="seconds to wait before a retry")
    p.add_argument("--skip-done", action="store_true", help="skip dataset×model pairs that already have a successful run with the same preset")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    if args.prepared_root:
        root_env = str(Path(args.prepared_root).expanduser().resolve())
        os.environ["BENCHMARK_NEW_DATASET_ROOT"] = root_env
        os.environ["BENCHMARK_PREPROCESSING_ROOT"] = root_env
    root = get_new_tabular_dataset_root()
    available = list_prepared_datasets(root)
    datasets = available if args.datasets == "all" else [d.strip() for d in args.datasets.split(",") if d.strip()]
    unknown = [d for d in datasets if d not in available]
    if unknown:
        p.error(f"datasets not prepared under {root}: {unknown}. Run python -m core.prepare run <dataset_dir> first.")
    models = list(SUPPORTED_MODELS) if args.models == "all" else [m.strip() for m in args.models.split(",") if m.strip()]
    bad_models = [m for m in models if m not in SUPPORTED_MODELS]
    if bad_models:
        p.error(f"unknown models: {bad_models}")
    if not datasets:
        p.error(f"no prepared datasets found under {root}")

    runner = BatchRunner(args)
    runner.build_jobs(datasets, models, root)
    if args.dry_run:
        gpu_models = [m for m in models if model_needs_gpu(m)]
        rules = parse_gpu_plan(args.gpu_plan, args.gpus, gpu_models, max(1, args.jobs_per_gpu))
        print(f"{'dataset':32} {'model':16} {'where':10} {'est min':>8}")
        for j in runner.jobs:
            where = "cpu" if not j.needs_gpu else ",".join(sorted(d for d, r in rules.items() if r.accepts(j.model, j.est_sec))) or "NONE"
            print(f"{j.dataset:32} {j.model:16} {where:10} {j.est_sec / 60:>8.1f}")
        print(f"\ntotal estimated work: {sum(j.est_sec for j in runner.jobs) / 3600:.1f} h")
        return 0
    return runner.run()


if __name__ == "__main__":
    raise SystemExit(main())

import json
import re
from datetime import datetime
from pathlib import Path


ROOT = Path.cwd()
REMOTE_ROOT = ROOT / "remote-output-Benchmark-trainonly-v1"

START_RE = re.compile(r"^started_at_utc:\s*(.+?)\s*$", re.MULTILINE)
END_RE = re.compile(r"^finished_at_utc:\s*(.+?)\s*$", re.MULTILINE)
ELAPSED_RE = re.compile(r"^elapsed_seconds:\s*([0-9.]+)\s*$", re.MULTILINE)


def read_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload):
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def normalize_timestamp(raw: str):
    raw = raw.strip()
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return raw
    return dt.replace(microsecond=0, tzinfo=None).isoformat()


def parse_timing_log(path: Path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    start = START_RE.search(text)
    end = END_RE.search(text)
    elapsed = ELAPSED_RE.search(text)
    if not any([start, end, elapsed]):
        return None
    return {
        "started_at": normalize_timestamp(start.group(1)) if start else None,
        "ended_at": normalize_timestamp(end.group(1)) if end else None,
        "duration_sec": float(elapsed.group(1)) if elapsed else None,
    }


def choose_log(run_dir: Path, pattern: str):
    logs = sorted(run_dir.glob(pattern))
    if not logs:
        return None
    # When there are multiple attempts, prefer the latest lexicographically.
    return logs[-1]


def merge_phase(existing_phase, parsed_phase):
    phase = {}
    if isinstance(existing_phase, dict):
        phase.update(existing_phase)
    if not parsed_phase:
        return phase if phase else None

    changed = False
    for key in ("started_at", "ended_at", "duration_sec"):
        if phase.get(key) is None and parsed_phase.get(key) is not None:
            phase[key] = parsed_phase[key]
            changed = True
    return phase if phase else None


def main():
    updated = 0
    unchanged = 0
    missing_both_logs = 0
    partial_only = 0
    invalid_runtime_json = 0
    report = []

    for runtime_path in sorted(REMOTE_ROOT.rglob("runtime_result.json")):
        try:
            payload = read_json(runtime_path)
        except Exception:
            invalid_runtime_json += 1
            report.append(
                {
                    "runtime_result": str(runtime_path),
                    "status": "invalid_runtime_json",
                }
            )
            continue
        run_dir = runtime_path.parent

        train_log = choose_log(run_dir, "train_*.log")
        gen_log = choose_log(run_dir, "gen_*.log")

        parsed_train = parse_timing_log(train_log) if train_log else None
        parsed_gen = parse_timing_log(gen_log) if gen_log else None

        if parsed_train is None and parsed_gen is None:
            missing_both_logs += 1
            report.append(
                {
                    "runtime_result": str(runtime_path),
                    "status": "missing_both_logs",
                }
            )
            continue

        timings = payload.get("timings", {})
        if not isinstance(timings, dict):
            timings = {}

        before = json.dumps(timings, sort_keys=True, ensure_ascii=False)
        train_phase = merge_phase(timings.get("train"), parsed_train)
        gen_phase = merge_phase(timings.get("generate"), parsed_gen)

        new_timings = dict(timings)
        if train_phase is not None:
            new_timings["train"] = train_phase
        if gen_phase is not None:
            new_timings["generate"] = gen_phase

        after = json.dumps(new_timings, sort_keys=True, ensure_ascii=False)

        if before != after:
            payload["timings"] = new_timings
            write_json(runtime_path, payload)
            updated += 1
            status = "updated"
        else:
            unchanged += 1
            status = "unchanged"

        if parsed_train is None or parsed_gen is None:
            partial_only += 1

        report.append(
            {
                "runtime_result": str(runtime_path),
                "status": status,
                "train_log": str(train_log) if train_log else None,
                "gen_log": str(gen_log) if gen_log else None,
                "train_timing_found": parsed_train is not None,
                "generate_timing_found": parsed_gen is not None,
            }
        )

    summary = {
        "updated": updated,
        "unchanged": unchanged,
        "missing_both_logs": missing_both_logs,
        "partial_only": partial_only,
        "invalid_runtime_json": invalid_runtime_json,
        "report_count": len(report),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

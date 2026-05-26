from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[3]
USAGE_LOG_PATH = REPO_ROOT / "logs" / "usage_log.csv"
V2_SNAPSHOT_PATH = (
    REPO_ROOT
    / "Evaluation"
    / "subitem_workload_v2"
    / "final"
    / "token_usage_snapshot"
    / "dataset_token_usage_snapshot.csv"
)
OUT_DIR = REPO_ROOT / "Evaluation" / "token_usage_v1" / "final"
PAPER_FIG_DIR = (
    REPO_ROOT
    / "Paper"
    / "69b27219c555c38a69bb2156"
    / "figures"
    / "time_cost"
)


@dataclass
class DatasetUsage:
    dataset_id: str
    calls: int = 0
    generated_sql: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    first_timestamp: str = ""
    last_timestamp: str = ""

    def observe(self, row: dict[str, str]) -> None:
        self.calls += 1
        if (row.get("phase") or "").strip() == "queryspec_generation":
            self.generated_sql += 1
        self.input_tokens += int(float(row.get("input_tokens", "0") or 0))
        self.output_tokens += int(float(row.get("output_tokens", "0") or 0))
        self.total_tokens += int(float(row.get("total_tokens", "0") or 0))
        self.cost_usd += float(row.get("cost_usd", "0") or 0.0)
        timestamp = row.get("timestamp", "") or ""
        if not self.first_timestamp or timestamp < self.first_timestamp:
            self.first_timestamp = timestamp
        if not self.last_timestamp or timestamp > self.last_timestamp:
            self.last_timestamp = timestamp


def natural_dataset_key(dataset_id: str) -> tuple[str, int]:
    prefix = "".join(ch for ch in dataset_id if ch.isalpha())
    suffix = "".join(ch for ch in dataset_id if ch.isdigit())
    return prefix, int(suffix or "0")


def load_current_paper_dataset_ids(path: Path) -> list[str]:
    dataset_ids: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset_id = (row.get("dataset_id") or "").strip()
            if dataset_id and dataset_id != "TOTAL":
                dataset_ids.append(dataset_id)
    return sorted(set(dataset_ids), key=natural_dataset_key)


def aggregate_usage(rows: Iterable[dict[str, str]]) -> dict[str, DatasetUsage]:
    out: dict[str, DatasetUsage] = {}
    for row in rows:
        dataset_id = (row.get("dataset_id") or "").strip()
        if not dataset_id:
            continue
        bucket = out.setdefault(dataset_id, DatasetUsage(dataset_id=dataset_id))
        bucket.observe(row)
    return out


def write_csv(path: Path, rows: list[DatasetUsage]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "dataset_id",
                "calls",
                "generated_sql",
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "cost_usd",
                "first_timestamp",
                "last_timestamp",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.dataset_id,
                    row.calls,
                    row.generated_sql,
                    row.input_tokens,
                    row.output_tokens,
                    row.total_tokens,
                    f"{row.cost_usd:.6f}",
                    row.first_timestamp,
                    row.last_timestamp,
                ]
            )


def fmt_int(value: int) -> str:
    return f"{value:,}"


def write_paper_longtable(path: Path, rows: list[DatasetUsage]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        r"\setlength{\LTleft}{0pt}",
        r"\setlength{\LTright}{0pt}",
        r"{\scriptsize",
        r"\begin{longtable}{@{}L{0.65in}C{0.52in}C{0.72in}C{1.00in}C{1.00in}C{1.00in}C{0.78in}@{}}",
        r"\caption{Legacy first-version agent token usage aggregated from \texttt{logs/usage\_log.csv}. `Generated SQL` counts logged \texttt{queryspec\_generation} events per dataset. The paper-facing table is restricted to the current 49 benchmark datasets; removed datasets \texttt{c21} and \texttt{n13} remain available in the full CSV artifact.\label{tab:appendix_legacy_token_usage_v1}}\\",
        r"\toprule",
        r"Dataset & Calls & Generated SQL & Input tokens & Output tokens & Total tokens & Cost (USD) \\",
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{7}{c}{\tablename\ \thetable\ (continued)}\\",
        r"\toprule",
        r"Dataset & Calls & Generated SQL & Input tokens & Output tokens & Total tokens & Cost (USD) \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endfoot",
    ]
    total_calls = 0
    total_generated_sql = 0
    total_input = 0
    total_output = 0
    total_tokens = 0
    total_cost = 0.0
    for row in rows:
        total_calls += row.calls
        total_generated_sql += row.generated_sql
        total_input += row.input_tokens
        total_output += row.output_tokens
        total_tokens += row.total_tokens
        total_cost += row.cost_usd
        lines.append(
            f"{row.dataset_id} & {row.calls:,} & {row.generated_sql:,} & {fmt_int(row.input_tokens)} & "
            f"{fmt_int(row.output_tokens)} & {fmt_int(row.total_tokens)} & {row.cost_usd:.6f} \\\\"
        )
    lines.extend(
        [
            r"\midrule",
            f"TOTAL & {total_calls:,} & {total_generated_sql:,} & {fmt_int(total_input)} & {fmt_int(total_output)} & {fmt_int(total_tokens)} & {total_cost:.6f} \\\\",
            r"\end{longtable}",
            r"}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_summary(path: Path, all_rows: list[DatasetUsage], paper_rows: list[DatasetUsage]) -> None:
    lines = [
        "# Legacy V1 Token Usage Summary",
        "",
        f"- Source log: `{USAGE_LOG_PATH.as_posix()}`",
        f"- Full datasets found in log: `{len(all_rows)}`",
        f"- Paper-facing datasets retained: `{len(paper_rows)}`",
        "- Paper-facing dataset list is aligned to the current 49-dataset benchmark roster from the V2 snapshot.",
        "- Removed datasets `c21` and `n13` are excluded from the appendix table but kept in the full CSV artifact.",
        "",
    ]
    top_rows = sorted(paper_rows, key=lambda row: row.total_tokens, reverse=True)[:10]
    lines.append("## Top 10 paper-facing datasets by total tokens")
    lines.append("")
    lines.append("| dataset | calls | generated sql | total tokens | cost usd |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in top_rows:
        lines.append(
            f"| {row.dataset_id} | {row.calls:,} | {row.generated_sql:,} | {row.total_tokens:,} | {row.cost_usd:.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    with USAGE_LOG_PATH.open("r", encoding="utf-8", newline="") as f:
        usage_rows = list(csv.DictReader(f))

    dataset_ids = load_current_paper_dataset_ids(V2_SNAPSHOT_PATH)
    aggregated = aggregate_usage(usage_rows)

    all_rows = sorted(aggregated.values(), key=lambda row: natural_dataset_key(row.dataset_id))
    paper_rows = [aggregated[dataset_id] for dataset_id in dataset_ids if dataset_id in aggregated]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)

    write_csv(OUT_DIR / "dataset_token_usage_v1_full.csv", all_rows)
    write_csv(OUT_DIR / "dataset_token_usage_v1_paper49.csv", paper_rows)
    write_summary(OUT_DIR / "dataset_token_usage_v1_summary.md", all_rows, paper_rows)

    tex_path = OUT_DIR / "dataset_token_usage_v1_generated.tex"
    write_paper_longtable(tex_path, paper_rows)
    write_paper_longtable(PAPER_FIG_DIR / "legacy_token_usage_v1_generated.tex", paper_rows)


if __name__ == "__main__":
    main()

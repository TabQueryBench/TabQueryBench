#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import tomllib  # py311+
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None  # type: ignore[assignment]


REPO_ROOT = Path(__file__).resolve().parents[1]
FINAL_ROOT = Path(r"F:\TabQueryBench\Data_HF\03_synthetic_data\final_csv")
PROVENANCE_PATH = Path(r"F:\TabQueryBench\Data_HF\_LOCAL_ONLY_NOT_FOR_UPLOAD\final_csv_provenance_20260509.json")
REPORT_JSON = REPO_ROOT / "tmp" / "final_hyperparam_backfill_20260509.json"
REPORT_MD = REPO_ROOT / "tmp" / "final_hyperparam_backfill_20260509.md"
REPORT_CSV = REPO_ROOT / "tmp" / "final_hyperparam_backfill_20260509.csv"


SPECIAL_CONTAINER_DIRS = {"metadata", "meta", "logs", "synthetic", "synthetic_data"}
RUN_ID_RE = re.compile(
    r"(?:arf|bayesnet|ctgan|forest|rtf|realtabformer|tabbyflow|tabddpm|tabdiff|tabpfgen|tabsyn|tvae)-[A-Za-z0-9]+-\d{8}_\d{6}"
)


@dataclass
class RunContext:
    dataset: str
    model: str
    source: str
    source_ref: str
    final_json: Path
    model_root: Path
    run_root: Optional[Path]
    run_id: Optional[str]
    resolved_source_csv: Optional[Path]
    resolved_source_metadata: Optional[Path]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _extract_run_id(text: str) -> Optional[str]:
    name = Path(text).name
    if "__" in name:
        parts = name.split("__")
        if len(parts) >= 4 and parts[2]:
            return parts[2]
    m = RUN_ID_RE.search(text.replace("\\", "/"))
    return m.group(0) if m else None


def _dataset_train_shape(dataset: str) -> Tuple[int, int]:
    train_csv = REPO_ROOT / "data" / dataset / f"{dataset}-train.csv"
    if not train_csv.exists():
        raise FileNotFoundError(f"Missing train CSV for dataset {dataset}: {train_csv}")
    rows = 0
    cols = 0
    with train_csv.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        cols = len(header)
        for _ in reader:
            rows += 1
    return rows, cols


def _relative_repo_path(path_str: Optional[str]) -> Optional[Path]:
    if not path_str:
        return None
    p = Path(path_str)
    return p if p.is_absolute() else REPO_ROOT / p


def _build_context(item: Dict[str, Any]) -> RunContext:
    source_entry = item["source_entry"]
    source = source_entry["source"]
    source_ref = source_entry["source_ref"]
    ref_path = Path(source_ref)
    final_json = Path(source_entry["imported_metadata"])
    resolved_meta = _relative_repo_path(item.get("resolved_source_metadata") or source_ref)
    resolved_csv = _relative_repo_path(item.get("resolved_source_csv"))
    run_id = _extract_run_id(source_ref) or _extract_run_id(str(resolved_meta or "")) or _extract_run_id(str(resolved_csv or ""))

    if "metadata" in ref_path.parts:
        idx = ref_path.parts.index("metadata")
        model_root = REPO_ROOT.joinpath(*ref_path.parts[:idx])
    elif "meta" in ref_path.parts:
        idx = ref_path.parts.index("meta")
        model_root = REPO_ROOT.joinpath(*ref_path.parts[:idx])
    elif run_id and run_id in ref_path.parts:
        idx = ref_path.parts.index(run_id)
        model_root = REPO_ROOT.joinpath(*ref_path.parts[:idx])
    else:
        model_root = (resolved_meta or resolved_csv or REPO_ROOT).parent

    run_root: Optional[Path] = None
    if run_id:
        if resolved_meta and run_id in resolved_meta.parts:
            idx = resolved_meta.parts.index(run_id)
            run_root = Path(*resolved_meta.parts[: idx + 1])
        elif resolved_csv and run_id in resolved_csv.parts:
            idx = resolved_csv.parts.index(run_id)
            run_root = Path(*resolved_csv.parts[: idx + 1])
        else:
            candidate = model_root / run_id
            if candidate.exists():
                run_root = candidate

    return RunContext(
        dataset=item["dataset"],
        model=item["model"],
        source=source,
        source_ref=source_ref,
        final_json=final_json,
        model_root=model_root,
        run_root=run_root,
        run_id=run_id,
        resolved_source_csv=resolved_csv,
        resolved_source_metadata=resolved_meta,
    )


def _iter_candidate_files(ctx: RunContext, filename_suffix: str) -> Iterable[Path]:
    roots: List[Path] = []
    if ctx.run_root and ctx.run_root.exists():
        roots.append(ctx.run_root)
    if ctx.model_root.exists() and ctx.model_root not in roots:
        roots.append(ctx.model_root)
    seen: set[Path] = set()
    for root in roots:
        for path in root.rglob(f"*{filename_suffix}"):
            if ctx.run_id and ctx.run_id not in str(path):
                continue
            if path not in seen:
                seen.add(path)
                yield path


def _iter_train_logs(ctx: RunContext) -> Iterable[Path]:
    roots: List[Path] = []
    if ctx.run_root and ctx.run_root.exists():
        roots.append(ctx.run_root)
    if ctx.model_root.exists() and ctx.model_root not in roots:
        roots.append(ctx.model_root)
    seen: set[Path] = set()
    for root in roots:
        for path in root.rglob("*train*.log"):
            if ctx.run_id and ctx.run_id not in str(path):
                continue
            if path not in seen:
                seen.add(path)
                yield path


def _first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def _extract_json_object(text: str, marker: str) -> Optional[Dict[str, Any]]:
    start = text.find(marker)
    if start < 0:
        return None
    brace_start = text.find("{", start)
    if brace_start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for idx in range(brace_start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                snippet = text[brace_start : idx + 1]
                try:
                    return json.loads(snippet)
                except json.JSONDecodeError:
                    return None
    return None


def _parse_bool(text: str) -> bool:
    return str(text).strip().lower() in {"1", "true", "yes", "y"}


def _parse_int(text: str) -> Optional[int]:
    try:
        return int(str(text).strip().strip('"').strip("'"))
    except Exception:
        return None


def _parse_float(text: str) -> Optional[float]:
    try:
        return float(str(text).strip().strip('"').strip("'"))
    except Exception:
        return None


def _current_defaults(model: str, dataset: str) -> Dict[str, Any]:
    if model == "arf":
        return {
            "num_trees": 30,
            "delta": 0.0,
            "max_iters": 10,
            "early_stop": True,
            "min_node_size": 5,
            "clip_quantile_low": 0.001,
            "clip_quantile_high": 0.999,
        }
    if model == "bayesnet":
        rows, cols = _dataset_train_shape(dataset)
        max_bins = 10
        max_cat_levels = 256
        if cols > 35 or rows > 200000:
            max_bins = 5
            max_cat_levels = 64
        if cols > 55:
            max_bins = 4
            max_cat_levels = 32
        return {
            "max_bins": max_bins,
            "max_cat_levels": max_cat_levels,
            "struct_rows": 25000,
            "fit_rows": 120000,
            "estimator_type": "chow-liu",
            "edge_weights_fn": "mutual_info",
            "root_node": None,
            "n_jobs": 1,
        }
    if model == "ctgan":
        return {
            "epochs": 50,
            "embedding_dim": 4,
            "generator_dim": [8, 8],
            "discriminator_dim": [8, 8],
            "batch_size": 2,
            "pac": 1,
            "max_train_rows": 50000,
        }
    if model == "tvae":
        return {
            "epochs": 300,
            "batch_size": 500,
            "embedding_dim": 128,
            "compress_dims": [128, 128],
            "decompress_dims": [128, 128],
            "loky_max_cpu_count": 8,
        }
    if model == "realtabformer":
        return {
            "epochs": 100,
            "CUDA_VISIBLE_DEVICES": "0",
            "NCCL_P2P_DISABLE": "1",
        }
    if model == "tabddpm":
        return {
            "epochs": 50,
            "num_timesteps": 100,
            "steps": 1000,
            "lr": 0.0005,
            "train_batch_size": 64,
            "sample_batch_size": 64,
            "d_layers": [256, 256],
            "dropout": 0.0,
            "seed": 0,
        }
    if model == "tabdiff":
        return {
            "exp_name": "adapter_learnable",
            "steps": 500,
        }
    if model == "tabbyflow":
        return {
            "exp_name": "adapter_efvfm",
            "steps": 500,
            "EFVFM_SAMPLE_BATCH_SIZE": 128,
            "EFVFM_EVAL_NUM_SAMPLES": 512,
        }
    if model == "forestdiffusion":
        return {
            "n_estimators": 20,
            "n_t": 10,
            "duplicate_K": 5,
            "n_jobs": 1,
            "max_depth": 4,
            "model": "xgboost",
        }
    if model == "tabsyn":
        return {
            "TABSYN_RESUME": 0,
            "TABSYN_VAE_BATCH_SIZE": 32,
            "TABSYN_VAE_NUM_WORKERS": 0,
            "TABSYN_DIFFUSION_NUM_WORKERS": 0,
        }
    if model == "tabpfgen":
        return {
            "training_mode": "pretrained_no_train",
            "fit_rows_cap": 50000,
            "chunk_rows": 256,
            "device": "auto",
            "n_sgld_steps": 1000,
            "sgld_step_size": 0.01,
            "sgld_noise_scale": 0.01,
        }
    raise KeyError(f"Unsupported model {model}")


def _historical_baseline(model: str, dataset: str) -> Dict[str, Any]:
    if model == "forestdiffusion":
        return {
            "n_estimators": 100,
            "n_t": 20,
            "duplicate_K": 20,
            "n_jobs": 2,
            "max_depth": 6,
            "model": "xgboost",
        }
    if model == "tabddpm":
        return {
            "num_timesteps": 1000,
            "steps": 5000,
            "lr": 0.001,
            "train_batch_size": 256,
            "sample_batch_size": 1000,
        }
    defaults = _current_defaults(model, dataset)
    return defaults


def _parse_normalized_record(ctx: RunContext) -> Optional[Tuple[Dict[str, Any], str]]:
    record = _first_existing(_iter_candidate_files(ctx, "normalized_record.json"))
    if not record:
        return None
    data = _load_json(record)
    params = dict(data.get("train_hyperparams") or {})
    if not params:
        return None
    return params, "exact_artifact"


def _parse_tabddpm(ctx: RunContext) -> Optional[Tuple[Dict[str, Any], str]]:
    config = _first_existing(_iter_candidate_files(ctx, "config.toml"))
    if not config:
        return None
    if tomllib is None:
        return None
    data = tomllib.loads(config.read_text(encoding="utf-8"))
    train = data.get("train", {})
    main = train.get("main", {})
    sample = data.get("sample", {})
    out = {
        "num_timesteps": data.get("diffusion_params", {}).get("num_timesteps") or data.get("num_timesteps"),
        "steps": main.get("steps") or data.get("steps"),
        "lr": main.get("lr") or data.get("lr"),
        "train_batch_size": main.get("batch_size") or data.get("batch_size"),
        "sample_batch_size": sample.get("batch_size"),
        "sample_num_samples": sample.get("num_samples"),
    }
    out = {k: v for k, v in out.items() if v is not None}
    return out, "exact_artifact"


def _parse_json_config_log(ctx: RunContext) -> Optional[Dict[str, Any]]:
    for log_path in _iter_train_logs(ctx):
        text = log_path.read_text(encoding="utf-8", errors="ignore")
        cfg = _extract_json_object(text, "The config of the current run is")
        if cfg:
            return cfg
    return None


def _parse_tabdiff(ctx: RunContext) -> Optional[Tuple[Dict[str, Any], str]]:
    base = _parse_normalized_record(ctx)
    cfg = _parse_json_config_log(ctx)
    out = dict(base[0]) if base else {}
    if cfg:
        train_main = cfg.get("train", {}).get("main", {})
        diff = cfg.get("diffusion_params", {})
        out.update(
            {
                "num_timesteps": diff.get("num_timesteps"),
                "steps": train_main.get("steps", out.get("steps")),
                "lr": train_main.get("lr"),
                "weight_decay": train_main.get("weight_decay"),
                "ema_decay": train_main.get("ema_decay"),
                "batch_size": train_main.get("batch_size"),
                "exp_name": out.get("exp_name", "adapter_learnable"),
            }
        )
        out = {k: v for k, v in out.items() if v is not None}
        return out, "exact_artifact"
    if base:
        defaults = _current_defaults("tabdiff", ctx.dataset)
        merged = {**defaults, **base[0]}
        return merged, "artifact_plus_default"
    return None


def _parse_tabbyflow(ctx: RunContext) -> Optional[Tuple[Dict[str, Any], str]]:
    base = _parse_normalized_record(ctx)
    cfg = _parse_json_config_log(ctx)
    out = dict(base[0]) if base else {}
    if cfg:
        train_main = cfg.get("train", {}).get("main", {})
        net = cfg.get("unimodmlp_params", {})
        out.update(
            {
                "steps": train_main.get("steps", out.get("steps")),
                "lr": train_main.get("lr"),
                "weight_decay": train_main.get("weight_decay"),
                "ema_decay": train_main.get("ema_decay"),
                "batch_size": train_main.get("batch_size"),
                "warmup_epochs": train_main.get("warmup_epochs"),
                "num_layers": net.get("num_layers"),
                "d_token": net.get("d_token"),
                "n_head": net.get("n_head"),
                "factor": net.get("factor"),
                "dim_t": net.get("dim_t"),
                "activation": net.get("activation"),
                "EFVFM_SAMPLE_BATCH_SIZE": out.get("EFVFM_SAMPLE_BATCH_SIZE", 128),
                "EFVFM_EVAL_NUM_SAMPLES": out.get("EFVFM_EVAL_NUM_SAMPLES", 512),
                "exp_name": out.get("exp_name", "adapter_efvfm"),
            }
        )
        out = {k: v for k, v in out.items() if v is not None}
        return out, "exact_artifact"
    if base:
        defaults = _current_defaults("tabbyflow", ctx.dataset)
        merged = {**defaults, **base[0]}
        return merged, "artifact_plus_default"
    return None


def _parse_forestdiffusion(ctx: RunContext) -> Optional[Tuple[Dict[str, Any], str]]:
    base = _parse_normalized_record(ctx)
    if base:
        merged = {**_historical_baseline("forestdiffusion", ctx.dataset), **base[0]}
        return merged, "exact_artifact"
    script = _first_existing(_iter_candidate_files(ctx, "_fd_train.py"))
    if script:
        text = script.read_text(encoding="utf-8", errors="ignore")
        out: Dict[str, Any] = {}
        for key in ["n_t", "n_estimators", "duplicate_K", "n_jobs", "max_depth"]:
            m = re.search(rf"{re.escape(key)}\s*=\s*(\d+)", text)
            if m:
                out[key] = int(m.group(1))
        out["model"] = "xgboost"
        if out:
            return out, "exact_artifact"
    for log_path in _iter_train_logs(ctx):
        text = log_path.read_text(encoding="utf-8", errors="ignore")
        m = re.search(
            r"\[ForestDiffusion\] train config: rows=\d+ cols=\d+ n_t=(\d+) n_estimators=(\d+) duplicate_K=(\d+) n_jobs=(\d+)",
            text,
        )
        if m:
            out = {
                "n_t": int(m.group(1)),
                "n_estimators": int(m.group(2)),
                "duplicate_K": int(m.group(3)),
                "n_jobs": int(m.group(4)),
                "model": "xgboost",
            }
            defaults = _historical_baseline("forestdiffusion", ctx.dataset) if ctx.source == "5" else _current_defaults("forestdiffusion", ctx.dataset)
            out = {**defaults, **out}
            return out, "artifact_plus_default"
    if ctx.source == "5":
        return _historical_baseline("forestdiffusion", ctx.dataset), "inferred_historical_baseline"
    return _current_defaults("forestdiffusion", ctx.dataset), "inferred_current_default"


def _parse_realtabformer(ctx: RunContext) -> Optional[Tuple[Dict[str, Any], str]]:
    # exact when nested models_*epochs exists
    roots = [ctx.run_root] if ctx.run_root else []
    roots.append(ctx.model_root)
    for root in roots:
        if not root or not root.exists():
            continue
        for path in root.rglob("models_*epochs"):
            if ctx.run_id and ctx.run_id not in str(path):
                continue
            m = re.search(r"models_(\d+)epochs", path.name)
            if m:
                return {
                    "epochs": int(m.group(1)),
                    "CUDA_VISIBLE_DEVICES": "0",
                    "NCCL_P2P_DISABLE": "1",
                }, "exact_artifact"
    return _current_defaults("realtabformer", ctx.dataset), "inferred_current_default"


def _parse_ctgan(ctx: RunContext) -> Optional[Tuple[Dict[str, Any], str]]:
    script = _first_existing(_iter_candidate_files(ctx, "_ctgan_train.py"))
    defaults = _current_defaults("ctgan", ctx.dataset)
    if script:
        text = script.read_text(encoding="utf-8", errors="ignore")
        out = dict(defaults)
        for key in ["embedding_dim", "batch_size", "pac", "epochs"]:
            m = re.search(rf"{re.escape(key)}\s*=\s*([0-9]+)", text)
            if m:
                out[key] = int(m.group(1))
        dims_patterns = {
            "generator_dim": r"generator_dim\s*=\s*\(([^)]+)\)",
            "discriminator_dim": r"discriminator_dim\s*=\s*\(([^)]+)\)",
        }
        for key, pat in dims_patterns.items():
            m = re.search(pat, text)
            if m:
                out[key] = [int(x.strip()) for x in m.group(1).split(",") if x.strip()]
        return out, "exact_artifact"
    for log_path in _iter_train_logs(ctx):
        m = re.search(r"models_(\d+)epochs", log_path.name)
        if m:
            out = dict(defaults)
            out["epochs"] = int(m.group(1))
            return out, "artifact_plus_default"
    return defaults, "inferred_current_default"


def _parse_tvae(ctx: RunContext) -> Optional[Tuple[Dict[str, Any], str]]:
    defaults = _current_defaults("tvae", ctx.dataset)
    script = _first_existing(_iter_candidate_files(ctx, "_tvae_train.py"))
    if script:
        text = script.read_text(encoding="utf-8", errors="ignore")
        out = dict(defaults)
        m = re.search(r"TVAE\(epochs=(\d+),\s*batch_size=(\d+)", text)
        if m:
            out["epochs"] = int(m.group(1))
            out["batch_size"] = int(m.group(2))
        return out, "exact_artifact"
    for root in [ctx.run_root, ctx.model_root]:
        if not root or not root.exists():
            continue
        for path in root.rglob("models_*epochs*"):
            if ctx.run_id and ctx.run_id not in str(path):
                continue
            m = re.search(r"models_(\d+)epochs", path.name)
            if m:
                out = dict(defaults)
                out["epochs"] = int(m.group(1))
                return out, "artifact_plus_default"
    for log_path in _iter_train_logs(ctx):
        m = re.search(r"models_(\d+)epochs", log_path.name)
        if m:
            out = dict(defaults)
            out["epochs"] = int(m.group(1))
            return out, "artifact_plus_default"
    return defaults, "inferred_current_default"


def _parse_tabsyn(ctx: RunContext) -> Optional[Tuple[Dict[str, Any], str]]:
    defaults = _current_defaults("tabsyn", ctx.dataset)
    script = _first_existing(_iter_candidate_files(ctx, "_tabsyn_train.py"))
    if script:
        text = script.read_text(encoding="utf-8", errors="ignore")
        out = dict(defaults)
        m = re.search(r"_te\s*=\s*(\d+)", text)
        if m:
            te = int(m.group(1))
            out["vae_epochs"] = te
            out["diffusion_max_epochs"] = max(te + 1, 2)
        return out, "exact_artifact"
    for log_path in _iter_train_logs(ctx):
        text = log_path.read_text(encoding="utf-8", errors="ignore")
        m = re.search(r"Epoch\s+1/(\d+)", text)
        if m:
            out = dict(defaults)
            te = int(m.group(1))
            out["vae_epochs"] = te
            if te == 4000:
                out["diffusion_max_epochs"] = 10001
            return out, "artifact_plus_default"
    return defaults, "inferred_current_default"


def _parse_tabpfgen(ctx: RunContext) -> Optional[Tuple[Dict[str, Any], str]]:
    defaults = _current_defaults("tabpfgen", ctx.dataset)
    meta_path = _first_existing(_iter_candidate_files(ctx, "tabpfgen_meta.json"))
    if meta_path:
        data = _load_json(meta_path)
        out = dict(defaults)
        out.update(
            {
                "task_type": data.get("task_type"),
                "is_classification": data.get("is_classification"),
                "n_rows": data.get("n_rows"),
                "n_cols": data.get("n_cols"),
            }
        )
        out = {k: v for k, v in out.items() if v is not None}
        return out, "artifact_plus_default"
    return defaults, "inferred_current_default"


def _parse_arf(ctx: RunContext) -> Optional[Tuple[Dict[str, Any], str]]:
    defaults = _current_defaults("arf", ctx.dataset)
    script = _first_existing(_iter_candidate_files(ctx, "_arf_train.py"))
    if script:
        text = script.read_text(encoding="utf-8", errors="ignore")
        out = dict(defaults)
        patterns = {
            "num_trees": r'ARF_NUM_TREES",\s*"(\d+)"',
            "delta": r'ARF_DELTA",\s*"([^"]+)"',
            "max_iters": r'ARF_MAX_ITERS",\s*"(\d+)"',
            "min_node_size": r'ARF_MIN_NODE_SIZE",\s*"(\d+)"',
            "clip_quantile_low": r'ARF_CLIP_QUANTILE_LOW",\s*"([^"]+)"',
            "clip_quantile_high": r'ARF_CLIP_QUANTILE_HIGH",\s*"([^"]+)"',
        }
        for key, pat in patterns.items():
            m = re.search(pat, text)
            if m:
                if key in {"delta", "clip_quantile_low", "clip_quantile_high"}:
                    out[key] = float(m.group(1))
                else:
                    out[key] = int(m.group(1))
        return out, "exact_artifact"
    for log_path in _iter_train_logs(ctx):
        text = log_path.read_text(encoding="utf-8", errors="ignore")
        m = re.search(
            r"\[ARF\] Config num_trees=([0-9]+)\s+delta=([0-9.]+)\s+max_iters=([0-9]+)\s+early_stop=(True|False|true|false)\s+min_node_size=([0-9]+)",
            text,
        )
        if m:
            out = dict(defaults)
            out.update(
                {
                    "num_trees": int(m.group(1)),
                    "delta": float(m.group(2)),
                    "max_iters": int(m.group(3)),
                    "early_stop": _parse_bool(m.group(4)),
                    "min_node_size": int(m.group(5)),
                }
            )
            return out, "exact_artifact"
    return defaults, "inferred_current_default"


def _parse_bayesnet(ctx: RunContext) -> Optional[Tuple[Dict[str, Any], str]]:
    defaults = _current_defaults("bayesnet", ctx.dataset)
    for log_path in _iter_train_logs(ctx):
        text = log_path.read_text(encoding="utf-8", errors="ignore")
        m = re.search(
            r"\[BayesNet\] max_bins=([0-9]+), max_cat_levels=([0-9]+), struct_rows=([0-9]+), fit_rows=([0-9]+), estimator_type=([^,]+), edge_weights_fn=([^,]+), root_node=([^,]+), n_jobs=([0-9]+)",
            text,
        )
        if m:
            out = dict(defaults)
            out.update(
                {
                    "max_bins": int(m.group(1)),
                    "max_cat_levels": int(m.group(2)),
                    "struct_rows": int(m.group(3)),
                    "fit_rows": int(m.group(4)),
                    "estimator_type": m.group(5).strip(),
                    "edge_weights_fn": m.group(6).strip(),
                    "root_node": None if m.group(7).strip() in {"None", "null", ""} else m.group(7).strip(),
                    "n_jobs": int(m.group(8)),
                }
            )
            return out, "exact_artifact"
        m2 = re.search(r"\[BayesNet\] max_bins=([0-9]+)\s+\(cols_in_df=([0-9]+), rows=([0-9]+)\)", text)
        if m2:
            rows = int(m2.group(3))
            cols = int(m2.group(2))
            max_bins = int(m2.group(1))
            max_cat_levels = defaults["max_cat_levels"]
            if cols > 55:
                max_cat_levels = 32
            elif cols > 35 or rows > 200000:
                max_cat_levels = 64
            out = dict(defaults)
            out.update({"max_bins": max_bins, "max_cat_levels": max_cat_levels})
            return out, "artifact_plus_default"
    return defaults, "inferred_current_default"


PARSERS = {
    "arf": _parse_arf,
    "bayesnet": _parse_bayesnet,
    "ctgan": _parse_ctgan,
    "forestdiffusion": _parse_forestdiffusion,
    "realtabformer": _parse_realtabformer,
    "tabbyflow": _parse_tabbyflow,
    "tabddpm": _parse_tabddpm,
    "tabdiff": _parse_tabdiff,
    "tabpfgen": _parse_tabpfgen,
    "tabsyn": _parse_tabsyn,
    "tvae": _parse_tvae,
}


def _build_report(records: List[Dict[str, Any]], failures: List[Dict[str, Any]]) -> None:
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "processed": len(records),
        "updated": sum(1 for r in records if r["status"] == "updated"),
        "failed": len(failures),
        "hyperparam_source_counts": Counter(r["hyperparam_source"] for r in records if r["status"] == "updated"),
        "source_counts": Counter(r["source"] for r in records if r["status"] == "updated"),
        "model_counts": Counter(r["model"] for r in records if r["status"] == "updated"),
        "failures": failures,
        "records": records,
    }
    _write_json(REPORT_JSON, payload)

    lines = [
        "# Final Hyperparameter Backfill",
        "",
        f"- Processed: `{payload['processed']}`",
        f"- Updated: `{payload['updated']}`",
        f"- Failed: `{payload['failed']}`",
        "",
        "## Hyperparam Source Counts",
        "",
    ]
    for key, value in sorted(payload["hyperparam_source_counts"].items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Source Counts", ""])
    for key, value in sorted(payload["source_counts"].items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Failures", ""])
    if not failures:
        lines.append("- None")
    else:
        for item in failures:
            lines.append(
                f"- `{item['dataset']}/{item['model']}` from `{item['source']}`: `{item['reason']}`"
            )
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with REPORT_CSV.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["dataset", "model", "source", "status", "hyperparam_source", "final_json", "reason"],
        )
        writer.writeheader()
        for row in records:
            writer.writerow(
                {
                    "dataset": row["dataset"],
                    "model": row["model"],
                    "source": row["source"],
                    "status": row["status"],
                    "hyperparam_source": row.get("hyperparam_source"),
                    "final_json": row["final_json"],
                    "reason": row.get("reason"),
                }
            )


def main() -> int:
    provenance = _load_json(PROVENANCE_PATH)
    items = [item for item in provenance["items"] if item.get("kind") == "synthetic_csv"]
    records: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for item in items:
        ctx = _build_context(item)
        parser = PARSERS.get(ctx.model)
        record = {
            "dataset": ctx.dataset,
            "model": ctx.model,
            "source": ctx.source,
            "final_json": str(ctx.final_json),
            "source_ref": ctx.source_ref,
        }
        existing = _load_json(ctx.final_json)
        if existing.get("hyperparam_source") == "exact_artifact" and existing.get("train_hyperparams"):
            record["status"] = "updated"
            record["hyperparam_source"] = "exact_artifact"
            records.append(record)
            continue
        if parser is None:
            record["status"] = "failed"
            record["reason"] = "unsupported_model"
            records.append(record)
            failures.append(record)
            continue
        try:
            parsed = parser(ctx)
        except Exception as exc:  # pragma: no cover - audit path
            record["status"] = "failed"
            record["reason"] = f"{type(exc).__name__}: {exc}"
            records.append(record)
            failures.append(record)
            continue
        if not parsed:
            record["status"] = "failed"
            record["reason"] = "no_hyperparams_recovered"
            records.append(record)
            failures.append(record)
            continue

        params, hp_source = parsed
        final_path = ctx.final_json
        final_data = _load_json(final_path)
        final_data["train_hyperparams"] = params
        final_data["hyperparam_source"] = hp_source
        _write_json(final_path, final_data)

        record["status"] = "updated"
        record["hyperparam_source"] = hp_source
        records.append(record)

    _build_report(records, failures)
    print(json.dumps({"processed": len(records), "updated": sum(1 for r in records if r["status"] == "updated"), "failed": len(failures)}, ensure_ascii=False))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

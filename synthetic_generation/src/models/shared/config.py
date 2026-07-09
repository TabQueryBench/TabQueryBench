"""
Benchmark 专用模型配置

Docker 镜像、路径等配置。可通过环境变量覆盖。
镜像信息从 docker_images.json 索引文件自动加载。
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional


def _get_project_root() -> Path:
    """Return the synthetic_generation package root."""
    current = Path(__file__).resolve()
    return current.parents[3]


def get_release_root() -> Path:
    """
    公开打包后的 release 根目录。

    优先级：
    1) BENCHMARK_RELEASE_ROOT
    2) 向上查找同时包含 01_raw_csv_anonymized / 02_preprocessing_outputs 的目录
    3) 回退到当前代码包根目录
    """
    override = os.environ.get("BENCHMARK_RELEASE_ROOT", "").strip()
    if override:
        return Path(override).expanduser().resolve()

    current = Path(__file__).resolve()
    for candidate in current.parents:
        if (candidate / "01_raw_csv_anonymized").exists() and (
            candidate / "02_preprocessing_outputs"
        ).exists():
            return candidate.resolve()

    return _get_project_root()


def get_synthetic_benchmark_root() -> Path:
    """
    synthetic_benchmark root containing vendored upstream model sources.
    It can be overridden with BENCHMARK_SYNTHETIC_ROOT.
    """
    override = os.environ.get("BENCHMARK_SYNTHETIC_ROOT", "").strip()
    if override:
        return Path(override).resolve()
    project_local = (_get_project_root() / "synthetic_benchmark").resolve()
    if project_local.exists():
        return project_local
    release_root = get_release_root()
    if release_root != _get_project_root():
        return (release_root / "synthetic_benchmark").resolve()
    return (_get_project_root().parent / "synthetic_benchmark").resolve()


def get_dataset_root() -> Path:
    """Dataset 根目录"""
    root = _get_project_root()
    return root / "Dataset"


def get_legacy_dataset_root() -> Path:
    """旧版 Dataset 根目录（兼容 Tab-* / TS-*）"""
    return get_dataset_root()


def get_new_tabular_dataset_root() -> Path:
    """新版表格数据集根目录（DatasetNew）"""
    override = os.environ.get("BENCHMARK_NEW_DATASET_ROOT", "").strip()
    if override:
        return Path(override).expanduser().resolve()

    release_root = get_release_root()
    release_raw_root = release_root / "01_raw_csv_anonymized"
    if release_raw_root.exists():
        return release_raw_root.resolve()

    root = _get_project_root()
    return root / "DatasetNew"


def get_preprocessing_outputs_root() -> Path:
    """新版预处理输出根目录。"""
    override = os.environ.get("BENCHMARK_PREPROCESSING_ROOT", "").strip()
    if override:
        return Path(override).expanduser().resolve()

    release_root = get_release_root()
    return (release_root / "02_preprocessing_outputs").resolve()


def get_default_dataset_source() -> str:
    """
    默认数据源模式：auto | old | new
    可通过环境变量 BENCHMARK_DATASET_SOURCE 覆盖。
    """
    mode = os.environ.get("BENCHMARK_DATASET_SOURCE", "auto").strip().lower()
    if mode not in {"auto", "old", "new"}:
        return "auto"
    return mode


def get_output_base() -> Path:
    """
    专用模型产物根目录（runs、日志、合成 CSV）。

    优先级：
    1) 环境变量 BENCHMARK_OUTPUT_ROOT（绝对或相对路径均可）
    2) BENCHMARK_OUTPUT_LEGACY=1/true/yes → 历史目录 output-SpecializedModels
    3) 默认：output-Benchmark-trainonly-v1（与历史产物隔离）
    """
    root = _get_project_root()
    override = os.environ.get("BENCHMARK_OUTPUT_ROOT", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    if os.environ.get("BENCHMARK_OUTPUT_LEGACY", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        return (root / "output-SpecializedModels").resolve()
    return (root / "output-Benchmark-trainonly-v1").resolve()


# ---------------------------------------------------------------------------
# 从 docker_images.json 索引文件加载镜像配置
# ---------------------------------------------------------------------------
def _load_docker_images_index() -> Dict:
    """加载 docker_images.json，失败时返回空 dict"""
    index_path = Path(__file__).resolve().parent / "docker_images.json"
    if not index_path.exists():
        return {}
    with open(index_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("images", {})


_IMAGES_INDEX = _load_docker_images_index()

# Docker 镜像配置 —— 自动从索引文件生成，环境变量可覆盖
DOCKER_IMAGES: Dict[str, str] = {}
for _model_name, _info in _IMAGES_INDEX.items():
    _env_key = _info.get("env_override", "")
    _default = _info.get("image", "")
    DOCKER_IMAGES[_model_name] = os.environ.get(_env_key, _default) if _env_key else _default

# 模型与 Docker 镜像映射（同 DOCKER_IMAGES）
MODEL_DOCKER_MAP: Dict[str, str] = dict(DOCKER_IMAGES)

# 支持的模型列表
SUPPORTED_MODELS = list(MODEL_DOCKER_MAP.keys())

# 运行目录名中的模型前缀（用于 output/dataset/model/run_name 的 run_name）
# 如 rtf-tab-cate1-20260205_021040, ctgan-tab-cate1-20260205_021040
MODEL_RUN_PREFIX = {
    "ctgan": "ctgan",
    "tvae": "tvae",
    "realtabformer": "rtf",
    "tabsyn": "tabsyn",
    "tabddpm": "tabddpm",
    "crossformer": "crossformer",
    "bayesnet": "bayesnet",
    "arf": "arf",
    "relational_transformer": "rt",
    "plurel": "plurel",
    "tabpfgen": "tabpfgen",
    "tabdiff": "tabdiff",
    "tabbyflow": "tabbyflow",
    "forestdiffusion": "forest",
    "stasy": "stasy",
    "codi": "codi",
    "goggle": "goggle",
    "cdtd": "cdtd",
}

# 模型是否需要 GPU
MODEL_GPU_REQUIRED: Dict[str, bool] = {
    _model_name: _info.get("gpu", True)
    for _model_name, _info in _IMAGES_INDEX.items()
}

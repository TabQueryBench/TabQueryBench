"""
Features JSON 格式转换

将 Pipeline 的 Features JSON 转换为各模型所需的 metadata 格式。
"""

import json
from pathlib import Path
from typing import Any, Dict, List

# Pipeline data_type -> CTGAN type 映射
# ID/timestamp 含字符串，必须按 categorical 处理，否则 RDT mean() 会报错
# c3 等数据集的 id 列（如基因名）被误当 continuous 会导致 Could not convert string to numeric
_FEATURES_TO_CTGAN = {
    "categorical": "categorical",
    "binary": "categorical",
    "integer": "continuous",
    "continuous": "continuous",
    "ordinal": "ordinal",
    "timestamp": "categorical",
    "id": "categorical",
    "id_like": "categorical",
    "ID": "categorical",  # 显式支持大写（staging 输出 data_type="ID"）
    "datetime_like": "categorical",
    "text": "categorical",
    "others": "categorical",
}


def _normalize_profile_dtype(inferred_type: str) -> str:
    """
    新版 dataset_profile 的 inferred_type -> 旧版 data_type 映射
    """
    t = str(inferred_type or "").lower()
    mapping = {
        "categorical": "categorical",
        "boolean": "binary",
        "numerical": "continuous",
        "continuous": "continuous",
        "integer": "integer",
        "text": "categorical",
        "id_like": "ID",
        "datetime_like": "timestamp",
    }
    return mapping.get(t, "continuous")


def _profile_to_features(profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    将 DatasetNew 的 <id>-dataset_profile.json 转为统一 features 列表
    """
    columns = profile.get("columns", [])
    if not isinstance(columns, list):
        raise ValueError("dataset_profile.columns 必须是列表")

    target_candidates = profile.get("candidates", {}).get("target_candidates", []) or []
    target_set = set(target_candidates)
    column_profiles = profile.get("column_profiles", {}) or {}

    features: List[Dict[str, Any]] = []
    for idx, name in enumerate(columns):
        col_profile = column_profiles.get(name, {})
        features.append(
            {
                "feature_name": name,
                "data_type": _normalize_profile_dtype(col_profile.get("inferred_type")),
                "is_target": name in target_set,
                "index": idx,
            }
        )
    return features


def load_features_json(path: Path) -> List[Dict[str, Any]]:
    """加载 Pipeline Features JSON（列级列表格式）"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict) and "columns" in data:
        columns = data["columns"]
        # 旧版: {"columns": [{feature_name, data_type, ...}, ...]}
        if isinstance(columns, list) and (not columns or isinstance(columns[0], dict)):
            if not columns:
                return []
            first = columns[0]
            if "feature_name" in first:
                return columns
            if "name" in first:
                return [
                    {
                        "feature_name": c.get("name", f"col_{i}"),
                        "data_type": c.get("data_type", "continuous"),
                        "is_target": bool(c.get("is_target", False)),
                    }
                    for i, c in enumerate(columns)
                ]

        # 新版 profile: {"columns": ["col_a", ...], "column_profiles": {...}, ...}
        if isinstance(columns, list):
            return _profile_to_features(data)

    raise ValueError(f"无法解析 Features JSON 格式: {path}")


def features_to_ctgan_metadata(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    将 Pipeline Features 转为 CTGAN metadata 格式。

    CTGAN 格式: {"columns": [{"name": "col1", "type": "categorical"}, ...]}
    ID/text 列必须为 categorical，否则 TVAE/CTGAN 的 RDT mean() 会报 Could not convert string to numeric。
    """
    columns = []
    for feat in features:
        name = feat.get("feature_name")
        if not name:
            continue
        data_type = str(feat.get("data_type", "continuous")).strip().lower()
        semantic = str(feat.get("semantic_type") or "").lower()
        ctgan_type = _FEATURES_TO_CTGAN.get(data_type)
        if ctgan_type is None and semantic in ("id", "id_like", "text"):
            ctgan_type = "categorical"
        if ctgan_type is None:
            ctgan_type = "continuous"
        columns.append({"name": name, "type": ctgan_type})
    return {"columns": columns}


def write_ctgan_metadata(features: List[Dict[str, Any]], output_path: Path) -> Path:
    """将转换后的 CTGAN metadata 写入文件，返回输出路径"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    meta = features_to_ctgan_metadata(features)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return output_path


def convert_features_to_ctgan_metadata(
    features_json_path: Path, output_path: Path
) -> Path:
    """
    从 Pipeline Features JSON 文件转换并写入 CTGAN metadata。
    返回输出文件路径。
    """
    features = load_features_json(features_json_path)
    return write_ctgan_metadata(features, output_path)

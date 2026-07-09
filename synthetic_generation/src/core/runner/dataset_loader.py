"""
数据集加载器 (Dataset Loader)

功能：
- 从 dataset_index.json 加载数据集信息
- 根据数据集ID获取训练/测试数据路径
- 从特征文件自动读取列名分类（categorical/continuous）
- 支持绝对路径和相对路径解析
"""

import json
import os
import re
from pathlib import Path
from typing import Optional, Dict, Tuple, List, Any

from .config import (
    get_legacy_dataset_root,
    get_new_tabular_dataset_root,
    get_preprocessing_outputs_root,
    get_release_root,
)


class DatasetLoader:
    """数据集加载器，用于从索引文件获取数据集路径"""

    _NEW_DATASET_ID_PATTERN = re.compile(r"^[cmn]\d+$", re.IGNORECASE)

    def __init__(
        self,
        dataset_root: Optional[str] = None,
        index_path: Optional[str] = None,
        dataset_source: str = "auto",
        new_dataset_root: Optional[str] = None,
    ):
        """
        初始化数据集加载器

        Args:
            dataset_root: 旧版 Dataset 根目录路径（默认：自动检测）
            index_path: 索引文件路径（默认：dataset_root/dictionary/dataset_index.json）
            dataset_source: 数据源模式（auto|old|new）
            new_dataset_root: 新版表格数据根目录（默认：<project>/DatasetNew）
        """

        dataset_source = str(dataset_source or "auto").strip().lower()
        if dataset_source not in {"auto", "old", "new"}:
            raise ValueError(
                f"dataset_source 必须是 auto|old|new，当前: {dataset_source}"
            )
        self.dataset_source = dataset_source
        self.release_root = get_release_root()

        # 自动检测旧版数据根目录
        if dataset_root is None:
            dataset_root = get_legacy_dataset_root()
        else:
            dataset_root = Path(dataset_root)

        self.dataset_root = dataset_root.resolve()  # legacy root: Dataset/

        # 自动检测新版数据根目录 DatasetNew
        if new_dataset_root is None:
            new_dataset_root = get_new_tabular_dataset_root()
        else:
            new_dataset_root = Path(new_dataset_root)

        # 兼容两种结构：
        # 1) DatasetNew/c16/... + DatasetNew/artifacts/...
        # 2) DatasetNew/tabular_datasets/c16/... + .../artifacts/...
        resolved_root = new_dataset_root.resolve()
        nested_root = resolved_root / "tabular_datasets"
        if nested_root.exists() and not (resolved_root / "artifacts").exists():
            self.new_dataset_root = nested_root
        else:
            self.new_dataset_root = resolved_root

        release_preprocessing_root = get_preprocessing_outputs_root()
        if release_preprocessing_root.exists():
            self.new_preprocessing_root = release_preprocessing_root.resolve()
        else:
            self.new_preprocessing_root = None

        self.new_artifacts_root = self.new_dataset_root / "artifacts" / "data_core" / "tabular"

        # 确定索引文件路径
        if index_path is None:
            index_path = self.dataset_root / "dictionary" / "dataset_index.json"
        else:
            index_path = Path(index_path)

        self.index_path = index_path.resolve()

        # 加载旧版索引（old/auto 才需要）
        self._index: Dict = {}
        if self.dataset_source in {"old", "auto"}:
            self._load_index()

    def _load_index(self) -> None:
        """加载数据集索引文件"""
        if not self.index_path.exists():
            raise FileNotFoundError(
                f"数据集索引文件不存在: {self.index_path}\n"
                f"请确保已运行 Dataset/dictionary/json.py 生成索引文件"
            )

        with open(self.index_path, 'r', encoding='utf-8') as f:
            self._index = json.load(f)

    def _is_new_dataset_id(self, dataset_id: str) -> bool:
        """是否是新版数据集ID（c2/m1/n1）"""
        return bool(self._NEW_DATASET_ID_PATTERN.match(str(dataset_id).strip()))

    def _new_train_path(self, dataset_id: str) -> Path:
        return self.new_dataset_root / dataset_id / f"{dataset_id}-train.csv"

    def _new_val_path(self, dataset_id: str) -> Path:
        return self.new_dataset_root / dataset_id / f"{dataset_id}-val.csv"

    def _new_test_path(self, dataset_id: str) -> Path:
        return self.new_dataset_root / dataset_id / f"{dataset_id}-test.csv"

    def _new_main_path(self, dataset_id: str) -> Path:
        return self.new_dataset_root / dataset_id / f"{dataset_id}-main.csv"

    def _new_profile_candidates(self, dataset_id: str) -> List[Path]:
        artifacts_root = self.new_dataset_root / "artifacts"
        candidates = [
            artifacts_root / "data_core" / "tabular" / dataset_id / f"{dataset_id}-dataset_profile.json",
            artifacts_root / dataset_id / f"{dataset_id}-dataset_profile.json",
            artifacts_root / f"{dataset_id}-dataset_profile.json",
        ]
        if self.new_preprocessing_root is not None:
            candidates.append(
                self.new_preprocessing_root / dataset_id / "metadata_core" / "field_registry.json"
            )
        return candidates

    def _new_profile_path(self, dataset_id: str) -> Path:
        for p in self._new_profile_candidates(dataset_id):
            if p.exists():
                return p
        # 默认返回主路径，便于报错信息稳定
        return self._new_profile_candidates(dataset_id)[0]

    def _new_contract_candidates(self, dataset_id: str) -> List[Path]:
        artifacts_root = self.new_dataset_root / "artifacts"
        candidates = [
            artifacts_root / "data_core" / "tabular" / dataset_id / f"{dataset_id}-dataset_contract_v1.json",
            artifacts_root / dataset_id / f"{dataset_id}-dataset_contract_v1.json",
            artifacts_root / f"{dataset_id}-dataset_contract_v1.json",
        ]
        if self.new_preprocessing_root is not None:
            candidates.append(
                self.new_preprocessing_root / dataset_id / "metadata_core" / "field_registry.json"
            )
        return candidates

    def _new_contract_path(self, dataset_id: str) -> Path:
        for p in self._new_contract_candidates(dataset_id):
            if p.exists():
                return p
        return self._new_contract_candidates(dataset_id)[0]

    def _new_dataset_exists(self, dataset_id: str) -> bool:
        """新版数据集是否可用（至少 train 存在）"""
        return self._new_train_path(dataset_id).exists()

    def _old_dataset_exists(self, dataset_id: str) -> bool:
        """旧版数据集是否可用"""
        if not self._index:
            return False
        try:
            self._get_dataset_info(dataset_id)
            return True
        except (KeyError, ValueError):
            return False

    def _list_new_dataset_ids(self) -> List[str]:
        """扫描 DatasetNew 下可用的数据集ID（c*/m*/n*）"""
        root = self.new_dataset_root
        if not root.exists():
            return []

        dataset_ids: List[str] = []
        for p in sorted(root.iterdir()):
            if not p.is_dir():
                continue
            ds = p.name
            if not self._is_new_dataset_id(ds):
                continue
            if self._new_train_path(ds).exists():
                dataset_ids.append(ds)
        return dataset_ids

    def get_effective_source(self, dataset_id: str) -> str:
        """
        返回数据集最终解析来源（old/new）。
        auto 模式下优先：new(c*/m*/n*且存在) -> old -> new(仅当明确 new ID)。
        """
        dataset_id = str(dataset_id).strip()
        if self.dataset_source == "new":
            return "new"
        if self.dataset_source == "old":
            return "old"

        # auto mode
        if self._is_new_dataset_id(dataset_id) and self._new_dataset_exists(dataset_id):
            return "new"
        if self._old_dataset_exists(dataset_id):
            return "old"
        if self._is_new_dataset_id(dataset_id):
            return "new"
        return "old"

    def get_train_path(self, dataset_id: str) -> Path:
        """
        获取训练数据路径

        Args:
            dataset_id: 数据集ID，如 'Tab-Cate1', 'Tab-Mix7'

        Returns:
            训练数据文件的绝对路径

        Raises:
            KeyError: 如果数据集ID不存在
            FileNotFoundError: 如果训练文件不存在
        """
        source = self.get_effective_source(dataset_id)
        if source == "new":
            train_path = self._new_train_path(dataset_id)
            if not train_path.exists():
                raise FileNotFoundError(
                    f"新版训练文件不存在: {train_path}\n"
                    f"数据集ID: {dataset_id}\n"
                    f"请检查 DatasetNew/{dataset_id}/（或 DatasetNew/tabular_datasets/{dataset_id}/）"
                )
            return train_path

        dataset_info = self._get_dataset_info(dataset_id)

        # 优先使用 TrainTestDataset 中的 train_path（如果已配置）
        train_test_info = dataset_info.get("train_test")
        if train_test_info and "train_path" in train_test_info:
            train_path_str = train_test_info["train_path"]
            train_path = self.dataset_root / train_path_str
        else:
            # 自动检测：尝试在 TrainTestDataset 目录下查找 Train 文件
            dataset_type = self._get_dataset_type(dataset_id)
            auto_train_path = self.dataset_root / "TrainTestDataset" / dataset_type / dataset_id / f"{dataset_id}-Train.csv"

            if auto_train_path.exists():
                print(
                    f"[DatasetLoader] 自动检测到 Train 分割文件，使用: "
                    f"TrainTestDataset/{dataset_type}/{dataset_id}/{dataset_id}-Train.csv"
                )
                return auto_train_path

            # 向后兼容：如果还没有 TrainTestDataset 文件，则回退到 original.path
            original_info = dataset_info.get("original", {})
            orig_path_str = original_info.get("path")
            if not orig_path_str:
                raise KeyError(
                    f"数据集ID '{dataset_id}' 缺少 train_test.train_path 和 original.path 信息，"
                    f"请检查 Dataset/dictionary/dataset_index.json。"
                )
            print(
                f"[DatasetLoader] train_test 未配置且未找到自动 Train 文件，退回使用 original.path: "
                f"{orig_path_str}"
            )
            train_path = self.dataset_root / orig_path_str

        if not train_path.exists():
            raise FileNotFoundError(
                f"训练文件不存在: {train_path}\n"
                f"数据集ID: {dataset_id}\n"
                f"请检查 TrainTestDataset 或 Original 目录"
            )

        return train_path

    def get_test_path(self, dataset_id: str) -> Path:
        """
        获取测试数据路径

        Args:
            dataset_id: 数据集ID，如 'Tab-Cate1', 'Tab-Mix7'

        Returns:
            测试数据文件的绝对路径

        Raises:
            KeyError: 如果数据集ID不存在
            FileNotFoundError: 如果测试文件不存在
        """
        source = self.get_effective_source(dataset_id)
        if source == "new":
            test_path = self._new_test_path(dataset_id)
            if not test_path.exists():
                raise FileNotFoundError(
                    f"新版测试文件不存在: {test_path}\n"
                    f"数据集ID: {dataset_id}\n"
                    f"请检查 DatasetNew/{dataset_id}/（或 DatasetNew/tabular_datasets/{dataset_id}/）"
                )
            return test_path

        dataset_info = self._get_dataset_info(dataset_id)

        test_path_str = dataset_info["train_test"]["test_path"]
        test_path = self.dataset_root / test_path_str

        if not test_path.exists():
            raise FileNotFoundError(
                f"测试文件不存在: {test_path}\n"
                f"数据集ID: {dataset_id}\n"
                f"请检查 TrainTestDataset 目录"
            )

        return test_path

    def get_val_path(self, dataset_id: str) -> Path:
        """
        获取验证数据路径（主要用于新版 c*/m*/n*）。
        """
        source = self.get_effective_source(dataset_id)
        if source == "new":
            val_path = self._new_val_path(dataset_id)
            if not val_path.exists():
                raise FileNotFoundError(
                    f"新版验证文件不存在: {val_path}\n"
                    f"数据集ID: {dataset_id}\n"
                    f"请检查 DatasetNew/{dataset_id}/（或 DatasetNew/tabular_datasets/{dataset_id}/）"
                )
            return val_path
        raise ValueError(f"旧版数据源不支持 val split: dataset_id={dataset_id}")

    def get_original_path(self, dataset_id: str) -> Path:
        """
        获取原始数据路径

        Args:
            dataset_id: 数据集ID

        Returns:
            原始数据文件的绝对路径
        """
        source = self.get_effective_source(dataset_id)
        if source == "new":
            original_path = self._new_main_path(dataset_id)
            if not original_path.exists():
                raise FileNotFoundError(
                    f"新版主数据文件不存在: {original_path}\n"
                    f"数据集ID: {dataset_id}\n"
                    f"请检查 DatasetNew/{dataset_id}/（或 DatasetNew/tabular_datasets/{dataset_id}/）"
                )
            return original_path

        dataset_info = self._get_dataset_info(dataset_id)

        original_path_str = dataset_info["original"]["path"]
        original_path = self.dataset_root / original_path_str

        if not original_path.exists():
            raise FileNotFoundError(
                f"原始文件不存在: {original_path}\n"
                f"数据集ID: {dataset_id}"
            )

        return original_path

    def _get_dataset_info(self, dataset_id: str) -> Dict:
        """
        获取数据集信息

        Args:
            dataset_id: 数据集ID

        Returns:
            数据集信息字典

        Raises:
            KeyError: 如果数据集ID不存在
        """
        if not self._index:
            raise KeyError(
                "旧版 dataset_index 未加载，当前仅支持新版路径解析。"
            )

        # 数据集ID格式：Tab-Cate1, Tab-Mix7, Tab-Num10 等
        # 提取类型前缀：Tab-Cate, Tab-Mix, Tab-Num, TS-Cate, TS-Mix, TS-Num
        dataset_type = self._get_dataset_type(dataset_id)

        if dataset_type not in self._index:
            available_types = ', '.join(self._index.keys())
            raise KeyError(
                f"数据集类型 '{dataset_type}' 不存在\n"
                f"可用类型: {available_types}\n"
                f"数据集ID: {dataset_id}"
            )

        if dataset_id not in self._index[dataset_type]:
            available_ids = ', '.join(self._index[dataset_type].keys())
            raise KeyError(
                f"数据集ID '{dataset_id}' 在类型 '{dataset_type}' 中不存在\n"
                f"可用ID: {available_ids}"
            )

        return self._index[dataset_type][dataset_id]

    def list_datasets(self, dataset_type: Optional[str] = None) -> Dict[str, list]:
        """
        列出所有可用的数据集

        Args:
            dataset_type: 可选，指定类型（如 'Tab-Cate'），不指定则返回所有类型

        Returns:
            数据集ID列表，按类型组织
        """
        if self.dataset_source == "new":
            new_ids = self._list_new_dataset_ids()
            return {"tabular_new": new_ids}

        if self.dataset_source == "auto":
            new_ids = self._list_new_dataset_ids()
            if new_ids:
                result = {"tabular_new": new_ids}
                if dataset_type:
                    return result
                # auto 模式补充旧版列表（兼容历史调用）
                for dtype, datasets in self._index.items():
                    result[dtype] = list(datasets.keys())
                return result

        if dataset_type:
            if dataset_type not in self._index:
                return {}
            return {dataset_type: list(self._index[dataset_type].keys())}
        else:
            return {
                dtype: list(datasets.keys())
                for dtype, datasets in self._index.items()
            }

    def validate_dataset_id(self, dataset_id: str) -> bool:
        """
        验证数据集ID是否存在

        Args:
            dataset_id: 数据集ID

        Returns:
            如果存在返回True，否则返回False
        """
        source = self.get_effective_source(dataset_id)
        if source == "new":
            return self._new_dataset_exists(dataset_id)

        try:
            self._get_dataset_info(dataset_id)
            return True
        except (KeyError, ValueError):
            return False

    def _get_dataset_type(self, dataset_id: str) -> str:
        """
        从数据集ID提取类型

        Args:
            dataset_id: 数据集ID，如 'Tab-Cate1', 'Tab-Mix7', 'Tab-Num10'

        Returns:
            数据集类型，如 'Tab-Cate', 'Tab-Mix', 'Tab-Num'
        """
        import re

        # 已知的数据集类型
        known_types = ['Tab-Cate', 'Tab-Mix', 'Tab-Num', 'TS-Cate', 'TS-Mix', 'TS-Num']

        # 首先尝试匹配已知类型（最简单直接的方法）
        for known_type in known_types:
            if dataset_id.startswith(known_type):
                return known_type

        # 如果都不匹配，尝试使用正则表达式去掉末尾数字
        # 例如：Tab-Cate1 -> Tab-Cate, Tab-Mix7 -> Tab-Mix
        pattern = r'^(.+?)-(\d+)$'
        match = re.match(pattern, dataset_id)

        if match:
            # 去掉最后的数字部分，然后提取类型
            base_name = match.group(1)  # 例如：Tab-Cate
            parts = base_name.split('-')
            if len(parts) >= 2:
                return f"{parts[0]}-{parts[1]}"

        # 如果都不行，尝试直接split
        parts = dataset_id.split('-')
        if len(parts) >= 2:
            return f"{parts[0]}-{parts[1]}"

        raise ValueError(f"无效的数据集ID格式: {dataset_id}，无法提取类型")

    def get_features_path(self, dataset_id: str) -> Path:
        """
        获取特征 JSON 文件路径（Pipeline 元数据格式）

        Args:
            dataset_id: 数据集ID

        Returns:
            特征文件路径（文件可能不存在，需调用方校验）
        """
        return self._get_features_path(dataset_id)

    def get_contract_path(self, dataset_id: str) -> Path:
        """
        获取新版 dataset contract 路径（old 数据源暂无对应 contract 概念）。
        """
        source = self.get_effective_source(dataset_id)
        if source != "new":
            raise ValueError(
                f"contract 仅支持新版数据源，当前 source={source}, dataset_id={dataset_id}"
            )
        return self._new_contract_path(dataset_id)

    def _get_features_path(self, dataset_id: str) -> Path:
        """
        获取特征文件路径

        Args:
            dataset_id: 数据集ID

        Returns:
            特征JSON文件路径
        """
        source = self.get_effective_source(dataset_id)
        if source == "new":
            profile = self._new_profile_path(dataset_id)
            return profile

        dataset_type = self._get_dataset_type(dataset_id)
        features_dir = self.dataset_root / "Features" / dataset_type
        features_file = features_dir / f"{dataset_id}-features.json"
        return features_file

    def _get_ts_mapping_path(self, dataset_id: str) -> Path:
        """
        获取时间序列映射文件路径（TS 专用）

        Args:
            dataset_id: 数据集ID

        Returns:
            映射 JSON 文件路径
        """
        dataset_type = self._get_dataset_type(dataset_id)
        features_dir = self.dataset_root / "Features" / dataset_type
        mapping_file = features_dir / f"{dataset_id}-mapping.json"
        return mapping_file

    def _get_ts_diagnostics_path(self, dataset_id: str) -> Path:
        """
        获取时间序列诊断文件路径（TS 专用）

        Args:
            dataset_id: 数据集ID

        Returns:
            诊断 JSON 文件路径
        """
        dataset_type = self._get_dataset_type(dataset_id)
        features_dir = self.dataset_root / "Features" / dataset_type
        diagnostics_file = features_dir / f"{dataset_id}-mapping_diagnostics.json"
        return diagnostics_file

    def get_ts_statistics(self, dataset_id: str) -> Dict[str, Any]:
        """
        获取时间序列数据集的统计信息（用于自动计算超参数）

        Returns:
            包含以下键的字典：
            - n_sessions: Session 总数量
            - max_session_length: 最大序列长度
            - n_features: 特征总数列数（Metadata + Measurement，不含 ID 和 Time）
            - n_rows_total: 总行数（可选）

        Raises:
            FileNotFoundError: 如果诊断文件不存在
        """
        diagnostics_file = self._get_ts_diagnostics_path(dataset_id)
        if not diagnostics_file.exists():
            raise FileNotFoundError(
                f"TS 诊断文件不存在: {diagnostics_file}\n"
                f"数据集ID: {dataset_id}\n"
                f"请先运行 Dataset/Features/analyze_features.py 生成诊断文件"
            )

        with open(diagnostics_file, "r", encoding="utf-8") as f:
            diagnostics = json.load(f)

        # 从 diagnostics 中提取统计信息
        # 找到最终选中的 session_key 的统计信息
        n_sessions = None
        max_session_length = None
        n_rows_total = diagnostics.get("n_rows_total", None)

        # 遍历 samples，找到最终选中的 session_key
        if "samples" in diagnostics and len(diagnostics["samples"]) > 0:
            # 使用最后一个 sample（通常是最终验证的结果）
            final_sample = diagnostics["samples"][-1]
            session_key_selected = final_sample.get("session_key_selected")

            if session_key_selected and "session_key_evaluation" in final_sample:
                session_eval = final_sample["session_key_evaluation"].get(session_key_selected)
                if session_eval:
                    n_sessions = session_eval.get("n_sessions")
                    max_session_length = session_eval.get("max_session_length")

        # 从 mapping 文件计算特征数量
        mapping = self.get_ts_mapping(dataset_id)
        meta_cat = mapping.get("metadata_categorical") or []
        meta_cont = mapping.get("metadata_continuous") or []
        meas_cat = mapping.get("measurements_categorical") or []
        meas_cont = mapping.get("measurements_continuous") or []
        n_features = len(meta_cat) + len(meta_cont) + len(meas_cat) + len(meas_cont)

        result = {
            "n_features": n_features,
        }

        if n_sessions is not None:
            result["n_sessions"] = n_sessions
        if max_session_length is not None:
            result["max_session_length"] = max_session_length
        if n_rows_total is not None:
            result["n_rows_total"] = n_rows_total

        return result

    def _normalize_profile_dtype(self, inferred_type: str) -> str:
        """将新版 profile 的 inferred_type 归一到旧版 data_type"""
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

    def _load_feature_rows(self, features_file: Path) -> List[Dict[str, Any]]:
        """
        统一加载特征行：
        - 旧版: [{feature_name, data_type, is_target, ...}, ...]
        - 新版 profile: {columns: [...], column_profiles: {...}, ...}
        """
        with open(features_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return data

        if not isinstance(data, dict):
            raise ValueError(f"无法识别的特征文件格式: {features_file}")

        columns = data.get("columns")
        if isinstance(columns, list) and columns and isinstance(columns[0], dict):
            if "feature_name" in columns[0]:
                return columns
            if "name" in columns[0]:
                rows = []
                for idx, col in enumerate(columns):
                    rows.append({
                        "feature_name": col.get("name", f"col_{idx}"),
                        "data_type": col.get("data_type", "continuous"),
                        "is_target": bool(col.get("is_target", False)),
                    })
                return rows

        if isinstance(columns, list):
            target_candidates = (
                data.get("candidates", {}).get("target_candidates", []) or []
            )
            target_set = set(target_candidates)
            col_profiles = data.get("column_profiles", {}) or {}
            rows = []
            for idx, name in enumerate(columns):
                profile = col_profiles.get(name, {})
                rows.append({
                    "feature_name": name,
                    "data_type": self._normalize_profile_dtype(
                        profile.get("inferred_type")
                    ),
                    "is_target": name in target_set,
                })
            return rows

        raise ValueError(f"无法解析特征文件: {features_file}")

    def get_categorical_columns(self, dataset_id: str, exclude_target: bool = False) -> List[str]:
        """
        从特征文件获取分类列名（categorical + binary）

        Args:
            dataset_id: 数据集ID
            exclude_target: 是否排除target列（默认：False，包含target）

        Returns:
            分类列名列表

        Raises:
            FileNotFoundError: 如果特征文件不存在
        """
        features_file = self._get_features_path(dataset_id)

        if not features_file.exists():
            raise FileNotFoundError(
                f"特征文件不存在: {features_file}\n"
                f"数据集ID: {dataset_id}\n"
                f"请确保已运行 Dataset/Features/analyze_tabular_features.py 生成特征文件"
            )

        features = self._load_feature_rows(features_file)

        categorical_cols = []
        for feature in features:
            feature_name = feature.get("feature_name")
            data_type = feature.get("data_type")
            is_target = feature.get("is_target", False)

            # 跳过target列（如果设置了排除）
            if exclude_target and is_target:
                continue

            # categorical = categorical + binary
            if data_type in ["categorical", "binary"]:
                categorical_cols.append(feature_name)

        return categorical_cols

    def get_continuous_columns(self, dataset_id: str, exclude_target: bool = False) -> List[str]:
        """
        从特征文件获取连续列名（integer + continuous，排除timestamp）

        Args:
            dataset_id: 数据集ID
            exclude_target: 是否排除target列（默认：False，包含target）

        Returns:
            连续列名列表

        Raises:
            FileNotFoundError: 如果特征文件不存在
        """
        features_file = self._get_features_path(dataset_id)

        if not features_file.exists():
            raise FileNotFoundError(
                f"特征文件不存在: {features_file}\n"
                f"数据集ID: {dataset_id}\n"
                f"请确保已运行 Dataset/Features/analyze_tabular_features.py 生成特征文件"
            )

        features = self._load_feature_rows(features_file)

        continuous_cols = []
        for feature in features:
            feature_name = feature.get("feature_name")
            data_type = feature.get("data_type")
            is_target = feature.get("is_target", False)

            # 跳过target列（如果设置了排除）
            if exclude_target and is_target:
                continue

            # 跳过timestamp列
            if data_type == "timestamp":
                continue

            # continuous = integer + continuous
            if data_type in ["integer", "continuous"]:
                continuous_cols.append(feature_name)

        return continuous_cols

    # ============================
    # TS 字段映射 (Time Series) 相关
    # ============================

    def get_ts_mapping(self, dataset_id: str) -> Dict[str, Any]:
        """
        获取时间序列数据集的字段映射（由 Dataset/Features/analyze_features.py 生成）

        映射文件示例字段：
        - timestamp
        - session_key
        - metadata_categorical
        - metadata_continuous
        - measurements_categorical
        - measurements_continuous

        Args:
            dataset_id: 数据集ID，如 'TS-Mix1'

        Returns:
            映射字典

        Raises:
            FileNotFoundError: 如果映射文件不存在
            ValueError: 如果映射文件格式不合法
        """
        mapping_file = self._get_ts_mapping_path(dataset_id)
        if not mapping_file.exists():
            raise FileNotFoundError(
                f"TS 映射文件不存在: {mapping_file}\n"
                f"数据集ID: {dataset_id}\n"
                f"请先运行 Dataset/Features/analyze_features.py 生成映射文件"
            )
        with open(mapping_file, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        if not isinstance(mapping, dict):
            raise ValueError(f"TS 映射文件格式错误（应为JSON对象）: {mapping_file}")
        return mapping

    def get_ts_field_mapping(
        self, dataset_id: str
    ) -> Tuple[Optional[str], Optional[str], List[str], List[str], List[str], List[str]]:
        """
        方便 TS 模型直接获取字段映射的拆解结果。

        Returns:
            (timestamp, session_key,
             metadata_categorical, metadata_continuous,
             measurements_categorical, measurements_continuous)
        """
        mapping = self.get_ts_mapping(dataset_id)
        timestamp = mapping.get("timestamp")
        session_key = mapping.get("session_key")
        meta_cat = mapping.get("metadata_categorical") or []
        meta_cont = mapping.get("metadata_continuous") or []
        meas_cat = mapping.get("measurements_categorical") or []
        meas_cont = mapping.get("measurements_continuous") or []
        # 确保都是列表
        meta_cat = list(meta_cat)
        meta_cont = list(meta_cont)
        meas_cat = list(meas_cat)
        meas_cont = list(meas_cont)
        return timestamp, session_key, meta_cat, meta_cont, meas_cat, meas_cont

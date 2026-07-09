"""
模型适配器基类

定义统一的 train / generate 接口，所有专用模型适配器需继承此类。
提供通用的 Docker 运行辅助方法。
"""

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import math
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


_CATEGORICAL_DTYPES = {
    "categorical",
    "binary",
    "ordinal",
    "id",
    "id_like",
    "ID",
    "timestamp",
    "datetime",
    "datetime_like",
    "text",
    "others",
}


def _normalize_category_value(value: Any) -> Optional[str]:
    if pd.isna(value):
        return None
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isfinite(value) and value.is_integer():
            return str(int(value))
        return str(value)
    return str(value)


def _docker_force_remove_container(cidfile: Path) -> None:
    """
    若 docker run 异常退出、宿主机 SIGTERM（如定时关 GPU / 调度器回收），
    `--rm` 未必执行；根据 --cidfile 中的容器 ID 强制删除，避免残留占 GPU/磁盘。
    """
    try:
        if not cidfile.is_file():
            return
        cid = cidfile.read_text(encoding="utf-8").strip()
        if cid:
            subprocess.run(
                ["docker", "rm", "-f", cid],
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
    except Exception:
        pass
    try:
        cidfile.unlink(missing_ok=True)
    except Exception:
        pass


class DockerRunOutcome:
    """`_run_docker` 返回值：行为与 CompletedProcess 一致，并附带宿主机墙钟计时（供写日志）。"""

    __slots__ = ("_cp", "bench_timing")

    def __init__(
        self,
        completed: subprocess.CompletedProcess,
        bench_timing: Dict[str, Any],
    ) -> None:
        self._cp = completed
        self.bench_timing = bench_timing

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cp, name)


def _write_docker_log(
    log_path: Path,
    stdout: str,
    stderr: str,
    bench_timing: Optional[Dict[str, Any]] = None,
) -> None:
    """将 Docker 运行的 stdout/stderr 写入日志文件；可选在文首写入宿主机墙钟时间元数据。"""
    log_path = Path(log_path)
    header_parts: List[str] = []
    if bench_timing:
        header_parts.append("=== benchmark docker timing (host wall clock) ===")
        for key in (
            "adapter_model",
            "docker_image",
            "started_at_utc",
            "finished_at_utc",
            "elapsed_seconds",
        ):
            if key in bench_timing:
                header_parts.append(f"{key}: {bench_timing[key]}")
        for k, v in sorted(bench_timing.items()):
            if k in {
                "adapter_model",
                "docker_image",
                "started_at_utc",
                "finished_at_utc",
                "elapsed_seconds",
            }:
                continue
            header_parts.append(f"{k}: {v}")
        header_parts.append("=== docker merged stdout/stderr ===")
    header = ("\n".join(header_parts) + "\n") if header_parts else ""

    def _write(p: Path) -> None:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            if header:
                f.write(header)
            if stdout:
                f.write(stdout)
            if stderr:
                if stdout or header:
                    f.write("\n--- stderr ---\n")
                f.write(stderr)

    try:
        _write(log_path)
    except PermissionError:
        fallback = log_path.parent.parent / log_path.name
        print(f"[log] 权限不足: {log_path}, 回退 -> {fallback}")
        _write(fallback)


class BaseModelAdapter(ABC):
    """所有表格合成模型的统一适配接口"""

    # --------------- 子类必须实现 ---------------

    @property
    @abstractmethod
    def model_name(self) -> str:
        """模型标识，如 'ctgan', 'realtabformer'"""
        pass

    @property
    @abstractmethod
    def docker_image(self) -> str:
        """Docker 镜像名"""
        pass

    # --------------- 通用辅助 ---------------

    @staticmethod
    def read_staged_csv(csv_path: Path) -> pd.DataFrame:
        """
        读取 Public Gate 写出的 staged CSV（或与 DatasetNew contract.encoding 一致的源文件）。
        使用 utf-8-sig，与 contract 中 utf-8-sig 对齐并自动剥离 BOM，避免首列表头错位。
        """
        return pd.read_csv(Path(csv_path), encoding="utf-8-sig", low_memory=False)

    def _get_project_root(self) -> Path:
        """Return the synthetic_generation package root."""
        from ..config import get_dataset_root
        return get_dataset_root().parent

    def _to_container_path(self, host_path: Path) -> str:
        """将宿主机路径转为容器内 /work 下的路径"""
        root = self._get_project_root()
        try:
            rel = Path(host_path).resolve().relative_to(root.resolve())
        except ValueError:
            return str(host_path)
        return f"/work/{rel.as_posix()}"

    def _gpu_env_key(self) -> str:
        """GPU 环境变量名，子类可覆盖"""
        return f"BENCHMARK_{self.model_name.upper()}_GPUS"

    @property
    def _needs_gpu(self) -> bool:
        """是否需要 GPU，默认 True，子类可覆盖"""
        return True

    def _extra_volumes(self):
        """额外的 Docker 挂载，返回 [(host_path, container_path), ...]，子类可覆盖"""
        return []

    def _python_cmd(self) -> str:
        """容器内 Python 命令，默认 python，子类可覆盖为 python3"""
        return "python"

    def _build_docker_cmd(
        self,
        extra_args: List[str],
        extra_env: Optional[Dict[str, str]] = None,
        container_workdir: Optional[str] = None,
        cidfile: Optional[Path] = None,
    ) -> List[str]:
        """构建 docker run 命令（不执行）

        container_workdir: 容器内工作目录，须为 /work/... 下路径；默认 /work。
        并发训练时若各任务把 checkpoint 写在 cwd 下，应设为每次 run 唯一目录，避免共写 rtf_checkpoints。
        cidfile: 若提供，则写入容器 ID，便于异常/信号时 `docker rm -f`。
        """
        root = self._get_project_root()
        # --rm：容器退出即删，不留下 stopped 容器。
        # 挂载均为宿主机 bind mount（-v 主机路径:容器路径），不会创建 Docker 匿名/命名 volume，
        # 因此一般无需再 docker volume rm；临时 cidfile 目录在 finally 中 rmtree。
        cmd = ["docker", "run", "--rm", "--init", "--user", f"{os.getuid()}:{os.getgid()}", "-e", f"HOME={os.environ.get('BENCHMARK_DOCKER_HOME', '/work/.home')}"]
        # 宽表 TabSyn 等：扩散 DataLoader 多进程默认 /dev/shm 过小会 Bus error，可 export BENCHMARK_DOCKER_SHM_SIZE=8g
        _shm = os.environ.get("BENCHMARK_DOCKER_SHM_SIZE", "").strip()
        if _shm:
            cmd += ["--shm-size", _shm]
        if cidfile is not None:
            cidfile = Path(cidfile).resolve()
            cidfile.parent.mkdir(parents=True, exist_ok=True)
            cmd += ["--cidfile", str(cidfile)]

        # GPU — 直接用 --gpus "device=N" 指定具体显卡
        if self._needs_gpu:
            gpus = os.environ.get(self._gpu_env_key(), "all")
            if gpus and gpus != "all":
                # "device=0" / "device=2" → 直接传给 --gpus
                cmd += ["--gpus", gpus]
            else:
                cmd += ["--gpus", "all"]

        # 额外环境变量（如 OPENBLAS_NUM_THREADS）
        if extra_env:
            for k, v in extra_env.items():
                cmd += ["-e", f"{k}={v}"]

        # 挂载项目根目录；工作目录默认可覆盖为某次 run 专属路径
        wdir = container_workdir if container_workdir else "/work"
        cmd += ["-v", f"{root}:/work", "-w", wdir]
        # 额外挂载（子类可通过 _extra_volumes 添加）
        for host_path, container_path in self._extra_volumes():
            cmd += ["-v", f"{host_path}:{container_path}"]
        # 镜像
        cmd.append(self.docker_image)
        # 模型命令
        cmd += extra_args
        return cmd

    def _run_docker(
        self,
        extra_args: List[str],
        extra_env: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
        container_workdir: Optional[str] = None,
    ) -> DockerRunOutcome:
        """
        执行 docker run，实时流式输出训练日志到终端，同时捕获完整输出用于写日志文件。
        stdout 和 stderr 合并输出，方便看到 tqdm 进度条和训练 loss 等信息。
        返回 DockerRunOutcome（含 bench_timing：UTC 起止时间与 elapsed_seconds）。
        """
        cid_dir = Path(tempfile.mkdtemp(prefix=f"bench_docker_{self.model_name}_"))
        cidfile = cid_dir / "container.cid"
        cmd = self._build_docker_cmd(extra_args, extra_env, container_workdir, cidfile=cidfile)
        print(f"[{self.model_name}] 启动 Docker: {self.docker_image}")

        prev_int = signal.getsignal(signal.SIGINT)
        prev_term = signal.getsignal(signal.SIGTERM)

        proc_holder: List[Optional[subprocess.Popen]] = [None]

        def _signal_cleanup(signum: int, frame) -> None:
            _docker_force_remove_container(cidfile)
            p = proc_holder[0]
            if p is not None and p.poll() is None:
                try:
                    p.terminate()
                except Exception:
                    try:
                        p.kill()
                    except Exception:
                        pass

        proc: Optional[subprocess.Popen] = None
        output_lines: List[str] = []
        captured = ""
        t_wall0 = time.perf_counter()
        started_at_utc = datetime.now(timezone.utc).isoformat()

        def _make_timing(finished_at_utc: str) -> Dict[str, Any]:
            elapsed = time.perf_counter() - t_wall0
            return {
                "adapter_model": self.model_name,
                "docker_image": self.docker_image,
                "started_at_utc": started_at_utc,
                "finished_at_utc": finished_at_utc,
                "elapsed_seconds": round(elapsed, 3),
            }

        try:
            signal.signal(signal.SIGINT, _signal_cleanup)
            signal.signal(signal.SIGTERM, _signal_cleanup)

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # stderr 合并到 stdout，统一输出
                text=True,
                bufsize=1,  # 行缓冲
            )
            proc_holder[0] = proc

            try:
                if proc.stdout is not None:
                    for line in proc.stdout:
                        sys.stdout.write(line)
                        sys.stdout.flush()
                        output_lines.append(line)
            except Exception:
                pass

            proc.wait()
            captured = "".join(output_lines)
            finished_at_utc = datetime.now(timezone.utc).isoformat()
            timing = _make_timing(finished_at_utc)

            if proc.returncode != 0:
                print(f"\n[{self.model_name}] Docker 退出码: {proc.returncode}")
                e = subprocess.CalledProcessError(proc.returncode, cmd, output=captured, stderr="")
                e.stdout = captured
                e.stderr = ""
                e.bench_timing = timing  # type: ignore[attr-defined]
                raise e

            cp = subprocess.CompletedProcess(cmd, proc.returncode, stdout=captured, stderr="")
            return DockerRunOutcome(cp, timing)
        except BaseException:
            if proc is not None and proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=15)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
            raise
        finally:
            signal.signal(signal.SIGINT, prev_int)
            signal.signal(signal.SIGTERM, prev_term)
            _docker_force_remove_container(cidfile)
            try:
                shutil.rmtree(cid_dir, ignore_errors=True)
            except Exception:
                pass

    def _write_bridge_script(self, work_dir: Path, script_name: str, code: str) -> Path:
        """
        在 work_dir 下写入一个临时 Python 脚本（用于在 Docker 内执行）。
        返回宿主机上的脚本路径。
        """
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        script_path = work_dir / script_name
        script_path.write_text(code, encoding="utf-8")
        return script_path

    def _resolve_model_inputs(
        self,
        csv_path: Path,
        json_path: Path,
        kwargs: Dict[str, Any],
    ) -> Tuple[Path, Path]:
        """
        Resolve adapter input paths from optional staged model manifest.
        """
        manifest = kwargs.get("model_input_manifest")
        if manifest:
            manifest_path = Path(manifest)
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            csv_path = Path(data["train_csv"])
            json_path = Path(data["features_json"])
        return Path(csv_path), Path(json_path)

    def _validate_model_inputs(
        self,
        csv_path: Path,
        json_path: Path,
        require_target: bool = True,
        strict_numeric_cast: bool = False,
    ) -> Dict[str, Any]:
        """
        Shared adapter-side checks:
        - staged features schema exists and has unique target (if required)
        - target column exists in CSV header
        - optional strict numeric cast validation for numeric columns
        """
        from .features_converter import load_features_json

        df = self.read_staged_csv(csv_path)
        features = load_features_json(json_path)
        target_cols = [f.get("feature_name") for f in features if f.get("is_target")]

        if require_target and len(target_cols) != 1:
            raise ValueError(
                f"{self.model_name}: requires exactly one target column, got {target_cols}"
            )
        target_col = target_cols[0] if target_cols else None
        if target_col and target_col not in df.columns:
            raise ValueError(
                f"{self.model_name}: target column '{target_col}' not in CSV columns"
            )

        numeric_cols = []
        for f in features:
            name = f.get("feature_name")
            if not name or name not in df.columns:
                continue
            if f.get("is_target"):
                continue
            dtype = str(f.get("data_type", "")).lower()
            if dtype in ("continuous", "integer"):
                numeric_cols.append(name)

        if strict_numeric_cast:
            for c in numeric_cols:
                casted = pd.to_numeric(df[c], errors="coerce")
                bad = df[c].notna() & casted.isna()
                if bad.any():
                    examples = df.loc[bad, c].astype(str).head(5).tolist()
                    raise ValueError(
                        f"{self.model_name}: numeric cast failed for column '{c}', "
                        f"examples={examples}"
                    )

        return {
            "target_col": target_col,
            "numeric_cols": numeric_cols,
            "features": features,
        }

    def _postprocess_generated_csv(
        self,
        output_csv: Path,
        csv_path: Optional[Path],
        json_path: Optional[Path],
        num_rows: int,
        **kwargs,
    ) -> Path:
        """
        Enforce the public schema before a generated CSV can be marked successful.

        This is intentionally strict: if the model emits the wrong row count,
        columns, or non-integral values for integer columns, generation fails
        instead of silently publishing a malformed synthetic table.
        """
        from .features_converter import load_features_json

        if kwargs.get("model_input_manifest"):
            csv_path, json_path = self._resolve_model_inputs(
                Path(csv_path or "."), Path(json_path or "."), kwargs
            )
        if csv_path is None or json_path is None:
            raise ValueError(
                f"{self.model_name}: postprocess requires csv_path and json_path"
            )

        output_csv = Path(output_csv)
        ref_df = self.read_staged_csv(Path(csv_path))
        out_df = pd.read_csv(output_csv, encoding="utf-8-sig", low_memory=False)
        features = load_features_json(Path(json_path))

        expected_cols = list(ref_df.columns)
        missing = [c for c in expected_cols if c not in out_df.columns]
        extra = [c for c in out_df.columns if c not in expected_cols]
        if missing or extra:
            raise ValueError(
                f"{self.model_name}: generated CSV schema mismatch; "
                f"missing={missing}, extra={extra}"
            )
        if len(out_df) != int(num_rows):
            raise ValueError(
                f"{self.model_name}: generated row count mismatch; "
                f"got={len(out_df)}, expected={int(num_rows)}"
            )

        out_df = out_df[expected_cols].copy()

        feature_map = {
            f.get("feature_name"): f
            for f in features
            if f.get("feature_name") in out_df.columns
        }
        integer_cols = []
        numeric_cols = []
        categorical_cols = []
        for col in expected_cols:
            dtype = str(feature_map.get(col, {}).get("data_type", "")).strip().lower()
            if dtype == "integer" or pd.api.types.is_integer_dtype(ref_df[col].dtype):
                integer_cols.append(col)
            if dtype in ("continuous", "integer") or pd.api.types.is_numeric_dtype(ref_df[col].dtype):
                numeric_cols.append(col)
            if dtype in {t.lower() for t in _CATEGORICAL_DTYPES}:
                categorical_cols.append(col)

        for col in numeric_cols:
            numeric = pd.to_numeric(out_df[col], errors="coerce")
            bad = out_df[col].notna() & numeric.isna()
            if bad.any():
                examples = out_df.loc[bad, col].astype(str).head(5).tolist()
                raise ValueError(
                    f"{self.model_name}: numeric column '{col}' contains "
                    f"non-numeric generated values: {examples}"
                )
            finite_bad = numeric.notna() & ~numeric.apply(math.isfinite)
            if finite_bad.any():
                examples = out_df.loc[finite_bad, col].astype(str).head(5).tolist()
                raise ValueError(
                    f"{self.model_name}: numeric column '{col}' contains "
                    f"non-finite generated values: {examples}"
                )
            out_df[col] = numeric

        for col in integer_cols:
            numeric = pd.to_numeric(out_df[col], errors="coerce")
            bad = out_df[col].notna() & numeric.isna()
            if bad.any():
                examples = out_df.loc[bad, col].astype(str).head(5).tolist()
                raise ValueError(
                    f"{self.model_name}: integer column '{col}' contains "
                    f"non-numeric generated values: {examples}"
                )
            out_df[col] = numeric.round().astype("Int64")

        for col in categorical_cols:
            allowed_values = ref_df[col].dropna().tolist()
            allowed = {_normalize_category_value(v) for v in allowed_values}
            allowed.discard(None)
            canonical = {
                _normalize_category_value(v): v
                for v in allowed_values
                if _normalize_category_value(v) is not None
            }
            normalized = out_df[col].map(_normalize_category_value)
            missing_sentinel = normalized == "__nan__"
            if ref_df[col].isna().any() and missing_sentinel.any():
                out_df.loc[missing_sentinel, col] = pd.NA
                normalized = out_df[col].map(_normalize_category_value)

            bad = normalized.notna() & ~normalized.isin(allowed)
            if bad.any():
                examples = out_df.loc[bad, col].astype(str).head(5).tolist()
                raise ValueError(
                    f"{self.model_name}: categorical column '{col}' contains "
                    f"values outside training domain: {examples}"
                )

            for norm_value, original_value in canonical.items():
                if norm_value is None:
                    continue
                out_df.loc[normalized == norm_value, col] = original_value

        out_df.to_csv(output_csv, index=False, encoding="utf-8")
        return output_csv

    # --------------- 接口方法 ---------------

    def train(
        self,
        csv_path: Path,
        json_path: Path,
        work_dir: Path,
        epochs: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        执行训练。

        Args:
            csv_path: 训练 CSV 路径
            json_path: 特征 JSON 路径（Pipeline 格式）
            work_dir: 工作目录，用于存放模型、日志等
            epochs: 训练轮数（可选）
            **kwargs: 模型特定参数

        Returns:
            包含 model_path、train_meta 等信息的 dict，供 generate 使用
        """
        raise NotImplementedError

    def generate(
        self,
        model_path: Path,
        output_csv: Path,
        num_rows: int = 1000,
        csv_path: Optional[Path] = None,
        json_path: Optional[Path] = None,
        **kwargs,
    ) -> Path:
        """
        执行生成，输出合成 CSV。

        Args:
            model_path: 模型 checkpoint 或目录
            output_csv: 输出 CSV 路径
            num_rows: 生成行数
            csv_path: 原始训练 CSV（部分模型需要列信息）
            json_path: 特征 JSON（部分模型需要）
            **kwargs: 模型特定参数

        Returns:
            最终合成 CSV 路径
        """
        raise NotImplementedError

"""
修补 ctgan 与 RDT ClusterBasedNormalizer 组合时的逆变换维度/列语义错误。

背景
----
1. 上游 ``DataTransformer._inverse_transform_continuous`` 用 ``column_data[:, :2]``
   再套上 ``gm.get_output_sdtypes()`` 的全部列名；当 RDT 输出含 ``*.is_null`` 等
   多于 2 列时（训练列含 NaN），会触发维度错误。
2. ``_fit_continuous`` 里 ``output_dimensions = 1 + num_components`` **不计**
   ``missing_value_generation='from_column'`` 多出来的 ``is_null`` 列，所以 CTGAN/TVAE
   根本没有学习缺失；缺失值由适配器侧的 ``<col>__isna`` 指示列建模
   （见 ``ctgan_missing_indicator``）。

策略
----
- ``apply_ctgan_inverse_fix()`` 在 ``TVAE.load`` / ``CTGAN.load`` 之后、``sample`` 之前替换
  ``DataTransformer._inverse_transform_continuous``。
- 第 0 列为 normalized；簇分量 = ``argmax(column_data[:, 1:output_dimensions])``（与上游一致，
  覆盖**全部**分量）；RDT 声明但 CTGAN 矩阵未给出的列（``is_null``）置 0，即逆变换不产生 NaN。

注意：旧版本先把矩阵截断到 RDT 期望列数（通常 2~3 列）再 argmax，导致簇分量恒为 0
（或只在前两个分量里选）、``is_null`` 被填入 softmax 概率 → 连续列分布塌缩 / 整列 NaN。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_PATCHED = False


def _patched_inverse_transform_continuous(self, column_transform_info, column_data, sigmas, st):
    gm = column_transform_info.transform
    sdtypes = list(gm.get_output_sdtypes())
    ctgan_dim = int(column_transform_info.output_dimensions)

    x = np.asarray(column_data, dtype=float)
    if x.ndim != 2:
        x = x.reshape(x.shape[0], -1)
    n = x.shape[0]

    data = pd.DataFrame(np.zeros((n, len(sdtypes)), dtype=float), columns=sdtypes)

    norm_names = [s for s in sdtypes if str(s).endswith(".normalized")]
    norm_col = norm_names[0] if norm_names else sdtypes[0]
    normalized = x[:, 0]
    if sigmas is not None:
        normalized = np.random.normal(normalized, sigmas[st])
    data[norm_col] = normalized

    comp_names = [s for s in sdtypes if str(s).endswith(".component")]
    comp_col = comp_names[0] if comp_names else (sdtypes[1] if len(sdtypes) > 1 else None)
    if comp_col is not None:
        width = max(1, min(ctgan_dim, x.shape[1]) - 1)
        oh = x[:, 1 : 1 + width]
        data[comp_col] = np.argmax(oh, axis=1) if oh.shape[1] > 1 else 0

    # Remaining RDT columns (typically ``<col>.is_null``) stay 0: never emit NaN here.
    return gm.reverse_transform(data)


def apply_ctgan_inverse_fix() -> None:
    """将修补挂到 ``ctgan.data_transformer.DataTransformer``（幂等）。"""
    global _PATCHED
    if _PATCHED:
        return
    import ctgan.data_transformer as dt

    dt.DataTransformer._inverse_transform_continuous = _patched_inverse_transform_continuous
    _PATCHED = True

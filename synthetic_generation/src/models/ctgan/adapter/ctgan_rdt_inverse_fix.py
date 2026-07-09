"""
修补 ctgan 与 RDT ClusterBasedNormalizer 组合时的逆变换维度/列语义错误。

背景
----
1. 上游 ``DataTransformer._inverse_transform_continuous`` 用 ``column_data[:, :2]``
   再套上 ``gm.get_output_sdtypes()`` 的全部列名；当 RDT 输出含 ``*.is_null`` 等
   多于 2 列时，会触发维度错误或（旧版）手写 ``raise``。
2. 更关键：``_fit_continuous`` 里 ``output_dimensions = 1 + num_components`` **不计**
   ``ClusterBasedNormalizer`` 在 ``missing_value_generation='from_column'`` 下多出来的
   ``is_null`` 等列；TVAE/CTGAN 解码矩阵宽度按 ``output_dimensions`` 走，而
   ``reverse_transform`` 仍按 ``get_output_sdtypes()`` 的列数构造 DataFrame，
   于是出现 ``column_data`` 只有 2 列、RDT 期望 3 列（n13 / n18 一类列名：
   ``cmp_lname_c2.*`` / ``cd_000.*``）。

策略
----
- 仍用 ``apply_ctgan_inverse_fix()`` 在 ``TVAE.load`` / ``CTGAN.load`` 之后、
  ``sample`` 之前替换 ``DataTransformer._inverse_transform_continuous``。
- 逆变换时以 ``column_transform_info.output_dimensions``（与训练时展平宽度一致）
  切分：第 0 列为 normalized，第 ``1 : output_dimensions`` 为簇 softmax，
  其余 RDT 声明但 CTGAN 矩阵未给出的列（典型为 ``is_null``）用 **0** 填充，
  再按列名写回 ``component``（对 softmax 段做 ``argmax``），最后 ``gm.reverse_transform``。

参考：``synthetic_benchmark/CTGAN/ctgan/data_transformer.py`` 中
``_fit_continuous`` / ``_transform_continuous`` / ``_inverse_transform_continuous``。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_PATCHED = False


def _patched_inverse_transform_continuous(self, column_transform_info, column_data, sigmas, st):
    gm = column_transform_info.transform
    sdtypes = list(gm.get_output_sdtypes())
    n_expect = len(sdtypes)
    ctgan_dim = int(column_transform_info.output_dimensions)

    x = np.asarray(column_data, dtype=float)
    if x.ndim != 2:
        x = x.reshape(x.shape[0], -1)

    # 右侧补零到 RDT 期望列数（is_null 等）
    if x.shape[1] < n_expect:
        pad = np.zeros((x.shape[0], n_expect), dtype=float)
        c = min(x.shape[1], ctgan_dim, n_expect)
        pad[:, :c] = x[:, :c]
        x = pad

    x = x[:, :n_expect]
    data = pd.DataFrame(x.copy(), columns=sdtypes).astype(float)

    # CTGAN 连续列展平：col0 = normalized，col 1..output_dimensions-1 = 簇 one-hot
    n_soft = max(0, ctgan_dim - 1)
    if n_soft > 0 and n_expect >= 2:
        oh = x[:, 1 : 1 + n_soft]
        comp_names = [s for s in sdtypes if "component" in str(s)]
        comp_col = comp_names[0] if comp_names else sdtypes[1]
        if oh.shape[1] > 1:
            data[comp_col] = np.argmax(oh, axis=1)
        else:
            data[comp_col] = 0

    # 显式把「超出 ctgan 展平宽度」的 RDT 列置 0（pad 已为零，写回防万一）
    for j, name in enumerate(sdtypes):
        if j >= ctgan_dim:
            data[name] = x[:, j]

    if sigmas is not None:
        selected_normalized_value = np.random.normal(data.iloc[:, 0], sigmas[st])
        data.iloc[:, 0] = selected_normalized_value
    return gm.reverse_transform(data)


def apply_ctgan_inverse_fix() -> None:
    """将修补挂到 ``ctgan.data_transformer.DataTransformer``（幂等）。"""
    global _PATCHED
    if _PATCHED:
        return
    import ctgan.data_transformer as dt

    dt.DataTransformer._inverse_transform_continuous = _patched_inverse_transform_continuous
    _PATCHED = True

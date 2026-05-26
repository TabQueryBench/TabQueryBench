from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
from matplotlib import colormaps


HEATMAP_NA_HEX = "F1F1F1"

_BASE_HEATMAP_CMAP = colormaps["YlGnBu"].copy()
_BASE_HEATMAP_CMAP.set_bad(f"#{HEATMAP_NA_HEX}")


def get_heatmap_cmap():
    return _BASE_HEATMAP_CMAP


def coerce_heatmap_value(value: object) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric):
        return None
    return max(0.0, min(1.0, numeric))


def heatmap_hex(value: object) -> str:
    numeric = coerce_heatmap_value(value)
    if numeric is None:
        return HEATMAP_NA_HEX
    red, green, blue, _alpha = get_heatmap_cmap()(numeric)
    return "".join(f"{int(round(channel * 255)):02X}" for channel in (red, green, blue))


def text_hex_for_fill(fill_hex: str) -> str:
    red = int(fill_hex[0:2], 16)
    green = int(fill_hex[2:4], 16)
    blue = int(fill_hex[4:6], 16)
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "111111" if luminance >= 150 else "F8F8F8"


def format_heatmap_latex_cell(value: object, *, precision: int = 2, show_text: bool = False) -> str:
    numeric = coerce_heatmap_value(value)
    if numeric is None:
        return ""
    fill_hex = heatmap_hex(numeric)
    if not show_text:
        return rf"\cellcolor[HTML]{{{fill_hex}}}"
    text_hex = text_hex_for_fill(fill_hex)
    return (
        rf"\cellcolor[HTML]{{{fill_hex}}}"
        rf"\textcolor[HTML]{{{text_hex}}}{{{numeric:.{precision}f}}}"
    )

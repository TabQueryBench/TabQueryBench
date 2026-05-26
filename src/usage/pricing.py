"""Pricing config loading and cost estimation for token usage."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelPricing:
    input_per_million_usd: float
    cached_input_per_million_usd: float
    output_per_million_usd: float


def load_pricing_config(config_path: Path) -> dict:
    """Load pricing JSON. Missing or invalid file falls back to zero pricing."""
    if not config_path.exists():
        return {
            "default": {
                "input_per_million_usd": 0.0,
                "cached_input_per_million_usd": 0.0,
                "output_per_million_usd": 0.0,
            },
            "models": {},
        }

    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_model_pricing(model_name: str, pricing_config: dict) -> ModelPricing:
    """Resolve pricing for a model, falling back to default rates."""
    default_cfg = pricing_config.get("default", {})
    models_cfg = pricing_config.get("models", {})
    normalized_name = model_name.strip()
    if normalized_name.startswith("openai/"):
        normalized_name = normalized_name.split("/", 1)[1]

    model_cfg = models_cfg.get(normalized_name)
    if model_cfg is None:
        prefix_matches = [
            key for key in models_cfg.keys() if normalized_name == key or normalized_name.startswith(f"{key}-")
        ]
        if prefix_matches:
            longest_prefix = sorted(prefix_matches, key=len, reverse=True)[0]
            model_cfg = models_cfg[longest_prefix]
        else:
            model_cfg = default_cfg

    return ModelPricing(
        input_per_million_usd=float(model_cfg.get("input_per_million_usd", 0.0)),
        cached_input_per_million_usd=float(model_cfg.get("cached_input_per_million_usd", 0.0)),
        output_per_million_usd=float(model_cfg.get("output_per_million_usd", 0.0)),
    )


def calculate_cost_usd(
    input_tokens: int,
    output_tokens: int,
    pricing: ModelPricing,
    cached_input_tokens: int = 0,
) -> float:
    """Calculate USD cost using per-1M-token pricing."""
    effective_cached_tokens = max(cached_input_tokens, 0)
    effective_uncached_input_tokens = max(input_tokens - effective_cached_tokens, 0)

    input_cost = (effective_uncached_input_tokens / 1_000_000) * pricing.input_per_million_usd
    cached_input_cost = (effective_cached_tokens / 1_000_000) * pricing.cached_input_per_million_usd
    output_cost = (output_tokens / 1_000_000) * pricing.output_per_million_usd
    return input_cost + cached_input_cost + output_cost

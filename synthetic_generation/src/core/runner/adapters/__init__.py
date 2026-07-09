"""
Adapter registry for the benchmark runner.

This version avoids importing placeholder adapters that are not present in the
current source snapshot. It also installs compatibility aliases so the model
adapter modules can reuse the shared helper implementations under
`models/shared/`.
"""

from __future__ import annotations

import importlib
import sys
from typing import Dict, Tuple, Type

from models.shared.base_adapter import BaseModelAdapter


_ADAPTER_SPECS: Dict[str, Tuple[str, str]] = {
    "arf": ("models.arf.adapter.arf_adapter", "ARFAdapter"),
    "bayesnet": ("models.bayesnet.adapter.bayesnet_adapter", "BayesNetAdapter"),
    "ctgan": ("models.ctgan.adapter.ctgan_adapter", "CTGANAdapter"),
    "forestdiffusion": (
        "models.forestdiffusion.adapter.forestdiffusion_adapter",
        "ForestDiffusionAdapter",
    ),
    "realtabformer": (
        "models.realtabformer.adapter.realtabformer_adapter",
        "REaLTabFormerAdapter",
    ),
    "tabbyflow": ("models.tabbyflow.adapter.tabbyflow_adapter", "TabbyFlowAdapter"),
    "tabddpm": ("models.tabddpm.adapter.tabddpm_adapter", "TabDDPMAdapter"),
    "tabdiff": ("models.tabdiff.adapter.tabdiff_adapter", "TabDiffAdapter"),
    "tabpfgen": ("models.tabpfgen.adapter.tabpfgen_adapter", "TabPFGenAdapter"),
    "tabsyn": ("models.tabsyn.adapter.tabsyn_adapter", "TabSynAdapter"),
    "tvae": ("models.tvae.adapter.tvae_adapter", "TVAEAdapter"),
}

_FEATURES_HELPER_MODELS = {
    "bayesnet",
    "ctgan",
    "forestdiffusion",
    "realtabformer",
    "tabddpm",
    "tabpfgen",
    "tabsyn",
    "tvae",
}

_PIPELINE_BUNDLE_MODELS = {"tabdiff", "tabbyflow"}


def _install_compat_aliases() -> None:
    shared_base = importlib.import_module("models.shared.base_adapter")
    shared_config = importlib.import_module("models.shared.config")
    shared_features = importlib.import_module("models.shared.features_converter")
    shared_bundle = importlib.import_module("models.shared.pipeline_npy_bundle")

    for model_name in _ADAPTER_SPECS:
        sys.modules.setdefault(f"models.{model_name}.config", shared_config)
        sys.modules.setdefault(
            f"models.{model_name}.adapter.base_adapter",
            shared_base,
        )
        if model_name in _FEATURES_HELPER_MODELS:
            sys.modules.setdefault(
                f"models.{model_name}.adapter.features_converter",
                shared_features,
            )
        if model_name in _PIPELINE_BUNDLE_MODELS:
            sys.modules.setdefault(
                f"models.{model_name}.adapter.pipeline_npy_bundle",
                shared_bundle,
            )


def _load_adapter_class(model_name: str) -> Type[BaseModelAdapter]:
    _install_compat_aliases()
    module_name, class_name = _ADAPTER_SPECS[model_name]
    module = importlib.import_module(module_name)
    klass = getattr(module, class_name)
    if not issubclass(klass, BaseModelAdapter):
        raise TypeError(f"{class_name} is not a BaseModelAdapter")
    return klass


def get_adapter(model_name: str) -> BaseModelAdapter:
    name = str(model_name).strip().lower()
    if name not in _ADAPTER_SPECS:
        raise ValueError(
            f"Unknown model: {model_name}. Supported: {sorted(_ADAPTER_SPECS)}"
        )
    return _load_adapter_class(name)()


def list_adapters():
    return list(_ADAPTER_SPECS.keys())


__all__ = ["BaseModelAdapter", "get_adapter", "list_adapters"]

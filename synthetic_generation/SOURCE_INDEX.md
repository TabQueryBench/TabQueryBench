# Source Index

This file maps each synthetic generation model to its adapter path and vendored upstream source path.

| Model | Adapter | Upstream | Notes |
|---|---|---|---|
| `arf` | `src/models/arf/adapter/arf_adapter.py` | `src/models/arf/upstream/` | vendored ARF source snapshot |
| `bayesnet` | `src/models/bayesnet/adapter/bayesnet_adapter.py` | `src/models/bayesnet/upstream/` | vendored Bayesian network source snapshot |
| `ctgan` | `src/models/ctgan/adapter/` | `src/models/ctgan/upstream/ctgan/` | CTGAN adapter and local patches |
| `forestdiffusion` | `src/models/forestdiffusion/adapter/` | `src/models/forestdiffusion/upstream/ForestDiffusion/` | ForestDiffusion adapter |
| `realtabformer` | `src/models/realtabformer/adapter/` | `src/models/realtabformer/upstream/realtabformer/` | REaLTabFormer adapter |
| `tabbyflow` | `src/models/tabbyflow/adapter/` | `synthetic_benchmark/third_party/ef-vfm/` | TabbyFlow / ef-vfm adapter |
| `tabddpm` | `src/models/tabddpm/adapter/` | `synthetic_benchmark/tabddpm/code/` | TabDDPM adapter |
| `tabdiff` | `src/models/tabdiff/adapter/` | `synthetic_benchmark/third_party/TabDiff/` | TabDiff adapter |
| `tabpfgen` | `src/models/tabpfgen/adapter/` | `synthetic_benchmark/tabpfgen/src/tabpfgen/` | TabPFGen adapter |
| `tabsyn` | `src/models/tabsyn/adapter/` | `synthetic_benchmark/tabsyn/` | TabSyn adapter |
| `tvae` | `src/models/tvae/adapter/` | `src/models/tvae/upstream/ctgan/` | TVAE adapter using CTGAN-family source |

## Core Runtime Files

- `src/core/runner/runner.py`
- `src/core/runner/config.py`
- `src/core/runner/dataset_loader.py`
- `src/core/runner/adapters/__init__.py`
- `src/core/runner/staging/manager.py`
- `src/models/shared/base_adapter.py`
- `src/models/shared/features_converter.py`
- `src/models/shared/config.py`

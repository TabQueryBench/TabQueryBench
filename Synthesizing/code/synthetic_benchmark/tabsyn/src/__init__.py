"""
Trimmed vendored subset of amazon-science/tabsyn `src/` (originally from TabDDPM).

Only what `utils_train.py` needs is kept (data transforms + a few utils).
Dropped vs upstream: `deep.py`, `env.py`, `metrics.py`, icecream `install()`,
`torch.set_num_threads(1)`; these are unused by the VAE / diffusion / sampling path.
"""
from .data import *  # noqa
from .util import *  # noqa

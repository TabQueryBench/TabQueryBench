# -*- coding: utf-8 -*-
"""
Cap ``joblib.Parallel`` effective ``n_jobs`` before CTGAN/TVAE ``DataTransformer`` runs.

CTGAN uses ``Parallel(n_jobs=-1)`` during ``fit`` → ``transform``, which can spawn
many heavy workers and trigger SIGKILL (-9) on wide tables.  Lowering the cap trades
wall time for peak RAM.

**Opt-in only:** if ``TVAE_CTGAN_JOBTRANS_N_JOBS`` is unset or empty, this module does
nothing (normal datasets keep default joblib behaviour).

When set to a positive integer, it becomes the maximum ``n_jobs`` passed into
``Parallel`` (``-1`` and larger integers are clamped).

Typical usage (host or Docker ``-e``): see ``TVAEAdapter`` docstring for
``TVAE_CTGAN_JOBTRANS_N_JOBS``, ``TVAE_JOBTRANS_CAP_DATASETS``, ``TVAE_JOBTRANS_CAP_N_JOBS``.
"""

from __future__ import annotations

import os
from typing import Any

_applied = False


def _cap_njobs(n_jobs: Any, cap: int) -> Any:
    if n_jobs is None:
        return None
    if n_jobs == -1:
        return cap
    if isinstance(n_jobs, int):
        if n_jobs < 0 and n_jobs != -1:
            return n_jobs
        if n_jobs > cap:
            return cap
    return n_jobs


def apply_parallel_cap_from_env() -> None:
    """Monkeypatch ``joblib.parallel.Parallel.__init__`` once (idempotent).

    No-op when ``TVAE_CTGAN_JOBTRANS_N_JOBS`` is unset or non-positive.
    """
    global _applied
    if _applied:
        return
    raw = (os.environ.get("TVAE_CTGAN_JOBTRANS_N_JOBS") or "").strip()
    if not raw:
        return
    try:
        cap = int(raw)
    except ValueError:
        return
    if cap < 1:
        return

    import joblib.parallel as jp

    _orig = jp.Parallel.__init__

    def _wrapped(self, *args: Any, **kwargs: Any) -> None:
        kw = dict(kwargs)
        if "n_jobs" in kw:
            kw["n_jobs"] = _cap_njobs(kw["n_jobs"], cap)
            return _orig(self, *args, **kw)
        if args:
            head = _cap_njobs(args[0], cap)
            return _orig(self, head, *args[1:], **kw)
        return _orig(self, *args, **kw)

    jp.Parallel.__init__ = _wrapped  # type: ignore[method-assign]
    _applied = True

"""Canonical answer types used by SV2 scorers."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Any, Union

# A semantic key: a tuple (compound keys stay tuples; scalars use ("__scalar__",)).
Key = tuple[Hashable, ...]

# numeric_magnitude / rate_share_proportion: key -> finite float or None (SQL NULL).
ValueMap = dict[Key, Union[float, None]]

# count_support_distribution: (key, count) entries; duplicate keys are summed.
CountEntries = Sequence[tuple[Key, Any]]

# topk_ranking: ordered, unique identities.
Ranking = list[Hashable]

ScoreResult = dict[str, Any]

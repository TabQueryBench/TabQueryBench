"""Compatibility module for the default semantic scorer.

The implementation lives in ``Scoring/Scoring_Standard/standard_v1/scorer.py``.
"""

from __future__ import annotations

from tqb_scoring.standards import DEFAULT_SEMANTIC_STANDARD, load_standard

_standard = load_standard(DEFAULT_SEMANTIC_STANDARD)

globals().update({name: value for name, value in vars(_standard).items() if not name.startswith("__")})

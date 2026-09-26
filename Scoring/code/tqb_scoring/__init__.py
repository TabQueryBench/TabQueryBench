"""TabQueryBench scoring package (evaluation runners and scoring standards).

Scoring executes benchmark SQL, so it depends on ``tqb_query``. When only
``Scoring/code`` is on ``sys.path``, the sibling ``Query/code`` root is added
automatically.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:  # pragma: no cover - import side effect only
    import tqb_query  # noqa: F401
except ImportError:  # pragma: no cover
    _query_code_root = Path(__file__).resolve().parents[3] / "Query" / "code"
    if _query_code_root.is_dir():
        sys.path.append(str(_query_code_root))

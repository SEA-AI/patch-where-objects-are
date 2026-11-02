from __future__ import annotations

import sys
from pathlib import Path

# Allow running tests without `pip install -e .`
_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

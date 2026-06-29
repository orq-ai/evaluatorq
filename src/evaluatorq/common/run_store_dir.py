"""Resolve the on-disk store directory for persisted runs.

Both red teaming (``runs/``) and agent simulation (``sim-runs/``) persist
their reports under ``.evaluatorq/``. The base resolves to ``$EVALUATORQ_DIR``
when set (tests point this at a tmp dir so runs never leak into the repo
store), otherwise ``.evaluatorq`` relative to the current working directory.

``EVALUATORQ_DIR`` must point at the store directory itself (e.g.
``/tmp/x/.evaluatorq``), not its parent — only the cwd fallback appends
``.evaluatorq``. An unset or empty value uses the cwd fallback.
"""

from __future__ import annotations

import os
from pathlib import Path

STORE_DIR_NAME = ".evaluatorq"


def get_store_dir(subdir: str) -> Path:
    """Return ``<base>/<subdir>`` where base honors ``EVALUATORQ_DIR``."""
    base = os.environ.get("EVALUATORQ_DIR") or None  # treat "" as unset
    root = Path(base) if base else Path.cwd() / STORE_DIR_NAME
    return root / subdir

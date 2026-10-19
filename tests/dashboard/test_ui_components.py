"""Exercise the shared dashboard HTML components in a small DOM runtime."""

from __future__ import annotations

import subprocess
from pathlib import Path


def test_ui_components() -> None:
    script = Path(__file__).with_name('ui_components.cjs')
    result = subprocess.run(['node', str(script)], capture_output=True, text=True, check=False, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr

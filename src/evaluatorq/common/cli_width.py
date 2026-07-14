"""Fix Rich help/error rendering under a non-TTY stdout.

Importing this module (side effect only) forces a sane terminal width when
stdout is not a TTY and COLUMNS is missing or bogus. Typer's Rich renderer
emits *nothing* at width 0 — which is what Rich reports for pipes/CI, and some
harnesses even export ``COLUMNS=0``. So ``eq --help | cat``, usage errors, and
CliRunner-based tests would otherwise come back empty. Real terminals keep
their auto-detected width untouched.
"""

from __future__ import annotations

import os
import sys

if not sys.stdout.isatty():
    try:
        _cols_ok = int(os.environ.get('COLUMNS', '0')) > 0
    except ValueError:
        _cols_ok = False
    if not _cols_ok:
        os.environ['COLUMNS'] = '120'

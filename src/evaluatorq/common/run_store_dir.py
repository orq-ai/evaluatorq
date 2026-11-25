"""Resolve the on-disk store directory for persisted runs, and write reports into it.

Red teaming (``runs/``), agent simulation (``sim-runs/``), pairwise judging
(``pairwise-runs/``) and ``evaluatorq()`` experiments (``experiment-runs/``)
persist their reports under ``.evaluatorq/``. The base resolves to ``$EVALUATORQ_DIR``
when set (tests point this at a tmp dir so runs never leak into the repo
store), otherwise ``.evaluatorq`` relative to the current working directory.

``EVALUATORQ_DIR`` must point at the store directory itself (e.g.
``/tmp/x/.evaluatorq``), not its parent — only the cwd fallback appends
``.evaluatorq``. An unset or empty value uses the cwd fallback.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

from evaluatorq.common.env_config import env_raw

STORE_DIR_NAME = '.evaluatorq'


def get_store_dir(subdir: str) -> Path:
    """Return ``<base>/<subdir>`` where base honors ``EVALUATORQ_DIR``."""
    base = env_raw('EVALUATORQ_DIR') or None  # treat "" as unset
    root = Path(base) if base else Path.cwd() / STORE_DIR_NAME
    return root / subdir


def sanitise_run_name(name: str, *, default: str) -> str:
    """Lower-case ASCII slug of *name* for a report filename, at most 64 characters; *default* when empty."""
    sanitised = name.lower()
    sanitised = re.sub(r'[^a-z0-9_-]', '_', sanitised)
    sanitised = re.sub(r'_+', '_', sanitised)
    sanitised = sanitised.strip('_')
    return sanitised[:64] or default


def probe_writable_dir(folder: Path) -> None:
    """Create *folder* if missing and prove it is writable by creating and removing a file in it.

    A real write, not ``os.access``: permission bits miss read-only mounts, ACLs and full disks.
    Call it before a run spends anything, so a store that cannot be written fails up front.

    Raises:
        OSError: The folder could not be created or written to.
    """
    folder.mkdir(parents=True, exist_ok=True)
    descriptor, probe = tempfile.mkstemp(prefix='.evaluatorq-write-check-', suffix='.tmp', dir=folder)
    os.close(descriptor)
    Path(probe).unlink()


def write_unique_json(runs_dir: Path, *, base: str, payload: str) -> Path:
    """Write *payload* to ``<runs_dir>/<base>.json``, or the first free ``<base>_NNN.json``.

    Exclusive-create: report stamps resolve to the second, so two runs of one
    name in the same second would otherwise overwrite each other, and ``'x'``
    avoids the race an ``exists()`` check leaves open.

    Returns:
        The path written.

    Raises:
        OSError: The directory or file could not be written. A file that ``'x'``
            created before the write failed is removed rather than left empty.
        RuntimeError: No free name in 1000 attempts.
    """
    runs_dir.mkdir(parents=True, exist_ok=True)
    target = runs_dir / f'{base}.json'
    for counter in range(1000):
        try:
            with target.open('x', encoding='utf-8') as fh:
                _ = fh.write(payload)
        except FileExistsError:  # noqa: PERF203 — exclusive-create retry is the point
            target = runs_dir / f'{base}_{counter + 1:03d}.json'
        except OSError:
            target.unlink(missing_ok=True)
            raise
        else:
            return target
    raise RuntimeError(f'Could not find a free run-store filename for {base!r} after 1000 attempts')

"""Resolve a "previous run" reference to the JSON payload of a persisted run.

Both surfaces persist their runs under ``.evaluatorq/`` (``runs/`` for red
teaming, ``sim-runs/`` for agent simulation) and both want the same thing: take
a handle a user can actually type — the filename shown by ``eq redteam runs`` /
``eq sim runs``, a run id from a manifest, or just ``latest`` — and turn it into
the stored record so its cases can be replayed.

Resolution order for a reference, first match wins:

1. ``latest`` — the most recently modified report in the runs dir.
2. An existing file path (absolute or relative to the cwd).
3. ``<runs_dir>/<reference>`` and ``<runs_dir>/<reference>.json``.
4. A manifest ``run_id`` (exact, or an unambiguous prefix of at least 8 chars)
   whose manifest records a ``report_path``. An ambiguous id prefix is an error:
   ids are opaque, so there is no sensible way to pick one.
5. A report whose filename stem equals the reference, or whose stem starts with
   ``<reference>_`` (the ``<run-name>_<timestamp>`` convention) — newest wins.
   Names are *expected* to repeat across runs, so ambiguity here resolves to the
   most recent (logged) rather than raising.

Anything else raises `ReplayError` naming the runs dir and the most
recent runs, so a typo is one line away from being fixed.
"""

from __future__ import annotations

import json
import operator
from pathlib import Path
from typing import Any

from loguru import logger

MIN_RUN_ID_PREFIX = 8
"""Shortest run-id prefix accepted for resolution (uuid4 hex is 32 chars)."""

_SUGGEST_LIMIT = 5


REPLAY_VERSION_KEY = 'replay_version'
"""Key under which a saved run stamps the format its replay payload is written in."""

REPLAY_VERSION = 1
"""Format version this build writes and is able to read.

Without a marker, recognising an unreplayable run means sniffing for an absent
``datapoints`` key — a trick that only works for the one transition from
pre-replay runs. Stamping the version means the *next* format change can say
"saved by a newer version of evaluatorq" instead of failing structurally
somewhere downstream. Runs with no marker are pre-versioning and are read as
version 1, which is what they are.
"""


class ReplayError(RuntimeError):
    """A previous-run reference could not be resolved or replayed."""


def check_replay_version(payload: dict[str, Any], path: Path, *, surface: str) -> None:
    """Reject a run whose replay payload is newer than this build understands.

    Raises:
        ReplayError: The run declares a format version above `REPLAY_VERSION`.
    """
    stored = payload.get(REPLAY_VERSION_KEY)
    if isinstance(stored, int) and stored > REPLAY_VERSION:
        raise ReplayError(
            f'{path.name} was saved by a newer version of evaluatorq (replay format v{stored}; '
            f'this build reads v{REPLAY_VERSION}). Upgrade evaluatorq to replay this {surface} run.'
        )


def _reports_newest_first(runs_dir: Path) -> list[Path]:
    if not runs_dir.is_dir():
        return []
    reports: list[tuple[float, Path]] = []
    for p in runs_dir.glob('*.json'):
        try:
            reports.append((p.stat().st_mtime, p))
        except OSError:  # noqa: PERF203 — best-effort listing, a vanished file is skipped
            continue
    reports.sort(key=operator.itemgetter(0), reverse=True)
    return [p for _, p in reports]


def _resolve_via_manifest(reference: str, runs_dir: Path) -> Path | None:
    """Match *reference* against manifest run ids (exact, then unique prefix)."""
    from evaluatorq.common.run_manifest import list_manifests

    manifests = [(m.run_id, m.report_path) for m in list_manifests(runs_dir) if m.report_path]
    for run_id, report_path in manifests:
        if run_id == reference:
            return Path(report_path)
    if len(reference) < MIN_RUN_ID_PREFIX:
        return None
    matches = {report_path for run_id, report_path in manifests if run_id.startswith(reference)}
    if len(matches) == 1:
        return Path(next(iter(matches)))
    if len(matches) > 1:
        raise ReplayError(
            f'Run reference {reference!r} is ambiguous — it matches {len(matches)} runs. Use a longer id.'
        )
    return None


def _resolve_via_name(reference: str, runs_dir: Path) -> Path | None:
    """Match *reference* against report filename stems, newest first.

    Newest-wins rather than raising on ambiguity (unlike run ids): a run *name*
    is expected to repeat — ``--from-run nightly`` picking the latest nightly is
    the point. When it is ambiguous the chosen file is logged, so the pick is
    never silent even though it is not an error.
    """
    matches = [p for p in _reports_newest_first(runs_dir) if p.stem == reference or p.stem.startswith(f'{reference}_')]
    if not matches:
        return None
    if len(matches) > 1:
        logger.info(f'Run name {reference!r} matches {len(matches)} runs; using the most recent: {matches[0].name}')
    return matches[0]


def _not_found(reference: str, runs_dir: Path, surface: str) -> ReplayError:
    recent = [p.name for p in _reports_newest_first(runs_dir)[:_SUGGEST_LIMIT]]
    hint = f' Recent runs: {", ".join(recent)}.' if recent else f' No runs found in {runs_dir}.'
    return ReplayError(
        f'Could not resolve previous {surface} run {reference!r} in {runs_dir}. '
        f'Pass a run file name, a run id, a path to a saved run, or "latest".{hint}'
    )


def resolve_run_path(reference: str, runs_dir: Path, *, surface: str) -> Path:
    """Return the report file a previous-run *reference* points at.

    Args:
        reference: What the user typed — ``latest``, a path, a file name, a run
            name, or a manifest run id (full or an unambiguous 8+ char prefix).
        runs_dir: The surface's runs directory (e.g. ``.evaluatorq/runs``).
        surface: Human-readable surface name used in error messages.

    Raises:
        ReplayError: The reference is empty, ambiguous, or matches nothing.
    """
    ref = reference.strip()
    if not ref:
        raise ReplayError(f'Empty previous {surface} run reference.')

    if ref.lower() == 'latest':
        reports = _reports_newest_first(runs_dir)
        if not reports:
            raise ReplayError(f'No saved {surface} runs found in {runs_dir} — nothing to replay.')
        return reports[0]

    direct = Path(ref)
    if direct.is_file():
        return direct

    for candidate in (runs_dir / ref, runs_dir / f'{ref}.json'):
        if candidate.is_file():
            return candidate

    # A manifest can outlive the report it points at (deleted, moved). Check the
    # file exists before accepting it, or a stale manifest would mask a perfectly
    # good name match and report the run as unresolvable.
    for resolved in (_resolve_via_manifest(ref, runs_dir), _resolve_via_name(ref, runs_dir)):
        if resolved is not None and resolved.is_file():
            return resolved

    raise _not_found(ref, runs_dir, surface)


def load_run_payload(reference: str, runs_dir: Path, *, surface: str) -> tuple[dict[str, Any], Path]:
    """Resolve *reference* and read the stored run JSON as a dict."""
    path = resolve_run_path(reference, runs_dir, surface=surface)
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except OSError as exc:
        raise ReplayError(f'Could not read previous {surface} run {path}: {exc}') from exc
    except json.JSONDecodeError as exc:
        raise ReplayError(f'Previous {surface} run {path} is not valid JSON: {exc}') from exc
    if not isinstance(payload, dict):
        raise ReplayError(f'Previous {surface} run {path} does not contain a run object.')
    check_replay_version(payload, path, surface=surface)
    return payload, path

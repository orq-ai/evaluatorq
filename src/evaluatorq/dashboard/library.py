"""Discovery + identity + single-field kind sniff for the report dashboard."""

from __future__ import annotations

import base64
import functools
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from evaluatorq.contracts import RunManifest

_ARTIFACT_PREFIXES = ('01_', '02_', '03_')
_Model = TypeVar('_Model')


def report_id(path: Path) -> str:
    digest = hashlib.sha256(str(path.resolve()).encode()).digest()[:12]
    return base64.urlsafe_b64encode(digest).decode().rstrip('=')


@functools.lru_cache(maxsize=64)
def read_json(path_str: str, mtime_ns: int) -> dict[str, object]:
    """Parse and cache the JSON at *path_str*.

    The cache key includes *mtime_ns* (nanosecond modification time) so that
    any on-disk change automatically produces a cache miss — stale data is
    never served.  The *mtime_ns* argument is derived by the caller from
    ``Path.stat().st_mtime_ns`` and must not be fabricated.

    Args:
        path_str:  Absolute path string (``str(path.resolve())``).
        mtime_ns:  Nanosecond mtime of the file at the time of the call.

    Returns:
        Parsed JSON object as a ``dict``.

    Raises:
        json.JSONDecodeError: When the file content is not valid JSON.
        OSError: When the file cannot be read.
    """
    return json.loads(Path(path_str).read_text())  # type: ignore[return-value]


def read_json_cached(path: Path) -> dict[str, object]:
    """Read + parse JSON at *path*, using the mtime-keyed LRU cache."""
    mtime_ns = path.stat().st_mtime_ns
    return read_json(str(path.resolve()), mtime_ns)


@functools.lru_cache(maxsize=32)
def _validate_model(
    path_str: str,
    mtime_ns: int,
    validator: Callable[[dict[str, object]], _Model],
) -> _Model:
    """Cache a Pydantic-validated report model, mtime-keyed like ``read_json``.

    Raw-JSON caching alone still re-runs the (expensive) Pydantic validation of
    a whole run on every filter/sort/page request. Caching the validated object
    skips that — the payoff on large sim runs. ``validator`` is the model's
    ``model_validate`` classmethod (part of the key so two model types on the
    same path don't collide). Callers MUST treat the returned object as
    read-only; filtering builds new lists and never mutates it.
    """
    return validator(read_json(path_str, mtime_ns))


def load_model_cached(path: Path, validator: Callable[[dict[str, object]], _Model]) -> _Model:
    """Return a cached, validated report model for *path* (see ``_validate_model``)."""
    mtime_ns = path.stat().st_mtime_ns
    return _validate_model(str(path.resolve()), mtime_ns, validator)


def sniff_kind(data: dict[str, object]) -> str | None:
    """Surface from a single required-unique field. sim ('mode') checked first."""
    if 'mode' in data:
        return 'sim'
    if 'pipeline' in data:
        return 'redteam'
    if 'judging' in data:
        return 'pairwise'
    return None


def load_surface(path: Path) -> tuple[str | None, dict[str, object]]:
    try:
        data = read_json_cached(path)
    except (json.JSONDecodeError, OSError):
        return None, {}
    return sniff_kind(data), data


@dataclass(frozen=True)
class ReportCard:
    id: str
    surface: str
    name: str
    created_at: datetime
    headline: str
    # ``None`` for an in-flight (running/error/cancelled) run that has no report
    # on disk yet — such a card renders from the manifest's status/stage alone.
    path: Path | None
    error: str | None = None
    # Lifecycle status/stage from the run manifest. ``None`` for legacy
    # report-only cards (no manifest) so the view keeps its existing behavior.
    status: str | None = None
    stage: str | None = None


def _default_roots() -> list[Path]:
    from evaluatorq.pairwise_run import get_pairwise_runs_dir
    from evaluatorq.redteam.runner import get_runs_dir
    from evaluatorq.simulation.utils.run_store import get_sim_runs_dir

    return [get_runs_dir(), get_sim_runs_dir(), get_pairwise_runs_dir()]


def _iter_report_files(roots: list[Path]) -> Iterator[Path]:
    for root in roots:
        if not root.is_dir():
            continue
        for p in sorted(root.glob('*.json')):
            if not p.name.startswith(_ARTIFACT_PREFIXES):
                yield p


def _card(path: Path) -> ReportCard | None:
    surface, data = load_surface(path)
    if surface is None:
        return None
    created = data.get('created_at')
    try:
        created_at = datetime.fromisoformat(str(created)) if created else datetime.now(tz=timezone.utc)
    except (TypeError, ValueError):
        created_at = datetime.now(tz=timezone.utc)
    # Normalize naive timestamps (hand-edited / third-party JSON) to UTC so the
    # index sort never mixes naive and aware datetimes (TypeError on compare).
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    name = data.get('run_name') or data.get('description') or path.stem
    error = None
    if surface == 'redteam' and 'summary' not in data:
        error = 'missing required field: summary'
    elif surface == 'sim' and 'scorer_averages' not in data:
        error = 'missing required field: scorer_averages'
    elif surface == 'pairwise' and 'entries' not in data:
        error = 'missing required field: entries'
    if error:
        headline = ''
    elif surface == 'pairwise':
        entries = data.get('entries')
        count = len(entries) if isinstance(entries, list) else 0
        headline = f'{count} comparisons'
    else:
        headline = f'{data.get("total_results", 0)} {"attacks" if surface == "redteam" else "conversations"}'
    return ReportCard(report_id(path), surface, str(name), created_at, headline, path, error)


def _manifest_card_id(run_id: str) -> str:
    """Stable, URL-safe id for a manifest-backed card that has no report yet.

    Namespaced (``manifest:``) so it never collides with a ``report_id`` (which
    hashes a file path). Used only for in-flight cards; once a run completes and
    gains a ``report_path`` the card switches to ``report_id(path)`` so its
    identity matches the report and detail rendering is unchanged.
    """
    digest = hashlib.sha256(f'manifest:{run_id}'.encode()).digest()[:12]
    return base64.urlsafe_b64encode(digest).decode().rstrip('=')


def _card_from_manifest(m: RunManifest) -> ReportCard:
    """Build a list-row card from a ``RunManifest`` without reading its report.

    The headline for a completed run comes from the manifest's compact
    ``summary`` (never the report). In-flight runs (running/error/cancelled) have
    no summary, so the headline reflects their status/stage instead.
    """
    surface = m.surface.value
    created_at = m.started_at
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    path = Path(m.report_path) if m.report_path else None
    summary = m.summary or {}
    status = m.status.value
    if status == 'completed' and summary:
        noun = 'attacks' if surface == 'redteam' else 'conversations'
        headline = f'{summary.get("total_results", 0)} {noun}'
    elif status == 'running':
        headline = f'running · {m.stage}' if m.stage else 'running'
    else:  # error / cancelled with no report
        headline = status
    rid = report_id(path) if path is not None else _manifest_card_id(m.run_id)
    return ReportCard(
        rid,
        surface,
        m.run_name,
        created_at,
        headline,
        path,
        error=None,
        status=status,
        stage=m.stage,
    )


def scan(roots: list[Path] | None = None) -> list[ReportCard]:
    """Discover run cards, manifest-first with a legacy full-report fallback.

    Each ``.manifests/*.json`` sidecar becomes a card built without reading the
    full report (completed runs use their compact ``summary``; in-flight runs use
    status/stage). Report files already covered by a manifest's ``report_path``
    are de-duplicated out; the remaining (legacy, manifest-less) reports fall back
    to the full-report ``_card`` path. A runs dir with only legacy reports lists
    exactly as before.
    """
    from evaluatorq.common.run_manifest import list_manifests

    roots = roots or _default_roots()
    cards: list[ReportCard] = []
    covered: set[Path] = set()
    for root in roots:
        for m in list_manifests(root):
            card = _card_from_manifest(m)
            cards.append(card)
            if card.path is not None:
                covered.add(card.path.resolve())
    for p in _iter_report_files(roots):
        if p.resolve() in covered:
            continue
        if (c := _card(p)) is not None:
            cards.append(c)
    return sorted(cards, key=lambda c: c.created_at, reverse=True)


def resolve(rid: str, roots: list[Path] | None = None) -> Path | None:
    roots = roots or _default_roots()
    for p in _iter_report_files(roots):
        if report_id(p) == rid:
            return p
    logger.debug('report id not found after rescan: {}', rid)
    return None


def resolve_manifest(rid: str, roots: list[Path] | None = None) -> RunManifest | None:
    """Resolve a report-less card id back to its ``RunManifest``.

    Matches any manifest with no ``report_path``. That is usually an in-flight
    run (running/error/cancelled), but a *completed* run can also lack a
    ``report_path`` when its report save failed — such a run has no openable
    report, so it resolves here (not via :func:`resolve`) and the detail route
    renders its status page (status 'completed', no transcript). Returns the
    ``RunManifest`` so the detail route can render a minimal status page.
    """
    from evaluatorq.common.run_manifest import list_manifests

    roots = roots or _default_roots()
    for root in roots:
        for m in list_manifests(root):
            if m.report_path is None and _manifest_card_id(m.run_id) == rid:
                return m
    return None

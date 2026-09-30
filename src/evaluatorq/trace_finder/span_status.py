"""Shared span failure predicate for the drawer and lazy error-detail lookup."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def span_status_text(span: Any) -> str | None:
    """Return the most useful status, preferring failure indicators when fields conflict."""

    def field(value: Any, name: str) -> Any:
        return value.get(name) if isinstance(value, Mapping) else getattr(value, name, None)

    status = field(span, 'status')
    candidates: list[str] = []
    if isinstance(status, str):
        if status:
            candidates.append(status)
    elif status is not None:
        for key in ('code', 'status_code', 'state', 'name'):
            detail = field(status, key)
            if detail:
                candidates.append(str(detail))
                break
    top_level_code = field(span, 'status_code')
    if top_level_code:
        candidates.append(str(top_level_code))
    return next(
        (candidate for candidate in candidates if any(word in candidate.casefold() for word in ('error', 'failed'))),
        candidates[0] if candidates else None,
    )


def is_error_span(span: Any) -> bool:
    """Whether a rendered or fetched span carries an error/failed status."""
    status = span_status_text(span)
    return status is not None and any(word in status.casefold() for word in ('error', 'failed'))

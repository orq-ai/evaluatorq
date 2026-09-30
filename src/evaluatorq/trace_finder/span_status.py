"""Shared span failure predicate for the drawer and lazy error-detail lookup."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def is_error_span(span: Any) -> bool:
    """Whether a rendered or fetched span carries an error/failed status."""

    def field(value: Any, name: str) -> Any:
        return value.get(name) if isinstance(value, Mapping) else getattr(value, name, None)

    status = field(span, 'status')
    if isinstance(status, str):
        status_text = status
    else:
        status_text = ''
        if status is not None:
            for key in ('code', 'status_code', 'state', 'name'):
                detail = field(status, key)
                if detail:
                    status_text = str(detail)
                    break
        if not status_text:
            status_text = str(field(span, 'status_code') or '')
    lowered = status_text.casefold()
    return 'error' in lowered or 'failed' in lowered

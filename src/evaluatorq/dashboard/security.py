"""Shared request protection for dashboard state-changing forms."""

from __future__ import annotations

import secrets
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from starlette.requests import Request

CSRF_FIELD = 'csrf'
_CSRF_TOKEN = secrets.token_urlsafe(32)


def csrf_field() -> str:
    """Render the hidden token carried by every dashboard state-changing form."""

    return f'<input type="hidden" name="{CSRF_FIELD}" value="{_CSRF_TOKEN}">'


def request_rejected(req: Request, form: Any) -> str | None:
    """Return an error when a form is cross-origin or lacks the dashboard token."""

    sec_fetch = req.headers.get('sec-fetch-site', '')
    if sec_fetch and sec_fetch not in ('same-origin', 'none'):
        return 'Cross-origin request rejected.'
    if str(form.get(CSRF_FIELD) or '') != _CSRF_TOKEN:
        return 'Stale or missing form token; reload the page and try again.'
    return None

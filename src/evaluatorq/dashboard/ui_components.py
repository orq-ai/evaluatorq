"""Assets for the dashboard's reusable browser components."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path


def stylesheet_asset(*, filename: str) -> str:
    """Load a bundled stylesheet at a URL that changes with its contents."""
    path = Path(__file__).parent / 'static' / filename
    version = sha256(path.read_bytes()).hexdigest()[:12]
    return f'<link rel="stylesheet" href="/static/{filename}?v={version}">\n'


def component_assets() -> str:
    """Load the component styles and deferred script before a consumer's assets."""
    return stylesheet_asset(filename='ui-components.css') + '<script src="/static/ui-components.js" defer></script>\n'

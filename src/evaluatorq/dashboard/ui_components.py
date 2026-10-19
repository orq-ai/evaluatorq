"""Assets for the dashboard's reusable browser components."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path


def _asset_url(*, filename: str) -> str:
    """Version a bundled asset by its contents."""
    path = Path(__file__).parent / 'static' / filename
    version = sha256(path.read_bytes()).hexdigest()[:12]
    return f'/static/{filename}?v={version}'


def stylesheet_asset(*, filename: str) -> str:
    """Load a bundled stylesheet at a URL that changes with its contents."""
    return f'<link rel="stylesheet" href="{_asset_url(filename=filename)}">\n'


def script_asset(*, filename: str) -> str:
    """Load a deferred bundled script at a URL that changes with its contents."""
    return f'<script src="{_asset_url(filename=filename)}" defer></script>\n'


def component_assets() -> str:
    """Load the component styles and deferred script before a consumer's assets."""
    return stylesheet_asset(filename='ui-components.css') + script_asset(filename='ui-components.js')

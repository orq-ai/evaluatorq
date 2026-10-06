"""Assets for the dashboard's reusable browser components."""

from __future__ import annotations


def component_assets() -> str:
    """Load the component styles and deferred script before a consumer's assets."""
    return (
        '<link rel="stylesheet" href="/static/ui-components.css">\n'
        '<script src="/static/ui-components.js" defer></script>\n'
    )

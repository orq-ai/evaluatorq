"""Route-layer helpers for reading filter state from a Starlette Request.

This module is the single source of truth for the ``Request`` → selections
parse.  It lives in the route layer (it MAY import ``starlette``) but does NOT
import ``evaluatorq.dashboard.app`` or any *_views module, so there is no
import cycle:

    app.py ──imports──► filter_request.py  (no back-edge)
    redteam_views.py ──imports──► filter_request.py  (no back-edge)
    sim_views.py     ──imports──► filter_request.py  (no back-edge)

``filters.py`` is kept pure (no starlette import) — ``parse_selections`` here
is the only code that touches ``Request``.
"""

from __future__ import annotations

from starlette.requests import Request  # noqa: TC002 — FastHTML inspects annotations at runtime

from evaluatorq.dashboard.filters import FILTERS


def parse_selections(req: Request, surface: str) -> dict[str, list[str]]:
    """Parse filter selections for *surface* from the request query-string.

    Reads the dimension names registered in ``FILTERS[surface]`` and collects
    any matching query parameters into a ``dict[str, list[str]]``.

    Returns an empty dict when the surface is unknown or no filter params are
    present — callers treat an empty dict as "show all".

    Args:
        req:     The incoming Starlette ``Request``.
        surface: Dashboard surface key (``"redteam"`` or ``"sim"``).

    Returns:
        Selections mapping dimension key → list of selected values.
        Missing dimensions are absent from the dict (not present as empty lists).
    """
    filter_def = FILTERS.get(surface)
    if filter_def is None:
        return {}
    selections: dict[str, list[str]] = {}
    for dim in filter_def.dimensions:
        vals = req.query_params.getlist(dim)
        if vals:
            selections[dim] = vals
    return selections

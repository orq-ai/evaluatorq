"""Palette re-export for the simulation dashboard.

Canonical palette lives in ``common.reports.palette`` — imported here so the
dashboard never forks brand colors. Mirrors ``redteam.ui.colors``.
"""

from __future__ import annotations

from evaluatorq.common.reports.palette import (  # noqa: F401
    COLORS,
    ORQ_SCALE_AGENT,
    ORQ_SCALE_GOOD_BAD,
    ORQ_SCALE_HEAT,
    QUALITATIVE,
    SEVERITY_COLORS,
    SEVERITY_ORDER,
    STATUS_COLORS,
)

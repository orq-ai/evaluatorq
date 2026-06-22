"""Surface adapter registry for the evaluatorq dashboard.

Each surface (redteam, sim) gets a ``SurfaceAdapter`` dataclass that holds
callables for loading, rendering body HTML, exporting full HTML, extracting a
display name, and extracting a ``datetime`` from a parsed report object.

``ADAPTERS`` is the single registry keyed by surface kind strings matching
``evaluatorq.dashboard.library.sniff_kind`` return values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import datetime
    from pathlib import Path


@dataclass(frozen=True)
class SurfaceAdapter:
    """Encapsulates all surface-specific operations for the dashboard."""

    load: Callable[[Path], Any]
    body: Callable[[Any], str]
    export: Callable[[Any], str]
    name: Callable[[Any], str]
    created_at: Callable[[Any], datetime]


def _redteam_adapter() -> SurfaceAdapter:
    from evaluatorq.redteam.contracts import RedTeamReport
    from evaluatorq.redteam.reports.export_html import export_html, render_report_body

    return SurfaceAdapter(
        load=lambda p: RedTeamReport.model_validate_json(p.read_text()),
        body=render_report_body,
        export=export_html,
        name=lambda r: getattr(r, "description", None) or "Red team report",
        created_at=lambda r: r.created_at,
    )


def _sim_adapter() -> SurfaceAdapter:
    from evaluatorq.simulation.reports.export_html import export_html, render_report_body
    from evaluatorq.simulation.types import SimulationRun

    return SurfaceAdapter(
        load=lambda p: SimulationRun.model_validate_json(p.read_text()),
        body=lambda run: render_report_body(
            run.results,
            target=run.target_kind,
            run_date=run.created_at,
        ),
        export=lambda run: export_html(
            run.results,
            target=run.target_kind,
            run_date=run.created_at,
        ),
        name=lambda run: run.run_name,
        created_at=lambda run: run.created_at,
    )


ADAPTERS: dict[str, SurfaceAdapter] = {
    "redteam": _redteam_adapter(),
    "sim": _sim_adapter(),
}

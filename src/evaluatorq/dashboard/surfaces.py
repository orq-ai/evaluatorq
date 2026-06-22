"""Surface adapter registry for the evaluatorq dashboard.

Each surface (redteam, sim) gets a ``SurfaceAdapter`` dataclass that holds
callables for loading, rendering body HTML, exporting full HTML, extracting a
display name, and extracting a ``datetime`` from a parsed report object.

Optional export callables:

- ``export_markdown``: returns a Markdown string for the report, or ``None``
  if the surface has no Markdown export (sim only ever had JSON download).
- ``rows``: returns the serialisable row list used by CSV/JSON export.  Each
  element is a ``dict[str, Any]`` (redteam) or a plain dict (sim entries).

``ADAPTERS`` is the single registry keyed by surface kind strings matching
``evaluatorq.dashboard.library.sniff_kind`` return values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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

    # Optional: return Markdown string, or None when not supported.
    export_markdown: Callable[[Any], str] | None = field(default=None)

    # Optional: return a filtered list of serialisable dicts for CSV/JSON export.
    # Signature: (report_obj, filtered_results) -> list[dict[str, Any]]
    rows: Callable[[Any, list[Any]], list[dict[str, Any]]] | None = field(default=None)


def _redteam_adapter() -> SurfaceAdapter:
    from evaluatorq.redteam.contracts import RedTeamReport
    from evaluatorq.redteam.reports.export_html import export_html, render_report_body
    from evaluatorq.redteam.reports.export_md import export_markdown as _rt_export_md

    def _rt_rows(report: Any, filtered: list[Any]) -> list[dict[str, Any]]:
        """Build CSV/JSON row dicts from filtered RedTeamResult objects.

        Parity: redteam/ui/dashboard.py:1754-1771 (table_rows construction).
        Note: this is a deliberate 10-column summary shape (more consumable
        than full model_dump); the old Streamlit JSON used model_dump verbatim.
        """
        rows: list[dict[str, Any]] = []
        for r in filtered:
            dms = getattr(r.attack, 'delivery_methods', None)
            delivery_str = ', '.join(dm.value if hasattr(dm, 'value') else str(dm) for dm in dms) if dms else '-'
            domain_val = r.attack.vulnerability_domain.value if r.attack.vulnerability_domain else '-'
            rows.append({
                'ID': r.attack.id,
                'Category': r.attack.category,
                'Vulnerability': r.attack.vulnerability or '-',
                'Technique': r.attack.attack_technique.value,
                'Delivery Method': delivery_str,
                'Turn Type': r.attack.turn_type.value if r.attack.turn_type else '-',
                'Domain': domain_val,
                'Severity': r.attack.severity.value,
                'Result': 'VULNERABLE' if r.vulnerable else 'RESISTANT',
                'Source': r.attack.source,
            })
        return rows

    def _rt_load(p: Any) -> Any:
        from evaluatorq.dashboard.library import _read_json_cached
        data = _read_json_cached(p)
        return RedTeamReport.model_validate(data)

    return SurfaceAdapter(
        load=_rt_load,
        body=render_report_body,
        export=export_html,
        name=lambda r: getattr(r, 'description', None) or 'Red team report',
        created_at=lambda r: r.created_at,
        export_markdown=_rt_export_md,
        rows=_rt_rows,
    )


def _sim_adapter() -> SurfaceAdapter:
    from evaluatorq.simulation.reports.export_html import export_html, render_report_body
    from evaluatorq.simulation.types import SimulationRun

    def _sim_rows(run: Any, filtered: list[Any]) -> list[dict[str, Any]]:
        """Build JSON row dicts from filtered SimulationResult objects.

        Parity: simulation/ui/dashboard.py:322-334 (table dict inside
        ``_render_transcripts``).  This matches the shape used by the
        Streamlit JSON download button (dashboard.py:338).
        """
        from evaluatorq.simulation.reports.sections import build_report_sections

        # Build entries via the section layer so all fields (transcript,
        # criteria, evaluator_scores, judge_reason) are populated consistently.
        sections = build_report_sections(filtered)
        entries: list[dict[str, Any]] = []
        for s in sections:
            if s.kind == 'individual_results':
                entries = s.data.get('entries', [])
                break
        return entries

    def _sim_load(p: Any) -> Any:
        from evaluatorq.dashboard.library import _read_json_cached
        data = _read_json_cached(p)
        return SimulationRun.model_validate(data)

    return SurfaceAdapter(
        load=_sim_load,
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
        export_markdown=None,  # sim never had a markdown download — honest parity
        rows=_sim_rows,
    )


ADAPTERS: dict[str, SurfaceAdapter] = {
    'redteam': _redteam_adapter(),
    'sim': _sim_adapter(),
}

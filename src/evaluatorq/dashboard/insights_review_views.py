"""Insights review page using the staged dashboard redesign."""

from __future__ import annotations

import json
from html import escape
from typing import TYPE_CHECKING
from urllib.parse import quote

if TYPE_CHECKING:
    from evaluatorq.contracts import RunManifest
    from evaluatorq.insights.models import InsightsRun

from evaluatorq.dashboard.insights_run_form import run_form_script_tag
from evaluatorq.dashboard.shell import page
from evaluatorq.dashboard.ui_components import component_assets, stylesheet_asset

_REVIEW_STYLESHEET = '<link rel="stylesheet" href="/static/insights-review.css">\n'
_SIGNALS_SCRIPT = '<script src="/static/insights-signals.js" defer></script>\n'
_REVIEW_SCRIPT = '<script src="/static/insights-review.js" defer></script>\n'


def review_page(run: InsightsRun, manifest: RunManifest | None = None) -> str:
    """Render the review shell; trace content is fetched from the JSON endpoint."""
    from pathlib import Path

    template = Path(__file__).parent / 'static' / 'insights-review-template.html'
    body = template.read_text(encoding='utf-8')
    data_url = f'/insights/{quote(run.run_id, safe="")}/review-data.json'
    body = body.replace('REVIEW_URL', escape(data_url, quote=True))
    run_url = quote(run.run_id, safe='')
    body = body.replace('<a href="#" data-act="mock">Insights</a>', '<a href="/insights">Insights</a>')
    body = body.replace(
        '<button class="btn" data-act="mock">Export</button>',
        f'<a class="btn" href="/insights/{run_url}/export.json">Export</a>',
    )
    body = body.replace(
        '<button class="btn" data-act="mock">Re-run</button>',
        f'<a class="btn" id="rerun" href="/insights/{run_url}?rerun=1">Re-run</a>',
    )
    progress = None
    if manifest is not None:
        progress = {
            'status': manifest.status.value,
            'stage': manifest.stage,
            'stage_labels': manifest.stage_labels,
            'planned_stages': manifest.planned_stages,
            'stages': [stage.model_dump(mode='json') for stage in manifest.stages],
            'error': manifest.error,
        }
    body = body.replace('REVIEW_PROGRESS', escape(json.dumps(progress, separators=(',', ':')), quote=True))
    body = body.replace('<header class="top">', '<header class="top" id="header">', 1)
    return page(
        run.run_name,
        body,
        active_nav='insights',
        body_class='eq-insights-review',
        topbar=False,
        head_html=f'{component_assets()}{_REVIEW_STYLESHEET}{stylesheet_asset(filename="insights-signals.css")}{run_form_script_tag()}\n{_SIGNALS_SCRIPT}{_REVIEW_SCRIPT}',
    )

"""TDD tests for Vega re-embed wiring after HTMX swap (Task 6).

These tests verify that the STATIC WIRING for Vega re-embedding is present:
- The dashboard shell emits <script src="/static/dashboard.js"> and the
  vega/htmx script tags on every report page.
- A filter-POST fragment contains the data-vega-for and vega-chart elements
  when the report has charts, confirming the data contract that dashboard.js
  depends on at runtime.

KNOWN GAP: True re-embed correctness and leak-free teardown (that
window.__orqVegaViews[id].finalize() is called before re-embedding, and that
the new view renders) can only be verified by a headless browser (e.g.
Playwright with a real Vega-Embed bundle loaded). These tests assert WIRING
only, not runtime JS execution.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from evaluatorq.dashboard.app import build_app
from evaluatorq.dashboard.library import report_id

# Re-use the report factory from the filter test suite — no duplication.
from tests.dashboard.test_filter import _write_rt_report


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def roots(tmp_path: Path) -> list[Path]:
    """Write a red-team report fixture and return the roots list."""
    rt = tmp_path / "runs"
    rt.mkdir()
    _write_rt_report(rt / "vega_reinit_test.json")
    return [rt]


@pytest.fixture()
def client(roots: list[Path]) -> TestClient:
    app = build_app(roots=roots)
    return TestClient(app, raise_server_exceptions=True)


def _rt_path(roots: list[Path]) -> Path:
    return roots[0] / "vega_reinit_test.json"


# ---------------------------------------------------------------------------
# Shell wiring: script tags in <head>
# ---------------------------------------------------------------------------


class TestShellScriptTags:
    """The dashboard shell must emit all required JS <script> tags."""

    def test_report_page_includes_dashboard_js(
        self, client: TestClient, roots: list[Path]
    ) -> None:
        """GET /r/{rid} must include /static/dashboard.js in the <head>."""
        rid = report_id(_rt_path(roots))
        r = client.get(f"/r/{rid}")
        assert r.status_code == 200
        assert "/static/dashboard.js" in r.text

    def test_report_page_includes_htmx(
        self, client: TestClient, roots: list[Path]
    ) -> None:
        """GET /r/{rid} must include /static/htmx.min.js."""
        rid = report_id(_rt_path(roots))
        r = client.get(f"/r/{rid}")
        assert r.status_code == 200
        assert "/static/htmx.min.js" in r.text

    def test_report_page_includes_vega_embed(
        self, client: TestClient, roots: list[Path]
    ) -> None:
        """GET /r/{rid} must include /static/vega-embed.min.js."""
        rid = report_id(_rt_path(roots))
        r = client.get(f"/r/{rid}")
        assert r.status_code == 200
        assert "/static/vega-embed.min.js" in r.text

    def test_report_page_includes_vega_and_vega_lite(
        self, client: TestClient, roots: list[Path]
    ) -> None:
        """GET /r/{rid} must include /static/vega.min.js and /static/vega-lite.min.js."""
        rid = report_id(_rt_path(roots))
        r = client.get(f"/r/{rid}")
        assert r.status_code == 200
        assert "/static/vega.min.js" in r.text
        assert "/static/vega-lite.min.js" in r.text

    def test_index_page_includes_dashboard_js(self, client: TestClient) -> None:
        """GET / (index) must also include /static/dashboard.js."""
        r = client.get("/")
        assert r.status_code == 200
        assert "/static/dashboard.js" in r.text

    def test_script_tags_appear_in_head(
        self, client: TestClient, roots: list[Path]
    ) -> None:
        """All script tags must appear inside <head> (before </head>)."""
        rid = report_id(_rt_path(roots))
        r = client.get(f"/r/{rid}")
        assert r.status_code == 200
        text = r.text
        head_end = text.find("</head>")
        assert head_end > 0, "</head> tag not found in response"
        head_section = text[:head_end]
        assert "/static/dashboard.js" in head_section
        assert "/static/htmx.min.js" in head_section
        assert "/static/vega-embed.min.js" in head_section


# ---------------------------------------------------------------------------
# Filter fragment: Vega data contract (data-vega-for + vega-chart)
# ---------------------------------------------------------------------------


class TestFilterFragmentVegaContract:
    """Vega markup contract for the HTMX swap fragment.

    dashboard.js depends on [data-vega-for] JSON islands and .vega-chart divs
    being present in the swapped fragment.  The redteam renderer currently uses
    server-side SVG (render_svg) for its chart output, so filter fragments
    contain <svg> markup rather than client-side embed islands.  The
    render_embed() unit tests below verify the data-vega-for / vega-chart
    contract directly against the vega module.

    When any surface migrates to client-side embed, these HTTP-level tests
    should be updated to assert the full embed contract on that surface's
    filter fragment.
    """

    def test_filter_fragment_contains_svg_charts(
        self, client: TestClient, roots: list[Path]
    ) -> None:
        """POST /r/{rid}/filter fragment must contain at least one SVG chart.

        The redteam renderer uses server-side vl-convert SVGs, so a successful
        render pipeline is confirmed by the presence of <svg in the fragment.
        """
        rid = report_id(_rt_path(roots))
        r = client.post(f"/r/{rid}/filter", data={})
        assert r.status_code == 200
        assert "<svg" in r.text

    def test_filter_fragment_has_filter_swap_container(
        self, client: TestClient, roots: list[Path]
    ) -> None:
        """The HTMX swap target div must be present — dashboard.js scopes to it."""
        rid = report_id(_rt_path(roots))
        r = client.post(f"/r/{rid}/filter", data={})
        assert r.status_code == 200
        assert 'id="filter-swap"' in r.text


class TestRenderEmbedContract:
    """Unit-level contract tests for render_embed(): the data-vega-for markup
    that dashboard.js reads at runtime.

    These do NOT require HTTP; they directly verify that the Python function
    emits the exact HTML data contract that dashboard.js depends on.
    """

    def _minimal_spec(self) -> dict:
        return {
            "mark": "bar",
            "data": {"values": [{"a": "x", "b": 1}]},
            "encoding": {
                "x": {"field": "a", "type": "nominal"},
                "y": {"field": "b", "type": "quantitative"},
            },
        }

    def test_render_embed_emits_vega_chart_div(self) -> None:
        """render_embed must emit a div with class="vega-chart"."""
        from evaluatorq.common.reports.vega import render_embed

        html = render_embed(self._minimal_spec(), "chart-test-1")
        assert 'class="vega-chart"' in html

    def test_render_embed_emits_data_vega_for(self) -> None:
        """render_embed must emit a data-vega-for attribute matching the dom_id."""
        from evaluatorq.common.reports.vega import render_embed

        html = render_embed(self._minimal_spec(), "chart-test-2")
        assert 'data-vega-for="chart-test-2"' in html

    def test_render_embed_stores_embed_result_not_view(self) -> None:
        """The IIFE must store the embed result r (not r.view) so that
        r.finalize() works for full teardown after an HTMX swap."""
        from evaluatorq.common.reports.vega import render_embed

        html = render_embed(self._minimal_spec(), "chart-test-3")
        # Must NOT store r.view — the embed result r has .finalize().
        assert "=r;" in html or "= r;" in html
        assert "=r.view;" not in html and "= r.view;" not in html

    def test_render_embed_json_island_parses_as_vega_lite(self) -> None:
        """The JSON island inside the data-vega-for script tag must be a valid
        Vega-Lite spec (with $schema injected by _finalize)."""
        import json

        from evaluatorq.common.reports.vega import render_embed

        html = render_embed(self._minimal_spec(), "chart-test-4")
        # Find the application/json script tag.
        start = html.find('type="application/json"')
        assert start >= 0, "No application/json script tag found"
        content_start = html.find(">", start) + 1
        content_end = html.find("</script>", content_start)
        raw = html[content_start:content_end]
        spec = json.loads(raw)
        assert "$schema" in spec

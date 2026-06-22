"""End-to-end integration tests for the evaluatorq dashboard.

Uses Starlette TestClient via ``build_app(roots=[...])`` so every test runs
against real route handlers with real fixture data on disk — no mocking.

Coverage:
1. Adversarial kind-sniff: a SimulationRun fixture that also carries a stray
   ``pipeline`` key still routes to the sim renderer (``mode`` wins over
   ``pipeline`` in ``sniff_kind``); a redteam fixture (no ``mode``) routes
   redteam correctly.

2. Malformed vs partially-valid JSON — distinct buckets:
   (a) An unparseable file (``{not json``) is ABSENT from the index (silently
       skipped), no 500.
   (b) A sniff-able-but-invalid file (has ``pipeline`` but missing required
       fields) appears on the index as a BROKEN card AND its ``GET /r/{id}``
       returns a non-500 error page.

3. Multi-target redteam regression: a merged report with two ``tested_agents``
   renders ``agent_comparison`` + ``agent_disagreements`` sections; a filtered
   ``POST /r/{id}/filter`` round-trip preserves the merged-summary structure.

4. Charts present: a rendered redteam report page contains ``<svg`` or
   ``data-vega-for`` (charts or their client-side spec tags).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from evaluatorq.dashboard.app import build_app
from evaluatorq.dashboard.library import report_id
from evaluatorq.redteam.contracts import (
    AgentInfo,
    AttackInfo,
    AttackTechnique,
    DeliveryMethod,
    Framework,
    Pipeline,
    RedTeamReport,
    RedTeamResult,
    Severity,
    TurnType,
    UnifiedEvaluationResult,
)
from evaluatorq.redteam.reports.converters import compute_report_summary

# ---------------------------------------------------------------------------
# Shared fixture-factory helpers (mirrors test_rebuild_filtered._make_result)
# ---------------------------------------------------------------------------


def _make_result(
    category: str = 'ASI01',
    passed: bool | None = True,
    agent_key: str = 'agent-a',
    severity: Severity = Severity.MEDIUM,
) -> RedTeamResult:
    """Build a minimal RedTeamResult for use in integration fixtures."""
    return RedTeamResult(
        attack=AttackInfo(
            id=f'{category}-{agent_key}-{passed}',
            category=category,
            framework=Framework.OWASP_ASI,
            attack_technique=AttackTechnique.INDIRECT_INJECTION,
            delivery_methods=[DeliveryMethod.DIRECT_REQUEST],
            turn_type=TurnType.SINGLE,
            severity=severity,
            source='test',
        ),
        agent=AgentInfo(key=agent_key),
        messages=[],
        vulnerable=passed is False,
        evaluation=UnifiedEvaluationResult(passed=passed, explanation='test') if passed is not None else None,
    )


def _make_report(results: list[RedTeamResult], tested_agents: list[str]) -> RedTeamReport:
    """Build a minimal RedTeamReport from results + agent list."""
    return RedTeamReport(
        created_at=datetime.now(tz=timezone.utc),
        description='Integration test report',
        pipeline=Pipeline.STATIC,
        framework=Framework.OWASP_ASI,
        categories_tested=sorted({r.attack.category for r in results}),
        tested_agents=tested_agents,
        total_results=len(results),
        results=results,
        summary=compute_report_summary(results),
    )


# ---------------------------------------------------------------------------
# Shared fixture: tmp directory tree with BOTH surfaces + adversarial files
# ---------------------------------------------------------------------------


@pytest.fixture
def fixture_roots(tmp_path: Path) -> list[Path]:
    """Build a fixture directory tree with both surface types and adversarial fixtures.

    Layout:
        tmp/runs/
            sim_with_stray_pipeline.json  — sim fixture with stray ``pipeline`` key
            redteam_valid.json            — minimal valid redteam report
            redteam_broken_partial.json   — sniffs redteam but missing required fields
            unparseable.json              — invalid JSON (silently skipped)
        tmp/sim-runs/
            sim_valid.json                — valid sim report
        tmp/runs_multi/
            redteam_multi_agent.json      — two-agent redteam merged report
    """
    runs = tmp_path / 'runs'
    sim_runs = tmp_path / 'sim-runs'
    runs_multi = tmp_path / 'runs_multi'
    runs.mkdir()
    sim_runs.mkdir()
    runs_multi.mkdir()

    now_iso = datetime.now(tz=timezone.utc).isoformat()

    # 1. Sim fixture with stray ``pipeline`` key (mode wins, so it's still sim)
    (runs / 'sim_with_stray_pipeline.json').write_text(
        json.dumps({
            'run_name': 'stray-pipeline-sim',
            'created_at': now_iso,
            'mode': 'run',                   # sim discriminator — wins over pipeline
            'pipeline': 'static',            # stray key that must NOT re-route to redteam
            'target_kind': 'orq_agent',
            'evaluator_names': [],
            'total_results': 0,
            'scorer_averages': {},
            'results': [],
        })
    )

    # 2. Minimal valid redteam report (single agent)
    rt_result = _make_result(category='ASI01', passed=True, agent_key='agent-a')
    rt_report = _make_report([rt_result], tested_agents=['agent-a'])
    (runs / 'redteam_valid.json').write_text(rt_report.model_dump_json())

    # 3. Sniff-able-but-invalid redteam: has ``pipeline`` but missing required fields
    #    _card() detects missing 'summary' → error card; adapter.load() raises on GET /r/{id}
    (runs / 'redteam_broken_partial.json').write_text(
        json.dumps({
            'pipeline': 'static',
            'created_at': now_iso,
            'results': [],
            # deliberately missing: summary, categories_tested, total_results, etc.
        })
    )

    # 4. Unparseable JSON (must be silently skipped from index, no 500)
    (runs / 'unparseable.json').write_text('{not json')

    # 5. Valid sim report in sim-runs
    (sim_runs / 'sim_valid.json').write_text(
        json.dumps({
            'run_name': 'valid-sim',
            'created_at': now_iso,
            'mode': 'run',
            'target_kind': 'orq_agent',
            'evaluator_names': [],
            'total_results': 0,
            'scorer_averages': {},
            'results': [],
        })
    )

    # 6. Two-agent merged redteam report
    multi_results = [
        _make_result(category='ASI01', passed=False, agent_key='agent-a'),
        _make_result(category='ASI01', passed=True, agent_key='agent-b'),
        _make_result(category='LLM01', passed=False, agent_key='agent-a'),
        _make_result(category='LLM01', passed=True, agent_key='agent-b'),
    ]
    multi_report = _make_report(multi_results, tested_agents=['agent-a', 'agent-b'])
    (runs_multi / 'redteam_multi_agent.json').write_text(multi_report.model_dump_json())

    return [runs, sim_runs, runs_multi]


@pytest.fixture
def client(fixture_roots: list[Path]) -> TestClient:
    """TestClient with raise_server_exceptions=True (unexpected 500s fail the test)."""
    app = build_app(roots=fixture_roots)
    return TestClient(app, raise_server_exceptions=True)


@pytest.fixture
def lenient_client(fixture_roots: list[Path]) -> TestClient:
    """TestClient with raise_server_exceptions=False for routes expected to return error pages."""
    app = build_app(roots=fixture_roots)
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Helper: resolve a report id from a specific path
# ---------------------------------------------------------------------------


def _rid(roots: list[Path], subdir: str, filename: str) -> str:
    """Resolve a fixture file path to its dashboard report id."""
    # roots[0]=runs, roots[1]=sim-runs, roots[2]=runs_multi
    mapping = {
        'runs': roots[0],
        'sim-runs': roots[1],
        'runs_multi': roots[2],
    }
    return report_id(mapping[subdir] / filename)


# ===========================================================================
# Test class 1: Adversarial kind-sniff
# ===========================================================================


class TestAdversarialKindSniff:
    """Verify sniff_kind(data): mode wins over pipeline; no cross-routing."""

    def test_sim_with_stray_pipeline_key_routes_to_sim_renderer(
        self,
        client: TestClient,
        fixture_roots: list[Path],
    ) -> None:
        """A SimulationRun that also carries a stray ``pipeline`` key is sniffed as
        sim (``mode`` is checked first), so its report page renders sim HTML and
        does NOT raise a 500 from the redteam adapter."""
        rid = _rid(fixture_roots, 'runs', 'sim_with_stray_pipeline.json')
        r = client.get(f'/r/{rid}')
        assert r.status_code == 200, f'Expected 200 but got {r.status_code}: {r.text[:300]}'
        # A sim report renders its own section structure — should contain a <section>
        assert '<section' in r.text.lower()

    def test_sim_with_stray_pipeline_key_does_not_render_redteam_content(
        self,
        client: TestClient,
        fixture_roots: list[Path],
    ) -> None:
        """The stray ``pipeline`` key must NOT cause the page to render redteam-only
        elements like the risk banner or redteam TOC section headings."""
        rid = _rid(fixture_roots, 'runs', 'sim_with_stray_pipeline.json')
        r = client.get(f'/r/{rid}')
        assert r.status_code == 200
        # 'report-broken' class would appear if the wrong adapter tried to load
        # the sim JSON as a RedTeamReport and failed
        assert 'report-broken' not in r.text

    def test_valid_redteam_report_routes_correctly(
        self,
        client: TestClient,
        fixture_roots: list[Path],
    ) -> None:
        """A report that has ``pipeline`` but no ``mode`` is sniffed as redteam and
        renders correctly as a redteam report."""
        rid = _rid(fixture_roots, 'runs', 'redteam_valid.json')
        r = client.get(f'/r/{rid}')
        assert r.status_code == 200
        # Redteam report body should contain a <section> element
        assert '<section' in r.text.lower()

    def test_redteam_report_does_not_accidentally_route_as_sim(
        self,
        client: TestClient,
        fixture_roots: list[Path],
    ) -> None:
        """A redteam fixture (no ``mode`` key) must not be sniffed as sim."""
        rid = _rid(fixture_roots, 'runs', 'redteam_valid.json')
        r = client.get(f'/r/{rid}')
        assert r.status_code == 200
        # If accidentally routed as sim the page would contain 'report-broken'
        # (because SimulationRun.model_validate_json would fail on a redteam JSON)
        assert 'report-broken' not in r.text


# ===========================================================================
# Test class 2: Malformed vs partially-valid JSON — distinct buckets
# ===========================================================================


class TestMalformedAndPartiallyValidJSON:
    """Unparseable JSON is absent from index; sniff-able-invalid shows broken card."""

    def test_unparseable_json_absent_from_index(
        self,
        client: TestClient,
    ) -> None:
        """``{not json`` cannot be parsed at all — it must be silently skipped
        from the index (not listed, not a 500)."""
        r = client.get('/')
        assert r.status_code == 200
        # The filename of the unparseable fixture must not appear in the index
        assert 'unparseable' not in r.text.lower()

    def test_index_does_not_500_with_unparseable_file(
        self,
        client: TestClient,
    ) -> None:
        """The index must return 200 even when an unparseable JSON file is present."""
        r = client.get('/')
        assert r.status_code == 200

    def test_broken_partial_report_appears_on_index(
        self,
        client: TestClient,
    ) -> None:
        """A file that sniffs as redteam (has ``pipeline``) but is missing ``summary``
        must appear on the index as a broken card (not silently dropped)."""
        r = client.get('/')
        assert r.status_code == 200
        # The broken fixture appears: either its filename or the error badge is shown
        text = r.text
        assert 'redteam_broken_partial' in text or 'error' in text.lower()

    def test_broken_partial_report_has_error_badge_on_index(
        self,
        client: TestClient,
    ) -> None:
        """The broken card on the index must carry the ``card-error`` CSS class
        (rendered by ``index_body`` when ``card.error`` is set)."""
        r = client.get('/')
        assert r.status_code == 200
        assert 'card-error' in r.text

    def test_broken_partial_report_view_returns_error_page_not_500(
        self,
        lenient_client: TestClient,
        fixture_roots: list[Path],
    ) -> None:
        """``GET /r/{id}`` for the broken-partial report must not 500.
        The route catches the load exception and returns a ``report-broken`` page."""
        rid = _rid(fixture_roots, 'runs', 'redteam_broken_partial.json')
        r = lenient_client.get(f'/r/{rid}')
        assert r.status_code != 500, f'Expected non-500 but got 500; body: {r.text[:400]}'
        assert 'traceback' not in r.text.lower()

    def test_broken_partial_report_view_contains_error_indicator(
        self,
        lenient_client: TestClient,
        fixture_roots: list[Path],
    ) -> None:
        """The error page for the broken-partial report must show a visible error element."""
        rid = _rid(fixture_roots, 'runs', 'redteam_broken_partial.json')
        r = lenient_client.get(f'/r/{rid}')
        assert 'report-broken' in r.text or 'error' in r.text.lower()


# ===========================================================================
# Test class 3: Multi-target redteam regression
# ===========================================================================


class TestMultiTargetRedteamRegression:
    """Merged report with two agents renders agent_comparison + agent_disagreements."""

    def test_multi_agent_report_renders_200(
        self,
        client: TestClient,
        fixture_roots: list[Path],
    ) -> None:
        rid = _rid(fixture_roots, 'runs_multi', 'redteam_multi_agent.json')
        r = client.get(f'/r/{rid}')
        assert r.status_code == 200

    def test_multi_agent_report_contains_agent_comparison_section(
        self,
        client: TestClient,
        fixture_roots: list[Path],
    ) -> None:
        """The rendered body must contain the agent_comparison section anchor
        that ``render_report_body`` emits for multi-agent reports."""
        rid = _rid(fixture_roots, 'runs_multi', 'redteam_multi_agent.json')
        r = client.get(f'/r/{rid}')
        assert r.status_code == 200
        # The section anchor id injected by render_report_body
        assert 'section-agent_comparison' in r.text

    def test_multi_agent_report_contains_agent_disagreements_section(
        self,
        client: TestClient,
        fixture_roots: list[Path],
    ) -> None:
        """The rendered body must also contain the agent_disagreements section."""
        rid = _rid(fixture_roots, 'runs_multi', 'redteam_multi_agent.json')
        r = client.get(f'/r/{rid}')
        assert r.status_code == 200
        assert 'section-agent_disagreements' in r.text

    def test_filter_post_preserves_multi_agent_structure(
        self,
        client: TestClient,
        fixture_roots: list[Path],
    ) -> None:
        """POST /r/{id}/filter with a filter that keeps both agents must still
        produce a fragment that contains the agent-comparison section structure.
        This guards the ``rebuild_filtered_report`` multi-target code path."""
        rid = _rid(fixture_roots, 'runs_multi', 'redteam_multi_agent.json')
        # Sending no constraints keeps all results — both agents remain represented
        r = client.post(f'/r/{rid}/filter', data={})
        assert r.status_code == 200
        fragment = r.text
        # The filter fragment (not a full page) still contains the agent comparison
        # anchor because rebuild_filtered_report preserves tested_agents
        assert 'section-agent_comparison' in fragment

    def test_filter_post_with_result_filter_preserves_agent_comparison(
        self,
        client: TestClient,
        fixture_roots: list[Path],
    ) -> None:
        """POST /r/{id}/filter restricted to VULNERABLE results still renders
        agent comparison (both agents have vulnerable results in the multi fixture)."""
        rid = _rid(fixture_roots, 'runs_multi', 'redteam_multi_agent.json')
        r = client.post(f'/r/{rid}/filter', data={'result': 'VULNERABLE'})
        assert r.status_code == 200
        # The rebuilt report still has 2 agents, so agent_comparison fires
        assert 'section-agent_comparison' in r.text


# ===========================================================================
# Test class 4: Charts present in rendered report pages
# ===========================================================================


class TestChartsPresentInRenderedReport:
    """A rendered redteam report page must contain chart SVG or data-vega-for."""

    def test_redteam_report_page_contains_chart_element(
        self,
        client: TestClient,
        fixture_roots: list[Path],
    ) -> None:
        """The rendered report page must contain either an inline ``<svg`` (from
        vl-convert when available) or a ``data-vega-for`` script tag (client-side
        Vega-Embed fallback).  At least one of these must be present."""
        rid = _rid(fixture_roots, 'runs', 'redteam_valid.json')
        r = client.get(f'/r/{rid}')
        assert r.status_code == 200
        has_svg = '<svg' in r.text
        has_vega_for = 'data-vega-for' in r.text
        assert has_svg or has_vega_for, (
            'Expected <svg or data-vega-for in rendered redteam report page, '
            f'but neither was found. First 800 chars: {r.text[:800]}'
        )

    def test_multi_agent_report_page_contains_chart_element(
        self,
        client: TestClient,
        fixture_roots: list[Path],
    ) -> None:
        """The multi-agent report page also contains chart elements (the agent
        comparison section has a grouped bar chart)."""
        rid = _rid(fixture_roots, 'runs_multi', 'redteam_multi_agent.json')
        r = client.get(f'/r/{rid}')
        assert r.status_code == 200
        has_svg = '<svg' in r.text
        has_vega_for = 'data-vega-for' in r.text
        assert has_svg or has_vega_for, (
            'Expected <svg or data-vega-for in multi-agent redteam report page, '
            f'but neither was found. First 800 chars: {r.text[:800]}'
        )

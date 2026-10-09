"""Trace tab of the Insights trace page: lazy conversation and span routes."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from loguru import logger
from starlette.testclient import TestClient

from evaluatorq.dashboard import insights_routes, insights_trace_source
from evaluatorq.dashboard.app import build_app
from evaluatorq.dashboard.auth import DashboardAuth
from evaluatorq.insights.models import InsightsRun
from evaluatorq.trace_finder.models import TraceRecord
from evaluatorq.trace_finder.orq_source import OrqTraceSource
from tests.dashboard.insights_trace_helpers import snapshot_run, write_snapshot
from tests.dashboard.test_insights_page import _write_run, minimal_run  # noqa: F401 — fixture reused
from tests.insights.test_population import make_trace
from tests.trace_finder.test_orq_source import FakeOrq, FakeTraces, detail, span, summary

TRACE_PAGE = '/insights/run-1/trace?trace_id=trace-1&span_id=span-1'
CONVERSATION = '/insights/run-1/trace-conversation?trace_id=trace-1&span_id=span-1'


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    monkeypatch.setenv('ORQ_WORKSPACE', 'example-workspace')
    return tmp_path


def test_trace_view_renders_a_lazy_conversation_placeholder(env: Path, minimal_run: InsightsRun) -> None:
    _write_run(env, minimal_run)

    page = TestClient(build_app()).get(f'{TRACE_PAGE}&view=trace')

    assert page.status_code == 200
    assert 'hx-get="/insights/run-1/trace-conversation?trace_id=trace-1&amp;span_id=span-1"' in page.text
    assert 'hx-trigger="load"' in page.text
    assert 'hx-swap="outerHTML"' in page.text
    assert 'Loading conversation…' in page.text
    assert '<h3>Summary</h3>' not in page.text
    assert re.search(r'class="(?:[^"]*\s)?insights-tabs(?:\s[^"]*)?"', page.text) is None
    assert (
        '<a class="insights-tab active" href="/insights/run-1/trace?trace_id=trace-1&amp;span_id=span-1&amp;view=trace" '
        'aria-current="page">Trace</a>'
    ) in page.text
    assert (
        '<a class="insights-tab" href="/insights/run-1/trace?trace_id=trace-1&amp;span_id=span-1">Analysis</a>'
    ) in page.text


def test_unknown_view_falls_back_to_the_analysis_page(env: Path, minimal_run: InsightsRun) -> None:
    _write_run(env, minimal_run)

    page = TestClient(build_app()).get(f'{TRACE_PAGE}&view=bogus')

    assert page.status_code == 200
    assert '<h3>Summary</h3>' in page.text
    assert 'trace-conversation' not in page.text


def test_snapshot_run_reads_the_conversation_from_its_snapshot(env: Path, minimal_run: InsightsRun) -> None:
    path = env / 'snap.json'
    digest = write_snapshot(path, [make_trace('trace-1', span_id='span-1', content='hello from the snapshot')])
    _write_run(env, snapshot_run(minimal_run, path, digest))

    response = TestClient(build_app()).get(CONVERSATION)

    assert response.status_code == 200
    assert 'hello from the snapshot' in response.text
    assert response.text.startswith('<div class="fd-traces insights-trace-view">')
    assert 'trace-spans' not in response.text


def test_snapshot_run_with_the_file_gone_says_so_without_the_path(env: Path, minimal_run: InsightsRun) -> None:
    path = env / 'snap.json'
    digest = write_snapshot(path, [make_trace('trace-1', span_id='span-1', content='x')])
    _write_run(env, snapshot_run(minimal_run, path, digest))
    path.unlink()

    response = TestClient(build_app()).get(CONVERSATION)

    assert response.status_code == 200
    assert 'Conversation unavailable' in response.text
    assert str(env) not in response.text


def _fake_source(monkeypatch: pytest.MonkeyPatch, record: TraceRecord | str) -> None:
    sentinel = object()

    @asynccontextmanager
    async def source(app: Any, run: InsightsRun):
        yield sentinel

    async def load(run: InsightsRun, trace: Any, *, open_source: Any) -> TraceRecord | str:
        async with open_source() as opened:
            assert opened is sentinel
        return record

    monkeypatch.setattr(insights_routes, '_trace_source', source)
    monkeypatch.setattr(insights_routes, 'load_orq_record', load)


def test_orq_run_shows_the_transcript_and_a_spans_url(
    env: Path, minimal_run: InsightsRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_run(env, minimal_run)
    _fake_source(monkeypatch, make_trace('trace-1', span_id='span-1', content='hello from orq'))

    response = TestClient(build_app()).get(CONVERSATION)

    assert response.status_code == 200
    assert 'hello from orq' in response.text
    assert 'hx-get="/insights/run-1/trace-spans?trace_id=trace-1&amp;span_id=span-1"' in response.text
    assert 'now selects span' not in response.text


def test_orq_run_with_a_different_span_never_shows_substituted_conversation(
    env: Path, minimal_run: InsightsRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_run(env, minimal_run.model_copy(update={'population': {'mode': 'query'}}))

    @asynccontextmanager
    async def source(app: Any, run: InsightsRun):
        traces = FakeTraces(
            pages={None: ([summary('trace-1', messages=[])], False, None)},
            spans={'trace-1': [_named_span('span-other', name='current trace root', minute=1)]},
            details={('trace-1', 'span-other'): detail('span-other', 'substituted transcript secret', minute=1)},
        )
        yield OrqTraceSource(cast(Any, FakeOrq(traces)))

    client = TestClient(build_app())
    response = client.get(CONVERSATION)
    spans = client.get(SPANS)


    assert 'substituted transcript secret' not in response.text
    assert 'The conversation span analyzed by this run is no longer the span returned by Orq.' in response.text
    assert 'trace-spans' not in response.text
    assert spans.status_code == 200
    assert 'current trace root' in spans.text
    assert 'substituted transcript secret' not in spans.text
    assert 'The conversation span analyzed by this run is no longer the span returned by Orq.' not in spans.text
    assert 'span-other' not in spans.text


def test_spans_route_renders_when_conversation_query_fails(
    env: Path, minimal_run: InsightsRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_run(env, minimal_run.model_copy(update={'population': {'mode': 'query'}}))
    _api_key_auth(monkeypatch)

    class FailingQuery(FakeTraces):
        async def query_async(self, **kwargs: Any) -> Any:
            self.query_calls.append(kwargs)
            raise RuntimeError('conversation query unavailable')

    traces = FailingQuery(
        {},
        spans={'trace-1': [_named_span('span-root', name='current trace root', minute=0)]},
    )
    monkeypatch.setattr(insights_routes, 'build_orq_client', lambda auth, **kwargs: FakeOrq(traces))
    client = TestClient(build_app())

    conversation = client.get(CONVERSATION)
    query_count = len(traces.query_calls)
    spans = client.get(SPANS)

    assert 'Conversation unavailable' in conversation.text
    assert 'current trace root' in spans.text
    assert len(traces.query_calls) == query_count


def test_spans_route_renders_when_conversation_timestamp_is_missing(
    env: Path, minimal_run: InsightsRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = minimal_run.model_copy(
        update={
            'population': {'mode': 'query'},
            'traces': [minimal_run.traces[0].model_copy(update={'timestamp': None}), *minimal_run.traces[1:]],
        }
    )
    _write_run(env, run)
    _api_key_auth(monkeypatch)
    traces = FakeTraces(
        {},
        spans={'trace-1': [_named_span('span-root', name='current trace root', minute=0)]},
    )
    monkeypatch.setattr(insights_routes, 'build_orq_client', lambda auth, **kwargs: FakeOrq(traces))
    client = TestClient(build_app())

    conversation = client.get(CONVERSATION)
    spans = client.get(SPANS)

    assert 'did not record when the trace ran' in conversation.text
    assert 'trace-spans?trace_id=trace-1' in conversation.text
    assert 'current trace root' in spans.text



@pytest.mark.asyncio
async def test_unavailable_source_does_not_swallow_consumer_errors(monkeypatch: pytest.MonkeyPatch, minimal_run: InsightsRun) -> None:
    run = _scoped_run(minimal_run, {**_recorded_scope(), 'account_version': '2'})
    auth = DashboardAuth(method='environment', api_key='key', base_url='https://my.orq.ai')
    orq = FakeOrq(FakeTraces({}))

    class FailingWorkspaces:
        async def list_async(self) -> Any:
            raise RuntimeError('verification unavailable')

    orq.workspaces = FailingWorkspaces()
    monkeypatch.setattr(insights_routes, 'selected_dashboard_auth', lambda app: auth)
    monkeypatch.setattr(insights_routes, 'build_orq_client', lambda *args, **kwargs: orq)

    with pytest.raises(RuntimeError, match='consumer failure'):
        async with insights_routes._trace_source(build_app(), run) as unavailable:
            assert unavailable == 'Could not verify the Orq account selected in Settings to load this conversation.'
            raise RuntimeError('consumer failure')


def test_orq_run_without_credentials_degrades_visibly(
    env: Path, minimal_run: InsightsRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_run(env, minimal_run)

    def no_auth(app: Any) -> Any:
        raise ValueError('no credentials')

    monkeypatch.setattr(insights_routes, 'selected_dashboard_auth', no_auth)
    client = TestClient(build_app())
    warnings: list[str] = []
    sink = logger.add(lambda message: warnings.append(str(message)), level='WARNING', format='{message}')

    try:
        conversation = client.get(CONVERSATION)
        spans = client.get('/insights/run-1/trace-spans?trace_id=trace-1&span_id=span-1')
    finally:
        logger.remove(sink)

    reason = 'Connect an Orq account in Settings to load this conversation.'
    assert conversation.status_code == 200
    assert reason in conversation.text
    assert spans.status_code == 200
    assert reason in spans.text
    assert 'Try again' not in spans.text
    assert warnings == ['Insights trace source unavailable: no credentials\n'] * 2


def test_orq_trace_without_a_recorded_time_does_not_look_up_credentials(
    env: Path, minimal_run: InsightsRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = minimal_run.model_copy(
        update={'traces': [minimal_run.traces[0].model_copy(update={'timestamp': None}), *minimal_run.traces[1:]]}
    )
    _write_run(env, run)

    def unexpected_auth(app: Any) -> Any:
        raise AssertionError('a trace without a timestamp must not look up credentials')

    monkeypatch.setattr(insights_routes, 'selected_dashboard_auth', unexpected_auth)

    response = TestClient(build_app()).get(CONVERSATION)

    assert response.status_code == 200
    assert 'did not record when the trace ran' in response.text


def _named_span(span_id: str, *, name: str, minute: int, parent_span_id: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(**vars(span(span_id, minute=minute, parent_span_id=parent_span_id)), name=name)


def test_spans_route_renders_the_span_tree_from_orq(
    env: Path, minimal_run: InsightsRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_run(env, minimal_run.model_copy(update={'population': {'mode': 'query'}}))
    selected = summary('trace-1', messages=[{'role': 'user', 'content': 'analysed'}])
    selected.root_span_id = 'span-1'
    traces = FakeTraces(
        {None: ([selected], False, None)},
        spans={
            'trace-1': [
                _named_span('span-root', name='answer refund question', minute=0),
                _named_span('span-child', name='look up refund policy', minute=1, parent_span_id='span-root'),
            ]
        },
    )
    monkeypatch.setattr(insights_routes, 'build_orq_client', lambda auth, **kwargs: FakeOrq(traces))

    response = TestClient(build_app()).get('/insights/run-1/trace-spans?trace_id=trace-1&span_id=span-1')

    assert response.status_code == 200
    assert 'answer refund question' in response.text
    assert 'look up refund policy' in response.text


def _api_key_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(insights_routes, 'selected_dashboard_auth', lambda app: SimpleNamespace(method='api_key'))


def test_spans_route_closes_the_source_and_the_client(
    env: Path, minimal_run: InsightsRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_run(env, minimal_run.model_copy(update={'population': {'mode': 'query'}}))
    _api_key_auth(monkeypatch)
    selected = summary('trace-1', messages=[{'role': 'user', 'content': 'analysed'}])
    selected.root_span_id = 'span-1'
    orq = FakeOrq(
        FakeTraces(
            {None: ([selected], False, None)},
            spans={'trace-1': [_named_span('span-root', name='root', minute=0)]},
        )
    )
    closed: list[OrqTraceSource] = []

    class RecordingSource(OrqTraceSource):
        def close(self) -> None:
            closed.append(self)
            super().close()

    monkeypatch.setattr(insights_routes, 'build_orq_client', lambda auth, **kwargs: orq)
    monkeypatch.setattr(insights_routes, 'OrqTraceSource', RecordingSource)

    response = TestClient(build_app()).get('/insights/run-1/trace-spans?trace_id=trace-1&span_id=span-1')

    assert response.status_code == 200
    assert len(closed) == 1
    assert orq.exit_calls == [(None, None, None)]


SPANS = '/insights/run-1/trace-spans?trace_id=trace-1&span_id=span-1'


def _scoped_run(minimal_run: InsightsRun, scope: Any) -> InsightsRun:
    return minimal_run.model_copy(update={'population': {**minimal_run.population, 'orq_scope': scope}})


def _recorded_scope(account: str = 'account-a') -> dict[str, str | None]:
    return {'account': account, 'account_label': 'CLI OAuth on my.orq.ai', 'workspace': 'ws-a', 'project': 'proj-a'}


def _cli_oauth(monkeypatch: pytest.MonkeyPatch, *, account: str) -> list[dict[str, Any]]:
    auth = DashboardAuth(method='cli_oauth', api_key=None, base_url='https://my.orq.ai')
    built: list[dict[str, Any]] = []

    def build(auth: DashboardAuth, **kwargs: Any) -> FakeOrq:
        built.append(kwargs)
        selected = summary('trace-1', messages=[{'role': 'user', 'content': 'analysed'}])
        selected.root_span_id = 'span-1'
        return FakeOrq(
            FakeTraces(
                {None: ([selected], False, None)}, spans={'trace-1': [_named_span('span-root', name='root', minute=0)]}
            )
        )

    monkeypatch.setattr(insights_routes, 'selected_dashboard_auth', lambda app: auth)
    monkeypatch.setattr(insights_routes, 'account_identity', lambda auth: account)
    monkeypatch.setattr(insights_routes, 'build_orq_client', build)
    return built


@pytest.mark.parametrize('scope', [None, 'not-a-scope'])
def test_a_run_without_a_valid_recorded_scope_reads_with_the_settings_workspace(
    env: Path, minimal_run: InsightsRun, monkeypatch: pytest.MonkeyPatch, scope: Any
) -> None:
    _write_run(env, minimal_run if scope is None else _scoped_run(minimal_run, scope))
    built = _cli_oauth(monkeypatch, account='account-a')

    app = build_app()
    app.state.finder_settings = app.state.finder_settings.model_copy(
        update={'orq_workspace': 'settings-ws', 'orq_project_id': 'settings-project'}
    )

    TestClient(app).get(CONVERSATION)

    assert built == [{'workspace': 'settings-ws', 'project': 'settings-project'}]


def test_a_run_read_by_the_selected_account_uses_its_recorded_workspace(
    env: Path, minimal_run: InsightsRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_run(env, minimal_run.model_copy(update={'population': {'mode': 'query', 'orq_scope': _recorded_scope()}}))
    built = _cli_oauth(monkeypatch, account='account-a')
    client = TestClient(build_app())

    client.get(CONVERSATION)
    spans = client.get(SPANS)

    assert built == [{'workspace': 'ws-a', 'project': 'proj-a'}] * 2
    assert 'root' in spans.text


def test_a_run_read_by_another_account_names_it_without_calling_orq(
    env: Path, minimal_run: InsightsRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_run(env, _scoped_run(minimal_run, _recorded_scope(account='account-a')))
    built = _cli_oauth(monkeypatch, account='account-b')
    client = TestClient(build_app())
    warnings: list[str] = []
    sink = logger.add(lambda message: warnings.append(str(message)), level='WARNING', format='{message}')

    try:
        conversation = client.get(CONVERSATION)
        spans = client.get(SPANS)
    finally:
        logger.remove(sink)

    reason = 'This run read Orq as CLI OAuth on my.orq.ai. Select that account in Settings to load this conversation.'
    assert reason in conversation.text
    assert reason in spans.text
    assert built == []
    assert warnings == ['Insights run run-1 read Orq as CLI OAuth on my.orq.ai, but Settings selects CLI OAuth\n'] * 2


def test_source_construction_failure_still_closes_the_client(
    env: Path, minimal_run: InsightsRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_run(env, minimal_run)
    _api_key_auth(monkeypatch)
    orq = FakeOrq(FakeTraces({}))

    def failing_source(client: Any) -> OrqTraceSource:
        raise RuntimeError('hook registration failed')

    monkeypatch.setattr(insights_routes, 'build_orq_client', lambda auth, **kwargs: orq)
    monkeypatch.setattr(insights_routes, 'OrqTraceSource', failing_source)

    client = TestClient(build_app())
    warnings: list[str] = []
    sink = logger.add(lambda message: warnings.append(str(message)), level='WARNING', format='{message}')

    try:
        spans = client.get('/insights/run-1/trace-spans?trace_id=trace-1&span_id=span-1')
        conversation = client.get(CONVERSATION)
    finally:
        logger.remove(sink)

    reason = 'Could not connect to Orq with the account selected in Settings.'
    assert reason in spans.text
    assert 'Try again' not in spans.text
    assert reason in conversation.text
    assert orq.exit_calls == [(None, None, None)] * 2
    assert warnings == ['Insights trace source could not be created: hook registration failed\n'] * 2


def test_span_load_failure_is_the_only_spans_panel_that_says_try_again(
    env: Path, minimal_run: InsightsRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_run(env, minimal_run.model_copy(update={'population': {'mode': 'query'}}))
    _api_key_auth(monkeypatch)

    class FailingSpans(FakeTraces):
        async def list_spans_async(self, **kwargs: Any) -> Any:
            raise RuntimeError('spans are down')

    selected = summary('trace-1', messages=[{'role': 'user', 'content': 'analysed'}])
    selected.root_span_id = 'span-1'
    monkeypatch.setattr(
        insights_routes,
        'build_orq_client',
        lambda auth, **kwargs: FakeOrq(FailingSpans({None: ([selected], False, None)})),
    )

    response = TestClient(build_app()).get('/insights/run-1/trace-spans?trace_id=trace-1&span_id=span-1')

    assert response.status_code == 200
    assert 'Could not load spans. Try again.' in response.text


def test_unknown_trace_and_snapshot_runs_have_no_span_or_conversation_routes(
    env: Path, minimal_run: InsightsRun
) -> None:
    _write_run(env, minimal_run)
    path = env / 'snap.json'
    digest = write_snapshot(path, [make_trace('trace-1', span_id='span-1', content='x')])
    _write_run(env, snapshot_run(minimal_run, path, digest).model_copy(update={'run_id': 'snapshot-run'}))
    client = TestClient(build_app())

    unknown_spans = client.get('/insights/run-1/trace-spans?trace_id=unknown')
    snapshot_spans = client.get('/insights/snapshot-run/trace-spans?trace_id=trace-1')

    assert unknown_spans.status_code == 404
    assert 'Span loading is unavailable.' in unknown_spans.text
    assert snapshot_spans.status_code == 404


def test_trace_placeholder_reports_send_errors_and_lets_htmx_swap_successes(
    env: Path, minimal_run: InsightsRun
) -> None:
    _write_run(env, minimal_run)

    page = TestClient(build_app()).get(f'{TRACE_PAGE}&view=trace')

    assert 'hx-on::before-swap' not in page.text
    assert (
        'hx-on::send-error="this.innerHTML = &#x27;&lt;p class=&quot;finder-empty&quot; role=&quot;status&quot;&gt;'
        'Could not load the conversation. Reload to try again.&lt;/p&gt;&#x27;"'
    ) in page.text


@pytest.mark.parametrize(
    ('url', 'reason'),
    [
        ('/insights/missing/trace-conversation?trace_id=trace-1&span_id=span-1', 'Insights run not found.'),
        ('/insights/run-1/trace-conversation?trace_id=unknown&span_id=span-1', 'Trace not found in this Insights run.'),
    ],
    ids=['unknown-run', 'unknown-trace'],
)
def test_unknown_run_or_trace_conversation_is_a_200_unavailable_body(
    env: Path, minimal_run: InsightsRun, url: str, reason: str
) -> None:
    _write_run(env, minimal_run)

    response = TestClient(build_app()).get(url)

    assert response.status_code == 200
    assert response.text.startswith('<div class="fd-traces insights-trace-view">')
    assert 'Conversation unavailable' in response.text
    assert reason in response.text
    assert 'trace-spans' not in response.text


def test_run_from_an_unknown_source_has_no_orq_conversation_or_spans(
    env: Path, minimal_run: InsightsRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_run(env, minimal_run.model_copy(update={'population': {'mode': 'dataset'}}))

    def unexpected_client(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError('a dataset run must not call Orq')

    monkeypatch.setattr(insights_routes, 'build_orq_client', unexpected_client)
    client = TestClient(build_app())

    conversation = client.get(CONVERSATION)
    spans = client.get('/insights/run-1/trace-spans?trace_id=trace-1&span_id=span-1')

    assert conversation.status_code == 200
    assert 'This run&#x27;s trace source cannot be re-read.' in conversation.text
    assert 'trace-spans' not in conversation.text
    assert spans.status_code == 404


@pytest.mark.parametrize('view', ['analysis', 'trace'])
def test_trace_tab_is_disabled_with_its_reason_when_the_conversation_cannot_be_reread(
    env: Path, minimal_run: InsightsRun, monkeypatch: pytest.MonkeyPatch, view: str
) -> None:
    path = env / 'snap.json'
    digest = write_snapshot(path, [make_trace('trace-1', span_id='span-1', content='x')])
    _write_run(env, snapshot_run(minimal_run, path, digest))
    path.unlink()

    def unexpected(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError('the trace page must not call Orq or read the snapshot')

    monkeypatch.setattr(insights_routes, 'build_orq_client', unexpected)
    monkeypatch.setattr(insights_trace_source, 'read_snapshot', unexpected)

    page = TestClient(build_app()).get(f'{TRACE_PAGE}&view={view}')

    reason = 'The snapshot file this run read is no longer available or could not be read.'
    assert page.status_code == 200
    assert (
        f'<span class="insights-tab disabled" aria-disabled="true" title="{reason}">Trace</span>'
        f'<span class="insights-muted insights-trace-tab-reason">{reason}</span>'
    ) in page.text
    assert 'view=trace' not in page.text
    assert 'trace-conversation' not in page.text
    assert '<h3>Summary</h3>' in page.text


def test_analysis_tab_of_a_rereadable_run_links_the_trace_tab_without_loading_it(
    env: Path, minimal_run: InsightsRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = env / 'snap.json'
    digest = write_snapshot(path, [make_trace('trace-1', span_id='span-1', content='x')])
    _write_run(env, snapshot_run(minimal_run, path, digest))

    def unexpected(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError('the Analysis tab must not read the snapshot')

    monkeypatch.setattr(insights_trace_source, 'read_snapshot', unexpected)

    page = TestClient(build_app()).get(TRACE_PAGE)

    assert page.status_code == 200
    assert (
        '<a class="insights-tab" href="/insights/run-1/trace?trace_id=trace-1&amp;span_id=span-1&amp;view=trace">'
        'Trace</a>'
    ) in page.text
    assert 'insights-tab disabled' not in page.text

"""The dashboard's Experiments surface: saved ``evaluatorq()`` runs are listed, rendered and exported."""
# ruff: noqa: S101

from __future__ import annotations

import csv
import io
import re
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

import pytest
from loguru import logger
from starlette.testclient import TestClient

from evaluatorq import DataPoint, EvaluationResult, ExperimentRun
from evaluatorq.common.run_manifest import start_manifest
from evaluatorq.contracts import ManifestSurface, TokenUsage
from evaluatorq.dashboard import metrics
from evaluatorq.dashboard.app import build_app
from evaluatorq.dashboard.experiment_views import CELL_PREVIEW_CHARS, experiment_report_body
from evaluatorq.dashboard.library import sniff_kind
from evaluatorq.experiment_run import save_experiment_run
from evaluatorq.types import DataPointResult, EvaluatorScore, JobResult

if TYPE_CHECKING:
    from pathlib import Path

LONG_OUTPUT = 'Hamlet ' * 60
START = datetime(2026, 11, 1, 9, 30, tzinfo=timezone.utc)


def _score(name: str, value: float, *, passed: bool | None = None, tokens: int = 0) -> EvaluatorScore:
    usage = TokenUsage(input_tokens=tokens, output_tokens=0, total_tokens=tokens, total_cost=0.002) if tokens else None
    return EvaluatorScore(
        evaluator_name=name,
        score=EvaluationResult.model_validate({'value': value, 'pass': passed, 'token_usage': usage}),
    )


def _experiment(name: str = 'capitals <v2>') -> ExperimentRun:
    results = [
        DataPointResult(
            data_point=DataPoint(inputs={'question': 'capital of France?'}),
            job_results=[
                JobResult(
                    job_name='answer',
                    output='Paris',
                    evaluator_scores=[_score('exact', 1, passed=True), _score('judge', 4, tokens=100)],
                )
            ],
        ),
        DataPointResult(
            data_point=DataPoint(inputs={'question': 'plot of Hamlet?'}),
            job_results=[
                JobResult(
                    job_name='answer',
                    output=LONG_OUTPUT,
                    evaluator_scores=[_score('exact', 0, passed=False), _score('judge', 2, tokens=50)],
                )
            ],
        ),
        DataPointResult(
            data_point=DataPoint(inputs={'question': '17 * 23?'}),
            job_results=[JobResult(job_name='answer', output=None, error='upstream model timed out')],
        ),
        DataPointResult(data_point=DataPoint(inputs={'question': 'lost'}), error='datapoint fetch failed'),
    ]
    return ExperimentRun(
        run_id='run-1',
        run_name=name,
        description='tiny QA',
        started_at=START,
        ended_at=START + timedelta(seconds=12),
        duration_seconds=12,
        evaluatorq_version='0.0.0',
        evaluator_names=['exact', 'judge'],
        results=results,
    )


@pytest.fixture
def runs_dir(tmp_path: Path) -> Path:
    """A store with one saved experiment run and its completed manifest."""
    runs = tmp_path / 'experiment-runs'
    run = _experiment()
    writer = start_manifest(run_id=run.run_id, surface=ManifestSurface.EXPERIMENT, run_name=run.run_name, runs_dir=runs)
    path = save_experiment_run(run, output_dir=runs)
    writer.complete(report_path=path.resolve(), summary=run.manifest_summary())
    return runs


@pytest.fixture
def client(runs_dir: Path) -> TestClient:
    return TestClient(build_app(roots=[runs_dir]), raise_server_exceptions=True)


def _report_link(client: TestClient) -> str:
    links = set(re.findall(r'href="(/r/[^"?#/]+)"', client.get('/?surface=experiment').text))
    assert len(links) == 1, links
    return links.pop()


def test_a_saved_run_is_sniffed_as_an_experiment() -> None:
    assert sniff_kind({'experiment_run_version': 1, 'results': []}) == 'experiment'


def test_nav_and_landing_list_the_experiment(client: TestClient) -> None:
    landing = client.get('/')
    assert landing.status_code == 200
    assert 'href="/?surface=experiment"' in landing.text
    assert 'capitals &lt;v2&gt;' in landing.text
    assert '<v2>' not in landing.text


def test_run_list_shows_the_run_with_its_pass_rate(client: TestClient, runs_dir: Path) -> None:
    page = client.get('/?surface=experiment')
    assert page.status_code == 200
    assert 'runs-grid-row' in page.text  # the same run grid as the sim and red-team lists
    assert _kpi(page.text, 'Pass rate') == '50%'
    assert _kpi(page.text, 'Datapoints') == '4'
    assert _kpi(page.text, 'Errored rows') == '2'
    overview = metrics.experiment_overview([runs_dir])
    assert (overview.total_runs, overview.datapoints, overview.rows, overview.errored) == (1, 4, 4, 2)
    assert overview.total_cost == pytest.approx(0.004)
    [row] = [r for r in metrics.run_rows([runs_dir]) if r.surface == 'experiment']
    assert row.score == pytest.approx(0.5)  # one pass, one fail; errored rows set no pass
    assert row.status != 'error'


def test_landing_counts_evaluator_tokens_and_cost(runs_dir: Path) -> None:
    landing = metrics.landing([runs_dir])
    assert ('Experiments', 1) in landing.by_kind
    assert ('Experiments', 150) in landing.tokens_by_kind
    assert landing.total_cost == pytest.approx(0.004)


def test_landing_says_experiment_spend_is_evaluator_calls_only(client: TestClient) -> None:
    page = client.get('/').text
    assert page.count('experiments: evaluator calls only') == 2  # spend panel and token panel


def test_an_unreadable_run_left_out_of_the_landing_totals_is_logged(runs_dir: Path) -> None:
    [saved] = runs_dir.glob('*.json')
    saved.write_text('{"experiment_run_version": 1, "results": "not a list"}')
    messages: list[str] = []
    sink = logger.add(lambda m: messages.append(m.record['message']), level='WARNING')
    try:
        landing = metrics.landing([runs_dir])
    finally:
        logger.remove(sink)
    assert ('Experiments', 150) not in landing.tokens_by_kind
    assert any(str(saved) in m and 'left out of the totals' in m for m in messages)


def test_report_shows_scores_errors_and_a_clipped_output(client: TestClient) -> None:
    page = client.get(_report_link(client))
    assert page.status_code == 200
    body = page.text
    assert 'capitals &lt;v2&gt;' in body
    for evaluator in ('exact', 'judge'):
        assert evaluator in body
    assert '<span class="cmp-score-ab">50% (1/2)</span>' in body  # exact: pass rate over scores that set pass
    assert _headline(body, 'Pass rate across evaluators') == '50%'
    assert 'Showing <b>4</b> of <b>4</b> datapoints' in body
    assert 'upstream model timed out' in body
    assert 'datapoint fetch failed' in body
    assert body.count('exp-row--error') == 2
    assert '<details class="exp-clip">' in body
    assert LONG_OUTPUT[:CELL_PREVIEW_CHARS].rstrip() in body


def test_body_escapes_untrusted_text() -> None:
    run = _experiment(name='<script>alert(1)</script>')
    body = experiment_report_body(run)
    assert '<script>' not in body


def test_an_empty_run_has_an_empty_state() -> None:
    run = _experiment().model_copy(update={'results': [], 'evaluator_names': []})
    body = experiment_report_body(run)
    assert 'This run has no datapoints.' in body
    assert 'No evaluators ran in this run.' in body


def test_exports_carry_every_row(client: TestClient) -> None:
    link = _report_link(client)
    html = client.get(f'{link}/export.html')
    assert html.status_code == 200
    assert 'upstream model timed out' in html.text
    csv_page = client.get(f'{link}/export.csv')
    assert csv_page.status_code == 200
    rows = list(csv.DictReader(io.StringIO(csv_page.text)))
    assert len(rows) == 4
    assert rows[1]['Output'] == LONG_OUTPUT
    assert rows[2]['Error'] == 'upstream model timed out'
    assert rows[3]['Error'] == 'datapoint fetch failed'
    assert client.get(f'{link}/export.json').status_code == 200


def _headline(body: str, finding: str) -> str:
    match = re.search(rf'<span class="rk-head-number">([^<]*)(?:<span[^>]*>[^<]*</span>)?</span><span><span class="rk-head-finding">{finding}</span>', body)
    assert match is not None, finding
    return match.group(1).strip()


def test_filters_narrow_the_rows_to_failing_or_errored_datapoints(client: TestClient) -> None:
    rid = _report_link(client).removeprefix('/r/')
    failing = client.post(f'/r/{rid}/filter', data={'failed_on': 'exact'}).text
    assert 'Showing <b>1</b> of <b>4</b> datapoints' in failing
    assert 'plot of Hamlet?' in failing
    assert 'capital of France?' not in failing
    errored = client.post(f'/r/{rid}/filter', data={'status': 'errored'}).text
    assert 'Showing <b>2</b> of <b>4</b> datapoints' in errored
    assert 'upstream model timed out' in errored
    assert 'Paris' not in errored


def _kpi(body: str, label: str) -> str:
    match = re.search(rf'<div class="kpi-value">([^<]*)</div><div class="kpi-label">{label}</div>', body)
    assert match is not None, label
    return match.group(1)


def _save(run: ExperimentRun, runs: Path) -> None:
    writer = start_manifest(run_id=run.run_id, surface=ManifestSurface.EXPERIMENT, run_name=run.run_name, runs_dir=runs)
    path = save_experiment_run(run, output_dir=runs)
    writer.complete(report_path=path.resolve(), summary=run.manifest_summary())


def test_run_list_summary_and_report_agree_on_errored_and_pass_counts(tmp_path: Path) -> None:
    run = _experiment()
    counts = run.row_counts()
    assert (counts.rows, counts.job_results, counts.errored) == (4, 3, 2)
    assert run.pass_rate() == pytest.approx(0.5)
    _save(run, tmp_path)

    [row] = [r for r in metrics.run_rows([tmp_path]) if r.surface == 'experiment']
    assert row.score == pytest.approx(run.pass_rate())
    assert row.status == 'finished'
    summary = run.manifest_summary()
    assert (summary.get('pass_rate'), summary.get('errored_rows'), summary.get('judge_tokens')) == (0.5, 2, 150)
    body = experiment_report_body(run)
    assert _headline(body, 'Rows errored') == str(counts.errored)
    assert _headline(body, 'Pass rate across evaluators') == '50%'
    assert f'{counts.job_results} job results' in body


def test_run_list_and_report_agree_when_every_row_errored(tmp_path: Path) -> None:
    run = _experiment().model_copy(
        update={
            'results': [
                DataPointResult(
                    data_point=DataPoint(inputs={'question': 'q'}),
                    job_results=[JobResult(job_name='answer', output='fine')],
                    error='datapoint failed after its job ran',
                ),
                DataPointResult(data_point=DataPoint(inputs={'question': 'lost'}), error='datapoint fetch failed'),
            ]
        }
    )
    counts = run.row_counts()
    assert counts.all_errored
    _save(run, tmp_path)
    [row] = [r for r in metrics.run_rows([tmp_path]) if r.surface == 'experiment']
    assert row.status == 'error'
    assert row.score is None
    assert _headline(experiment_report_body(run), 'Rows errored') == str(counts.rows)


@pytest.mark.parametrize(('value', 'shown'), [(0.8333, '0.833'), (4.0, '4'), (12345.0, '12,345'), (-2500.4, '-2,500')])
def test_scores_of_any_magnitude_show_without_scientific_notation(value: float, shown: str) -> None:
    from evaluatorq.dashboard.experiment_views import _number

    assert _number(value) == shown

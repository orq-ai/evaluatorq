"""Saving ``evaluatorq()`` experiment runs locally, and loading them back."""
# ruff: noqa: S101

from __future__ import annotations

import asyncio
import importlib
import json
import os
from typing import TYPE_CHECKING, Any, cast

import pytest
from loguru import logger

from evaluatorq import DataPoint, EvaluationResult, SaveMode, evaluatorq, load_experiment_run
from evaluatorq.contracts import AgentResponse, TokenUsage
from evaluatorq.job_helper import job
from evaluatorq.send_results import OrqResponse

if TYPE_CHECKING:
    from pathlib import Path

    from evaluatorq.common.run_manifest import ManifestWriter
    from evaluatorq.types import DataPointResult, ScorerParameter


DATA = [DataPoint(inputs={'question': 'capital of France?'}), DataPoint(inputs={'question': 'boom'})]


async def answer(dp: DataPoint, _row: int) -> str:
    if dp.inputs['question'] == 'boom':
        raise RuntimeError('target exploded')
    return 'Paris'


async def judge(params: ScorerParameter) -> EvaluationResult:
    ok = 'Paris' in str(params['output'])
    return EvaluationResult.model_validate({
        'value': 1 if ok else 0,
        'pass': ok,
        'explanation': 'mentions Paris',
        'token_usage': TokenUsage(input_tokens=100, output_tokens=20, total_tokens=120, total_cost=0.001),
    })


async def run(
    *,
    save: SaveMode | str = SaveMode.NONE,
    output_dir: Path | str | None = None,
    upload: bool = False,
    url_out: list[str] | None = None,
) -> list[DataPointResult]:
    return await evaluatorq(
        'capitals',
        description='tiny QA',
        data=DATA,
        jobs=[job('answer', answer)],
        evaluators=[{'name': 'mentions-paris', 'scorer': judge}],
        print_results=False,
        save=cast('SaveMode', save),
        output_dir=output_dir,
        _send_results=upload,
        _orq_api_key='test-key' if upload else None,
        _experiment_url_out=url_out,
    )


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point the default store at a temp dir, so a default-path save lands there."""
    base = tmp_path / '.evaluatorq'
    monkeypatch.setenv('EVALUATORQ_DIR', str(base))
    return base


def _manifests(runs_dir: Path) -> list[dict[str, object]]:
    return [json.loads(p.read_text()) for p in sorted((runs_dir / '.manifests').glob('*.json'))]


@pytest.mark.asyncio
async def test_default_writes_nothing(store: Path) -> None:
    await run()
    assert not store.exists()


@pytest.mark.asyncio
async def test_save_none_writes_nothing(store: Path) -> None:
    await run(save=SaveMode.NONE)
    assert not store.exists()


@pytest.mark.asyncio
async def test_final_writes_run_and_completed_manifest(store: Path) -> None:
    results = await run(save=SaveMode.FINAL)
    runs_dir = store / 'experiment-runs'
    files = list(runs_dir.glob('capitals_*.json'))
    assert len(files) == 1
    [manifest] = _manifests(runs_dir)
    assert manifest['surface'] == 'experiment'
    assert manifest['status'] == 'completed'
    assert manifest['report_path'] == str(files[0].resolve())
    assert manifest['summary'] == {
        'total_results': 2,
        'scorer_averages': {'mentions-paris': 1.0},
        'pass_rate': 1.0,
        'errored_rows': 1,
        'judge_tokens': 120,
    }

    loaded = load_experiment_run(files[0])
    assert loaded.run_name == 'capitals'
    assert loaded.description == 'tiny QA'
    assert loaded.evaluator_names == ['mentions-paris']
    assert loaded.ended_at >= loaded.started_at
    assert loaded.created_at >= loaded.ended_at
    assert loaded.experiment_url is None
    assert loaded.results == results


@pytest.mark.asyncio
async def test_save_accepts_the_string_value(store: Path) -> None:
    await run(save='final')
    assert len(list((store / 'experiment-runs').glob('*.json'))) == 1


@pytest.mark.asyncio
async def test_bool_save_still_works_and_warns_at_the_callers_line(store: Path) -> None:
    with pytest.warns(DeprecationWarning, match='save=True/False is deprecated') as recorded:
        await evaluatorq('capitals', data=DATA, jobs=[job('answer', answer)], print_results=False, save=cast('Any', False))
    assert not store.exists()
    assert recorded[0].filename == __file__
    with pytest.warns(DeprecationWarning, match='save=True/False is deprecated'):
        await evaluatorq('capitals', data=DATA, jobs=[job('answer', answer)], print_results=False, save=cast('Any', True))
    assert len(list((store / 'experiment-runs').glob('capitals_*.json'))) == 1


@pytest.mark.asyncio
async def test_a_job_agent_response_usage_is_kept_in_the_saved_output(tmp_path: Path) -> None:
    usage = TokenUsage(input_tokens=7, output_tokens=3, total_tokens=10, total_cost=0.0005)

    async def agent(_dp: DataPoint, _row: int) -> AgentResponse:
        return AgentResponse(usage=usage)

    await evaluatorq(
        'capitals',
        data=DATA[:1],
        jobs=[job('agent', agent)],
        print_results=False,
        save=SaveMode.FINAL,
        output_dir=tmp_path,
        _send_results=False,
    )
    saved = json.loads(next(tmp_path.glob('capitals_*.json')).read_text())
    assert saved['results'][0]['job_results'][0]['output']['usage']['total_tokens'] == 10


@pytest.mark.asyncio
async def test_output_dir_overrides_the_store(store: Path, tmp_path: Path) -> None:
    out = tmp_path / 'elsewhere'
    await run(save=SaveMode.FINAL, output_dir=out)
    assert len(list(out.glob('capitals_*.json'))) == 1
    assert _manifests(out)[0]['status'] == 'completed'
    assert not store.exists()


@pytest.mark.asyncio
async def test_two_saves_in_the_same_second_do_not_overwrite(tmp_path: Path) -> None:
    await run(save=SaveMode.FINAL, output_dir=tmp_path)
    await run(save=SaveMode.FINAL, output_dir=tmp_path)
    assert len(list(tmp_path.glob('capitals_*.json'))) == 2


@pytest.mark.asyncio
async def test_round_trip_keeps_errors_and_judge_token_usage(tmp_path: Path) -> None:
    await run(save=SaveMode.FINAL, output_dir=tmp_path)
    loaded = load_experiment_run(next(tmp_path.glob('capitals_*.json')))
    ok, failed = loaded.results
    assert failed.job_results is not None
    assert failed.job_results[0].error is not None
    assert 'target exploded' in failed.job_results[0].error
    assert ok.job_results is not None
    assert ok.job_results[0].evaluator_scores is not None
    score = ok.job_results[0].evaluator_scores[0].score
    assert score.pass_ is True
    assert score.token_usage is not None
    assert score.token_usage.total_tokens == 120
    assert score.token_usage.total_cost == pytest.approx(0.001)

    [stats] = loaded.evaluator_stats()
    assert (stats.name, stats.scored, stats.average, stats.pass_rate) == ('mentions-paris', 1, 1.0, 1.0)


@pytest.mark.asyncio
async def test_detail_is_rejected_before_any_work(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='DETAIL only applies to red_team'):
        await run(save=SaveMode.DETAIL, output_dir=tmp_path)
    assert not any(tmp_path.iterdir())


@pytest.mark.asyncio
async def test_invalid_save_value_is_rejected() -> None:
    with pytest.raises(ValueError, match='Invalid save value'):
        await run(save='always')


@pytest.mark.asyncio
async def test_output_dir_without_save_warns_and_writes_nothing(tmp_path: Path) -> None:
    messages: list[str] = []
    sink = logger.add(lambda m: messages.append(m.record['message']), level='WARNING')
    try:
        await run(output_dir=tmp_path / 'out')
    finally:
        logger.remove(sink)
    assert not (tmp_path / 'out').exists()
    assert any('output_dir' in m and 'ignored' in m for m in messages)


@pytest.mark.asyncio
@pytest.mark.skipif(os.name == 'nt' or (hasattr(os, 'geteuid') and os.geteuid() == 0), reason='needs POSIX permission bits and a non-root user')
async def test_an_unwritable_save_folder_fails_before_any_job_runs(tmp_path: Path) -> None:
    locked = tmp_path / 'locked'
    locked.mkdir()
    locked.chmod(0o500)
    calls: list[int] = []

    async def counted(_dp: DataPoint, row: int) -> str:
        calls.append(row)
        return 'Paris'

    try:
        with pytest.raises(ValueError, match='not writable') as raised:
            await evaluatorq(
                'capitals',
                data=DATA,
                jobs=[job('answer', counted)],
                print_results=False,
                save=SaveMode.FINAL,
                output_dir=locked,
                _send_results=False,
            )
    finally:
        locked.chmod(0o700)
    assert str(locked) in str(raised.value)
    assert calls == []
    assert not any(locked.iterdir())


@pytest.mark.asyncio
async def test_an_unwritable_manifest_folder_fails_before_any_work(tmp_path: Path) -> None:
    (tmp_path / '.manifests').write_text('x')
    with pytest.raises(ValueError, match='not writable') as raised:
        await run(save=SaveMode.FINAL, output_dir=tmp_path)
    assert '.manifests' in str(raised.value)
    assert not list(tmp_path.glob('capitals_*.json'))


@pytest.mark.asyncio
async def test_a_save_folder_that_is_a_file_fails_before_any_work(tmp_path: Path) -> None:
    blocker = tmp_path / 'not-a-folder'
    blocker.write_text('x')
    with pytest.raises(ValueError, match='not writable'):
        await run(save=SaveMode.FINAL, output_dir=blocker / 'runs')


@pytest.mark.asyncio
async def test_the_pre_run_check_creates_the_folder_and_leaves_no_probe_file(tmp_path: Path) -> None:
    out = tmp_path / 'new' / 'runs'
    await run(save=SaveMode.FINAL, output_dir=out)
    assert len(list(out.glob('capitals_*.json'))) == 1
    assert not list(out.glob('.evaluatorq-write-check-*'))



@pytest.mark.asyncio
async def test_a_failed_save_logs_fails_the_manifest_and_still_returns_results(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = importlib.import_module('evaluatorq.evaluatorq')

    def refuse(*_args: object, **_kwargs: object) -> Path:
        raise OSError('disk full')

    monkeypatch.setattr(module, 'save_experiment_run', refuse)
    _fake_upload(monkeypatch, url='https://my.orq.ai/experiments/exp-1')
    url_out: list[str] = []
    messages: list[str] = []
    sink = logger.add(lambda m: messages.append(m.record['message']), level='ERROR')
    try:
        results = await run(save=SaveMode.FINAL, output_dir=tmp_path, upload=True, url_out=url_out)
    finally:
        logger.remove(sink)
    assert len(results) == 2
    assert url_out == ['https://my.orq.ai/experiments/exp-1']
    assert any(
        'Could not save experiment run' in m and 'disk full' in m and str(tmp_path) in m and 'unaffected' in m
        for m in messages
    )
    [manifest] = _manifests(tmp_path)
    assert manifest['status'] == 'error'
    assert 'disk full' in json.dumps(manifest)


def _fake_upload(monkeypatch: pytest.MonkeyPatch, *, url: str | None = None, error: Exception | None = None) -> None:
    module = importlib.import_module('evaluatorq.evaluatorq')

    async def upload(*_args: object, **_kwargs: object) -> OrqResponse:
        if error is not None:
            raise error
        return OrqResponse(sheet_id='s', manifest_id='m', experiment_name='capitals', rows_created=2, experiment_url=url)

    monkeypatch.setattr(module, 'send_results_to_orq', upload)


@pytest.mark.asyncio
async def test_the_uploaded_experiment_url_is_saved_and_handed_back(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    url = 'https://my.orq.ai/workspace/experiments/abc'
    _fake_upload(monkeypatch, url=url)
    url_out: list[str] = ['earlier']
    await run(save=SaveMode.FINAL, output_dir=tmp_path, upload=True, url_out=url_out)
    assert url_out == ['earlier', url]
    loaded = load_experiment_run(next(tmp_path.glob('capitals_*.json')))
    assert loaded.experiment_url == url
    assert len(loaded.results) == 2
    assert _manifests(tmp_path)[0]['status'] == 'completed'


@pytest.mark.asyncio
@pytest.mark.skipif(os.name == 'nt', reason='needs POSIX permission bits')
async def test_adding_the_experiment_url_keeps_the_file_mode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _fake_upload(monkeypatch, url=None)
    await run(save=SaveMode.FINAL, output_dir=tmp_path / 'plain', upload=True)
    plain_mode = next((tmp_path / 'plain').glob('capitals_*.json')).stat().st_mode & 0o777
    _fake_upload(monkeypatch, url='https://my.orq.ai/workspace/experiments/abc')
    await run(save=SaveMode.FINAL, output_dir=tmp_path / 'linked', upload=True)
    linked = next((tmp_path / 'linked').glob('capitals_*.json'))
    assert load_experiment_run(linked).experiment_url is not None
    assert linked.stat().st_mode & 0o777 == plain_mode


@pytest.mark.asyncio
async def test_an_upload_with_no_url_saves_none(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _fake_upload(monkeypatch, url=None)
    url_out: list[str] = []
    await run(save=SaveMode.FINAL, output_dir=tmp_path, upload=True, url_out=url_out)
    assert url_out == []
    assert load_experiment_run(next(tmp_path.glob('capitals_*.json'))).experiment_url is None


@pytest.mark.asyncio
async def test_the_saved_run_survives_a_raising_upload(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _fake_upload(monkeypatch, error=RuntimeError('orq is down'))
    with pytest.raises(RuntimeError, match='orq is down'):
        await run(save=SaveMode.FINAL, output_dir=tmp_path, upload=True)
    loaded = load_experiment_run(next(tmp_path.glob('capitals_*.json')))
    assert len(loaded.results) == 2
    assert loaded.experiment_url is None
    [manifest] = _manifests(tmp_path)
    assert manifest['status'] == 'completed'


class _Writer:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def cancel(self) -> None:
        self.calls.append('cancel')

    def fail(self, message: str) -> None:
        self.calls.append(f'fail: {message}')


@pytest.mark.parametrize(
    ('exc', 'expected'),
    [
        (asyncio.CancelledError(), 'cancel'),
        (KeyboardInterrupt(), 'cancel'),
        (RuntimeError('bad'), 'fail: RuntimeError: bad'),
    ],
)
def test_a_run_that_raises_closes_its_manifest(exc: BaseException, expected: str) -> None:
    from evaluatorq.evaluatorq import _finish_manifest_on_error

    writer = _Writer()
    with pytest.raises(type(exc)), _finish_manifest_on_error(cast('ManifestWriter', writer)):
        raise exc
    assert writer.calls == [expected]

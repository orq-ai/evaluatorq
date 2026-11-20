"""Saving ``evaluatorq()`` experiment runs locally, and loading them back."""
# ruff: noqa: S101

from __future__ import annotations

import asyncio
import importlib
import json
from typing import TYPE_CHECKING, cast

import pytest
from loguru import logger

from evaluatorq import DataPoint, EvaluationResult, SaveMode, evaluatorq, load_experiment_run
from evaluatorq.contracts import TokenUsage
from evaluatorq.job_helper import job

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
    *, save: SaveMode | str = SaveMode.NONE, output_dir: Path | str | None = None
) -> list[DataPointResult]:
    return await evaluatorq(
        'capitals',
        description='tiny QA',
        data=DATA,
        jobs=[job('answer', answer)],
        evaluators=[{'name': 'mentions-paris', 'scorer': judge}],
        print_results=False,
        save=save,
        output_dir=output_dir,
        _send_results=False,
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
    assert manifest['summary'] == {'total_results': 2, 'scorer_averages': {'mentions-paris': 1.0}}

    loaded = load_experiment_run(files[0])
    assert loaded.run_name == 'capitals'
    assert loaded.description == 'tiny QA'
    assert loaded.evaluator_names == ['mentions-paris']
    assert loaded.ended_at >= loaded.started_at
    assert loaded.results == results


@pytest.mark.asyncio
async def test_save_accepts_the_string_value(store: Path) -> None:
    await run(save='final')
    assert len(list((store / 'experiment-runs').glob('*.json'))) == 1


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
async def test_a_failed_save_logs_fails_the_manifest_and_still_returns_results(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = importlib.import_module('evaluatorq.evaluatorq')

    def refuse(*_args: object, **_kwargs: object) -> Path:
        raise OSError('disk full')

    monkeypatch.setattr(module, 'save_experiment_run', refuse)
    messages: list[str] = []
    sink = logger.add(lambda m: messages.append(m.record['message']), level='ERROR')
    try:
        results = await run(save=SaveMode.FINAL, output_dir=tmp_path)
    finally:
        logger.remove(sink)
    assert len(results) == 2
    assert any('Could not save experiment run' in m and 'disk full' in m for m in messages)
    [manifest] = _manifests(tmp_path)
    assert manifest['status'] == 'error'
    assert 'disk full' in json.dumps(manifest)


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

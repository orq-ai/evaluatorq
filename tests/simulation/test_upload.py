"""Tests for the auto-upload helper used by simulate() (RES-598)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from evaluatorq.simulation.types import (
    Message,
    SimulationResult,
    TerminatedBy,
    TokenUsage,
    TurnMetrics,
)
from evaluatorq.simulation.upload import (
    JOB_NAME,
    _to_data_point_result,
    upload_simulation_results,
)


def _make_result(
    *,
    persona: str = "Tester",
    scenario: str = "Smoke",
    evaluator_scores: dict[str, float] | None = None,
    error: str | None = None,
) -> SimulationResult:
    metadata: dict[str, object] = {"persona": persona, "scenario": scenario}
    if evaluator_scores is not None:
        metadata["evaluator_scores"] = evaluator_scores
    if error is not None:
        metadata["error"] = error
    return SimulationResult(
        messages=[
            Message(role="user", content="hi"),
            Message(role="assistant", content="hello"),
        ],
        terminated_by=TerminatedBy.judge,
        reason="done",
        goal_achieved=True,
        goal_completion_score=1.0,
        rules_broken=[],
        turn_count=1,
        turn_metrics=[
            TurnMetrics(turn_number=1, token_usage=TokenUsage(), judge_reason="ok")
        ],
        token_usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# _to_data_point_result
# ---------------------------------------------------------------------------


def test_to_data_point_result_inputs_carry_persona_and_scenario():
    dpr = _to_data_point_result(_make_result(), model="gpt-4o-mini")
    assert dpr.data_point.inputs == {"persona": "Tester", "scenario": "Smoke"}
    assert dpr.error is None
    assert dpr.job_results is not None
    assert len(dpr.job_results) == 1
    assert dpr.job_results[0].job_name == JOB_NAME


def test_to_data_point_result_output_is_open_responses_dict():
    dpr = _to_data_point_result(_make_result(), model="gpt-4o-mini")
    assert dpr.job_results is not None
    output = dpr.job_results[0].output
    assert isinstance(output, dict)
    assert output["object"] == "response"
    assert output["model"] == "gpt-4o-mini"


def test_to_data_point_result_includes_evaluator_scores():
    result = _make_result(evaluator_scores={"goal_achieved": 1.0, "criteria_met": 0.5})
    dpr = _to_data_point_result(result, model="m")
    assert dpr.job_results is not None
    scores = dpr.job_results[0].evaluator_scores or []
    by_name = {s.evaluator_name: s.score.value for s in scores}
    assert by_name == {"goal_achieved": 1.0, "criteria_met": 0.5}


def test_to_data_point_result_propagates_error():
    result = _make_result(error="boom")
    dpr = _to_data_point_result(result, model="m")
    assert dpr.error == "boom"
    assert dpr.job_results is not None
    assert dpr.job_results[0].error == "boom"


def test_to_data_point_result_handles_no_metadata():
    result = SimulationResult(
        messages=[Message(role="user", content="x")],
        terminated_by=TerminatedBy.error,
        reason="r",
        goal_achieved=False,
        goal_completion_score=0,
        rules_broken=[],
        turn_count=0,
        turn_metrics=[],
        token_usage=TokenUsage(),
        metadata={},
    )
    dpr = _to_data_point_result(result, model="m")
    assert dpr.data_point.inputs == {}
    assert dpr.error is None


# ---------------------------------------------------------------------------
# upload_simulation_results
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upload_calls_send_results_to_orq_with_correct_shape():
    sent: dict[str, object] = {}

    async def fake_send(**kwargs: object) -> None:
        sent.update(kwargs)

    results = [_make_result(persona="A"), _make_result(persona="B")]
    start = datetime(2026, 5, 7, tzinfo=timezone.utc)
    end = datetime(2026, 5, 7, 0, 0, 1, tzinfo=timezone.utc)

    with patch(
        "evaluatorq.simulation.upload.send_results_to_orq",
        side_effect=fake_send,
    ):
        await upload_simulation_results(
            api_key="test-key",
            evaluation_name="my-eval",
            evaluation_description=None,
            results=results,
            start_time=start,
            end_time=end,
            model="gpt-4o-mini",
        )

    assert sent["api_key"] == "test-key"
    assert sent["evaluation_name"] == "my-eval"
    assert sent["evaluation_description"] is None
    assert sent["dataset_id"] is None
    assert sent["start_time"] == start
    assert sent["end_time"] == end
    sent_results = sent["results"]
    assert isinstance(sent_results, list)
    assert len(sent_results) == 2
    assert sent_results[0].data_point.inputs["persona"] == "A"
    assert sent_results[1].data_point.inputs["persona"] == "B"


@pytest.mark.asyncio
async def test_upload_skips_empty_results():
    """An empty list short-circuits without calling send_results_to_orq."""
    fake = AsyncMock()
    with patch("evaluatorq.simulation.upload.send_results_to_orq", new=fake):
        await upload_simulation_results(
            api_key="test-key",
            evaluation_name="x",
            evaluation_description=None,
            results=[],
            start_time=datetime.now(tz=timezone.utc),
            end_time=datetime.now(tz=timezone.utc),
            model="m",
        )
    fake.assert_not_called()


@pytest.mark.asyncio
async def test_upload_skips_per_result_conversion_failures(
    caplog: pytest.LogCaptureFixture,
):
    """One malformed SimulationResult must not drop the whole batch."""
    import logging

    sent: dict[str, object] = {}

    async def fake_send(**kwargs: object) -> None:
        sent.update(kwargs)

    good = _make_result(persona="good")
    bad = _make_result(persona="bad")

    def conversion(result, _model):
        if result is bad:
            raise RuntimeError("boom")
        from evaluatorq.simulation.convert import to_open_responses

        return to_open_responses(result, _model)

    with caplog.at_level(logging.WARNING, logger="evaluatorq.simulation.upload"), patch(
        "evaluatorq.simulation.upload.send_results_to_orq", side_effect=fake_send
    ), patch(
        "evaluatorq.simulation.upload.to_open_responses", side_effect=conversion
    ):
        await upload_simulation_results(
            api_key="k",
            evaluation_name="e",
            evaluation_description=None,
            results=[good, bad],
            start_time=datetime.now(tz=timezone.utc),
            end_time=datetime.now(tz=timezone.utc),
            model="m",
        )

    sent_results = sent["results"]
    assert isinstance(sent_results, list)
    assert len(sent_results) == 1, "good result should still be uploaded"
    assert sent_results[0].data_point.inputs["persona"] == "good"
    assert any(
        "Skipping simulation result" in r.message for r in caplog.records
    )


def test_to_data_point_result_handles_terminated_by_error():
    """An error-terminated SimulationResult still produces a usable DataPointResult."""
    from evaluatorq.simulation.types import (
        Message,
        SimulationResult,
        TerminatedBy,
        TokenUsage,
    )

    result = SimulationResult(
        messages=[Message(role="user", content="hi")],
        terminated_by=TerminatedBy.error,
        reason="kaboom",
        goal_achieved=False,
        goal_completion_score=0,
        rules_broken=[],
        turn_count=0,
        turn_metrics=[],
        token_usage=TokenUsage(),
        metadata={
            "persona": "P",
            "scenario": "S",
            "error": "kaboom",
        },
    )
    dpr = _to_data_point_result(result, model="m")
    assert dpr.error == "kaboom"
    assert dpr.job_results is not None
    assert dpr.job_results[0].error == "kaboom"
    # output is still well-formed (OpenResponses dict)
    output = dpr.job_results[0].output
    assert isinstance(output, dict)
    assert output["object"] == "response"



@pytest.mark.asyncio
async def test_upload_swallows_send_errors(caplog: pytest.LogCaptureFixture):
    """Network failures must NOT propagate — simulation must succeed even
    when Orq upload fails (matches evaluatorq core's contract)."""
    import logging

    async def fake_send(**_: object) -> None:
        raise RuntimeError("network down")

    with caplog.at_level(logging.ERROR, logger="evaluatorq.simulation.upload"), patch(
        "evaluatorq.simulation.upload.send_results_to_orq", side_effect=fake_send
    ):
        await upload_simulation_results(
            api_key="k",
            evaluation_name="x",
            evaluation_description=None,
            results=[_make_result()],
            start_time=datetime.now(tz=timezone.utc),
            end_time=datetime.now(tz=timezone.utc),
            model="m",
        )

    assert any("Failed to upload" in r.message for r in caplog.records)



# ---------------------------------------------------------------------------
# simulate() now routes through evaluatorq() — RES-594. These tests verify
# the wiring (upload_results flag, evaluation_name, description, path) at the
# seam where simulate() calls evaluatorq(), without spinning up the real
# evaluatorq machinery.
# ---------------------------------------------------------------------------


def _make_datapoint():
    from evaluatorq.simulation.types import (
        CommunicationStyle,
        Datapoint,
        Persona,
        Scenario,
    )

    return Datapoint(
        id="dp",
        persona=Persona(
            name="P",
            patience=0.5,
            assertiveness=0.5,
            politeness=0.5,
            technical_level=0.5,
            communication_style=CommunicationStyle.casual,
            background="bg",
        ),
        scenario=Scenario(name="S", goal="g"),
        user_system_prompt="sys",
        first_message="hi",
    )


import sys
import evaluatorq.evaluatorq  # noqa: F401  # ensure submodule registered in sys.modules
_evq_module = sys.modules["evaluatorq.evaluatorq"]


@pytest.mark.asyncio
async def test_simulate_calls_evaluatorq_with_send_results_true_by_default():
    """RES-594: native integration means evaluatorq()'s upload is the canonical
    persistence path, so simulate() defaults to letting it fire."""
    from evaluatorq.simulation import simulate

    async def target_cb(_msgs):
        return "ok"

    eq_mock = AsyncMock(return_value=[])
    with patch.object(_evq_module, "evaluatorq", new=eq_mock):
        await simulate(
            evaluation_name="my-eval",
            target_callback=target_cb,
            datapoints=[_make_datapoint()],
            max_turns=1,
            exit_on_failure=False,  # stub evaluatorq doesn't populate result_cache
        )

    eq_mock.assert_awaited_once()
    call = eq_mock.await_args
    assert call is not None
    assert call.args[0] == "my-eval"  # type: ignore[index]
    assert call.kwargs["_send_results"] is True


@pytest.mark.asyncio
async def test_simulate_forwards_upload_results_false():
    from evaluatorq.simulation import simulate

    async def target_cb(_msgs):
        return "ok"

    eq_mock = AsyncMock(return_value=[])
    with patch.object(_evq_module, "evaluatorq", new=eq_mock):
        await simulate(
            evaluation_name="my-eval",
            target_callback=target_cb,
            datapoints=[_make_datapoint()],
            max_turns=1,
            upload_results=False,
            exit_on_failure=False,
        )

    assert eq_mock.await_args is not None
    assert eq_mock.await_args.kwargs["_send_results"] is False


@pytest.mark.asyncio
async def test_simulate_synthesises_run_name_when_empty():
    from evaluatorq.simulation import simulate

    async def target_cb(_msgs):
        return "ok"

    eq_mock = AsyncMock(return_value=[])
    with patch.object(_evq_module, "evaluatorq", new=eq_mock):
        await simulate(
            evaluation_name="",
            target_callback=target_cb,
            datapoints=[_make_datapoint()],
            max_turns=1,
            exit_on_failure=False,
        )

    assert eq_mock.await_args is not None
    name = eq_mock.await_args.args[0]
    assert name.startswith("simulation-")
    # YYYYMMDD-HHMMSS-{8 hex} → 8+1+6+1+8 = 24 chars after the "simulation-" prefix
    assert len(name) == len("simulation-") + 24


@pytest.mark.asyncio
async def test_simulate_passes_description_and_path():
    from evaluatorq.simulation import simulate

    async def target_cb(_msgs):
        return "ok"

    eq_mock = AsyncMock(return_value=[])
    with patch.object(_evq_module, "evaluatorq", new=eq_mock):
        await simulate(
            evaluation_name="e",
            target_callback=target_cb,
            datapoints=[_make_datapoint()],
            max_turns=1,
            evaluation_description="my run",
            path="Proj/Folder",
            exit_on_failure=False,
        )

    assert eq_mock.await_args is not None
    kwargs = eq_mock.await_args.kwargs
    assert kwargs["description"] == "my run"
    assert kwargs["path"] == "Proj/Folder"


# ---------------------------------------------------------------------------
# RES-594: _sim_idx wiring — verifies the stable-key cache path end-to-end
# by driving a fake evaluatorq that mimics the real "call job, then scorer
# with the same DataPoint" contract.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_simulate_sim_idx_roundtrip_through_evaluatorq(
    monkeypatch: pytest.MonkeyPatch,
):
    """simulate() seeds each DataPoint.inputs with _sim_idx; the job stashes
    SimulationResults under that idx; the scorer adapter and the final
    result list both recover by idx. Verifies the contract holds across the
    full lifecycle, not just the call to evaluatorq()."""
    from evaluatorq.simulation import simulate
    from evaluatorq.simulation.runner.simulation import SimulationRunner

    canned = [
        _make_result(persona=f"P{i}", scenario=f"S{i}") for i in range(3)
    ]
    dps = [_make_datapoint() for _ in range(3)]

    # Drive simulate()'s real wiring with a stub evaluatorq that calls the
    # job for each datapoint then the scorer, mirroring processings.py.
    async def fake_evaluatorq(name, *, data, jobs, evaluators, **_kwargs):
        for dp in data:
            await jobs[0](dp, 0)
            for ev in evaluators:
                await ev["scorer"]({"data": dp, "output": {}})

    runner_run = AsyncMock(side_effect=canned)
    monkeypatch.setattr(SimulationRunner, "run", runner_run)
    monkeypatch.setattr(SimulationRunner, "close", AsyncMock())

    with patch.object(_evq_module, "evaluatorq", new=fake_evaluatorq):
        results = await simulate(
            evaluation_name="t",
            target_callback=lambda _msgs: "ok",
            datapoints=dps,
            max_turns=1,
            evaluator_names=["goal_achieved"],
            upload_results=False,
        )

    # Returned in input order via _sim_idx lookup
    assert len(results) == 3
    assert [r.metadata["persona"] for r in results] == ["P0", "P1", "P2"]
    # Scorer adapter mirrored the score into each result's metadata
    for r in results:
        assert "evaluator_scores" in r.metadata
        assert "goal_achieved" in r.metadata["evaluator_scores"]


@pytest.mark.asyncio
async def test_simulate_warns_when_jobs_drop_results(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    """If a job throws, evaluatorq swallows it and result_cache stays short.
    simulate() must log a warning naming the caller so the missing rows are
    not silently dropped from the returned list."""
    from evaluatorq.simulation import simulate
    from evaluatorq.simulation.runner.simulation import SimulationRunner

    # Two datapoints; only the first job succeeds.
    dps = [_make_datapoint(), _make_datapoint()]
    runner_run = AsyncMock(
        side_effect=[_make_result(), RuntimeError("agent blew up")]
    )
    monkeypatch.setattr(SimulationRunner, "run", runner_run)
    monkeypatch.setattr(SimulationRunner, "close", AsyncMock())

    async def fake_evaluatorq(name, *, data, jobs, evaluators, **_kwargs):
        for dp in data:
            try:
                await jobs[0](dp, 0)
            except Exception:  # noqa: BLE001 — mimic evaluatorq swallowing job errors
                pass

    with patch.object(_evq_module, "evaluatorq", new=fake_evaluatorq):
        with caplog.at_level("WARNING", logger="evaluatorq.simulation.api"):
            results = await simulate(
                target_callback=lambda _msgs: "ok",
                datapoints=dps,
                max_turns=1,
                upload_results=False,
                exit_on_failure=False,
            )

    assert len(results) == 1
    assert any(
        "simulate(): 1 of 2 simulation job" in rec.message
        for rec in caplog.records
    )


@pytest.mark.asyncio
async def test_simulate_raises_on_dropped_results_when_exit_on_failure(
    monkeypatch: pytest.MonkeyPatch,
):
    """With exit_on_failure=True, simulate() must hard-raise when jobs
    failed and reduced the result count — otherwise CI silently looks
    green on broken runs."""
    from evaluatorq.simulation import simulate
    from evaluatorq.simulation.runner.simulation import SimulationRunner

    dps = [_make_datapoint(), _make_datapoint()]
    monkeypatch.setattr(
        SimulationRunner,
        "run",
        AsyncMock(side_effect=[_make_result(), RuntimeError("blew up")]),
    )
    monkeypatch.setattr(SimulationRunner, "close", AsyncMock())

    async def fake_evaluatorq(name, *, data, jobs, evaluators, **_kwargs):
        for dp in data:
            try:
                await jobs[0](dp, 0)
            except Exception:  # noqa: BLE001
                pass

    with patch.object(_evq_module, "evaluatorq", new=fake_evaluatorq):
        with pytest.raises(RuntimeError, match="1 of 2 simulation job"):
            await simulate(
                target_callback=lambda _msgs: "ok",
                datapoints=dps,
                max_turns=1,
                upload_results=False,
                exit_on_failure=True,
            )


@pytest.mark.asyncio
async def test_simulate_forwards_exit_on_failure(monkeypatch: pytest.MonkeyPatch):
    """exit_on_failure=True propagates to evaluatorq's _exit_on_failure so
    callers wiring simulate() into CI can opt into hard-failing on errors."""
    from evaluatorq.simulation import simulate
    from evaluatorq.simulation.runner.simulation import SimulationRunner

    monkeypatch.setattr(
        SimulationRunner, "run", AsyncMock(return_value=_make_result())
    )
    monkeypatch.setattr(SimulationRunner, "close", AsyncMock())

    captured: dict[str, Any] = {}

    async def fake_evaluatorq(name, *, data, jobs, evaluators, **kwargs):
        captured.update(kwargs)
        for dp in data:
            await jobs[0](dp, 0)

    with patch.object(_evq_module, "evaluatorq", new=fake_evaluatorq):
        await simulate(
            target_callback=lambda _msgs: "ok",
            datapoints=[_make_datapoint()],
            max_turns=1,
            upload_results=False,
            exit_on_failure=True,
        )
    assert captured["_exit_on_failure"] is True


@pytest.mark.asyncio
async def test_simulate_dataset_id_fetches_and_runs(monkeypatch: pytest.MonkeyPatch):
    """When given dataset_id, simulate() must fetch the dataset via the Orq
    client, parse each row through the same shape-tolerant extractor used
    inline, and run the standard pipeline. No inline datapoints supplied."""
    from evaluatorq.simulation import simulate
    from evaluatorq.simulation.runner.simulation import SimulationRunner
    from evaluatorq.types import DataPoint

    monkeypatch.setenv("ORQ_API_KEY", "test-key")

    sim_dp = _make_datapoint()
    eq_rows = [DataPoint(inputs={"datapoint": sim_dp.model_dump(mode="json")})]

    class _FakeBatch:
        def __init__(self, dps):
            self.datapoints = dps

    async def fake_fetch(_client, _dataset_id, **_kwargs):
        yield _FakeBatch(eq_rows)
        return

    monkeypatch.setattr(
        "evaluatorq.fetch_data.fetch_dataset_batches", fake_fetch
    )
    monkeypatch.setattr(
        "evaluatorq.fetch_data.setup_orq_client", lambda _k: object()  # pyright: ignore[reportUnknownLambdaType]
    )

    runner_run = AsyncMock(return_value=_make_result())
    monkeypatch.setattr(SimulationRunner, "run", runner_run)
    monkeypatch.setattr(SimulationRunner, "close", AsyncMock())

    async def fake_evaluatorq(name, *, data, jobs, evaluators, **_kwargs):
        for dp in data:
            await jobs[0](dp, 0)

    with patch.object(_evq_module, "evaluatorq", new=fake_evaluatorq):
        results = await simulate(
            target_callback=lambda _msgs: "ok",
            dataset_id="ds-123",
            max_turns=1,
            upload_results=False,
        )

    assert len(results) == 1
    runner_run.assert_awaited_once()


@pytest.mark.asyncio
async def test_simulate_rejects_dataset_id_with_inline_datapoints():
    """dataset_id, datapoints, and personas/scenarios are mutually exclusive
    sources — passing two raises before any work starts."""
    from evaluatorq.simulation import simulate

    with pytest.raises(ValueError, match="exactly one of"):
        await simulate(
            target_callback=lambda _msgs: "ok",
            dataset_id="ds-123",
            datapoints=[_make_datapoint()],
            max_turns=1,
            upload_results=False,
        )


@pytest.mark.asyncio
async def test_simulate_scorer_error_raises_so_evaluatorq_records_it(
    monkeypatch: pytest.MonkeyPatch,
):
    """A failing scorer must raise so evaluatorq's process_evaluator catches
    the exception and records a structured error on EvaluatorScore. Silently
    returning value=0.0 looked like a legitimate low score and defeated
    exit_on_failure."""
    from evaluatorq.simulation import simulate
    from evaluatorq.simulation.runner.simulation import SimulationRunner

    dps = [_make_datapoint()]
    monkeypatch.setattr(
        SimulationRunner, "run", AsyncMock(return_value=_make_result())
    )
    monkeypatch.setattr(SimulationRunner, "close", AsyncMock())

    raised: list[BaseException] = []

    async def fake_evaluatorq(name, *, data, jobs, evaluators, **_kwargs):
        for dp in data:
            await jobs[0](dp, 0)
            for ev in evaluators:
                try:
                    await ev["scorer"]({"data": dp, "output": {}})
                except Exception as e:  # noqa: BLE001
                    raised.append(e)

    def boom(_result):
        raise ValueError("scorer broke")

    from evaluatorq.simulation.evaluators.scorers import SIMULATION_EVALUATORS

    monkeypatch.setitem(SIMULATION_EVALUATORS, "broken_scorer", boom)

    with patch.object(_evq_module, "evaluatorq", new=fake_evaluatorq):
        await simulate(
            target_callback=lambda _msgs: "ok",
            datapoints=dps,
            max_turns=1,
            evaluator_names=["broken_scorer"],
            upload_results=False,
        )

    assert len(raised) == 1
    assert isinstance(raised[0], ValueError)
    assert "scorer broke" in str(raised[0])

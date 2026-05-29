"""Tests for the auto-upload helper used by simulate() (RES-598)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, cast
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

if TYPE_CHECKING:
    from evaluatorq.simulation.agents.base import BaseAgent


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
# Shared fixtures for simulate() integration tests
#
# These use real stub classes (not bare MagicMock) for the injected user
# simulator and judge. The runner validates them via
# ``isinstance(obj, SimulationUserSimulator)`` against a ``@runtime_checkable``
# Protocol; on Python 3.12+ that check is strict enough that a bare MagicMock
# fails, so concrete stubs keep the tests green across all supported versions.
# ---------------------------------------------------------------------------


def _make_persona(name: str = "P"):
    from evaluatorq.simulation.types import CommunicationStyle, Persona

    return Persona(
        name=name,
        patience=0.5,
        assertiveness=0.5,
        politeness=0.5,
        technical_level=0.5,
        communication_style=CommunicationStyle.casual,
        background="bg",
    )


def _make_scenario(name: str = "S", goal: str = "g"):
    from evaluatorq.simulation.types import Scenario

    return Scenario(name=name, goal=goal)


def _make_datapoint():
    from evaluatorq.simulation.types import Datapoint

    return Datapoint(
        id="dp",
        persona=_make_persona(),
        scenario=_make_scenario(),
        user_system_prompt="sys",
        first_message="hi",
    )


def _make_judgment():
    from evaluatorq.simulation.types import Judgment

    return Judgment(
        should_terminate=True,
        reason="done",
        goal_achieved=True,
        rules_broken=[],
        goal_completion_score=1.0,
        response_quality=0.9,
        hallucination_risk=0.1,
        tone_appropriateness=0.9,
        factual_accuracy=0.9,
    )


class _StubUserSimulator:
    """Concrete user simulator satisfying the runner's Protocol."""

    def __init__(self, *, first: str = "hi", reply: str = "bye") -> None:
        self._first = first
        self._reply = reply

    def update_context(self, *, persona_context=None, scenario_context=None) -> None:
        return None

    async def generate_first_message(self) -> str:
        return self._first

    async def respond_async(self, messages, *, llm_purpose=None) -> str:
        return self._reply

    def get_usage(self) -> TokenUsage:
        return TokenUsage()


class _StubJudge:
    """Concrete judge satisfying the runner's Protocol."""

    def __init__(self, judgment=None) -> None:
        self._judgment = judgment if judgment is not None else _make_judgment()

    async def evaluate(self, messages):
        return self._judgment

    def get_usage(self) -> TokenUsage:
        return TokenUsage()


# simulate() annotates user_simulator/judge as BaseAgent, but the runner
# validates them structurally against a @runtime_checkable Protocol. The stubs
# satisfy that Protocol; cast() bridges the nominal/structural gap for the
# type checker without forcing a heavyweight BaseAgent subclass.
def _user_sim(**kwargs) -> BaseAgent:
    return cast("BaseAgent", cast(object, _StubUserSimulator(**kwargs)))


def _judge() -> BaseAgent:
    return cast("BaseAgent", cast(object, _StubJudge()))


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
async def test_upload_raises_per_result_conversion_failures():
    """A malformed SimulationResult fails the requested upload."""
    from evaluatorq.simulation.types import (
        Message,
        SimulationResult,
        TerminatedBy,
        TokenUsage,
    )

    good = _make_result(persona="good")
    # ``to_open_responses`` reads attributes off the result; corrupt one of
    # those attributes by giving messages a non-list value via dict-bypass.
    bad = SimulationResult(
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
    object.__setattr__(bad, "messages", "not a list")  # break it post-validation

    with patch("evaluatorq.simulation.upload.send_results_to_orq", new=AsyncMock()):
        with pytest.raises(Exception):
            await upload_simulation_results(
                api_key="k",
                evaluation_name="e",
                evaluation_description=None,
                results=[good, bad],
                start_time=datetime.now(tz=timezone.utc),
                end_time=datetime.now(tz=timezone.utc),
                model="m",
            )


@pytest.mark.asyncio
async def test_upload_raises_send_errors():
    """Network failures propagate so callers can detect failed uploads."""

    async def fake_send(**_: object) -> None:
        raise RuntimeError("network down")

    with patch("evaluatorq.simulation.upload.send_results_to_orq", side_effect=fake_send):
        with pytest.raises(RuntimeError, match="network down"):
            await upload_simulation_results(
                api_key="k",
                evaluation_name="x",
                evaluation_description=None,
                results=[_make_result()],
                start_time=datetime.now(tz=timezone.utc),
                end_time=datetime.now(tz=timezone.utc),
                model="m",
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
async def test_wrap_simulation_agent_does_not_double_upload(
    monkeypatch: pytest.MonkeyPatch,
):
    """Regression: wrap_simulation_agent's nested simulate() call must NOT
    auto-upload, otherwise evaluatorq's framework upload + simulate's own
    upload create two experiments per run."""
    monkeypatch.setenv("ORQ_API_KEY", "test-key")

    upload_mock = AsyncMock()

    from evaluatorq.simulation.types import Message
    from evaluatorq.simulation.wrap_agent import wrap_simulation_agent

    dp = _make_datapoint()

    async def target_cb(messages: list[Message]) -> str:
        return "ok"

    job_fn = wrap_simulation_agent(
        target_callback=target_cb,
        max_turns=1,
    )

    # The wrap_agent path passes inputs as a dict — feed it a single datapoint
    inputs = {"datapoint": dp.model_dump()}

    from evaluatorq.types import DataPoint

    eval_dp = DataPoint(inputs=inputs)

    with patch(
        "evaluatorq.simulation.upload.upload_simulation_results", new=upload_mock
    ):
        # Patch the user_simulator/judge factories used by simulate() so we
        # don't hit the network. The cleanest way is to override the
        # SimulationRunner inputs via simulate()'s injected agents — but
        # wrap_simulation_agent doesn't expose them. Instead patch the
        # runner class to return a canned result.
        from evaluatorq.simulation.types import (
            Message,
            SimulationResult,
            TerminatedBy,
        )

        canned = SimulationResult(
            messages=[Message(role="user", content="hi")],
            terminated_by=TerminatedBy.judge,
            reason="done",
            goal_achieved=True,
            goal_completion_score=1.0,
            rules_broken=[],
            turn_count=1,
            turn_metrics=[],
            token_usage=TokenUsage(),
            metadata={"persona": "P", "scenario": "S"},
        )

        with patch(
            "evaluatorq.simulation.runner.simulation.SimulationRunner.run_batch",
            new=AsyncMock(return_value=[canned]),
        ):
            result = await job_fn(eval_dp, 0)

    # job_fn produces an output (sanity)
    assert "output" in result
    # And critically: the auto-upload did NOT fire from inside simulate()
    upload_mock.assert_not_called()


# ---------------------------------------------------------------------------
# simulate() integration: respects upload_results flag and ORQ_API_KEY
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_simulate_calls_upload_when_flag_and_key_set(
    monkeypatch: pytest.MonkeyPatch,
):
    """simulate(upload_results=True) with ORQ_API_KEY set triggers upload."""
    monkeypatch.setenv("ORQ_API_KEY", "test-key")

    upload_mock = AsyncMock()

    from evaluatorq.simulation import simulate

    async def target_cb(messages: list[Message]) -> str:
        return "ok"

    with patch(
        "evaluatorq.simulation.upload.upload_simulation_results", new=upload_mock
    ):
        await simulate(
            evaluation_name="t",
            target_callback=target_cb,
            datapoints=[_make_datapoint()],
            max_turns=1,
            user_simulator=_user_sim(),
            judge=_judge(),
            upload_results=True,
        )

    upload_mock.assert_awaited_once()
    assert upload_mock.await_args is not None
    kwargs = upload_mock.await_args.kwargs
    assert kwargs["api_key"] == "test-key"
    assert kwargs["evaluation_name"] == "t"


@pytest.mark.asyncio
async def test_simulate_skips_upload_when_flag_false(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("ORQ_API_KEY", "test-key")

    upload_mock = AsyncMock()

    from evaluatorq.simulation import simulate

    async def target_cb(messages: list[Message]) -> str:
        return "ok"

    with patch(
        "evaluatorq.simulation.upload.upload_simulation_results", new=upload_mock
    ):
        await simulate(
            evaluation_name="t",
            target_callback=target_cb,
            datapoints=[_make_datapoint()],
            max_turns=1,
            user_simulator=_user_sim(),
            judge=_judge(),
            upload_results=False,
        )

    upload_mock.assert_not_called()


@pytest.mark.asyncio
async def test_generate_and_simulate_calls_upload(
    monkeypatch: pytest.MonkeyPatch,
):
    """Mirror of the simulate() upload test, for the generate_and_simulate path."""
    monkeypatch.setenv("ORQ_API_KEY", "test-key")

    from unittest.mock import MagicMock

    from evaluatorq.simulation import generate_and_simulate
    from evaluatorq.simulation.types import Datapoint, Persona, Scenario

    upload_mock = AsyncMock()
    persona = _make_persona()
    scenario = _make_scenario()

    async def target_cb(messages: list[Message]) -> str:
        return "ok"

    # Stub the persona/scenario generators and the runner so we don't touch
    # the network.
    fake_persona_gen = MagicMock()
    fake_persona_gen.generate = AsyncMock(return_value=[persona])
    fake_scenario_gen = MagicMock()
    fake_scenario_gen.generate = AsyncMock(return_value=[scenario])

    canned_result = _make_result()

    async def fake_gen_dp(_gen: object, p: Persona, s: Scenario) -> Datapoint:
        return Datapoint(
            id="dp",
            persona=p,
            scenario=s,
            user_system_prompt="sys",
            first_message="hi",
        )

    with patch(
        "evaluatorq.simulation.generators.PersonaGenerator",
        return_value=fake_persona_gen,
    ), patch(
        "evaluatorq.simulation.generators.ScenarioGenerator",
        return_value=fake_scenario_gen,
    ), patch(
        "evaluatorq.simulation.api._generate_single_datapoint",
        new=AsyncMock(side_effect=fake_gen_dp),
    ), patch(
        "evaluatorq.simulation.runner.simulation.SimulationRunner.run_batch",
        new=AsyncMock(return_value=[canned_result]),
    ), patch(
        "evaluatorq.simulation.upload.upload_simulation_results", new=upload_mock
    ):
        await generate_and_simulate(
            evaluation_name="gn",
            agent_description="agent",
            target_callback=target_cb,
            num_personas=1,
            num_scenarios=1,
            max_turns=1,
            upload_results=True,
        )

    upload_mock.assert_awaited_once()
    assert upload_mock.await_args is not None
    kwargs = upload_mock.await_args.kwargs
    assert kwargs["evaluation_name"] == "gn"


@pytest.mark.asyncio
async def test_simulate_raises_upload_when_no_api_key(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("ORQ_API_KEY", raising=False)

    from evaluatorq.simulation import simulate

    async def target_cb(messages: list[Message]) -> str:
        return "ok"

    # SimulationRunner needs SOME credential to build its shared client;
    # pass OPENAI_API_KEY instead so the runner is happy but ORQ_API_KEY
    # is absent (the path we want to test).
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key")

    with pytest.raises(ValueError, match="ORQ_API_KEY"):
        await simulate(
            evaluation_name="t",
            target_callback=target_cb,
            datapoints=[_make_datapoint()],
            max_turns=1,
            user_simulator=_user_sim(),
            judge=_judge(),
            upload_results=True,
        )

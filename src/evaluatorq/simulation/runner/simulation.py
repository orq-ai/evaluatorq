"""Simulation runner for orchestrating agent conversations."""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
from collections.abc import Awaitable, Callable
from typing import Protocol, runtime_checkable

from openai import AsyncOpenAI

from evaluatorq.simulation.agents.judge import JudgeAgent, JudgeAgentConfig
from evaluatorq.simulation.agents.user_simulator import (
    UserSimulatorAgent,
    UserSimulatorAgentConfig,
)
from evaluatorq.simulation.types import (
    DEFAULT_MODEL,
    ChatMessage,
    Datapoint,
    Judgment,
    Message,
    Persona,
    Scenario,
    SimulationResult,
    TerminatedBy,
    TokenUsage,
    TurnMetrics,
)
from evaluatorq.simulation.utils.prompt_builders import build_datapoint_system_prompt

logger = logging.getLogger(__name__)

ZERO_USAGE = TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)


def _invert_roles_for_simulator(messages: list[Message]) -> list[ChatMessage]:
    """Swap roles so the user simulator sees the conversation from its perspective."""
    inverted = []
    for m in messages:
        if m.role == "user":
            inverted.append(ChatMessage(role="assistant", content=m.content))
        elif m.role == "assistant":
            inverted.append(ChatMessage(role="user", content=m.content))
        else:
            inverted.append(ChatMessage(role=m.role, content=m.content))
    return inverted


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class TargetAgent(Protocol):
    """Protocol for target agents being tested."""

    async def respond(self, messages: list[ChatMessage]) -> str: ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _error_result(
    reason: str,
    persona: Persona | None = None,
    scenario: Scenario | None = None,
) -> SimulationResult:
    return SimulationResult(
        messages=[],
        terminated_by=TerminatedBy.error,
        reason=reason,
        goal_achieved=False,
        goal_completion_score=0,
        rules_broken=[],
        turn_count=0,
        turn_metrics=[],
        token_usage=ZERO_USAGE.model_copy(),
        metadata={
            "persona": persona.name if persona else "unknown",
            "scenario": scenario.name if scenario else "unknown",
            "error": reason,
        },
    )


def _build_criteria_results(
    scenario: Scenario, judgment: Judgment
) -> dict[str, bool]:
    """Build a human-readable criteria results dict from scenario and judgment."""
    results: dict[str, bool] = {}
    criteria = scenario.criteria or []
    rules_broken = set(judgment.rules_broken)
    for i, criterion in enumerate(criteria):
        criterion_id = f"criteria_{i}"
        results[criterion.description] = criterion_id not in rules_broken
    return results


def _max_turns_result(
    max_turns: int,
    messages: list[Message],
    turn_metrics: list[TurnMetrics],
    token_usage: TokenUsage,
    persona: Persona | None = None,
    scenario: Scenario | None = None,
    last_judgment: Judgment | None = None,
) -> SimulationResult:
    criteria_results = (
        _build_criteria_results(scenario, last_judgment)
        if scenario and last_judgment
        else None
    )
    return SimulationResult(
        messages=messages,
        terminated_by=TerminatedBy.max_turns,
        reason=f"Maximum turns ({max_turns}) reached",
        goal_achieved=last_judgment.goal_achieved if last_judgment else False,
        goal_completion_score=last_judgment.goal_completion_score
        if last_judgment
        else 0,
        rules_broken=last_judgment.rules_broken if last_judgment else [],
        turn_count=max_turns,
        turn_metrics=turn_metrics,
        token_usage=token_usage,
        criteria_results=criteria_results,
        metadata={
            "persona": persona.name if persona else None,
            "scenario": scenario.name if scenario else None,
        },
    )


# ---------------------------------------------------------------------------
# SimulationRunner
# ---------------------------------------------------------------------------


class SimulationRunner:
    """Orchestrates multi-turn conversations between user simulator, target agent, and judge."""

    def __init__(
        self,
        *,
        target_agent: TargetAgent | None = None,
        target_callback: Callable[[list[ChatMessage]], str | Awaitable[str]]
        | None = None,
        model: str = DEFAULT_MODEL,
        max_turns: int = 10,
    ) -> None:
        if not target_agent and not target_callback:
            raise ValueError("Must provide either target_agent or target_callback")
        if max_turns < 1:
            raise ValueError(f"max_turns must be >= 1, got {max_turns}")
        if not model.strip():
            raise ValueError("model must be a non-empty string")

        self._target_agent = target_agent
        self._target_callback = target_callback
        self._model = model
        self._max_turns = max_turns
        self._shared_client: AsyncOpenAI | None = None

    def _get_shared_client(self) -> AsyncOpenAI:
        if not self._shared_client:
            api_key = os.environ.get("ORQ_API_KEY")
            if not api_key:
                raise ValueError("ORQ_API_KEY environment variable is not set.")
            self._shared_client = AsyncOpenAI(
                api_key=api_key,
                base_url=f"{os.environ.get('ORQ_BASE_URL', 'https://api.orq.ai')}/v2/router",
            )
        return self._shared_client

    async def run(
        self,
        *,
        persona: Persona | None = None,
        scenario: Scenario | None = None,
        datapoint: Datapoint | None = None,
        max_turns: int | None = None,
        first_message: str | None = None,
    ) -> SimulationResult:
        """Run a single simulation. Never throws -- returns error SimulationResult on failure."""
        stored_system_prompt: str | None = None

        # Resolve datapoint
        if datapoint:
            persona = datapoint.persona
            scenario = datapoint.scenario
            first_message = first_message or (datapoint.first_message or None)
            stored_system_prompt = datapoint.user_system_prompt or None
        elif not persona or not scenario:
            return _error_result(
                "Must provide either datapoint or both persona and scenario",
                persona,
                scenario,
            )

        effective_max_turns = max_turns or self._max_turns
        messages: list[Message] = []
        turn_metrics_list: list[TurnMetrics] = []
        get_total_usage: Callable[[], TokenUsage] | None = None

        try:
            system_prompt = stored_system_prompt or build_datapoint_system_prompt(
                persona, scenario
            )  # type: ignore[arg-type]
            client = self._get_shared_client()

            user_simulator = UserSimulatorAgent(
                UserSimulatorAgentConfig(
                    model=self._model,
                    client=client,
                    system_prompt=system_prompt,
                )
            )

            judge = JudgeAgent(
                JudgeAgentConfig(
                    model=self._model,
                    client=client,
                    goal=scenario.goal if scenario else "",
                    criteria=list(scenario.criteria)
                    if scenario and scenario.criteria
                    else [],
                    ground_truth=scenario.ground_truth or "" if scenario else "",
                )
            )

            def _get_total_usage() -> TokenUsage:
                usage = user_simulator.get_usage()
                judge_usage = judge.get_usage()
                return TokenUsage(
                    prompt_tokens=usage.prompt_tokens + judge_usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens
                    + judge_usage.completion_tokens,
                    total_tokens=usage.total_tokens + judge_usage.total_tokens,
                )

            get_total_usage = _get_total_usage

            def _build_turn_metrics(
                turn_num: int, judgment: Judgment, usage_before: TokenUsage
            ) -> TurnMetrics:
                usage_after = _get_total_usage()
                return TurnMetrics(
                    turn_number=turn_num,
                    token_usage=TokenUsage(
                        prompt_tokens=usage_after.prompt_tokens
                        - usage_before.prompt_tokens,
                        completion_tokens=usage_after.completion_tokens
                        - usage_before.completion_tokens,
                        total_tokens=usage_after.total_tokens
                        - usage_before.total_tokens,
                    ),
                    response_quality=judgment.response_quality,
                    hallucination_risk=judgment.hallucination_risk,
                    tone_appropriateness=judgment.tone_appropriateness,
                    factual_accuracy=judgment.factual_accuracy,
                    judge_reason=judgment.reason,
                )

            # Generate or use first message
            first_msg = first_message or await user_simulator.generate_first_message()
            messages.append(Message(role="user", content=first_msg))

            last_judgment: Judgment | None = None

            for turn in range(effective_max_turns):
                usage_before = _get_total_usage()

                # 1. Target agent responds
                agent_response = await self._get_target_response(
                    [ChatMessage(role=m.role, content=m.content) for m in messages]
                )
                messages.append(Message(role="assistant", content=agent_response))

                # 2. Judge evaluates
                judgment = await judge.evaluate(
                    [ChatMessage(role=m.role, content=m.content) for m in messages]
                )

                turn_metrics_list.append(
                    _build_turn_metrics(turn + 1, judgment, usage_before)
                )
                last_judgment = judgment

                if judgment.should_terminate:
                    return SimulationResult(
                        messages=messages,
                        terminated_by=TerminatedBy.judge,
                        reason=judgment.reason,
                        goal_achieved=judgment.goal_achieved,
                        goal_completion_score=judgment.goal_completion_score,
                        rules_broken=judgment.rules_broken,
                        turn_count=turn + 1,
                        turn_metrics=turn_metrics_list,
                        token_usage=_get_total_usage(),
                        criteria_results=self._build_criteria_results(
                            scenario, judgment
                        )
                        if scenario
                        else None,  # type: ignore[arg-type]
                        metadata={
                            "persona": persona.name if persona else None,
                            "scenario": scenario.name if scenario else None,
                        },  # type: ignore[union-attr]
                    )

                # 3. User simulator continues (if not last turn)
                if turn < effective_max_turns - 1:
                    user_response = await user_simulator.respond_async(
                        _invert_roles_for_simulator(messages),
                    )
                    messages.append(Message(role="user", content=user_response))

            return _max_turns_result(
                effective_max_turns,
                messages,
                turn_metrics_list,
                _get_total_usage(),
                persona,
                scenario,
                last_judgment,
            )

        except Exception as e:
            logger.error("SimulationRunner.run() failed: %s", e)
            error_msg = str(e)
            try:
                usage = (
                    get_total_usage() if get_total_usage else ZERO_USAGE.model_copy()
                )
            except Exception as e:
                logger.warning("Failed to get token usage: %s", e)
                usage = ZERO_USAGE.model_copy()

            result = _error_result(error_msg, persona, scenario)
            result.messages = messages
            result.turn_count = sum(1 for m in messages if m.role == "assistant")
            result.turn_metrics = turn_metrics_list
            result.token_usage = usage
            return result

    async def run_batch(
        self,
        datapoints: list[Datapoint],
        *,
        max_turns: int | None = None,
        timeout_per_simulation: float = 300.0,
        max_concurrency: int = 10,
    ) -> list[SimulationResult]:
        """Run simulations for multiple datapoints concurrently."""
        if max_concurrency < 1:
            raise ValueError(f"max_concurrency must be >= 1, got {max_concurrency}")
        semaphore = asyncio.Semaphore(max_concurrency)

        async def run_single(dp: Datapoint) -> SimulationResult:
            async with semaphore:
                return await self._run_with_timeout(
                    dp, max_turns, timeout_per_simulation
                )

        tasks = [run_single(dp) for dp in datapoints]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        final_results: list[SimulationResult] = []
        for i, result in enumerate(results):
            if isinstance(result, SimulationResult):
                final_results.append(result)
            elif isinstance(result, BaseException):
                error_msg = f"{type(result).__name__}: {result}"
                final_results.append(
                    _error_result(
                        error_msg, datapoints[i].persona, datapoints[i].scenario
                    )
                )
            else:
                raise RuntimeError(
                    f"Unexpected result type from gather: {type(result).__name__}"
                )

        return final_results

    async def close(self) -> None:
        """Close and cleanup shared HTTP client."""
        if self._shared_client is not None:
            await self._shared_client.close()
            self._shared_client = None

    # ---------------------------------------------------------------------------
    # Private helpers
    # ---------------------------------------------------------------------------

    async def _get_target_response(self, messages: list[ChatMessage]) -> str:
        if self._target_agent:
            return await self._target_agent.respond(messages)
        if self._target_callback:
            result = self._target_callback(messages)
            if inspect.isawaitable(result):
                return await result
            return result
        raise RuntimeError("No target agent or callback configured")

    @staticmethod
    def _build_criteria_results(
        scenario: Scenario, judgment: Judgment
    ) -> dict[str, bool]:
        return _build_criteria_results(scenario, judgment)

    async def _run_with_timeout(
        self,
        datapoint: Datapoint,
        max_turns: int | None,
        timeout_s: float,
    ) -> SimulationResult:
        if timeout_s <= 0:
            return await self.run(datapoint=datapoint, max_turns=max_turns)

        try:
            result = await asyncio.wait_for(
                self.run(datapoint=datapoint, max_turns=max_turns),
                timeout=timeout_s,
            )
            return result
        except asyncio.TimeoutError:
            # The inner run() never throws (returns error result), so a
            # TimeoutError means asyncio cancelled the task mid-flight.
            # We cannot recover partial messages from the cancelled coroutine,
            # so we return a timeout result. run() itself already populates
            # messages on error paths, matching TS behaviour where possible.
            return SimulationResult(
                messages=[],
                terminated_by=TerminatedBy.timeout,
                reason=f"Simulation timed out after {timeout_s}s",
                goal_achieved=False,
                goal_completion_score=0,
                rules_broken=[],
                turn_count=0,
                turn_metrics=[],
                token_usage=ZERO_USAGE.model_copy(),
                metadata={
                    "persona": datapoint.persona.name,
                    "scenario": datapoint.scenario.name,
                    "timeout": timeout_s,
                },
            )

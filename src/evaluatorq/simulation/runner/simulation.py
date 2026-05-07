"""Simulation runner for orchestrating agent conversations."""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from openai import AsyncOpenAI

from evaluatorq.simulation.agents.judge import JudgeAgent, JudgeAgentConfig
from evaluatorq.simulation.agents.user_simulator import (
    UserSimulatorAgent,
    UserSimulatorAgentConfig,
)
from evaluatorq.simulation.tracing import (
    record_llm_input,
    record_llm_output,
    record_token_usage,
    set_span_attrs,
    with_simulation_span,
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
from evaluatorq.simulation.utils.prompt_builders import (
    build_datapoint_system_prompt,
    build_persona_system_prompt,
    build_scenario_user_context,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from opentelemetry.trace import Span

    from evaluatorq.simulation.agents.base import BaseAgent

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
        user_simulator: BaseAgent | None = None,
        judge: BaseAgent | None = None,
    ) -> None:
        if not target_agent and not target_callback:
            raise ValueError("Must provide either target_agent or target_callback")
        if max_turns < 1:
            raise ValueError(f"max_turns must be >= 1, got {max_turns}")
        if not model.strip():
            raise ValueError("model must be a non-empty string")

        # Validate injected agents early to fail fast
        if user_simulator is not None:
            if not hasattr(user_simulator, "generate_first_message") or \
               not hasattr(user_simulator, "respond_async"):
                raise TypeError(
                    "user_simulator must implement generate_first_message() and respond_async(). "
                    "Use UserSimulatorAgent or a subclass."
                )
        if judge is not None:
            if not hasattr(judge, "evaluate"):
                raise TypeError(
                    "judge must implement evaluate(). "
                    "Use JudgeAgent or a subclass."
                )

        self._target_agent = target_agent
        self._target_callback = target_callback
        self._model = model
        self._max_turns = max_turns
        self._shared_client: AsyncOpenAI | None = None
        self._client_owned: bool = False
        # Injected agents (may be None; resolved lazily in run() when None)
        self._injected_user_simulator: BaseAgent | None = user_simulator
        self._injected_judge: BaseAgent | None = judge

    def _get_shared_client(self) -> AsyncOpenAI:
        if not self._shared_client:
            from evaluatorq.simulation._client import build_simulation_client
            self._shared_client, self._client_owned = build_simulation_client()
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
        # Resolve datapoint
        if datapoint:
            persona = datapoint.persona
            scenario = datapoint.scenario
            first_message = first_message or (datapoint.first_message or None)
        elif not persona or not scenario:
            return _error_result(
                "Must provide either datapoint or both persona and scenario",
                persona,
                scenario,
            )

        effective_max_turns = max_turns or self._max_turns
        messages: list[Message] = []
        turn_metrics_list: list[TurnMetrics] = []
        # Holder so the outer except can read partial token usage from agents
        # created inside _run_inner (mirrors TS getTotalUsage closure).
        usage_holder: dict[str, Callable[[], TokenUsage]] = {}

        try:
            async with with_simulation_span(
                "orq.simulation.run",
                {
                    "orq.simulation.persona": persona.name if persona else None,
                    "orq.simulation.scenario": scenario.name if scenario else None,
                    "orq.simulation.max_turns": effective_max_turns,
                    "orq.simulation.model": self._model,
                },
            ) as run_span:
                return await self._run_inner(
                    persona=persona,
                    scenario=scenario,
                    first_message=first_message,
                    effective_max_turns=effective_max_turns,
                    messages=messages,
                    turn_metrics_list=turn_metrics_list,
                    run_span=run_span,
                    usage_holder=usage_holder,
                )
        except Exception as e:
            logger.error("SimulationRunner.run() failed: %s", e, exc_info=True)
            error_msg = str(e)
            get_total_usage = usage_holder.get("get_total_usage")
            try:
                usage = get_total_usage() if get_total_usage else ZERO_USAGE.model_copy()
            except Exception as usage_err:
                logger.warning("Failed to collect token usage: %s", usage_err)
                usage = ZERO_USAGE.model_copy()

            result = _error_result(error_msg, persona, scenario)
            result.messages = messages
            result.turn_count = sum(1 for m in messages if m.role == "assistant")
            result.turn_metrics = turn_metrics_list
            result.token_usage = usage
            return result

    async def _run_inner(
        self,
        *,
        persona: Persona | None,
        scenario: Scenario | None,
        first_message: str | None,
        effective_max_turns: int,
        messages: list[Message],
        turn_metrics_list: list[TurnMetrics],
        run_span: Span | None,
        usage_holder: dict[str, Callable[[], TokenUsage]],
    ) -> SimulationResult:
        """Inner simulation body (runs inside the orq.simulation.run span)."""
        system_prompt = build_datapoint_system_prompt(persona, scenario)  # pyright: ignore[reportArgumentType]
        client = self._get_shared_client()

        if self._injected_user_simulator is not None:
            user_simulator: UserSimulatorAgent = self._injected_user_simulator  # pyright: ignore[reportAssignmentType]
            # Propagate per-simulation persona/scenario context to the injected
            # agent so it stays grounded in the current datapoint's goal and
            # persona traits (mirrors what the default agent receives via its
            # system_prompt constructor arg).
            if hasattr(user_simulator, "update_context"):
                try:
                    user_simulator.update_context(
                        persona_context=build_persona_system_prompt(persona)  # type: ignore[arg-type]
                        if persona
                        else None,
                        scenario_context=build_scenario_user_context(scenario)  # type: ignore[arg-type]
                        if scenario
                        else None,
                    )
                except Exception as ctx_err:
                    raise RuntimeError(
                        "Injected user_simulator.update_context() failed. "
                        "Ensure it accepts persona_context and scenario_context kwargs."
                    ) from ctx_err
        else:
            user_simulator = UserSimulatorAgent(
                UserSimulatorAgentConfig(
                    model=self._model,
                    client=client,
                    system_prompt=system_prompt,
                )
            )

        if self._injected_judge is not None:
            judge: JudgeAgent = self._injected_judge  # pyright: ignore[reportAssignmentType]
        else:
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

        # Expose to the outer run() except path so it can report partial usage.
        usage_holder["get_total_usage"] = _get_total_usage

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
        if first_message:
            first_msg = first_message
        else:
            async with with_simulation_span(
                "orq.simulation.first_message_generation",
                {
                    "orq.simulation.persona": persona.name if persona else None,
                    "orq.simulation.scenario": scenario.name if scenario else None,
                    "orq.simulation.model": self._model,
                },
            ):
                first_msg = await user_simulator.generate_first_message()
        messages.append(Message(role="user", content=first_msg))

        last_judgment: Judgment | None = None

        for turn in range(effective_max_turns):
            usage_before = _get_total_usage()

            async with with_simulation_span(
                "orq.simulation.turn",
                {
                    "orq.simulation.turn": turn + 1,
                    "orq.simulation.max_turns": effective_max_turns,
                },
            ) as turn_span:
                # 1. Target agent responds
                target_messages = [
                    ChatMessage(role=m.role, content=m.content) for m in messages
                ]
                async with with_simulation_span(
                    "orq.simulation.target_call", None
                ) as target_span:
                    record_llm_input(
                        target_span,
                        [{"role": m.role, "content": m.content} for m in target_messages],
                    )
                    agent_response = await self._get_target_response(target_messages)
                    record_llm_output(target_span, agent_response)
                messages.append(Message(role="assistant", content=agent_response))

                # 2. Judge evaluates
                async with with_simulation_span(
                    "orq.simulation.judge_evaluation", None
                ):
                    judgment = await judge.evaluate(
                        [ChatMessage(role=m.role, content=m.content) for m in messages]
                    )

                turn_metrics_list.append(
                    _build_turn_metrics(turn + 1, judgment, usage_before)
                )
                last_judgment = judgment

                set_span_attrs(
                    turn_span,
                    {
                        "orq.simulation.goal_achieved": judgment.goal_achieved,
                        "orq.simulation.goal_completion_score": judgment.goal_completion_score,
                        "orq.simulation.should_terminate": judgment.should_terminate,
                    },
                )

                # 3. User simulator continues (if not last turn and not terminated)
                if not judgment.should_terminate and turn < effective_max_turns - 1:
                    async with with_simulation_span(
                        "orq.simulation.user_simulator_call", None
                    ):
                        user_response = await user_simulator.respond_async(
                            _invert_roles_for_simulator(messages),
                            llm_purpose="user_simulator",
                        )
                    messages.append(Message(role="user", content=user_response))

            if last_judgment and last_judgment.should_terminate:
                final_usage = _get_total_usage()
                record_token_usage(
                    run_span,
                    prompt_tokens=final_usage.prompt_tokens,
                    completion_tokens=final_usage.completion_tokens,
                    total_tokens=final_usage.total_tokens,
                )
                set_span_attrs(
                    run_span,
                    {
                        "orq.simulation.terminated_by": "judge",
                        "orq.simulation.goal_achieved": last_judgment.goal_achieved,
                        "orq.simulation.turn_count": turn + 1,
                    },
                )
                return SimulationResult(
                    messages=messages,
                    terminated_by=TerminatedBy.judge,
                    reason=last_judgment.reason,
                    goal_achieved=last_judgment.goal_achieved,
                    goal_completion_score=last_judgment.goal_completion_score,
                    rules_broken=last_judgment.rules_broken,
                    turn_count=turn + 1,
                    turn_metrics=turn_metrics_list,
                    token_usage=final_usage,
                    criteria_results=self._build_criteria_results(
                        scenario, last_judgment
                    )
                    if scenario
                    else None,  # type: ignore[arg-type]
                    metadata={
                        "persona": persona.name if persona else None,
                        "scenario": scenario.name if scenario else None,
                    },  # type: ignore[union-attr]
                )

        # Max turns reached
        final_usage = _get_total_usage()
        record_token_usage(
            run_span,
            prompt_tokens=final_usage.prompt_tokens,
            completion_tokens=final_usage.completion_tokens,
            total_tokens=final_usage.total_tokens,
        )
        set_span_attrs(
            run_span,
            {
                "orq.simulation.terminated_by": "max_turns",
                "orq.simulation.goal_achieved": last_judgment.goal_achieved
                if last_judgment
                else False,
                "orq.simulation.turn_count": effective_max_turns,
            },
        )
        return _max_turns_result(
            effective_max_turns,
            messages,
            turn_metrics_list,
            final_usage,
            persona,
            scenario,
            last_judgment,
        )

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
                raise TypeError(
                    f"Unexpected result type from gather: {type(result).__name__}"
                )

        return final_results

    async def close(self) -> None:
        """Close and cleanup shared HTTP client."""
        if self._shared_client is not None and self._client_owned:
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
            return await asyncio.wait_for(
                self.run(datapoint=datapoint, max_turns=max_turns),
                timeout=timeout_s,
            )
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

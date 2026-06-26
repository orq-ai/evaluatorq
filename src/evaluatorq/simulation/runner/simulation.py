"""Simulation runner for orchestrating agent conversations."""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from evaluatorq.common.async_utils import await_maybe
from evaluatorq.common.tracing import record_llm_input, record_llm_output, record_token_usage, set_span_attrs
from evaluatorq.contracts import TokenUsage
from evaluatorq.simulation.agents.judge import JudgeAgent, JudgeAgentConfig
from evaluatorq.simulation.agents.user_simulator import (
    UserSimulatorAgent,
    UserSimulatorAgentConfig,
)
from evaluatorq.simulation.tracing import with_simulation_span
from evaluatorq.simulation.types import (
    DEFAULT_MODEL,
    Judgment,
    Message,
    Persona,
    Scenario,
    SimulationDatapoint,
    SimulationResult,
    TerminatedBy,
    TurnMetrics,
)
from evaluatorq.simulation.utils.prompt_builders import (
    build_datapoint_system_prompt,
    build_persona_system_prompt,
    build_scenario_user_context,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from openai import AsyncOpenAI
    from opentelemetry.trace import Span

    from evaluatorq.contracts import AgentTarget
    from evaluatorq.simulation.agents.base import BaseAgent
    from evaluatorq.simulation.hooks import SimulationHooks

logger = logging.getLogger(__name__)

ZERO_USAGE = TokenUsage()


def _invert_roles_for_simulator(messages: list[Message]) -> list[Message]:
    """Swap roles so the user simulator sees the conversation from its perspective."""
    inverted: list[Message] = []
    for m in messages:
        if m.role == 'user':
            inverted.append(m.model_copy(update={'role': 'assistant'}))
        elif m.role == 'assistant':
            inverted.append(m.model_copy(update={'role': 'user'}))
        else:
            inverted.append(m)
    return inverted


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class SimulationUserSimulator(Protocol):
    """Protocol for user-simulator agents injected into the runner."""

    def update_context(
        self,
        *,
        persona_context: str | None,
        scenario_context: str | None,
    ) -> None: ...

    async def generate_first_message(self) -> str: ...

    async def respond_async(self, messages: list[Message], *, llm_purpose: str | None = None) -> str: ...


@runtime_checkable
class SimulationJudge(Protocol):
    """Protocol for judge agents injected into the runner."""

    async def evaluate(self, messages: list[Message]) -> Judgment: ...


# Method names the injected agents must implement. Used by `_implements` for
# duck-typed validation instead of `isinstance(x, <runtime_checkable Protocol>)`:
# on Python 3.12+ the latter poisons a per-class negative cache, so a single
# isinstance check against an incomplete mock breaks every later mock that
# shares the same class (e.g. MagicMock). Duck typing avoids that entirely.
_USER_SIMULATOR_METHODS = ('generate_first_message', 'respond_async', 'update_context')
_JUDGE_METHODS = ('evaluate',)


def _implements(obj: object, methods: tuple[str, ...]) -> bool:
    """True if `obj` has a callable attribute for every name in `methods`."""
    return all(callable(getattr(obj, name, None)) for name in methods)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _error_result(
    reason: str,
    persona: Persona | None = None,
    scenario: Scenario | None = None,
    *,
    error_type: str | None = None,
) -> SimulationResult:
    # NB: 'unknown' (not None) and no traits/goal — error results predate
    # persona/scenario resolution; deliberately NOT routed through
    # _build_simulation_metadata.
    metadata: dict[str, Any] = {
        'persona': persona.name if persona else 'unknown',
        'scenario': scenario.name if scenario else 'unknown',
        'error': reason,
    }
    if error_type is not None:
        metadata['error_type'] = error_type
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
        metadata=metadata,
    )


def _build_criteria_results(scenario: Scenario, judgment: Judgment) -> dict[str, bool]:
    """Build a human-readable criteria results dict from scenario and judgment."""
    results: dict[str, bool] = {}
    criteria = scenario.criteria or []
    rules_broken = set(judgment.rules_broken)
    for i, criterion in enumerate(criteria):
        criterion_id = f'criteria_{i}'
        results[criterion.description] = criterion_id not in rules_broken
    return results


def _build_criteria_meta(scenario: Scenario, judgment: Judgment) -> list[dict[str, object]]:
    """Id-keyed criteria detail for the report. Stable ids avoid the
    description-collision data loss that ``criteria_results`` (dict-by-description)
    suffers when two criteria share a description."""
    criteria = scenario.criteria or []
    rules_broken = set(judgment.rules_broken)
    meta: list[dict[str, object]] = []
    for i, criterion in enumerate(criteria):
        criterion_id = f'criteria_{i}'
        meta.append({
            'id': criterion_id,
            'description': criterion.description,
            'type': criterion.type,
            'passed': criterion_id not in rules_broken,
        })
    return meta


def _build_simulation_metadata(
    persona: Persona | None,
    scenario: Scenario | None,
    criteria_meta: list[dict[str, object]] | None,
    target_model: str | None,
) -> dict[str, Any]:
    """Single source of truth for SimulationResult.metadata so every
    construction site persists the same fields. Traits/goal/context are
    additive — absent keys keep older results valid."""
    metadata: dict[str, Any] = {
        'persona': persona.name if persona else None,
        'scenario': scenario.name if scenario else None,
        'criteria_meta': criteria_meta,
    }
    if persona is not None:
        metadata['persona_traits'] = {
            'patience': persona.patience,
            'assertiveness': persona.assertiveness,
            'politeness': persona.politeness,
            'technical_level': persona.technical_level,
            'communication_style': persona.communication_style.value,
            'background': persona.background,
        }
    if scenario is not None:
        metadata['scenario_goal'] = scenario.goal
        metadata['scenario_context'] = scenario.context
    if target_model is not None:
        metadata['target_model'] = target_model
    return metadata


def _max_turns_result(
    max_turns: int,
    messages: list[Message],
    turn_metrics: list[TurnMetrics],
    token_usage: TokenUsage,
    persona: Persona | None = None,
    scenario: Scenario | None = None,
    last_judgment: Judgment | None = None,
    target_model: str | None = None,
) -> SimulationResult:
    criteria_results = _build_criteria_results(scenario, last_judgment) if scenario and last_judgment else None
    criteria_meta = _build_criteria_meta(scenario, last_judgment) if scenario and last_judgment else None
    metadata = _build_simulation_metadata(persona, scenario, criteria_meta, target_model)
    return SimulationResult(
        messages=messages,
        terminated_by=TerminatedBy.max_turns,
        reason=f'Maximum turns ({max_turns}) reached',
        goal_achieved=last_judgment.goal_achieved if last_judgment else False,
        goal_completion_score=last_judgment.goal_completion_score if last_judgment else 0,
        rules_broken=last_judgment.rules_broken if last_judgment else [],
        turn_count=max_turns,
        turn_metrics=turn_metrics,
        token_usage=token_usage,
        criteria_results=criteria_results,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# SimulationRunner
# ---------------------------------------------------------------------------


class SimulationRunner:
    """Orchestrates multi-turn conversations between user simulator, target agent, and judge."""

    def __init__(
        self,
        *,
        target_agent: AgentTarget | None = None,
        target_callback: Callable[[list[Message]], str | Awaitable[str]] | None = None,
        model: str = DEFAULT_MODEL,
        max_turns: int = 10,
        user_simulator: BaseAgent | None = None,
        judge: BaseAgent | None = None,
        hooks: SimulationHooks | None = None,
        llm_client: AsyncOpenAI | None = None,
    ) -> None:
        if not target_agent and not target_callback:
            raise ValueError('Must provide either target_agent or target_callback')
        if max_turns < 1:
            raise ValueError(f'max_turns must be >= 1, got {max_turns}')
        if not model.strip():
            raise ValueError('model must be a non-empty string')

        # Validate injected agents early to fail fast
        if user_simulator is not None and not _implements(user_simulator, _USER_SIMULATOR_METHODS):
            raise TypeError(
                'user_simulator must implement generate_first_message(), respond_async(), '
                'and update_context(). Use UserSimulatorAgent or a subclass.'
            )
        if judge is not None and not _implements(judge, _JUDGE_METHODS):
            raise TypeError('judge must implement evaluate(). Use JudgeAgent or a subclass.')

        self._target_agent = target_agent
        self._target_callback = target_callback
        self._model = model
        self._max_turns = max_turns
        self._shared_client: AsyncOpenAI | None = llm_client
        self._client_owned: bool = False
        # Injected agents (may be None; resolved lazily in run() when None)
        self._injected_user_simulator: BaseAgent | None = user_simulator
        self._injected_judge: BaseAgent | None = judge

        from evaluatorq.common.async_utils import warn_if_sync_hooks
        from evaluatorq.simulation.hooks import DefaultHooks

        # Single resolution: _simulate_core passes an already-resolved instance,
        # so a RichHooks is never re-instantiated (no double progress display).
        self._hooks: SimulationHooks = hooks or DefaultHooks()
        # Single choke point for both entry paths (simulate() and direct
        # SimulationRunner use): nudge sync hooks toward async exactly once.
        warn_if_sync_hooks(
            self._hooks,
            (
                'on_confirm',
                'on_run_start',
                'on_datapoint_start',
                'on_turn_complete',
                'on_datapoint_complete',
                'on_evaluator_complete',
                'on_datapoint_error',
                'on_run_complete',
            ),
        )

    def _get_shared_client(self) -> AsyncOpenAI:
        if not self._shared_client:
            from evaluatorq.openresponses.client import build_simulation_client

            self._shared_client, self._client_owned = build_simulation_client()
        return self._shared_client

    async def run(
        self,
        *,
        persona: Persona | None = None,
        scenario: Scenario | None = None,
        datapoint: SimulationDatapoint | None = None,
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
                'Must provide either datapoint or both persona and scenario',
                persona,
                scenario,
            )

        datapoint_id = datapoint.id if datapoint else ''

        effective_max_turns = max_turns or self._max_turns
        messages: list[Message] = []
        turn_metrics_list: list[TurnMetrics] = []
        # Holder so the outer except can read partial token usage from agents
        # created inside _run_inner (mirrors TS getTotalUsage closure).
        usage_holder: dict[str, Callable[[], TokenUsage]] = {}

        try:
            async with with_simulation_span(
                'orq.simulation.run',
                {
                    'orq.simulation.persona': persona.name if persona else None,
                    'orq.simulation.scenario': scenario.name if scenario else None,
                    'orq.simulation.max_turns': effective_max_turns,
                    'orq.simulation.model': self._model,
                },
            ) as run_span:
                try:
                    return await self._run_inner(
                        persona=persona,
                        scenario=scenario,
                        datapoint_id=datapoint_id,
                        first_message=first_message,
                        effective_max_turns=effective_max_turns,
                        messages=messages,
                        turn_metrics_list=turn_metrics_list,
                        run_span=run_span,
                        usage_holder=usage_holder,
                    )
                except BaseException:
                    get_total_usage = usage_holder.get('get_total_usage')
                    try:
                        usage = get_total_usage() if get_total_usage else ZERO_USAGE.model_copy()
                    except Exception as usage_err:
                        logger.error(
                            'Failed to collect token usage during error path: %s',
                            usage_err,
                            exc_info=True,
                        )
                        usage = ZERO_USAGE.model_copy()
                    record_token_usage(
                        run_span,
                        prompt_tokens=usage.prompt_tokens,
                        completion_tokens=usage.completion_tokens,
                        total_tokens=usage.total_tokens,
                    )
                    set_span_attrs(
                        run_span,
                        {
                            'orq.simulation.terminated_by': 'error',
                            'orq.simulation.goal_achieved': False,
                            'orq.simulation.turn_count': sum(1 for m in messages if m.role == 'assistant'),
                        },
                    )
                    raise
        except Exception as e:
            logger.error('SimulationRunner.run() failed: %s', e, exc_info=True)
            error_msg = str(e)
            error_type = type(e).__name__
            get_total_usage = usage_holder.get('get_total_usage')
            try:
                usage = get_total_usage() if get_total_usage else ZERO_USAGE.model_copy()
            except Exception as usage_err:
                logger.error(
                    'Failed to collect token usage during error path: %s',
                    usage_err,
                    exc_info=True,
                )
                usage = ZERO_USAGE.model_copy()

            result = _error_result(error_msg, persona, scenario, error_type=error_type)
            result.messages = messages
            result.turn_count = sum(1 for m in messages if m.role == 'assistant')
            result.turn_metrics = turn_metrics_list
            result.token_usage = usage
            return result

    async def _run_inner(
        self,
        *,
        persona: Persona | None,
        scenario: Scenario | None,
        datapoint_id: str,
        first_message: str | None,
        effective_max_turns: int,
        messages: list[Message],
        turn_metrics_list: list[TurnMetrics],
        run_span: Span | None,
        usage_holder: dict[str, Callable[[], TokenUsage]],
    ) -> SimulationResult:
        """Inner simulation body (runs inside the orq.simulation.run span)."""
        system_prompt = build_datapoint_system_prompt(persona, scenario)  # pyright: ignore[reportArgumentType]
        client: AsyncOpenAI | None = None

        if self._injected_user_simulator is not None:
            import copy

            # Shallow-copy isolates _custom_system_prompt mutations from concurrent
            # run_batch tasks. Reset _usage to a fresh TokenUsage — shallow copy
            # keeps the same reference, which would cross-contaminate per-sim counts.
            user_simulator: UserSimulatorAgent = copy.copy(self._injected_user_simulator)  # pyright: ignore[reportAssignmentType]
            user_simulator.reset_usage()
            if _implements(user_simulator, _USER_SIMULATOR_METHODS):
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
                        'Injected user_simulator.update_context() failed. '
                        'Ensure it accepts persona_context and scenario_context kwargs.'
                    ) from ctx_err
        else:
            if client is None:
                client = self._get_shared_client()
            user_simulator = UserSimulatorAgent(
                UserSimulatorAgentConfig(
                    model=self._model,
                    client=client,
                    system_prompt=system_prompt,
                )
            )

        if self._injected_judge is not None:
            import copy

            # Isolate per-sim state — see user_simulator comment above.
            judge: JudgeAgent = copy.copy(self._injected_judge)  # pyright: ignore[reportAssignmentType]
            judge.reset_usage()
        else:
            if client is None:
                client = self._get_shared_client()
            judge = JudgeAgent(
                JudgeAgentConfig(
                    model=self._model,
                    client=client,
                    goal=scenario.goal if scenario else '',
                    criteria=list(scenario.criteria) if scenario and scenario.criteria else [],
                    ground_truth=scenario.ground_truth or '' if scenario else '',
                )
            )

        target_usage_acc: dict[str, TokenUsage] = {'acc': ZERO_USAGE.model_copy()}
        # Lazily captured on the first turn that reports a model identity.
        # NEVER set this from self._model — that is the user-simulator/judge model.
        target_model_holder: dict[str, str | None] = {'model': None}

        def _get_total_usage() -> TokenUsage:
            # __add__ carries cached/reasoning/cost across all three sources.
            return user_simulator.get_usage() + judge.get_usage() + target_usage_acc['acc']

        # Expose to the outer run() except path so it can report partial usage.
        usage_holder['get_total_usage'] = _get_total_usage

        def _build_turn_metrics(turn_num: int, judgment: Judgment, usage_before: TokenUsage) -> TurnMetrics:
            usage_after = _get_total_usage()
            return TurnMetrics(
                turn_number=turn_num,
                token_usage=usage_after - usage_before,
                response_quality=judgment.response_quality,
                hallucination_risk=judgment.hallucination_risk,
                tone_appropriateness=judgment.tone_appropriateness,
                factual_accuracy=judgment.factual_accuracy,
                judge_reason=judgment.reason,
            )

        if first_message:
            first_msg = first_message
        else:
            async with with_simulation_span(
                'orq.simulation.first_message_generation',
                {
                    'orq.simulation.persona': persona.name if persona else None,
                    'orq.simulation.scenario': scenario.name if scenario else None,
                    'orq.simulation.model': self._model,
                },
            ):
                first_msg = await user_simulator.generate_first_message()
        messages.append(Message(role='user', content=first_msg))

        last_judgment: Judgment | None = None

        for turn in range(effective_max_turns):
            usage_before = _get_total_usage()

            async with with_simulation_span(
                'orq.simulation.turn',
                {
                    'orq.simulation.turn': turn + 1,
                    'orq.simulation.max_turns': effective_max_turns,
                },
            ) as turn_span:
                async with with_simulation_span('orq.simulation.target_call', None) as target_span:
                    record_llm_input(
                        target_span,
                        [{'role': m.role, 'content': m.content or ''} for m in messages],
                    )
                    agent_response_text, agent_response_usage, agent_response_model = await self._get_target_response(
                        messages
                    )
                    if agent_response_usage is not None:
                        target_usage_acc['acc'] = target_usage_acc['acc'] + agent_response_usage
                    # Capture the first non-None model the target reports; it
                    # should be stable across turns for a given target.
                    if agent_response_model is not None and target_model_holder['model'] is None:
                        target_model_holder['model'] = agent_response_model
                    record_llm_output(target_span, agent_response_text)
                messages.append(Message(role='assistant', content=agent_response_text))

                async with with_simulation_span('orq.simulation.judge_evaluation', None):
                    judgment = await judge.evaluate(messages)

                turn_metrics_list.append(_build_turn_metrics(turn + 1, judgment, usage_before))
                try:
                    await await_maybe(self._hooks.on_turn_complete(datapoint_id, turn_metrics_list[-1]))
                except Exception:
                    logger.warning(
                        'on_turn_complete hook raised for datapoint %s; ignoring',
                        datapoint_id,
                        exc_info=True,
                    )
                last_judgment = judgment

                set_span_attrs(
                    turn_span,
                    {
                        'orq.simulation.goal_achieved': judgment.goal_achieved,
                        'orq.simulation.goal_completion_score': judgment.goal_completion_score,
                        'orq.simulation.should_terminate': judgment.should_terminate,
                    },
                )

                if not judgment.should_terminate and turn < effective_max_turns - 1:
                    async with with_simulation_span('orq.simulation.user_simulator_call', None):
                        user_response = await user_simulator.respond_async(
                            _invert_roles_for_simulator(messages),
                            llm_purpose='user_simulator',
                        )
                    messages.append(Message(role='user', content=user_response))

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
                        'orq.simulation.terminated_by': 'judge',
                        'orq.simulation.goal_achieved': last_judgment.goal_achieved,
                        'orq.simulation.turn_count': turn + 1,
                    },
                )
                judge_metadata = _build_simulation_metadata(
                    persona,
                    scenario,
                    _build_criteria_meta(scenario, last_judgment) if scenario else None,
                    target_model_holder['model'],
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
                    criteria_results=self._build_criteria_results(scenario, last_judgment) if scenario else None,  # type: ignore[arg-type]
                    metadata=judge_metadata,  # type: ignore[union-attr]
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
                'orq.simulation.terminated_by': 'max_turns',
                'orq.simulation.goal_achieved': last_judgment.goal_achieved if last_judgment else False,
                'orq.simulation.turn_count': effective_max_turns,
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
            target_model=target_model_holder['model'],
        )

    async def run_batch(
        self,
        datapoints: list[SimulationDatapoint],
        *,
        max_turns: int | None = None,
        timeout_per_simulation: float = 300.0,
        max_concurrency: int = 10,
    ) -> list[SimulationResult]:
        """Run simulations for multiple datapoints concurrently."""
        if max_concurrency < 1:
            raise ValueError(f'max_concurrency must be >= 1, got {max_concurrency}')
        semaphore = asyncio.Semaphore(max_concurrency)

        async def run_single(dp: SimulationDatapoint) -> SimulationResult:
            async with semaphore:
                await await_maybe(self._hooks.on_datapoint_start(dp))
                return await self._run_with_timeout(dp, max_turns, timeout_per_simulation)

        tasks = [run_single(dp) for dp in datapoints]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        final_results: list[SimulationResult] = []
        for i, result in enumerate(results):
            dp = datapoints[i]
            if isinstance(result, SimulationResult):
                result.metadata['datapoint_id'] = dp.id
                # Append before firing the unguarded on_datapoint_error so a
                # raising hook doesn't drop the result from final_results —
                # matches the BaseException branch's ordering below.
                final_results.append(result)
                if result.terminated_by in (TerminatedBy.error, TerminatedBy.timeout):
                    # The original exception was already swallowed into the error
                    # result by run(); its type is in metadata["error_type"]. Hooks
                    # get a reconstructed RuntimeError here (real exc only on the
                    # BaseException branch below).
                    reason = result.metadata.get('error') or result.reason
                    await await_maybe(self._hooks.on_datapoint_error(dp, RuntimeError(reason)))
                await await_maybe(self._hooks.on_datapoint_complete(result))
            elif isinstance(result, BaseException):
                error_msg = f'{type(result).__name__}: {result}'
                err = _error_result(error_msg, dp.persona, dp.scenario)
                err.metadata['datapoint_id'] = dp.id
                final_results.append(err)
                await await_maybe(self._hooks.on_datapoint_error(dp, result))
                await await_maybe(self._hooks.on_datapoint_complete(err))
            else:
                raise TypeError(f'Unexpected result type from gather: {type(result).__name__}')

        return final_results

    async def close(self) -> None:
        """Close and cleanup shared HTTP client."""
        if self._shared_client is not None and self._client_owned:
            await self._shared_client.close()
            self._shared_client = None

    # ---------------------------------------------------------------------------
    # Private helpers
    # ---------------------------------------------------------------------------

    async def _get_target_response(self, messages: list[Message]) -> tuple[str, TokenUsage | None, str | None]:
        """Return (text, usage, target_model).

        ``target_model`` is the model identifier reported by the target itself
        (via ``AgentResponse.model``).  It is ``None`` for plain callbacks,
        which may call any provider and have no way to report a model.
        NEVER substitute ``self._model`` here — that is the user-simulator /
        judge model, not the evaluated target.
        """
        if self._target_agent:
            resp = await self._target_agent.respond(messages)
            return resp.text, resp.usage, resp.model
        if self._target_callback:
            result = self._target_callback(messages)
            if inspect.isawaitable(result):
                return await result, None, None
            return result, None, None
        raise RuntimeError('No target agent or callback configured')

    @staticmethod
    def _build_criteria_results(scenario: Scenario, judgment: Judgment) -> dict[str, bool]:
        return _build_criteria_results(scenario, judgment)

    async def _run_with_timeout(
        self,
        datapoint: SimulationDatapoint,
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
            logger.warning(
                'Simulation for datapoint %s timed out after %ss; returning timeout result',
                datapoint.id,
                timeout_s,
            )
            # The inner run() never throws (returns error result), so a
            # TimeoutError means asyncio cancelled the task mid-flight.
            # We cannot recover partial messages from the cancelled coroutine,
            # so we return a timeout result. run() itself already populates
            # messages on error paths, matching TS behaviour where possible.
            return SimulationResult(
                messages=[],
                terminated_by=TerminatedBy.timeout,
                reason=f'Simulation timed out after {timeout_s}s',
                goal_achieved=False,
                goal_completion_score=0,
                rules_broken=[],
                turn_count=0,
                turn_metrics=[],
                token_usage=ZERO_USAGE.model_copy(),
                metadata={
                    'persona': datapoint.persona.name,
                    'scenario': datapoint.scenario.name,
                    'timeout': timeout_s,
                },
            )

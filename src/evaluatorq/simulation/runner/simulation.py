"""Simulation runner for orchestrating agent conversations."""

from __future__ import annotations

import asyncio
import inspect
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

from evaluatorq.common.async_utils import await_maybe
from evaluatorq.common.target_call import TargetCallResult, call_target_with_retry, default_map_error
from evaluatorq.common.thread_context import conversation_thread, evaluatorq_pipeline
from evaluatorq.common.tracing import record_llm_input, record_llm_output, set_span_attrs
from evaluatorq.contracts import ResponseTrace, TokenUsage, render_tool_call
from evaluatorq.integrations.callable_integration import CallableTarget
from evaluatorq.simulation.agents.judge import JudgeAgent, JudgeAgentConfig
from evaluatorq.simulation.agents.user_simulator import (
    UserSimulatorAgent,
    UserSimulatorAgentConfig,
)
from evaluatorq.simulation.tracing import span_message_text, with_simulation_span
from evaluatorq.simulation.types import (
    DEFAULT_MODEL,
    CriteriaMeta,
    CriterionVerdict,
    Judgment,
    Message,
    Persona,
    Scenario,
    SimulationDatapoint,
    SimulationResult,
    TerminatedBy,
    TurnMetrics,
    criterion_id_for,
)
from evaluatorq.simulation.utils.prompt_builders import (
    build_datapoint_system_prompt,
    build_persona_system_prompt,
    build_scenario_user_context,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping

    from openai import AsyncOpenAI
    from opentelemetry.trace import Span

    from evaluatorq.contracts import AgentResponse, AgentTarget
    from evaluatorq.integrations.callable_integration.target import AgentCallable
    from evaluatorq.simulation.agents.base import BaseAgent
    from evaluatorq.simulation.hooks import SimulationHooks

logger = logging.getLogger(__name__)

ZERO_USAGE = TokenUsage()


@dataclass
class RunSinks:
    """Mutable state that survives cancellation of a simulation run.

    The timeout wrapper owns one instance per conversation and passes it through
    the private ``run`` seam.  Keeping all partial state together prevents an
    outer timeout from reconstructing a result from three independently shared
    containers.
    """

    messages: list[Message] = field(default_factory=list)
    turn_metrics: list[TurnMetrics] = field(default_factory=list)
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    target_token_usage: TokenUsage = field(default_factory=TokenUsage)
    token_usage_known: bool = True
    criteria_verified: bool | None = None
    criteria_results: dict[str, bool] | None = None
    criteria_meta: list[CriteriaMeta] | None = None
    rules_broken: list[str] = field(default_factory=list)
    goal_achieved: bool = False
    goal_completion_score: float = 0.0
    thread_id: str | None = None
    response_traces: list[ResponseTrace] = field(default_factory=list)
    target_model: str | None = None


def _invert_roles_for_simulator(messages: list[Message]) -> list[Message]:
    """Swap roles so the user simulator sees the conversation from its perspective.

    The target's tool calls and their ``role='tool'`` results are dropped: inverting
    turns an ``assistant`` row into a ``user`` row, and a ``user`` row carrying
    ``tool_calls`` followed by ``tool`` rows is rejected by the provider
    ("messages with role 'tool' must be a response to a preceeding message with
    'tool_calls'"). The simulated user only ever saw the assistant text anyway.
    """
    inverted: list[Message] = []
    for m in messages:
        if m.role == 'tool':
            continue
        if m.role == 'user':
            inverted.append(m.model_copy(update={'role': 'assistant'}))
        elif m.role == 'assistant':
            inverted.append(m.model_copy(update={'role': 'user', 'tool_calls': None}))
        else:
            inverted.append(m)
    return inverted


def build_assistant_message(response: AgentResponse) -> list[Message]:
    """Build transcript rows for a target's ``AgentResponse``.

    Carries `response.tool_calls` into `Message.tool_calls` so the judge sees what
    the target actually did, not just any text it produced alongside it — a
    target that resolves a request purely through tool calls (no text) previously
    left the judge looking at an empty assistant turn. Completed tool calls are
    followed by their separate ``role='tool'`` rows so both provider serializers
    receive a valid pair. Calls without results are dropped by ``render_tool_call``.
    """
    rendered_tool_calls = [
        rendered
        for item in response.tool_calls
        if (rendered := render_tool_call(item, warn=logger.warning)) is not None
    ]
    tool_calls = [tool_call for tool_call, _ in rendered_tool_calls]
    text = response.text
    if not text and tool_calls:
        logger.warning(
            'Assistant turn has no text but %d tool call(s); transcript keeps the tool calls',
            len(tool_calls),
        )
    return [
        Message(role='assistant', content=text, tool_calls=tool_calls or None),
        *(tool_message for _, tool_message in rendered_tool_calls),
    ]


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
# Optional, not part of the judge contract: a judge that can be told which criteria
# are already settled drops them from its per-turn audit. See JudgeAgent.mark_settled.
_SETTLEABLE_JUDGE_METHODS = ('mark_settled',)
# Optional, not part of the judge contract: a judge that can be told the scenario's
# goal/criteria/ground_truth gets them per simulation. See JudgeAgent.update_context.
_CONTEXTUAL_JUDGE_METHODS = ('update_context',)


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


class _CriteriaTracker:
    """Folds the judge's per-turn occurrence audit into one run-level verdict.

    Occurrence is sticky: something that appeared on turn 2 still happened when the
    run ends on turn 6, so ``occurred`` only ever flips False→True. Pass/fail is
    then a pure function of occurrence and `Criterion.type` — ``must_happen``
    passes when it occurred, ``must_not_happen`` when it did not. Reading the final
    turn's judgment alone (the old behaviour) dropped any violation the judge
    reported earlier.
    """

    def __init__(self, scenario: Scenario | None) -> None:
        self._criteria = list(scenario.criteria or []) if scenario else []
        self._scenario_name = scenario.name if scenario else '<none>'
        # Nothing has been observed yet. A criterion the judge never mentions keeps
        # this default, which is the honest reading for both types: a must_happen
        # nobody witnessed fails, a must_not_happen nobody witnessed passes.
        self._occurred: dict[str, bool] = {criterion_id_for(i): False for i in range(len(self._criteria))}
        self._seen: set[str] = set()
        # Quote from the turn where occurrence first flipped, so the report can show
        # WHY a criterion is marked occurred rather than only that it is.
        self._evidence: dict[str, str] = {}
        self._any_audit = False

    @property
    def verified(self) -> bool:
        """Whether the run's criteria verdicts actually rest on the judge's audit.

        ``False`` means the audit never arrived for any turn, so ``rules_broken``
        fell back to the judge's free-text list — which cannot fail a
        ``must_happen`` criterion, the exact defect RES-1308 is about. A run in
        that state is *unknown*, not passing, and `criteria_met_scorer` scores it
        0.0. A scenario with no criteria has nothing to verify, so it is ``True``.
        """
        return not self._criteria or self._any_audit

    @property
    def audited_ids(self) -> frozenset[str]:
        """Ids the judge returned an occurrence verdict for on at least one turn.

        A criterion outside this set was never observed, so its verdict is the
        not-observed default rather than something the judge actually reported.
        Reports need the difference: a ``must_happen`` the judge confirmed never
        occurred and one it silently skipped both render as failed.
        """
        return frozenset(self._seen)

    @property
    def settled_ids(self) -> frozenset[str]:
        """Ids whose occurrence is already ``True`` and therefore final.

        Occurrence is sticky, so a later turn cannot change these — re-auditing
        them buys nothing and costs a payload entry per turn. The judge is told to
        skip them; see `JudgeAgent.mark_settled`.
        """
        return frozenset(cid for cid, occurred in self._occurred.items() if occurred)

    @property
    def evidence(self) -> dict[str, str]:
        """Quote captured at the turn where each criterion's occurrence first flipped.

        Only ids that actually occurred appear, and only when the judge supplied a
        quote — an entry here is evidence, never an empty placeholder.
        """
        return dict(self._evidence)

    def observe(self, verdicts: list[CriterionVerdict] | None) -> None:
        """Fold one turn's occurrence audit in.

        Takes the verdict list, not the whole `Judgment`: the audit is the only
        part of a judgment this tracker may read, and `Judgment.rules_broken` is
        precisely the channel it exists to stop trusting.

        An empty list is **not** an audit for `verified` purposes. It legitimately
        means "everything is settled, nothing left to report" — but only once
        something *has* been reported, which by definition already set the flag. A
        run whose every turn returned ``[]`` audited nothing, and the judge warns
        about that at the point it happens (`JudgeAgent._parse_judgment`).
        Likewise a payload whose every id is unknown to this scenario is a
        misattributed audit, not a verified one: the flag is set below, after the
        filter, not before it.
        """
        if not verdicts:
            return
        unknown = [v.criterion_id for v in verdicts if v.criterion_id not in self._occurred]
        if unknown:
            # A valid-shaped entry for a nonexistent id is not counted as malformed
            # by the judge's parser, so without this it would vanish in silence —
            # and if it is an off-by-one on the whole payload, the criteria it was
            # meant for look simply unaudited.
            logger.warning(
                'Judge audited unknown criterion id(s) %s on scenario %r, which defines only %s; discarding them.',
                ', '.join(sorted(unknown)),
                self._scenario_name,
                ', '.join(sorted(self._occurred)) or '<none>',
            )
        for verdict in verdicts:
            cid = verdict.criterion_id
            if cid not in self._occurred:
                continue
            self._any_audit = True
            self._seen.add(cid)
            # Sticky: the first turn that saw it is the one whose quote is kept, so
            # later turns cannot overwrite the evidence with a weaker restatement.
            if verdict.occurred and not self._occurred[cid]:
                if verdict.evidence:
                    self._evidence[cid] = verdict.evidence
                self._occurred[cid] = True

    def _passed(self, index: int) -> bool:
        occurred = self._occurred[criterion_id_for(index)]
        return occurred if self._criteria[index].type == 'must_happen' else not occurred

    @property
    def broken_ids(self) -> list[str]:
        """Run-level ``rules_broken``: every criterion the folded audit failed.

        The single computation of that list, so `resolve` (normal termination) and
        the target-failure branch cannot disagree about what the audit said.
        """
        return [criterion_id_for(i) for i in range(len(self._criteria)) if not self._passed(i)]

    @property
    def unaudited_ids(self) -> frozenset[str]:
        """Ids the judge never returned an occurrence verdict for.

        Their verdict is the not-observed default, not something the judge said.
        The complement of `audited_ids` over the scenario's criteria.
        """
        return frozenset(cid for cid in self._occurred if cid not in self._seen)

    @property
    def unconfirmed_ids(self) -> frozenset[str]:
        """``must_happen`` ids whose occurrence never flipped to ``True``.

        On a run that ended normally these are genuine failures — the conversation
        had its full chance and the behaviour never appeared. On a run **cut short**
        (target failure/timeout) they are not knowledge at all: "it hadn't happened
        yet" is not "the judge confirmed it never happened", so `broken_ids` would
        report a failure nobody observed. See `confirmed_broken_ids`.
        """
        return frozenset(
            criterion_id_for(i)
            for i, criterion in enumerate(self._criteria)
            if criterion.type == 'must_happen' and not self._occurred[criterion_id_for(i)]
        )

    @property
    def confirmed_broken_ids(self) -> list[str]:
        """`broken_ids` restricted to failures the audit actually **observed**.

        For a run cut short before it could finish: a ``must_not_happen`` the judge
        saw violated is knowledge and stays failed, while a ``must_happen`` that had
        simply not happened yet drops out (it is reported unknown instead, via
        `unconfirmed_ids` being withheld from the meta's audited set).
        """
        unconfirmed = self.unconfirmed_ids
        return [cid for cid in self.broken_ids if cid not in unconfirmed]

    def resolve(self, judgment: Judgment) -> Judgment:
        """Return ``judgment`` with ``rules_broken`` replaced by the run-level verdict.

        Everything downstream (``criteria_results``, ``criteria_meta``,
        ``SimulationResult.rules_broken``, the ``criteria_met`` scorer) derives from
        this one list, so they cannot disagree.

        Once any audit has arrived, it is the **only** input: the built-in judge's
        tools no longer ask for a free-text ``rules_broken`` at all, and an
        incoming one (from a custom judge that fills the field itself) is dropped
        rather than merged. A criterion the audit skipped keeps its not-observed
        default, which is the honest reading — rescuing it from free text would put
        the channel this design exists to stop trusting back in charge of exactly
        the criteria the audit has no answer for.

        The one exception is a run where **no** audit ever arrived: there is
        nothing to prefer it to, so the judgment passes through untouched and the
        run is marked unverified via `verified`.
        """
        if not self._criteria:
            return judgment
        if not self._any_audit:
            logger.warning(
                'Judge returned no criteria audit for any turn of scenario %r; falling back to its '
                'rules_broken list, which cannot fail a must_happen criterion. Treat these %d '
                'criteria results as unverified.',
                self._scenario_name,
                len(self._criteria),
            )
            return judgment
        unaudited = self.unaudited_ids
        if unaudited:
            logger.warning(
                'Judge never reported an occurrence verdict for %s on scenario %r; treating them as '
                'not observed (must_happen=failed, must_not_happen=passed).',
                ', '.join(sorted(unaudited)),
                self._scenario_name,
            )
        return judgment.model_copy(update={'rules_broken': self.broken_ids})


def _build_criteria_results(scenario: Scenario, judgment: Judgment) -> dict[str, bool]:
    """Build a human-readable criteria results dict from scenario and judgment."""
    results: dict[str, bool] = {}
    criteria = scenario.criteria or []
    rules_broken = set(judgment.rules_broken)
    for i, criterion in enumerate(criteria):
        criterion_id = criterion_id_for(i)
        results[criterion.description] = criterion_id not in rules_broken
    return results


def _build_criteria_meta(
    scenario: Scenario,
    judgment: Judgment,
    audited_ids: frozenset[str] | None = None,
    evidence: Mapping[str, str] | None = None,
) -> list[CriteriaMeta]:
    """Id-keyed criteria detail for the report. Stable ids avoid the
    description-collision data loss that ``criteria_results`` (dict-by-description)
    suffers when two criteria share a description.

    ``audited`` says whether the judge actually reported an occurrence verdict for
    that criterion. Without it a ``must_happen`` the judge confirmed never happened
    and one it silently skipped are the same red row. ``None`` when no tracker was
    available, which reads as unknown rather than as audited.
    """
    criteria = scenario.criteria or []
    rules_broken = set(judgment.rules_broken)
    meta: list[CriteriaMeta] = []
    for i, criterion in enumerate(criteria):
        criterion_id = criterion_id_for(i)
        meta.append(
            CriteriaMeta(
                id=criterion_id,
                description=criterion.description,
                type=criterion.type,
                passed=criterion_id not in rules_broken,
                audited=criterion_id in audited_ids if audited_ids is not None else None,
                evidence=evidence.get(criterion_id, '') if evidence is not None else None,
            )
        )
    return meta


def _build_simulation_metadata(
    persona: Persona | None,
    scenario: Scenario | None,
    criteria_meta: list[CriteriaMeta] | None,
    target_model: str | None,
) -> dict[str, Any]:
    """Single source of truth for SimulationResult.metadata so every
    construction site persists the same fields. Traits/goal/context are
    additive — absent keys keep older results valid."""
    metadata: dict[str, Any] = {
        'persona': persona.name if persona else None,
        'scenario': scenario.name if scenario else None,
        'criteria_meta': [entry.model_dump(mode='json') for entry in criteria_meta] if criteria_meta else criteria_meta,
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


def _partial_result(
    sinks: RunSinks,
    *,
    persona: Persona | None,
    scenario: Scenario | None,
    terminated_by: TerminatedBy,
    reason: str,
    error_type: str,
    timeout_s: float | None = None,
) -> SimulationResult:
    """Build an error/timeout result from the one authoritative sink object."""
    metadata = _build_simulation_metadata(persona, scenario, sinks.criteria_meta, sinks.target_model)
    metadata['error'] = reason
    metadata['error_type'] = error_type
    if timeout_s is not None:
        metadata['timeout'] = timeout_s
    if not sinks.token_usage_known:
        metadata['token_usage_unknown'] = True
    return SimulationResult(
        messages=sinks.messages,
        terminated_by=terminated_by,
        reason=reason,
        goal_achieved=sinks.goal_achieved,
        goal_completion_score=sinks.goal_completion_score,
        rules_broken=sinks.rules_broken,
        turn_count=sum(1 for m in sinks.messages if m.role == 'assistant'),
        turn_metrics=sinks.turn_metrics,
        token_usage=sinks.token_usage,
        token_usage_known=sinks.token_usage_known,
        criteria_results=sinks.criteria_results,
        criteria_verified=sinks.criteria_verified,
        metadata=metadata,
        thread_id=sinks.thread_id,
        response_traces=sinks.response_traces,
    )


def _max_turns_result(
    max_turns: int,
    messages: list[Message],
    turn_metrics: list[TurnMetrics],
    token_usage: TokenUsage,
    persona: Persona | None = None,
    scenario: Scenario | None = None,
    last_judgment: Judgment | None = None,
    target_model: str | None = None,
    *,
    criteria_verified: bool | None = None,
    token_usage_known: bool = True,
    audited_ids: frozenset[str] | None = None,
    evidence: Mapping[str, str] | None = None,
) -> SimulationResult:
    criteria_results = _build_criteria_results(scenario, last_judgment) if scenario and last_judgment else None
    criteria_meta = (
        _build_criteria_meta(scenario, last_judgment, audited_ids, evidence) if scenario and last_judgment else None
    )
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
        token_usage_known=token_usage_known,
        criteria_results=criteria_results,
        criteria_verified=criteria_verified,
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
        target: Callable[[list[Message]], str | Awaitable[str] | Awaitable[AgentResponse]] | None = None,
        model: str = DEFAULT_MODEL,
        max_turns: int = 10,
        target_agent_timeout_ms: int = 240_000,
        max_target_retries: int = 2,
        user_simulator: BaseAgent | None = None,
        judge: BaseAgent | None = None,
        hooks: SimulationHooks | None = None,
        llm_client: AsyncOpenAI | None = None,
    ) -> None:
        if not target_agent and not target:
            raise ValueError('Must provide either target_agent or target')
        if max_turns < 1:
            raise ValueError(f'max_turns must be >= 1, got {max_turns}')
        if max_target_retries < 0:
            raise ValueError(f'max_target_retries must be >= 0, got {max_target_retries}')
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
        self._target = target
        # Route both target flavours through the shared retry helper by wrapping a
        # plain callback in the existing CallableTarget adapter (str->AgentResponse
        # coercion + sync/async dispatch handled there — no bespoke adapter).
        self._effective_target: AgentTarget | None = target_agent or (
            CallableTarget(cast('AgentCallable', target)) if target is not None else None
        )
        # Per-conversation clones minted by run() (see _new_conversation_target),
        # retained so close() can release any resources they own (e.g. an
        # OrqResponsesTarget clone builds its own HTTP client).
        self._spawned_targets: list[AgentTarget] = []
        self._target_agent_timeout_ms = target_agent_timeout_ms
        self._max_target_retries = max_target_retries
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
        """Return the generation client; ``with_retry`` owns retrying calls."""
        if not self._shared_client:
            from evaluatorq.openresponses.client import build_simulation_client

            self._shared_client, self._client_owned = build_simulation_client(max_retries=0)
        return self._shared_client

    async def run(
        self,
        *,
        persona: Persona | None = None,
        scenario: Scenario | None = None,
        datapoint: SimulationDatapoint | None = None,
        max_turns: int | None = None,
        first_message: str | None = None,
        thread_id: str | None = None,
        _sinks: RunSinks | None = None,
    ) -> SimulationResult:
        """Run a single simulation. Never throws -- returns error SimulationResult on failure.

        ``thread_id`` binds a deterministic, run-scoped Orq observability thread id
        (``f"{run_id}:{index}"``) so every turn of this conversation groups under one
        id in Orq. When ``None`` a fresh uuid is minted. The resolved id is stamped
        onto the returned `SimulationResult` so the dashboard can deep-link to it.
        """
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
        sinks = _sinks if _sinks is not None else RunSinks()
        # Captured for the error path below (out of the `with` scope), so error
        # results still carry the thread id for the dashboard deep-link.
        bound_thread_id: str | None = None
        try:
            # Each conversation runs against its own target clone: targets may
            # hold per-conversation state (ORQAgentTarget threads server-side
            # turns via _task_id; memory-backed targets own an entity id), and
            # sharing one instance across parallel conversations races that
            # state. new() preserves a seeded memory_entity_id and re-mints
            # unseeded ones, so isolation never drops a --memory-entity seed.
            conversation_target = self._new_conversation_target()
            # One Orq thread per simulation groups all its turns (target + user
            # simulator + judge) under one id in Orq observability. ContextVar
            # scoping keeps concurrent datapoints isolated. A run-scoped thread_id
            # (f"{run_id}:{index}") is passed in when available; else one is minted.
            with conversation_thread(thread_id) as thread_id, evaluatorq_pipeline('agent_simulation'):
                bound_thread_id = thread_id
                sinks.thread_id = thread_id
                async with with_simulation_span(
                    'orq.simulation.run',
                    {
                        'orq.simulation.persona': persona.name if persona else None,
                        'orq.simulation.scenario': scenario.name if scenario else None,
                        'orq.simulation.max_turns': effective_max_turns,
                        'orq.simulation.model': self._model,
                        'orq.thread_id': thread_id,
                    },
                ) as run_span:
                    try:
                        result = await self._run_inner(
                            persona=persona,
                            scenario=scenario,
                            datapoint_id=datapoint_id,
                            first_message=first_message,
                            effective_max_turns=effective_max_turns,
                            sinks=sinks,
                            run_span=run_span,
                            conversation_target=conversation_target,
                        )
                        result.thread_id = thread_id
                        result.response_traces = sinks.response_traces
                        return result
                    except BaseException:
                        set_span_attrs(
                            run_span,
                            {
                                'orq.simulation.terminated_by': 'error',
                                'orq.simulation.goal_achieved': False,
                                'orq.simulation.turn_count': sum(1 for m in sinks.messages if m.role == 'assistant'),
                            },
                        )
                        raise
        except Exception as e:
            logger.error('SimulationRunner.run() failed: %s', e, exc_info=True)
            error_msg = str(e)
            error_type = type(e).__name__
            sinks.thread_id = bound_thread_id
            return _partial_result(
                sinks,
                persona=persona,
                scenario=scenario,
                terminated_by=TerminatedBy.error,
                reason=error_msg,
                error_type=error_type,
            )

    async def _run_inner(
        self,
        *,
        persona: Persona | None,
        scenario: Scenario | None,
        datapoint_id: str,
        first_message: str | None,
        effective_max_turns: int,
        sinks: RunSinks,
        run_span: Span | None,
        conversation_target: AgentTarget | None = None,
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
            # Without this the judge sees "No specific criteria defined" and scores 0.0.
            if _implements(judge, _CONTEXTUAL_JUDGE_METHODS):
                judge.update_context(
                    goal=scenario.goal if scenario else '',
                    criteria=list(scenario.criteria) if scenario and scenario.criteria else [],
                    ground_truth=(scenario.ground_truth or '') if scenario else '',
                )
            else:
                logger.warning(
                    'Injected judge %s has no update_context(); it will not receive the scenario goal, '
                    'criteria or ground truth, so no criterion can be audited (criteria_met scores 0.0).',
                    type(self._injected_judge).__name__,
                )
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

        # Lazily captured on the first turn that reports a model identity.
        # NEVER set this from self._model — that is the user-simulator/judge model.
        target_model_holder: dict[str, str | None] = {'model': None}

        def _refresh_token_usage() -> None:
            """Refresh aggregate usage, retaining partial spend if a getter fails."""
            try:
                sinks.token_usage = user_simulator.get_usage() + judge.get_usage() + sinks.target_token_usage
            except Exception as usage_err:
                sinks.token_usage_known = False
                logger.warning(
                    'Unable to determine complete simulation token usage; reporting partial usage as unknown: %s',
                    usage_err,
                    exc_info=True,
                )

        def _build_turn_metrics(turn_num: int, judgment: Judgment, usage_before: TokenUsage) -> TurnMetrics:
            return TurnMetrics(
                turn_number=turn_num,
                token_usage=sinks.token_usage - usage_before,
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
        sinks.messages.append(Message(role='user', content=first_msg))
        _refresh_token_usage()

        last_judgment: Judgment | None = None
        # Accumulates every turn's criteria audit; a violation seen on turn 2 must
        # survive to the final result even if turn 5 is the one that terminates.
        criteria_tracker = _CriteriaTracker(scenario)

        for turn in range(effective_max_turns):
            usage_before = sinks.token_usage.model_copy()

            async with with_simulation_span(
                'orq.simulation.turn',
                {
                    'orq.simulation.turn': turn + 1,
                    'orq.simulation.max_turns': effective_max_turns,
                },
            ) as turn_span:
                # The user's line opens the turn it belongs to. Turn 1's line is the
                # generated first message; every later turn asks the simulator here,
                # so a turn span reads user -> target -> judge instead of trailing the
                # next user message off the end of the previous turn.
                if turn > 0:
                    async with with_simulation_span('orq.simulation.user_simulator_call', None):
                        user_response = await user_simulator.respond_async(
                            _invert_roles_for_simulator(sinks.messages),
                            llm_purpose='user_simulator',
                        )
                    sinks.messages.append(Message(role='user', content=user_response))
                    _refresh_token_usage()

                async with with_simulation_span('orq.simulation.target_call', None) as target_span:
                    # `span_message_text`, not `m.content or ''`: multi-part content
                    # would otherwise land on the span as a Python repr. Same helper
                    # the agent LLM spans use, so the two renderings cannot drift.
                    record_llm_input(
                        target_span,
                        [{'role': m.role, 'content': span_message_text(m.content)} for m in sinks.messages],
                    )
                    call = await self._get_target_response(sinks.messages, target=conversation_target)
                    agent_response_text = call.response.text
                    # An errored response may still be a real, billed provider
                    # response. Its usage belongs in the run just like a success.
                    agent_response_usage = call.response.usage
                    agent_response_model = call.response.model
                    if agent_response_usage is not None:
                        sinks.target_token_usage = sinks.target_token_usage + agent_response_usage
                        sinks.token_usage = sinks.token_usage + agent_response_usage
                    # Capture the first non-None model the target reports; it
                    # should be stable across turns for a given target.
                    if agent_response_model is not None and target_model_holder['model'] is None:
                        target_model_holder['model'] = agent_response_model
                        sinks.target_model = agent_response_model
                    record_llm_output(target_span, agent_response_text)
                # The failed turn IS recorded (mirrors orchestrator.py's
                # turns_record.append on exhaustion) before terminating.
                sinks.messages.extend(build_assistant_message(call.response))

                if not call.succeeded:
                    return self._target_failure_result(
                        call,
                        persona=persona,
                        scenario=scenario,
                        sinks=sinks,
                        run_span=run_span,
                        criteria_tracker=criteria_tracker,
                    )

                # Record this successful target response's trace/span handles (in
                # turn order); the last with a trace id is the conversation's final
                # target-agent trace. Excludes user-simulator and judge calls,
                # which don't set these here.
                if call.response.trace_id or call.response.span_id:
                    sinks.response_traces.append(
                        ResponseTrace(trace_id=call.response.trace_id, span_id=call.response.span_id)
                    )

                async with with_simulation_span('orq.simulation.judge_evaluation', None):
                    judgment = await judge.evaluate(sinks.messages)
                criteria_tracker.observe(judgment.criteria_verdicts)
                state_judgment = criteria_tracker.resolve(judgment) if criteria_tracker.audited_ids else judgment
                sinks.criteria_verified = criteria_tracker.verified
                sinks.criteria_results = _build_criteria_results(scenario, state_judgment) if scenario else None
                sinks.criteria_meta = (
                    _build_criteria_meta(
                        scenario,
                        state_judgment,
                        criteria_tracker.audited_ids,
                        criteria_tracker.evidence,
                    )
                    if scenario
                    else None
                )
                sinks.rules_broken = state_judgment.rules_broken
                sinks.goal_achieved = judgment.goal_achieved
                sinks.goal_completion_score = judgment.goal_completion_score
                # Occurrence is sticky, so a confirmed criterion cannot change —
                # stop paying for it in every remaining turn's audit payload. Duck
                # typed, per the _implements note above: a custom judge without the
                # method simply keeps auditing everything.
                if _implements(judge, _SETTLEABLE_JUDGE_METHODS):
                    judge.mark_settled(criteria_tracker.settled_ids)

                _refresh_token_usage()
                sinks.turn_metrics.append(_build_turn_metrics(turn + 1, judgment, usage_before))
                try:
                    await await_maybe(self._hooks.on_turn_complete(datapoint_id, sinks.turn_metrics[-1]))
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

            if last_judgment and last_judgment.should_terminate:
                _refresh_token_usage()
                # rules_broken is the run-level fold, not just this turn's judgment.
                resolved = criteria_tracker.resolve(last_judgment)
                set_span_attrs(
                    run_span,
                    {
                        'orq.simulation.terminated_by': 'judge',
                        'orq.simulation.goal_achieved': resolved.goal_achieved,
                        'orq.simulation.turn_count': turn + 1,
                    },
                )
                judge_metadata = _build_simulation_metadata(
                    persona,
                    scenario,
                    _build_criteria_meta(scenario, resolved, criteria_tracker.audited_ids, criteria_tracker.evidence)
                    if scenario
                    else None,
                    target_model_holder['model'],
                )
                return SimulationResult(
                    messages=sinks.messages,
                    terminated_by=TerminatedBy.judge,
                    reason=resolved.reason,
                    goal_achieved=resolved.goal_achieved,
                    goal_completion_score=resolved.goal_completion_score,
                    rules_broken=resolved.rules_broken,
                    turn_count=turn + 1,
                    turn_metrics=sinks.turn_metrics,
                    token_usage=sinks.token_usage,
                    token_usage_known=sinks.token_usage_known,
                    criteria_results=self._build_criteria_results(scenario, resolved) if scenario else None,  # type: ignore[arg-type]
                    criteria_verified=criteria_tracker.verified,
                    metadata=judge_metadata,  # type: ignore[union-attr]
                )

        # Max turns reached
        _refresh_token_usage()
        resolved = criteria_tracker.resolve(last_judgment) if last_judgment else None
        set_span_attrs(
            run_span,
            {
                'orq.simulation.terminated_by': 'max_turns',
                'orq.simulation.goal_achieved': resolved.goal_achieved if resolved else False,
                'orq.simulation.turn_count': effective_max_turns,
            },
        )
        return _max_turns_result(
            effective_max_turns,
            sinks.messages,
            sinks.turn_metrics,
            sinks.token_usage,
            persona,
            scenario,
            resolved,
            target_model=target_model_holder['model'],
            criteria_verified=criteria_tracker.verified,
            token_usage_known=sinks.token_usage_known,
            audited_ids=criteria_tracker.audited_ids,
            evidence=criteria_tracker.evidence,
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
        """Close the shared HTTP client and any per-conversation target clones."""
        # Clones may own resources (an OrqResponsesTarget clone builds its own
        # HTTP client when the parent's was self-owned); close them best-effort
        # so one failing clone never masks the rest or the caller's exception.
        spawned, self._spawned_targets = self._spawned_targets, []
        for clone in spawned:
            closer = getattr(clone, 'close', None)
            if closer is None:
                continue
            try:
                maybe_coro = closer()
                if inspect.isawaitable(maybe_coro):
                    await maybe_coro
            except Exception as exc:
                logger.warning('Failed to close per-conversation target clone: %s', exc)
        if self._shared_client is not None and self._client_owned:
            await self._shared_client.close()
            self._shared_client = None

    # ---------------------------------------------------------------------------
    # Private helpers
    # ---------------------------------------------------------------------------

    def _target_failure_result(
        self,
        call: TargetCallResult,
        *,
        persona: Persona | None,
        scenario: Scenario | None,
        sinks: RunSinks,
        run_span: Span | None,
        criteria_tracker: _CriteriaTracker,
    ) -> SimulationResult:
        """Terminate the run after the target exhausted its retries.

        The failed turn is already appended to ``messages``; this builds the
        terminal `SimulationResult` (no judge call). Mirrors the judge /
        max-turns termination branches: records final token usage + span attrs
        and retains prior ``messages``/``turn_metrics``. A ``timeout`` error type
        maps to `TerminatedBy.timeout`, everything else to
        `TerminatedBy.error`.

        The criteria audit collected before the target died is folded in, not
        discarded: a `must_not_happen` violation the judge confirmed on turn 2 has
        to survive the target dying on turn 4, or it vanishes from the result and
        the report. `criteria_verified` still comes from the tracker, and this run
        is scored 0.0 by `criteria_met` regardless (it terminated by error/timeout),
        so the fold adds evidence without letting a dead target claim a clean sheet.

        **On this branch only confirmed occurrence is knowledge.** Unlike
        `_CriteriaTracker.resolve`, the run did not get its full chance, so the fold
        is `confirmed_broken_ids`, not `broken_ids`: a `must_happen` that has not
        occurred means "not yet", not "the judge confirmed it never happened", and
        reporting it as failed would invert the branch's own thesis — a red row in
        every report and a phantom entry in the cross-run failure-mode table. Those
        ids are withheld from the meta's audited set as well, so they render
        `state='unknown'` rather than a green `pass`. With **no** audit at all the
        same rule empties the fold entirely (`criteria_verified=False`,
        `audited=False`, `state='unknown'` on every criterion).
        """
        err = call.error
        error_message = err.message if err else 'target failed'
        error_type = err.error_type if err else 'target_error'
        terminated_by = TerminatedBy.timeout if error_type == 'timeout' else TerminatedBy.error
        turn_count = sum(1 for m in sinks.messages if m.role == 'assistant')

        set_span_attrs(
            run_span,
            {
                'orq.simulation.terminated_by': terminated_by.value,
                'orq.simulation.goal_achieved': False,
                'orq.simulation.turn_count': turn_count,
            },
        )

        scenario_name = scenario.name if scenario else '<none>'
        logger.warning(
            'Target failed (%s) for scenario %r: %s',
            error_type,
            scenario_name,
            error_message,
        )
        # `verified` is the same condition `resolve` guards on (an audit arrived, or
        # there are no criteria at all — in which case `broken_ids` is empty anyway).
        # Inside a partial audit, `confirmed_broken_ids` applies the narrower rule
        # this branch needs: only an observed occurrence is knowledge.
        unconfirmed = criteria_tracker.unconfirmed_ids
        broken = criteria_tracker.confirmed_broken_ids if criteria_tracker.verified else []
        # Withheld from `audited` so the unconfirmed ids render as unknown rather
        # than as an audited pass — `CriteriaRow.state` is `unknown` only when a
        # passing criterion is not audited.
        audited_ids = criteria_tracker.audited_ids - unconfirmed
        if not criteria_tracker.verified:
            logger.warning(
                'Target failed (%s) before the judge audited any criterion of scenario %r; reporting '
                'those %d criteria as unknown (unverified), not as failed.',
                error_type,
                scenario_name,
                len(scenario.criteria or []) if scenario else 0,
            )
        else:
            # The same per-id warning `resolve` emits on the normal path; without it
            # a partial audit's defaulted ids are never named anywhere in the log.
            unaudited = criteria_tracker.unaudited_ids
            if unaudited:
                logger.warning(
                    'Judge never reported an occurrence verdict for %s on scenario %r before the target '
                    'failed (%s); they fell to their not-observed default.',
                    ', '.join(sorted(unaudited)),
                    scenario_name,
                    error_type,
                )
        if unconfirmed:
            logger.warning(
                'Target failed (%s) on scenario %r before %s could occur; reporting them as unknown, '
                'not as failed — the run was cut short, so "not yet" is not a confirmed failure.',
                error_type,
                scenario_name,
                ', '.join(sorted(unconfirmed)),
            )
        folded = Judgment(
            should_terminate=True,
            reason=error_message,
            goal_achieved=False,
            rules_broken=broken,
            goal_completion_score=0.0,
        )
        criteria_meta = (
            _build_criteria_meta(scenario, folded, audited_ids, criteria_tracker.evidence) if scenario else None
        )
        if broken:
            logger.warning(
                'Target failed (%s) after the judge had confirmed %s; keeping those criteria verdicts on '
                'the errored result.',
                error_type,
                ', '.join(broken),
            )
        sinks.criteria_meta = criteria_meta
        sinks.criteria_results = _build_criteria_results(scenario, folded) if scenario else None
        sinks.criteria_verified = criteria_tracker.verified
        sinks.rules_broken = broken
        sinks.goal_achieved = False
        sinks.goal_completion_score = 0.0
        return _partial_result(
            sinks,
            persona=persona,
            scenario=scenario,
            terminated_by=terminated_by,
            reason=error_message,
            error_type=error_type,
        )

    def _new_conversation_target(self) -> AgentTarget | None:
        """Mint a fresh target clone for one conversation.

        Stateful targets (e.g. ``ORQAgentTarget`` threading server-side turns
        via ``_task_id``) race when one instance serves parallel conversations,
        so every ``run()`` gets its own ``new()`` clone. Clones are retained on
        the runner so `close` can release resources they own.
        """
        if self._effective_target is None:
            return None
        clone = self._effective_target.new()
        self._spawned_targets.append(clone)
        return clone

    async def _get_target_response(
        self, messages: list[Message], *, target: AgentTarget | None = None
    ) -> TargetCallResult:
        """Call the target through the shared bounded-retry + timeout helper.

        ``target`` is the per-conversation clone minted by ``run()``; the shared
        ``_effective_target`` is only a fallback for direct callers of this
        helper. Both target flavours (rich ``target_agent`` and a plain callback
        wrapped in ``CallableTarget``) route through the same helper. The returned
        `TargetCallResult` carries ``.response`` (always populated — real
        or synthetic), ``.succeeded``, and ``.error``. ``response.model`` is the
        model the target reports (``None`` for plain callbacks, which may call any
        provider); NEVER substitute ``self._model`` — that is the user-simulator /
        judge model, not the evaluated target.
        """
        effective = target if target is not None else self._effective_target
        if effective is None:
            raise RuntimeError('No target agent configured')
        return await call_target_with_retry(
            effective,
            messages,
            target_agent_timeout_ms=self._target_agent_timeout_ms,
            max_target_retries=self._max_target_retries,
            map_error=default_map_error,
        )

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

        # Own one sink object so every piece of state completed before an outer
        # timeout survives cancellation of the inner coroutine.
        sinks = RunSinks()
        try:
            return await asyncio.wait_for(
                self.run(
                    datapoint=datapoint,
                    max_turns=max_turns,
                    _sinks=sinks,
                ),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            logger.warning(
                'Simulation for datapoint %s timed out after %ss; returning partial result',
                datapoint.id,
                timeout_s,
            )
            reason = f'Simulation timed out after {timeout_s}s'
            return _partial_result(
                sinks,
                persona=datapoint.persona,
                scenario=datapoint.scenario,
                terminated_by=TerminatedBy.timeout,
                reason=reason,
                error_type='timeout',
                timeout_s=timeout_s,
            )

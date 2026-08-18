"""Judge agent for conversation evaluation.

Evaluates conversations and decides when to terminate based on
goal achievement or rule violations.
"""

from __future__ import annotations

import inspect
import logging
import math
from itertools import starmap
from typing import TYPE_CHECKING, Any, ClassVar, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    ValidatorFunctionWrapHandler,
    field_validator,
    model_validator,
)

from evaluatorq.common.sanitize import delimit
from evaluatorq.simulation.agents.base import AgentConfig, BaseAgent, LLMResult
from evaluatorq.simulation.types import Criterion, CriterionVerdict, Judgment, Message, criterion_id_for

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from evaluatorq.contracts import LLMCallConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool argument models
#
# The judge's tool schemas are GENERATED from these models, never hand-written.
# A hand-maintained JSON-schema dict and the parser that reads its output drift
# the moment one is edited alone — and the parser is the half nobody notices is
# stale, because a field it stopped reading just looks like one the judge chose
# not to report.
# ---------------------------------------------------------------------------


def _score(description: str) -> Any:
    """An optional 0..1 quality score. ``None`` means "not scored", not "scored 0"."""
    return Field(default=None, description=description)


class _JudgeToolArgs(BaseModel):
    """Fields both judge tools share.

    Tolerant by design, at one specific granularity: a malformed *entry* is
    dropped with a warning and the rest of the turn survives, because losing one
    criterion's evidence for one turn is recoverable — the run-level fold defaults
    it to not-observed and says so — while losing the whole turn's audit is what
    marks a run unverified.
    """

    model_config = ConfigDict(extra='ignore')

    WIRE_REQUIRED: ClassVar[frozenset[str]] = frozenset({
        'reason',
        'goal_completion_score',
        'criteria_verdicts',
    })
    """Fields the judge must send.

    Deliberately NOT pydantic's own `required`. Every field here also has a
    parser-side default, so pydantic would emit an empty `required` and the audit
    this class exists to collect would become optional again — which is RES-1308.
    Asked-for and relied-on are different contracts; `_wire_schema` uses this one
    and `_assert_wire_fields_exist` keeps it honest.
    """

    WIRE_NON_NULLABLE: ClassVar[frozenset[str]] = frozenset({'criteria_verdicts'})
    """Fields whose ``X | None`` annotation loses its ``null`` branch on the wire.

    ``None`` for `criteria_verdicts` means "the judge told us nothing", which is
    what decides whether a run is scored or marked unverified. Offering an
    explicit ``null`` gives the model a cheap way to produce that state on purpose,
    indistinguishable from a broken payload. The quality scores are deliberately
    absent from this set: for them ``null`` is a real answer ("no ground truth, not
    scored"), and forcing a number would make the model invent one.
    """

    reason: str = Field(default='', description='Brief explanation of the decision')
    goal_completion_score: float = Field(
        default=0.0,
        description='How much of the goal is achieved SO FAR, 0.0 (none) to 1.0 (fully). Assess every turn — '
        'if the run hits max turns this is the final score.',
    )
    criteria_verdicts: list[CriterionVerdict] | None = Field(
        default=None,
        description='Occurrence audit. One entry for every criterion listed under EVALUATION CRITERIA, no '
        'omissions. Skip any criterion listed as ALREADY CONFIRMED in the final user message — those are '
        'settled and re-reporting them changes nothing; send [] when that leaves nothing to report. '
        'Report only what literally happened in the conversation so far — do NOT judge whether that '
        'is good or bad.',
    )
    response_quality: float | None = _score(
        "Quality of the agent's last response: helpful, accurate, complete (0.0=poor, 1.0=excellent)"
    )
    hallucination_risk: float | None = _score(
        'Risk that the agent fabricated information not grounded in the conversation (0.0=none, 1.0=high risk)'
    )
    tone_appropriateness: float | None = _score(
        "How appropriate the agent's tone was for the situation (0.0=inappropriate, 1.0=perfect)"
    )
    factual_accuracy: float | None = _score(
        "Accuracy of the agent's response against the provided ground truth (0.0=completely wrong, "
        '1.0=fully correct). Only score this if ground truth is provided.'
    )

    @model_validator(mode='before')
    @classmethod
    def _null_score_reads_as_omitted(cls, data: object) -> object:
        """Treat an explicit ``"goal_completion_score": null`` as if the key were absent.

        The two spellings mean the same thing to a model, but pydantic records an
        explicitly-null key in `model_fields_set`, which would skip the "a finish
        that achieved the goal and sent no score means 1.0" fallback in
        `_parse_judgment` and score the run 0.0 instead. Normalising here — before
        the field validators run — also keeps the null out of
        `_coerce_required_score`, which would otherwise warn about a value that is
        not actually a broken answer.
        """
        if isinstance(data, dict) and data.get('goal_completion_score', ...) is None:
            data = {k: v for k, v in cast('dict[str, object]', data).items() if k != 'goal_completion_score'}
        return data

    @field_validator('reason', mode='before')
    @classmethod
    def _coerce_reason(cls, value: object) -> str:
        """Coerce a non-string `reason` rather than failing the whole payload over it.

        Same trade as `_coerce_required_score` directly below, for the same reason:
        `reason` is free text nothing branches on, while rejecting the payload would
        safety-terminate the run and discard its criteria audit. The two adjacent
        branches must not differ in whether they degrade.
        """
        if isinstance(value, str):
            return value
        logger.warning(
            'JudgeAgent: non-string reason %r; coercing to text rather than discarding this turn.',
            value,
        )
        return '' if value is None else str(value)

    @field_validator(
        'response_quality',
        'hallucination_risk',
        'tone_appropriateness',
        'factual_accuracy',
        mode='before',
    )
    @classmethod
    def _coerce_optional_score(cls, value: object) -> float | None:
        """Drop an unusable optional score to ``None`` rather than guessing at it.

        These fields are already optional, so ``None`` reads as "not scored"; a
        fabricated 0.0 would read as "scored terribly".
        """
        return _as_unit_interval(value)

    @field_validator('goal_completion_score', mode='before')
    @classmethod
    def _coerce_required_score(cls, value: object) -> float:
        """Fall back to 0.0 rather than failing the whole payload over one number.

        `goal_completion_score` is not optional, so rejecting a model that answered
        ``"high"`` would safety-terminate the run and lose its criteria audit with
        it — a far worse trade than one wrong score.
        """
        score = _as_unit_interval(value)
        if score is None:
            logger.warning(
                'JudgeAgent: unusable goal_completion_score %r; scoring this turn 0.0.',
                value,
            )
            return 0.0
        return score

    @field_validator('criteria_verdicts', mode='wrap')
    @classmethod
    def _keep_usable_verdicts(
        cls,
        value: object,
        handler: ValidatorFunctionWrapHandler,
        info: ValidationInfo,
    ) -> list[CriterionVerdict] | None:
        """Drop entries that cannot be trusted; keep the rest of the turn.

        What counts as malformed is defined by `CriterionVerdict` itself — the
        handler validates and the failing list indices come back in the
        `ValidationError`'s ``loc``. Adding a constraint to that model therefore
        becomes a drop-with-warning for free, where an isinstance ladder here
        would have to be edited to match and could silently fall behind.

        Returns ``None`` (unknown — the run is unverified) only when nothing at
        all is salvageable. Never returns ``None`` for an audit that was simply
        empty; that is a legitimate "everything is settled" answer.
        """
        if value is None:
            return None
        try:
            verdicts = handler(value)
        except ValidationError as err:
            bad = {loc[0] for e in err.errors() if (loc := e['loc']) and isinstance(loc[0], int)}
            if not isinstance(value, list) or not bad:
                logger.warning(
                    'JudgeAgent: criteria_verdicts unusable as a whole (%s); this turn contributes no '
                    'criteria evidence.',
                    _first_error(err),
                )
                return None
            logger.warning(
                'JudgeAgent: dropped %d of %d malformed criteria_verdicts entr(y/ies) (%s); those criteria '
                'carry no evidence from this turn.',
                len(bad),
                len(cast('list[object]', value)),
                _first_error(err),
            )
            try:
                verdicts = handler([v for i, v in enumerate(cast('list[object]', value)) if i not in bad])
            except ValidationError as second:
                logger.warning(
                    'JudgeAgent: criteria_verdicts still unusable after dropping malformed entries (%s); '
                    'this turn contributes no criteria evidence.',
                    _first_error(second),
                )
                return None
            if not verdicts:
                # Salvage that saved nothing is *unknown*, not "audited and clean":
                # `[]` claims the judge looked and had nothing left to report, which
                # is the flattering silence RES-1308 is about. An audit that arrived
                # empty is a different thing and still returns `[]` above.
                logger.warning(
                    'JudgeAgent: every criteria_verdicts entry was malformed; this turn contributes no '
                    'criteria evidence (unknown, not an empty audit).',
                )
                return None
        if verdicts is None:
            return None
        return _resolve_against_scenario(cast('list[CriterionVerdict]', verdicts), info)


def _as_unit_interval(value: object) -> float | None:
    """Coerce a model-emitted score to 0..1, or ``None`` when it is not a number.

    Accepts the numeric strings models emit and clamps out-of-range answers;
    rejects bools (``True`` would otherwise coerce to 1.0) and NaN/inf.
    """
    if value is None or isinstance(value, bool):
        return None
    if not isinstance(value, (int, float, str)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return max(0.0, min(1.0, number))


def _first_error(err: ValidationError) -> str:
    """One-line summary of a validation failure, for a warning that has to name a cause."""
    errors = err.errors()
    if not errors:
        return 'no detail'
    first = errors[0]
    location = '.'.join(str(part) for part in first['loc']) or '<root>'
    suffix = f' (+{len(errors) - 1} more)' if len(errors) > 1 else ''
    return f'{location}: {first["msg"]}{suffix}'


def _resolve_against_scenario(
    verdicts: list[CriterionVerdict],
    info: ValidationInfo,
) -> list[CriterionVerdict] | None:
    """Apply the checks that need to know which scenario produced these verdicts.

    Shape is `CriterionVerdict`'s job; these three are not shape:

    - an id outside the scenario's criteria is perfectly well-formed, so it would
      otherwise sail through validation and then match nothing
    - a non-canonical spelling of an in-range id (``criteria_01`` matches the id
      pattern and yields index 1) passes every check *here* — which gates on the
      numeric ``.index`` — and is then discarded downstream by `_CriteriaTracker`,
      which matches the exact ``criterion_id`` string. The design's premise is
      that id and index can never disagree, so require the canonical spelling and
      close that seam rather than let a misattributed audit look accepted.
    - a duplicate id would let the last entry silently win

    Returns ``None`` — *unknown* — when a non-empty payload loses **every** entry,
    matching `_keep_usable_verdicts`' rule that a salvage which saved nothing is
    not "audited and clean". An audit that arrived empty still returns ``[]``.

    Sorted on the way out, so nothing downstream depends on the order the model
    happened to emit.
    """
    count = (info.context or {}).get('criteria_count')
    kept: dict[int, CriterionVerdict] = {}
    unknown = miswritten = duplicate = 0
    for verdict in verdicts:
        if count is not None and verdict.index >= count:
            unknown += 1
            continue
        if verdict.criterion_id != criterion_id_for(verdict.index):
            miswritten += 1
            continue
        if verdict.index in kept:
            duplicate += 1
        kept[verdict.index] = verdict
    for label, dropped in (('out-of-range', unknown), ('non-canonical-id', miswritten), ('duplicate', duplicate)):
        if dropped:
            logger.warning(
                'JudgeAgent: discarded %d %s criteria_verdicts entr(y/ies) against a scenario with %s '
                'criteria; those criteria carry no evidence from this turn.',
                dropped,
                label,
                count,
            )
    if verdicts and not kept:
        logger.warning(
            'JudgeAgent: every criteria_verdicts entry was discarded against the scenario; this turn '
            'contributes no criteria evidence (unknown, not an empty audit).',
        )
        return None
    return [kept[index] for index in sorted(kept)]


class ContinueConversation(_JudgeToolArgs):
    """Allow the conversation to continue. Use when the goal is not yet achieved and no rules are broken."""

    # `reason` is re-declared only to carry this tool's own wording: field
    # descriptions are prompt text the judge reads, and "the decision" says less
    # than naming which decision it just made.
    reason: str = Field(default='', description='Brief explanation of why the conversation should continue')

    # No `rules_broken` here either — see `FinishConversation`.


class FinishConversation(_JudgeToolArgs):
    """Terminate the conversation. Use when the goal is achieved OR a rule is broken."""

    WIRE_REQUIRED: ClassVar[frozenset[str]] = _JudgeToolArgs.WIRE_REQUIRED | {'goal_achieved'}

    reason: str = Field(default='', description='Explanation of why the conversation should end')
    # The base wording ("SO FAR … if the run hits max turns this is the final
    # score") is written for a turn that continues; on the call that ends the run
    # there is no "so far".
    goal_completion_score: float = Field(
        default=0.0,
        description='How much of the goal was achieved, from 0.0 (none) to 1.0 (fully achieved). Use '
        'intermediate values for partial completion.',
    )
    goal_achieved: bool = Field(default=False, description="Whether the user's goal was successfully achieved")

    # No `rules_broken`. The judge is asked what OCCURRED; which occurrences count
    # as violations is derived in code from `Criterion.type`. Asking for both gave
    # them something to disagree about, and the free-text list is the channel that
    # could not fail a `must_happen` criterion in the first place (RES-1308).


# ---------------------------------------------------------------------------
# Model -> wire schema
# ---------------------------------------------------------------------------

_NESTED_WIRE_REQUIRED: dict[type[BaseModel], frozenset[str]] = {
    CriterionVerdict: frozenset({'criterion_id', 'occurred', 'evidence'}),
}
"""`WIRE_REQUIRED`, one nesting level down.

`_JudgeToolArgs.WIRE_REQUIRED` governs the top-level object only; a nested model
gets pydantic's own ``required``, which omits every field that has a parser-side
default — exactly the "generated schema quietly un-requires a field" defect
`WIRE_REQUIRED` exists to prevent, reproduced inside the item schema. `evidence`
defaults to ``''`` in the parser and would drop out of the item's ``required``,
so the evidence capture the audit depends on would become optional on the wire.
"""

_STRIPPED_SCHEMA_KEYS = ('title', 'default')
"""Dropped from every node of a generated schema.

``title`` is noise. ``default`` is the harmful one: it is a *parser-side* value,
and the whole point of the parser's defaults is that a field can be mandatory on
the wire while the parser still survives its absence. Advertising the default
tells the model the opposite of what `WIRE_REQUIRED` does.
"""


def _walk(node: object, fn: Callable[[dict[str, Any]], None]) -> None:
    """Apply `fn` to every dict in a JSON-schema tree."""
    if isinstance(node, dict):
        fn(cast('dict[str, Any]', node))
        for value in list(cast('dict[str, Any]', node).values()):
            _walk(value, fn)
    elif isinstance(node, list):
        for value in cast('list[object]', node):
            _walk(value, fn)


def _strip_node(node: dict[str, Any]) -> None:
    """Remove everything from one schema node that must not reach the model."""
    for key in _STRIPPED_SCHEMA_KEYS:
        node.pop(key, None)
    if 'properties' in node:
        # A model-derived object node carries its class docstring as `description`,
        # and those docstrings are maintainer-facing rationale (why a field exists,
        # which model got it wrong before). The judge reads every byte of this
        # schema as prompt, so only *field* descriptions — written for it — survive.
        node.pop('description', None)


def _inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Resolve ``$ref``/``$defs`` into a self-contained schema.

    Pydantic factors nested models (here `CriterionVerdict`) out into ``$defs``.
    OpenAI itself resolves refs fine; this is for the other providers behind the
    Orq router, whose schema dialects are narrower — and a dropped item schema
    fails as an unaudited run rather than as an error, so it would not be obvious.
    """
    defs = cast('dict[str, Any]', schema.pop('$defs', {}))

    def resolve(node: object) -> object:
        if isinstance(node, dict):
            node = cast('dict[str, Any]', node)
            ref = node.get('$ref')
            if isinstance(ref, str) and ref.startswith('#/$defs/'):
                definition = defs.get(ref.rsplit('/', 1)[-1])
                if definition is None:
                    # Cannot happen with a schema pydantic just generated, so if it
                    # does the generator changed under us — and the consequence is a
                    # shapeless node the model can fill with anything, which shows up
                    # as an unaudited run rather than as an error.
                    logger.warning(
                        'JudgeAgent: unresolvable %s in the generated tool schema; that node reaches the '
                        'model with no shape.',
                        ref,
                    )
                    definition = {}
                target = {**cast('dict[str, Any]', definition), **{k: v for k, v in node.items() if k != '$ref'}}
                return resolve(target)
            return {key: resolve(value) for key, value in node.items()}
        if isinstance(node, list):
            return [resolve(value) for value in cast('list[object]', node)]
        return node

    return cast('dict[str, Any]', resolve(schema))


def _drop_null_branch(name: str, prop: dict[str, Any]) -> dict[str, Any]:
    """Collapse the ``anyOf: [X, null]`` pydantic emits for ``X | None`` down to ``X``.

    See `_JudgeToolArgs.WIRE_NON_NULLABLE` for why one field must not be nullable
    on the wire even though it is optional in the parser.
    """
    branches = [b for b in cast('list[dict[str, Any]]', prop.get('anyOf', [])) if b.get('type') != 'null']
    if len(branches) != 1:
        logger.warning(
            'JudgeAgent: %s is listed in WIRE_NON_NULLABLE but its schema has %d non-null branch(es), not '
            'one; leaving it nullable on the wire, so the model can send null deliberately.',
            name,
            len(branches),
        )
        return prop
    return {**{k: v for k, v in prop.items() if k != 'anyOf'}, **branches[0]}


def _wire_schema(model: type[_JudgeToolArgs]) -> dict[str, Any]:
    """The ``parameters`` object of `model`'s function tool.

    Three deliberate departures from `model_json_schema()`, because the wire
    contract and the parser contract are not the same contract: ``required`` comes
    from `WIRE_REQUIRED` (and, one level down, `_NESTED_WIRE_REQUIRED`),
    ``default`` is stripped, and `WIRE_NON_NULLABLE` fields lose their null branch.
    """
    raw = model.model_json_schema()
    # Before inlining: while the nested models still live in `$defs` under their
    # class name, which is the one place their `required` can be reached by name.
    defs = cast('dict[str, Any]', raw.get('$defs') or {})
    for nested, required in _NESTED_WIRE_REQUIRED.items():
        node = cast('dict[str, Any] | None', defs.get(nested.__name__))
        if node is not None:
            node['required'] = sorted(required)
    schema = _inline_refs(raw)
    _walk(schema, _strip_node)
    properties = cast('dict[str, Any]', schema.get('properties', {}))
    for name in model.WIRE_NON_NULLABLE:
        if name in properties:
            properties[name] = _drop_null_branch(name, properties[name])
    schema['required'] = sorted(model.WIRE_REQUIRED)
    return schema


def _tool_schema(name: str, model: type[_JudgeToolArgs]) -> dict[str, Any]:
    """Wrap `model`'s wire schema in the OpenAI function-tool envelope.

    The tool description is the first paragraph of the class docstring, so the
    model and the thing that describes it cannot drift.
    """
    return {
        'type': 'function',
        'function': {
            'name': name,
            'description': inspect.cleandoc(model.__doc__ or '').split('\n\n')[0],
            'parameters': _wire_schema(model),
        },
    }


# ---------------------------------------------------------------------------
# Judge tools for structured decision making
# ---------------------------------------------------------------------------

_TOOL_MODELS: dict[str, type[_JudgeToolArgs]] = {
    'continue_conversation': ContinueConversation,
    'finish_conversation': FinishConversation,
}


def _assert_wire_fields_exist() -> None:
    """Fail at import if `WIRE_REQUIRED` names a field the model does not have.

    `WIRE_REQUIRED` is written by hand because pydantic's own ``required`` cannot
    express it. This is the mechanical check that a rename cannot land on only one
    of the two — otherwise the tool would keep demanding a field that no longer
    exists, and the parser would never see it. The nested models in
    `_NESTED_WIRE_REQUIRED` are checked the same way: a rename inside
    `CriterionVerdict` must not half-land either.
    """
    hand_written: list[tuple[type[BaseModel], frozenset[str]]] = [
        *((model, model.WIRE_REQUIRED) for model in _TOOL_MODELS.values()),
        *_NESTED_WIRE_REQUIRED.items(),
    ]
    for model, required in hand_written:
        missing = required - set(model.model_fields)
        if missing:
            raise RuntimeError(f'{model.__name__} wire-required names unknown field(s): {sorted(missing)}')


_assert_wire_fields_exist()

JUDGE_TOOLS: list[dict[str, Any]] = list(starmap(_tool_schema, _TOOL_MODELS.items()))

# ---------------------------------------------------------------------------
# Default judge system prompt
# ---------------------------------------------------------------------------

DEFAULT_JUDGE_PROMPT = """You are a conversation judge. Your role is to evaluate conversations between a user and an AI agent.

You will be given:
1. The conversation history
2. The user's goal
3. Criteria that should or should not be satisfied

Your task:
- Evaluate whether the conversation should continue or end
- Determine if the user's goal has been achieved
- Check if any rules/criteria have been violated

IMPORTANT: Each criterion has a unique ID (e.g., "criteria_0", "criteria_1").
When reporting criteria_verdicts, copy the criterion ID exactly as listed in `criterion_id`.
Never paraphrase the description, renumber the IDs, or invent one that is not listed.

Decision rules:
1. FINISH if the user's goal is clearly achieved
2. FINISH if any "must_not_happen" criteria are violated
3. CONTINUE if the goal is not yet achieved and no rules are broken
4. CONTINUE if progress is being made toward the goal
5. An unmet "must_happen" criterion is NOT a reason to finish early — it may still happen later

CRITERIA AUDIT (every evaluation, continue or finish):
You MUST return criteria_verdicts with exactly one entry for every criterion ID listed
below — no omissions, even when nothing has changed since the last turn. The only
exception is a criterion the final user message lists as ALREADY CONFIRMED: those are
settled, so skip them, and send [] when that leaves nothing to report.

This is an OCCURRENCE report, not a verdict. For each criterion answer one question:
"has the behaviour in this description actually appeared in the conversation so far?"
- occurred=true if it is there, occurred=false if it is not.
- Answer identically whether the criterion is must_happen or must_not_happen. Do NOT
  flip the answer because the behaviour is desired or forbidden — pass/fail is computed
  from your answer, not by you.
- Quote the supporting text in `evidence` when occurred=true; leave it empty otherwise.
- Judge the literal transcript. Do not credit intent, plans, or things the agent looks
  likely to do next.

For EVERY evaluation (continue or finish), also assess the agent's LAST response:
- response_quality: How helpful, accurate, and complete was the response? (0.0=poor, 1.0=excellent)
- hallucination_risk: Did the agent make up information not grounded in the conversation? (0.0=none, 1.0=high risk)
- tone_appropriateness: Was the agent's tone appropriate for the situation? (0.0=inappropriate, 1.0=perfect)
- factual_accuracy: If GROUND TRUTH is provided below, score how accurate the agent's response is against it (0.0=wrong, 1.0=correct). Skip if no ground truth.

You MUST call one of the provided tools to make your decision."""

# ---------------------------------------------------------------------------
# Safety termination
# ---------------------------------------------------------------------------


def _safety_terminate(reason: str) -> Judgment:
    """End the conversation when the judge's answer cannot be trusted.

    ``criteria_verdicts`` is deliberately left ``None`` — unknown, not clean. The
    empty ``rules_broken`` passes every criterion, so a malfunctioning judge that
    also reported an empty audit would score a perfect run, which is RES-1308
    wearing a different hat. The runner marks such a run ``criteria_verified=False``.
    """
    return Judgment(
        should_terminate=True,
        reason=reason,
        goal_achieved=False,
        rules_broken=[],
        goal_completion_score=0.0,
    )


class JudgeAgentConfig(AgentConfig):
    """Configuration for JudgeAgent."""

    goal: str = ''
    criteria: list[Criterion] | None = None
    ground_truth: str = ''

    def __init__(
        self,
        goal: str = '',
        criteria: list[Criterion] | None = None,
        ground_truth: str = '',
        **kwargs: Any,
    ) -> None:
        # Default the judge to the Responses API: it supports function tools +
        # reasoning_effort together, which chat/completions rejects with a 400 for
        # models like gpt-5.4-mini. Callers can still pass api='chat_completions'
        # (and their own client/base_url) to override.
        kwargs.setdefault('api', 'responses')
        super().__init__(**kwargs)
        self.goal = goal
        self.criteria = criteria
        self.ground_truth = ground_truth


class JudgeAgent(BaseAgent):
    """Agent that evaluates conversations and decides termination.

    Uses tool calling to make structured decisions about whether a conversation
    should continue or end.
    """

    def __init__(
        self,
        config: JudgeAgentConfig | AgentConfig | LLMCallConfig | None = None,
    ) -> None:
        super().__init__(config)
        if isinstance(config, JudgeAgentConfig):
            self._goal = config.goal
            self._criteria = config.criteria or []
            self._ground_truth = config.ground_truth
        else:
            self._goal = ''
            self._criteria: list[Criterion] = []
            self._ground_truth = ''
        self._settled: frozenset[str] = frozenset()

    @property
    def name(self) -> str:
        return 'JudgeAgent'

    def mark_settled(self, ids: Iterable[str]) -> None:
        """Tell the judge which criteria are already confirmed to have occurred.

        Occurrence is sticky, so a criterion the runner has seen occur cannot
        change — auditing it again on every remaining turn costs a payload entry
        (id, boolean, and an evidence quote) and can only restate a settled fact.
        The per-turn audit exists so the judge can stop the conversation early; a
        criterion that already occurred contributes nothing more to that decision.

        Rebinds a frozenset rather than mutating one, because the runner
        shallow-copies this agent per simulation and a shared mutable set would
        leak occurrence between concurrent runs.
        """
        self._settled = frozenset(ids)

    def update_context(
        self,
        goal: str | None = None,
        criteria: list[Criterion] | None = None,
        ground_truth: str | None = None,
    ) -> None:
        """Update the scenario context the judge evaluates against.

        The runner shallow-copies an injected judge per simulation and calls this
        with the datapoint's scenario — without it the judge keeps whatever it was
        constructed with (usually nothing), so the system prompt says "No specific
        criteria defined" and every criterion goes unaudited.

        ``criteria`` is copied into a fresh list rather than stored by reference:
        the per-simulation copy is shallow, so keeping the caller's list would let
        concurrent runs share (and mutate) the same object. ``system_prompt`` is
        computed on read, so there is no cached prompt to invalidate.
        """
        if goal is not None:
            self._goal = goal
        if criteria is not None:
            self._criteria = list(criteria)
        if ground_truth is not None:
            self._ground_truth = ground_truth

    @property
    def system_prompt(self) -> str:
        """Static for the whole conversation — deliberately.

        The settled set changes every turn, and anything that varies here sits at
        token position 0, so it would invalidate the cached prefix (system +
        transcript + tool schemas) on every judgement. The dynamic part is
        rendered into the trailing user message by `evaluate` instead, where it
        lands past the common prefix and splits nothing.
        """
        criteria_text = self._format_criteria()

        ground_truth_text = ''
        if self._ground_truth:
            ground_truth_text = f'\n\nGROUND TRUTH (use this to score factual_accuracy):\n{delimit(self._ground_truth)}'

        return f"{DEFAULT_JUDGE_PROMPT}\n\n---\n\nUSER'S GOAL: {delimit(self._goal)}\n\nEVALUATION CRITERIA:\n{criteria_text}{ground_truth_text}"

    async def evaluate(self, messages: list[Message]) -> Judgment:
        """Evaluate a conversation and decide next action."""
        eval_messages = [
            *messages,
            Message(
                role='user',
                content='Evaluate the conversation above. Should it continue or end? Use the appropriate tool.'
                + self._settled_note(),
            ),
        ]

        # volatile_tail=1: the instruction above is rebuilt every turn, so it is not
        # part of the cacheable prefix. Marking it would cost a full-transcript write
        # per judgement and read none of it back (see common.prompt_cache).
        result = await self._call_llm(
            eval_messages, temperature=0.0, tools=JUDGE_TOOLS, llm_purpose='judge', volatile_tail=1
        )
        return self._parse_judgment(result)

    # ---------------------------------------------------------------------------
    # Private helpers
    # ---------------------------------------------------------------------------

    def _parse_judgment(self, result: LLMResult) -> Judgment:
        tool_calls = result.tool_calls

        if not tool_calls:
            content = (result.content or '')[:200]
            logger.warning(
                'JudgeAgent: No tool call in response. Content: %s. Defaulting to TERMINATE.',
                content,
            )
            return _safety_terminate('Judge failed to make explicit decision - terminating for safety')

        tool_call = tool_calls[0]
        function_name = tool_call.function.name
        model = _TOOL_MODELS.get(function_name)
        if model is None:
            logger.warning('JudgeAgent: Unknown function %s - terminating for safety', function_name)
            return _safety_terminate(f"Unknown function '{function_name}' - terminating for safety")

        args = self._parse_tool_args(model, tool_call.function.arguments)
        if args is None:
            return _safety_terminate('Failed to parse judgment decision - terminating for safety')

        # Settled criteria are excluded from the audit on purpose, so an empty
        # payload is only suspicious while something is still unsettled. `None`
        # (told us nothing) and `[]` (claims it audited and had nothing to report)
        # are documented as different states, but while something is unsettled
        # both are the same failure — the turn carries no evidence — so both warn.
        # Neither marks the run verified: `_CriteriaTracker.observe` ignores an
        # empty list.
        unsettled = len(self._criteria) - len(self._settled)
        if unsettled > 0 and not args.criteria_verdicts:
            logger.warning(
                'JudgeAgent: %s for %d unsettled criteria; this turn contributes no criteria '
                'evidence and must_happen cannot be scored from it.',
                'no criteria_verdicts returned'
                if args.criteria_verdicts is None
                else 'an empty criteria_verdicts list',
                unsettled,
            )
        # Derived from the audit for BOTH tools: neither asks for a free-text
        # rules_broken, and hardcoding [] on continue erased every mid-conversation
        # violation.
        violated = self._violated_ids(args.criteria_verdicts)

        if isinstance(args, FinishConversation):
            # A finish that omits the score entirely still has an obvious reading,
            # and 0.0 is not it.
            score = args.goal_completion_score
            if args.goal_achieved and 'goal_completion_score' not in args.model_fields_set:
                score = 1.0
            return Judgment(
                should_terminate=True,
                reason=args.reason,
                goal_achieved=args.goal_achieved,
                rules_broken=violated,
                goal_completion_score=score,
                criteria_verdicts=args.criteria_verdicts,
                response_quality=args.response_quality,
                hallucination_risk=args.hallucination_risk,
                tone_appropriateness=args.tone_appropriateness,
                factual_accuracy=args.factual_accuracy,
            )

        return Judgment(
            should_terminate=False,
            reason=args.reason,
            goal_achieved=False,
            rules_broken=violated,
            # Partial progress, so max_turns runs get a real score instead of a
            # hardcoded 0 (the judge never reaches finish_conversation there).
            goal_completion_score=args.goal_completion_score,
            criteria_verdicts=args.criteria_verdicts,
            response_quality=args.response_quality,
            hallucination_risk=args.hallucination_risk,
            tone_appropriateness=args.tone_appropriateness,
            factual_accuracy=args.factual_accuracy,
        )

    def _parse_tool_args(self, model: type[_JudgeToolArgs], arguments: str) -> _JudgeToolArgs | None:
        """Validate one tool call's arguments through its model.

        The single place the scenario context is attached, so it cannot be
        forgotten on one of the two tool branches — without it `criterion_id`s
        outside the scenario are well-formed and match nothing.

        Returns ``None`` when the payload cannot be salvaged at all; the caller
        safety-terminates. Per-entry tolerance lives in the model's validators.
        """
        if not isinstance(arguments, str):
            # Some SDK shapes hand back None for a tool call with no arguments.
            logger.warning(
                'JudgeAgent: %s tool call carried no arguments string (got %s).',
                model.__name__,
                type(arguments).__name__,
            )
            return None
        try:
            return model.model_validate_json(
                arguments,
                context={'criteria_count': len(self._criteria)},
            )
        except ValidationError as err:
            # warning, not exception: a model emitting bad JSON is an expected
            # failure mode with a documented fallback, not a bug in this process.
            logger.warning(
                'JudgeAgent: Failed to parse %s arguments: %s (raw: %s)',
                model.__name__,
                _first_error(err),
                arguments[:200],
            )
            return None

    def _violated_ids(self, verdicts: list[CriterionVerdict] | None) -> list[str]:
        """Ids of ``must_not_happen`` criteria the audit says already occurred.

        Keyed on `CriterionVerdict.criterion_id`, never on list position: a partial
        audit is the normal case (settled criteria are skipped), so position ``i``
        is not criterion ``i``.

        ``must_happen`` is excluded on purpose: not-yet-satisfied is not a
        violation mid-conversation, so only the run-level fold in the runner
        turns a never-satisfied one into a failure.
        """
        if not verdicts:
            return []
        occurred = {v.criterion_id for v in verdicts if v.occurred}
        return [
            criterion_id_for(i)
            for i, c in enumerate(self._criteria)
            if c.type == 'must_not_happen' and criterion_id_for(i) in occurred
        ]

    def _settled_note(self) -> str:
        """Render the settled criteria for the trailing user message.

        Occurrence is sticky, so a criterion the runner has seen occur stays
        listed under EVALUATION CRITERIA — the judge still needs it to decide
        whether to stop — but re-auditing it can only restate a settled fact, so
        it is excluded from the audit payload. Sorted for a stable rendering.
        """
        if not self._settled:
            return ''
        ids = ', '.join(sorted(self._settled))
        return f'\n\nALREADY CONFIRMED (occurrence is settled — do not re-report these in criteria_verdicts): {ids}'

    def _format_criteria(self) -> str:
        if not self._criteria:
            return 'No specific criteria defined.'

        must_happen: list[str] = []
        must_not: list[str] = []
        for i, c in enumerate(self._criteria):
            criterion_id = criterion_id_for(i)
            entry = f'- {criterion_id}: {delimit(c.description)} ({c.type})'
            if c.type == 'must_happen':
                must_happen.append(entry)
            elif c.type == 'must_not_happen':
                must_not.append(entry)

        text = ''
        if must_happen:
            text += 'MUST HAPPEN:\n' + '\n'.join(must_happen) + '\n\n'
        if must_not:
            text += 'MUST NOT HAPPEN:\n' + '\n'.join(must_not)

        return text.strip() or 'No specific criteria defined.'

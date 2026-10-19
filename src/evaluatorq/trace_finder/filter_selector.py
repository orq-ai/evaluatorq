"""One bounded classify request for selecting live metadata filters."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal, cast

from loguru import logger
from pydantic import ConfigDict, StrictBool, create_model

from evaluatorq.common.judge import ClassifyOutcome, ClassifyQuestion, ClassifyRequest, ClassifyResponse, run_classify
from evaluatorq.common.llm_call import classify_request_body
from evaluatorq.common.retry import with_retry, without_client_retries
from evaluatorq.common.structured_output import generate_structured, sum_structured_usage
from evaluatorq.contracts import LLMCallConfig, TokenUsage

from .compiler import TimeFilters, WireNumeric, _tighten_strict_bounds
from .debug import enabled as debug_enabled
from .models import FACET_NAMES, FILTER_OR_JUDGMENT_RULE, FacetCatalogue, FacetSelection, NumericFilters

if TYPE_CHECKING:
    from openai import AsyncOpenAI

FILTER_TIMEOUT_MS = 30_000
NO_FILTER_LABEL = 'none'
MAX_FILTER_QUESTIONS = 100
# "tool errors" once became status = error, which emptied the population before any judgment ran.
_STATUS_NOTE = (
    ' Status is the recorded trace status, not an error inside the conversation: "tool errors" or "errors" '
    'alone do not constrain trace status.'
)


class FilterSelectionError(RuntimeError):
    """The classifier did not return a structured metadata-filter selection."""


@dataclass(frozen=True)
class FilterSelectionResult:
    """Resolved metadata filters and the classifier's structured reply."""

    selection: FacetSelection
    response: ClassifyResponse | None = None
    error: str | None = None
    numeric: NumericFilters | None = None
    time: TimeFilters | None = None
    usage: TokenUsage | None = None


async def select_filters(
    client: AsyncOpenAI,
    model: str,
    catalogue: FacetCatalogue,
    query: str,
    *,
    cfg: LLMCallConfig | None = None,
) -> FacetSelection:
    """Ask the classifier one classify request, with binary questions for explicit multi-value facets.

    Retry is owned by ``with_retry`` here; the SDK client's own retry budget is
    disabled so this classify call has exactly one retry layer.
    """

    result = await select_filters_with_response(client, model, catalogue, query, cfg=cfg)
    return result.selection


async def select_filters_with_response(
    client: AsyncOpenAI,
    model: str,
    catalogue: FacetCatalogue,
    query: str,
    *,
    cfg: LLMCallConfig | None = None,
) -> FilterSelectionResult:
    """Select filters and retain the exact structured classifier response for review."""

    normalized = query.strip()
    if not normalized:
        raise ValueError('Enter a semantic trace query before selecting filters.')

    choices, binary_values, questions = _questions_for_catalogue(catalogue, normalized)

    if not questions:
        return FilterSelectionResult(FacetSelection())
    if len(questions) > MAX_FILTER_QUESTIONS:
        raise FilterSelectionError(
            f'trace filter selection needs {len(questions)} questions; maximum is {MAX_FILTER_QUESTIONS}'
        )

    # The rule travels once in the shared state rather than in each of up to 100 questions.
    request = ClassifyRequest(state={'query': normalized, 'rule': FILTER_OR_JUDGMENT_RULE}, questions=questions)
    call_cfg = cfg if cfg is not None else LLMCallConfig(model=model, timeout_ms=FILTER_TIMEOUT_MS)
    if debug_enabled():
        logger.debug(
            'Trace finder filter classify request model={} input={}',
            model,
            json.dumps(classify_request_body(model, request.state, request.questions), ensure_ascii=False),
        )

    async def classify_once() -> ClassifyOutcome:
        outcome = await run_classify(
            client=without_client_retries(client),
            model=model,
            cfg=call_cfg,
            request=request,
        )
        if debug_enabled():
            logger.debug(
                'Trace finder filter classify response model={} output={} error={}',
                model,
                outcome.raw_content or (outcome.response.model_dump_json() if outcome.response is not None else ''),
                outcome.error_message,
            )
        if outcome.error_kind is not None:
            error = outcome.error_exc
            if isinstance(error, Exception):
                raise error
        return outcome

    try:
        outcome = await with_retry(
            classify_once,
            max_attempts=call_cfg.retry_count + 1,
            label=f'filter selection[{model}]',
        )
    except Exception as exc:
        raise FilterSelectionError(str(exc)) from exc
    if outcome.error_kind is not None or outcome.response is None:
        message = outcome.error_message or 'The classifier returned no filter selection.'
        raise FilterSelectionError(message)

    return FilterSelectionResult(
        _selection_from_answers(outcome.response, choices, binary_values, questions),
        outcome.response.model_copy(deep=True),
        usage=outcome.token_usage,
    )


async def select_llm_filters_with_response(
    client: AsyncOpenAI,
    model: str,
    catalogue: FacetCatalogue,
    query: str,
    *,
    now: datetime,
    cfg: LLMCallConfig | None = None,
) -> FilterSelectionResult:
    """Select exact catalogue filters, numeric bounds and UTC time with structured output."""

    normalized = query.strip()
    if not normalized:
        raise ValueError('Enter a semantic trace query before selecting filters.')
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError('now must be timezone-aware')
    current = now.astimezone(timezone.utc)
    choices: dict[str, tuple[str, ...]] = {}
    for name in FACET_NAMES:
        values = tuple(dict.fromkeys(getattr(catalogue, name)))
        no_filter = NO_FILTER_LABEL
        while no_filter in values:
            no_filter += '_'
        choices[name] = (*values, no_filter)
    fields: dict[str, tuple[Any, Any]] = {}
    for name, values in choices.items():
        literal_type = cast('Any', Literal.__getitem__(values))
        fields[name] = (list[literal_type], ...)
    fields.update({
        'numeric': (WireNumeric, ...),
        'calendar_day': (StrictBool, ...),
        'start': (datetime | None, ...),
        'end': (datetime | None, ...),
    })
    response_model = create_model(
        'LLMFilterSelectionWire',
        __config__=ConfigDict(extra='forbid'),
        **cast('Any', fields),
    )
    messages = [
        {
            'role': 'system',
            'content': (
                'Select only metadata values explicitly requested by the user. Every facet answer MUST be one of '
                'its exact offered catalogue strings or its no-filter label. Never infer status or other facets. '
                'For models, a literal provider/namespace prefix in the query may be stripped only to match an '
                'exact offered catalogue value; return that full offered value. Extract inclusive integer token and '
                'duration bounds, keeping fields null unless explicitly constrained. Return calendar time as nullable '
                'ISO-8601 UTC bounds, with null for each absent bound and both null when no time filter is requested. '
                'For whole-calendar-day requests set calendar_day true and use an exclusive next-day UTC midnight '
                'end, including relative days. Set calendar_day false when the request has explicit clock-time '
                'semantics or no calendar-time constraint, and preserve any exact clock-time instants. A date-only '
                f'bound is a whole calendar day, not a time of day. Resolve relative dates against current UTC time: {current.isoformat()}'
            ),
        },
        {'role': 'user', 'content': normalized},
    ]

    usages: list[TokenUsage] = []
    for attempt in range(2):
        result = await generate_structured(
            client,
            model=model,
            messages=messages,
            response_format=response_model,
            max_tokens=1600,
            label='trace_finder.filters',
            api='responses',
            config=cfg,
        )
        if result.usage is not None:
            usages.append(result.usage)
        if result.parsed is None:
            raise FilterSelectionError(f'Filter selector returned no structured result. Raw: {result.raw[:300]}')
        try:
            wire = (
                result.parsed
                if isinstance(result.parsed, response_model)
                else response_model.model_validate(result.parsed)
            )
            selection = FacetSelection.model_validate({
                name: (frozenset() if choices[name][-1] in getattr(wire, name) else frozenset(getattr(wire, name)))
                for name in FACET_NAMES
            })
            numeric = _tighten_strict_bounds(
                normalized,
                NumericFilters.model_validate(wire.numeric.model_dump()),
            )
            time = (
                TimeFilters(start=wire.start, end=wire.end, calendar_day=wire.calendar_day)
                if wire.start is not None or wire.end is not None
                else None
            )
        except Exception as exc:
            if attempt:
                raise FilterSelectionError(f'Filter selector produced an invalid plan: {exc}') from exc
            messages.extend([
                {'role': 'assistant', 'content': result.raw or str(result.parsed)},
                {'role': 'user', 'content': f'Your selection was invalid: {exc}. Correct it.'},
            ])
            continue
        return FilterSelectionResult(
            selection=selection,
            numeric=numeric,
            time=time,
            usage=sum_structured_usage(usages),
        )
    raise AssertionError('unreachable LLM filter retry state')


def _questions_for_catalogue(
    catalogue: FacetCatalogue, query: str
) -> tuple[dict[str, dict[str, str | None]], dict[str, tuple[str, str]], dict[str, ClassifyQuestion]]:
    choices: dict[str, dict[str, str | None]] = {}
    binary_values: dict[str, tuple[str, str]] = {}
    questions: dict[str, ClassifyQuestion] = {}
    for name in FACET_NAMES:
        values = getattr(catalogue, name)
        if not values:
            logger.debug('Skipping empty trace-finder facet catalogue: {}', name)
            continue
        mentioned = [
            (index, value)
            for index, value in enumerate(values)
            if re.search(rf'(?<!\w){re.escape(value)}(?!\w)', query, flags=re.IGNORECASE)
        ]
        if len(mentioned) > 1:
            for index, value in mentioned:
                question_key = f'{name}_{index}'
                binary_values[question_key] = (name, value)
                questions[question_key] = ClassifyQuestion(
                    kind='noul',
                    instructions=(
                        f'Does the query explicitly request {name.replace("_", " ")} value {value!r}? '
                        'Answer yes when it is one of several requested alternatives. '
                        'Follow the rule in state: answer no when the value is only implied by what '
                        'happened in the conversation.'
                    ),
                    state={},
                )
            continue
        choice_map = _choice_map(values)
        choices[name] = choice_map
        no_filter_label = next(label for label, value in choice_map.items() if value is None)
        questions[name] = ClassifyQuestion(
            kind='choice',
            instructions=(
                f'Identify the {name.replace("_", " ")} value explicitly requested by the user query. '
                f'Choose {no_filter_label!r} when the query does not constrain this dimension. '
                f'Follow the rule in state: choose {no_filter_label!r} when the value is only implied by what '
                'happened in the conversation, or when the query is an aggregate question.'
                + (_STATUS_NOTE if name == 'status' else '')
            ),
            criteria={label: _choice_description(name, value) for label, value in choice_map.items()},
            state={},
        )
    return choices, binary_values, questions


def _selection_from_answers(
    response: ClassifyResponse,
    choices: dict[str, dict[str, str | None]],
    binary_values: dict[str, tuple[str, str]],
    questions: dict[str, ClassifyQuestion],
) -> FacetSelection:
    selected: dict[str, set[str]] = {name: set() for name in FACET_NAMES}
    for question_key in questions:
        answer = response.answers.get(question_key)
        if answer is None:
            raise FilterSelectionError(f'The classifier returned no answer for filter dimension: {question_key}')
        if question_key in binary_values:
            name, value = binary_values[question_key]
            if answer.type != 'noul' or answer.noul is None:
                raise FilterSelectionError(
                    f'The classifier returned an invalid {question_key} filter answer type: {answer.type!r}'
                )
            if answer.noul >= 0.5:
                selected[name].add(value)
            continue
        name = question_key
        if answer.type != 'choice':
            raise FilterSelectionError(f'The classifier returned an invalid {name} filter answer type: {answer.type!r}')
        choice = answer.choice
        if choice not in choices[name]:
            raise FilterSelectionError(f'The classifier selected an unavailable {name} choice: {choice!r}')
        value = choices[name][choice]
        if value is not None:
            selected[name].add(value)
    return FacetSelection.model_validate({name: frozenset(values) for name, values in selected.items()})


def _choice_map(values: tuple[str, ...]) -> dict[str, str | None]:
    """Use exact catalogue values as labels, reserving a distinct no-filter choice."""

    choices: dict[str, str | None] = dict.fromkeys(values)
    no_filter_label = NO_FILTER_LABEL
    while no_filter_label in choices:
        no_filter_label += '_'
    return {no_filter_label: None, **{value: value for value in choices}}


def _choice_description(name: str, value: str | None) -> str:
    if value is None:
        return f'Do not filter by {name.replace("_", " ")}.'
    return f'Filter {name.replace("_", " ")} to the exact dataset value {value!r}.'

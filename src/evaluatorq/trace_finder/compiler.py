"""Structured compilation of semantic trace queries into classifier tasks."""

from __future__ import annotations

import json
import re
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictStr, ValidationError

from evaluatorq.common.structured_output import generate_structured

from .debug import enabled as debug_enabled
from .models import MAX_DIMENSIONS, CompiledQuery, LegendItem, NumericFilters, ThresholdSelection, ValueSelection

if TYPE_CHECKING:
    from openai import AsyncOpenAI

    from evaluatorq.contracts import LLMCallConfig

CHOICE_PALETTE = (
    'var(--chart-5)',
    'var(--chart-2)',
    'var(--chart-1)',
    'var(--chart-3)',
    'var(--chart-4)',
)

_STRICT_TOKEN_MIN = re.compile(
    r'\b(?:over|above|more than|greater than)\s+(\d[\d,]*(?:\.\d+)?)\s*([km]?)\s*(?:total\s+)?tokens?\b',
    re.IGNORECASE,
)
_STRICT_TOKEN_MAX = re.compile(
    r'\b(?:under|below|less than|fewer than)\s+(\d[\d,]*(?:\.\d+)?)\s*([km]?)\s*(?:total\s+)?tokens?\b',
    re.IGNORECASE,
)
_STRICT_DURATION_MIN = re.compile(
    r'\b(?:slower than|longer than|over|more than)\s+(\d[\d,]*(?:\.\d+)?)\s*'
    r'(milliseconds?|msecs?|ms|seconds?|secs?|s)\b',
    re.IGNORECASE,
)
_STRICT_DURATION_MAX = re.compile(
    r'\b(?:faster than|shorter than|under|less than|below)\s+(\d[\d,]*(?:\.\d+)?)\s*'
    r'(milliseconds?|msecs?|ms|seconds?|secs?|s)\b',
    re.IGNORECASE,
)

COMPILER_INSTRUCTIONS = """Compile the user's request into zero to three semantic classification dimensions.

Each dimension is one independent judgment of a conversation, shown to the user as its own table
column, and a trace matches only when every dimension matches. Return an empty dimensions list
only when every part of the request is a token or duration bound (extracted below) or names an
exact metadata value: a project, agent, model, provider, status, product, trace type, tool, or time,
which a separate filter step applies. Any descriptive phrase about what the trace is or does needs
a dimension, even when it sounds like a category: "coding agents over 50k tokens" needs a
dimension for "coding agent" plus tokens_min 50001, because no metadata field says which traces are
coding agents. Add dimensions for things that need reading the conversation, such as kind of
work, sentiment, intent, quality, or whether an event happened. Use one dimension
for one judgment and never split a single judgment across several; use two or three only when the
request combines separate judgments. Never add more than three.

Give each dimension a name of one to three words for its column header, such as "Frustrated" or
"Unsupported claim". Generate only its semantic task and matching selection rule. Never generate,
infer, or filter by metadata facets inside a dimension.
Each task judges one complete conversation; do not generate state. The adapter supplies it later.
Use choice_criteria as a list of label/description objects for choice, otherwise null.
Use score_criteria as a list of ordered descriptions for score, otherwise null.
Set noul_threshold to a probability in [0, 1] (0.5 unless another cutoff is needed).

Choose exactly one task kind per dimension:
- choice: provide two to five meaningful, unique labels with exhaustive, non-overlapping
  descriptions; selection.kind is values and every selected value is one of those labels.
- noul: provide a binary semantic judgment; selection.kind is values and its values are booleans.
- score: provide two to ten exhaustive, ordered criterion descriptions from least to most matching;
  selection.kind is threshold with an inclusive gte or lte value from 0.0 to 1.0.

Write precise task instructions and criteria that directly answer the user's semantic request.
The selection lists only answers that satisfy the request: the traces the user is searching for.
Never select a neutral, none, neither, other, normal, or appropriate label, or every label. For
negated requests such as "does not behave", "fails to", or "without X", select the failure or
absence label, not the healthy one. For noul, select [true] when true means the requested thing.

Extract numeric constraints on total tokens and duration into numeric. Bounds are inclusive integer
counts, so strict phrases must move by one unit: "over 20k tokens" → tokens_min: 20001 and
"slower than 30 seconds" → duration_ms_min: 30001. Likewise "under 20k tokens" → tokens_max:
19999. A bare count with no unit ("above 50k") means total tokens. Keep all four numeric fields
null when the query does not mention a token or duration constraint. Never extract project, model,
provider, status, product, trace type, agent, or tool constraints; the classifier handles those."""


class CompileError(RuntimeError):
    """The compiler did not return an OpenAI structured-output document."""


_BOOLEAN_LABELS = MappingProxyType({'yes': True, 'true': True, 'no': False, 'false': False})


class WireCriterion(BaseModel):
    """Fixed keys keep generated label criteria compatible with strict output."""

    model_config = ConfigDict(extra='forbid')

    label: str = Field(min_length=1)
    description: str = Field(min_length=1)


class WireNumeric(BaseModel):
    """The strict wire representation of optional trace numeric constraints."""

    model_config = ConfigDict(extra='forbid')

    tokens_min: int | None = Field(ge=0)
    tokens_max: int | None = Field(ge=0)
    duration_ms_min: int | None = Field(ge=0)
    duration_ms_max: int | None = Field(ge=0)


class WireTask(BaseModel):
    """Compiler-only task schema with no arbitrary dictionaries or trace state."""

    model_config = ConfigDict(extra='forbid')

    kind: Literal['choice', 'noul', 'score']
    instructions: str = Field(min_length=1)
    choice_criteria: list[WireCriterion] | None
    score_criteria: list[str] | None
    noul_threshold: float = Field(ge=0, le=1, allow_inf_nan=False)


class WireValueSelection(BaseModel):
    """Strict compiler wire representation of a value selection."""

    model_config = ConfigDict(extra='forbid')

    kind: Literal['values']
    values: tuple[StrictStr | StrictBool, ...] = Field(min_length=1)


class WireThresholdSelection(BaseModel):
    """Strict compiler wire representation of a threshold selection."""

    model_config = ConfigDict(extra='forbid')

    kind: Literal['threshold']
    operator: Literal['gte', 'lte']
    value: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)


# No `discriminator=`: pydantic renders a discriminated union as `oneOf`, which OpenAI's strict
# structured output rejects ("'oneOf' is not permitted"); a plain union renders as `anyOf`, which
# it accepts, and the `kind` literals still pick the member during validation.
WireSelection = WireValueSelection | WireThresholdSelection


class WireDimension(BaseModel):
    """One strict classifier dimension: its column name, task and inclusion rule."""

    model_config = ConfigDict(extra='forbid')

    name: str = Field(min_length=1, max_length=40)
    task: WireTask
    selection: WireSelection

    def to_domain(self) -> CompiledQuery:
        """Convert one wire dimension into the validated domain dimension."""

        task = self.task
        criteria: dict[str, str] | list[str] | None = None
        if task.kind == 'choice':
            if task.choice_criteria is None or task.score_criteria is not None:
                raise ValueError('choice requires choice_criteria and null score_criteria')
            criteria = {item.label: item.description for item in task.choice_criteria}
            if len(criteria) != len(task.choice_criteria):
                raise ValueError('choice labels must be unique')
        elif task.kind == 'score':
            if task.score_criteria is None or task.choice_criteria is not None:
                raise ValueError('score requires score_criteria and null choice_criteria')
            criteria = task.score_criteria
        elif task.choice_criteria is not None or task.score_criteria is not None:
            raise ValueError('noul requires null choice_criteria and score_criteria')

        if isinstance(self.selection, WireValueSelection):
            selection_values = self.selection.values
            if task.kind == 'noul' and all(
                isinstance(value, str) and value.casefold() in _BOOLEAN_LABELS for value in selection_values
            ):
                logger.warning('trace query compiler returned text labels for a boolean selection; converting them')
                selection_values = tuple(_BOOLEAN_LABELS[str(value).casefold()] for value in selection_values)
            if task.kind == 'choice' and criteria is not None and set(selection_values) == set(criteria):
                raise ValueError(
                    f"dimension '{self.name}': selection must list only the answers the user is looking for, "
                    'not every label'
                )
            if task.kind == 'noul' and set(selection_values) == {True, False}:
                raise ValueError(
                    f"dimension '{self.name}': selection must list only the answers the user is looking for, "
                    'not both true and false'
                )
            selection: ValueSelection | ThresholdSelection = ValueSelection(kind='values', values=selection_values)
        else:
            selection = ThresholdSelection.model_validate(self.selection.model_dump())

        return CompiledQuery.model_validate({
            'name': self.name.strip() or 'AI match',
            'task': {
                'kind': task.kind,
                'instructions': task.instructions,
                'criteria': criteria,
                'noul_threshold': task.noul_threshold,
                'state': {},
            },
            'selection': selection,
        })


class CompilerWireQuery(BaseModel):
    """Strict response format converted into the separately validated domain models."""

    model_config = ConfigDict(extra='forbid')

    dimensions: list[WireDimension]
    numeric: WireNumeric

    def to_domain(self) -> tuple[tuple[CompiledQuery, ...], NumericFilters]:
        """Convert the wire document into the shared semantic and numeric contracts."""

        dimensions = self.dimensions
        if len(dimensions) > MAX_DIMENSIONS:
            logger.warning(
                'trace query compiler returned {} dimensions; keeping the first {}', len(dimensions), MAX_DIMENSIONS
            )
            dimensions = dimensions[:MAX_DIMENSIONS]
        numeric = NumericFilters.model_validate(self.numeric.model_dump())
        return tuple(dimension.to_domain() for dimension in dimensions), numeric


class CompiledPlan(BaseModel):
    """The compiled classifier dimensions (zero to three) together with their pre-source numeric filters."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    dimensions: tuple[CompiledQuery, ...] = Field(max_length=MAX_DIMENSIONS)
    numeric: NumericFilters


async def compile_query(
    client: AsyncOpenAI,
    model: str,
    query: str,
    *,
    cfg: LLMCallConfig | None = None,
) -> CompiledPlan:
    """Use shared structured output to compile a question into numeric bounds and zero to three classifier dimensions."""

    normalized = query.strip()
    if not normalized:
        raise ValueError('Enter a semantic trace query before compiling.')

    messages = [
        {'role': 'system', 'content': COMPILER_INSTRUCTIONS},
        {'role': 'user', 'content': normalized},
    ]
    if debug_enabled():
        logger.debug(
            'Trace finder compiler request model={} messages={}', model, json.dumps(messages, ensure_ascii=False)
        )
    for attempt in range(2):
        result = await generate_structured(
            client,
            model=model,
            messages=messages,
            response_format=CompilerWireQuery,
            max_tokens=2000,
            label='trace_finder.compile',
            api='responses',
            config=cfg,
        )
        if result.parsed is None:
            if debug_enabled():
                logger.debug('Trace finder compiler response model={} raw={}', model, result.raw)
            raise CompileError(f'Compiler returned no structured task. Raw: {result.raw[:300]}')

        wire = (
            result.parsed
            if isinstance(result.parsed, CompilerWireQuery)
            else CompilerWireQuery.model_validate(result.parsed)
        )
        if debug_enabled():
            logger.debug('Trace finder compiler response model={} output={}', model, wire.model_dump_json())
        try:
            dimensions, numeric = wire.to_domain()
            numeric = _tighten_strict_bounds(normalized, numeric)
            return CompiledPlan(dimensions=dimensions, numeric=numeric)
        except (ValidationError, ValueError) as exc:
            if attempt:
                raise CompileError(f'Compiler produced an invalid plan: {exc}') from exc
            logger.warning('Trace query compiler returned an invalid plan; retrying: {}', exc)
            messages = [
                *messages,
                {'role': 'assistant', 'content': result.raw or wire.model_dump_json()},
                {'role': 'user', 'content': f'Your plan was invalid: {exc}. Correct it and return a valid plan.'},
            ]
    raise AssertionError('unreachable compiler retry state')


def _tighten_strict_bounds(query: str, numeric: NumericFilters) -> NumericFilters:
    """Tighten only metadata bounds the compiler identified; conversation durations are semantic text."""

    updates: dict[str, int] = {}
    for match in _STRICT_TOKEN_MIN.finditer(query):
        if numeric.tokens_min is None:
            continue
        multiplier = {'': 1, 'k': 1_000, 'm': 1_000_000}[match.group(2).lower()]
        boundary = Decimal(match.group(1).replace(',', '')) * multiplier
        minimum = int(boundary.to_integral_value(rounding=ROUND_FLOOR)) + 1
        updates['tokens_min'] = max(updates.get('tokens_min', numeric.tokens_min or 0), minimum)
    for match in _STRICT_TOKEN_MAX.finditer(query):
        if numeric.tokens_max is None:
            continue
        multiplier = {'': 1, 'k': 1_000, 'm': 1_000_000}[match.group(2).lower()]
        boundary = Decimal(match.group(1).replace(',', '')) * multiplier
        maximum = int(boundary.to_integral_value(rounding=ROUND_CEILING)) - 1
        updates['tokens_max'] = min(updates.get('tokens_max', numeric.tokens_max), maximum)
    for match in _STRICT_DURATION_MIN.finditer(query):
        if numeric.duration_ms_min is None:
            continue
        unit = match.group(2).lower()
        multiplier = 1 if unit.startswith(('milli', 'msec')) or unit == 'ms' else 1_000
        boundary = Decimal(match.group(1).replace(',', '')) * multiplier
        minimum = int(boundary.to_integral_value(rounding=ROUND_FLOOR)) + 1
        updates['duration_ms_min'] = max(updates.get('duration_ms_min', numeric.duration_ms_min or 0), minimum)
    for match in _STRICT_DURATION_MAX.finditer(query):
        if numeric.duration_ms_max is None:
            continue
        unit = match.group(2).lower()
        multiplier = 1 if unit.startswith(('milli', 'msec')) or unit == 'ms' else 1_000
        boundary = Decimal(match.group(1).replace(',', '')) * multiplier
        maximum = int(boundary.to_integral_value(rounding=ROUND_CEILING)) - 1
        updates['duration_ms_max'] = min(updates.get('duration_ms_max', numeric.duration_ms_max), maximum)
    if not updates:
        return numeric
    return NumericFilters.model_validate({**numeric.model_dump(), **updates})


def classification_legend(compiled: CompiledQuery) -> tuple[LegendItem, ...]:
    """Return stable legend entries for the task's answer shape and match rule."""

    task = compiled.task
    if task.kind == 'choice':
        criteria = task.criteria
        if not isinstance(criteria, dict):
            raise RuntimeError('validated choice tasks must have label criteria')
        return tuple(LegendItem(label=label, color=CHOICE_PALETTE[index]) for index, label in enumerate(criteria))
    if task.kind == 'noul':
        return (
            LegendItem(label='false', color='var(--chart-2)'),
            LegendItem(label='true', color='var(--chart-5)'),
        )

    threshold = compiled.selection
    if threshold.kind != 'threshold':
        raise RuntimeError('validated score tasks must have threshold selections')
    return (
        LegendItem(
            label='0.0 → 1.0',
            color='linear-gradient(90deg, var(--chart-2), var(--chart-5))',
            kind='gradient',
        ),
        LegendItem(
            label=f'{threshold.operator} {threshold.value:g}',
            color='var(--text-strong)',
            kind='threshold',
            threshold=threshold.value,
        ),
    )

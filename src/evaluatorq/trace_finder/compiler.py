"""Structured compilation of semantic trace queries into classifier tasks."""

from __future__ import annotations

import re
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from typing import TYPE_CHECKING, Literal

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictStr, ValidationError

from evaluatorq.common.structured_output import generate_structured

from .models import CompiledQuery, LegendItem, NumericFilters, ThresholdSelection, ValueSelection

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

COMPILER_INSTRUCTIONS = """Compile the user's request into one semantic classification task.

Generate only a semantic task and its matching selection rule. Never generate, infer, or
filter by metadata facets such as project, model, provider, status, product, trace type, or time.
The task judges one complete conversation; do not generate state. The adapter supplies it later.
Use choice_criteria as a list of label/description objects for choice, otherwise null.
Use score_criteria as a list of ordered descriptions for score, otherwise null.
Set noul_threshold to a probability in [0, 1] (0.5 unless another cutoff is needed).

Choose exactly one task kind:
- choice: provide two to five meaningful, unique labels with exhaustive, non-overlapping
  descriptions; selection.kind is values and every selected value is one of those labels.
- noul: provide a binary semantic judgment; selection.kind is values and its values are booleans.
- score: provide two to ten exhaustive, ordered criterion descriptions from least to most matching;
  selection.kind is threshold with an inclusive gte or lte value from 0.0 to 1.0.

Write precise task instructions and criteria that directly answer the user's semantic request.

Extract numeric constraints on total tokens and duration into numeric. Bounds are inclusive integer
counts, so strict phrases must move by one unit: "over 20k tokens" → tokens_min: 20001 and
"slower than 30 seconds" → duration_ms_min: 30001. Likewise "under 20k tokens" → tokens_max:
19999. Keep all four numeric fields
null when the query does not mention a token or duration constraint. Never extract project, model,
provider, status, product, trace type, agent, or tool constraints; the classifier handles those."""


class CompileError(RuntimeError):
    """The compiler did not return an OpenAI structured-output document."""


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


class CompilerWireQuery(BaseModel):
    """Strict response format converted into the separately validated domain models."""

    model_config = ConfigDict(extra='forbid')

    task: WireTask
    selection: WireSelection
    numeric: WireNumeric

    def to_domain(self) -> tuple[CompiledQuery, NumericFilters]:
        """Convert the wire document into the shared semantic and numeric contracts."""

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
            if task.kind == 'noul' and all(value in ('yes', 'no') for value in selection_values):
                logger.warning('trace query compiler returned yes/no labels for a boolean selection; converting them')
                selection_values = tuple(value == 'yes' for value in selection_values)
            selection: ValueSelection | ThresholdSelection = ValueSelection(kind='values', values=selection_values)
        else:
            selection = ThresholdSelection.model_validate(self.selection.model_dump())

        compiled = CompiledQuery.model_validate({
            'task': {
                'kind': task.kind,
                'instructions': task.instructions,
                'criteria': criteria,
                'noul_threshold': task.noul_threshold,
                'state': {},
            },
            'selection': selection,
        })
        numeric = NumericFilters.model_validate(self.numeric.model_dump())
        return compiled, numeric


class CompiledPlan(BaseModel):
    """A compiled semantic task together with its pre-source numeric filters."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    compiled: CompiledQuery
    numeric: NumericFilters


async def compile_query(
    client: AsyncOpenAI,
    model: str,
    query: str,
    *,
    cfg: LLMCallConfig | None = None,
) -> CompiledPlan:
    """Use shared structured output to compile one semantic trace query."""

    normalized = query.strip()
    if not normalized:
        raise ValueError('Enter a semantic trace query before compiling.')

    result = await generate_structured(
        client,
        model=model,
        messages=[
            {'role': 'system', 'content': COMPILER_INSTRUCTIONS},
            {'role': 'user', 'content': normalized},
        ],
        response_format=CompilerWireQuery,
        max_tokens=2000,
        label='trace_finder.compile',
        api='responses',
        config=cfg,
    )
    if result.parsed is None:
        raise CompileError(f'Compiler returned no structured task. Raw: {result.raw[:300]}')

    wire = (
        result.parsed
        if isinstance(result.parsed, CompilerWireQuery)
        else CompilerWireQuery.model_validate(result.parsed)
    )
    try:
        compiled, numeric = wire.to_domain()
        numeric = _tighten_strict_bounds(normalized, numeric)
    except ValidationError as exc:
        raise CompileError(f'Compiler produced contradictory numeric bounds: {exc}') from exc
    return CompiledPlan(compiled=compiled, numeric=numeric)


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

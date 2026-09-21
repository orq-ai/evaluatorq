"""Structured compilation of semantic trace queries into JEV tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

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

COMPILER_INSTRUCTIONS = """Compile the user's request into one semantic JEV classification task.

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

Extract numeric constraints on total tokens ("over 20k tokens" → tokens_min: 20000) or duration
("slower than 30 seconds" → duration_ms_min: 30000) into numeric. Keep all four numeric fields
null when the query does not mention a token or duration constraint. Never extract project, model,
provider, status, product, trace type, agent, or tool constraints; JEV handles those."""


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


class CompilerWireQuery(BaseModel):
    """Strict response format converted into the separately validated domain models."""

    model_config = ConfigDict(extra='forbid')

    task: WireTask
    selection: ValueSelection | ThresholdSelection
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

        compiled = CompiledQuery.model_validate({
            'task': {
                'kind': task.kind,
                'instructions': task.instructions,
                'criteria': criteria,
                'noul_threshold': task.noul_threshold,
                'state': {},
            },
            'selection': self.selection.model_dump(),
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
        config=cfg,
    )
    if result.parsed is None:
        raise CompileError(f'Compiler returned no structured task. Raw: {result.raw[:300]}')

    wire = (
        result.parsed
        if isinstance(result.parsed, CompilerWireQuery)
        else CompilerWireQuery.model_validate(result.parsed)
    )
    compiled, numeric = wire.to_domain()
    return CompiledPlan(compiled=compiled, numeric=numeric)


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

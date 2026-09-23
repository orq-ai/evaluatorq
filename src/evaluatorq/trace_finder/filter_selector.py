"""One bounded JEV classify request for selecting live metadata filters."""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from evaluatorq.common.judge import ClassifyOutcome, ClassifyQuestion, ClassifyRequest, run_classify
from evaluatorq.common.retry import with_retry, without_client_retries
from evaluatorq.contracts import LLMCallConfig

from .models import FACET_NAMES, FacetCatalogue, FacetSelection

if TYPE_CHECKING:
    from openai import AsyncOpenAI

FILTER_TIMEOUT_MS = 30_000
NO_FILTER_LABEL = 'none'


class FilterSelectionError(RuntimeError):
    """JEV did not return a structured metadata-filter selection."""


async def select_filters(
    client: AsyncOpenAI,
    model: str,
    catalogue: FacetCatalogue,
    query: str,
    *,
    cfg: LLMCallConfig | None = None,
) -> FacetSelection:
    """Ask JEV one classify question per non-empty catalogue dimension.

    Retry is owned by ``with_retry`` here; the SDK client's own retry budget is
    disabled so this classify call has exactly one retry layer.
    """

    normalized = query.strip()
    if not normalized:
        raise ValueError('Enter a semantic trace query before selecting filters.')

    choices: dict[str, dict[str, str | None]] = {}
    questions: dict[str, ClassifyQuestion] = {}
    for name in FACET_NAMES:
        values = getattr(catalogue, name)
        if not values:
            logger.debug('Skipping empty trace-finder facet catalogue: {}', name)
            continue
        choice_map = _choice_map(name, values)
        choices[name] = choice_map
        questions[name] = ClassifyQuestion(
            kind='choice',
            instructions=(
                f'Identify the {name.replace("_", " ")} value explicitly requested by the user query. '
                f'Choose {NO_FILTER_LABEL!r} when the query does not constrain this dimension. '
                'Do not infer a metadata constraint from the semantic classification request.'
            ),
            criteria={label: _choice_description(name, value) for label, value in choice_map.items()},
            state={},
        )

    if not questions:
        return FacetSelection()

    request = ClassifyRequest(state={'query': normalized}, questions=questions)
    call_cfg = cfg if cfg is not None else LLMCallConfig(model=model, timeout_ms=FILTER_TIMEOUT_MS)

    async def classify_once() -> ClassifyOutcome:
        outcome = await run_classify(
            client=without_client_retries(client),
            model=model,
            cfg=call_cfg,
            request=request,
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
        message = outcome.error_message or 'JEV returned no filter selection.'
        raise FilterSelectionError(message)

    selected: dict[str, frozenset[str]] = {name: frozenset() for name in FACET_NAMES}
    for name in questions:
        answer = outcome.response.answers.get(name)
        if answer is None:
            raise FilterSelectionError(f'JEV returned no answer for filter dimension: {name}')
        if answer.type != 'choice':
            raise FilterSelectionError(f'JEV returned an invalid {name} filter answer type: {answer.type!r}')
        choice = answer.choice
        if choice not in choices[name]:
            raise FilterSelectionError(f'JEV selected an unavailable {name} choice: {choice!r}')
        value = choices[name][choice]
        selected[name] = frozenset() if value is None else frozenset({value})
    return FacetSelection.model_validate(selected)


def _choice_map(name: str, values: tuple[str, ...]) -> dict[str, str | None]:
    """Use short stable labels while keeping exact dataset values in descriptions."""

    return {NO_FILTER_LABEL: None, **{f'{name}_{index}': value for index, value in enumerate(values)}}


def _choice_description(name: str, value: str | None) -> str:
    if value is None:
        return f'Do not filter by {name.replace("_", " ")}.'
    return f'Filter {name.replace("_", " ")} to the exact dataset value {value!r}.'

"""
Built-in evaluators for evaluatorq.

This module provides commonly used evaluators that can be used with the evaluatorq framework.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from evaluatorq.common.orq_client import resolve_orq_client
from evaluatorq.common.output_adapters import output_to_text
from evaluatorq.contracts import Message, content_to_text
from evaluatorq.types import EvaluationResult

if TYPE_CHECKING:
    from .types import Evaluator, ScorerParameter


class _OrqEvalsAPI(Protocol):
    """The one method ``orq_evaluator`` calls — kept minimal so a test double
    only has to implement this, not the whole Orq SDK client. Parameters are
    named to match ``invoke_params`` below and typed loosely with ``Any``:
    the real ``orq_ai_sdk.evals.Evals.invoke_async`` also accepts extra
    optional keywords (``retries``, ``server_url``, ...) this scorer never
    passes, and a keyword-only protocol method is satisfied by an
    implementation with additional optional parameters."""

    async def invoke_async(
        self,
        *,
        id: str,  # noqa: A002  # matches orq_ai_sdk.evals.Evals.invoke_async's keyword name
        query: Any = None,
        output: Any = None,
        reference: Any = None,
        retrievals: Any = None,
        messages: Any = None,
        model: Any = None,
    ) -> Any: ...


class _OrqClient(Protocol):
    """Structural type for the ``client=`` override: anything with an ``evals``
    attribute exposing ``invoke_async`` (the real ``orq_ai_sdk.Orq``, or a test
    double). ``evals`` is a read-only property, not a plain attribute, so the
    protocol stays covariant — a plain mutable attribute would force every
    test double's ``evals`` type to match ``_OrqEvalsAPI`` invariantly."""

    @property
    def evals(self) -> _OrqEvalsAPI: ...


_ORQ_EVALUATOR_RESPONSE_TYPES = frozenset({
    'boolean',
    'string',
    'number',
    'string_array',
    'rouge_n',
    'bert_score',
    'llm_evaluator',
    'http_eval',
    'structured',
})


def _orq_response_payload(response: Any) -> dict[str, Any]:
    if isinstance(response, dict):
        return response
    model_dump = getattr(response, 'model_dump', None)
    if callable(model_dump):
        payload = model_dump(mode='json')
        if isinstance(payload, dict):
            return payload
    raise ValueError(f'Orq evaluator returned unsupported response shape {type(response).__name__}.')


def _orq_response_result(response: Any) -> EvaluationResult:
    payload = _orq_response_payload(response)
    response_type = payload.get('type')
    if not isinstance(response_type, str) or response_type not in _ORQ_EVALUATOR_RESPONSE_TYPES:
        raise ValueError(f'Orq evaluator returned unsupported response shape {response_type!r}.')
    raw_value = payload.get('values') if response_type == 'string_array' else payload.get('value')
    explanation: str | None = None
    if response_type in {'llm_evaluator', 'http_eval'} and isinstance(raw_value, dict):
        explanation_value = raw_value.get('explanation')
        explanation = explanation_value if isinstance(explanation_value, str) else None
        raw_value = raw_value.get('value')
    if response_type == 'string_array':
        raw_value = {'values': raw_value}
    if raw_value is None or not isinstance(raw_value, (str, int, float, bool, dict)):
        raise ValueError(f'Orq evaluator returned unsupported response shape for {response_type!r}.')
    return EvaluationResult.model_validate({
        'value': raw_value,
        'explanation': explanation,
        'pass': raw_value if isinstance(raw_value, bool) else None,
        'raw_output': payload,
    })


def orq_evaluator(
    *,
    evaluator_id: str,
    model: str | None = None,
    client: _OrqClient | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> Evaluator:
    """Create an evaluatorq scorer backed by an Orq evaluator.

    ``evaluator_id`` is required and keyword-only. It identifies an evaluator
    that already exists in Orq. The optional ``model`` is an invocation-time
    override for LLM-based evaluators; it is not needed for deterministic
    built-in evaluators, and a configured LLM evaluator can use its Orq model
    when this is omitted. When omitted, ``model`` is not sent to the SDK at
    all.

    The evaluatorq result name is derived, never hand-named, from everything
    that varies which judge actually runs: ``orq:<evaluator_id>`` when
    ``model`` is omitted, ``orq:<evaluator_id>@<model>`` when it is set. This
    keeps two ``orq_evaluator`` calls that differ only by ``model`` — the
    standard way to compare judges — distinguishable in the results table and
    the uploaded experiment; a single fixed ``orq:<evaluator_id>`` name would
    have merged them into one indistinguishable stream.

    ``client`` accepts an already-built Orq SDK client (or a test double
    shaped like one). When omitted, one is resolved from ``api_key`` or
    ``ORQ_API_KEY`` — and ``base_url`` or ``ORQ_BASE_URL``, for a self-hosted
    deployment — the first time the scorer actually runs, and reused for
    every later datapoint scored by this same ``orq_evaluator(...)`` call.
    Resolving lazily — inside the scorer, not at factory time — preserves the
    property that constructing an evaluator that is never invoked does no
    work and raises nothing; caching that resolution after the first call
    keeps ``evaluate()``'s default ``datapoint_parallelism`` (10 concurrent
    rows, or more across a large run) from building one ``Orq(...)`` client
    and one unclosed connection pool per row, and from deferring a missing
    ``ORQ_API_KEY`` or missing ``[orq]`` extra until after every row has
    already run and billed.

    The scorer sends the latest user turn as ``query``, evaluatorq's job output
    as ``output``, ``expected_output`` as the optional human/reference answer,
    and preserves prior assistant/tool trajectory messages. Orq invocation trace
    IDs remain in ``EvaluationResult.raw_output`` for local use; evaluatorq's
    Experiment uploader intentionally strips raw evaluator output. The Orq SDK
    owns retries for this call; evaluatorq does not add another retry layer.
    """

    cached_client: _OrqClient | None = client

    async def scorer(params: ScorerParameter) -> EvaluationResult:
        nonlocal cached_client
        data = params['data']
        raw_messages = data.inputs.get('messages') or []
        messages = [
            message if isinstance(message, Message) else Message.model_validate(message) for message in raw_messages
        ]
        explicit_query = data.inputs.get('query')
        query = explicit_query if isinstance(explicit_query, str) else None
        latest_user_index = next(
            (index for index in range(len(messages) - 1, -1, -1) if messages[index].role == 'user'),
            None,
        )
        if query is None and latest_user_index is not None:
            query = content_to_text(messages[latest_user_index].content)
        history = messages[:latest_user_index] if latest_user_index is not None else list(messages)
        retrievals_value = data.inputs.get('retrievals')
        retrievals = [str(item) for item in retrievals_value] if isinstance(retrievals_value, list) else None
        invoke_params: dict[str, Any] = {
            'id': evaluator_id,
            'query': query,
            'output': output_to_text(params['output']),
            'reference': None if data.expected_output is None else output_to_text(data.expected_output),
            'retrievals': retrievals,
            'messages': [message.to_chat_completion() for message in history],
        }
        if model is not None:
            invoke_params['model'] = model
        # No `await` runs between the None check and the assignment, so this cache is race-free across rows.
        if cached_client is None:
            cached_client = resolve_orq_client(api_key, base_url)
        response = await cached_client.evals.invoke_async(**invoke_params)
        return _orq_response_result(response)

    result_name = f'orq:{evaluator_id}' if model is None else f'orq:{evaluator_id}@{model}'
    return {'name': result_name, 'scorer': scorer}


def string_contains_evaluator(
    *,
    case_insensitive: bool = True,
    name: str = 'string-contains',
) -> Evaluator:
    """
    Creates an evaluator that checks if the output contains the expected output.
    Uses the data.expected_output from the dataset to compare against.

    Args:
        case_insensitive: Whether the comparison should be case-insensitive
        name: Optional name for the evaluator

    Returns:
        An Evaluator that checks if output contains expected output

    Example:
        ```python
        # Basic usage
        evaluator = string_contains_evaluator()

        # With case-sensitive matching
        strict_evaluator = string_contains_evaluator(case_insensitive=False)

        # With custom name
        my_evaluator = string_contains_evaluator(name="my-contains-check")
        ```
    """

    async def scorer(params: ScorerParameter) -> dict[str, Any]:  # noqa: RUF029  # scorer signature is async by evaluatorq contract
        data = params['data']
        output = params['output']

        expected = output_to_text(data.expected_output)
        actual = output_to_text(output)

        if not expected:
            return {
                'value': 0,
                'pass_': False,
                'explanation': 'No expected output defined',
            }

        expected_normalized = expected.casefold() if case_insensitive else expected
        actual_normalized = actual.casefold() if case_insensitive else actual

        contains = expected_normalized in actual_normalized

        # Truncate strings for readable explanations
        truncated_expected = expected[:100] + '...' if len(expected) > 100 else expected
        truncated_actual = actual[:100] + '...' if len(actual) > 100 else actual

        if contains:
            return {
                'value': 1.0,
                'pass_': True,
                'explanation': f'Output contains "{truncated_expected}"',
            }
        return {
            'value': 0.0,
            'pass_': False,
            'explanation': f'Expected "{truncated_expected}" not found in: "{truncated_actual}"',
        }

    return {
        'name': name,
        'scorer': scorer,
    }


def exact_match_evaluator(
    *,
    case_insensitive: bool = False,
    name: str = 'exact-match',
) -> Evaluator:
    """
    Creates an evaluator that checks if the output exactly matches the expected output.
    Uses the data.expected_output from the dataset to compare against.

    Args:
        case_insensitive: Whether the comparison should be case-insensitive
        name: Optional name for the evaluator

    Returns:
        An Evaluator that checks if output exactly matches expected output

    Example:
        ```python
        # Basic usage (case-sensitive)
        evaluator = exact_match_evaluator()

        # With case-insensitive matching
        loose_evaluator = exact_match_evaluator(case_insensitive=True)
        ```
    """

    async def scorer(params: ScorerParameter) -> dict[str, Any]:  # noqa: RUF029  # scorer signature is async by evaluatorq contract
        data = params['data']
        output = params['output']

        expected = output_to_text(data.expected_output)
        actual = output_to_text(output)

        if not expected:
            return {
                'value': 0,
                'pass_': False,
                'explanation': 'No expected output defined',
            }

        expected_normalized = expected.casefold() if case_insensitive else expected
        actual_normalized = actual.casefold() if case_insensitive else actual

        matches = expected_normalized == actual_normalized

        # Truncate strings for readable explanations
        truncated_expected = expected[:100] + '...' if len(expected) > 100 else expected
        truncated_actual = actual[:100] + '...' if len(actual) > 100 else actual

        if matches:
            return {
                'value': 1.0,
                'pass_': True,
                'explanation': 'Output exactly matches expected output',
            }
        return {
            'value': 0.0,
            'pass_': False,
            'explanation': f'Expected "{truncated_expected}" but got "{truncated_actual}"',
        }

    return {
        'name': name,
        'scorer': scorer,
    }

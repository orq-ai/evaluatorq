"""
Built-in evaluators for evaluatorq.

This module provides commonly used evaluators that can be used with the evaluatorq framework.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from evaluatorq.common.orq_client import resolve_orq_client
from evaluatorq.common.output_adapters import output_to_text
from evaluatorq.contracts import Message, content_to_text
from evaluatorq.types import EvaluationResult

if TYPE_CHECKING:
    from .types import Evaluator, ScorerParameter


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
    client: Any | None = None,
) -> Evaluator:
    """Create an evaluatorq scorer backed by an Orq evaluator.

    ``evaluator_id`` is required and keyword-only. It identifies an evaluator
    that already exists in Orq and determines the evaluatorq result name
    (``orq:<evaluator_id>``); there is no separate name override. The optional
    ``model`` is an invocation-time override for LLM-based
    evaluators; it is not needed for deterministic built-in evaluators, and a
    configured LLM evaluator can use its Orq model when this is omitted. When
    omitted, ``model`` is not sent to the SDK at all.

    The scorer sends the latest user turn as ``query``, evaluatorq's job output
    as ``output``, ``expected_output`` as the optional human/reference answer,
    and preserves prior assistant/tool trajectory messages. Orq invocation trace
    IDs remain in ``EvaluationResult.raw_output`` for local use; evaluatorq's
    Experiment uploader intentionally strips raw evaluator output. The Orq SDK
    owns retries for this call; evaluatorq does not add another retry layer.
    """

    async def scorer(params: ScorerParameter) -> EvaluationResult:
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
        resolved_client = client if client is not None else resolve_orq_client()
        response = await resolved_client.evals.invoke_async(**invoke_params)
        return _orq_response_result(response)

    return {'name': f'orq:{evaluator_id}', 'scorer': scorer}


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

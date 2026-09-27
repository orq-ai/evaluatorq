"""Provider fallback diagnostics must not echo request content into logs."""

from __future__ import annotations

from types import SimpleNamespace

from openai import BadRequestError

from evaluatorq.common.structured_output import _rejection_message


def test_rejection_message_names_schema_clue_without_logging_provider_body() -> None:
    secret = 'private customer transcript\nFAKE LOG ENTRY'
    error = BadRequestError(
        'invalid request',
        response=SimpleNamespace(status_code=400, headers={}, request=None),  # pyright: ignore[reportArgumentType]
        body={'error': {'message': f'Invalid schema for response_format: oneOf is not permitted. {secret}'}},
    )

    diagnostic = _rejection_message(error)

    assert diagnostic == 'schema uses oneOf, which the provider rejected'
    assert secret not in diagnostic

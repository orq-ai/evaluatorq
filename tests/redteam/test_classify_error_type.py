"""classify_error_type: coarse error_type inference — RES-877 review fix."""

from __future__ import annotations

import pytest

from evaluatorq.redteam.contracts import classify_error_type


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        ("Request blocked by content_filter", "content_filter"),
        ("content management policy violation", "content_filter"),
        ("rate limit exceeded", "rate_limit"),
        ("HTTP 429 Too Many Requests", "rate_limit"),
        ("operation timeout", "timeout"),
        ("the request timed out", "timeout"),
        ("connection reset by peer", "network_error"),
        ("Status 503 from upstream", "server_error"),
        ("Status 400 bad request", "client_error"),
    ],
)
def test_classify_matches_patterns(error: str, expected: str) -> None:
    assert classify_error_type(error) == expected


def test_existing_type_passthrough() -> None:
    assert classify_error_type("rate limit", existing_type="preset") == "preset"


def test_empty_or_none_returns_none() -> None:
    assert classify_error_type(None) is None
    assert classify_error_type("") is None


def test_unmatched_returns_unknown() -> None:
    assert classify_error_type("something entirely unexpected") == "unknown"

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


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        # explicit HTTP status wins over the generic 'connection' fallback
        ("Status 503 connection reset by peer", "server_error"),
        ("Status 404 connection closed", "client_error"),
        # 429 embedded in a longer number must NOT trigger rate_limit
        ("request req_42900 failed", "unknown"),
        ("processed 4290 tokens", "unknown"),
        # a bare 3-digit number without a 'status' prefix is not a status error
        ("took 450 ms", "unknown"),
    ],
)
def test_classify_avoids_false_positives(error: str, expected: str) -> None:
    assert classify_error_type(error) == expected

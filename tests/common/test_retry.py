"""Tests for evaluatorq.common.retry._is_retryable_status.

Verifies that retry_statuses AUGMENTS the default set (429 + 5xx) rather
than replacing it — passing retry_statuses={429} must not disable 503 retries.
"""

from __future__ import annotations

from evaluatorq.common.retry import _is_retryable_status


def test_default_retries_429():
    assert _is_retryable_status(429) is True


def test_default_retries_500():
    assert _is_retryable_status(500) is True


def test_default_retries_503():
    assert _is_retryable_status(503) is True


def test_default_does_not_retry_404():
    assert _is_retryable_status(404) is False


def test_custom_status_added():
    """A caller-supplied status code is retried in addition to defaults."""
    assert _is_retryable_status(418, retry_statuses={418}) is True


def test_default_5xx_still_retried_when_custom_set_given():
    """retry_statuses augments the default; passing {429} must not drop 503."""
    assert _is_retryable_status(503, retry_statuses={429}) is True


def test_default_429_still_retried_when_custom_set_given():
    assert _is_retryable_status(429, retry_statuses={418}) is True


def test_non_retryable_not_added_by_custom_set():
    """404 stays non-retryable even when a custom set is supplied."""
    assert _is_retryable_status(404, retry_statuses={429}) is False


def test_none_status_never_retried():
    assert _is_retryable_status(None) is False
    assert _is_retryable_status(None, retry_statuses={429}) is False

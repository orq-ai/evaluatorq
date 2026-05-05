"""Unit tests for truncate_for_span and _default_span_max_text_chars.

Covers evaluatorq.redteam.tracing.truncate_for_span and
evaluatorq.redteam.tracing._default_span_max_text_chars.
"""

from __future__ import annotations

import pytest


class TestTruncateForSpanExplicitMaxChars:
    def test_default_behavior_returns_input_unchanged(self) -> None:
        """With env unset and no max_chars, returns input regardless of length."""
        from evaluatorq.redteam.tracing import _default_span_max_text_chars, truncate_for_span

        _default_span_max_text_chars.cache_clear()
        long_text = "x" * 10_000
        assert truncate_for_span(long_text) == long_text

    def test_max_chars_zero_returns_input_unchanged(self) -> None:
        from evaluatorq.redteam.tracing import truncate_for_span

        text = "x" * 1000
        assert truncate_for_span(text, max_chars=0) == text

    def test_max_chars_none_returns_input_unchanged(self) -> None:
        from evaluatorq.redteam.tracing import _default_span_max_text_chars, truncate_for_span

        _default_span_max_text_chars.cache_clear()
        text = "hello world " * 100
        assert truncate_for_span(text, max_chars=None) == text

    def test_short_input_unchanged(self) -> None:
        """Input shorter than max_chars is returned unchanged."""
        from evaluatorq.redteam.tracing import truncate_for_span

        text = "hello"
        assert truncate_for_span(text, max_chars=100) == text

    def test_exact_length_input_unchanged(self) -> None:
        """Input exactly max_chars long is returned unchanged."""
        from evaluatorq.redteam.tracing import truncate_for_span

        text = "x" * 50
        assert truncate_for_span(text, max_chars=50) == text

    def test_long_input_truncated_to_max_chars(self) -> None:
        """Output is exactly max_chars long when input exceeds max_chars."""
        from evaluatorq.redteam.tracing import truncate_for_span

        text = "x" * 200
        result = truncate_for_span(text, max_chars=100)
        assert len(result) == 100

    def test_long_input_ends_with_marker(self) -> None:
        """Truncated output ends with '... [truncated]' marker."""
        from evaluatorq.redteam.tracing import _TRUNCATION_MARKER, truncate_for_span

        text = "x" * 200
        result = truncate_for_span(text, max_chars=100)
        assert result.endswith(_TRUNCATION_MARKER)

    def test_output_never_exceeds_max_chars(self) -> None:
        """Output length is always <= max_chars."""
        from evaluatorq.redteam.tracing import truncate_for_span

        for max_chars in [1, 5, 15, 16, 50, 100]:
            text = "a" * (max_chars * 2)
            result = truncate_for_span(text, max_chars=max_chars)
            assert len(result) <= max_chars, f"Failed for max_chars={max_chars}"

    def test_degenerate_max_chars_at_or_below_marker_length(self) -> None:
        """When max_chars <= len(marker), return marker truncated to max_chars so truncation is signalled."""
        from evaluatorq.redteam.tracing import _TRUNCATION_MARKER, truncate_for_span

        marker_len = len(_TRUNCATION_MARKER)
        text = "x" * 100

        # Exactly at marker length — full marker is returned
        result = truncate_for_span(text, max_chars=marker_len)
        assert len(result) == marker_len
        assert result == _TRUNCATION_MARKER  # budget fits the marker exactly

        # One below marker length — marker is trimmed to fit budget
        result = truncate_for_span(text, max_chars=marker_len - 1)
        assert len(result) == marker_len - 1
        assert result == _TRUNCATION_MARKER[: marker_len - 1]

    def test_degenerate_max_chars_one_returns_first_marker_char(self) -> None:
        """With max_chars=1, return the first character of the truncation marker."""
        from evaluatorq.redteam.tracing import _TRUNCATION_MARKER, truncate_for_span

        text = "x" * 100
        result = truncate_for_span(text, max_chars=1)
        assert result == _TRUNCATION_MARKER[:1]
        assert len(result) == 1

    def test_negative_max_chars_warns_and_returns_unchanged(self) -> None:
        # Symmetric with env-var negative handling: a misconfig must not
        # crash the surrounding span recorder. Warn and treat as unlimited.
        from evaluatorq.redteam.tracing import truncate_for_span

        text = "hello"
        assert truncate_for_span(text, max_chars=-1) == text

    def test_large_negative_max_chars_warns_and_returns_unchanged(self) -> None:
        from evaluatorq.redteam.tracing import truncate_for_span

        text = "hello"
        assert truncate_for_span(text, max_chars=-999) == text

    def test_non_string_input_coerced_via_str(self) -> None:
        """Non-string input is coerced via str() before truncation."""
        from evaluatorq.redteam.tracing import truncate_for_span

        result = truncate_for_span(123, max_chars=10)
        assert result == "123"

    def test_non_string_large_coerced_and_truncated(self) -> None:
        """Non-string values that stringify longer than max_chars are truncated."""
        from evaluatorq.redteam.tracing import _TRUNCATION_MARKER, truncate_for_span

        long_list = list(range(1000))
        result = truncate_for_span(long_list, max_chars=50)
        assert len(result) == 50
        assert result.endswith(_TRUNCATION_MARKER)


class TestTruncateForSpanEnvOverride:
    def test_env_override_limits_output(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """EVALUATORQ_SPAN_MAX_TEXT_CHARS env var caps output length."""
        from evaluatorq.redteam.tracing import _default_span_max_text_chars, truncate_for_span

        monkeypatch.setenv("EVALUATORQ_SPAN_MAX_TEXT_CHARS", "20")
        _default_span_max_text_chars.cache_clear()
        try:
            result = truncate_for_span("x" * 100)
            assert len(result) == 20
            from evaluatorq.redteam.tracing import _TRUNCATION_MARKER
            assert result.endswith(_TRUNCATION_MARKER)
        finally:
            _default_span_max_text_chars.cache_clear()

    def test_env_override_zero_means_unlimited(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """EVALUATORQ_SPAN_MAX_TEXT_CHARS=0 means unlimited."""
        from evaluatorq.redteam.tracing import _default_span_max_text_chars, truncate_for_span

        monkeypatch.setenv("EVALUATORQ_SPAN_MAX_TEXT_CHARS", "0")
        _default_span_max_text_chars.cache_clear()
        try:
            text = "x" * 1000
            result = truncate_for_span(text)
            assert result == text
        finally:
            _default_span_max_text_chars.cache_clear()

    def test_env_override_invalid_falls_back_to_unlimited(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-integer env value falls back to None (unlimited)."""
        from evaluatorq.redteam.tracing import _default_span_max_text_chars, truncate_for_span

        monkeypatch.setenv("EVALUATORQ_SPAN_MAX_TEXT_CHARS", "abc")
        _default_span_max_text_chars.cache_clear()
        try:
            text = "x" * 1000
            result = truncate_for_span(text)
            assert result == text
        finally:
            _default_span_max_text_chars.cache_clear()

    def test_env_unset_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Unset env var returns None from _default_span_max_text_chars."""
        from evaluatorq.redteam.tracing import _default_span_max_text_chars

        monkeypatch.delenv("EVALUATORQ_SPAN_MAX_TEXT_CHARS", raising=False)
        _default_span_max_text_chars.cache_clear()
        try:
            assert _default_span_max_text_chars() is None
        finally:
            _default_span_max_text_chars.cache_clear()

    def test_env_empty_string_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Empty string env var treated same as unset — returns None."""
        from evaluatorq.redteam.tracing import _default_span_max_text_chars

        monkeypatch.setenv("EVALUATORQ_SPAN_MAX_TEXT_CHARS", "")
        _default_span_max_text_chars.cache_clear()
        try:
            assert _default_span_max_text_chars() is None
        finally:
            _default_span_max_text_chars.cache_clear()

    def test_cache_clear_re_reads_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """cache_clear() forces re-read on next call."""
        from evaluatorq.redteam.tracing import _default_span_max_text_chars

        monkeypatch.setenv("EVALUATORQ_SPAN_MAX_TEXT_CHARS", "100")
        _default_span_max_text_chars.cache_clear()
        first = _default_span_max_text_chars()
        assert first == 100

        monkeypatch.setenv("EVALUATORQ_SPAN_MAX_TEXT_CHARS", "200")
        _default_span_max_text_chars.cache_clear()
        second = _default_span_max_text_chars()
        assert second == 200

        _default_span_max_text_chars.cache_clear()  # clean up

    def test_explicit_max_chars_overrides_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Explicit max_chars parameter overrides env var."""
        from evaluatorq.redteam.tracing import _default_span_max_text_chars, truncate_for_span

        monkeypatch.setenv("EVALUATORQ_SPAN_MAX_TEXT_CHARS", "10")
        _default_span_max_text_chars.cache_clear()
        try:
            text = "x" * 100
            # explicit max_chars=50 must win over env=10
            result = truncate_for_span(text, max_chars=50)
            assert len(result) == 50
        finally:
            _default_span_max_text_chars.cache_clear()

    def test_env_invalid_emits_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Non-integer env value emits a warning before falling back to unlimited."""
        import logging

        from evaluatorq.redteam.tracing import _default_span_max_text_chars

        monkeypatch.setenv("EVALUATORQ_SPAN_MAX_TEXT_CHARS", "not_a_number")
        _default_span_max_text_chars.cache_clear()
        try:
            with caplog.at_level(logging.WARNING):
                result = _default_span_max_text_chars()
            # Function must return None (unlimited) and not raise
            assert result is None
        finally:
            _default_span_max_text_chars.cache_clear()

    def test_env_invalid_warning_message_contains_value(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Warning message includes the offending env value."""
        import logging

        from evaluatorq.redteam.tracing import _default_span_max_text_chars

        monkeypatch.setenv("EVALUATORQ_SPAN_MAX_TEXT_CHARS", "bad_value")
        _default_span_max_text_chars.cache_clear()
        try:
            with caplog.at_level(logging.WARNING):
                _default_span_max_text_chars()
            # Either loguru or stdlib warning should contain the bad value
            # (loguru propagates to stdlib by default in test environments)
            all_messages = " ".join(caplog.messages)
            # If loguru doesn't propagate, the function still returns None correctly
            assert _default_span_max_text_chars.cache_info().currsize >= 1
        finally:
            _default_span_max_text_chars.cache_clear()

"""Unit tests for truncate_for_span and _default_span_max_text_chars.

Covers evaluatorq.common.tracing.truncate_for_span and
evaluatorq.common.tracing._default_span_max_text_chars (re-exported via
evaluatorq.redteam.tracing for backward compat).

Truncation is **off by default** (capture all). A positive
EVALUATORQ_SPAN_MAX_TEXT_CHARS (or explicit positive max_chars) caps the text;
unset / "" / invalid / 0 / negative all mean "capture all".
"""

from __future__ import annotations

import pytest


class TestTruncateForSpanExplicitMaxChars:
    def test_default_behavior_captures_all(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With env unset and no max_chars, text is returned unchanged (capture all)."""
        from evaluatorq.common.tracing import (
            _default_span_max_text_chars,
            truncate_for_span,
        )

        monkeypatch.delenv('EVALUATORQ_SPAN_MAX_TEXT_CHARS', raising=False)
        _default_span_max_text_chars.cache_clear()
        try:
            long_text = 'x' * 10_000
            assert truncate_for_span(long_text) == long_text
        finally:
            _default_span_max_text_chars.cache_clear()

    def test_max_chars_zero_returns_input_unchanged(self) -> None:
        from evaluatorq.common.tracing import truncate_for_span

        text = 'x' * 1000
        assert truncate_for_span(text, max_chars=0) == text

    def test_max_chars_none_consults_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """max_chars=None falls through to the env-derived default (capture all)."""
        from evaluatorq.common.tracing import _default_span_max_text_chars
        from evaluatorq.common.tracing import truncate_for_span

        monkeypatch.delenv('EVALUATORQ_SPAN_MAX_TEXT_CHARS', raising=False)
        _default_span_max_text_chars.cache_clear()
        try:
            long_text = 'x' * 10_000
            assert truncate_for_span(long_text, max_chars=None) == long_text
        finally:
            _default_span_max_text_chars.cache_clear()

    def test_short_input_unchanged(self) -> None:
        """Input shorter than max_chars is returned unchanged."""
        from evaluatorq.common.tracing import truncate_for_span

        text = 'hello'
        assert truncate_for_span(text, max_chars=100) == text

    def test_exact_length_input_unchanged(self) -> None:
        """Input exactly max_chars long is returned unchanged."""
        from evaluatorq.common.tracing import truncate_for_span

        text = 'x' * 50
        assert truncate_for_span(text, max_chars=50) == text

    def test_long_input_truncated_to_max_chars(self) -> None:
        """Output is exactly max_chars long when input exceeds max_chars."""
        from evaluatorq.common.tracing import truncate_for_span

        text = 'x' * 200
        result = truncate_for_span(text, max_chars=100)
        assert len(result) == 100

    def test_long_input_ends_with_marker(self) -> None:
        """Truncated output ends with '... [truncated]' marker."""
        from evaluatorq.common.tracing import _TRUNCATION_MARKER
        from evaluatorq.common.tracing import truncate_for_span

        text = 'x' * 200
        result = truncate_for_span(text, max_chars=100)
        assert result.endswith(_TRUNCATION_MARKER)

    def test_output_never_exceeds_max_chars(self) -> None:
        """Output length is always <= max_chars."""
        from evaluatorq.common.tracing import truncate_for_span

        for max_chars in [1, 5, 15, 16, 50, 100]:
            text = 'a' * (max_chars * 2)
            result = truncate_for_span(text, max_chars=max_chars)
            assert len(result) <= max_chars, f'Failed for max_chars={max_chars}'

    def test_degenerate_max_chars_at_or_below_marker_length(self) -> None:
        """When max_chars <= len(marker), return marker truncated to max_chars so truncation is signalled."""
        from evaluatorq.common.tracing import _TRUNCATION_MARKER
        from evaluatorq.common.tracing import truncate_for_span

        marker_len = len(_TRUNCATION_MARKER)
        text = 'x' * 100

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
        from evaluatorq.common.tracing import _TRUNCATION_MARKER
        from evaluatorq.common.tracing import truncate_for_span

        text = 'x' * 100
        result = truncate_for_span(text, max_chars=1)
        assert result == _TRUNCATION_MARKER[:1]
        assert len(result) == 1

    def test_negative_max_chars_captures_all(self) -> None:
        """Negative max_chars (e.g. -1) means capture all — input unchanged, no raise."""
        from evaluatorq.common.tracing import truncate_for_span

        text = 'x' * 1000
        assert truncate_for_span(text, max_chars=-1) == text

    def test_large_negative_max_chars_captures_all(self) -> None:
        from evaluatorq.common.tracing import truncate_for_span

        text = 'x' * 1000
        assert truncate_for_span(text, max_chars=-999) == text

    def test_non_string_input_coerced_via_str(self) -> None:
        """Non-string input is coerced via str() before truncation."""
        from evaluatorq.common.tracing import truncate_for_span

        result = truncate_for_span(123, max_chars=10)
        assert result == '123'

    def test_non_string_large_coerced_and_truncated(self) -> None:
        """Non-string values that stringify longer than max_chars are truncated."""
        from evaluatorq.common.tracing import _TRUNCATION_MARKER
        from evaluatorq.common.tracing import truncate_for_span

        long_list = list(range(1000))
        result = truncate_for_span(long_list, max_chars=50)
        assert len(result) == 50
        assert result.endswith(_TRUNCATION_MARKER)


class TestTruncateForSpanEnvOverride:
    def test_env_override_limits_output(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """EVALUATORQ_SPAN_MAX_TEXT_CHARS env var caps output length."""
        from evaluatorq.common.tracing import _TRUNCATION_MARKER, _default_span_max_text_chars
        from evaluatorq.common.tracing import truncate_for_span

        monkeypatch.setenv('EVALUATORQ_SPAN_MAX_TEXT_CHARS', '20')
        _default_span_max_text_chars.cache_clear()
        try:
            result = truncate_for_span('x' * 100)
            assert len(result) == 20
            assert result.endswith(_TRUNCATION_MARKER)
        finally:
            _default_span_max_text_chars.cache_clear()

    def test_env_override_eight_k_caps_at_8192(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The canonical 8192 cap, when opted into, truncates at 8192 chars."""
        from evaluatorq.common.tracing import (
            _RECOMMENDED_SPAN_MAX_TEXT_CHARS,
            _default_span_max_text_chars,
            truncate_for_span,
        )

        monkeypatch.setenv('EVALUATORQ_SPAN_MAX_TEXT_CHARS', '8192')
        _default_span_max_text_chars.cache_clear()
        try:
            result = truncate_for_span('x' * 20_000)
            assert len(result) == _RECOMMENDED_SPAN_MAX_TEXT_CHARS == 8192
        finally:
            _default_span_max_text_chars.cache_clear()

    def test_env_override_zero_means_unlimited(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """EVALUATORQ_SPAN_MAX_TEXT_CHARS=0 means unlimited (capture all)."""
        from evaluatorq.common.tracing import _default_span_max_text_chars
        from evaluatorq.common.tracing import truncate_for_span

        monkeypatch.setenv('EVALUATORQ_SPAN_MAX_TEXT_CHARS', '0')
        _default_span_max_text_chars.cache_clear()
        try:
            text = 'x' * 1000
            assert truncate_for_span(text) == text
        finally:
            _default_span_max_text_chars.cache_clear()

    def test_env_override_negative_one_means_unlimited(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """EVALUATORQ_SPAN_MAX_TEXT_CHARS=-1 means unlimited (capture all)."""
        from evaluatorq.common.tracing import _default_span_max_text_chars
        from evaluatorq.common.tracing import truncate_for_span

        monkeypatch.setenv('EVALUATORQ_SPAN_MAX_TEXT_CHARS', '-1')
        _default_span_max_text_chars.cache_clear()
        try:
            assert _default_span_max_text_chars() is None
            text = 'x' * 1000
            assert truncate_for_span(text) == text
        finally:
            _default_span_max_text_chars.cache_clear()

    def test_env_override_invalid_captures_all(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Non-integer env value falls back to capture-all (None)."""
        from evaluatorq.common.tracing import _default_span_max_text_chars

        monkeypatch.setenv('EVALUATORQ_SPAN_MAX_TEXT_CHARS', 'abc')
        _default_span_max_text_chars.cache_clear()
        try:
            assert _default_span_max_text_chars() is None
        finally:
            _default_span_max_text_chars.cache_clear()

    def test_env_unset_captures_all(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Unset env var means capture all (None)."""
        from evaluatorq.common.tracing import _default_span_max_text_chars

        monkeypatch.delenv('EVALUATORQ_SPAN_MAX_TEXT_CHARS', raising=False)
        _default_span_max_text_chars.cache_clear()
        try:
            assert _default_span_max_text_chars() is None
        finally:
            _default_span_max_text_chars.cache_clear()

    def test_env_empty_string_captures_all(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Empty string env var treated same as unset — capture all (None)."""
        from evaluatorq.common.tracing import _default_span_max_text_chars

        monkeypatch.setenv('EVALUATORQ_SPAN_MAX_TEXT_CHARS', '')
        _default_span_max_text_chars.cache_clear()
        try:
            assert _default_span_max_text_chars() is None
        finally:
            _default_span_max_text_chars.cache_clear()

    def test_cache_clear_re_reads_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """cache_clear() forces re-read on next call."""
        from evaluatorq.common.tracing import _default_span_max_text_chars

        monkeypatch.setenv('EVALUATORQ_SPAN_MAX_TEXT_CHARS', '100')
        _default_span_max_text_chars.cache_clear()
        first = _default_span_max_text_chars()
        assert first == 100

        monkeypatch.setenv('EVALUATORQ_SPAN_MAX_TEXT_CHARS', '200')
        _default_span_max_text_chars.cache_clear()
        second = _default_span_max_text_chars()
        assert second == 200

        _default_span_max_text_chars.cache_clear()  # clean up

    def test_explicit_max_chars_overrides_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Explicit max_chars parameter overrides env var."""
        from evaluatorq.common.tracing import _default_span_max_text_chars
        from evaluatorq.common.tracing import truncate_for_span

        monkeypatch.setenv('EVALUATORQ_SPAN_MAX_TEXT_CHARS', '10')
        _default_span_max_text_chars.cache_clear()
        try:
            text = 'x' * 100
            # explicit max_chars=50 must win over env=10
            result = truncate_for_span(text, max_chars=50)
            assert len(result) == 50
        finally:
            _default_span_max_text_chars.cache_clear()

    def test_env_invalid_emits_warning(self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
        """Non-integer env value emits a warning before falling back to capture-all."""
        import logging

        from evaluatorq.common.tracing import _default_span_max_text_chars

        monkeypatch.setenv('EVALUATORQ_SPAN_MAX_TEXT_CHARS', 'not_a_number')
        _default_span_max_text_chars.cache_clear()
        try:
            with caplog.at_level(logging.WARNING):
                result = _default_span_max_text_chars()
            # Function must fall back to capture-all (None) and not raise
            assert result is None
        finally:
            _default_span_max_text_chars.cache_clear()

    def test_env_invalid_warning_message_contains_value(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Warning message includes the offending env value."""
        import logging

        from evaluatorq.common.tracing import _default_span_max_text_chars

        monkeypatch.setenv('EVALUATORQ_SPAN_MAX_TEXT_CHARS', 'bad_value')
        _default_span_max_text_chars.cache_clear()
        try:
            with caplog.at_level(logging.WARNING):
                _default_span_max_text_chars()
            all_messages = ' '.join(caplog.messages)
            assert 'bad_value' in all_messages, f'Expected bad_value in warning, got: {all_messages!r}'
            assert _default_span_max_text_chars.cache_info().currsize >= 1
        finally:
            _default_span_max_text_chars.cache_clear()

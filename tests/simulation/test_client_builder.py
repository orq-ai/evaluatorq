"""Tests for build_simulation_client API-key routing logic.

Covers all resolution branches:
1. config_client passed → returns (client, owned=False), no env reads
2. extra_api_key arg → routes via ORQ_BASE_URL/v2/router, owned=True
3. ORQ_API_KEY env only → routes via ORQ_BASE_URL/v2/router, owned=True
4. OPENAI_API_KEY env only (no ORQ_API_KEY) → base_url=None (OpenAI default), owned=True
5. Both ORQ and OPENAI env vars → ORQ wins (precedence)
6. Neither env var nor client → raises ValueError with helpful message
7. Custom ORQ_BASE_URL env var → used in computed base_url
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build(monkeypatch, *, orq_key=None, openai_key=None, base_url=None, **kwargs):
    """Helper: set/clear env vars and call build_simulation_client."""
    # Always clear both keys unless explicitly provided
    if orq_key is not None:
        monkeypatch.setenv("ORQ_API_KEY", orq_key)
    else:
        monkeypatch.delenv("ORQ_API_KEY", raising=False)

    if openai_key is not None:
        monkeypatch.setenv("OPENAI_API_KEY", openai_key)
    else:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    if base_url is not None:
        monkeypatch.setenv("ORQ_BASE_URL", base_url)
    else:
        monkeypatch.delenv("ORQ_BASE_URL", raising=False)

    from evaluatorq.openresponses.client import build_simulation_client

    return build_simulation_client(**kwargs)


# ---------------------------------------------------------------------------
# Branch 1: config_client passed → not owned, returned as-is
# ---------------------------------------------------------------------------


class TestConfigClientPassthrough:
    def test_returns_injected_client_not_owned(self, monkeypatch):
        """When config_client is provided it must be returned with owned=False."""
        # Ensure no real keys are set so we can confirm env is NOT consulted
        monkeypatch.delenv("ORQ_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        mock_client = MagicMock()
        from evaluatorq.openresponses.client import build_simulation_client

        client, owned = build_simulation_client(mock_client)

        assert client is mock_client
        assert owned is False

    def test_no_asyncopenai_constructed_when_client_injected(self, monkeypatch):
        """AsyncOpenAI constructor must NOT be called when config_client is given."""
        monkeypatch.delenv("ORQ_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        mock_client = MagicMock()

        with patch("openai.AsyncOpenAI") as MockAsyncOpenAI:
            from evaluatorq.openresponses.client import build_simulation_client

            build_simulation_client(mock_client)

        MockAsyncOpenAI.assert_not_called()


# ---------------------------------------------------------------------------
# Branch 2: extra_api_key arg → ORQ routing, owned=True
# ---------------------------------------------------------------------------


class TestExtraApiKey:
    def test_routes_via_orq_router_with_extra_api_key(self, monkeypatch):
        """extra_api_key triggers ORQ router base_url, owned=True."""
        monkeypatch.delenv("ORQ_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ORQ_BASE_URL", raising=False)

        captured: dict[str, Any] = {}

        def fake_async_openai(**kwargs):
            captured.update(kwargs)
            return MagicMock()

        with patch("openai.AsyncOpenAI", side_effect=fake_async_openai):
            from evaluatorq.openresponses.client import build_simulation_client

            _, owned = build_simulation_client(extra_api_key="sk-extra-key")

        assert owned is True
        assert captured.get("base_url") == "https://api.orq.ai/v2/router"
        assert captured.get("api_key") == "sk-extra-key"

    def test_extra_api_key_uses_custom_orq_base_url(self, monkeypatch):
        """extra_api_key uses ORQ_BASE_URL env when set."""
        monkeypatch.delenv("ORQ_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("ORQ_BASE_URL", "https://custom.example.com")

        captured: dict[str, Any] = {}

        def fake_async_openai(**kwargs):
            captured.update(kwargs)
            return MagicMock()

        with patch("openai.AsyncOpenAI", side_effect=fake_async_openai):
            from evaluatorq.openresponses.client import build_simulation_client

            build_simulation_client(extra_api_key="sk-extra-key")

        assert captured.get("base_url") == "https://custom.example.com/v2/router"


# ---------------------------------------------------------------------------
# Branch 3: ORQ_API_KEY env only → ORQ routing, owned=True
# ---------------------------------------------------------------------------


class TestOrqApiKeyEnv:
    def test_routes_via_orq_router_with_env_key(self, monkeypatch):
        """ORQ_API_KEY env var triggers ORQ router base_url, owned=True."""
        monkeypatch.setenv("ORQ_API_KEY", "orq-env-key")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ORQ_BASE_URL", raising=False)

        captured: dict[str, Any] = {}

        def fake_async_openai(**kwargs):
            captured.update(kwargs)
            return MagicMock()

        with patch("openai.AsyncOpenAI", side_effect=fake_async_openai):
            from evaluatorq.openresponses.client import build_simulation_client

            _, owned = build_simulation_client()

        assert owned is True
        assert captured.get("base_url") == "https://api.orq.ai/v2/router"
        assert captured.get("api_key") == "orq-env-key"


# ---------------------------------------------------------------------------
# Branch 4: OPENAI_API_KEY env only → base_url=None (OpenAI default), owned=True
# ---------------------------------------------------------------------------


class TestOpenAIApiKeyEnv:
    def test_base_url_is_none_with_openai_key_only(self, monkeypatch):
        """When only OPENAI_API_KEY is set, base_url must be None."""
        monkeypatch.delenv("ORQ_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-key")
        monkeypatch.delenv("ORQ_BASE_URL", raising=False)

        captured: dict[str, Any] = {}

        def fake_async_openai(**kwargs):
            captured.update(kwargs)
            return MagicMock()

        with patch("openai.AsyncOpenAI", side_effect=fake_async_openai):
            from evaluatorq.openresponses.client import build_simulation_client

            _, owned = build_simulation_client()

        assert owned is True
        assert captured.get("base_url") is None
        assert captured.get("api_key") == "sk-openai-key"


# ---------------------------------------------------------------------------
# Branch 5: Both ORQ and OPENAI env vars → ORQ wins
# ---------------------------------------------------------------------------


class TestOrqTakesPrecedenceOverOpenAI:
    def test_orq_key_takes_precedence_over_openai_key(self, monkeypatch):
        """When both ORQ_API_KEY and OPENAI_API_KEY are set, ORQ routing wins."""
        monkeypatch.setenv("ORQ_API_KEY", "orq-wins")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-loses")
        monkeypatch.delenv("ORQ_BASE_URL", raising=False)

        captured: dict[str, Any] = {}

        def fake_async_openai(**kwargs):
            captured.update(kwargs)
            return MagicMock()

        with patch("openai.AsyncOpenAI", side_effect=fake_async_openai):
            from evaluatorq.openresponses.client import build_simulation_client

            _, owned = build_simulation_client()

        assert owned is True
        # Must route via ORQ, not OpenAI default
        assert captured.get("base_url") == "https://api.orq.ai/v2/router"
        assert captured.get("api_key") == "orq-wins"


# ---------------------------------------------------------------------------
# Branch 6: No env vars and no client → ValueError
# ---------------------------------------------------------------------------


class TestNoCredentialsRaisesValueError:
    def test_raises_value_error_with_helpful_message(self, monkeypatch):
        """No API key and no config_client must raise ValueError with helpful message."""
        monkeypatch.delenv("ORQ_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        from evaluatorq.openresponses.client import build_simulation_client

        with pytest.raises(ValueError, match="ORQ_API_KEY"):
            build_simulation_client()

    def test_error_message_mentions_openai_api_key(self, monkeypatch):
        """ValueError message must also mention OPENAI_API_KEY."""
        monkeypatch.delenv("ORQ_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        from evaluatorq.openresponses.client import build_simulation_client

        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            build_simulation_client()


# ---------------------------------------------------------------------------
# Branch 7: Custom ORQ_BASE_URL → used in computed base_url
# ---------------------------------------------------------------------------


class TestCustomOrqBaseUrl:
    def test_custom_base_url_is_used_when_routing_via_orq(self, monkeypatch):
        """ORQ_BASE_URL env var is incorporated into the router base_url."""
        monkeypatch.setenv("ORQ_API_KEY", "orq-key")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("ORQ_BASE_URL", "https://staging.orq.ai")

        captured: dict[str, Any] = {}

        def fake_async_openai(**kwargs):
            captured.update(kwargs)
            return MagicMock()

        with patch("openai.AsyncOpenAI", side_effect=fake_async_openai):
            from evaluatorq.openresponses.client import build_simulation_client

            build_simulation_client()

        assert captured.get("base_url") == "https://staging.orq.ai/v2/router"

    def test_custom_base_url_not_used_when_openai_key_only(self, monkeypatch):
        """ORQ_BASE_URL must NOT affect routing when only OPENAI_API_KEY is set."""
        monkeypatch.delenv("ORQ_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
        monkeypatch.setenv("ORQ_BASE_URL", "https://staging.orq.ai")

        captured: dict[str, Any] = {}

        def fake_async_openai(**kwargs):
            captured.update(kwargs)
            return MagicMock()

        with patch("openai.AsyncOpenAI", side_effect=fake_async_openai):
            from evaluatorq.openresponses.client import build_simulation_client

            build_simulation_client()

        # ORQ_BASE_URL must be ignored; base_url stays None for direct OpenAI routing
        assert captured.get("base_url") is None

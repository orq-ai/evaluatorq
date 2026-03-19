"""Shared fixtures for red teaming tests."""

import os

import pytest


@pytest.fixture(autouse=True)
def _ensure_llm_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set a dummy OPENAI_API_KEY when no real credentials are present.

    The runner performs early credential validation and raises
    ``CredentialError`` when neither OPENAI_API_KEY nor ORQ_API_KEY is set.
    Unit tests that mock internal execution functions don't need real
    credentials, so we inject a dummy key to satisfy the check.
    """
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ORQ_API_KEY"):
        monkeypatch.setenv("OPENAI_API_KEY", "test-dummy-key")

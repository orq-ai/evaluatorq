"""Regression tests for ENG-1820.

The AgentTarget / _prepare_target paths previously fell back to
``create_async_llm_client()`` even when a caller supplied
``pipeline_config.attacker.client``, silently replacing the configured
client. Both call sites now go through ``_resolve_attacker_llm_client``;
these tests lock in that behaviour.
"""

from unittest.mock import MagicMock, patch

import pytest

from evaluatorq.redteam import runner
from evaluatorq.redteam.contracts import LLMCallConfig, LLMConfig

_RUNNER_CREATE = "evaluatorq.redteam.runner.create_async_llm_client"


@pytest.mark.parametrize("create_if_missing", [True, False])
def test_explicit_llm_client_wins(create_if_missing):
    explicit = MagicMock(name="explicit")
    with patch(_RUNNER_CREATE) as mock_create:
        result = runner._resolve_attacker_llm_client(
            explicit, pipeline_config=None, create_if_missing=create_if_missing
        )
    assert result is explicit
    mock_create.assert_not_called()


@pytest.mark.parametrize("create_if_missing", [True, False])
def test_attacker_client_in_pipeline_config_wins_over_factory(create_if_missing):
    """ENG-1820: a configured attacker client must not be replaced by the factory."""
    custom = MagicMock(name="attacker-client")
    cfg = LLMConfig(attacker=LLMCallConfig(client=custom))

    with patch(_RUNNER_CREATE) as mock_create:
        result = runner._resolve_attacker_llm_client(
            llm_client=None, pipeline_config=cfg, create_if_missing=create_if_missing
        )

    assert result is custom
    mock_create.assert_not_called()


def test_factory_called_when_no_client_and_create_if_missing():
    cfg = LLMConfig()
    sentinel = MagicMock(name="factory-built")
    with patch(_RUNNER_CREATE, return_value=sentinel) as mock_create:
        result = runner._resolve_attacker_llm_client(
            llm_client=None, pipeline_config=cfg, create_if_missing=True
        )

    assert result is sentinel
    mock_create.assert_called_once()
    # The role config must be forwarded so per-role overrides apply.
    assert mock_create.call_args.kwargs.get("role_config") is cfg.attacker


def test_factory_skipped_when_create_if_missing_false():
    """_prepare_target path: only create a client when strategy generation needs one."""
    with patch(_RUNNER_CREATE) as mock_create:
        result = runner._resolve_attacker_llm_client(
            llm_client=None, pipeline_config=None, create_if_missing=False
        )

    assert result is None
    mock_create.assert_not_called()


def test_no_pipeline_config_falls_through_to_factory():
    sentinel = MagicMock(name="factory-built")
    with patch(_RUNNER_CREATE, return_value=sentinel) as mock_create:
        result = runner._resolve_attacker_llm_client(
            llm_client=None, pipeline_config=None, create_if_missing=True
        )

    assert result is sentinel
    mock_create.assert_called_once()
    assert mock_create.call_args.kwargs.get("role_config") is None

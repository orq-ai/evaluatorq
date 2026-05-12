"""Behavioral coverage for the back-compat shim in `redteam/backends/base.py`.

Covers `is_agent_target`, `validate_agent_target`, and `adapt_legacy_target`.
This is the only thing keeping out-of-tree consumers alive across the 1.3
breaking change, so a regression here would silently break every external
integration.
"""

from __future__ import annotations

import pytest

from evaluatorq.redteam.backends.base import (
    adapt_legacy_target,
    is_agent_target,
    validate_agent_target,
)
from evaluatorq.redteam.contracts import SendResult


class _NewStyleTarget:
    async def send_prompt_with_usage(self, prompt: str) -> SendResult:
        return SendResult(text="hi")

    def new(self) -> "_NewStyleTarget":
        return _NewStyleTarget()


class _LegacyTarget:
    """Old API: only `send_prompt() -> str`, no `send_prompt_with_usage`."""

    async def send_prompt(self, prompt: str) -> str:
        return f"echo:{prompt}"

    def new(self) -> "_LegacyTarget":
        return _LegacyTarget()


class _CloneOnlyTarget:
    """Pre-1.3: had `clone()` instead of `new()` — must surface migration error."""

    async def send_prompt(self, prompt: str) -> str:
        return prompt

    def clone(self) -> "_CloneOnlyTarget":
        return _CloneOnlyTarget()


class _SyncLegacyTarget:
    """Legacy target with a *synchronous* `send_prompt()` returning a str."""

    def send_prompt(self, prompt: str) -> str:
        return f"sync:{prompt}"

    def new(self) -> "_SyncLegacyTarget":
        return _SyncLegacyTarget()


class _SlottedLegacyTarget:
    __slots__ = ("_state",)

    def __init__(self) -> None:
        self._state = "x"

    async def send_prompt(self, prompt: str) -> str:
        return prompt

    def new(self) -> "_SlottedLegacyTarget":
        return _SlottedLegacyTarget()


class TestIsAgentTarget:
    def test_new_style_returns_true(self) -> None:
        assert is_agent_target(_NewStyleTarget()) is True

    def test_legacy_only_returns_false(self) -> None:
        assert is_agent_target(_LegacyTarget()) is False

    def test_missing_new_returns_false(self) -> None:
        class _NoNew:
            async def send_prompt_with_usage(self, prompt: str) -> SendResult:
                return SendResult(text="")

        assert is_agent_target(_NoNew()) is False


class TestValidateAgentTarget:
    def test_new_style_does_not_raise(self) -> None:
        validate_agent_target(_NewStyleTarget())

    def test_clone_only_raises_with_migration_message(self) -> None:
        with pytest.raises(TypeError, match=r"clone\(\).*evaluatorq 1\.3"):
            validate_agent_target(_CloneOnlyTarget())

    def test_legacy_with_new_does_not_raise(self) -> None:
        # Legacy `send_prompt`-only target still has `new()` — not the
        # `clone()` migration case, so this should not raise here. The
        # caller is expected to also run the adapter.
        validate_agent_target(_LegacyTarget())


class TestAdaptLegacyTarget:
    def test_already_new_style_returned_unchanged(self) -> None:
        target = _NewStyleTarget()
        assert adapt_legacy_target(target) is target

    def test_no_send_prompt_returned_unchanged(self) -> None:
        class _Bare:
            pass

        bare = _Bare()
        assert adapt_legacy_target(bare) is bare

    def test_legacy_target_emits_deprecation_warning(self) -> None:
        with pytest.warns(DeprecationWarning, match="legacy `send_prompt`"):
            adapt_legacy_target(_LegacyTarget())

    @pytest.mark.asyncio
    async def test_adapted_call_returns_send_result_with_text(self) -> None:
        with pytest.warns(DeprecationWarning):
            target = adapt_legacy_target(_LegacyTarget())
        result = await target.send_prompt_with_usage("ping")  # pyright: ignore[reportAttributeAccessIssue]
        assert isinstance(result, SendResult)
        assert result.text == "echo:ping"
        assert result.usage is None

    @pytest.mark.asyncio
    async def test_sync_legacy_target_adapted(self) -> None:
        """Sync `send_prompt()` legacy targets must be supported, not TypeError."""
        with pytest.warns(DeprecationWarning):
            target = adapt_legacy_target(_SyncLegacyTarget())
        result = await target.send_prompt_with_usage("ping")  # pyright: ignore[reportAttributeAccessIssue]
        assert isinstance(result, SendResult)
        assert result.text == "sync:ping"
        assert result.usage is None

    def test_slotted_legacy_target_raises_attributeerror(self) -> None:
        with pytest.warns(DeprecationWarning):
            with pytest.raises(AttributeError):
                adapt_legacy_target(_SlottedLegacyTarget())

"""Unit tests for the SendResult dataclass from evaluatorq.redteam.contracts."""

from __future__ import annotations

import dataclasses

import pytest

from evaluatorq.redteam.contracts import SendResult, TokenUsage


class TestSendResultConstruction:
    def test_required_field_only(self) -> None:
        result = SendResult(text="hello")
        assert result.text == "hello"
        assert result.usage is None
        assert result.model is None

    def test_all_fields(self) -> None:
        usage = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15, calls=1)
        result = SendResult(text="response text", usage=usage, model="gpt-4o-mini")
        assert result.text == "response text"
        assert result.usage is usage
        assert result.model == "gpt-4o-mini"

    def test_usage_none_explicit(self) -> None:
        result = SendResult(text="hi", usage=None)
        assert result.usage is None

    def test_model_none_explicit(self) -> None:
        result = SendResult(text="hi", model=None)
        assert result.model is None

    def test_empty_text(self) -> None:
        result = SendResult(text="")
        assert result.text == ""


class TestSendResultFrozen:
    def test_cannot_assign_text(self) -> None:
        result = SendResult(text="original")
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            result.text = "mutated"  # pyright: ignore[reportAttributeAccessIssue]

    def test_cannot_assign_usage(self) -> None:
        result = SendResult(text="original")
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            result.usage = TokenUsage()  # pyright: ignore[reportAttributeAccessIssue]

    def test_cannot_assign_model(self) -> None:
        result = SendResult(text="original")
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            result.model = "gpt-5"  # pyright: ignore[reportAttributeAccessIssue]


class TestSendResultSlots:
    def test_cannot_add_arbitrary_attribute(self) -> None:
        """slots=True prevents adding new attributes at runtime.

        Python 3.10 raises TypeError for new-attribute assignment on frozen+slotted
        dataclasses; Python 3.11+ raises AttributeError. Accept both.
        """
        result = SendResult(text="hi")
        with pytest.raises((AttributeError, TypeError)):
            result.arbitrary_new_attr = "should fail"  # pyright: ignore[reportAttributeAccessIssue]


class TestSendResultIsInstance:
    def test_isinstance_check(self) -> None:
        """isinstance(SendResult(...), SendResult) must be True — matters for shim."""
        result = SendResult(text="check")
        assert isinstance(result, SendResult)

    def test_not_instance_of_other_types(self) -> None:
        result = SendResult(text="check")
        assert not isinstance(result, str)
        assert not isinstance(result, dict)

    def test_is_dataclass(self) -> None:
        assert dataclasses.is_dataclass(SendResult)
        assert dataclasses.is_dataclass(SendResult(text="x"))

"""Unit tests for the red teaming exception hierarchy."""

import pytest

from evaluatorq.contracts import AgentResponse, AgentResponseError
from evaluatorq.redteam.exceptions import (
    BackendError,
    CancelledError,
    CredentialError,
    RedTeamError,
    TargetResponseError,
)


class TestExceptionHierarchy:
    """Tests verifying the exception class hierarchy."""

    def test_credential_error_is_subclass_of_redteam_error(self):
        """CredentialError must inherit from RedTeamError."""
        assert issubclass(CredentialError, RedTeamError)

    def test_backend_error_is_subclass_of_redteam_error(self):
        """BackendError must inherit from RedTeamError."""
        assert issubclass(BackendError, RedTeamError)

    def test_cancelled_error_is_subclass_of_redteam_error(self):
        """CancelledError must inherit from RedTeamError."""
        assert issubclass(CancelledError, RedTeamError)

    def test_target_response_error_is_subclass_of_redteam_error(self):
        """TargetResponseError must belong to the red-team exception hierarchy."""
        assert issubclass(TargetResponseError, RedTeamError)

    @pytest.mark.parametrize(
        "exc_class",
        [CredentialError, BackendError, CancelledError],
    )
    def test_all_subclasses_caught_by_redteam_error(self, exc_class):
        """All concrete exceptions can be caught with except RedTeamError."""
        with pytest.raises(RedTeamError):
            raise exc_class("test message")

    def test_target_response_error_is_caught_by_redteam_error(self):
        """TargetResponseError accepts and preserves a target error response."""
        response = AgentResponse(error=AgentResponseError(message='test message', error_type='target_error'))

        with pytest.raises(RedTeamError):
            raise TargetResponseError(response)

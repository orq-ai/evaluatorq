"""Unit tests for the red teaming exception hierarchy."""

import pytest

from evaluatorq.redteam.exceptions import (
    BackendError,
    CancelledError,
    CredentialError,
    RedTeamError,
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

    @pytest.mark.parametrize(
        "exc_class",
        [CredentialError, BackendError, CancelledError],
    )
    def test_all_subclasses_caught_by_redteam_error(self, exc_class):
        """All concrete exceptions can be caught with except RedTeamError."""
        with pytest.raises(RedTeamError):
            raise exc_class("test message")

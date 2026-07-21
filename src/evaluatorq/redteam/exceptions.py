"""Domain-specific exceptions for the evaluatorq.redteam package."""

from evaluatorq.contracts import AgentResponse


class RedTeamError(Exception):
    """Base exception for all red teaming errors."""


class CredentialError(RedTeamError):
    """Missing or invalid API credentials (e.g. ORQ_API_KEY not set)."""


class BackendError(RedTeamError):
    """Unsupported or unavailable backend."""


class DatasetError(RedTeamError):
    """Failed to download or load a red team dataset (network, auth, or parse error)."""


class CancelledError(RedTeamError):
    """Pipeline run was cancelled by the user via hooks."""


class TargetResponseError(RedTeamError):
    """Wrap a target-returned error marker so it follows target retry handling."""

    def __init__(self, response: AgentResponse) -> None:
        self.response = response
        response_error = response.error
        if response_error is None:
            raise ValueError('TargetResponseError requires an AgentResponse.error marker')
        self.error = response_error
        super().__init__(response_error.message)

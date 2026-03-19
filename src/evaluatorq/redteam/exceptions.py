"""Domain-specific exceptions for the evaluatorq.redteam package."""


class RedTeamError(Exception):
    """Base exception for all red teaming errors."""


class CredentialError(RedTeamError):
    """Missing or invalid API credentials (e.g. ORQ_API_KEY not set)."""


class BackendError(RedTeamError):
    """Unsupported or unavailable backend."""


class CancelledError(RedTeamError):
    """Pipeline run was cancelled by the user via hooks."""

"""Domain-specific exceptions for the evaluatorq.redteam package."""


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


class StrategyGenerationError(RedTeamError):
    """The attacker model produced no attack strategies for a vulnerability.

    Raised by objective generation when the call fails (a content-filter block that
    survived its retries, a transport error, an unparseable reply), and by the strategy
    planner when a vulnerability is left with zero strategies because generation failed.
    The message names the attacker model, since that is the setting to change.
    """

"""Resolve the deprecated ``parallelism=`` alias for ``datapoint_parallelism=``.

The old name was ambiguous once a second concurrency knob existed: it counted
datapoints, never LLM requests, and could not be sized against a provider limit.
Shared so every entry point deprecates it on the same schedule and with the same
message.
"""

from __future__ import annotations

import warnings


def resolve_datapoint_parallelism(
    datapoint_parallelism: int | None,
    parallelism: int | None,
    *,
    default: int,
    caller: str,
) -> int:
    """Return the effective datapoint concurrency, warning if the old name was used.

    The new name wins when both are supplied, so a caller mid-migration cannot be
    silently held to the old value.
    """
    if parallelism is None:
        return default if datapoint_parallelism is None else datapoint_parallelism

    warnings.warn(
        f'{caller}(parallelism=...) is deprecated; use datapoint_parallelism= for the '
        'datapoint count, or llm_parallelism= to bound concurrent LLM requests.',
        DeprecationWarning,
        stacklevel=3,
    )
    if datapoint_parallelism is not None:
        warnings.warn(
            f'{caller}() got both parallelism= and datapoint_parallelism=; using datapoint_parallelism.',
            DeprecationWarning,
            stacklevel=3,
        )
        return datapoint_parallelism
    return parallelism

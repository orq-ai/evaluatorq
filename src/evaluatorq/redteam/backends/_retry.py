"""Retry-configuration warnings for red-team backends.

Module-private. This lives here rather than in ``common/retry.py`` because it
reads ``LLMConfig`` field defaults, and ``common/`` is shared infrastructure
that must not import from ``redteam/``. The generic, surface-agnostic half of
this concern — ``without_client_retries`` — stays in ``common/retry.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from evaluatorq.redteam.contracts import LLMConfig

if TYPE_CHECKING:
    from collections.abc import Iterable


def warn_ignored_target_retries(
    target_name: str,
    *,
    retry_count: int | None = None,
    retry_on_codes: Iterable[int] | None = None,
) -> None:
    """Warn when pipeline retry settings do not own target-call retries.

    Target calls are retried by ``call_target_with_retry``. The settings still
    may configure non-target SDK operations (notably ORQ context and cleanup),
    but they are intentionally ignored or overridden at the target boundary.
    Defaults come from the Pydantic field metadata so checking whether a caller
    changed a setting does not instantiate a configuration object.

    Each field is compared independently: a caller that re-passes one setting at
    its default while leaving the other unset has changed nothing, and must not
    be warned at.
    """
    if retry_count is None and retry_on_codes is None:
        return
    count_changed = retry_count is not None and retry_count != LLMConfig.model_fields['retry_count'].default
    codes_changed = retry_on_codes is not None and retry_on_codes != LLMConfig.model_fields['retry_on_codes'].default
    if not count_changed and not codes_changed:
        return
    logger.warning(
        f'Ignoring retry_count and retry_on_codes for {target_name} target calls; '
        'call_target_with_retry owns target retries and the SDK retry budget is '
        'disabled at the target-call boundary'
    )

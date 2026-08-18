"""Retry-configuration warnings for red-team backends.

Module-private. Lives here, not in ``common/retry.py``, because it reads
``LLMConfig`` defaults and ``common/`` must not import from ``redteam/``.
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

    ``call_target_with_retry`` owns them; these settings still configure non-target
    SDK operations (ORQ context and cleanup), so they are ignored only at the target
    boundary. Each field is compared against its Pydantic default independently, so
    re-passing one at its default does not warn.
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

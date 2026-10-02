"""Load the Orq model catalogue for a dashboard model menu."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from loguru import logger

from evaluatorq.common.llm_client import resolve_llm_client
from evaluatorq.common.model_catalogue import ModelInfo, ModelKind, get_model_info, models_by_provider
from evaluatorq.common.orq_client import close_orq_client
from evaluatorq.dashboard.auth import DashboardAuth, build_auth_clients

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from openai import AsyncOpenAI


@asynccontextmanager
async def _catalogue_client(auth: DashboardAuth) -> AsyncIterator[AsyncOpenAI]:
    """The client the catalogue is read through, closing whatever this opened.

    Yields:
        The client to read the catalogue with.
    """
    if auth.method == 'cli_oauth':
        orq, llm = build_auth_clients(auth)
        try:
            yield llm
        finally:
            await close_orq_client(orq)
            await llm.close()
        return
    resolved = resolve_llm_client(
        extra_api_key=auth.api_key,
        orq_host=auth.base_url,
        require_orq=True,
        max_retries=0,
    )
    try:
        yield resolved.client
    finally:
        if resolved.owned:
            await resolved.client.close()


async def model_groups(auth: DashboardAuth, kind: ModelKind) -> dict[str, list[str]]:
    """Models of ``kind`` grouped by provider, or ``{}`` when the catalogue cannot be loaded."""
    try:
        async with _catalogue_client(auth) as client:
            return await models_by_provider(client, kind=kind)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        logger.warning('Orq model catalogue unavailable for {} models ({}): {}', kind, type(exc).__name__, exc)
        return {}


async def catalogue_entry(auth: DashboardAuth, model: str) -> ModelInfo | None:
    """The catalogue entry for ``model``, or ``None`` when it is unlisted or the catalogue cannot be loaded."""
    try:
        async with _catalogue_client(auth) as client:
            return await get_model_info(model, client)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        logger.warning('Orq model catalogue unavailable for {} ({}): {}', model, type(exc).__name__, exc)
        return None

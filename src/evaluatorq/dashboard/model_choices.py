"""Load the Orq model catalogue for a dashboard model menu."""

from __future__ import annotations

from evaluatorq.common.llm_client import resolve_llm_client
from evaluatorq.common.model_catalogue import ModelKind, models_by_provider
from evaluatorq.common.orq_client import close_orq_client
from evaluatorq.dashboard.auth import DashboardAuth, build_auth_clients


async def model_groups(auth: DashboardAuth, kind: ModelKind) -> dict[str, list[str]]:
    """Models of ``kind`` grouped by provider, or ``{}`` when the catalogue cannot be loaded."""
    try:
        if auth.method == 'cli_oauth':
            orq, llm = build_auth_clients(auth)
            try:
                return await models_by_provider(llm, kind=kind)
            finally:
                await close_orq_client(orq)
                await llm.close()
        resolved = resolve_llm_client(
            extra_api_key=auth.api_key,
            orq_host=auth.base_url,
            require_orq=True,
            max_retries=0,
        )
        try:
            return await models_by_provider(resolved.client, kind=kind)
        finally:
            if resolved.owned:
                await resolved.client.close()
    except (ImportError, OSError, RuntimeError, ValueError):
        return {}

"""Backend resolution for dynamic red teaming runtime."""

from __future__ import annotations

import os

from openai import AsyncOpenAI

from evaluatorq.redteam.backends.base import BackendBundle
from evaluatorq.redteam.backends.openai import (
    NoopMemoryCleanup,
    OpenAIContextProvider,
    OpenAIErrorMapper,
    OpenAITargetFactory,
)


def create_async_llm_client() -> AsyncOpenAI:
    """Create an OpenAI-compatible async client.

    Preference order:
    1. ORQ router-style env (`ORQ_API_KEY` + `ROUTER_BASE_URL`)
    2. Standard OpenAI env (`OPENAI_API_KEY` [+ optional `OPENAI_BASE_URL`])
    """
    orq_api_key = os.getenv('ORQ_API_KEY')
    router_base_url = os.getenv('ROUTER_BASE_URL')
    if orq_api_key and router_base_url:
        return AsyncOpenAI(api_key=orq_api_key, base_url=router_base_url)

    openai_api_key = os.getenv('OPENAI_API_KEY')
    if openai_api_key:
        openai_base_url = os.getenv('OPENAI_BASE_URL')
        if openai_base_url:
            return AsyncOpenAI(api_key=openai_api_key, base_url=openai_base_url)
        return AsyncOpenAI(api_key=openai_api_key)

    msg = (
        'Missing LLM credentials. Set either ORQ_API_KEY+ROUTER_BASE_URL '
        'or OPENAI_API_KEY (optionally OPENAI_BASE_URL).'
    )
    raise RuntimeError(msg)


def resolve_backend(backend: str = 'orq') -> BackendBundle:
    """Resolve runtime backend bundle with lazy optional imports."""
    normalized = backend.strip().lower()
    if normalized == 'openai':
        client = create_async_llm_client()
        return BackendBundle(
            name='openai',
            target_factory=OpenAITargetFactory(client),
            context_provider=OpenAIContextProvider(),
            memory_cleanup=NoopMemoryCleanup(),
            error_mapper=OpenAIErrorMapper(),
        )

    if normalized != 'orq':
        msg = f"Unsupported backend: {backend!r}. Expected 'orq' or 'openai'."
        raise ValueError(msg)

    try:
        from evaluatorq.redteam.backends.orq import ORQErrorMapper, create_orq_backend

        target_factory, context_provider, memory_cleanup = create_orq_backend()
    except (ImportError, RuntimeError) as exc:
        msg = "ORQ backend requested but ORQ dependencies are unavailable. Install ORQ extras or use backend='openai'."
        raise RuntimeError(msg) from exc

    return BackendBundle(
        name='orq',
        target_factory=target_factory,
        context_provider=context_provider,
        memory_cleanup=memory_cleanup,
        error_mapper=ORQErrorMapper(),
    )

"""ORQ backend implementation for dynamic red teaming.

Consolidates all ORQ SDK-specific agent code behind the backend ABCs
defined in ``backends.base``. Other modules import these concrete classes
when they need ORQ-specific behavior, or accept the base classes when they
want to be backend-agnostic.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from typing import Any, cast

from loguru import logger

from evaluatorq.redteam.backends.registry import ORQ_DEFAULT_BASE_URL
from evaluatorq.redteam.contracts import TargetKind
from evaluatorq.redteam.exceptions import CredentialError

try:
    from orq_ai_sdk import Orq as _Orq

    _orq_cls: Any = _Orq
except ImportError:
    _orq_cls = None


def _get_orq_api_key() -> str:
    """Read ORQ_API_KEY from environment."""
    key = os.environ.get('ORQ_API_KEY', '')
    if not key:
        msg = 'ORQ_API_KEY environment variable is not set'
        raise CredentialError(msg)
    return key


def _get_orq_server_url() -> str:
    """Read ORQ_BASE_URL from environment and strip /v3/router for SDK use."""
    url = os.environ.get('ORQ_BASE_URL', ORQ_DEFAULT_BASE_URL)
    return url.rstrip('/').removesuffix('/v3/router')


from evaluatorq.contracts import AgentTarget, Message
from evaluatorq.redteam.backends._errors import extract_provider_error_code, extract_status_code
from evaluatorq.redteam.backends.base import Backend
from evaluatorq.redteam.contracts import (
    PIPELINE_CONFIG,
    AgentContext,
    AgentResponse,
    KnowledgeBaseInfo,
    MemoryStoreInfo,
    OutputMessage,
    TextOutputItem,
    TokenUsage,
    ToolCallOutputItem,
    ToolInfo,
)
from evaluatorq.common.tracing import record_token_usage, set_span_attrs, truncate_for_span
from evaluatorq.redteam.tracing import with_redteam_span


async def _orq_cleanup_memory(orq_client: Any, ctx: AgentContext, entity_ids: list[str]) -> None:
    """Delete memory entities for each (memory store, entity_id) combination."""
    for ms in ctx.memory_stores:
        if not ms.key:
            logger.warning(f'Memory store {ms.id} has no key, skipping cleanup')
            continue
        for entity_id in entity_ids:
            try:
                await asyncio.to_thread(
                    orq_client.memory_stores.delete_memory,
                    memory_store_key=ms.key,
                    memory_entity_id=entity_id,
                )
                logger.debug(f'Deleted memory entity {entity_id} from store {ms.key}')
            except Exception as e:  # noqa: PERF203
                if extract_status_code(e) == 404:
                    continue
                logger.warning(f'Failed to cleanup memory entity {entity_id} from {ms.key}: {e}')


def _orq_map_error(exc: Exception) -> tuple[str, str]:
    """Map an ORQ exception to a normalized (error_code, error_message) tuple."""
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    status_code = extract_status_code(exc)
    provider_code = extract_provider_error_code(exc)
    if status_code is not None:
        return f'orq.http.{status_code}', f'{type(exc).__name__}: {exc}'
    if provider_code:
        return f'orq.code.{provider_code}', f'{type(exc).__name__}: {exc}'
    if 'timeout' in name or 'timed out' in text:
        return 'orq.timeout', f'{type(exc).__name__}: {exc}'
    if 'auth' in name or 'unauthorized' in text or 'forbidden' in text:
        return 'orq.auth', f'{type(exc).__name__}: {exc}'
    if 'ratelimit' in name or '429' in text:
        return 'orq.rate_limit', f'{type(exc).__name__}: {exc}'
    return 'orq.unknown', f'{type(exc).__name__}: {exc}'


class ORQAgentTarget(AgentTarget):
    """Target adapter for ORQ agents.

    Wraps the ORQ SDK to satisfy the ``AgentTarget`` ABC.
    """

    def __init__(
        self,
        agent_key: str,
        orq_client: Any,
        memory_entity_id: str | None = None,
        model: str | None = None,
        timeout_ms: int | None = None,
    ):
        """Initialize the ORQ agent target with client and configuration.

        ORQ agents can have server-side memory stores attached. Every target
        instance owns a ``memory_entity_id`` so parallel jobs stay isolated;
        if not provided, one is generated. The pipeline reads this attribute
        after construction to track entities for cleanup.
        """
        if memory_entity_id is None:
            memory_entity_id = f'red-team-{uuid.uuid4().hex[:12]}'
        super().__init__(memory_entity_id=memory_entity_id)
        timeout_ms = timeout_ms or PIPELINE_CONFIG.target_agent_timeout_ms
        self.agent_key = agent_key
        self.orq_client = orq_client
        self.model = model
        self._timeout_ms = timeout_ms
        self._task_id: str | None = None
        self._cached_context: AgentContext | None = None

    async def respond(self, messages: list[Message]) -> AgentResponse:
        """Send the last user message to the ORQ agents endpoint, threaded via ``task_id``.

        The ORQ agents endpoint holds conversation state server-side via
        ``task_id``, so ``respond`` forwards only the last user turn. Prior turns
        in ``messages`` are assumed consistent with the server-side history that
        ``task_id`` references; the first call (``task_id is None``) seeds it.

        Caller contract: ``messages[-1].role == "user"``. A mismatch with the
        server-side history is a caller bug.

        Token usage is accumulated across pending-tool-call continuations.
        """
        if not messages or messages[-1].role != "user":
            raise ValueError(
                "ORQAgentTarget.respond requires messages[-1].role == 'user'. "
                "Server-side conversation state is held via task_id."
            )
        prompt = messages[-1].content or ""
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0
        total_calls = 0

        def _accumulate_usage(resp: object) -> None:
            """Accumulate token usage from a response into running totals."""
            nonlocal total_prompt_tokens, total_completion_tokens, total_tokens, total_calls
            resp_usage = getattr(resp, 'usage', None)
            if resp_usage is None:
                return
            prompt_tokens = int(getattr(resp_usage, 'prompt_tokens', 0) or 0)
            completion_tokens = int(getattr(resp_usage, 'completion_tokens', 0) or 0)
            raw_total = getattr(resp_usage, 'total_tokens', None)
            if raw_total is not None and int(raw_total) > 0:
                total = int(raw_total)
            else:
                total = prompt_tokens + completion_tokens
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens
            total_tokens += total
            total_calls += 1

        def _extract_text(resp: object) -> str:
            """Extract the first non-empty text part from an agent response."""
            output = getattr(resp, 'output', None) or []
            for item in output:
                parts = getattr(item, 'parts', None) or []
                for part in parts:
                    if getattr(part, 'kind', None) == 'text':
                        text = getattr(part, 'text', None)
                        if isinstance(text, str) and text.strip():
                            return text
            return ''

        def _pending_tool_call_ids(resp: object) -> list[str]:
            """Return IDs of pending tool calls from a response."""
            pending = getattr(resp, 'pending_tool_calls', None) or []
            ids: list[str] = []
            for call in pending:
                call_id = getattr(call, 'id', None)
                if not call_id and isinstance(call, dict):
                    call_id = call.get('id')
                if isinstance(call_id, str) and call_id.strip():
                    ids.append(call_id)
            return ids

        def _extract_tool_call_items(resp: object) -> list[ToolCallOutputItem]:
            """Extract tool calls from a response as ToolCallOutputItem instances."""
            pending = getattr(resp, 'pending_tool_calls', None) or []
            items: list[ToolCallOutputItem] = []
            for call in pending:
                name = getattr(call, 'name', None) or (call.get('name') if isinstance(call, dict) else None) or ''
                args_raw = getattr(call, 'arguments', None) or (call.get('arguments') if isinstance(call, dict) else None) or {}
                if isinstance(args_raw, str):
                    try:
                        json.loads(args_raw)
                        args_str = args_raw
                    except (json.JSONDecodeError, ValueError):
                        logger.warning(
                            f"Failed to parse tool call arguments as JSON for tool '{name}'; "
                            f"wrapping raw string under 'raw' key. Raw: {args_raw!r:.200}"
                        )
                        args_str = json.dumps({'raw': args_raw})
                else:
                    try:
                        args_str = json.dumps(args_raw if isinstance(args_raw, dict) else {})
                    except (TypeError, ValueError):
                        args_str = json.dumps(args_raw, default=str)
                call_id = getattr(call, 'id', None) or (call.get('id') if isinstance(call, dict) else None)
                kwargs: dict[str, Any] = {'name': str(name), 'arguments': args_str}
                if isinstance(call_id, str) and call_id:
                    kwargs['id'] = call_id
                items.append(ToolCallOutputItem(**kwargs))
            return items

        async with with_redteam_span(
            f'agent {self.agent_key}',
            {
                'orq.redteam.llm_purpose': 'target',
                'gen_ai.system': 'orq',
                'gen_ai.request.model': self.model or self.agent_key,
                'gen_ai.input.messages': json.dumps(
                    [{'role': 'user', 'content': truncate_for_span(prompt)}],
                    ensure_ascii=False,
                ),
            },
        ) as span:
            try:
                kwargs: dict[str, Any] = {
                    'agent_key': self.agent_key,
                    'message': {'role': 'user', 'parts': [{'kind': 'text', 'text': prompt}]},
                    'background': False,
                }
                if self._task_id is not None:
                    kwargs['task_id'] = self._task_id
                if self.memory_entity_id:
                    kwargs['memory'] = {'entity_id': self.memory_entity_id}

                response = await asyncio.to_thread(self.orq_client.agents.responses.create, **kwargs)

                if response.task_id:
                    self._task_id = response.task_id

                _accumulate_usage(response)
                text_response = _extract_text(response)
                all_tool_calls: list[ToolCallOutputItem] = _extract_tool_call_items(response)

                # Some agent tool flows require client-provided tool_result parts.
                # Continue the same task with synthetic tool results so the thread can progress.
                max_tool_continuations = PIPELINE_CONFIG.max_tool_continuations
                pending_ids = _pending_tool_call_ids(response)
                continuation_count = 0
                while pending_ids and continuation_count < max_tool_continuations:
                    continuation_count += 1
                    logger.debug(
                        f'{self.agent_key}: resolving {len(pending_ids)} pending tool call(s) '
                        f'via synthetic tool_result (step {continuation_count}/{max_tool_continuations})'
                    )

                    tool_parts = [
                        {
                            'kind': 'tool_result',
                            'tool_call_id': tool_call_id,
                            'result': {
                                'ok': False,
                                'error': 'Tool execution unavailable in red-teaming harness',
                            },
                        }
                        for tool_call_id in pending_ids
                    ]
                    response = await asyncio.to_thread(
                        self.orq_client.agents.responses.create,
                        agent_key=self.agent_key,
                        message={'role': 'tool', 'parts': tool_parts},
                        task_id=self._task_id,
                        background=False,
                    )
                    if response.task_id:
                        self._task_id = response.task_id
                    _accumulate_usage(response)
                    extracted = _extract_text(response)
                    if extracted:
                        text_response = extracted
                    all_tool_calls.extend(_extract_tool_call_items(response))
                    pending_ids = _pending_tool_call_ids(response)

                if pending_ids:
                    raise RuntimeError(
                        f'Unresolved pending tool calls after {max_tool_continuations} continuations '
                        f'({len(pending_ids)} remaining)'
                    )

                response_model = getattr(response, 'model', None)
                if response_model:
                    set_span_attrs(span, {'gen_ai.response.model': str(response_model)})

                result_text = text_response or ''
                set_span_attrs(
                    span,
                    {
                        'gen_ai.output.messages': json.dumps(
                            [{'role': 'assistant', 'content': truncate_for_span(result_text)}],
                            ensure_ascii=False,
                        ),
                    },
                )

                usage = (
                    TokenUsage(
                        prompt_tokens=total_prompt_tokens,
                        completion_tokens=total_completion_tokens,
                        total_tokens=total_tokens,
                        calls=total_calls,
                    )
                    if total_calls > 0
                    else None
                )
                output: list[OutputMessage] = cast('list[OutputMessage]', list(all_tool_calls))
                output.append(TextOutputItem(text=result_text, annotations=[]))
                return AgentResponse(
                    output=output,
                    usage=usage,
                    model=str(response_model) if response_model else None,
                    response_id=getattr(response, 'task_id', None),
                    finish_reason=None,
                )

            except Exception as e:
                logger.error(f'ORQ agent call failed: {e}')
                raise
            finally:
                if total_calls > 0:
                    record_token_usage(
                        span,
                        prompt_tokens=total_prompt_tokens,
                        completion_tokens=total_completion_tokens,
                        total_tokens=total_tokens,
                        calls=total_calls,
                    )

    def new(self) -> ORQAgentTarget:
        """Return a fresh target instance with isolated state.

        Each call gets its own ``memory_entity_id`` (auto-generated in
        ``__init__``) and own ``_task_id`` so parallel jobs never share
        server-side memory or conversation state.
        """
        return ORQAgentTarget(
            agent_key=self.agent_key,
            orq_client=self.orq_client,
            model=self.model,
            timeout_ms=self._timeout_ms,
        )

    target_kind: TargetKind = TargetKind.AGENT
    """Used by the runner to populate report metadata correctly."""

    @property
    def name(self) -> str:
        """Return the agent key as the display name for reports and tracing."""
        return self.agent_key

    async def get_agent_context(self) -> AgentContext:
        """Return agent context for this target's agent key."""
        if self._cached_context is not None:
            return self._cached_context
        logger.debug(f'Retrieving agent context for: {self.agent_key}')
        agent_data = await asyncio.to_thread(
            self.orq_client.agents.retrieve,
            agent_key=self.agent_key,
        )

        tools: list[ToolInfo] = []
        settings = getattr(agent_data, 'settings', None)
        if settings and hasattr(settings, 'tools') and settings.tools:
            tools.extend(
                ToolInfo(
                    name=getattr(tool, 'key', None) or getattr(tool, 'display_name', None) or tool.id,
                    description=getattr(tool, 'description', None),
                    parameters=None,
                )
                for tool in settings.tools
            )

        raw_kb_ids: list[str] = []
        if hasattr(agent_data, 'knowledge_bases') and agent_data.knowledge_bases:
            raw_kb_ids = [getattr(kb, 'knowledge_id', None) or str(kb) for kb in agent_data.knowledge_bases]

        raw_ms_ids: list[str] = []
        if hasattr(agent_data, 'memory_stores') and agent_data.memory_stores:
            raw_ms_ids = [ms if isinstance(ms, str) else getattr(ms, 'key', str(ms)) for ms in agent_data.memory_stores]

        enrichment_tasks: list[Any] = [self._enrich_knowledge_base(kb_id) for kb_id in raw_kb_ids]
        enrichment_tasks.extend(self._enrich_memory_store(ms_id) for ms_id in raw_ms_ids)

        enriched_results = await asyncio.gather(*enrichment_tasks) if enrichment_tasks else []
        knowledge_bases = [r for r in enriched_results if isinstance(r, KnowledgeBaseInfo)]
        memory_stores = [r for r in enriched_results if isinstance(r, MemoryStoreInfo)]

        model_raw = getattr(agent_data, 'model', None)
        model_id = getattr(model_raw, 'id', None) if model_raw is not None else None

        self._cached_context = AgentContext(
            key=self.agent_key,
            display_name=getattr(agent_data, 'display_name', None),
            description=getattr(agent_data, 'description', None),
            system_prompt=getattr(agent_data, 'system_prompt', None),
            instructions=getattr(agent_data, 'instructions', None),
            tools=tools,
            memory_stores=memory_stores,
            knowledge_bases=knowledge_bases,
            model=model_id,
        )
        return self._cached_context

    async def _enrich_knowledge_base(self, kb_id: str) -> KnowledgeBaseInfo:
        """Retrieve full knowledge base details from ORQ API."""
        try:
            kb = await asyncio.to_thread(self.orq_client.knowledge.retrieve, knowledge_id=kb_id)
            return KnowledgeBaseInfo(
                id=kb_id,
                key=getattr(kb, 'key', None),
                name=getattr(kb, 'key', None),
                description=getattr(kb, 'description', None) or None,
            )
        except Exception as e:
            logger.warning(f'Failed to enrich knowledge base {kb_id}: {e} — attack strategies will use limited context')
            return KnowledgeBaseInfo(id=kb_id)

    async def _enrich_memory_store(self, ms_key: str) -> MemoryStoreInfo:
        """Retrieve full memory store details via ORQ SDK."""
        try:
            ms = await asyncio.to_thread(
                self.orq_client.memory_stores.retrieve,
                memory_store_key=ms_key,
            )
            return MemoryStoreInfo(
                id=getattr(ms, 'id', ms_key),
                key=getattr(ms, 'key', ms_key),
                description=getattr(ms, 'description', None) or None,
            )
        except Exception as e:
            logger.warning(f'Failed to enrich memory store {ms_key}: {e} — attack strategies will use limited context')
            return MemoryStoreInfo(id=ms_key)

    def map_error(self, exc: Exception) -> tuple[str, str]:
        """Map an exception to a normalized error code and message tuple."""
        return _orq_map_error(exc)

    async def cleanup_memory(self, ctx: AgentContext, entity_ids: list[str]) -> None:
        """Clean up memory entities created during red teaming."""
        await _orq_cleanup_memory(self.orq_client, ctx, entity_ids)


class ORQBackend(Backend):
    """Backend for ORQ-hosted agents.

    Owns the ORQ SDK client. Creates ``ORQAgentTarget`` instances per job.
    Performs memory cleanup via the SDK. Maps ORQ exceptions to a normalized
    error taxonomy.
    """

    def __init__(self, *, orq_client: Any = None, timeout_ms: int | None = None) -> None:
        super().__init__(name="orq")
        timeout_ms = timeout_ms or PIPELINE_CONFIG.target_agent_timeout_ms
        self._timeout_ms = timeout_ms
        if orq_client is not None:
            self._orq_client = orq_client
        else:
            if _orq_cls is None:
                raise ImportError(
                    "ORQ backend requires the orq-ai-sdk package. "
                    "Install with: pip install evaluatorq[orq]"
                )
            self._orq_client = _orq_cls(
                api_key=_get_orq_api_key(),
                server_url=_get_orq_server_url(),
                timeout_ms=self._timeout_ms,
            )

    def create_target(self, agent_key: str) -> ORQAgentTarget:
        return ORQAgentTarget(
            agent_key=agent_key,
            orq_client=self._orq_client,
            timeout_ms=self._timeout_ms,
        )

    async def cleanup_memory(self, ctx: AgentContext, entity_ids: list[str]) -> None:
        await _orq_cleanup_memory(self._orq_client, ctx, entity_ids)

    def map_error(self, exc: Exception) -> tuple[str, str]:
        return _orq_map_error(exc)


def create_orq_agent_target(
    agent_key: str,
    orq_client: Any = None,
    timeout_ms: int | None = None,
) -> ORQAgentTarget:
    """Create an ORQAgentTarget from environment config."""
    timeout_ms = timeout_ms or PIPELINE_CONFIG.target_agent_timeout_ms
    if orq_client is None:
        if _orq_cls is None:
            raise ImportError('ORQ backend requires the orq-ai-sdk package.')
        orq_client = _orq_cls(
            api_key=_get_orq_api_key(),
            server_url=_get_orq_server_url(),
            timeout_ms=timeout_ms,
        )
    return ORQAgentTarget(agent_key=agent_key, orq_client=orq_client, timeout_ms=timeout_ms)

"""Custom AgentTarget that handles function-tool callbacks in-process.

evaluatorq's built-in orq backend stubs all pending tool calls with an
error result, so it can't be used for agents with real function tools.
This class implements the AgentTarget protocol with a tool-call loop
that dispatches to local Python handlers.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from evaluatorq.redteam.contracts import (
    AgentContext,
    AgentResponse,
    KnowledgeBaseInfo,
    TextOutputItem,
    TokenUsage,
    ToolInfo,
)

from agent_build.demo_data import DemoState
from agent_build.handlers import (
    handle_get_policy,
    handle_issue_refund,
    handle_lookup_order,
)

MAX_TOOL_CONTINUATIONS = 12


class RefundAgentTarget:
    """AgentTarget for the refund agent — handles its 3 function tools locally."""

    memory_entity_id: str | None = None

    def __init__(self, agent_key: str, orq_client: Any | None = None) -> None:
        if orq_client is None:
            from orq_ai_sdk import Orq  # lazy: tests pass a mock

            orq_client = Orq(api_key=os.environ['ORQ_API_KEY'])
        self.agent_key = agent_key
        self.name = agent_key
        self.orq_client = orq_client
        self._task_id: str | None = None
        self._state = DemoState()
        # IDs of tool_calls from a prior turn that we never answered (e.g. because
        # `_invoke` failed or we hit MAX_TOOL_CONTINUATIONS). orq's task history
        # requires every assistant tool_call to be followed by a tool message, so
        # we drain these with synthetic error results before the next user turn.
        self._unanswered_tool_call_ids: list[str] = []

    def new(self) -> RefundAgentTarget:
        return RefundAgentTarget(agent_key=self.agent_key, orq_client=self.orq_client)

    async def get_agent_context(self) -> AgentContext:
        from agent_build.build_agent import (
            AGENT_MODEL_ID,
            AGENTS,
            KB_KEY,
            TOOL_SCHEMAS,
        )

        prompt_by_key = {key: prompt for key, _display, prompt in AGENTS}
        display_by_key = {key: display for key, display, _prompt in AGENTS}
        tools = [
            ToolInfo(
                name=name,
                description=schema['description'],
                parameters={
                    'type': 'object',
                    'properties': schema['parameters']['properties'],
                    'required': schema['parameters']['required'],
                },
            )
            for name, schema in TOOL_SCHEMAS.items()
        ]
        return AgentContext(
            key=self.agent_key,
            display_name=display_by_key.get(self.agent_key, self.agent_key),
            description='Customer service refund agent (red-teaming demo).',
            instructions=prompt_by_key[self.agent_key],
            tools=tools,
            knowledge_bases=[KnowledgeBaseInfo(id=KB_KEY, key=KB_KEY, name='Refund policy KB')],
            model=AGENT_MODEL_ID,
        )

    async def send_prompt(self, prompt: str) -> AgentResponse:
        await self._drain_unanswered_tool_calls()
        response = await self._invoke(message={'role': 'user', 'parts': [{'kind': 'text', 'text': prompt}]})
        text = self._extract_text(response)
        usage = self._extract_usage(response)
        pending = self._pending_calls(response)
        for _step in range(1, MAX_TOOL_CONTINUATIONS + 1):
            if not pending:
                break
            parts = [self._dispatch(c) for c in pending]
            pending_ids_before_invoke = [p.get('tool_call_id') for p in parts if p.get('tool_call_id')]
            self._unanswered_tool_call_ids = pending_ids_before_invoke
            response = await self._invoke(message={'role': 'tool', 'parts': parts})
            self._unanswered_tool_call_ids = []
            step_usage = self._extract_usage(response)
            usage = usage + step_usage if usage is not None else step_usage
            new_text = self._extract_text(response)
            if new_text:
                text = new_text
            pending = self._pending_calls(response)
        if pending:
            self._unanswered_tool_call_ids = [
                getattr(c, 'id', None) or (c.get('id') if isinstance(c, dict) else None) for c in pending
            ]
            self._unanswered_tool_call_ids = [i for i in self._unanswered_tool_call_ids if i]
            raise RuntimeError(
                f'too many tool-call continuations (>{MAX_TOOL_CONTINUATIONS}); agent stuck in tool-call loop'
            )
        return AgentResponse(
            output=[TextOutputItem(text=text or '')],
            usage=usage,
            response_id=self._task_id,
            finish_reason=getattr(response, 'finish_reason', None),
        )

    async def _drain_unanswered_tool_calls(self) -> None:
        """Synthesise error tool_results for any pending tool_calls left from a
        prior turn so orq's task history is consistent before the next user
        message."""
        if not self._unanswered_tool_call_ids:
            return
        parts = [
            {
                'kind': 'tool_result',
                'tool_call_id': call_id,
                'result': {'ok': False, 'error': 'aborted: previous turn ended before tool result was delivered'},
            }
            for call_id in self._unanswered_tool_call_ids
        ]
        self._unanswered_tool_call_ids = []
        try:
            await self._invoke(message={'role': 'tool', 'parts': parts})
        except Exception:
            # If draining itself fails, the task is unrecoverable — drop task_id
            # so the next user message starts a fresh conversation.
            self._task_id = None

    async def _invoke(self, *, message: dict) -> Any:
        kwargs: dict[str, Any] = {
            'agent_key': self.agent_key,
            'message': message,
            'background': False,
        }
        if self._task_id is not None:
            kwargs['task_id'] = self._task_id
        response = await asyncio.to_thread(self.orq_client.agents.responses.create, **kwargs)
        new_task_id = getattr(response, 'task_id', None)
        if new_task_id:
            self._task_id = new_task_id
        return response

    def _dispatch(self, call: Any) -> dict:
        # call may be either:
        #   - a ToolCallPart from output[].parts[] (kind='tool_call', tool_call_id, tool_name, arguments)
        #   - a PendingToolCalls (id, function={name, arguments})
        # Prefer ToolCallPart fields because orq's task history tracks
        # tool_call_id from the part, not the PendingToolCalls.id.
        call_id = (
            getattr(call, 'tool_call_id', None)
            or getattr(call, 'id', None)
            or (call.get('tool_call_id') if isinstance(call, dict) else None)
            or (call.get('id') if isinstance(call, dict) else None)
        )
        name = getattr(call, 'tool_name', None)
        if name is None:
            func = getattr(call, 'function', None)
            if func is not None:
                name = getattr(func, 'name', None)
        if name is None and isinstance(call, dict):
            name = call.get('tool_name') or call.get('name')

        raw_args = getattr(call, 'arguments', None)
        if raw_args is None:
            func = getattr(call, 'function', None)
            if func is not None:
                raw_args = getattr(func, 'arguments', None)
        if raw_args is None and isinstance(call, dict):
            raw_args = call.get('arguments')
        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                args = {}
        else:
            args = dict(raw_args or {})

        # Every pending tool_call MUST be answered, even on handler error —
        # otherwise orq's task history is left with an unanswered tool_call and
        # the next user turn 400s with "tool_call_ids did not have response messages".
        try:
            if name == 'lookup_order':
                result: dict = handle_lookup_order(self._state, **args)
            elif name == 'issue_refund':
                result = handle_issue_refund(self._state, **args)
            elif name == 'get_policy':
                result = handle_get_policy(orq_client=self.orq_client, **args)
            else:
                result = {'ok': False, 'error': f'unknown_tool:{name}'}
        except Exception as exc:
            result = {'ok': False, 'error': f'{type(exc).__name__}: {exc}'}

        return {'kind': 'tool_result', 'tool_call_id': call_id, 'result': result}

    def _pending_calls(self, resp: Any) -> list[Any]:
        # Prefer ToolCallPart entries from output[].parts[] — those carry
        # tool_call_id (the id orq's task history matches against). Fall
        # back to pending_tool_calls if no parts present.
        parts_calls: list[Any] = []
        output = getattr(resp, 'output', None) or []
        for item in output:
            parts_calls.extend(
                part for part in (getattr(item, 'parts', None) or []) if getattr(part, 'kind', None) == 'tool_call'
            )
        if parts_calls:
            return parts_calls
        return list(getattr(resp, 'pending_tool_calls', None) or [])

    def _extract_text(self, resp: Any) -> str:
        # AgentResponseMessage.parts shape with kind='text', text=str
        output = getattr(resp, 'output', None) or []
        for item in output:
            parts = getattr(item, 'parts', None) or []
            for part in parts:
                if getattr(part, 'kind', None) == 'text':
                    return getattr(part, 'text', '') or ''
        return ''

    def _extract_usage(self, resp: Any) -> TokenUsage | None:
        usage = getattr(resp, 'usage', None)
        if usage is None:
            return None
        return TokenUsage(
            prompt_tokens=int(getattr(usage, 'prompt_tokens', 0) or 0),
            completion_tokens=int(getattr(usage, 'completion_tokens', 0) or 0),
            total_tokens=int(getattr(usage, 'total_tokens', 0) or 0),
            calls=1,
        )

"""Live check: does the judge actually hit the Anthropic prompt cache?

Runs a real `JudgeAgent` against the Orq router on both API paths, three
judgements each with a growing transcript, and prints the per-call cache
counters. Needs `ORQ_API_KEY`.

    uv run python scripts/manual_tests/prompt_cache_judge_check.py

What to expect: both paths place a positioned breakpoint, and `volatile_tail=1`
keeps it off the instruction the judge rebuilds each turn. Call 1 writes; calls
2-3 read that write back and write only the newly appended turns. ``responses``
reads slightly more than ``chat_completions`` because its system prompt sits in
`instructions`, outside the marked input, so the whole input prefix caches.

The prefix is salted with a uuid so every run starts cold; without that, a
second run inside the 5m TTL reads the first run's entry and every number lies.
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid
from typing import Any

from evaluatorq.simulation.agents.judge import JudgeAgent, JudgeAgentConfig
from evaluatorq.simulation.types import Criterion, Message

MODEL = os.environ.get('CACHE_CHECK_MODEL', 'anthropic/claude-sonnet-4-6')

# Anthropic caches nothing below ~1024 tokens, so the prefix has to be genuinely
# large or every counter reads 0 and the run proves nothing.
_FILLER = (
    'The support agent must confirm the account identifier before disclosing any '
    'billing detail, must never quote a refund figure it has not verified against '
    'the ledger, and must escalate to a human whenever the customer disputes a '
    'charge more than once in the same conversation. '
)


def _big_goal(salt: str) -> str:
    return f'Run {salt}. Evaluate a customer-support agent. ' + _FILLER * 40


def _criteria() -> list[Criterion]:
    return [
        Criterion(description='The agent verifies the account identifier before sharing billing detail.',
                  type='must_happen'),
        Criterion(description='The agent escalates a repeated dispute to a human.', type='must_happen'),
        Criterion(description='The agent never invents a refund amount.', type='must_not_happen'),
    ]


def _transcript(turns: int, salt: str) -> list[Message]:
    """A growing, append-only transcript: turn N is a strict prefix of turn N+1."""
    messages: list[Message] = []
    for i in range(turns):
        messages.append(Message(role='user', content=f'[{salt}] Question {i}: I was charged twice. ' + _FILLER * 8))
        messages.append(Message(role='assistant', content=f'Answer {i}: let me verify your account. ' + _FILLER * 8))
    return messages


def _record(client: Any, sink: list[Any]) -> None:
    """Capture raw SDK usage without reaching into the agent's internals."""
    chat_create = client.chat.completions.create
    responses_create = client.responses.create

    async def chat(**kwargs: Any) -> Any:
        response = await chat_create(**kwargs)
        sink.append(response.usage)
        return response

    async def responses(**kwargs: Any) -> Any:
        response = await responses_create(**kwargs)
        sink.append(response.usage)
        return response

    client.chat.completions.create = chat
    client.responses.create = responses


def _counters(usage: Any) -> tuple[int, int, int]:
    """(input, cache_read, cache_write) across the chat and responses shapes."""
    if usage is None:
        return (0, 0, 0)
    total = getattr(usage, 'prompt_tokens', None) or getattr(usage, 'input_tokens', 0) or 0
    details = getattr(usage, 'prompt_tokens_details', None) or getattr(usage, 'input_tokens_details', None)
    read = getattr(usage, 'cache_read_input_tokens', 0) or (getattr(details, 'cached_tokens', 0) if details else 0) or 0
    write = getattr(usage, 'cache_creation_input_tokens', 0) or 0
    if not write:
        creation = getattr(usage, 'cache_creation', None)
        write = (getattr(creation, 'ephemeral_5m_input_tokens', 0) or 0) if creation else 0
    return (int(total), int(read), int(write))


async def _run(api: str, salt: str) -> list[tuple[int, int, int]]:
    judge = JudgeAgent(
        JudgeAgentConfig(
            goal=_big_goal(salt),
            criteria=_criteria(),
            model=MODEL,
            api=api,
        )
    )
    usages: list[Any] = []
    _record(judge._client, usages)  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001

    for turns in (3, 4, 5):
        await judge.evaluate(_transcript(turns, salt))
    await judge.close()
    return [_counters(u) for u in usages]


async def main() -> int:
    if not os.environ.get('ORQ_API_KEY'):
        print('ORQ_API_KEY is not set')
        return 1

    salt = uuid.uuid4().hex[:12]
    print(f'model={MODEL} salt={salt}\n')

    failures: list[str] = []
    for api in ('chat_completions', 'responses'):
        print(f'--- api={api} ---')
        try:
            rows = await _run(api, f'{salt}-{api}')
        except Exception as exc:  # noqa: BLE001 - a manual probe reports, it does not raise
            print(f'  FAILED: {type(exc).__name__}: {exc}\n')
            failures.append(f'{api}: {type(exc).__name__}')
            continue

        for i, (total, read, write) in enumerate(rows, start=1):
            print(f'  call {i}: input={total:>6}  cache_read={read:>6}  cache_write={write:>6}')

        reads_after_first = sum(read for _, read, _ in rows[1:])
        verdict = 'PASS' if reads_after_first > 0 else 'FAIL — no cache read on any later call'
        print(f'  => {verdict}\n')
        if verdict.startswith('FAIL'):
            failures.append(f'{api}: no cache read')

    if failures:
        print('FAILURES: ' + '; '.join(failures))
        return 1
    print('done')
    return 0


if __name__ == '__main__':
    sys.exit(asyncio.run(main()))

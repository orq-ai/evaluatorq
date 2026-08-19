"""Probe: is a Responses breakpoint positionable, like on Chat Completions?

It is, and that is what `common.prompt_cache.mark_responses_input` ships. This
script is the evidence, and the regression probe that would catch the router
dropping per-item support: the top-level `cache_control` body field marks the end
of the *whole* input, so it cannot be kept off a rebuilt trailing item and reads
nothing back. That is why it is deliberately not sent.

Each mode below runs two calls that share a long prefix and differ only in a
rebuilt trailing item. A mode "works" if call 2 reports a cache read.

    uv run python scripts/manual_tests/prompt_cache_responses_probe.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid
from typing import Any

from evaluatorq.common.llm_client import resolve_llm_client

MODEL = os.environ.get('CACHE_CHECK_MODEL', 'anthropic/claude-sonnet-4-6')

_FILLER = (
    'The support agent must confirm the account identifier before disclosing any '
    'billing detail, must never quote a refund figure it has not verified against '
    'the ledger, and must escalate whenever a charge is disputed twice. '
)


def _text_item(role: str, text: str, *, marked: bool = False) -> dict[str, Any]:
    part_type = 'output_text' if role == 'assistant' else 'input_text'
    part: dict[str, Any] = {'type': part_type, 'text': text}
    if marked:
        part['cache_control'] = {'type': 'ephemeral'}
    return {'role': role, 'content': [part]}


def _input(salt: str, turns: int, *, mark_prefix_end: bool, tail: str) -> list[dict[str, Any]]:
    """Long shared prefix, then a trailing item that differs between calls."""
    items = [_text_item('user', f'[{salt}] Context: ' + _FILLER * 60)]
    for i in range(turns):
        items.append(_text_item('user', f'Question {i}: I was charged twice. ' + _FILLER * 6))
        items.append(_text_item('assistant', f'Answer {i}: verifying your account. ' + _FILLER * 6))
    if mark_prefix_end:
        # Mark the END OF THE PERSISTED PREFIX — the thing volatile_tail does on
        # chat completions. The rebuilt tail is appended after it, unmarked.
        items[-1] = _text_item('assistant', items[-1]['content'][0]['text'], marked=True)
    items.append(_text_item('user', tail))
    return items


def _read_tokens(usage: Any) -> int:
    details = getattr(usage, 'input_tokens_details', None)
    return int(
        getattr(usage, 'cache_read_input_tokens', 0)
        or (getattr(details, 'cached_tokens', 0) if details else 0)
        or 0
    )


async def _probe(client: Any, label: str, *, per_item: bool, top_level: bool) -> None:
    salt = uuid.uuid4().hex[:12]
    reads: list[int] = []
    for call in (1, 2):
        kwargs: dict[str, Any] = {
            'model': MODEL,
            'input': _input(salt, 4, mark_prefix_end=per_item, tail=f'Rebuilt instruction, revision {call}.'),
            'max_output_tokens': 16,
        }
        if top_level:
            kwargs['extra_body'] = {'cache_control': {'type': 'ephemeral'}}
        try:
            response = await client.responses.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 - probing what the router rejects is the point
            print(f'  {label:<34} call {call}: REJECTED {type(exc).__name__}: {str(exc)[:160]}')
            return
        reads.append(_read_tokens(response.usage))
    print(f'  {label:<34} reads={reads}  {"HIT" if reads[-1] > 0 else "no read"}')


async def main() -> int:
    if not os.environ.get('ORQ_API_KEY'):
        print('ORQ_API_KEY is not set')
        return 1
    client = resolve_llm_client().client
    print(f'model={MODEL}\n')
    print('Two calls per mode; they share a long prefix and differ in the trailing item.')
    print('A cache read on call 2 means the breakpoint landed on the shared prefix.\n')

    await _probe(client, 'top-level only (rejected: 0 reads)', per_item=False, top_level=True)
    await _probe(client, 'per-item on prefix end (what we ship)', per_item=True, top_level=False)
    await _probe(client, 'per-item + top-level', per_item=True, top_level=True)
    await _probe(client, 'neither (control)', per_item=False, top_level=False)
    return 0


if __name__ == '__main__':
    sys.exit(asyncio.run(main()))

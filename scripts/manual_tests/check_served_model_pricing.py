"""Show which model a call was priced against (RES-1573).

Run it with an ORQ key and read the two ids in each line. When they differ, the
cost came from the model that answered rather than the one that was asked for,
which is the whole point of the change: a router
(`orq/autorouter-anthropic-balanced`) resolves per request, and its catalogue
entry is one headline rate that does not match the model it picked.

    ORQ_API_KEY=... uv run python scripts/manual_tests/check_served_model_pricing.py

Pass model ids as arguments to try others.
"""

import asyncio
import sys

from evaluatorq.common.llm_call import execute_chat_completion
from evaluatorq.common.llm_client import resolve_llm_client

DEFAULT_MODELS = ('openai/gpt-5.6-luna', 'anthropic/claude-sonnet-4-6', 'orq/autorouter-anthropic-balanced')


async def main() -> None:
    models = sys.argv[1:] or list(DEFAULT_MODELS)
    resolved = resolve_llm_client()
    for model in models:
        try:
            response, usage = await execute_chat_completion(
                client=resolved.client,
                model=model,
                messages=[{'role': 'user', 'content': 'Reply with the single word: ok'}],
                span=None,
                timeout_s=60,
                max_tokens=16,
                inject_trace_headers=False,
            )
        except Exception as exc:  # noqa: BLE001 - a 404 for an unshipped router is a result, not a crash
            print(f'requested={model!r:32} error={type(exc).__name__}: {str(exc)[:120]}')
            continue
        cost = usage.total_cost if usage else None
        priced = usage.priced_calls if usage else 0
        print(f'requested={model!r:32} served={response.model!r:28} cost={cost} priced_calls={priced}')


if __name__ == '__main__':
    asyncio.run(main())

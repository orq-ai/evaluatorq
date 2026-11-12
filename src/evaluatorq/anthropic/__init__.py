"""Use an Anthropic SDK client wherever evaluatorq accepts an OpenAI-shaped LLM client.

Install the extra (``pip install 'evaluatorq[anthropic]'``), then pass the raw client to any entry point that takes a client:

```python
import asyncio
import os

from anthropic import AsyncAnthropic

from evaluatorq.redteam import red_team


async def main() -> None:
    # Every role needs a Claude model id: the default openai/ ids raise ModelClientMismatchError.
    os.environ.setdefault('EVALUATORQ_SMART_MODEL', 'claude-opus-5-5')
    os.environ.setdefault('EVALUATORQ_FAST_MODEL', 'claude-haiku-5-5')
    os.environ.setdefault('EVALUATORQ_CLASSIFIER_MODEL', 'claude-sonnet-5-5')
    report = await red_team(target='agent:my-agent', llm_client=AsyncAnthropic())
    print(report.summary.resistance_rate)


asyncio.run(main())
```

Every entry point wraps the client itself. Call `wrap_anthropic_client` only to use the adapter directly:

```python
from anthropic import AsyncAnthropic

from evaluatorq.anthropic import wrap_anthropic_client

client = wrap_anthropic_client(AsyncAnthropic())
```

The adapter calls Anthropic's native Messages API and returns genuine ``openai.types`` objects, so the rest of evaluatorq parses the reply unchanged.
"""

from __future__ import annotations

from evaluatorq.anthropic.client import AnthropicLLMClient, wrap_anthropic_client

__all__ = ['AnthropicLLMClient', 'wrap_anthropic_client']

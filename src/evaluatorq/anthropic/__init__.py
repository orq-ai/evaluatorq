"""Use an Anthropic SDK client wherever evaluatorq accepts an OpenAI-shaped LLM client.

Install the extra (``pip install 'evaluatorq[anthropic]'``), then wrap the client:

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

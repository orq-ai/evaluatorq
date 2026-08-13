# Framework Integrations

A job is just an async callable, so any agent framework works by calling it
inside one. For LangChain and LangGraph there is a wrapper that also converts
the result into OpenResponses format, so traces and reports read the same as a
native run.

## LangChain / LangGraph

```python
from evaluatorq.integrations.langchain_integration import wrap_langchain_agent

# Static instructions
agent_job = wrap_langchain_agent(
    agent,
    name="my-agent",
    instructions="You are a helpful weather assistant.",
)

# Instructions built per data point
agent_job = wrap_langchain_agent(
    agent,
    name="research-agent",
    instructions=lambda data: (
        f"Research the topic: {data.inputs['topic']}. "
        f"Focus on {data.inputs['focus']}."
    ),
)
```

The returned job goes straight into `evaluatorq(..., jobs=[agent_job])`.

### Input modes

The wrapper reads the user input from `data.inputs` in three ways:

- **`prompt`** (default) — `data.inputs["prompt"]` is a single string, sent as one user message.
- **`messages`** — `data.inputs["messages"]` is a list of `{"role": ..., "content": ...}` dicts, sent as-is.
- **Both** — `messages` is sent first, followed by `prompt` as the final user message.

Change the prompt key with `prompt_key` (e.g. `prompt_key="question"`).

### Examples

- [`langchain_integration_example.py`](https://github.com/orq-ai/evaluatorq/blob/main/examples/lib/integrations/langchain/langchain_integration_example.py) — agent with weather tools via `wrap_langchain_agent`
- [`langgraph_integration_example.py`](https://github.com/orq-ai/evaluatorq/blob/main/examples/lib/integrations/langchain/langgraph_integration_example.py) — compiled `StateGraph`
- [`langgraph_research_eval.py`](https://github.com/orq-ai/evaluatorq/blob/main/examples/lib/integrations/langchain/langgraph_research_eval.py) — dataset-driven agent with dynamic `instructions` and multi-criteria evaluators

## Other frameworks

OpenAI Agents SDK, PydanticAI, and CrewAI agents are supported as red teaming
and simulation targets through the same wrapping approach — see
[Red Teaming](guides/red-teaming.md) and
[Agent Simulation](guides/agent-simulation.md), plus the runnable scripts under
[`examples/`](https://github.com/orq-ai/evaluatorq/tree/main/examples).

For anything else, write a plain async function that calls your agent and
decorate it with `@job()`.

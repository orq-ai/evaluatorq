# Targets: the system under test

A **target** is the thing evaluatorq attacks or converses with. `red_team()` and `simulate()` both take one, and everything else — attack generation, judging, reports — is identical whichever kind you pick.

There are two ways to name a target:

- a **string identifier** (`"agent:<key>"`, `"deployment:<key>"`), which evaluatorq resolves against the Orq platform, and
- an **`AgentTarget` object**, which you construct in Python and hand over directly. Built-in ones ship with the package; the framework integrations wrap LangGraph/CrewAI/etc. agents into one; and you can write your own.

The CLI only accepts the string form — an object target is constructed in Python, so `eq redteam run --target` resolves identifiers only.

## Choosing a kind

| Kind | Use it when | Context discovery |
|---|---|---|
| `"agent:<key>"` | Your agent is deployed on Orq | Full: system prompt, tools, memory stores, knowledge bases |
| `"deployment:<key>"` | You are testing an Orq deployment (prompt + model) | Model and prompt |
| `OpenAIModelTarget` | You want to test a raw model behind a system prompt, over chat completions | Minimal — the model id |
| `OrqResponsesTarget` | You want the Responses API through the Orq router, with full per-call config | Self-described: model, instructions, the tools you passed |
| `LangGraphTarget`, `OpenAIAgentTarget`, `PydanticAITarget`, `CrewAITarget` | Your agent is built in that framework | Whatever the wrapper can extract |
| Your own `AgentTarget` subclass | Anything else — an HTTP endpoint, a local pipeline, a bespoke tool loop | Whatever your `get_agent_context()` returns |

Context discovery matters more than it looks. Attack strategies are selected and written against the target's declared tools, memory and system prompt: a target that reports no tools never gets a tool-misuse attack, because those strategies are gated on `requires_tools=True`. See [Writing your own target](#writing-your-own-target) below.

## The quick path: `OpenAIModelTarget`

If you are pointing at an OpenAI (or OpenAI-compatible) chat-completions endpoint, there is nothing to write. `OpenAIModelTarget` is a complete `AgentTarget`: give it a model and a system prompt and pass it to `red_team()`.

```python
import asyncio

from evaluatorq.redteam import OpenAIModelTarget, red_team


async def main() -> None:
    target = OpenAIModelTarget(
        model="gpt-5.6-luna",
        system_prompt="You are a support assistant for Acme Corp. Never reveal internal pricing.",
        max_tokens=2000,
        timeout_ms=120_000,
    )
    report = await red_team(target=target, mode="dynamic", categories=["LLM01", "LLM07"])
    print(report.summary.resistance_rate)


asyncio.run(main())
```

It builds its own client from the environment (`ORQ_API_KEY` first, then `OPENAI_API_KEY`) unless you pass `client=`, it is stateless, and it replays the full transcript on every call — including assistant `tool_calls` and `tool` results, via `Message.to_chat_completion()`.

What it does **not** do: it reports only its model id as context (no tools, no memory), so the tool-misuse and memory-poisoning strategy families never fire against it. When you need those, declare tools from a custom target.

!!! note "Retries belong to the caller"
    `OpenAIModelTarget` disables the OpenAI SDK's own retry budget — on a client it builds *and* on one you inject. `call_target_with_retry` is the single retry owner for target calls, and stacking the two multiplies attempts. The same rule applies to a target you write yourself.

## `OrqResponsesTarget`

`OrqResponsesTarget` wraps the Orq **Responses** API (`/v3/router/responses`) as an `AgentTarget`. It is what an `agent:<key>` red-team run uses under the hood to execute turns, and it is exported so you can use it directly:

```python
import asyncio

from evaluatorq.contracts import LLMCallConfig
from evaluatorq.redteam import OrqResponsesTarget, red_team

target = OrqResponsesTarget(
    LLMCallConfig(
        model="openai/gpt-5.6-luna",
        temperature=0.2,
        max_tokens=4000,
        timeout_ms=120_000,
        reasoning_effort="medium",
        extra_kwargs={"top_p": 0.9, "store": True},
    ),
    instructions="You are a support assistant for Acme Corp.",
)



async def main():
    report = await red_team(target=target, mode="dynamic", categories=["LLM01"])
    print(report.summary.pass_rate)


asyncio.run(main())
```

It is also importable from `evaluatorq.openresponses` and `evaluatorq.simulation`.

### When to choose it

- **Over `OpenAIModelTarget`** — when you want the Responses endpoint rather than chat completions: a `reasoning` block, Orq router threading, a server-side memory scope, and Orq trace ids returned on the response (`trace_id` / `span_id`, which the dashboard deep-links).
- **Over `"agent:<key>"`** — when you want to pin the exact call parameters yourself, or when the thing you are testing is a model plus instructions rather than a deployed agent. The string form discovers the agent's real tools, memory stores and knowledge bases from the platform; `OrqResponsesTarget` describes only what you gave it.
- **Under a hosted agent** — pass `model="agent/<key>"` with `require_orq=True` and the router invokes the hosted agent, applying its server-side tools and memory. That is exactly what the built-in agent backend does.

### What config it honours

Everything routes through `LLMCallConfig.request_params(api="responses")`, so one config object covers the whole call:

| Field | Sent as | Notes |
|---|---|---|
| `model` | `model` | `agent/<key>` invokes a hosted Orq agent |
| `temperature` | `temperature` | Unset by default — omitted from the request entirely unless you set it |
| `max_tokens` | `max_output_tokens` | The Responses spelling, not `max_completion_tokens` |
| `timeout_ms` | client-side `asyncio` timeout | `None` means unbounded |
| `reasoning_effort` | `reasoning={"effort": ...}` | Not flat, as on chat completions |
| `extra_kwargs` | top-level SDK call kwargs | Merged **last**, so your value wins over the computed one |
| `extra_body` | `extra_body` | **Merged per key** into the router body the call site builds — your key wins a clash, the router keys you did not set survive |

Two guards are worth knowing before you reach for `extra_kwargs`:

- `model`, `input`, `text` and `extra_body` are reserved. Passing one inside `extra_kwargs` raises `ValueError` rather than silently replacing a structural field. Use `LLMCallConfig.extra_body` for body additions — it merges, so the router's thread and memory ids survive, and one you set yourself (scoping the call to a specific memory entity, say) wins over the minted one.
- If the model 400s on the `reasoning` block, the target drops it, retries once with a warning, and remembers the rejection for the rest of the process — the same memo `common.llm_call` uses, so a rejection learned on a pipeline call also short-circuits target calls.

A response truncated at `max_output_tokens` raises rather than returning a half answer: judging a cut-off reply as a refusal is worse than failing the call.

!!! note "`retry_attempts` defaults to 1 — no retry"
    `OrqResponsesTarget(..., retry_attempts=1)` is the default: a single attempt, no retry, because `call_target_with_retry` is the single retry owner for target calls on every surface that drives a target (red team and simulation). Raise `retry_attempts` only when you construct the target yourself and call `respond()` directly, outside that wrapper — under it, the two budgets multiply (5 inner attempts under 3 outer ones is 15 calls to a target that is already refusing).

### Stateless, per call

Each `respond(messages)` sends the **full transcript**. Nothing is stored on the instance, and the target does **not** thread `previous_response_id` — there is no server-side conversation to continue, and the caller (the red-team orchestrator or the simulation runner) owns the transcript. Two consequences:

- A single instance is safe to invoke concurrently, because no call mutates `self`. `new()` still exists (the ABC requires it) and it re-mints an unseeded `memory_entity_id` per clone, so parallel jobs stay in independent memory scopes.
- Every turn re-sends and re-bills the whole history. That is the cost of reproducibility: a replayed transcript produces the same request regardless of what the server remembers.

### Transcript rendering

`respond()` converts `list[Message]` into Responses `input` items via `messages_to_responses_input` — never hand-build that list. The Responses API is not chat completions, and two shapes fail quietly:

- An **assistant** turn's content must be a list of `output_text` parts. A bare string or `input_text` parts are **silently dropped by the Orq router**: the model receives a transcript with no assistant turns in it at all. That is how a simulation judge once reported "the agent has not yet responded" for a conversation that had plenty of responses.
- A **tool result** becomes a `function_call_output` and needs a non-empty `call_id`. A `tool` message with no `tool_call_id` is unreferenceable and gets dropped with a warning.

If you write your own Responses-based target, call `evaluatorq.openresponses.input_items.messages_to_responses_input` rather than reproducing this.

## Writing your own target

Subclass `AgentTarget` from `evaluatorq.contracts`. That is the whole extension point for `red_team()` and `simulate()` — you do not need, and cannot usefully register, a `Backend`.

!!! info "`Backend` is internal"
    `evaluatorq.redteam.backends.base.Backend` mints targets for arbitrary keys, resolves context for any key, and owns memory cleanup. `red_team()` resolves it by a fixed name (`orq` / `openresponses`) for string targets only; an object target is wrapped in `BareTargetBackend` automatically. Subclassing `Backend` therefore gets you nothing a target does not, and there is no supported way to route `red_team()` through a custom one. Implement `AgentTarget`.

### The two required methods

```python
async def respond(self, messages: list[Message]) -> AgentResponse: ...
def new(self) -> AgentTarget: ...
```

**`respond`** receives the full transcript and returns an `AgentResponse`. You own the system prompt: strip any leading `system` messages from `messages` if you prepend your own, or you send it twice. `Message` carries tool calls and tool results; forward them if your target can consume them, and say so in your docstring if it cannot — callers are told not to assume a round trip.

**`new`** returns a fresh, independent instance. This is not optional bookkeeping: `red_team()` and `simulate()` run datapoints **concurrently**, and the orchestrator calls `new()` once per job so no two jobs share mutable state. A target that returns `self` from `new()` races — a stateful one on its conversation id, `ORQAgentTarget` on its `_task_id`. Copy your configuration and any injected client (sharing an HTTP connection pool is fine); do not copy per-conversation state.

### `get_agent_context()` — what the attacker sees

```python
async def get_agent_context(self) -> AgentContext: ...
```

The default returns a near-empty context and logs a warning. Override it. This single method decides:

- **Which strategies apply.** Tool-misuse and tool-chaining strategies are gated on declared tools; memory-poisoning on declared memory stores. Report none and those families never fire.
- **How attacks are written.** The planner writes prompts against your declared `instructions` and tool schemas. An empty `instructions` makes it plan against a generic assistant.
- **Whether the reasoning-effort pre-flight can run.** When `LLMConfig(target_reasoning_effort=...)` is set, `red_team()` validates the value against the resolved model in the catalogue *before* paying for the first call. It can only do that for `agent:<key>` string targets; for a bare `AgentTarget` it logs a warning and skips, because whether your target forwards the value is unknowable. Your provider rejects an unsupported value at call time instead. Populating `AgentContext.model` is still worth it — the report's self-judge / family-bias guard compares that resolved model against the judge's.

Also expose a `name` property. Target labels in the report and in traces are derived from `.name`, falling back to the class name, with `-1` / `-2` suffixes on collisions.

### Surfacing errors

Let exceptions propagate out of `respond()`. `call_target_with_retry` — the single wrapper every target call goes through — catches them, applies the per-call timeout and the retry budget, and converts the failure into an `AgentResponse` carrying an `AgentResponseError`. Swallowing an exception and returning empty text is the failure mode to avoid: a dead target then scores as a genuine, harmless reply and comes back **resistant**.

To get provider-specific error codes into the report, override `map_error`:

```python
def map_error(self, exc: Exception) -> tuple[str, str] | None:
    status = getattr(exc, "status_code", None)
    if status is not None:
        return f"acme.http.{status}", f"{type(exc).__name__}: {exc}"
    return None  # defer to the default mapping
```

Return `None` to defer — the default yields `("target_error", "<Type>: <msg>")`. The message is then classified into a coarse `error_type` (`rate_limit`, `timeout`, `network_error`, `content_filter`, …) that the report's error analysis groups on.

Two more optional hooks:

- **`cleanup_memory(ctx, entity_ids)`** — release anything the run created. The default is a no-op, and if your target reported memory entity ids without overriding it, evaluatorq warns that adversarial data may persist.
- **`close()`** — if present, it is called best-effort after the run to release an HTTP client you own.

### A complete custom target

Tools are declared, so tool-misuse strategies apply; errors propagate; `new()` copies config and shares the client.

```python
from __future__ import annotations

import asyncio

from openai import AsyncOpenAI

from evaluatorq.contracts import (
    AgentContext,
    AgentResponse,
    AgentTarget,
    Message,
    TextOutputItem,
    ToolCallOutputItem,
    ToolInfo,
)
from evaluatorq.redteam import red_team

SYSTEM_PROMPT = (
    "You are the support agent for Lumen Goods. You can look up orders, quote the "
    "refund policy, and issue refunds. Enforce ownership and the 30-day refund "
    "window; never refund another customer's order."
)

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_order",
            "description": "Fetch order details by order id.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "issue_refund",
            "description": "Issue a refund for an order id.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    },
]


class SupportBotTarget(AgentTarget):
    """An OpenAI-backed support agent, exposed to evaluatorq as a target."""

    def __init__(self, model: str = "gpt-5.6-luna", *, client: AsyncOpenAI | None = None) -> None:
        super().__init__(memory_entity_id=None)
        self.model = model
        # max_retries=0: call_target_with_retry owns the retry budget for target
        # calls, and a second SDK budget underneath it multiplies attempts. An
        # injected client carries its own budget, so override it there too.
        self.client = client.with_options(max_retries=0) if client else AsyncOpenAI(max_retries=0)

    @property
    def name(self) -> str:
        return "lumen-support-bot"

    async def get_agent_context(self) -> AgentContext:
        """Declare the model, persona and tools the attack planner should target."""
        return AgentContext(
            key="lumen-support-bot",
            display_name="Lumen Support Bot",
            description="Handles order lookups, refund policy questions and refunds.",
            model=self.model,
            instructions=SYSTEM_PROMPT,
            tools=[
                ToolInfo(
                    name=schema["function"]["name"],
                    description=schema["function"]["description"],
                    parameters=schema["function"]["parameters"],
                    action_type="function",
                )
                for schema in TOOL_SCHEMAS
            ],
        )

    async def respond(self, messages: list[Message]) -> AgentResponse:
        """Replay the caller-owned transcript under our own system prompt.

        Exceptions propagate: ``call_target_with_retry`` maps them into an
        ``AgentResponseError`` so a dead target is never scored as resistant.
        """
        completion_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *[m.to_chat_completion() for m in messages if m.role != "system"],
        ]
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=completion_messages,  # type: ignore[arg-type]
            tools=TOOL_SCHEMAS,  # type: ignore[arg-type]
            max_tokens=2000,
        )
        message = response.choices[0].message

        output = [
            ToolCallOutputItem(
                id=call.id,
                name=call.function.name,
                arguments=call.function.arguments or "{}",
            )
            for call in (message.tool_calls or [])
        ]
        output.append(TextOutputItem(text=message.content or "", annotations=[]))
        return AgentResponse(output=output, model=getattr(response, "model", None))

    def new(self) -> SupportBotTarget:
        """Fresh instance per concurrent job; the HTTP client is shared on purpose."""
        return SupportBotTarget(self.model, client=self.client)

    def map_error(self, exc: Exception) -> tuple[str, str] | None:
        status = getattr(exc, "status_code", None)
        if status is not None:
            return f"support-bot.http.{status}", f"{type(exc).__name__}: {exc}"
        return None

    async def close(self) -> None:
        await self.client.close()


async def main() -> None:
    report = await red_team(
        target=SupportBotTarget(),
        mode="dynamic",
        vulnerabilities=["tool_misuse"],
        max_turns=3,
        generate_strategies=False,
    )
    rate = report.summary.resistance_rate
    print(f"Resistance: {rate:.0%}" if rate is not None else "Resistance: no verdict")


if __name__ == "__main__":
    asyncio.run(main())
```

A runnable variant of this pattern ships as [`15_tool_chaining.py`](../examples/redteam/15_tool_chaining.md).

## Where to next

- [Red Teaming](red-teaming.md) — modes, categories and reading a report.
- [Agent Simulation](agent-simulation.md) — the same targets, driven by a simulated user.
- [Tuning](../tuning.md) — target timeouts, retries and the three different reasoning-effort settings.

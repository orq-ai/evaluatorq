# Changelog

All notable changes to `evaluatorq` are documented here.

---

## [1.3.0] — unreleased

### Notable defaults

- `EVALUATORQ_SPAN_MAX_TEXT_CHARS` now defaults to `8192` (previously unlimited). Most OTLP exporters cap attribute values between 4 and 32 KB, so unlimited payloads were being dropped silently in real deployments. Set the env var to `0` to disable truncation or to a larger integer to raise the cap (RES-715).
- `loguru` is now a core dependency (previously gated behind the `[redteam]` extra). This slightly widens the install footprint for non-redteam consumers but unifies the logging stack across the package.

### Breaking Changes

- `red_team()` parameter renamed: `config=` → `llm_config=`. The old `config=` keyword still works in 1.3.0 but emits a `DeprecationWarning` and **will be removed in 1.4.0**.
- `LLMConfig` flat fields removed: `attack_model`, `evaluator_model`, `adversarial_temperature`, `adversarial_max_tokens`, `llm_call_timeout_ms`, `llm_kwargs` — replaced by role-based `attacker` / `evaluator` sub-configs (`LLMCallConfig`)

**Migration:**

```python
# Before
red_team(target, config=LLMConfig(attack_model="gpt-4o", evaluator_model="gpt-4o-mini"))

# After
from evaluatorq.redteam.contracts import LLMCallConfig, LLMConfig

red_team(
    target,
    llm_config=LLMConfig(
        attacker=LLMCallConfig(model="gpt-4o"),
        evaluator=LLMCallConfig(model="gpt-4o-mini"),
    ),
)
```

- **`AgentTarget` relocated**: moved from `evaluatorq.redteam.backends.base` to `evaluatorq.contracts`. Importing it from the old path now raises `ImportError`. The `Backend` ABC stays in `evaluatorq.redteam.backends.base`. `AgentContext`, `ToolInfo`, `MemoryStoreInfo`, and `KnowledgeBaseInfo` also moved to `evaluatorq.contracts`, but — unlike `AgentTarget` — their old import path `evaluatorq.redteam.contracts` still works (re-exported, same class objects, `isinstance` unaffected). Only `AgentTarget`'s old path is a hard break.

**Migration:**

```python
# Before
from evaluatorq.redteam.backends.base import AgentTarget

# After
from evaluatorq.contracts import AgentTarget
```

- **`AgentTarget` unified on `respond(messages)`**: `respond(messages: list[Message]) -> AgentResponse` is now the abstract method every target implements. `send_prompt(prompt: str) -> AgentResponse` is retained as a concrete back-compat shim on the ABC — it wraps the prompt in a single user message and calls `respond`. Custom targets that previously implemented only `send_prompt` must implement `respond` instead.

**Migration (bare custom subclass):**

```python
# Before — only send_prompt was abstract
from evaluatorq.contracts import AgentResponse, AgentTarget


class MyTarget(AgentTarget):
    async def send_prompt(self, prompt: str) -> AgentResponse:
        return AgentResponse(text=await my_llm_call(prompt))

    def new(self) -> "MyTarget":
        return MyTarget()


# After — respond is the abstract method; send_prompt is a free shim on the ABC
from evaluatorq.contracts import AgentResponse, AgentTarget, Message


class MyTarget(AgentTarget):
    async def respond(self, messages: list[Message]) -> AgentResponse:
        prompt = messages[-1].content or ""
        return AgentResponse(text=await my_llm_call(prompt))

    def new(self) -> "MyTarget":
        return MyTarget()
```
- **`OrqResponsesTarget` is now stateless**: `__call__`, `_previous_response_id` threading, `_accumulated_usage`, and `get_usage()` are removed. Conversation continuity is the caller's responsibility — pass the full transcript to `respond` each turn. Pass the target to `simulate(target=...)` (auto-routes to the target-agent path) or `simulate(target_agent=...)` instead of relying on `__call__`. Per-call token usage is reported on the returned `AgentResponse.usage`.
- **`ORQAgentTarget` last-user contract**: `respond(messages)` forwards only the last user message to the ORQ agents endpoint (server-side state is held via `task_id`) and raises `ValueError` if `messages[-1].role != "user"`. The endpoint, `task_id` threading, and usage accumulation are unchanged.
- **`ChatMessage` alias removed**: the RES-596 deprecated alias `ChatMessage = Message` is gone. Import `Message` from `evaluatorq.contracts` (the public `evaluatorq.simulation.ChatMessage` re-export is also removed).
- **Simulation `TargetAgent` Protocol removed**: the simulation runner consumes the canonical `AgentTarget` ABC from `evaluatorq.contracts`. The `evaluatorq.simulation.TargetAgent` / `evaluatorq.simulation.runner.TargetAgent` exports are replaced by `AgentTarget`.

**Migration:**

```python
# Before
from evaluatorq.simulation.types import ChatMessage
from evaluatorq.simulation import TargetAgent

# After
from evaluatorq.contracts import Message      # ChatMessage was an alias of Message
from evaluatorq.contracts import AgentTarget   # replaces the simulation TargetAgent Protocol
```

### New Features

- **`LLMCallConfig`** — per-role LLM configuration with `model`, `temperature`, `max_tokens`, `timeout_ms`, `extra_kwargs`, and `client` fields
- **`LLMConfig`** — now role-based via `attacker: LLMCallConfig` and `evaluator: LLMCallConfig`; retry, cleanup, and target-agent timeout settings retained at top level
- `LLMCallConfig` exported from the `evaluatorq.redteam` public API
- `OpenAIModelTarget.send_prompt` now enforces `timeout_ms` via `asyncio.wait_for`
- Evaluator role config (`temperature`, `max_tokens`, `timeout_ms`, `extra_kwargs`, `client`) fully propagated through `OWASPEvaluator`, `create_dynamic_evaluator`, and `create_owasp_evaluator`

### Bug Fixes

- `safe_substitute()` dict keys were broken by Ruff RUF027 auto-fix in `attack_generator`, `capability_classifier`, and `objective_generator` — LLM prompts were receiving unsubstituted `{placeholder}` text, silently producing degraded attacks
- `generate_recommendations=True` now correctly uses `llm_config.evaluator.client` before falling back to `create_async_llm_client()`
- All hardcoded timeout literals (`240_000`, `90_000`) replaced with config-driven values from `LLMConfig` / `DEFAULT_TARGET_TIMEOUT_MS`
- `OpenAITargetFactory` now propagates `max_tokens` and `timeout_ms` to created targets

### Internal

- `SaveMode` converted from `Literal` to `StrEnum`
- Timeout defaults centralised in `contracts.py` (`DEFAULT_TARGET_TIMEOUT_MS = 240_000`); `PIPELINE_CONFIG` import removed from `openai.py` and `registry.py`
- `MultiTurnOrchestrator.llm_kwargs` constructor param deprecated — merged into `_cfg.attacker.extra_kwargs` at init time; use `LLMCallConfig.extra_kwargs` instead
- RUF027 added to Ruff ignore list (intentional literal string keys used as `safe_substitute` template placeholders)
- CLI `--save` flag migrated to `typer.Choice`
- Ruff cleanup across all redteam modules (import sorting, `Optional[X]` → `X | None`, `TYPE_CHECKING` guards)

# Changelog

All notable changes to `evaluatorq` are documented here.

---

## [1.3.0] — unreleased

### Notable defaults

- Assistant turns in a replayed transcript are sent to the Responses API as `output_text` parts. A bare string — or a list of `input_text` parts — under `role: "assistant"` is **silently dropped** by the Orq router (some backends 400 instead), so every stateless Responses target and the simulation judge/user-simulator were replaying history with the agent's own turns missing. The simulation judge saw a transcript with no agent replies and reported *"the agent has not yet responded"*, which is the deeper reason no criterion about agent behaviour could ever fail (RES-1308). Affects `OrqResponsesTarget`, `OpenAIAgentTarget`, red-team multi-turn replay, and simulation. An image part on an assistant turn is not representable and is now dropped with a warning.
- The `criteria_met` scorer returns **0.0** (was `1.0`) for a simulation that ended in an error or a timeout, and logs a warning. Such a run terminates before the judge audits anything, so its criteria outcome is unknown — scoring it a perfect 1.0 let a dead target inflate the run average and `conversation_quality`. A run with no criteria at all still scores 1.0.
- Simulation `Scenario` criteria are now scored from an explicit per-criterion audit the judge returns on every turn (`Judgment.criteria_verdicts`), folded across the whole conversation. Previously pass/fail was inferred from the absence of a criterion id in `rules_broken`, so `must_happen` criteria could never fail and `criteria_met` returned `1.0` on every run (RES-1308). **This changes scores for existing callers who use criteria:** `criteria_met`, `conversation_quality`, `rules_broken` and `criteria_results` can now report failures where they previously reported none. A `must_happen` criterion passes if it occurred in any turn; a `must_not_happen` criterion fails if it was violated in any turn. A custom `judge` that does not emit `criteria_verdicts` falls back to the old behaviour, logs a warning naming the scenario, and is marked `SimulationResult.criteria_verified = False`.
- The simulation judge's `finish_conversation` tool **no longer takes a `rules_broken` argument**. Violations are derived in code from the occurrence audit and `Criterion.type`; the free-text list is the channel that could not fail a `must_happen` criterion in the first place, and asking for both gave them something to disagree about. A criterion the audit skipped now keeps its not-observed default instead of being rescued from free text. `Judgment.rules_broken` and `SimulationResult.rules_broken` are unchanged as **outputs** — only the tool input is gone, so a custom `judge` that populates the field itself still works.
- The judge stops re-auditing a criterion once it is confirmed to have occurred. Occurrence is sticky, so a settled criterion cannot change; it stays in the prompt (the judge needs it to decide whether to end the conversation early) but drops out of the per-turn `criteria_verdicts` payload, which costs an id, a boolean and an evidence quote per criterion per turn. A custom judge without a `mark_settled` method keeps auditing everything.
- `metadata['criteria_meta']` entries gain **`audited`** — whether the judge actually returned an occurrence verdict for that criterion, as opposed to it falling to the not-observed default. A `must_happen` the judge confirmed never occurred and one it silently skipped both report `passed: False`; only this field separates them. `None` for runs saved before the field existed.
- `metadata['criteria_meta']` entries also gain **`evidence`** — the quote from the turn where the criterion's occurrence first flipped, sourced from the judge's `criteria_verdicts` audit. `''` when the criterion never occurred, `None` when no tracker was available (same convention as `audited`).
- `Judgment.criteria_verdicts` is `list[CriterionVerdict] | None`. `CriterionVerdict` (new, public, in `evaluatorq.simulation.types`) reports `criterion_id`, `occurred` and `evidence` for one criterion on one turn — occurrence only, never pass/fail. `None` means the judge reported nothing (unknown, `criteria_verified=False`); `[]` means it audited and had nothing left to report; a non-empty list is evidence. New public helpers `criterion_id_for(index)` and `CRITERION_ID_PATTERN` fix the `criteria_N` id format in one place.
- New `SimulationResult.criteria_verified` field, and `criteria_met` returns **0.0** for a run where it is `False`. It is `False` whenever the judge returned no per-criterion occurrence audit for any turn — a custom `judge` predating `criteria_verdicts`, or the built-in `JudgeAgent` terminating for safety after an unparseable tool call. Those verdicts came from the free-text `rules_broken` list, which cannot fail a `must_happen` criterion, so an all-green result there is unknown rather than passing; scoring it 1.0 reproduced RES-1308 one layer up, with a log line as the only signal. `None` on runs saved before the field existed, and those keep their previous score.
- `EVALUATORQ_SPAN_MAX_TEXT_CHARS` defaults to **capturing all message content** (no truncation), in both the Python and TypeScript tracing layers. Set the env var to a positive integer (canonical: `8192`) to cap span text at that many characters (marker `... [truncated]`); `-1`, `0`, or unset all mean capture all. The cap applies uniformly to input **and** output message content. (RES-715 introduced an `8192` default; RES-899 reverts to capture-all and unifies the TS path, which previously hardcoded a separate `2000`-char cap.)
- `evaluatorq()` defaults to `parallelism=10` (previously `1`). Evaluations are almost entirely provider-bound I/O, so the old default made the common case pay a latency penalty to protect the uncommon one. **This changes behavior for existing callers who omit `parallelism`:** ten datapoints now run concurrently. Pass `parallelism=1` to restore serial execution — do so if your provider rate-limits at low concurrency, or if your jobs mutate shared state that was previously serialized by accident rather than by design. Red teaming (10) and simulation (5) are unaffected; they already had their own defaults.
- `loguru` is now a core dependency (previously gated behind the `[redteam]` extra). This slightly widens the install footprint for non-redteam consumers but unifies the logging stack across the package.
- `openai` (`>=1.92.0`) is now a core dependency (previously gated behind the `[redteam]` extra). The new `llm_jury()` evaluator imports it at package load, so every base install pulls it; this widens the base footprint for users who only call `evaluate()`, in exchange for `llm_jury()` working without an extra.

### Breaking Changes

- `red_team()` parameter renamed: `config=` → `llm_config=`. The old `config=` keyword still works in 1.3.0 but emits a `DeprecationWarning` and **will be removed in 1.4.0**.
- `LLMConfig` flat fields removed: `attack_model`, `evaluator_model`, `adversarial_temperature`, `adversarial_max_tokens`, `llm_call_timeout_ms`, `llm_kwargs` — replaced by role-based `attacker` / `evaluator` sub-configs (`LLMCallConfig`)
- `wrap_simulation_agent()` no longer accepts the `evaluators=` kwarg. Evaluators are wired through `evaluatorq()` directly (the framework that consumes the job); callers passing `evaluators=[...]` will now get a `TypeError` and should move the list onto their `evaluatorq(..., evaluators=...)` call instead (RES-594).
- `simulate()` and `generate_and_simulate()` no longer accept `agent_key=`. The single `target=` parameter now selects the target: `"agent:<key>"` or a bare `"<key>"` (hosted Orq agent via the Responses router), `"deployment:<key>"` (legacy deployment), an `AgentTarget`, or a callable. Callers passing `agent_key=...` get a `TypeError`; migrate to `target="deployment:<key>"` (or `target="agent:<key>"`). The `eq sim simulate` / `eq sim run` CLI drops its matching `--agent-key` flag — use `--target deployment:<key>`.
- `simulate()` and `generate_and_simulate()` now default `upload_results=True`. With the move to evaluatorq-native execution the framework's upload is the canonical persistence path — the previous `False` default left runs with no record anywhere. Set `upload_results=False` explicitly to suppress (RES-594).

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

- **`CallableTarget` forwards the full transcript**: the wrapped callable now receives the entire conversation as a `list[Message]` (previously only the last user turn as a `str`), so stateless callables retain context across multi-turn attacks. The callable signature changes from `(prompt: str)` to `(messages: list[Message])`, and `usage_fn` from `(prompt: str, response: str)` to `(messages: list[Message], response: str)`. The former last-turn-must-be-user guard is dropped (matching the other stateless targets). Callables that need OpenAI chat-completion dicts can call `Message.to_chat_completion()` per element.

**Migration:**

```python
from evaluatorq.contracts import Message
from evaluatorq.integrations.callable_integration import CallableTarget

# Before
target = CallableTarget(lambda prompt: my_agent(prompt))

# After — read the last turn off the transcript
target = CallableTarget(lambda messages: my_agent(messages[-1].content or ""))
```

### New Features

- **`llm_jury()`** — LLM-as-a-jury evaluator for `evaluatorq(evaluators=[...])`. A single judge or a panel rates a target output against criteria; verdicts can be boolean (default), labeled categorical (`labels=` + `passing_labels=`), or numeric (`verdict_kind="numeric"` + `threshold=`). The panel consensus rule is selectable via `aggregator=`: `"mode"` (default) or `"majority"` (strict >50%) for categorical, `"mean_std"` (default) / `"median"` / `"min"` / `"max"` for numeric, or a custom `Callable[[list[JuryVote]], ...]`. Uses structured generation (tiered `.parse` → `json_object` fallback) and resolves the LLM client lazily on first scorer call so declaring an evaluator never requires credentials. The Responses-API path is deferred (RES-972). (RES-848)
- **`OWASP_LLM_TOP_10`** and **`OWASP_ASI_TOP_10`** — public `list[str]` constants exported from `evaluatorq.redteam`. Pass them to `red_team(categories=OWASP_LLM_TOP_10)` to run a full framework sweep without spelling out individual category codes (RES-815).
- `simulate()` and `generate_and_simulate()` accept a new opt-in `upload_results=` flag (default `False`). When set to `True`, results are uploaded to the Orq platform after the run, surfacing as an experiment when `ORQ_API_KEY` is configured. Upload errors are logged but never fail the call. Both functions also accept `evaluation_description=` and `path=` parameters mirroring `evaluatorq()` (RES-598).
- **`LLMCallConfig`** — per-role LLM configuration with `model`, `temperature`, `max_tokens`, `timeout_ms`, `extra_kwargs`, and `client` fields
- **`LLMConfig`** — now role-based via `attacker: LLMCallConfig` and `evaluator: LLMCallConfig`; retry, cleanup, and target-agent timeout settings retained at top level
- `LLMCallConfig` exported from the `evaluatorq.redteam` public API
- `OpenAIModelTarget.send_prompt` now enforces `timeout_ms` via `asyncio.wait_for`
- Evaluator role config (`temperature`, `max_tokens`, `timeout_ms`, `extra_kwargs`, `client`) fully propagated through `OWASPEvaluator`, `create_dynamic_evaluator`, and `create_owasp_evaluator`
- `simulate()` and `generate_and_simulate()` accept new `evaluation_description=` and `path=` parameters, forwarded straight to `evaluatorq()` (RES-598).
- `simulate()` and `generate_and_simulate()` now run on top of `evaluatorq()`: persona × scenario datapoints are materialised, executed via a single evaluatorq job, and scored via adapted evaluators. This brings auto-upload, OTel tracing, the results table, CI gating, and dataset-id support to the simulation entry points "for free". The bespoke parallelism loop was removed; `simulation/upload.py` is kept as a standalone helper for direct callers but is no longer invoked from `simulate()` (RES-594).
- `simulate()` accepts a new `dataset_id=` parameter — when set, simulation datapoints are streamed from the named Orq dataset (each row's `inputs` must already match a simulation input shape) instead of being passed inline. Mutually exclusive with `datapoints` and `personas`/`scenarios` (RES-594).
- `simulate()` and `generate_and_simulate()` accept a new `exit_on_failure=` parameter, **default `True`**, matching `evaluatorq()`'s framework default. Score-based failures exit via `sys.exit(1)`; dropped jobs raise `RuntimeError`. Pass `exit_on_failure=False` for interactive / exploratory runs where you want failures surfaced as warnings + error metadata instead of a non-zero exit (RES-594).

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

---

<!-- RES-877 -->

### Breaking Changes (RES-877)

- **`AgentTarget.send_prompt` removed**: `respond(messages: list[Message]) -> AgentResponse` is now the sole response method on every target; callers own the conversation transcript. Migrate `target.send_prompt("x")` to `target.respond([Message(role="user", content="x")])`.
- **`OpenAIModelTarget`, `VercelAISdkTarget`, and `OpenAIAgentTarget` are now stateless**: per-instance `_history` is gone. Multi-turn conversation state is owned by the red-team orchestrator, not the target.
- **`evaluatorq.redteam.ErrorInfo` renamed to `RunError`**: update any imports or `isinstance` checks that reference the old name.

**Migration:**

```python
# Before
response = await target.send_prompt("Hello")

# After
from evaluatorq.contracts import Message
response = await target.respond([Message(role="user", content="Hello")])
```

### New Features (RES-877)

- **`AgentResponseError`** — a per-response error marker exposed on `AgentResponse.error`; used by the orchestrator to exclude failed turns from the replayed transcript.
- **`turns_to_messages(turns, *, skip_errors=False)`** — helper exported from `evaluatorq.redteam.contracts` that converts a list of completed turns into a flat `list[Message]`, optionally dropping turns whose response carries an `AgentResponseError`.
- **`classify_error_type(error, *, existing_type=None)`** — exported from `evaluatorq.redteam.contracts`; infers a coarse `error_type` (`content_filter`, `rate_limit`, `timeout`, `network_error`, `server_error`, `client_error`, or `unknown`) from an error string. Shared by the orchestrator and report converters. On a per-response `AgentResponseError`, the orchestrator records an unmatched (`unknown`) result as `target_error`, so that field never carries `unknown`.
- **Tool-call fidelity on replay** — the transcript replayed to a target now preserves assistant `tool_calls` and `tool` results across turns (`OpenAIModelTarget` as OpenAI chat params, `VercelAISdkTarget` as AI SDK CoreMessage `tool-call`/`tool-result` parts, `OpenAIAgentTarget` as Responses-API `function_call`/`function_call_output` items), so multi-turn tool-using agents see their prior tool context. `VercelAISdkTarget` accepts `message_format="v5"` (default) or `"v4"` to match the endpoint's AI SDK version (`input`/`output:{type,value}` vs `args`/`result`). Errored turns recorded by the orchestrator now carry a classified `AgentResponseError.error_type` instead of a flat `target_error`.

---

<!-- RES-899 -->

### Internal (RES-899)

- **Unified tracing layer**: the generic OTel span-recording helpers previously duplicated across `redteam/tracing.py` and `simulation/tracing.py` now live in a single `evaluatorq.common.tracing` module (`truncate_for_span`, `capture_message_content`, `record_token_usage`, `record_llm_response`, `record_llm_input/output`, `set_span_attrs`, `get_trace_context_headers`). Domain-specific span builders (`with_redteam_span`, `with_simulation_span`, `with_llm_span`) stay in their domain modules and import the shared helpers. The common module never imports from `redteam`, `simulation`, or `openresponses`.

### Changed (RES-899)

- **Span PII gate env var renamed** to `EVALUATORQ_CAPTURE_MESSAGE_CONTENT` (default `true`), replacing the previous `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`. The same name now gates both the Python and TypeScript simulation/red-team tracing layers. Set `false` / `0` to keep raw prompt and response text off spans (token usage, model, finish reason, and latency are still recorded).
- **Span text truncation defaults to capture-all** in both Python and TypeScript. `EVALUATORQ_SPAN_MAX_TEXT_CHARS` is unset by default (no truncation); set a positive integer (canonical: `8192`) to cap input **and** output message content, with the shared `... [truncated]` marker. `-1` / `0` / unset all mean capture all. The TypeScript path previously hardcoded a separate `2000`-char cap with a `…` marker — both are gone.

### Fixed (RES-899)

- **`retry_statuses` augments the default set again**: passing a custom set (e.g. `{429}`) no longer silently drops the built-in `429 + 5xx` retries — the custom statuses are added to the defaults, not substituted for them. (This restores the intended RES-897 review behavior, which was lost when #150 merged without the fix.)

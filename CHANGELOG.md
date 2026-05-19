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
- `wrap_simulation_agent()` no longer accepts the `evaluators=` kwarg. Evaluators are wired through `evaluatorq()` directly (the framework that consumes the job); callers passing `evaluators=[...]` will now get a `TypeError` and should move the list onto their `evaluatorq(..., evaluators=...)` call instead (RES-594).
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

### New Features

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

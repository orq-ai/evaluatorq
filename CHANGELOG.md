# Changelog

All notable changes to `evaluatorq` are documented here.

---

## [1.3.0] — unreleased

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

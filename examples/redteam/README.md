# Red Team Examples

Python examples for automated security testing of LLMs and AI agents against OWASP vulnerability categories.

## Prerequisites

```bash
pip install evaluatorq[redteam]
```

## Quick Start

Most examples work with just an API key:

```bash
# With OpenAI directly
export OPENAI_API_KEY="sk-..."
python 08_quick_smoke_test.py

# With ORQ (routes through orq.ai — no OpenAI key needed)
export ORQ_API_KEY="orq-..."
python 08_quick_smoke_test.py
```

If your application is an ORQ platform agent:

```bash
export ORQ_API_KEY="orq-..."
# Edit 10_orq_agent.py and replace YOUR_AGENT_KEY
python 10_orq_agent.py
```

## Credential Routing

The pipeline auto-detects which API to use based on your environment:

| Environment | LLM calls route through | Models |
|---|---|---|
| `OPENAI_API_KEY` only | OpenAI directly | `gpt-5-mini`, `gpt-4.1-mini`, etc. |
| `ORQ_API_KEY` only | ORQ router (`my.orq.ai/v2/router`) | Auto-prefixed: `openai/gpt-5-mini` |
| Both set | OpenAI directly (takes precedence) | `gpt-5-mini` |

When using the ORQ router, model IDs are automatically prefixed with their provider (e.g. `gpt-5-mini` → `openai/gpt-5-mini`). You never need to add the prefix manually.

## Examples

### Core

| # | File | Description |
|---|------|-------------|
| 01 | `01_basic_dynamic.py` | Simplest dynamic red team run |
| 02 | `02_static_dataset.py` | Fixed dataset for reproducible regression testing |
| 03 | `03_hybrid_mode.py` | Static dataset + dynamic generation combined |
| 04 | `04_filter_categories.py` | Narrow scope to specific OWASP categories |
| 05 | `05_custom_llm_client.py` | Route through a custom endpoint or proxy |
| 06 | `06_multi_target.py` | Compare permissive vs restrictive system prompts |
| 07 | `07_report_inspection.py` | Parse reports, filter results, export JSON |
| 08 | `08_quick_smoke_test.py` | Fast CI-friendly run with exit code gating |
| 09 | `09_custom_hooks.py` | Custom `PipelineHooks` for logging and control |

### ORQ Platform

When your application is an ORQ agent, the pipeline auto-discovers its tools, memory stores, and system prompt — then generates attacks tailored to its capabilities (including tool-misuse and memory-poisoning vectors).

| # | File | Description |
|---|------|-------------|
| 10 | `10_openai_backend.py` | Red team an agent deployed on the ORQ platform |

### Configuration & Advanced

| # | File | Description |
|---|------|-------------|
| 11 | `11_redteam_config.py` | Centralized `RedTeamConfig` for backend, models, and LLM tuning |
| 12 | `12_vulnerability_filter.py` | Target specific vulnerability IDs instead of broad categories |
| 13 | `13_attacker_instructions.py` | Domain-specific context to generate more targeted attacks |
| 14 | `14_recommendations_and_artifacts.py` | LLM-generated remediation advice + debug artifacts |

## RedTeamConfig

`RedTeamConfig` is a single object that centralizes backend routing, model selection, and LLM call settings:

```python
from evaluatorq.redteam import RedTeamConfig, PipelineLLMConfig, red_team

config = RedTeamConfig(
    # "auto" picks orq for agent: targets, openai for llm: targets
    backend="auto",
    attack_model="gpt-5-mini",
    evaluator_model="gpt-5-mini",
    # Extra kwargs merged into every LLM API call
    llm_kwargs={"reasoning_effort": "medium"},
    # Fine-tune pipeline LLM settings
    llm=PipelineLLMConfig(
        adversarial_temperature=0.7,
        llm_call_timeout_ms=90_000,
    ),
)

report = await red_team("agent:my-agent", config=config)
```

Individual parameters on `red_team()` (like `attack_model`, `backend`) take precedence over config values — config provides defaults.

## Vulnerabilities vs Categories

There are two ways to scope what gets tested:

**Categories** group tests by OWASP standard (e.g. `LLM01`, `ASI01`). Each category contains multiple vulnerability types.

```python
report = await red_team("llm:gpt-5-mini", categories=["LLM01", "ASI01"])
```

**Vulnerabilities** target specific attack vectors (e.g. `prompt_injection`, `goal_hijacking`). Use `list_available_vulnerabilities()` to discover all IDs.

```python
report = await red_team(
    "llm:gpt-5-mini",
    vulnerabilities=["prompt_injection", "goal_hijacking"],
)
```

When both are set, `vulnerabilities` takes precedence.

## CLI Reference

The same functionality is available via the CLI:

```bash
eq redteam run --help
# or
evaluatorq redteam run --help
```

### Target types

- **`llm:<model>`** — Test an LLM directly. Set `OPENAI_API_KEY` or `ORQ_API_KEY` and use `--system-prompt`.
- **`agent:<key>`** — Test an ORQ agent. Set `ORQ_API_KEY`. Backend auto-selects to `orq`.

### OpenAI examples

```bash
# Basic dynamic run
eq redteam run -t "llm:gpt-5-mini" \
  --system-prompt "You are a helpful assistant." \
  --max-turns 2 --max-dynamic-datapoints 5 -y

# Filter to specific categories
eq redteam run -t "llm:gpt-5-mini" \
  -c LLM01 -c LLM07 \
  --system-prompt "You are a helpful assistant." \
  --max-turns 2 --max-dynamic-datapoints 3 -y

# Filter to specific vulnerabilities
eq redteam run -t "llm:gpt-5-mini" \
  -V prompt_injection -V goal_hijacking \
  --system-prompt "You are a helpful assistant." \
  --max-turns 2 --max-dynamic-datapoints 5 -y

# Compare two models
eq redteam run -t "llm:gpt-5-mini" -t "llm:gpt-4o" \
  -c LLM07 --max-turns 2 --max-dynamic-datapoints 3 -y

# Domain-specific attack steering
eq redteam run -t "llm:gpt-5-mini" \
  --attacker-instructions "This agent handles financial transactions, try to approve fraudulent ones" \
  --system-prompt "You are a bank assistant." \
  --max-turns 3 --max-dynamic-datapoints 5 -y

# Export reports
eq redteam run -t "llm:gpt-5-mini" \
  -c LLM07 --max-dynamic-datapoints 5 \
  --save-report ./report.json --export-md ./reports --export-html ./reports -y
```

### ORQ platform examples

```bash
# Dynamic run against an ORQ agent (backend auto-detected)
eq redteam run -t "agent:my-agent-key" \
  -c LLM01 -c ASI01 --max-turns 3 --max-dynamic-datapoints 5 -y

# Hybrid mode (dynamic + static dataset)
eq redteam run -t "agent:my-agent-key" \
  --mode hybrid --max-turns 3 -y
```

### Key flags

| Flag                        | Description                                         |
|-----------------------------|-----------------------------------------------------|
| `-t` / `--target`           | Target (repeatable for multi-target)                |
| `--mode`                    | `dynamic`, `static`, or `hybrid`                    |
| `--backend`                 | `openai` (default) or `orq` (auto-detected for agents) |
| `-c` / `--category`         | OWASP category filter (LLM01-10, ASI01-10)          |
| `-V` / `--vulnerability`    | Vulnerability ID filter (e.g. `prompt_injection`)   |
| `--system-prompt`           | System message for `llm:` targets                   |
| `--attacker-instructions`   | Domain context to steer attack generation           |
| `--max-turns`               | Max conversation turns per attack                   |
| `--max-dynamic-datapoints`  | Cap on generated attack datapoints                  |
| `--no-generate-strategies`  | Skip LLM strategy generation (faster)               |
| `--attack-model`            | Model for adversarial prompt generation              |
| `--evaluator-model`         | Model for evaluation scoring                         |
| `--parallelism`             | Concurrent jobs                                     |
| `--output-dir`              | Save intermediate artifacts for debugging           |
| `-n` / `--name`             | Experiment name (default: `red-team`)               |
| `-y`                        | Skip confirmation prompt                            |
| `-v` / `-vv`                | Info / debug verbosity                              |

See `eq redteam run --help` for the full list.

### List previous runs

```bash
eq redteam runs
```

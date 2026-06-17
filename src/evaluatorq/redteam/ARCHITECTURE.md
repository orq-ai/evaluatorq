# Red Teaming — Architecture & Internals

Contributor-facing internals for the red teaming subpackage. For usage (running `red_team()`, modes, parameters, targets, CLI) see [README.md](README.md).

## Architecture

```
redteam/
├── contracts.py              # Shared types, enums, Pydantic schemas
├── vulnerability_registry.py # Single source of truth: vulnerability → framework mappings
├── runner.py                 # Unified red_team() entry point
├── cli.py                    # Typer CLI (eq redteam run / runs)
├── hooks.py                  # Pipeline lifecycle hooks (Default, Rich)
├── tracing.py                # OpenTelemetry span helpers
├── exceptions.py             # Custom exceptions
├── adaptive/                 # Pipeline brain
│   ├── agent_context.py      #   Retrieve agent config from ORQ API
│   ├── capability_classifier.py  #   LLM-based tool capability tagging
│   ├── strategy_planner.py   #   Select + generate attack strategies
│   ├── strategy_registry.py  #   Strategy lookup by vulnerability/category
│   ├── attack_generator.py   #   Fill templates + adapt prompts to context
│   ├── objective_generator.py #  Generate attack objectives
│   ├── orchestrator.py       #   Multi-turn adversarial attack loop
│   ├── evaluator.py          #   Vulnerability scoring
│   └── pipeline.py           #   evaluatorq DataPoint/Job/Evaluator wiring
├── backends/                 # Pluggable target backends
│   ├── base.py               #   Backend ABC + create_target() (AgentTarget lives in contracts)
│   ├── orq.py                #   ORQ platform agent target (default)
│   ├── openai.py             #   Direct OpenAI model target
│   ├── openresponses.py      #   OpenResponses (/responses) agent target
│   ├── registry.py           #   Backend resolution + LLM client factory
│   └── _errors.py            #   Status/provider error-code extraction helpers
├── frameworks/               # Framework-specific strategy libraries
│   ├── owasp_asi.py          #   ASI01–ASI10 attack strategies
│   ├── owasp_llm.py          #   LLM01–LLM09 attack strategies
│   └── owasp/                #   Evaluator prompts and bridge
├── reports/                  # Report generation and export
│   ├── converters.py         #   Result → RedTeamReport conversion
│   ├── display.py            #   Rich terminal summary tables
│   ├── export_html.py        #   HTML dashboard export
│   ├── export_md.py          #   Markdown report export
│   ├── recommendations.py    #   LLM-generated remediation advice
│   └── sections.py           #   Report section builders
├── runtime/                  # Job execution
│   └── jobs.py               #   Async job runner + job-name sanitizer
└── ui/                       # Streamlit interactive dashboard
    └── dashboard.py
```

## Key design decisions

- **Vulnerability is the atomic primitive** — strategies, evaluators, and datapoints bind to `Vulnerability` enum values. Framework categories are a derived mapping layer.
- **Agent-aware** — targets ORQ agents with tools, memory, and knowledge bases. The capability classifier ensures strategies only run when the agent has the required capabilities.
- **Adaptive attacks** — the adversarial LLM observes defenses and adjusts strategy mid-conversation.
- **Pluggable backends** — ORQ backend for production agents, OpenAI for raw models, or bring your own `AsyncOpenAI`-compatible client.
- **Parallel-safe** — each attack gets a unique memory entity ID, preventing cross-contamination. Post-run cleanup removes test data.

## OpenResponses backend (RES-540)

Target agents and deployments through the platform's `/responses` API in the
OpenResponses request shape:

```python
from evaluatorq.redteam import red_team
from evaluatorq.redteam.backends.registry import resolve_backend

bundle = resolve_backend("openresponses")
target = bundle.create_target("my-agent-id")

report = await red_team(target, vulnerabilities=["prompt_injection"])
```

Every attack the orchestrator runs is sent over the wire as:

```json
{"model": "my-agent-id",
 "input": [{"role": "user", "content": "<adversarial prompt>"}]}
```

The backend reuses `evaluatorq.openresponses.target.OrqResponsesTarget`, so
redteam and simulation share retry behavior, `previous_response_id` threading,
output parsing, and token-usage extraction.

Trace spans use `gen_ai.*` attributes plus `orq.openresponses.request` /
`orq.openresponses.response` so observability surfaces the exact payload that
went over the wire.

### Dataset helpers

For consumers that need to build OpenResponses payloads or load redteam static
datasets authored in OpenResponses input shape:

```python
from evaluatorq.openresponses import (
    build_openresponses_request,
    load_openresponses_dataset,
    turns_to_openresponses_input,
    redteam_sample_from_openresponses,
)
```

- `build_openresponses_request(model=..., prompt=..., conversation=...)` — assemble the wire payload.
- `load_openresponses_dataset(path)` — load a JSON / JSONL static redteam dataset authored as OpenResponses input arrays.
- `turns_to_openresponses_input(orchestrator_result.turns)` — convert an executed attack back into the OpenResponses input array (useful for replay, dataset capture, debugging).
- `redteam_sample_from_openresponses(input=..., openresponses_input=...)` — load datasets authored in OpenResponses format into the existing `RedTeamSample` schema.

### Registry shortcut

The backend is also registered for resolution via the standard registry, so it
slots into the same machinery as the ORQ / OpenAI backends:

```python
from evaluatorq.redteam.backends.registry import resolve_backend
bundle = resolve_backend("openresponses")
```

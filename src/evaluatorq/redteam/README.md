# Red Teaming

Adaptive red teaming for AI agents and LLMs — automatically discover security vulnerabilities using multi-turn adversarial attacks mapped to industry security frameworks.

## What it does

This subpackage probes AI agents and LLMs for security weaknesses using a multi-stage pipeline:

1. **Context retrieval** — Fetches the agent's configuration (tools, memory stores, knowledge bases, system prompt) from the ORQ platform, or accepts a system prompt for direct LLM targets.
2. **Capability classification** — An LLM semantically tags each tool (e.g. `file_system`, `code_execution`, `email`, `payment`) so attacks target real capabilities.
3. **Strategy planning** — Selects from 35+ hardcoded attack strategies filtered by agent capabilities, plus generates novel strategies tailored to the specific target via LLM.
4. **Multi-turn orchestration** — An adversarial LLM runs an adaptive attack loop: it observes the target's responses each turn and adjusts its approach.
5. **Evaluation** — Each attack result is scored against the relevant vulnerability evaluator to determine resistance or vulnerability.

## Vulnerability-first design

The atomic primitive is the **vulnerability**, not the framework category. Each vulnerability has a stable, framework-agnostic ID (e.g. `prompt_injection`, `goal_hijacking`) that maps to one or more framework categories. You can target attacks at either level:

```python
# Target specific vulnerabilities
report = await red_team("llm:gpt-5-mini", vulnerabilities=["prompt_injection", "goal_hijacking"])

# Or filter by framework category
report = await red_team("llm:gpt-5-mini", categories=["LLM01", "ASI01"])
```

### Supported vulnerabilities

18 vulnerability types across three domains:

| Domain | Vulnerability | Description |
|--------|--------------|-------------|
| **Agent** | `goal_hijacking` | Redirect the agent away from its intended objective |
| **Agent** | `tool_misuse` | Trick the agent into misusing its tools |
| **Agent** | `identity_privilege_abuse` | Exploit identity or permissions |
| **Agent** | `supply_chain` | Compromise via third-party dependencies |
| **Agent** | `code_execution` | Trigger unauthorized code execution |
| **Agent** | `memory_poisoning` | Inject malicious data into agent memory |
| **Agent** | `inter_agent_comms` | Exploit insecure agent-to-agent communication |
| **Agent** | `cascading_failures` | Trigger failure cascades across agents |
| **Agent** | `trust_exploitation` | Exploit human-agent trust boundaries |
| **Agent** | `rogue_agents` | Cause agents to act outside their boundaries |
| **Agent** | `excessive_agency` | Exploit overly broad agent permissions |
| **Model** | `prompt_injection` | Override instructions via injected prompts |
| **Model** | `sensitive_info_disclosure` | Extract confidential information |
| **Model** | `improper_output` | Produce harmful or unvalidated output |
| **Model** | `system_prompt_leakage` | Leak the system prompt contents |
| **Model** | `misinformation` | Generate false or misleading information |
| **Data** | `data_poisoning` | Corrupt training data or model weights |
| **Data** | `vector_embedding_weakness` | Exploit vector/embedding retrieval |

### Supported frameworks

Vulnerabilities map to industry security frameworks via `framework_mappings`:

| Framework | Categories | Description |
|-----------|-----------|-------------|
| **OWASP ASI** | ASI01–ASI10 | Agentic Security Initiative — agent-layer risks |
| **OWASP LLM Top 10** | LLM01–LLM09 | LLM-layer risks (prompt injection, data leakage, etc.) |

Some vulnerabilities map to multiple frameworks (e.g. `supply_chain` → ASI04 + LLM03).

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
│   ├── orq.py                #   ORQ platform agent target (default)
│   ├── openai.py             #   Direct OpenAI model target
│   └── registry.py           #   Backend resolution + LLM client factory
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
│   ├── jobs.py               #   Async job runner
│   └── orq_agent_job.py      #   ORQ agent job implementation
└── ui/                       # Streamlit interactive dashboard
    └── dashboard.py
```

## Key design decisions

- **Vulnerability is the atomic primitive** — strategies, evaluators, and datapoints bind to `Vulnerability` enum values. Framework categories are a derived mapping layer.
- **Agent-aware** — targets ORQ agents with tools, memory, and knowledge bases. The capability classifier ensures strategies only run when the agent has the required capabilities.
- **Adaptive attacks** — the adversarial LLM observes defenses and adjusts strategy mid-conversation.
- **Pluggable backends** — ORQ backend for production agents, OpenAI for raw models, or bring your own `AsyncOpenAI`-compatible client.
- **Parallel-safe** — each attack gets a unique memory entity ID, preventing cross-contamination. Post-run cleanup removes test data.

## Convention

Throughout this package: `passed=True` / `vulnerable=False` means the agent **resisted** the attack (good). `passed=False` / `vulnerable=True` means the agent was **vulnerable** (bad).

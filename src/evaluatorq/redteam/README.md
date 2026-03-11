# evaluatorq.redteam

Adaptive red teaming for AI agents — automatically discover security vulnerabilities by running OWASP-mapped adversarial attacks against your agents.

## What it does

This subpackage probes AI agents for security weaknesses using a multi-stage pipeline:

1. **Context retrieval** — Fetches the agent's actual configuration (tools, memory stores, knowledge bases, system prompt) from the ORQ platform.
2. **Capability classification** — An LLM semantically tags each tool (e.g. `file_system`, `code_execution`, `email`, `payment`) so attacks target real capabilities, not guessed ones.
3. **Strategy planning** — Selects from 22+ hardcoded OWASP attack strategies filtered by what the agent can actually do, plus generates novel attack strategies tailored to the specific agent via LLM.
4. **Multi-turn orchestration** — An adversarial LLM runs an adaptive attack loop: it observes the target's responses each turn and adjusts its approach, checking for objective achievement.
5. **OWASP evaluation** — Each attack result is scored against the relevant OWASP category to determine whether the agent resisted or was vulnerable.

Results map directly to **OWASP ASI** (Agentic Security Initiative) and **OWASP LLM Top 10** frameworks, producing per-category resistance rates and vulnerability summaries.

## Why not just use existing tools?

- **Agent-aware, not just LLM-aware** — Targets ORQ agents with tools, memory, and knowledge bases. Handles tool call flows, memory isolation, and multi-turn state.
- **Adaptive attacks** — The adversarial LLM observes defenses and changes strategy mid-conversation, rather than firing static prompts.
- **Context-driven** — Attack templates are filled with the agent's real tool names, memory stores, and description. The capability classifier ensures strategies only run when the agent has the required capabilities.
- **Parallel-safe** — Each attack job gets a unique memory entity ID, preventing cross-contamination between concurrent runs. Post-run cleanup removes all test data.
- **Pluggable backends** — Default ORQ backend for production agents, or swap to plain OpenAI for testing against raw models.

## Architecture

```
redteam/
├── contracts.py              # Shared types, enums, Pydantic schemas (stdlib + pydantic only)
├── adaptive/                 # The pipeline brain
│   ├── agent_context.py      #   Retrieve agent config from ORQ API
│   ├── capability_classifier.py  #   LLM-based tool capability tagging
│   ├── strategy_planner.py   #   Select + generate attack strategies
│   ├── attack_generator.py   #   Fill templates + adapt prompts to tools
│   ├── orchestrator.py       #   Multi-turn adversarial attack loop
│   ├── evaluator.py          #   OWASP vulnerability scoring
│   └── pipeline.py           #   evaluatorq DataPoint/Job/Evaluator wiring
├── backends/                 # Pluggable target backends
│   ├── orq.py                #   ORQ SDK agent target (default)
│   ├── openai.py             #   Direct OpenAI model target
│   └── registry.py           #   Backend resolution
└── frameworks/               # Hardcoded OWASP strategy libraries
    ├── owasp_asi.py          #   ASI01–ASI10 (agentic security)
    └── owasp_llm.py          #   LLM01–LLM10 (LLM top 10)
```

## OWASP coverage

| Framework | Categories | Examples |
|-----------|-----------|----------|
| **OWASP ASI** | ASI01–ASI10 | Goal Hijacking, Tool Misuse, RCE, Memory Poisoning, Trust Exploitation |
| **OWASP LLM** | LLM01–LLM10 | Prompt Injection, Sensitive Info Disclosure, System Prompt Leakage |

Hardcoded strategies cover ASI01, ASI02, ASI05, ASI06, ASI09, LLM01, LLM02, and LLM07. LLM-generated strategies can target any category.

## Convention

Throughout this package: `passed=True` means the agent **resisted** the attack (good). `passed=False` means **vulnerable** (bad).

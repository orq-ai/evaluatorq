# evaluatorq.redteam — Roadmap

**Last updated:** 2026-03-20

---

## Current State (v1)

The v1 red teaming engine is shipped. It covers:

- **19 vulnerabilities** across OWASP ASI (10) and OWASP LLM Top 10 (9)
- **Dynamic pipeline** — objective generation, capability-aware strategy selection, tool-adapted attack generation, LLM-as-judge evaluation
- **Static mode** — load pre-built attack datasets from HuggingFace or the orq.ai platform
- **Hybrid mode** — combine static datasets with dynamic generation
- **Multi-agent comparison** — run the same attacks against multiple agents, compare results side-by-side with disagreement analysis
- **Reporting** — Rich terminal, Markdown, HTML, JSON auto-save, Streamlit dashboard
- **Backends** — ORQ agents (via platform API) and any OpenAI-compatible model
- **CLI** — `eq redteam run`, `eq redteam runs` for history
- **Observability** — OpenTelemetry tracing, pipeline hooks
- **Security** — XML-escaping of traces, single-pass template substitution, prompt injection prevention

---

## P0 — Must Have

### 1. Responsible AI & Safety Vulnerabilities

Bias, toxicity, and safety vulnerabilities are table-stakes for enterprise red teaming. The HuggingFace dataset ([`orq/redteam-vulnerabilities`](https://huggingface.co/datasets/orq/redteam-vulnerabilities)) already contains 130 samples for bias, toxicity, and harmful content — they need to be wired into the pipeline.

- **Bias detection** — religion, politics, gender, race subtypes with LLM-as-judge evaluators
- **Toxicity detection** — profanity, insults, threats, mockery subtypes
- **Illegal activity** — weapons, drugs, violent crimes, cybercrime, child exploitation
- **Harmful content** — graphic/sexual content, personal safety (bullying, self-harm, dangerous challenges)

### 2. Domain-Specific Risk Vulnerabilities

Risk categories for agents giving inappropriate professional advice. The HF dataset already has 20 samples across these categories.

- **Legal advice risk** — detect agents providing specific legal advice without disclaimers
- **Medical advice risk** — detect agents providing medical diagnoses or treatment recommendations
- **Financial advice risk** — detect agents providing specific investment or financial advice

### 3. Custom Vulnerability API

Let users extend vulnerability coverage without modifying package internals.

- **Runtime registration API** — extensible registry so users can define custom vulnerabilities with no code changes
- **Custom evaluator criteria** — accept plain-text criteria that get wrapped in an LLM-as-judge prompt
- **Custom strategy attachment** — attach custom attack strategies to custom vulnerabilities

---

## P1 — Should Have

### 4. Compliance & Framework Mapping

Map vulnerabilities to industry-recognized frameworks for compliance reporting.

- **MITRE ATLAS mapping** — adversarial threat landscape for AI systems
- **NIST AI RMF mapping** — AI Risk Management Framework
- **Regulatory compliance mapping** — GDPR, EU AI Act, HIPAA, PCI DSS
- **OWASP compliance report** — one-click OWASP LLM Top 10 + ASI Top 10 compliance report as PDF/HTML
- **Pre-configured security profiles** — one-click profiles: "OWASP LLM Top 10", "EU AI Act", "GDPR"

### 5. Attack Method Expansion

High-value attack techniques proven effective and currently missing.

- **Multilingual attacks** — translate attacks to non-English languages; known bypass for English-trained safety filters
- **Encoding attacks** — Base64, ROT-13, Leetspeak deterministic transformations
- **Emotional/semantic manipulation** — social engineering using emotional pressure and semantic tricks
- **Context flooding** — flood context window to push system instructions out of attention
- **BadLikertJudge** — multi-turn attack using evaluative scales to extract harmful content
- **Tree jailbreaking** — branching conversation trees exploring multiple attack paths in parallel
- **Reuse simulated test cases** — skip attack regeneration on re-runs for faster iteration

### 6. Agentic Attack Plugins

Deeper agentic-specific attack coverage.

- **Tool discovery attacks** — probe agents to enumerate available tools and capabilities
- **Tool metadata poisoning** — test schema manipulation and description deception in agent tool definitions
- **Cross-context retrieval** — test tenant/user/role isolation in multi-tenant agent systems

### 7. Reporting & Regression

Make red teaming actionable over time.

- **Interactive report design** — 4-tab Streamlit dashboard
- **Historical comparison** — compare current run vs. previous runs with up to 4 comparison columns
- **Regression detection** — detect regressions and track improvement over time
- **DataFrame export** — `.to_df()` on results for data science workflows

### 8. Documentation

- **Documentation site** — getting started guide, vulnerability reference, CLI reference, custom vulnerabilities, backends, reports, API reference

---

## P2 — Nice to Have

### 9. Expanded PII & Intellectual Property

- **PII leakage subtypes** — extend current sensitive info disclosure with session leak, social manipulation, API/database access
- **Intellectual property** — imitation, copyright violations, trademark infringement

### 10. API & DX Polish

- **Sync API wrapper** — `red_team_sync()` that wraps `asyncio.run()`
- **YAML CLI configuration** — run red team from a YAML config file
- **Attack weighting** — per-attack `weight` parameter controlling selection probability
- **Exploitability ratings** — LOW/MEDIUM/HIGH exploitability metadata per attack method

### 11. Research Dataset Integration

- **Research dataset loader** — generic loader for HuggingFace datasets: BeaverTails, HarmBench, ToxicChat, DoNotAnswer
- **Domain-specific attack templates** — pre-built templates for healthcare, finance, e-commerce
- **CrowS-Pairs bias dataset** — integrate EuConform CrowS-Pairs for bias/discrimination evaluation

### 12. Advanced Security Testing

- **BFLA/BOLA/RBAC testing** — privilege escalation, function bypass, cross-customer access
- **System reconnaissance** — test for file metadata, database schema, and retrieval config leakage

---

## Out of Scope

| Item | Reason |
|------|--------|
| **Runtime guardrails** | Guardrails are a runtime concern, not a testing concern. |
| **RAG-specific plugins** | May revisit based on demand. |
| **CI/CD native integration** | The CLI can be called from any CI pipeline already. |
| **Web UI for results** | Streamlit dashboard + HTML export cover the local use case. |
| **Recursive hijacking / autonomous agent drift** | Low real-world prevalence with current agent architectures. |

---

## Contributing

We welcome contributions! If you're interested in working on any of these items, please open an issue or discussion to coordinate before starting work.

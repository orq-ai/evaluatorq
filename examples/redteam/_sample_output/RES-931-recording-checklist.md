# RES-931 recording checklist (per-framework demo videos)

Follow this to record one short screen video per external framework (Bauke's
request). Each run is ~1-2 minutes: dynamic mode, 3 attacks, up to 2 turns each.
The commands and targets are verified runnable; only the recording is manual.

## One-time setup

```bash
cd packages/evaluatorq-py          # repo root holding pyproject.toml
uv sync --frozen --all-extras      # installs the 4 framework extras
uv pip install -e .                # only if `import evaluatorq` fails from a bare python

# One key drives the agent model, the attacker LLM, and the judge (all route via Orq).
export ORQ_API_KEY=orq-...
```

## Record these four (one video each)

```bash
uv run python examples/redteam/17_langgraph_target.py     # LangGraph   -> LangGraphTarget
uv run python examples/redteam/18_openai_agents_target.py # OpenAI SDK  -> OpenAIAgentTarget
uv run python examples/redteam/19_pydantic_ai_target.py   # Pydantic AI -> PydanticAITarget
uv run python examples/redteam/20_crewai_target.py        # CrewAI      -> CrewAITarget
```

## What to capture in each video

1. The command line being run (shows the framework + wrapper).
2. The live attack progression (Rich hooks: each attack technique + verdict).
3. The final report table: resistance %, vulnerable / attacks.
4. For a vulnerable attack, the judge assessment line (the OWASP evaluator verdict).

Expected shape (matches the committed live output in
`RES-931-external-framework-runs.md`): LangGraph / OpenAI Agents / Pydantic AI
each take one indirect-injection (goal hijack via tool output); CrewAI resists
all three. Exact attacks vary run to run since the attacker LLM is generative.

## After recording

Drop the files in `docs/assets/redteam/` named `res931-<framework>.<ext>`
(e.g. `res931-langgraph.mp4` or `.gif`), then the embed slots in
`docs/guides/red-teaming.md` (marked with `RES-931 video slot` comments) get
uncommented to point at them. Ping me and I will wire them in.

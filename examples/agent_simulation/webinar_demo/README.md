# Demo — Simulating AI Agents

Working dir for the agent-simulation webinar: slides + runnable demo.

Structure mirrors the red-teaming demos (`../../redteam/refund_agent_demo`,
`../../redteam/crypto_stealing_demo`): a Quarto Reveal.js deck plus a runnable
agent the deck demos live. The presentation theme/assets are reused from those
decks (orq brand teal `#025558`, orange `q`).

## Status

**Scaffold.** `RUNBOOK.md` has the full demo flow (steps + exact commands + known
gaps), validated end-to-end. `presentation.qmd` is a bare themed skeleton — fill it
from the runbook once the flow is signed off.

- **`RUNBOOK.md`** — start here. The live-demo script: 3 Acts, real commands, gaps to call out.

## Slides

```bash
quarto render presentation.qmd     # → presentation.html + docs/
# or just open presentation.html in a browser (self-contained, no deps)
```

- `presentation.qmd` — deck source
- `_quarto.yml`, `orq-theme.css`, `styles.scss` — Quarto + brand theme (reused)
- `assets/` — brand media (logo, gradient background); add demo screenshots here

## Demo agent (to build)

The runnable demo goes here — a support agent driven by `evaluatorq.simulation.simulate()`
against a persona × scenario matrix, plus a hardening pass. Start from the examples one
level up (`../01_basic_simulation.py`, `../05_wrap_and_experiment.py`).

Planned layout (mirrors `refund_agent_demo/agent_build/`):

```
agent_build/
  build_agent.py     # idempotent orq setup (agent + tools)
  run_simulation.py  # drives simulate() over the persona/scenario matrix
  personas.py        # persona definitions
  scenarios.py       # scenario + criteria definitions
  tests/
```

## Prereqs

- `ORQ_API_KEY` exported in env
- Python 3.11+, [uv](https://docs.astral.sh/uv/)
- [Quarto](https://quarto.org) (only to re-render slides)

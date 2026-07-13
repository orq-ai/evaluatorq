# Demo — Simulating AI Agents

Working dir for the agent-simulation webinar: slides + runnable demo.

Structure mirrors the red-teaming demos (`../../redteam/refund_agent_demo`,
`../../redteam/crypto_stealing_demo`): a Quarto Reveal.js deck plus a runnable
agent the deck demos live. The presentation theme/assets are reused from those
decks (orq brand teal `#025558`, orange `q`).

## Start here

Everything below needs an Orq API key for the workspace that will host the demo.
Export it first (or copy `agent_build/.env.example` to `agent_build/.env` and fill it in):

```bash
export ORQ_API_KEY="sk-orq-..."     # the banking / CustomerDemo workspace key
```

Then provision the agent and run the flow:

```bash
cd examples/agent_simulation/webinar_demo
make provision      # create the Bank of Holland agent + tools + KB in Orq (idempotent)
make simulate       # run simulations against it
```

## Status

**Scaffold.** `RUNBOOK.md` has the full demo flow (steps + exact commands + known
gaps), validated end-to-end. `presentation.qmd` is a bare themed skeleton — fill it
from the runbook once the flow is signed off.

- **`RUNBOOK.md`** — the live-demo script: 3 Acts, real commands, gaps to call out.

## Slides

```bash
quarto render presentation.qmd     # → presentation.html + docs/
# or just open presentation.html in a browser (self-contained, no deps)
```

- `presentation.qmd` — deck source
- `_quarto.yml`, `orq-theme.css`, `styles.scss` — Quarto + brand theme (reused)
- `assets/` — brand media (logo, gradient background); add demo screenshots here

## Demo agent

The spine is the **Bank of Holland credit-card support
agent** — `azure/gpt-5-mini`, a 99-question Dutch/English FAQ knowledge base, and two
code tools (`get_card_info`, `get_transaction_details`). `make provision` recreates it
(and its tools + KB) in Orq from the definitions in `agent_build/`, using distinct demo
keys so it never clobbers the customer's live entities.

```
agent_build/
  provision.py            # idempotent Orq SDK provisioning (tools + KB + agent)
  config.py               # demo keys, path, embedding model
  assets/boh_faq.txt      # the FAQ ingested into the knowledge base
  orq_export/             # agent + tool + KB definitions exported from the platform
```

Run the demo flow with the Makefile: `make provision`, then `make generate`,
`make simulate`, `make rerun`, `make upload`. See **`RUNBOOK.md`** for the narrated
version with what each step demonstrates.

## Prereqs

- `ORQ_API_KEY` exported in env
- Python 3.11+, [uv](https://docs.astral.sh/uv/)
- [Quarto](https://quarto.org) (only to re-render slides)

# Demo — Red Teaming AI Agents

Runnable demo for the 30-minute talk at AI Builders Amsterdam.

## Slides

Quarto revealjs deck (`presentation.qmd`).

```bash
quarto render presentation.qmd
open docs/presentation.html
```

## Setup

```bash
cd demo
uv sync
cp .env.example .env   # fill in ORQ_API_KEY
```

## Run

Two terminals:

```bash
# T1: webapp
uv run uvicorn webapp.app:app --port 8001
```

```bash
# T2: red team
uv run python run.py
```

Open `http://localhost:8001/` in a browser to see the wallets tick.

## Tests

```bash
uv run pytest
```

## Finale (stage only)

```bash
uv run python finale.py
```

Opens a fake-shutdown overlay in the default browser. No OS action.

## Files

- `agents/` — `DemoAgent` (tool-capable, implements `AgentTarget`), vulnerable + secure subclasses
- `tools.py` — `send_email`, `send_crypto`, `run_shell`
- `webapp/` — FastAPI + SSE + static UI
- `run.py` — drives `red_team()` against both agents
- `compare.py` — renders side-by-side results table
- `finale.py` — one-shot prompt that trips `run_shell` → fake shutdown overlay

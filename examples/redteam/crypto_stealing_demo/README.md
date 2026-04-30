# Demo — Red Teaming AI Agents

Runnable demo for the 30-minute talk at AI Builders Amsterdam.

Two agents. Same tools. Different system prompts. Watch BTC leave Alice's wallet live.

![JARVIS vs HAL — demo setup](assets/slide-jarvis-vs-hal.png)

## The Attack

![Multi-turn attack transcript — JARVIS gets compromised](assets/slide-attack-transcript.png)

## Live Webapp

![Demo wallet — balances update live on transfer](assets/webapp.png)

## Terminal Run

![Capability probe and run plan](assets/cli-run-plan.jpg)

![Terminal output showing the red team pipeline](assets/cli-run-output.png)

## Results

![Side-by-side vulnerability comparison](assets/vulnerability-side-by-side.png)

![Platform results view](assets/platform-results.jpg)

![Dashboard explorer](assets/dashboard-explorer.png)

## Slides

Self-contained deck at `presentation.html` (open directly in browser, no dependencies).

To re-render from source:

```bash
quarto render presentation.qmd
```

## Setup

```bash
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

## Files

- `agents/` — `DemoAgent` (tool-capable, implements `AgentTarget`), vulnerable + secure subclasses
- `tools.py` — `send_email`, `send_crypto`, `run_shell`
- `webapp/` — FastAPI + SSE + static UI
- `run.py` — drives `red_team()` against both agents
- `compare.py` — renders side-by-side results table
- `presentation.html` — self-contained slide deck

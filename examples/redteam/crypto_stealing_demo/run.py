"""Driver script for the live demo.

Before running:
  1. Ensure ORQ_API_KEY + ORQ_BASE_URL are set (see .env.example).
  2. Start the webapp:  uv run uvicorn webapp.app:app --port 8001
  3. Open http://localhost:8001/ in a browser.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time

# Prevent rich from probing terminal width via cursor-position queries (CPR).
# Without this, rich sends \033[6n repeatedly; inside tmux the responses arrive
# after the script exits and appear as raw text in the shell prompt.
try:
    os.environ.setdefault("COLUMNS", str(os.get_terminal_size().columns))
except OSError:
    os.environ.setdefault("COLUMNS", "220")
from pathlib import Path

import httpx
from dotenv import load_dotenv
from evaluatorq.redteam import red_team, LLMCallConfig, LLMConfig

from agents.secure import HAL
from agents.vulnerable import JARVIS
from compare import render_side_by_side
from config import WEBAPP_URL

DEMO_DIR = Path(__file__).parent
RESULTS_DIR = DEMO_DIR / "results"


def _next_run_index() -> int:
    existing = [p.stem for p in RESULTS_DIR.glob("hal_*.json")]
    indices = [int(s.split("_")[-1]) for s in existing if s.split("_")[-1].isdigit()]
    return max(indices, default=0) + 1


def _preflight() -> None:
    env_file = DEMO_DIR / ".env"
    if not env_file.exists() and not os.environ.get("ORQ_API_KEY"):
        print(f"ERROR: {env_file} not found. Copy .env.example to .env and fill it in.", file=sys.stderr)
        sys.exit(2)

    missing = [k for k in ("ORQ_API_KEY",) if not os.environ.get(k)]
    if missing:
        print(f"ERROR: missing required env vars: {', '.join(missing)}", file=sys.stderr)
        print(f"       Check {env_file}", file=sys.stderr)
        sys.exit(2)

    base_url = os.environ.get("ORQ_BASE_URL", "https://my.orq.ai")
    try:
        r = httpx.get(f"{base_url.rstrip('/')}/health", timeout=5.0)
        if r.status_code >= 500:
            print(f"WARN: ORQ health check returned {r.status_code}", file=sys.stderr)
    except Exception as exc:
        print(f"WARN: ORQ not reachable at {base_url} ({exc})", file=sys.stderr)


async def main() -> None:
    load_dotenv(DEMO_DIR / ".env")
    _preflight()
    os.environ.pop("OPENAI_API_KEY", None)  # force ORQ router; OPENAI_API_KEY in shell would shadow it

    try:
        httpx.post(f"{WEBAPP_URL}/reset", timeout=2.0)
    except Exception as exc:
        print(f"WARN: webapp not reachable on {WEBAPP_URL} ({exc}). Continuing.", file=sys.stderr)

    attacker_instructions = (DEMO_DIR / "attacker_instructions.txt").read_text()
    RESULTS_DIR.mkdir(exist_ok=True)
    run_index = _next_run_index()

    start = time.time()
    model = LLMCallConfig(model="openai/gpt-5.4-mini")
    report = await red_team(
        target=[HAL(), JARVIS()],
        vulnerabilities=["prompt_injection", "goal_hijacking"],
        mode="dynamic",
        max_turns=6,
        max_dynamic_datapoints=10,
        max_static_datapoints=0,
        attacker_instructions=attacker_instructions,
        parallelism=20,
        llm_config=LLMConfig(attacker=model, evaluator=model),
        generate_recommendations=True,
        verbosity=0,
        name="AI Builders - Red Teaming Demo",
    )
    print(f"\nCompleted in {time.time() - start:.1f}s")

    dump = report.model_dump()
    for label in ["hal", "jarvis"]:
        filtered = {**dump, "results": [r for r in dump["results"] if r.get("agent", {}).get("key") == label.upper()]}
        out = RESULTS_DIR / f"{label}_{run_index:03d}.json"
        out.write_text(json.dumps(filtered, indent=2, default=str))
        print(f"-> {out}")

    render_side_by_side(
        str(RESULTS_DIR / f"hal_{run_index:03d}.json"),
        str(RESULTS_DIR / f"jarvis_{run_index:03d}.json"),
    )


if __name__ == "__main__":
    asyncio.run(main())

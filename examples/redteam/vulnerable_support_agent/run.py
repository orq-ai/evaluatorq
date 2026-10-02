"""Driver for the vulnerable support agent red-team demo.

Runs one adaptive red team against a DELIBERATELY INSECURE support agent for the
OWASP categories ASI01 (agentic goal hijacking) and LLM01 (prompt injection),
prints clean RichHooks output (good for a screen recording), and writes the
result JSON to results/.

Before running:
  1. Copy .env.example to .env and set ORQ_API_KEY (+ ORQ_BASE_URL if not default).
  2. uv run python run.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time

# Prevent rich from probing terminal width via cursor-position queries (CPR),
# which can leak escape codes into the shell prompt inside tmux.
try:
    os.environ.setdefault("COLUMNS", str(os.get_terminal_size().columns))
except OSError:
    os.environ.setdefault("COLUMNS", "220")

from pathlib import Path

from dotenv import load_dotenv
from evaluatorq.redteam import LLMCallConfig, LLMConfig, RichHooks, red_team

from agents.vulnerable import Ava
from config import MODEL

DEMO_DIR = Path(__file__).parent
RESULTS_DIR = DEMO_DIR / "results"

# ASI01 -> goal_hijacking, LLM01 -> prompt_injection (per evaluatorq's registry).
VULNERABILITIES = ["goal_hijacking", "prompt_injection"]


def _next_run_index() -> int:
    existing = [p.stem for p in RESULTS_DIR.glob("ava_*.json")]
    indices = [int(s.split("_")[-1]) for s in existing if s.split("_")[-1].isdigit()]
    return max(indices, default=0) + 1


def _preflight() -> None:
    env_file = DEMO_DIR / ".env"
    if not env_file.exists() and not os.environ.get("ORQ_API_KEY"):
        print(f"ERROR: {env_file} not found. Copy .env.example to .env and fill it in.", file=sys.stderr)
        sys.exit(2)

    if not os.environ.get("ORQ_API_KEY"):
        print("ERROR: missing required env var: ORQ_API_KEY", file=sys.stderr)
        print(f"       Check {env_file}", file=sys.stderr)
        sys.exit(2)


async def main() -> None:
    load_dotenv(DEMO_DIR / ".env")
    _preflight()
    os.environ.pop("OPENAI_API_KEY", None)  # force ORQ router; a shell OPENAI_API_KEY would shadow it

    attacker_instructions = (DEMO_DIR / "attacker_instructions.txt").read_text()
    RESULTS_DIR.mkdir(exist_ok=True)
    run_index = _next_run_index()

    model = LLMCallConfig(model=MODEL)

    start = time.time()
    report = await red_team(
        target=Ava(),
        vulnerabilities=VULNERABILITIES,
        mode="dynamic",
        max_turns=6,
        max_dynamic_datapoints=6,
        max_static_datapoints=0,
        attacker_instructions=attacker_instructions,
        datapoint_parallelism=6,
        llm_config=LLMConfig(attacker=model, evaluator=model),
        recommendations=True,
        hooks=RichHooks(skip_confirm=True),
        verbosity=0,
        name="Vulnerable Support Agent - Red Team",
    )
    print(f"\nCompleted in {time.time() - start:.1f}s")

    out = RESULTS_DIR / f"ava_{run_index:03d}.json"
    out.write_text(json.dumps(report.model_dump(), indent=2, default=str))
    print(f"-> {out}")


if __name__ == "__main__":
    asyncio.run(main())

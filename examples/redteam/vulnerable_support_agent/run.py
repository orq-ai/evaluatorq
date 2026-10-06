"""Driver for the vulnerable support agent red-team demo.

Runs one adaptive red team against a DELIBERATELY INSECURE support agent (Ava,
the default) for the OWASP categories ASI01 (agentic goal hijacking) and LLM01
(prompt injection), prints clean RichHooks output (good for a screen recording),
and writes the result JSON to results/ava_XXX.json.

Pass ``--target rex`` to run the same attacks against Rex, the hardened sibling
in agents/secure.py, for contrast; results go to results/rex_XXX.json.

Before running:
  1. Copy .env.example to .env and set ORQ_API_KEY (+ ORQ_BASE_URL if not default).
  2. uv run python run.py [--target {ava,rex}]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

DEMO_DIR = Path(__file__).parent
RESULTS_DIR = DEMO_DIR / "results"

# Load .env before importing config/agents: config.py reads DEMO_MODEL and
# DEMO_TARGET_MODEL at import time and agents bind TARGET_MODEL as a class attribute.
load_dotenv(DEMO_DIR / ".env")

# Prevent rich from probing terminal width via cursor-position queries (CPR),
# which can leak escape codes into the shell prompt inside tmux.
try:
    os.environ.setdefault("COLUMNS", str(os.get_terminal_size().columns))
except OSError:
    os.environ.setdefault("COLUMNS", "220")

from evaluatorq.redteam import LLMCallConfig, LLMConfig, RedTeamReport, RichHooks, red_team  # noqa: E402

from agents.secure import Rex  # noqa: E402
from agents.vulnerable import Ava  # noqa: E402
from config import MODEL, TARGET_MODEL  # noqa: E402

TARGETS = {"ava": Ava, "rex": Rex}

# ASI01 -> goal_hijacking, LLM01 -> prompt_injection (per evaluatorq's registry).
VULNERABILITIES = ["goal_hijacking", "prompt_injection"]


def _reserve_result_path(prefix: str) -> Path:
    """Atomically claim the next free results/<prefix>_XXX.json (exclusive create)."""
    existing = [p.stem for p in RESULTS_DIR.glob(f"{prefix}_*.json")]
    indices = [int(s.split("_")[-1]) for s in existing if s.split("_")[-1].isdigit()]
    index = max(indices, default=0) + 1
    while True:
        path = RESULTS_DIR / f"{prefix}_{index:03d}.json"
        try:
            with path.open("x"):
                return path
        except FileExistsError:
            index += 1


def _preflight() -> None:
    env_file = DEMO_DIR / ".env"
    if not env_file.exists() and not os.environ.get("ORQ_API_KEY"):
        print(f"ERROR: {env_file} not found. Copy .env.example to .env and fill it in.", file=sys.stderr)
        sys.exit(2)

    if not os.environ.get("ORQ_API_KEY"):
        print("ERROR: missing required env var: ORQ_API_KEY", file=sys.stderr)
        print(f"       Check {env_file}", file=sys.stderr)
        sys.exit(2)


def _print_summary(report: RedTeamReport) -> None:
    summary = report.summary
    print(f"\nTarget model: {TARGET_MODEL}  |  attacker/judge model: {MODEL}")
    if summary.no_verdict:
        print(
            "WARNING: no attack produced a verdict, so the target was never tested. "
            "Check the attacker/judge model and ORQ credentials before trusting this run.",
            file=sys.stderr,
        )
    for key, vuln in summary.by_vulnerability.items():
        print(f"  {key}: {vuln.vulnerabilities_found} vulnerable / {vuln.total_attacks} attacks")


async def main(target: str) -> None:
    _preflight()

    attacker_instructions = (DEMO_DIR / "attacker_instructions.txt").read_text()

    model = LLMCallConfig(model=MODEL)

    start = time.time()
    report = await red_team(
        target=TARGETS[target](),
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
    _print_summary(report)

    RESULTS_DIR.mkdir(exist_ok=True)
    out = _reserve_result_path(target)
    out.write_text(json.dumps(report.model_dump(), indent=2, default=str))
    print(f"-> {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--target",
        choices=sorted(TARGETS),
        default="ava",
        help="agent to attack: ava (vulnerable, default) or rex (hardened)",
    )
    asyncio.run(main(parser.parse_args().target))

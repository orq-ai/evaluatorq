"""Run an agent simulation and export the HTML + Markdown reports.

Usage:
    cd packages/evaluatorq-py
    uv run python scripts/manual_tests/simulation_report_demo.py

Uses a local target_callback (a customer-support agent driven by the ORQ
router) so no deployed agent is required. Writes report.html / report.md
next to this script.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

MODEL = "azure/gpt-4o-mini"
OUT_DIR = Path(__file__).resolve().parent


async def main() -> None:
    from evaluatorq.simulation import (
        CommunicationStyle,
        Criterion,
        Persona,
        Scenario,
        simulate,
    )
    from evaluatorq.simulation.reports import export_html, export_markdown
    from evaluatorq.contracts import Message

    api_key = os.environ["ORQ_API_KEY"]
    base_url = f"{os.environ.get('ORQ_BASE_URL', 'https://api.orq.ai')}/v3/router"
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    system = (
        "You are a friendly customer-support agent for 'Acme Cloud', a SaaS "
        "company. Be concise, helpful, and never rude. If you cannot help, "
        "explain next steps."
    )

    async def target_callback(messages: list[Message]) -> str:
        chat = [{"role": "system", "content": system}]
        for m in messages:
            chat.append({"role": m.role, "content": m.content or ""})
        resp = await client.chat.completions.create(
            model=MODEL, messages=chat, temperature=0.7
        )
        return resp.choices[0].message.content or ""

    personas = [
        Persona(
            name="Frustrated Customer",
            patience=0.2,
            assertiveness=0.8,
            politeness=0.4,
            technical_level=0.3,
            communication_style=CommunicationStyle.casual,
            background="Has been waiting days for a fix and is annoyed.",
        ),
        Persona(
            name="Polite Power User",
            patience=0.8,
            assertiveness=0.5,
            politeness=0.9,
            technical_level=0.9,
            communication_style=CommunicationStyle.formal,
            background="An experienced engineer who knows the product well.",
        ),
    ]

    scenarios = [
        Scenario(
            name="Billing question",
            goal="Find out why the latest invoice is higher than expected",
            context="The customer saw an unexpected charge on their bill.",
            criteria=[
                Criterion(description="Agent explains the charge clearly", type="must_happen"),
                Criterion(description="Agent is rude or dismissive", type="must_not_happen"),
            ],
        ),
        Scenario(
            name="API outage",
            goal="Get a workaround for a failing API endpoint",
            context="The customer's production integration is returning 500s.",
            criteria=[
                Criterion(description="Agent offers a concrete next step", type="must_happen"),
                Criterion(description="Agent dismisses the problem", type="must_not_happen"),
            ],
        ),
    ]

    print("Running simulation (2 personas x 2 scenarios)...")
    results = await simulate(
        evaluation_name="report-demo",
        target_callback=target_callback,
        personas=personas,
        scenarios=scenarios,
        max_turns=5,
        model=MODEL,
        evaluator_names=[
            "goal_achieved",
            "criteria_met",
            "turn_efficiency",
            "conversation_quality",
        ],
    )
    await client.close()

    print(f"Got {len(results)} results. Exporting reports...")
    html = export_html(results, target="Acme Cloud support agent")
    md = export_markdown(results, target="Acme Cloud support agent")

    (OUT_DIR / "report.html").write_text(html, encoding="utf-8")
    (OUT_DIR / "report.md").write_text(md, encoding="utf-8")
    print(f"Wrote {OUT_DIR / 'report.html'}")
    print(f"Wrote {OUT_DIR / 'report.md'}")


if __name__ == "__main__":
    asyncio.run(main())

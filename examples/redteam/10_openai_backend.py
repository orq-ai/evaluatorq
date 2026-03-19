"""Red team an ORQ platform agent.

When your application is deployed as an ORQ agent, the pipeline auto-discovers
its system prompt, tools, and memory stores, then generates attacks tailored
to its capabilities — including tool-misuse and memory-poisoning vectors that
aren't available with plain LLM targets.

Prerequisites:
    - ORQ_API_KEY set in environment
    - An agent deployed on https://my.orq.ai
    - Replace "YOUR_AGENT_KEY" below with your agent's key

Usage:
    ORQ_API_KEY=orq-... python 10_orq_agent.py
"""

import asyncio

from evaluatorq.redteam import red_team


async def main() -> None:
    # Replace with your agent key from the ORQ platform settings page.
    report = await red_team(
        "agent:YOUR_AGENT_KEY",
        backend="orq",
        mode="dynamic",
        categories=["LLM01", "LLM07", "ASI01", "ASI02"],
        max_dynamic_datapoints=5,
        max_turns=3,
        generate_strategies=False,
    )

    # The report includes auto-discovered agent context
    ctx = report.agent_context
    if ctx:
        tools = [t.name for t in ctx.tools] if ctx.tools else []
        memory = [m.key or m.id for m in ctx.memory_stores] if ctx.memory_stores else []
        print(f"Agent tools:  {', '.join(tools) or 'none'}")
        print(f"Agent memory: {', '.join(memory) or 'none'}")

    print(f"Resistance rate: {report.summary.resistance_rate:.0%}")
    print(f"Vulnerabilities: {report.summary.vulnerabilities_found}/{report.summary.total_attacks}")

    for result in report.results:
        if result.vulnerable:
            print(f"  VULNERABLE [{result.attack.category}]: {result.attack.vulnerability}")


if __name__ == "__main__":
    asyncio.run(main())

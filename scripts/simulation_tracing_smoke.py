"""Live tracing smoke test for the simulation module.

Runs a tiny 1-datapoint simulation against the Orq router with OTel tracing
enabled, then prints the trace ID(s) so you can look them up in the dashboard.

Requires ORQ_API_KEY in the environment (loaded from repo-root .env).

Usage:
    cd packages/evaluatorq-py
    uv run python scripts/simulation_tracing_smoke.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Any, Sequence

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env"))


# A tap processor that records every span we emit so we can print a tree
# locally and surface the trace ID(s).
class _TapProcessor:
    def __init__(self) -> None:
        self.spans: list[Any] = []

    def on_start(self, span: Any, parent_context: Any | None = None) -> None:
        return None

    def on_end(self, span: Any) -> None:
        self.spans.append(span)

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True


def _print_tree(spans: Sequence[Any]) -> None:
    by_id: dict[int, Any] = {s.context.span_id: s for s in spans}
    children: dict[int | None, list[Any]] = {}
    for s in spans:
        parent_id = s.parent.span_id if s.parent else None
        children.setdefault(parent_id, []).append(s)

    def _print(node: Any, depth: int) -> None:
        attrs = dict(node.attributes or {})
        hint_keys = (
            "orq.simulation.turn",
            "orq.simulation.terminated_by",
            "gen_ai.operation.name",
            "orq.simulation.llm_purpose",
            "gen_ai.usage.total_tokens",
        )
        hints = " ".join(
            f"{k.split('.')[-1]}={attrs[k]}" for k in hint_keys if k in attrs
        )
        print(f"{'  ' * depth}- {node.name}" + (f"  [{hints}]" if hints else ""))
        for child in children.get(node.context.span_id, []):
            _print(child, depth + 1)

    for root in children.get(None, []):
        _print(root, 0)


async def main() -> None:
    api_key = os.environ.get("ORQ_API_KEY")
    if not api_key:
        print("ERROR: ORQ_API_KEY not set")
        sys.exit(1)

    from evaluatorq.simulation import (
        CommunicationStyle,
        Persona,
        Scenario,
        simulate,
    )
    from evaluatorq.tracing.setup import init_tracing_if_needed

    print("=== Simulation tracing smoke test ===\n")

    # Initialize OTel tracing first so we can attach our local tap processor
    # alongside the OTLP exporter that init creates.
    initialized = await init_tracing_if_needed()
    print(f"Tracing initialized: {initialized}")

    tap = _TapProcessor()
    if initialized:
        from opentelemetry import trace

        provider = trace.get_tracer_provider()
        # TracerProvider.add_span_processor exists on the SDK provider
        if hasattr(provider, "add_span_processor"):
            provider.add_span_processor(tap)  # type: ignore[attr-defined]
            print("Local tap processor attached.\n")
        else:
            print("Provider has no add_span_processor; skipping local tap.\n")

    persona = Persona(
        name="Curious shopper",
        patience=0.6,
        assertiveness=0.5,
        politeness=0.7,
        technical_level=0.4,
        communication_style=CommunicationStyle.casual,
        background="Browsing for a new gadget",
    )

    scenario = Scenario(
        name="Product inquiry",
        goal="Ask one question about available products and get a useful answer",
        context="Quick chat about what the company offers",
    )

    print("Running simulate(): 1 persona x 1 scenario, max_turns=2 ...\n")
    results = await simulate(
        evaluation_name="tracing-smoke",
        agent_key="deployment_chat_reply",
        personas=[persona],
        scenarios=[scenario],
        max_turns=2,
        model="azure/gpt-4o-mini",
        evaluator_names=["goal_achieved"],
    )

    r = results[0]
    print(f"Result: terminated_by={r.terminated_by.value} turns={r.turn_count} "
          f"tokens={r.token_usage.total_tokens}")

    # Print the local span tree
    print("\n=== Local span tree (from tap) ===")
    _print_tree(tap.spans)
    print(f"=== {len(tap.spans)} spans captured locally ===\n")

    # Surface trace IDs (hex) so they can be searched in the dashboard
    trace_ids = sorted({format(s.context.trace_id, "032x") for s in tap.spans})
    print("Trace ID(s):")
    for tid in trace_ids:
        print(f"  {tid}")
    print(
        "\nLook these up in the Orq dashboard (Traces tab) or via the API "
        "/v2/traces?trace_id=<id>."
    )


if __name__ == "__main__":
    asyncio.run(main())

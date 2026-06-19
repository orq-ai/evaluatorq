#!/usr/bin/env python3
"""Run agent simulation via the SDK against a hosted Orq agent.

The target agent runs on Orq. For the refund demo agent, function tool calls
are handled locally using the same wrapper as the red-team demo. By default,
generation and judge/simulator LLM calls use the Orq AI Router with
openai/gpt-5.4-mini. Pass ``--sim-provider tailscale`` to switch those calls
back to the Tailscale OpenAI-compatible endpoint.

Usage:
    cd packages/evaluatorq-py
    export ORQ_API_KEY=...
    uv run python examples/agent_simulation/orq_agent_tailscale_openai.py \
        --agent refund-agent-fixed
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger as loguru_logger
from openai import AsyncOpenAI

from evaluatorq.common.llm_client import ORQ_DEFAULT_HOST, ORQ_ROUTER_SUFFIX
from evaluatorq.contracts import AgentContext, AgentResponse, AgentTarget, LLMCallConfig
from evaluatorq.openresponses.target import OrqResponsesTarget
from evaluatorq.redteam.backends.registry import resolve_backend
from evaluatorq.simulation import auto_save_run, build_simulation_run, generate_and_simulate
from evaluatorq.simulation.agents.user_simulator import UserSimulatorAgent
from evaluatorq.simulation.types import DEFAULT_EVALUATOR_NAMES, Message  # noqa: TC001

load_dotenv()

DEFAULT_ORQ_SIM_MODEL = "openai/gpt-5.4-mini"
DEFAULT_TAILSCALE_OPENAI_BASE_URL = "http://orq-research-workstation.siberian-pompano.ts.net:8613/v1"
REFUND_DEMO_ROOT = Path(__file__).parents[1] / "redteam" / "refund_agent_demo"


def _configure_logging(verbosity: int) -> None:
    # Two channels: stdlib (the lib's INFO diagnostics) and loguru (per-turn
    # hooks). Default keeps the chatty per-turn hooks at WARNING so the results
    # table and the ui-load hint this script prints at the end aren't buried.
    if verbosity < 0:
        stdlib_level, loguru_level = "ERROR", "ERROR"
    elif verbosity == 0:
        stdlib_level, loguru_level = "INFO", "WARNING"
    elif verbosity == 1:
        stdlib_level, loguru_level = "INFO", "INFO"
    else:
        stdlib_level, loguru_level = "DEBUG", "DEBUG"

    eq_logger = logging.getLogger("evaluatorq")
    eq_logger.setLevel(stdlib_level)
    if not eq_logger.handlers:  # attach once so INFO records actually emit (lastResort is WARNING)
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(message)s"))
        eq_logger.addHandler(handler)
    loguru_logger.remove()
    loguru_logger.add(sys.stderr, level=loguru_level)


async def _resolve_agent_description(agent_key: str) -> str:
    backend = resolve_backend("orq")
    ctx = await backend.resolve_context(agent_key)
    if not ctx.description:
        raise RuntimeError(
            f"Agent {agent_key!r} has no description; pass --description explicitly."
        )
    return ctx.description


async def _resolve_description(target: AgentTarget, agent_key: str) -> str:
    """Prefer the target's own context description — for a locally-wrapped agent
    that's the local source of truth. Fall back to the remote Orq agent's
    registered description only when the target exposes none."""
    ctx = await target.get_agent_context()
    if ctx.description:
        return ctx.description
    return await _resolve_agent_description(agent_key)


def _load_refund_agent_target_class():
    if str(REFUND_DEMO_ROOT) not in sys.path:
        sys.path.insert(0, str(REFUND_DEMO_ROOT))
    from agent_build.refund_target import RefundAgentTarget

    return RefundAgentTarget


class LocalRefundToolTarget(AgentTarget):
    """Sim AgentTarget adapter for the refund demo's local tool-call loop."""

    def __init__(self, agent_key: str) -> None:
        super().__init__()
        self.agent_key = agent_key
        self.name = agent_key
        self._refund_target_cls = _load_refund_agent_target_class()
        self._inners: dict[int, object] = {}

    def new(self) -> AgentTarget:
        return type(self)(self.agent_key)

    async def get_agent_context(self) -> AgentContext:
        # Local source of truth for the description that drives sim generation.
        # Never round-trip through the remote agent's registered description —
        # it drifts from the local prompts, and that mismatch produced
        # off-domain (webinar) scenarios. Same constant build_agent.py writes
        # to the remote agent.
        from agent_build.prompts import AGENT_DESCRIPTION

        return AgentContext(key=self.agent_key, description=AGENT_DESCRIPTION)

    async def respond(self, messages: list[Message]) -> AgentResponse:
        latest_user = next((m for m in reversed(messages) if m.role == "user"), None)
        if latest_user is None:
            raise ValueError("LocalRefundToolTarget requires at least one user message")
        conversation_id = id(messages)
        inner = self._inners.get(conversation_id)
        if inner is None:
            inner = self._refund_target_cls(agent_key=self.agent_key)
            self._inners[conversation_id] = inner
        response = await inner.send_prompt(latest_user.content or "")
        if isinstance(response, AgentResponse):
            return response
        return AgentResponse.model_validate(response)


def _build_target(agent_key: str) -> AgentTarget:
    normalized = agent_key.removeprefix("agent/")
    if normalized.startswith("refund-agent-"):
        return LocalRefundToolTarget(normalized)
    return OrqResponsesTarget(
        LLMCallConfig(
            model=f"agent/{normalized}",
            api="responses",
            timeout_ms=240_000,
        ),
        require_orq=True,
    )


async def _resolve_sim_model(client: AsyncOpenAI, explicit_model: str | None) -> str:
    if explicit_model:
        return explicit_model

    try:
        models = await client.models.list()
    except Exception:
        # Some single-model OpenAI-compatible servers ignore the model field but
        # do not implement /models. Keep a non-empty placeholder for the SDK.
        return "default"

    model_ids = [model.id for model in models.data if getattr(model, "id", None)]
    if not model_ids:
        return "default"
    if len(model_ids) > 1:
        print(f"Multiple models reported; using first: {model_ids[0]}")
    return model_ids[0]


def _build_orq_router_client() -> AsyncOpenAI:
    api_key = os.getenv("ORQ_API_KEY")
    if not api_key:
        raise SystemExit("ORQ_API_KEY is not set")
    host = os.getenv("ORQ_BASE_URL", ORQ_DEFAULT_HOST).rstrip("/")
    return AsyncOpenAI(api_key=api_key, base_url=f"{host}{ORQ_ROUTER_SUFFIX}")


async def _resolve_sim_client_and_model(args: argparse.Namespace) -> tuple[AsyncOpenAI, str]:
    if args.sim_provider == "orq":
        return _build_orq_router_client(), args.sim_model or DEFAULT_ORQ_SIM_MODEL

    client = AsyncOpenAI(
        api_key=os.getenv("TAILSCALE_OPENAI_API_KEY", "local"),
        base_url=args.tailscale_openai_base_url,
    )
    model = args.sim_model or os.getenv("TAILSCALE_OPENAI_MODEL")
    return client, await _resolve_sim_model(client, model)


def _build_responses_user_simulator(
    *,
    provider: str,
    model: str,
    client: AsyncOpenAI,
) -> UserSimulatorAgent | None:
    if provider != "orq":
        return None
    config = LLMCallConfig(model=model, api="responses", client=client)
    return UserSimulatorAgent(config)


def _print_results_summary(results: list) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    passed = sum(r.goal_achieved for r in results)
    total = len(results)
    scores = [r.goal_completion_score for r in results]
    avg_score = sum(scores) / total if total else 0.0

    table = Table(title=f"Simulation Results — {passed}/{total} goal achieved", title_style="bold")
    table.add_column("#", justify="right", style="dim")
    table.add_column("Result")
    table.add_column("Score", justify="right")
    table.add_column("Turns", justify="right")
    table.add_column("Terminated by")
    for i, result in enumerate(results, 1):
        ok = result.goal_achieved
        table.add_row(
            str(i),
            "[green]PASS[/green]" if ok else "[red]FAIL[/red]",
            f"[{'green' if ok else 'red'}]{result.goal_completion_score:.2f}[/]",
            str(result.turn_count),
            str(result.terminated_by),
        )
    console.print(table)
    console.print(
        f"Pass rate: [bold]{passed / total:.0%}[/bold]  ·  avg score: [bold]{avg_score:.2f}[/bold]"
        if total
        else "No results."
    )


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run evaluatorq sim against an Orq agent"
    )
    parser.add_argument("--agent", default="refund-agent-fixed", help="Orq agent key")
    parser.add_argument(
        "--description",
        default=None,
        help="Agent description. Defaults to the description fetched from Orq.",
    )
    parser.add_argument(
        "--sim-model",
        default=None,
        help=(
            "Simulation model for generation, user simulator, and judge. "
            "Defaults to openai/gpt-5.4-mini for --sim-provider orq; for "
            "tailscale, defaults to TAILSCALE_OPENAI_MODEL, then the first "
            "/models entry, then 'default'."
        ),
    )
    parser.add_argument(
        "--sim-provider",
        choices=("orq", "tailscale"),
        default=os.getenv("SIM_PROVIDER", "orq"),
        help="Provider for generation, user simulator, and judge. Defaults to orq.",
    )
    parser.add_argument(
        "--tailscale-openai-base-url",
        default=os.getenv("TAILSCALE_OPENAI_BASE_URL", DEFAULT_TAILSCALE_OPENAI_BASE_URL),
        help=(
            "OpenAI-compatible base URL for generation, user simulator, and judge. "
            "Defaults to TAILSCALE_OPENAI_BASE_URL or the Orq research workstation MagicDNS name."
        ),
    )
    parser.add_argument(
        "--num-datapoints",
        type=int,
        default=None,
        help="Exact number of datapoints to generate. Overrides --num-personas/--num-scenarios.",
    )
    parser.add_argument("--num-personas", type=int, default=1)
    parser.add_argument("--num-scenarios", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--parallelism", type=int, default=1)
    parser.add_argument(
        "--upload-results",
        action="store_true",
        help="Upload the resulting evaluatorq experiment to Orq.",
    )
    parser.add_argument(
        "--run-output",
        default=None,
        help="Explicit path for the run JSON. Default: auto-saved to .evaluatorq/sim-runs/.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip saving the run JSON entirely.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="Increase verbosity (-v info logs, -vv debug logs).",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress non-error output.",
    )
    args = parser.parse_args()
    verbosity = -1 if args.quiet else args.verbose
    _configure_logging(verbosity)

    if not os.getenv("ORQ_API_KEY"):
        raise SystemExit("ORQ_API_KEY is not set")
    if args.num_datapoints is not None and args.num_datapoints < 1:
        raise SystemExit("--num-datapoints must be >= 1")

    num_personas = args.num_datapoints or args.num_personas
    num_scenarios = 1 if args.num_datapoints is not None else args.num_scenarios

    generation_client, sim_model = await _resolve_sim_client_and_model(args)
    user_simulator = _build_responses_user_simulator(
        provider=args.sim_provider,
        model=sim_model,
        client=generation_client,
    )
    target = _build_target(args.agent)

    description = args.description or await _resolve_description(target, args.agent)

    try:
        results = await generate_and_simulate(
            evaluation_name=f"sim:{args.agent}:tailscale-openai",
            agent_description=description,
            target=target,
            num_personas=num_personas,
            num_scenarios=num_scenarios,
            sim_model=sim_model,
            generation_client=generation_client,
            max_turns=args.max_turns,
            parallelism=args.parallelism,
            evaluator_names=DEFAULT_EVALUATOR_NAMES,
            user_simulator=user_simulator,
            upload_results=args.upload_results,
        )
    finally:
        close_target = getattr(target, "close", None)
        if close_target is not None:
            await close_target()
        await generation_client.close()

    _print_results_summary(results)

    # Save after the summary so the "load in UI" hint is the last line printed.
    # (generate_and_simulate's save= would log it mid-run, before the table.)
    if not args.no_save:
        run = build_simulation_run(
            run_name=f"sim:{args.agent}:tailscale-openai",
            mode="simulate",
            target_kind="callback",
            evaluator_names=DEFAULT_EVALUATOR_NAMES,
            results=results,
        )
        if args.run_output:
            saved = Path(args.run_output)
            saved.parent.mkdir(parents=True, exist_ok=True)
            saved.write_text(run.model_dump_json(indent=2), encoding="utf-8")
        else:
            saved = auto_save_run(run=run, run_name=run.run_name)
        print(f'\nLoad in dashboard:  evaluatorq sim ui "{saved}"')


if __name__ == "__main__":
    asyncio.run(main())

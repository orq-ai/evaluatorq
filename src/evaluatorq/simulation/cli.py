"""CLI for evaluatorq agent simulation.

Three execution verbs:
    generate  — datapoints only (no simulation)
    simulate  — simulate pre-built datapoints
    run       — generate then simulate in one shot

``run`` is a convenience: it generates datapoints in-process and simulates them
under a single pipeline span (via ``generate_and_simulate``). It is not a literal
``generate`` + ``simulate`` pipe — it does not require an intermediate file. To
capture the exact generated inputs for reproducible re-runs, pass
``--save-datapoints PATH`` (then re-feed that file to ``sim simulate --datapoints``).

Usage:
    evaluatorq sim generate --agent-description "..." --output dp.jsonl
    evaluatorq sim simulate --datapoints dp.jsonl --agent-key my-agent
    evaluatorq sim run --agent-description "..." --openai-model gpt-4o-mini
    evaluatorq sim run --agent-description "..." --agent-key my-agent --save-datapoints dp.jsonl
    evaluatorq sim export --input results.jsonl --output payload.json
    evaluatorq sim validate-dataset dp.jsonl
    evaluatorq sim runs
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

import typer

from evaluatorq.simulation.types import DEFAULT_EVALUATOR_NAMES, DEFAULT_MODEL

app = typer.Typer(
    name="sim",
    help="Agent simulation commands.",
    no_args_is_help=True,
)

_SIM_RUNS_DIR_NAME = Path(".evaluatorq") / "sim-runs"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _configure_logging(*, verbose: bool, quiet: bool) -> None:
    if verbose:
        level = logging.DEBUG
    elif quiet:
        level = logging.WARNING
    else:
        level = logging.INFO

    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
        stream=sys.stderr,
    )
    try:
        from loguru import logger as loguru_logger

        loguru_logger.remove()
        loguru_logger.add(sys.stderr, level=logging.getLevelName(level))
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Target resolution
# ---------------------------------------------------------------------------


def _resolve_target(
    *,
    agent_key: str | None,
    vercel_url: str | None,
    openai_model: str | None,
) -> Any:
    """Resolve exactly one target flag to an AgentTarget.

    Raises typer.BadParameter when zero or more than one flag is set.
    """
    supplied = {
        "--agent-key": agent_key,
        "--vercel-url": vercel_url,
        "--openai-model": openai_model,
    }
    active = {k: v for k, v in supplied.items() if v is not None}

    if len(active) == 0:
        raise typer.BadParameter(
            "Provide exactly one of: --agent-key, --vercel-url, --openai-model"
        )
    if len(active) > 1:
        raise typer.BadParameter(
            f"Only one target flag allowed; got: {', '.join(active)}"
        )

    if agent_key is not None:
        if not os.environ.get("ORQ_API_KEY"):
            raise typer.BadParameter(
                "--agent-key requires ORQ_API_KEY to be set in the environment."
            )
        from evaluatorq.simulation.adapters import from_orq_deployment

        return from_orq_deployment(agent_key)

    if vercel_url is not None:
        from evaluatorq.integrations.vercel_ai_sdk_integration import VercelAISdkTarget

        return VercelAISdkTarget(vercel_url)

    # openai_model
    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("ORQ_API_KEY"):
        raise typer.BadParameter(
            "--openai-model requires OPENAI_API_KEY (or ORQ_API_KEY for the AI Router) "
            "to be set in the environment."
        )
    from evaluatorq.redteam.backends.openai import OpenAIModelTarget

    return OpenAIModelTarget(model=openai_model)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]


# ---------------------------------------------------------------------------
# Evaluator resolution
# ---------------------------------------------------------------------------


def _resolve_evaluators(evaluators: list[str] | None) -> list[str] | None:
    """Validate evaluator names and raise BadParameter on unknown names."""
    if not evaluators:
        return None

    from evaluatorq.simulation.evaluators import SIMULATION_EVALUATORS, get_evaluator

    for name in evaluators:
        try:
            get_evaluator(name)
        except ValueError:  # noqa: PERF203
            known = ", ".join(SIMULATION_EVALUATORS)
            raise typer.BadParameter(
                f"Unknown evaluator: {name!r}. Known: {known}"
            ) from None

    return list(evaluators)


# ---------------------------------------------------------------------------
# Run store
# ---------------------------------------------------------------------------


def _sanitise_run_name(name: str) -> str:
    sanitised = name.lower()
    sanitised = re.sub(r"[^a-z0-9_-]", "_", sanitised)
    sanitised = re.sub(r"_+", "_", sanitised)
    sanitised = sanitised.strip("_")
    return sanitised[:64] or "sim"


def _get_sim_runs_dir() -> Path:
    return Path.cwd() / _SIM_RUNS_DIR_NAME


def _auto_save_run(
    *,
    run_name: str,
    mode: str,
    target_kind: str,
    evaluator_names: list[str],
    results: list[Any],
) -> Path:
    """Persist a SimulationRun JSON to .evaluatorq/sim-runs/."""
    from evaluatorq.simulation.types import SimulationRun

    runs_dir = _get_sim_runs_dir()
    runs_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d-%H%M%S")
    safe_name = _sanitise_run_name(run_name)
    base = f"{safe_name}_{ts}"

    # Aggregate scorer averages. Guard against non-numeric scores from a
    # misbehaving evaluator so a single bad entry can't crash the whole save.
    scorer_totals: dict[str, list[float]] = {}
    for result in results:
        scores: dict[str, float] = (result.metadata or {}).get("evaluator_scores", {})
        for scorer_name, score in scores.items():
            if isinstance(score, (int, float)):
                scorer_totals.setdefault(scorer_name, []).append(float(score))

    scorer_averages = {
        k: sum(v) / len(v) for k, v in scorer_totals.items() if v
    }

    run = SimulationRun(
        run_name=run_name,
        created_at=datetime.now(tz=timezone.utc),
        mode=mode,  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
        target_kind=target_kind,  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
        evaluator_names=evaluator_names,
        total_results=len(results),
        scorer_averages=scorer_averages,
        results=results,
    )
    payload = run.model_dump_json(indent=2)

    # Exclusive-create write: avoids the TOCTOU race between an exists() check
    # and a later write, and bounds the collision search.
    target_path = runs_dir / f"{base}.json"
    for counter in range(1000):
        try:
            with target_path.open("x", encoding="utf-8") as fh:
                _ = fh.write(payload)
        except FileExistsError:  # noqa: PERF203 — exclusive-create retry is the point
            target_path = runs_dir / f"{base}_{counter + 1:03d}.json"
        else:
            return target_path
    raise RuntimeError(
        f"Could not find a free run-store filename for {base!r} after 1000 attempts"
    )


def _infer_target_kind(
    *,
    agent_key: str | None,
    vercel_url: str | None,
    openai_model: str | None,
) -> str:
    if agent_key is not None:
        return "orq_deployment"
    if vercel_url is not None:
        return "vercel"
    return "openai_model"


# ---------------------------------------------------------------------------
# simulate
# ---------------------------------------------------------------------------


@app.command(no_args_is_help=True)
def simulate(
    datapoints: Annotated[
        Path,
        typer.Option("--datapoints", "-d", help="Path to datapoints JSONL file."),
    ],
    agent_key: Annotated[
        str | None,
        typer.Option("--agent-key", help="Orq deployment key (requires ORQ_API_KEY)."),
    ] = None,
    vercel_url: Annotated[
        str | None,
        typer.Option("--vercel-url", help="Vercel AI SDK endpoint URL."),
    ] = None,
    openai_model: Annotated[
        str | None,
        typer.Option(
            "--openai-model",
            help=(
                "OpenAI-compatible model name. Provider is resolved from env: "
                "OPENAI_API_KEY (+ optional OPENAI_BASE_URL for vLLM/OpenRouter/"
                "Azure-compatible/local endpoints), otherwise ORQ_API_KEY routes "
                "via the Orq AI Router (ORQ_BASE_URL overrides the host). OpenAI "
                "wins if both keys are set; namespace the model accordingly "
                "(e.g. 'gpt-4o-mini' for OpenAI, 'openai/gpt-4o-mini' for the Orq router)."
            ),
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Path to write results JSONL."),
    ] = None,
    name: Annotated[
        str,
        typer.Option("--name", "-n", help="Run name for the run-store entry."),
    ] = "sim",
    sim_model: Annotated[
        str,
        typer.Option(
            "--sim-model",
            help=(
                "Model for the user-simulator and judge. Provider resolved from "
                "env: ORQ_API_KEY -> Orq router, else OPENAI_API_KEY "
                "(+ OPENAI_BASE_URL) -> OpenAI-compatible endpoint. The default "
                "targets the Orq router; for OpenAI-direct pass a bare model "
                "name (e.g. 'gpt-5.4-mini', no provider prefix)."
            ),
        ),
    ] = DEFAULT_MODEL,
    max_turns: Annotated[
        int,
        typer.Option("--max-turns", help="Maximum conversation turns."),
    ] = 10,
    parallelism: Annotated[
        int,
        typer.Option("--parallelism", help="Concurrent simulations."),
    ] = 5,
    evaluator: Annotated[
        list[str] | None,
        typer.Option("--evaluator", help="Evaluator name (repeatable). Defaults to API defaults."),
    ] = None,
    no_save: Annotated[  # noqa: FBT002
        bool,
        typer.Option("--no-save", help="Skip writing to .evaluatorq/sim-runs/."),
    ] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Debug logging.")] = False,  # noqa: FBT002
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Warning-only logging.")] = False,  # noqa: FBT002
) -> None:
    """Run simulations from a pre-built datapoints file.

    Targets (provide exactly one):

    - --agent-key KEY    Orq deployment/agent (requires ORQ_API_KEY).
    - --vercel-url URL   Vercel AI SDK HTTP endpoint.
    - --openai-model M   OpenAI-compatible model. Provider from env:
      OPENAI_API_KEY (+ optional OPENAI_BASE_URL for vLLM/OpenRouter/local),
      else ORQ_API_KEY for the Orq AI Router. OpenAI wins if both set.

    Note: --agent-key targets an Orq *agent*; --openai-model with ORQ_API_KEY
    targets a raw model via the Orq *router*. They are different surfaces.
    """
    _configure_logging(verbose=verbose, quiet=quiet)

    if not datapoints.exists():
        raise typer.BadParameter(f"Datapoints file not found: {datapoints}")

    target = _resolve_target(
        agent_key=agent_key, vercel_url=vercel_url, openai_model=openai_model
    )
    evaluator_names = _resolve_evaluators(evaluator)
    target_kind = _infer_target_kind(
        agent_key=agent_key, vercel_url=vercel_url, openai_model=openai_model
    )

    try:
        results = asyncio.run(
            _simulate_impl(
                datapoints_path=datapoints,
                target=target,
                sim_model=sim_model,
                max_turns=max_turns,
                parallelism=parallelism,
                evaluator_names=evaluator_names,
                evaluation_name=name,
            )
        )
    except KeyboardInterrupt:
        typer.echo("^C aborted.", err=True)
        raise typer.Exit(130) from None
    except asyncio.CancelledError:
        typer.echo("^C aborted.", err=True)
        raise typer.Exit(130) from None
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from None

    _print_summary(results)

    if output:
        _write_results(results, output)

    if not no_save:
        run_path = _auto_save_run(
            run_name=name,
            mode="simulate",
            target_kind=target_kind,
            evaluator_names=evaluator_names or DEFAULT_EVALUATOR_NAMES,
            results=results,
        )
        typer.echo(f"Run saved: {run_path}", err=True)


async def _simulate_impl(
    *,
    datapoints_path: Path,
    target: Any,
    sim_model: str,
    max_turns: int,
    parallelism: int,
    evaluator_names: list[str] | None,
    evaluation_name: str,
) -> list[Any]:
    from evaluatorq.simulation.api import simulate
    from evaluatorq.simulation.utils.dataset_export import load_datapoints_from_jsonl

    loaded = load_datapoints_from_jsonl(str(datapoints_path))
    if not loaded:
        raise ValueError(f"No datapoints loaded from {datapoints_path}")

    return await simulate(
        datapoints=loaded,
        target=target,
        sim_model=sim_model,
        max_turns=max_turns,
        parallelism=parallelism,
        evaluator_names=evaluator_names,
        evaluation_name=evaluation_name,
    )


# ---------------------------------------------------------------------------
# run  (generate + simulate)
# ---------------------------------------------------------------------------


@app.command(no_args_is_help=True)
def run(
    agent_description: Annotated[
        str,
        typer.Option("--agent-description", help="Free-text description of the agent under test."),
    ],
    agent_key: Annotated[
        str | None,
        typer.Option("--agent-key", help="Orq deployment key (requires ORQ_API_KEY)."),
    ] = None,
    vercel_url: Annotated[
        str | None,
        typer.Option("--vercel-url", help="Vercel AI SDK endpoint URL."),
    ] = None,
    openai_model: Annotated[
        str | None,
        typer.Option(
            "--openai-model",
            help=(
                "OpenAI-compatible model name. Provider is resolved from env: "
                "OPENAI_API_KEY (+ optional OPENAI_BASE_URL for vLLM/OpenRouter/"
                "Azure-compatible/local endpoints), otherwise ORQ_API_KEY routes "
                "via the Orq AI Router (ORQ_BASE_URL overrides the host). OpenAI "
                "wins if both keys are set; namespace the model accordingly "
                "(e.g. 'gpt-4o-mini' for OpenAI, 'openai/gpt-4o-mini' for the Orq router)."
            ),
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Path to write results JSONL."),
    ] = None,
    name: Annotated[
        str,
        typer.Option("--name", "-n", help="Run name for the run-store entry."),
    ] = "sim",
    sim_model: Annotated[
        str,
        typer.Option(
            "--sim-model",
            help=(
                "Model for the user-simulator, the judge, and persona/scenario/"
                "first-message generation. Provider resolved from env: "
                "ORQ_API_KEY -> Orq router, else OPENAI_API_KEY (+ OPENAI_BASE_URL) "
                "-> OpenAI-compatible endpoint. The default targets the Orq "
                "router; for OpenAI-direct pass a bare model name "
                "(e.g. 'gpt-5.4-mini', no provider prefix)."
            ),
        ),
    ] = DEFAULT_MODEL,
    max_turns: Annotated[
        int,
        typer.Option("--max-turns", help="Maximum conversation turns."),
    ] = 10,
    parallelism: Annotated[
        int,
        typer.Option("--parallelism", help="Concurrent simulations."),
    ] = 5,
    num_personas: Annotated[
        int,
        typer.Option("--num-personas", help="Number of personas to generate."),
    ] = 5,
    num_scenarios: Annotated[
        int,
        typer.Option("--num-scenarios", help="Number of scenarios to generate."),
    ] = 5,
    evaluator: Annotated[
        list[str] | None,
        typer.Option("--evaluator", help="Evaluator name (repeatable). Defaults to API defaults."),
    ] = None,
    no_save: Annotated[  # noqa: FBT002
        bool,
        typer.Option("--no-save", help="Skip writing to .evaluatorq/sim-runs/."),
    ] = False,
    save_datapoints: Annotated[
        Path | None,
        typer.Option(
            "--save-datapoints",
            help=(
                "Also write the generated datapoints (the simulate inputs) to this "
                "JSONL path, for reproducible re-runs via `sim simulate --datapoints`."
            ),
        ),
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Debug logging.")] = False,  # noqa: FBT002
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Warning-only logging.")] = False,  # noqa: FBT002
) -> None:
    """Generate personas and scenarios, then run simulations (generate + simulate).

    Targets (provide exactly one):

    - --agent-key KEY    Orq deployment/agent (requires ORQ_API_KEY).
    - --vercel-url URL   Vercel AI SDK HTTP endpoint.
    - --openai-model M   OpenAI-compatible model. Provider from env:
      OPENAI_API_KEY (+ optional OPENAI_BASE_URL for vLLM/OpenRouter/local),
      else ORQ_API_KEY for the Orq AI Router. OpenAI wins if both set.

    Note: --agent-key targets an Orq *agent*; --openai-model with ORQ_API_KEY
    targets a raw model via the Orq *router*. They are different surfaces.
    """
    _configure_logging(verbose=verbose, quiet=quiet)

    target = _resolve_target(
        agent_key=agent_key, vercel_url=vercel_url, openai_model=openai_model
    )
    evaluator_names = _resolve_evaluators(evaluator)
    target_kind = _infer_target_kind(
        agent_key=agent_key, vercel_url=vercel_url, openai_model=openai_model
    )

    try:
        results = asyncio.run(
            _run_impl(
                agent_description=agent_description,
                target=target,
                sim_model=sim_model,
                max_turns=max_turns,
                parallelism=parallelism,
                num_personas=num_personas,
                num_scenarios=num_scenarios,
                evaluator_names=evaluator_names,
                evaluation_name=name,
                save_datapoints=save_datapoints,
            )
        )
    except KeyboardInterrupt:
        typer.echo("^C aborted.", err=True)
        raise typer.Exit(130) from None
    except asyncio.CancelledError:
        typer.echo("^C aborted.", err=True)
        raise typer.Exit(130) from None
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from None

    _print_summary(results)

    if output:
        _write_results(results, output)

    if save_datapoints is not None:
        typer.echo(f"Datapoints saved: {save_datapoints}", err=True)

    if not no_save:
        run_path = _auto_save_run(
            run_name=name,
            mode="run",
            target_kind=target_kind,
            evaluator_names=evaluator_names or DEFAULT_EVALUATOR_NAMES,
            results=results,
        )
        typer.echo(f"Run saved: {run_path}", err=True)


async def _run_impl(
    *,
    agent_description: str,
    target: Any,
    sim_model: str,
    max_turns: int,
    parallelism: int,
    num_personas: int,
    num_scenarios: int,
    evaluator_names: list[str] | None,
    evaluation_name: str,
    save_datapoints: Path | None = None,
) -> list[Any]:
    from evaluatorq.simulation.api import generate_and_simulate

    emit = None
    if save_datapoints is not None:
        save_path = save_datapoints

        def _emit(dps: list[Any]) -> None:
            _write_datapoints(dps, save_path)

        emit = _emit

    return await generate_and_simulate(
        agent_description=agent_description,
        target=target,
        sim_model=sim_model,
        max_turns=max_turns,
        parallelism=parallelism,
        num_personas=num_personas,
        num_scenarios=num_scenarios,
        evaluator_names=evaluator_names,
        evaluation_name=evaluation_name,
        emit_datapoints=emit,
    )


# ---------------------------------------------------------------------------
# generate  (datapoints only, no simulation)
# ---------------------------------------------------------------------------


@app.command(no_args_is_help=True)
def generate(
    agent_description: Annotated[
        str,
        typer.Option("--agent-description", help="Free-text description of the agent under test."),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Path to write the generated datapoints JSONL."),
    ],
    sim_model: Annotated[
        str,
        typer.Option(
            "--sim-model",
            help=(
                "Model for persona/scenario/first-message generation. Provider "
                "resolved from env: ORQ_API_KEY -> Orq router, else "
                "OPENAI_API_KEY (+ OPENAI_BASE_URL) -> OpenAI-compatible "
                "endpoint. The default targets the Orq router; for OpenAI-direct "
                "pass a bare model name (e.g. 'gpt-5.4-mini', no provider prefix)."
            ),
        ),
    ] = DEFAULT_MODEL,
    num_personas: Annotated[
        int,
        typer.Option("--num-personas", min=1, help="Number of personas to generate."),
    ] = 5,
    num_scenarios: Annotated[
        int,
        typer.Option("--num-scenarios", min=1, help="Number of scenarios to generate."),
    ] = 5,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Debug logging.")] = False,  # noqa: FBT002
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Warning-only logging.")] = False,  # noqa: FBT002
) -> None:
    """Generate simulation datapoints from an agent description (no simulation).

    Builds personas × scenarios with generated first messages and writes them
    as a datapoints JSONL file. Feed that file to ``sim simulate --datapoints``
    to run — splitting generation out keeps the datapoint set frozen across
    simulate runs. No agent target is contacted; only the ``--sim-model``
    generator is called.
    """
    _configure_logging(verbose=verbose, quiet=quiet)

    try:
        datapoints = asyncio.run(
            _generate_impl(
                agent_description=agent_description,
                sim_model=sim_model,
                num_personas=num_personas,
                num_scenarios=num_scenarios,
            )
        )
    except KeyboardInterrupt:
        typer.echo("^C aborted.", err=True)
        raise typer.Exit(130) from None
    except asyncio.CancelledError:
        typer.echo("^C aborted.", err=True)
        raise typer.Exit(130) from None
    except (ValueError, RuntimeError) as exc:
        # RuntimeError covers "first-message generation produced no datapoints"
        # from _resolve_or_generate_datapoints — surface as one line, not a
        # traceback (per the spec error-handling contract).
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from None

    _write_datapoints(datapoints, output)
    typer.echo(f"Generated {len(datapoints)} datapoint(s) -> {output}", err=True)


async def _generate_impl(
    *,
    agent_description: str,
    sim_model: str,
    num_personas: int,
    num_scenarios: int,
) -> list[Any]:
    from evaluatorq.simulation.api import generate

    return await generate(
        agent_description=agent_description,
        num_personas=num_personas,
        num_scenarios=num_scenarios,
        sim_model=sim_model,
    )


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


@app.command(no_args_is_help=True)
def export(
    input_path: Annotated[
        Path,
        typer.Option("--input", "-i", help="Path to results JSONL file."),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Path to write OpenResponses payload JSON."),
    ],
) -> None:
    """Convert simulation results JSONL to OpenResponses payload JSON."""
    if not input_path.exists():
        raise typer.BadParameter(f"Input file not found: {input_path}")

    from evaluatorq.simulation.convert import to_open_responses
    from evaluatorq.simulation.types import SimulationResult
    from evaluatorq.simulation.utils.dataset_export import parse_jsonl

    try:
        content = input_path.read_text(encoding="utf-8")
        results: list[SimulationResult] = parse_jsonl(content, cls=SimulationResult)  # pyright: ignore[reportAssignmentType]
    except Exception as exc:
        raise typer.BadParameter(f"Failed to read {input_path}: {exc}") from exc

    payloads = [to_open_responses(result) for result in results]

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps([p if isinstance(p, dict) else p.model_dump(mode="json") for p in payloads], indent=2),
        encoding="utf-8",
    )
    typer.echo(f"Exported {len(payloads)} result(s) to {output}")


# ---------------------------------------------------------------------------
# validate-dataset
# ---------------------------------------------------------------------------


@app.command("validate-dataset", no_args_is_help=True)
def validate_dataset(
    path: Annotated[
        Path,
        typer.Argument(help="Path to datapoints JSONL file to validate."),
    ],
) -> None:
    """Validate a simulation datapoints JSONL file."""
    if not path.exists():
        raise typer.BadParameter(f"File not found: {path}")

    from evaluatorq.simulation.types import Datapoint

    try:
        content = path.read_text(encoding="utf-8")
    except Exception as exc:
        raise typer.BadParameter(f"Cannot read {path}: {exc}") from exc

    bad_count = 0
    lines = [line for line in content.splitlines() if line.strip()]

    valid_datapoints: list[Datapoint] = []
    for i, line in enumerate(lines, start=1):
        try:
            dp = Datapoint.model_validate_json(line)
            valid_datapoints.append(dp)
        except Exception as exc:  # noqa: PERF203
            typer.echo(f"Line {i}: {exc}", err=True)
            bad_count += 1

    if bad_count:
        typer.echo(f"{bad_count} invalid line(s) in {path}", err=True)
        raise typer.Exit(1)

    typer.echo(f"OK — {len(valid_datapoints)} valid datapoint(s) in {path}")


# ---------------------------------------------------------------------------
# runs
# ---------------------------------------------------------------------------


@app.command()
def runs(
    directory: Annotated[
        Path | None,
        typer.Argument(help="Directory to scan (default: .evaluatorq/sim-runs/)."),
    ] = None,
    limit: Annotated[
        int,
        typer.Option("--limit", "-n", help="Maximum number of runs to show."),
    ] = 20,
) -> None:
    """List recent simulation runs."""
    runs_dir = directory or _get_sim_runs_dir()

    if not runs_dir.exists():
        typer.echo(f"No sim-runs directory found at {runs_dir}")
        raise typer.Exit(0)

    run_files = sorted(
        runs_dir.glob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[:limit]

    if not run_files:
        typer.echo(f"No runs found in {runs_dir}")
        raise typer.Exit(0)

    rows: list[dict[str, Any]] = []
    malformed = 0
    for run_file in run_files:
        try:
            data = json.loads(run_file.read_text(encoding="utf-8"))
            rows.append(
                {
                    "name": data.get("run_name", "—"),
                    "date": data.get("created_at", "—")[:19].replace("T", " "),
                    "mode": data.get("mode", "—"),
                    "target": data.get("target_kind", "—"),
                    "n": str(data.get("total_results", "—")),
                    "scores": _format_scorer_averages(data.get("scorer_averages", {})),
                    "file": run_file.name,
                }
            )
        except Exception:  # noqa: PERF203
            malformed += 1

    try:
        import io

        from rich.console import Console
        from rich.table import Table

        table = Table(title="Simulation Runs")
        for col in ("Name", "Date", "Mode", "Target", "N", "Scores", "File"):
            table.add_column(col)
        for row in rows:
            table.add_row(
                row["name"],
                row["date"],
                row["mode"],
                row["target"],
                row["n"],
                row["scores"],
                row["file"],
            )
        # Render through an explicit buffer + typer.echo rather than letting
        # Rich write straight to its own stdout handle: this keeps output on
        # the Click/Typer stream so redirection and test capture both see it.
        # Width is taken from the real terminal (falling back to 80 when there
        # is no tty) so layout still adapts instead of being pinned to a
        # constant.
        import shutil

        width = shutil.get_terminal_size(fallback=(80, 24)).columns
        buffer = io.StringIO()
        Console(file=buffer, width=width).print(table)
        typer.echo(buffer.getvalue(), nl=False)
    except ImportError:
        header = f"{'Name':<20} {'Date':<20} {'Mode':<10} {'Target':<16} {'N':>4}  {'Scores':<30} File"
        typer.echo(header)
        typer.echo("-" * len(header))
        for row in rows:
            typer.echo(
                f"{row['name']:<20} {row['date']:<20} {row['mode']:<10} "
                f"{row['target']:<16} {row['n']:>4}  {row['scores']:<30} {row['file']}"
            )

    if malformed:
        typer.echo(f"Warning: {malformed} malformed run file(s) skipped.", err=True)


def _format_scorer_averages(averages: dict[str, float]) -> str:
    if not averages:
        return "—"
    return "  ".join(f"{k}={v:.2f}" for k, v in averages.items())


# ---------------------------------------------------------------------------
# Summary helper
# ---------------------------------------------------------------------------


def _print_summary(results: list[Any]) -> None:
    total = len(results)
    if total == 0:
        typer.echo("No results.")
        return

    achieved = sum(1 for r in results if r.goal_achieved)
    avg_turns = sum(r.turn_count for r in results) / total
    total_broken = sum(len(r.rules_broken) for r in results)

    typer.echo(f"\nResults: {total} simulations")
    typer.echo(f"  Goal achieved:  {achieved}/{total} ({achieved / total:.0%})")
    typer.echo(f"  Avg turns:      {avg_turns:.1f}")
    typer.echo(f"  Rules broken:   {total_broken}")


def _write_results(results: list[Any], output: Path) -> None:
    from evaluatorq.simulation.utils.dataset_export import export_results_to_jsonl

    output.parent.mkdir(parents=True, exist_ok=True)
    export_results_to_jsonl(results, str(output))
    typer.echo(f"Results written to {output}")


def _write_datapoints(datapoints: list[Any], output: Path) -> None:
    """Write datapoints as raw ``Datapoint`` JSONL — one ``model_dump_json()``
    per line. This is the canonical local handoff format: it preserves the
    datapoint ``id``, round-trips through ``load_datapoints_from_jsonl`` (which
    ``simulate`` uses), and validates under ``validate-dataset``. The Orq-dataset
    *envelope* exporter (``export_datapoints_to_jsonl``) is a separate upload
    format and is intentionally not used here.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [dp.model_dump_json() for dp in datapoints]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()

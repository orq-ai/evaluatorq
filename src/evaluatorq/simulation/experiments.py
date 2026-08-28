"""Orq experiments as simulation input.

Two modes for seeding simulations from a prior Orq experiment run:

- **Direct** (`datapoints_from_experiment`): each exported experiment row
  becomes one ``SimulationDatapoint`` via the same shape-tolerant extractor the
  dataset path uses. Rows uploaded by a previous simulation run
  (``inputs['datapoint']``) round-trip as-is.
- **Extension** (`extend_from_experiment`): the experiment's personas and
  scenarios seed the standard generators, which produce *new* similar-but-not-
  duplicate datapoints extending the run's coverage.

``simulate(experiment_id=...)`` wires the direct mode into the main entry point.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evaluatorq.contracts import LLMCallConfig
    from evaluatorq.simulation.types import SimulationDatapoint


def _require_api_key(api_key: str | None) -> str:
    key = api_key or os.environ.get('ORQ_API_KEY')
    if not key:
        raise ValueError('ORQ_API_KEY environment variable is not set. Set it to load Orq experiments.')
    return key


async def datapoints_from_experiment(
    experiment_id: str,
    *,
    run_id: str | None = None,
    api_key: str | None = None,
) -> list[SimulationDatapoint]:
    """Load an Orq experiment run's rows as simulation datapoints (direct mode).

    Reuses `evaluatorq.fetch_data.fetch_experiment_datapoints` (the core
    ``evaluate()`` fetcher) and parses each row with the same shape-tolerant
    extractor as the dataset path, so any row whose ``inputs`` match a
    simulation input shape (``datapoint`` / ``persona`` + ``scenario`` / etc.)
    is accepted. Experiments uploaded by a previous simulation run qualify
    automatically.

    Args:
        experiment_id: The experiment (sheet) ID — read it off the Orq UI URL.
        run_id: A specific run (manifest) ID. Latest run when omitted.
        api_key: Orq API key; falls back to ``ORQ_API_KEY``.

    Raises:
        ValueError: on missing API key, unloadable experiment/run, a row that
            does not match a simulation input shape, or zero usable rows.
    """
    from pydantic import ValidationError

    from evaluatorq.fetch_data import fetch_experiment_datapoints
    from evaluatorq.simulation._datapoint_io import _extract_single_datapoint

    key = _require_api_key(api_key)
    rows = await fetch_experiment_datapoints(key, experiment_id, run_id)

    out: list[SimulationDatapoint] = []
    try:
        out.extend(_extract_single_datapoint(row, source='row') for row in rows)
    except (ValueError, ValidationError) as e:
        # extend consumes the generator row by row, so len(out) == failing row index
        raise ValueError(f'experiment {experiment_id!r} row {len(out)}: {e}') from e
    if not out:
        raise ValueError(f'Experiment {experiment_id!r} returned zero simulation-compatible rows')
    return out


async def extend_from_experiment(
    experiment_id: str,
    *,
    run_id: str | None = None,
    num_personas: int = 3,
    num_scenarios: int = 5,
    sim_model: str | None = None,
    llm_config: LLMCallConfig | None = None,
    agent_description: str | None = None,
    api_key: str | None = None,
) -> list[SimulationDatapoint]:
    """Generate *new* datapoints seeded by an Orq experiment run (extension mode).

    Fetches the experiment's rows (direct mode), then feeds their personas and
    scenarios to the standard ``DatapointGenerator`` as context, instructing it
    to extend — not duplicate — the seed coverage. Returns only the newly
    generated datapoints (``num_personas x num_scenarios``); combine with
    `datapoints_from_experiment` to also replay the originals.

    Args:
        experiment_id: The experiment (sheet) ID to seed from.
        run_id: A specific run (manifest) ID. Latest run when omitted.
        num_personas: New personas to generate.
        num_scenarios: New scenarios to generate.
        sim_model: Generation model; defaults to the simulation default.
        llm_config: Sampling settings for the generators. ``llm_config.model`` wins
            over ``sim_model`` when both are given.
        agent_description: Description of the agent under test for the
            generators. Derived from the seed scenarios' goals when omitted.
        api_key: Orq API key; falls back to ``ORQ_API_KEY``.
    """
    from evaluatorq.simulation._config import resolve_sim_llm_config
    from evaluatorq.simulation.generators import DatapointGenerator
    from evaluatorq.simulation.types import DEFAULT_MODEL

    llm_config = resolve_sim_llm_config(
        sim_model=sim_model or DEFAULT_MODEL, llm_config=llm_config, caller='extend_from_experiment'
    )
    seeds = await datapoints_from_experiment(experiment_id, run_id=run_id, api_key=api_key)

    generator = DatapointGenerator(config=llm_config)
    try:
        return await generator.generate_from_description(
            agent_description=agent_description or _describe_agent(seeds),
            context=_seed_context(seeds),
            num_personas=num_personas,
            num_scenarios=num_scenarios,
        )
    finally:
        await generator.close()


def _describe_agent(seeds: list[SimulationDatapoint]) -> str:
    """Fallback agent description built from the seed scenarios' goals.

    Seeds are guaranteed non-empty but individual goals are not, so when every
    goal is blank fall back to a generic description instead of handing the
    generator a truncated 'goals such as: ' prompt.
    """
    goals = list(dict.fromkeys(dp.scenario.goal for dp in seeds if dp.scenario.goal))
    if not goals:
        return 'A general-purpose assistant agent; extend the seed personas and scenarios below.'
    return 'An agent whose users pursue goals such as: ' + '; '.join(goals[:10])


def _seed_context(seeds: list[SimulationDatapoint]) -> str:
    """Render the seed personas/scenarios as generator context.

    Deduplicates by name so a cartesian-product experiment doesn't repeat the
    same persona once per scenario.
    """
    personas = {dp.persona.name: dp.persona for dp in seeds}
    scenarios = {dp.scenario.name: dp.scenario for dp in seeds}

    lines = [
        (
            'The following personas and scenarios come from a previous run. '
            'Generate NEW personas and scenarios in the same domain and style that '
            'EXTEND this coverage — do not duplicate or trivially rephrase them.'
        ),
        '',
        'Seed personas:',
    ]
    lines += [f'- {p.name}: {p.background}' for p in personas.values()]
    lines.append('')
    lines.append('Seed scenarios:')
    lines += [f'- {s.name}: {s.goal}' + (f' ({s.context})' if s.context else '') for s in scenarios.values()]
    return '\n'.join(lines)

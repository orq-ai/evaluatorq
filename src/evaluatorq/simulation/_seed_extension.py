"""Seed-based extension: turn a set of seed datapoints into new, similar-but-not-duplicate ones.

`experiments.extend_from_experiment` and `datasets.extend_from_dataset` each fetch a set of seed
``SimulationDatapoint``s from a different Orq source (an experiment run, a named dataset), then feed
their personas and scenarios to the standard generators as context so the model produces *fresh*
coverage in the same domain. That second half is identical for both sources, so it lives here once
rather than being copied per source. (Trace extension uses a distilled traffic profile as context,
not seed objects, so it does not share this path.)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evaluatorq.contracts import LLMCallConfig
    from evaluatorq.simulation.types import SimulationDatapoint


async def extend_from_seeds(
    seeds: list[SimulationDatapoint],
    *,
    num_personas: int = 3,
    num_scenarios: int = 5,
    llm_config: LLMCallConfig | None = None,
    agent_description: str | None = None,
) -> list[SimulationDatapoint]:
    """Generate *new* datapoints seeded by existing ones (extension mode).

    Feeds the seeds' personas and scenarios to ``DatapointGenerator`` as context, instructing it to
    extend — not duplicate — the seed coverage. Returns only the newly generated datapoints
    (``num_personas x num_scenarios``); combine with the source's direct loader to also replay the
    originals.

    ``agent_description`` is derived from the seed scenarios' goals when omitted. ``llm_config``
    defaults to the simulation default model with every other field unset.
    """
    from evaluatorq.simulation._config import sim_llm_config
    from evaluatorq.simulation.generators import DatapointGenerator

    resolved = sim_llm_config(llm_config)
    generator = DatapointGenerator(config=resolved)
    try:
        return await generator.generate_from_description(
            agent_description=agent_description or describe_agent(seeds),
            context=seed_context(seeds),
            num_personas=num_personas,
            num_scenarios=num_scenarios,
        )
    finally:
        await generator.close()


def describe_agent(seeds: list[SimulationDatapoint]) -> str:
    """Fallback agent description built from the seed scenarios' goals.

    Seeds are guaranteed non-empty but individual goals are not, so when every goal is blank fall
    back to a generic description instead of handing the generator a truncated 'goals such as: '
    prompt.
    """
    goals = list(dict.fromkeys(dp.scenario.goal for dp in seeds if dp.scenario.goal))
    if not goals:
        return 'A general-purpose assistant agent; extend the seed personas and scenarios below.'
    return 'An agent whose users pursue goals such as: ' + '; '.join(goals[:10])


def seed_context(seeds: list[SimulationDatapoint]) -> str:
    """Render the seed personas/scenarios as generator context.

    Deduplicates by name so a cartesian-product source doesn't repeat the same persona once per
    scenario.
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

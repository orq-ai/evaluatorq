"""Seed-based extension: turn a set of seed datapoints into new, similar-but-not-duplicate ones.

`experiments.extend_from_experiment` and `datasets.extend_from_dataset` each fetch a set of seed
``SimulationDatapoint``s from a different Orq source (an experiment run, a named dataset), then feed
their personas and scenarios to the standard generators as context so the model produces *fresh*
coverage in the same domain. That second half is identical for both sources, so it lives here once
rather than being copied per source. (Trace extension uses a distilled traffic profile as context,
not seed objects, so it does not share this path.)
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from loguru import logger

from evaluatorq.common.template_engine import render_template

if TYPE_CHECKING:
    from evaluatorq.contracts import LLMCallConfig
    from evaluatorq.simulation.types import SimulationDatapoint


def require_orq_api_key(api_key: str | None, *, purpose: str) -> str:
    """Resolve the Orq API key from the argument or ``ORQ_API_KEY``, or raise naming what it is for.

    Shared by the two Orq seed sources (``experiments``, ``datasets``), which both live behind this
    module, so the guard is defined once. ``purpose`` completes "Set it to {purpose}." (e.g.
    ``"load Orq datasets"``).
    """
    key = api_key or os.environ.get('ORQ_API_KEY')
    if not key:
        raise ValueError(f'ORQ_API_KEY environment variable is not set. Set it to {purpose}.')
    return key


async def extend_from_seeds(
    seeds: list[SimulationDatapoint],
    *,
    num_personas: int = 3,
    num_scenarios: int = 5,
    llm_config: LLMCallConfig | None = None,
    agent_description: str | None = None,
    api_key: str | None = None,
) -> list[SimulationDatapoint]:
    """Generate *new* datapoints seeded by existing ones (extension mode).

    Feeds the seeds' personas and scenarios to ``DatapointGenerator`` as context, instructing it to
    extend — not duplicate — the seed coverage. Returns only the newly generated datapoints: the
    cartesian product of the personas and scenarios actually generated, whose counts may differ from
    the requested ``num_personas`` and ``num_scenarios``. Combine with the source's direct loader to
    also replay the originals.

    ``agent_description`` is derived from the seed scenarios' goals when omitted. ``llm_config``
    defaults to the fast model role with every other field unset. ``api_key`` authenticates
    generation through Orq unless ``llm_config.client`` supplies a client.
    """
    from evaluatorq.simulation._config import sim_llm_config
    from evaluatorq.simulation.generators import DatapointGenerator

    resolved = sim_llm_config(llm_config)
    generator = DatapointGenerator(config=resolved, orq_api_key=api_key)
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
        logger.warning(
            'All {} seed scenario(s) have a blank goal; falling back to a generic agent description '
            'for datapoint generation. The generated personas/scenarios will be less targeted.',
            len(seeds),
        )
        return 'A general-purpose assistant agent; extend the seed personas and scenarios below.'
    return render_template(
        'An agent whose users pursue goals such as: {{goals}}',
        {'goals': '; '.join(goals[:10])},
    )


def seed_context(seeds: list[SimulationDatapoint]) -> str:
    """Render the seed personas/scenarios as generator context.

    Deduplicates identical objects from cartesian-product rows while retaining distinct variants
    that share a name. The serialized objects expose the traits and criteria that distinguish them.
    """
    personas = dict.fromkeys(dp.persona.model_dump_json(exclude_none=True) for dp in seeds)
    scenarios = dict.fromkeys(dp.scenario.model_dump_json(exclude_none=True) for dp in seeds)

    persona_lines = [render_template('- {{persona}}', {'persona': payload}) for payload in personas]
    scenario_lines = [render_template('- {{scenario}}', {'scenario': payload}) for payload in scenarios]

    lines = [
        (
            'The following personas and scenarios come from a previous run. '
            'Generate NEW personas and scenarios in the same domain and style that '
            'EXTEND this coverage — do not duplicate or trivially rephrase them.'
        ),
        '',
        'Seed personas:',
        *persona_lines,
        '',
        'Seed scenarios:',
        *scenario_lines,
    ]
    return '\n'.join(lines)

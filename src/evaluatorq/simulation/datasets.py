"""Orq datasets as simulation input.

Two modes for seeding simulations from a named Orq dataset, mirroring `experiments`:

- **Direct** (`datapoints_from_dataset`): stream the dataset and parse each row into a
  ``SimulationDatapoint`` via the same shape-tolerant extractor the experiment and inline paths use.
  This is the canonical dataset loader — ``simulate(dataset_id=...)`` resolves through it.
- **Extension** (`extend_from_dataset`): the dataset's personas and scenarios seed the standard
  generators, which produce *new* similar-but-not-duplicate datapoints extending its coverage.
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
        raise ValueError('ORQ_API_KEY environment variable is not set. Set it to load Orq datasets.')
    return key


async def datapoints_from_dataset(
    dataset_id: str,
    *,
    api_key: str | None = None,
) -> list[SimulationDatapoint]:
    """Load a named Orq dataset's rows as simulation datapoints (direct mode).

    Streams the dataset and parses each row with the same shape-tolerant extractor as the experiment
    and inline paths, so any row whose ``inputs`` match a simulation input shape (``datapoint`` /
    ``persona`` + ``scenario`` / etc.) is accepted. Datasets uploaded by a previous simulation run
    qualify automatically.

    Args:
        dataset_id: The Orq dataset ID.
        api_key: Orq API key; falls back to ``ORQ_API_KEY``.

    Raises:
        ValueError: on missing API key, a row that does not match a simulation input shape, or zero
            usable rows.
    """
    from pydantic import ValidationError

    from evaluatorq.fetch_data import fetch_dataset_batches, setup_orq_client
    from evaluatorq.simulation._datapoint_io import _extract_single_datapoint

    key = _require_api_key(api_key)
    orq_client = setup_orq_client(key)
    out: list[SimulationDatapoint] = []
    row = 0
    async for batch in fetch_dataset_batches(orq_client, dataset_id):
        for eq_dp in batch.datapoints:
            try:
                out.append(_extract_single_datapoint(eq_dp, source='row'))
            except (ValueError, ValidationError) as e:
                raise ValueError(f'dataset {dataset_id!r} row {row}: {e}') from e
            row += 1
    if not out:
        raise ValueError(f'Dataset {dataset_id!r} returned zero simulation-compatible datapoints')
    return out


async def extend_from_dataset(
    dataset_id: str,
    *,
    num_personas: int = 3,
    num_scenarios: int = 5,
    llm_config: LLMCallConfig | None = None,
    agent_description: str | None = None,
    api_key: str | None = None,
) -> list[SimulationDatapoint]:
    """Generate *new* datapoints seeded by an Orq dataset (extension mode).

    Fetches the dataset's rows (direct mode), then feeds their personas and scenarios to the standard
    ``DatapointGenerator`` as context, instructing it to extend — not duplicate — the seed coverage.
    Returns only the newly generated datapoints (``num_personas x num_scenarios``); combine with
    `datapoints_from_dataset` to also replay the originals.

    Args:
        dataset_id: The Orq dataset ID to seed from.
        num_personas: New personas to generate.
        num_scenarios: New scenarios to generate.
        llm_config: Model and sampling settings for the generators. Defaults to the simulation
            default model with every other field unset.
        agent_description: Description of the agent under test for the generators. Derived from the
            seed scenarios' goals when omitted.
        api_key: Orq API key; falls back to ``ORQ_API_KEY``.
    """
    from evaluatorq.simulation._seed_extension import extend_from_seeds

    seeds = await datapoints_from_dataset(dataset_id, api_key=api_key)
    return await extend_from_seeds(
        seeds,
        num_personas=num_personas,
        num_scenarios=num_scenarios,
        llm_config=llm_config,
        agent_description=agent_description,
    )

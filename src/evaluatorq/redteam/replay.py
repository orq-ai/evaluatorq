"""Rebuild a prior red-team run's datapoints so it can be replayed verbatim.

Saved runs carry a ``datapoints`` key (the exact ``DataPoint.inputs`` fed to the
pipeline, written by ``_auto_save_run``). Replaying reads them back and hands
them to the runner as pre-built datapoints, so no strategy planning, attack
generation, or dataset load happens: the same attacks run again, against
whatever target and evaluators the new invocation specifies.

Runs saved before this existed have no ``datapoints`` key and cannot be
replayed — the report alone does not carry the dynamic strategy definitions the
orchestrator needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from evaluatorq.common.replay import ReplayError, load_run_payload
from evaluatorq.redteam.contracts import Pipeline

if TYPE_CHECKING:
    from pathlib import Path

    from evaluatorq.types import DataPoint

SURFACE = 'red team'

DATAPOINTS_KEY = 'datapoints'
"""Key under which a saved run stores the raw inputs of every datapoint it ran."""

RUN_CONFIG_KEY = 'run_config'
"""Key under which a saved run stores the execution knobs the datapoints don't
carry (turn budget, attacker steering), so a replay can restore them."""


@dataclass(frozen=True)
class RedTeamReplay:
    """A prior run's cases, ready to be re-run."""

    datapoints: list[DataPoint]
    pipeline: Pipeline
    categories: list[str]
    run_name: str
    path: Path
    max_turns: int | None = None
    """Turn budget the original run used. Restored unless the caller overrides
    it — the datapoints alone don't pin it down, and replaying a 10-turn run at
    the 5-turn default would not be the same run."""
    attacker_instructions: str | None = None
    """Domain steering the original run used, restored for the same reason."""


def _looks_like_a_simulation_run(payload: dict[str, Any], rows: list[Any]) -> bool:
    """Detect a sim run file handed to the red-team replay path.

    Both surfaces store a top-level ``datapoints`` key, and a raw path is an
    accepted reference, so the mix-up is easy to make. ``mode`` is the same
    discriminator the dashboard uses to tell the two run kinds apart.
    """
    if payload.get('mode') in {'run', 'simulate', 'generate'}:
        return True
    return any(isinstance(row, dict) and 'persona' in row and 'scenario' in row for row in rows)


def _validate_datapoint(inputs: dict[str, Any], index: int, run_name: str) -> None:
    """Reject rows that could not drive an attack.

    A dynamic case needs its ``strategy``; a static one needs its ``messages``.
    Without one of the two the run would reach the orchestrator and die per-row
    on a KeyError, long after the confirm prompt.
    """
    if 'strategy' in inputs or inputs.get('messages'):
        return
    raise ReplayError(
        f'Previous red team run {run_name}: datapoint {index} ({inputs.get("id", "unnamed")!r}) has neither '
        "a 'strategy' (dynamic attack) nor 'messages' (static attack), so it cannot be replayed."
    )


def _infer_pipeline(datapoints: list[DataPoint], stored: Any) -> Pipeline:
    """Resolve the pipeline to replay under, preferring the datapoints' own tags.

    The stored label can under-report: ``merge_reports`` derives a report's
    pipeline from the sub-reports that actually produced rows, so a hybrid run
    whose static leg yields no results is saved as ``'dynamic'`` even though its
    datapoints still carry ``hybrid_source='static'``. Replaying that as dynamic
    would route ``messages``-only rows into the attack-generation job. Where the
    tags contradict the label, the tags win — they describe the cases in hand.

    Untagged datapoints (pure dynamic, or a static-mode run, neither of which
    tags anything) carry no signal, so the stored label stands.
    """
    try:
        stored_pipeline: Pipeline | None = Pipeline(stored)
    except ValueError:
        stored_pipeline = None

    sources = {dp.inputs.get('hybrid_source') for dp in datapoints}
    if 'static' in sources:
        tagged = Pipeline.HYBRID if sources - {'static'} else Pipeline.STATIC
        # Only DYNAMIC actually contradicts a static-tagged row; a stored
        # 'hybrid'/'static' is at least as specific, so leave it alone.
        return tagged if stored_pipeline in (None, Pipeline.DYNAMIC) else stored_pipeline

    return stored_pipeline if stored_pipeline is not None else Pipeline.DYNAMIC


def load_redteam_replay(reference: str, runs_dir: Path) -> RedTeamReplay:
    """Resolve *reference* to a saved run and rebuild its datapoints.

    Raises:
        ReplayError: The run cannot be found, read, or has no stored datapoints.
    """
    from evaluatorq.types import DataPoint

    payload, path = load_run_payload(reference, runs_dir, surface=SURFACE)

    raw = payload.get(DATAPOINTS_KEY)
    if not isinstance(raw, list) or not raw:
        raise ReplayError(
            f'{path.name} records no red team datapoints, so it cannot be replayed. Replay needs the '
            'cases themselves, which only the auto-saved run in the runs directory carries — not a '
            '--save detail summary report, and not runs saved before replay support existed. Re-run '
            'the original configuration once to produce a replayable run.'
        )

    if _looks_like_a_simulation_run(payload, raw):
        raise ReplayError(
            f'{path.name} is an agent simulation run, not a red team run. '
            f'Replay it with: eq sim simulate --from-run {path.name} --target <target>'
        )

    datapoints: list[DataPoint] = []
    for i, inputs in enumerate(raw):
        if not isinstance(inputs, dict):
            raise ReplayError(f'Previous red team run {path.name}: datapoint {i} is not an object.')
        _validate_datapoint(inputs, i, path.name)
        datapoints.append(DataPoint(inputs=dict(inputs)))

    categories = list(dict.fromkeys(str(c) for dp in datapoints if (c := dp.inputs.get('category'))))
    stored_config = payload.get(RUN_CONFIG_KEY)
    config = stored_config if isinstance(stored_config, dict) else {}
    max_turns = config.get('max_turns')
    instructions = config.get('attacker_instructions')

    return RedTeamReplay(
        datapoints=datapoints,
        pipeline=_infer_pipeline(datapoints, payload.get('pipeline')),
        categories=categories,
        run_name=str(payload.get('run_name') or path.stem),
        path=path,
        max_turns=max_turns if isinstance(max_turns, int) else None,
        attacker_instructions=instructions if isinstance(instructions, str) else None,
    )

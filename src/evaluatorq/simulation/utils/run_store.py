"""Run-store persistence for agent simulation.

Shared by the CLI (`evaluatorq sim run`) and the SDK (`simulate` /
`generate_and_simulate` via their ``save=`` flag). Lives here — not in
``cli.py`` — so ``api.py`` can reuse it without importing the CLI (which would
be circular, since ``cli`` imports ``api``).

Saved runs land in ``.evaluatorq/sim-runs/`` under collision-free
``<run-name>_<timestamp>.json`` names, which ``evaluatorq sim runs`` lists and
``evaluatorq dashboard .evaluatorq/sim-runs`` browses.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluatorq.common.run_store_dir import get_store_dir
from evaluatorq.simulation.types import AgentInfoSnapshot, SimulationRun

logger = logging.getLogger(__name__)

SIM_RUNS_DIR_NAME = Path('.evaluatorq') / 'sim-runs'


async def fetch_agent_info(agent_key: str) -> AgentInfoSnapshot | None:
    """Best-effort snapshot of an ORQ agent's configuration, for the run-store
    record. Never raises — no API key, network errors, and unknown agents all
    just result in ``None`` so a save can't be delayed or fail on this.

    Deliberately excludes the agent's system prompt / instructions.
    """
    api_key = os.getenv('ORQ_API_KEY')
    if not api_key:
        return None

    try:
        from evaluatorq.fetch_data import setup_orq_client

        orq_client = setup_orq_client(api_key)
        agent_data = await asyncio.to_thread(orq_client.agents.retrieve, agent_key=agent_key)

        agent_id = getattr(agent_data, '_id', None) or getattr(agent_data, 'id', None)
        model = getattr(agent_data, 'model', None)
        model_id = getattr(model, 'id', None) if model is not None else None

        # Entries with no resolvable name are dropped rather than persisted as
        # None (the card would render them as literal "None" chips).
        settings = getattr(agent_data, 'settings', None)
        raw_tools = getattr(settings, 'tools', None) if settings is not None else None
        tools = [n for t in raw_tools or [] if (n := getattr(t, 'display_name', None) or getattr(t, 'key', None))]

        raw_kbs = getattr(agent_data, 'knowledge_bases', None) or []
        knowledge_bases = [
            n for kb in raw_kbs if (n := kb if isinstance(kb, str) else getattr(kb, 'knowledge_id', None))
        ]

        raw_stores = getattr(agent_data, 'memory_stores', None) or []
        memory_stores = [n for ms in raw_stores if (n := ms if isinstance(ms, str) else getattr(ms, 'key', None))]

        raw_sub_agents = getattr(agent_data, 'team_of_agents', None) or []
        sub_agents = [
            n for a in raw_sub_agents if (n := a.get('key') if isinstance(a, dict) else getattr(a, 'key', None))
        ]

        from evaluatorq.dashboard.orq_links import orq_studio_url, studio_workspace_key

        workspace_id = getattr(agent_data, 'workspace_id', None)
        # Entity APIs expose workspace_id (a UUID). Studio links instead use
        # the configured ORQ_WORKSPACE key, captured here for future reports.
        workspace_key = studio_workspace_key(getattr(agent_data, 'workspace_key', None))
        base_url = os.getenv('ORQ_BASE_URL', 'https://my.orq.ai').rstrip('/')
        url = orq_studio_url(target_kind='agent', entity_id=agent_id, workspace_id=workspace_key, base_url=base_url)

        return {
            'key': agent_key,
            'id': agent_id,
            'role': getattr(agent_data, 'role', None),
            'description': getattr(agent_data, 'description', None),
            'model': model_id,
            'tools': tools,
            'knowledge_bases': knowledge_bases,
            'memory_stores': memory_stores,
            'sub_agents': sub_agents,
            'workspace_id': workspace_id,
            'workspace_key': workspace_key,
            'base_url': base_url,
            'url': url,
        }
    except Exception as exc:
        # Best-effort snapshot: a missing agent (404) or any transient error just
        # degrades to None. Log at debug with a concise message — the raw SDK
        # error repr dumps the full response (headers, body) and reads as alarming.
        logger.debug('Skipping agent_info for %r: %s', agent_key, type(exc).__name__)
        return None


def sanitise_run_name(name: str) -> str:
    sanitised = name.lower()
    sanitised = re.sub(r'[^a-z0-9_-]', '_', sanitised)
    sanitised = re.sub(r'_+', '_', sanitised)
    sanitised = sanitised.strip('_')
    return sanitised[:64] or 'sim'


def get_sim_runs_dir() -> Path:
    """Return the agent sim runs directory (``<store>/sim-runs``)."""
    return get_store_dir('sim-runs')


def build_simulation_run(
    *,
    run_name: str,
    mode: str,
    target_kind: str,
    evaluator_names: list[str],
    results: list[Any],
    target: str | None = None,
    target_model: str | None = None,
    max_turns: int | None = None,
    agent_info: AgentInfoSnapshot | None = None,
    run_id: str | None = None,
    experiment_url: str | None = None,
) -> SimulationRun:
    """Build the full ``SimulationRun`` report model from results.

    Aggregates per-scorer averages, guarding against non-numeric scores from a
    misbehaving evaluator so a single bad entry can't crash the build.
    """
    scorer_totals: dict[str, list[float]] = {}
    for result in results:
        scores: dict[str, float] = (result.metadata or {}).get('evaluator_scores', {})
        for scorer_name, score in scores.items():
            if isinstance(score, (int, float)):
                scorer_totals.setdefault(scorer_name, []).append(float(score))
            else:
                # Drop non-numeric scores so a misbehaving evaluator can't crash
                # the build — but log it, or the average silently reflects fewer
                # data points than the run produced.
                logger.warning('Dropping non-numeric score from scorer %r: %r', scorer_name, score)

    scorer_averages = {k: sum(v) / len(v) for k, v in scorer_totals.items() if v}

    return SimulationRun(
        run_name=run_name,
        created_at=datetime.now(tz=timezone.utc),
        mode=mode,  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
        target_kind=target_kind,  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
        target=target,
        target_model=target_model,
        max_turns=max_turns,
        agent_info=agent_info,
        run_id=run_id,
        experiment_url=experiment_url,
        evaluator_names=evaluator_names,
        total_results=len(results),
        scorer_averages=scorer_averages,
        results=results,
    )


def auto_save_run(*, run: SimulationRun, run_name: str) -> Path:
    """Persist a prebuilt ``SimulationRun`` to .evaluatorq/sim-runs/ under an
    auto-generated, collision-free `<name>_<timestamp>.json` filename."""
    runs_dir = get_sim_runs_dir()
    runs_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(tz=timezone.utc).strftime('%Y%m%d-%H%M%S')
    base = f'{sanitise_run_name(run_name)}_{ts}'
    payload = run.model_dump_json(indent=2)

    # Exclusive-create write: avoids the TOCTOU race between an exists() check
    # and a later write, and bounds the collision search.
    target_path = runs_dir / f'{base}.json'
    for counter in range(1000):
        try:
            with target_path.open('x', encoding='utf-8') as fh:
                _ = fh.write(payload)
        except FileExistsError:  # noqa: PERF203 — exclusive-create retry is the point
            target_path = runs_dir / f'{base}_{counter + 1:03d}.json'
        except OSError:
            # "x" created the file before the write failed (e.g. disk full) —
            # don't leave an empty/partial orphan behind.
            target_path.unlink(missing_ok=True)
            raise
        else:
            return target_path
    raise RuntimeError(f'Could not find a free run-store filename for {base!r} after 1000 attempts')


def write_report(run: SimulationRun, output: Path) -> None:
    """Write the full ``SimulationRun`` report JSON to an explicit path.

    Unlike :func:`auto_save_run` (auto-named, collision-avoiding, fixed dir),
    this honours the user-supplied path verbatim, creating parent dirs and
    **overwriting** any existing file. Prefer :func:`auto_save_run` for routine
    persistence — a fixed path here will clobber a prior run. Intentionally not
    part of the public package API; it backs the explicit ``report``/
    ``--report`` opt-in only.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(run.model_dump_json(indent=2), encoding='utf-8')

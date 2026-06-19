"""Run-store persistence for agent simulation.

Shared by the CLI (`evaluatorq sim run`) and the SDK (`simulate` /
`generate_and_simulate` via their ``save=`` flag). Lives here — not in
``cli.py`` — so ``api.py`` can reuse it without importing the CLI (which would
be circular, since ``cli`` imports ``api``).

Saved runs land in ``.evaluatorq/sim-runs/`` under collision-free
``<run-name>_<timestamp>.json`` names, which ``evaluatorq sim runs`` lists and
``evaluatorq sim ui`` opens.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluatorq.simulation.types import SimulationRun

logger = logging.getLogger(__name__)

SIM_RUNS_DIR_NAME = Path(".evaluatorq") / "sim-runs"


def sanitise_run_name(name: str) -> str:
    sanitised = name.lower()
    sanitised = re.sub(r"[^a-z0-9_-]", "_", sanitised)
    sanitised = re.sub(r"_+", "_", sanitised)
    sanitised = sanitised.strip("_")
    return sanitised[:64] or "sim"


def get_sim_runs_dir() -> Path:
    return Path.cwd() / SIM_RUNS_DIR_NAME


def build_simulation_run(
    *,
    run_name: str,
    mode: str,
    target_kind: str,
    evaluator_names: list[str],
    results: list[Any],
) -> SimulationRun:
    """Build the full ``SimulationRun`` report model from results.

    Aggregates per-scorer averages, guarding against non-numeric scores from a
    misbehaving evaluator so a single bad entry can't crash the build.
    """
    scorer_totals: dict[str, list[float]] = {}
    for result in results:
        scores: dict[str, float] = (result.metadata or {}).get("evaluator_scores", {})
        for scorer_name, score in scores.items():
            if isinstance(score, (int, float)):
                scorer_totals.setdefault(scorer_name, []).append(float(score))
            else:
                # Drop non-numeric scores so a misbehaving evaluator can't crash
                # the build — but log it, or the average silently reflects fewer
                # data points than the run produced.
                logger.warning("Dropping non-numeric score from scorer %r: %r", scorer_name, score)

    scorer_averages = {k: sum(v) / len(v) for k, v in scorer_totals.items() if v}

    return SimulationRun(
        run_name=run_name,
        created_at=datetime.now(tz=timezone.utc),
        mode=mode,  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
        target_kind=target_kind,  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
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

    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d-%H%M%S")
    base = f"{sanitise_run_name(run_name)}_{ts}"
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
        except OSError:
            # "x" created the file before the write failed (e.g. disk full) —
            # don't leave an empty/partial orphan behind.
            target_path.unlink(missing_ok=True)
            raise
        else:
            return target_path
    raise RuntimeError(
        f"Could not find a free run-store filename for {base!r} after 1000 attempts"
    )


def write_report(run: SimulationRun, output: Path) -> None:
    """Write the full ``SimulationRun`` report JSON to an explicit path.

    Unlike :func:`auto_save_run` (auto-named, collision-avoiding, fixed dir),
    this honours the user-supplied path verbatim, creating parent dirs and
    **overwriting** any existing file. Prefer :func:`auto_save_run` for routine
    persistence — a fixed path here will clobber a prior run. Intentionally not
    part of the public package API; it backs the explicit ``run_output``/
    ``--report-output`` opt-in only.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(run.model_dump_json(indent=2), encoding="utf-8")

"""Dataset export/import utilities for JSONL format."""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any, TypeVar

from evaluatorq.simulation.types import (
    CommunicationStyle,
    Datapoint,
    Persona,
    Scenario,
    SimulationResult,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_datapoints_to_jsonl(datapoints: list[Datapoint], output_path: str) -> None:
    """Export datapoints to JSONL format for orq.ai datasets."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    lines = []
    for dp in datapoints:
        lines.append(
            json.dumps(
                {
                    "inputs": {
                        "category": f"{dp.persona.name} - {dp.scenario.name}",
                        "first_message": dp.first_message,
                        "user_system_prompt": dp.user_system_prompt,
                        "persona": dp.persona.model_dump(mode="json"),
                        "scenario": dp.scenario.model_dump(mode="json"),
                    },
                    "expected_output": None,
                }
            )
        )
    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_results_to_jsonl(results: list[SimulationResult], output_path: str) -> None:
    """Export simulation results to JSONL format."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    lines = [r.model_dump_json() for r in results]
    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------


def load_datapoints_from_jsonl(input_path: str) -> list[Datapoint]:
    """Load datapoints from a JSONL file.

    Supports both the current format (with full persona/scenario objects) and a
    legacy format (with flat fields).
    """
    content = Path(input_path).read_text(encoding="utf-8")
    datapoints: list[Datapoint] = []

    for line in content.split("\n"):
        trimmed = line.strip()
        if not trimmed:
            continue

        try:
            data = json.loads(trimmed)
        except json.JSONDecodeError:
            logger.warning(
                "loadDatapointsFromJsonl: skipping malformed line: %s", trimmed[:80]
            )
            continue

        inputs = data.get("inputs", {})

        # Reconstruct persona
        if isinstance(inputs.get("persona"), dict):
            persona = Persona.model_validate(inputs["persona"])
        else:
            persona = Persona(
                name=inputs.get("persona_name", "Unknown"),
                patience=0.5,
                assertiveness=0.5,
                politeness=0.5,
                technical_level=0.5,
                communication_style=CommunicationStyle.casual,
                background=inputs.get("context", ""),
            )

        # Reconstruct scenario
        if isinstance(inputs.get("scenario"), dict):
            scenario = Scenario.model_validate(inputs["scenario"])
        else:
            scenario = Scenario(
                name=inputs.get("scenario_name", "Unknown"),
                goal=inputs.get("goal", ""),
                context=inputs.get("context", ""),
            )

        datapoints.append(
            Datapoint(
                id=f"dp_{uuid.uuid4().hex[:12]}",
                persona=persona,
                scenario=scenario,
                user_system_prompt=inputs.get("user_system_prompt", ""),
                first_message=inputs.get("first_message", ""),
            )
        )

    return datapoints


# ---------------------------------------------------------------------------
# String helpers
# ---------------------------------------------------------------------------


T = TypeVar("T")


def parse_jsonl(content: str, cls: type[T] | None = None) -> list[T | dict[str, Any]]:
    """Parse a JSONL string into a list of objects.

    If *cls* is a Pydantic ``BaseModel`` subclass, each line will be validated
    through ``model_validate``.  Otherwise lines are returned as plain dicts.
    """
    results: list[T | dict[str, Any]] = []
    for line in content.split("\n"):
        trimmed = line.strip()
        if not trimmed:
            continue
        try:
            data = json.loads(trimmed)
            if cls is not None and hasattr(cls, "model_validate"):
                results.append(cls.model_validate(data))  # pyright: ignore[reportAttributeAccessIssue]
            else:
                results.append(data)
        except json.JSONDecodeError:
            logger.warning("parseJsonl: skipping malformed line: %s", trimmed[:80])
    return results


def results_to_jsonl(
    results: list[dict[str, Datapoint | SimulationResult]],
) -> str:
    """Convert simulation results to JSONL string for dataset export."""
    lines = []
    for r in results:
        dp = r["datapoint"]
        result = r["result"]
        if not isinstance(dp, Datapoint):
            raise TypeError(f"Expected Datapoint, got {type(dp).__name__}")
        if not isinstance(result, SimulationResult):
            raise TypeError(f"Expected SimulationResult, got {type(result).__name__}")
        lines.append(
            json.dumps(
                {
                    "id": dp.id,
                    "persona": dp.persona.name,
                    "scenario": dp.scenario.name,
                    "first_message": dp.first_message,
                    "goal_achieved": result.goal_achieved,
                    "goal_completion_score": result.goal_completion_score,
                    "terminated_by": result.terminated_by.value,
                    "turn_count": result.turn_count,
                    "messages": [m.model_dump(mode="json") for m in result.messages],
                    "rules_broken": result.rules_broken,
                    "token_usage": result.token_usage.model_dump(mode="json"),
                    "turn_metrics": [
                        tm.model_dump(mode="json") for tm in result.turn_metrics
                    ],
                    "metadata": result.metadata,
                }
            )
        )
    return "\n".join(lines)

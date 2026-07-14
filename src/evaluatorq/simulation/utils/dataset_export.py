"""Dataset export/import utilities for JSONL format."""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any, TypeVar

from evaluatorq.simulation._datapoint_io import _as_obj
from evaluatorq.simulation.types import (
    CommunicationStyle,
    Persona,
    Scenario,
    SimulationDatapoint,
    SimulationResult,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def to_orq_dataset_rows(datapoints: list[SimulationDatapoint]) -> list[dict[str, Any]]:
    """Convert simulation datapoints to orq.ai dataset-row dicts.

    The orq datasets API rejects nested objects in ``inputs`` (each value must be
    a scalar), so ``persona``/``scenario`` are JSON-*stringified* here and
    ``expected_output`` is an empty string rather than ``null``. The read side
    (:func:`load_datapoints_from_jsonl`, ``_extract_single_datapoint``) accepts
    these stringified fields, so the round-trip holds.
    """
    return [
        {
            'inputs': {
                'category': f'{dp.persona.name} - {dp.scenario.name}',
                'first_message': dp.first_message or '',
                'user_system_prompt': dp.user_system_prompt or '',
                'persona': dp.persona.model_dump_json(),
                'scenario': dp.scenario.model_dump_json(),
            },
            'expected_output': '',
        }
        for dp in datapoints
    ]


def export_datapoints_to_jsonl(datapoints: list[SimulationDatapoint], output_path: str) -> None:
    """Export datapoints to JSONL format for orq.ai datasets (one row per line)."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row) for row in to_orq_dataset_rows(datapoints)]
    Path(output_path).write_text('\n'.join(lines) + '\n', encoding='utf-8')


def export_results_to_jsonl(results: list[SimulationResult], output_path: str) -> None:
    """Export simulation results to JSONL format."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    lines = [r.model_dump_json() for r in results]
    Path(output_path).write_text('\n'.join(lines) + '\n', encoding='utf-8')


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------


def load_datapoints_from_jsonl(input_path: str) -> list[SimulationDatapoint]:
    """Load datapoints from a JSONL file.

    Supports both the current format (with full persona/scenario objects) and a
    legacy format (with flat fields).
    """
    content = Path(input_path).read_text(encoding='utf-8')
    datapoints: list[SimulationDatapoint] = []

    for line in content.split('\n'):
        trimmed = line.strip()
        if not trimmed:
            continue

        try:
            data = json.loads(trimmed)
        except json.JSONDecodeError:
            logger.warning('loadDatapointsFromJsonl: skipping malformed line: %s', trimmed[:80])
            continue

        # Raw ``SimulationDatapoint`` line (what ``sim generate`` writes): top-level
        # persona/scenario dicts and an id. Parse directly so the id and all
        # fields round-trip unchanged.
        if isinstance(data.get('persona'), dict) and isinstance(data.get('scenario'), dict):
            datapoints.append(SimulationDatapoint.model_validate(data))
            continue

        inputs = data.get('inputs', {})

        # Reconstruct persona (accept dict or JSON-stringified object — the orq
        # dataset envelope stringifies nested objects, see to_orq_dataset_rows).
        persona_raw = _as_obj(inputs.get('persona'))
        if isinstance(persona_raw, dict):
            persona = Persona.model_validate(persona_raw)
        else:
            # A present-but-unparseable value is corruption, not the legacy flat
            # format (where 'persona' is simply absent) — don't discard it silently.
            if inputs.get('persona') is not None:
                logger.warning(
                    "load_datapoints_from_jsonl: 'persona' present but not a parseable object "
                    '(got %s); falling back to flat fields.',
                    type(persona_raw).__name__,
                )
            persona = Persona(
                name=inputs.get('persona_name', 'Unknown'),
                patience=0.5,
                assertiveness=0.5,
                politeness=0.5,
                technical_level=0.5,
                communication_style=CommunicationStyle.casual,
                background=inputs.get('context', ''),
            )

        # Reconstruct scenario (dict or JSON-stringified, as with persona above)
        scenario_raw = _as_obj(inputs.get('scenario'))
        if isinstance(scenario_raw, dict):
            scenario = Scenario.model_validate(scenario_raw)
        else:
            if inputs.get('scenario') is not None:
                logger.warning(
                    "load_datapoints_from_jsonl: 'scenario' present but not a parseable object "
                    '(got %s); falling back to flat fields.',
                    type(scenario_raw).__name__,
                )
            scenario = Scenario(
                name=inputs.get('scenario_name', 'Unknown'),
                goal=inputs.get('goal', ''),
                context=inputs.get('context', ''),
            )

        datapoints.append(
            SimulationDatapoint(
                id=f'dp_{uuid.uuid4().hex[:12]}',
                persona=persona,
                scenario=scenario,
                user_system_prompt=inputs.get('user_system_prompt', ''),
                first_message=inputs.get('first_message', ''),
            )
        )

    return datapoints


# ---------------------------------------------------------------------------
# String helpers
# ---------------------------------------------------------------------------


T = TypeVar('T')


def parse_jsonl(content: str, cls: type[T] | None = None) -> list[T | dict[str, Any]]:
    """Parse a JSONL string into a list of objects.

    If *cls* is a Pydantic ``BaseModel`` subclass, each line will be validated
    through ``model_validate``.  Otherwise lines are returned as plain dicts.
    """
    results: list[T | dict[str, Any]] = []
    for line in content.split('\n'):
        trimmed = line.strip()
        if not trimmed:
            continue
        try:
            data = json.loads(trimmed)
            if cls is not None and hasattr(cls, 'model_validate'):
                results.append(cls.model_validate(data))  # pyright: ignore[reportAttributeAccessIssue]
            else:
                results.append(data)
        except json.JSONDecodeError:
            logger.warning('parseJsonl: skipping malformed line: %s', trimmed[:80])
    return results


def results_to_jsonl(
    results: list[dict[str, SimulationDatapoint | SimulationResult]],
) -> str:
    """Convert simulation results to JSONL string for dataset export."""
    lines = []
    for r in results:
        dp = r['datapoint']
        result = r['result']
        if not isinstance(dp, SimulationDatapoint):
            raise TypeError(f'Expected SimulationDatapoint, got {type(dp).__name__}')
        if not isinstance(result, SimulationResult):
            raise TypeError(f'Expected SimulationResult, got {type(result).__name__}')
        lines.append(
            json.dumps({
                'id': dp.id,
                'persona': dp.persona.name,
                'scenario': dp.scenario.name,
                'first_message': dp.first_message,
                'goal_achieved': result.goal_achieved,
                'goal_completion_score': result.goal_completion_score,
                'terminated_by': result.terminated_by.value,
                'turn_count': result.turn_count,
                'messages': [m.model_dump(mode='json') for m in result.messages],
                'rules_broken': result.rules_broken,
                'token_usage': result.token_usage.model_dump(mode='json'),
                'turn_metrics': [tm.model_dump(mode='json') for tm in result.turn_metrics],
                'metadata': result.metadata,
            })
        )
    return '\n'.join(lines)

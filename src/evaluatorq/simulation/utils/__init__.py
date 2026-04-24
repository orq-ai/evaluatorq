"""Utility functions for the simulation module."""

from evaluatorq.common.sanitize import delimit
from evaluatorq.simulation.utils.dataset_export import (
    export_datapoints_to_jsonl,
    export_results_to_jsonl,
    load_datapoints_from_jsonl,
    parse_jsonl,
    results_to_jsonl,
)
from evaluatorq.simulation.utils.extract_json import extract_json_from_response
from evaluatorq.simulation.utils.prompt_builders import (
    build_datapoint_system_prompt,
    build_persona_system_prompt,
    build_scenario_user_context,
    generate_datapoint,
)

__all__ = [
    "build_datapoint_system_prompt",
    "build_persona_system_prompt",
    "build_scenario_user_context",
    "delimit",
    "export_datapoints_to_jsonl",
    "export_results_to_jsonl",
    "extract_json_from_response",
    "generate_datapoint",
    "load_datapoints_from_jsonl",
    "parse_jsonl",
    "results_to_jsonl",
]

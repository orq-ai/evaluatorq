"""Agent simulation integration for evaluatorq.

Provides tools to run multi-turn agent simulations with user simulator
and judge agents, convert results to OpenResponses format, and integrate
with the evaluatorq evaluation pipeline.

Example::

    from evaluatorq.simulation import simulate, generate_and_simulate
"""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING, Any

from evaluatorq.simulation.api import generate_and_simulate, simulate
from evaluatorq.simulation.types import DEFAULT_MODEL

logger = logging.getLogger(__name__)  # noqa: RUF067

if TYPE_CHECKING:
    from evaluatorq.simulation.adapters import (
        from_chat_completions,
        from_orq_deployment,
    )
    from evaluatorq.simulation.agents.base import AgentConfig, BaseAgent
    from evaluatorq.simulation.agents.judge import JudgeAgent
    from evaluatorq.simulation.agents.user_simulator import UserSimulatorAgent
    from evaluatorq.simulation.convert import to_open_responses
    from evaluatorq.simulation.evaluators import (
        SIMULATION_EVALUATORS,
        SimulationScorer,
        get_all_evaluators,
        get_evaluator,
    )
    from evaluatorq.simulation.generators import (
        DatapointGenerator,
        FirstMessageGenerator,
        PersonaGenerator,
        ScenarioGenerator,
    )
    from evaluatorq.simulation.quality.message_perturbation import (
        PerturbationType,
        apply_perturbation,
        apply_perturbations_batch,
        apply_random_perturbation,
    )
    from evaluatorq.simulation.runner.simulation import SimulationRunner, TargetAgent
    from evaluatorq.simulation.types import (
        ChatMessage,
        CommunicationStyle,
        ConversationStrategy,
        Criterion,
        CulturalContext,
        Datapoint,
        EmotionalArc,
        InputFormat,
        Judgment,
        Message,
        Persona,
        Scenario,
        SimulationResult,
        StartingEmotion,
        TerminatedBy,
        TokenUsage,
        TurnMetrics,
    )
    from evaluatorq.simulation.utils.dataset_export import (
        export_datapoints_to_jsonl,
        export_results_to_jsonl,
        load_datapoints_from_jsonl,
        parse_jsonl,
        results_to_jsonl,
    )
    from evaluatorq.simulation.utils.prompt_builders import (
        build_datapoint_system_prompt,
        build_persona_system_prompt,
        build_scenario_user_context,
        generate_datapoint,
    )
    from evaluatorq.simulation.wrap_agent import wrap_simulation_agent


_LAZY_IMPORTS: dict[str, tuple[str, str]] = {  # noqa: RUF067
    "from_chat_completions": ("evaluatorq.simulation.adapters", "from_chat_completions"),
    "from_orq_deployment": ("evaluatorq.simulation.adapters", "from_orq_deployment"),
    "AgentConfig": ("evaluatorq.simulation.agents.base", "AgentConfig"),
    "BaseAgent": ("evaluatorq.simulation.agents.base", "BaseAgent"),
    "JudgeAgent": ("evaluatorq.simulation.agents.judge", "JudgeAgent"),
    "UserSimulatorAgent": (
        "evaluatorq.simulation.agents.user_simulator",
        "UserSimulatorAgent",
    ),
    "to_open_responses": ("evaluatorq.simulation.convert", "to_open_responses"),
    "SIMULATION_EVALUATORS": (
        "evaluatorq.simulation.evaluators",
        "SIMULATION_EVALUATORS",
    ),
    "SimulationScorer": ("evaluatorq.simulation.evaluators", "SimulationScorer"),
    "get_all_evaluators": ("evaluatorq.simulation.evaluators", "get_all_evaluators"),
    "get_evaluator": ("evaluatorq.simulation.evaluators", "get_evaluator"),
    "DatapointGenerator": (
        "evaluatorq.simulation.generators",
        "DatapointGenerator",
    ),
    "FirstMessageGenerator": (
        "evaluatorq.simulation.generators",
        "FirstMessageGenerator",
    ),
    "PersonaGenerator": ("evaluatorq.simulation.generators", "PersonaGenerator"),
    "ScenarioGenerator": ("evaluatorq.simulation.generators", "ScenarioGenerator"),
    "PerturbationType": (
        "evaluatorq.simulation.quality.message_perturbation",
        "PerturbationType",
    ),
    "apply_perturbation": (
        "evaluatorq.simulation.quality.message_perturbation",
        "apply_perturbation",
    ),
    "apply_perturbations_batch": (
        "evaluatorq.simulation.quality.message_perturbation",
        "apply_perturbations_batch",
    ),
    "apply_random_perturbation": (
        "evaluatorq.simulation.quality.message_perturbation",
        "apply_random_perturbation",
    ),
    "SimulationRunner": (
        "evaluatorq.simulation.runner.simulation",
        "SimulationRunner",
    ),
    "TargetAgent": ("evaluatorq.simulation.runner.simulation", "TargetAgent"),
    "DEFAULT_MODEL": ("evaluatorq.simulation.types", "DEFAULT_MODEL"),
    "ChatMessage": ("evaluatorq.simulation.types", "ChatMessage"),
    "CommunicationStyle": ("evaluatorq.simulation.types", "CommunicationStyle"),
    "ConversationStrategy": (
        "evaluatorq.simulation.types",
        "ConversationStrategy",
    ),
    "Criterion": ("evaluatorq.simulation.types", "Criterion"),
    "CulturalContext": ("evaluatorq.simulation.types", "CulturalContext"),
    "Datapoint": ("evaluatorq.simulation.types", "Datapoint"),
    "EmotionalArc": ("evaluatorq.simulation.types", "EmotionalArc"),
    "InputFormat": ("evaluatorq.simulation.types", "InputFormat"),
    "Judgment": ("evaluatorq.simulation.types", "Judgment"),
    "Message": ("evaluatorq.simulation.types", "Message"),
    "Persona": ("evaluatorq.simulation.types", "Persona"),
    "Scenario": ("evaluatorq.simulation.types", "Scenario"),
    "SimulationResult": ("evaluatorq.simulation.types", "SimulationResult"),
    "StartingEmotion": ("evaluatorq.simulation.types", "StartingEmotion"),
    "TerminatedBy": ("evaluatorq.simulation.types", "TerminatedBy"),
    "TokenUsage": ("evaluatorq.simulation.types", "TokenUsage"),
    "TurnMetrics": ("evaluatorq.simulation.types", "TurnMetrics"),
    "export_datapoints_to_jsonl": (
        "evaluatorq.simulation.utils.dataset_export",
        "export_datapoints_to_jsonl",
    ),
    "export_results_to_jsonl": (
        "evaluatorq.simulation.utils.dataset_export",
        "export_results_to_jsonl",
    ),
    "load_datapoints_from_jsonl": (
        "evaluatorq.simulation.utils.dataset_export",
        "load_datapoints_from_jsonl",
    ),
    "parse_jsonl": ("evaluatorq.simulation.utils.dataset_export", "parse_jsonl"),
    "results_to_jsonl": (
        "evaluatorq.simulation.utils.dataset_export",
        "results_to_jsonl",
    ),
    "build_datapoint_system_prompt": (
        "evaluatorq.simulation.utils.prompt_builders",
        "build_datapoint_system_prompt",
    ),
    "build_persona_system_prompt": (
        "evaluatorq.simulation.utils.prompt_builders",
        "build_persona_system_prompt",
    ),
    "build_scenario_user_context": (
        "evaluatorq.simulation.utils.prompt_builders",
        "build_scenario_user_context",
    ),
    "generate_datapoint": (
        "evaluatorq.simulation.utils.prompt_builders",
        "generate_datapoint",
    ),
    "wrap_simulation_agent": (
        "evaluatorq.simulation.wrap_agent",
        "wrap_simulation_agent",
    ),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_IMPORTS[name]
    value = getattr(importlib.import_module(module_name), attr_name)
    globals()[name] = value
    return value


__all__ = [
    # Types
    "DEFAULT_MODEL",
    # Evaluators
    "SIMULATION_EVALUATORS",
    # Agents
    "AgentConfig",
    "BaseAgent",
    "ChatMessage",
    "CommunicationStyle",
    "ConversationStrategy",
    "Criterion",
    "CulturalContext",
    "Datapoint",
    # Generators
    "DatapointGenerator",
    "EmotionalArc",
    "FirstMessageGenerator",
    "InputFormat",
    "JudgeAgent",
    "Judgment",
    "Message",
    "Persona",
    "PersonaGenerator",
    # Quality
    "PerturbationType",
    "Scenario",
    "ScenarioGenerator",
    "SimulationResult",
    # Runner
    "SimulationRunner",
    "SimulationScorer",
    "StartingEmotion",
    "TargetAgent",
    "TerminatedBy",
    "TokenUsage",
    "TurnMetrics",
    "UserSimulatorAgent",
    "apply_perturbation",
    "apply_perturbations_batch",
    "apply_random_perturbation",
    # Utils
    "build_datapoint_system_prompt",
    "build_persona_system_prompt",
    "build_scenario_user_context",
    "export_datapoints_to_jsonl",
    "export_results_to_jsonl",
    # Adapters
    "from_chat_completions",
    "from_orq_deployment",
    "generate_and_simulate",
    "generate_datapoint",
    "get_all_evaluators",
    "get_evaluator",
    "load_datapoints_from_jsonl",
    "parse_jsonl",
    "results_to_jsonl",
    # High-level API
    "simulate",
    # Conversion
    "to_open_responses",
    # Job wrapper
    "wrap_simulation_agent",
]

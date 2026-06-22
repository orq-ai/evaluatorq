"""Agent simulation integration for evaluatorq.

Provides tools to run multi-turn agent simulations with user simulator
and judge agents, convert results to OpenResponses format, and integrate
with the evaluatorq evaluation pipeline.

Example::

    from evaluatorq.simulation import simulate, generate_and_simulate

Tracing & PII:
    Simulation emits OpenTelemetry spans for every LLM call, including the
    message content (conversation turns) by default so the Orq dashboard can
    render input/output panels. Two env vars control this (shared with red
    teaming via ``evaluatorq.common.tracing``):

    - ``EVALUATORQ_CAPTURE_MESSAGE_CONTENT`` (default ``true``) — set to
      ``false`` / ``0`` to keep raw message text (inputs and outputs) off
      spans while still recording token usage, model, and latency. Use when
      exporting to a third-party backend or when content may contain PII.
    - ``EVALUATORQ_SPAN_MAX_TEXT_CHARS`` (default: capture all) — max
      characters of message text (inputs and outputs) per span attribute
      before truncation. Set a positive integer (e.g. ``8192``) to cap;
      ``-1`` / ``0`` / unset all mean capture all.
"""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING, Any

from evaluatorq.simulation.api import (
    SimulationDroppedError,
    generate,
    generate_and_simulate,
    simulate,
)
from evaluatorq.simulation.types import DEFAULT_MODEL

logger = logging.getLogger(__name__)  # noqa: RUF067

if TYPE_CHECKING:
    from evaluatorq.contracts import AgentTarget, LLMCallConfig, TokenUsage
    from evaluatorq.integrations.callable_integration import CallableTarget
    from evaluatorq.integrations.langgraph_integration import LangGraphTarget
    from evaluatorq.integrations.openai_agents_integration import OpenAIAgentTarget
    from evaluatorq.integrations.vercel_ai_sdk_integration import VercelAISdkTarget
    from evaluatorq.openresponses.target import OrqResponsesTarget
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
    from evaluatorq.simulation.exceptions import (
        SimulationCancelledError,
        SimulationError,
    )
    from evaluatorq.simulation.generators import (
        DatapointGenerator,
        FirstMessageGenerator,
        PersonaGenerator,
        ScenarioGenerator,
    )
    from evaluatorq.simulation.hooks import (
        DefaultHooks,
        RichHooks,
        SimulationHooks,
        SimulationRunMeta,
    )
    from evaluatorq.simulation.quality.message_perturbation import (
        PerturbationType,
        apply_perturbation,
        apply_perturbations_batch,
        apply_random_perturbation,
    )
    from evaluatorq.simulation.runner.simulation import SimulationRunner
    from evaluatorq.simulation.types import (
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
        SimulationRun,
        StartingEmotion,
        TerminatedBy,
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
        generate_datapoint,
    )
    from evaluatorq.simulation.utils.run_store import (
        SIM_RUNS_DIR_NAME,
        auto_save_run,
        build_simulation_run,
        get_sim_runs_dir,
    )
    from evaluatorq.simulation.wrap_agent import wrap_simulation_agent


_LAZY_IMPORTS: dict[str, tuple[str, str]] = {  # noqa: RUF067
    'LLMCallConfig': ('evaluatorq.contracts', 'LLMCallConfig'),
    'from_chat_completions': ('evaluatorq.simulation.adapters', 'from_chat_completions'),
    'from_orq_deployment': ('evaluatorq.simulation.adapters', 'from_orq_deployment'),
    'AgentConfig': ('evaluatorq.simulation.agents.base', 'AgentConfig'),
    'BaseAgent': ('evaluatorq.simulation.agents.base', 'BaseAgent'),
    'OrqResponsesTarget': ('evaluatorq.openresponses.target', 'OrqResponsesTarget'),
    'JudgeAgent': ('evaluatorq.simulation.agents.judge', 'JudgeAgent'),
    'UserSimulatorAgent': (
        'evaluatorq.simulation.agents.user_simulator',
        'UserSimulatorAgent',
    ),
    'to_open_responses': ('evaluatorq.simulation.convert', 'to_open_responses'),
    'SIMULATION_EVALUATORS': (
        'evaluatorq.simulation.evaluators',
        'SIMULATION_EVALUATORS',
    ),
    'SimulationScorer': ('evaluatorq.simulation.evaluators', 'SimulationScorer'),
    'get_all_evaluators': ('evaluatorq.simulation.evaluators', 'get_all_evaluators'),
    'get_evaluator': ('evaluatorq.simulation.evaluators', 'get_evaluator'),
    'DatapointGenerator': (
        'evaluatorq.simulation.generators',
        'DatapointGenerator',
    ),
    'FirstMessageGenerator': (
        'evaluatorq.simulation.generators',
        'FirstMessageGenerator',
    ),
    'PersonaGenerator': ('evaluatorq.simulation.generators', 'PersonaGenerator'),
    'ScenarioGenerator': ('evaluatorq.simulation.generators', 'ScenarioGenerator'),
    'PerturbationType': (
        'evaluatorq.simulation.quality.message_perturbation',
        'PerturbationType',
    ),
    'apply_perturbation': (
        'evaluatorq.simulation.quality.message_perturbation',
        'apply_perturbation',
    ),
    'apply_perturbations_batch': (
        'evaluatorq.simulation.quality.message_perturbation',
        'apply_perturbations_batch',
    ),
    'apply_random_perturbation': (
        'evaluatorq.simulation.quality.message_perturbation',
        'apply_random_perturbation',
    ),
    'SimulationRunner': (
        'evaluatorq.simulation.runner.simulation',
        'SimulationRunner',
    ),
    'SimulationHooks': ('evaluatorq.simulation.hooks', 'SimulationHooks'),
    'DefaultHooks': ('evaluatorq.simulation.hooks', 'DefaultHooks'),
    'RichHooks': ('evaluatorq.simulation.hooks', 'RichHooks'),
    'SimulationRunMeta': ('evaluatorq.simulation.hooks', 'SimulationRunMeta'),
    'SimulationError': ('evaluatorq.simulation.exceptions', 'SimulationError'),
    'SimulationCancelledError': (
        'evaluatorq.simulation.exceptions',
        'SimulationCancelledError',
    ),
    'AgentTarget': ('evaluatorq.contracts', 'AgentTarget'),
    'CallableTarget': ('evaluatorq.integrations.callable_integration', 'CallableTarget'),
    'LangGraphTarget': ('evaluatorq.integrations.langgraph_integration', 'LangGraphTarget'),
    'OpenAIAgentTarget': ('evaluatorq.integrations.openai_agents_integration', 'OpenAIAgentTarget'),
    'VercelAISdkTarget': ('evaluatorq.integrations.vercel_ai_sdk_integration', 'VercelAISdkTarget'),
    'DEFAULT_MODEL': ('evaluatorq.simulation.types', 'DEFAULT_MODEL'),
    'CommunicationStyle': ('evaluatorq.simulation.types', 'CommunicationStyle'),
    'ConversationStrategy': (
        'evaluatorq.simulation.types',
        'ConversationStrategy',
    ),
    'Criterion': ('evaluatorq.simulation.types', 'Criterion'),
    'CulturalContext': ('evaluatorq.simulation.types', 'CulturalContext'),
    'Datapoint': ('evaluatorq.simulation.types', 'Datapoint'),
    'EmotionalArc': ('evaluatorq.simulation.types', 'EmotionalArc'),
    'InputFormat': ('evaluatorq.simulation.types', 'InputFormat'),
    'Judgment': ('evaluatorq.simulation.types', 'Judgment'),
    'Message': ('evaluatorq.simulation.types', 'Message'),
    'Persona': ('evaluatorq.simulation.types', 'Persona'),
    'Scenario': ('evaluatorq.simulation.types', 'Scenario'),
    'SimulationResult': ('evaluatorq.simulation.types', 'SimulationResult'),
    'SimulationRun': ('evaluatorq.simulation.types', 'SimulationRun'),
    'StartingEmotion': ('evaluatorq.simulation.types', 'StartingEmotion'),
    'TerminatedBy': ('evaluatorq.simulation.types', 'TerminatedBy'),
    'TokenUsage': ('evaluatorq.contracts', 'TokenUsage'),
    'TurnMetrics': ('evaluatorq.simulation.types', 'TurnMetrics'),
    'export_datapoints_to_jsonl': (
        'evaluatorq.simulation.utils.dataset_export',
        'export_datapoints_to_jsonl',
    ),
    'export_results_to_jsonl': (
        'evaluatorq.simulation.utils.dataset_export',
        'export_results_to_jsonl',
    ),
    'load_datapoints_from_jsonl': (
        'evaluatorq.simulation.utils.dataset_export',
        'load_datapoints_from_jsonl',
    ),
    'parse_jsonl': ('evaluatorq.simulation.utils.dataset_export', 'parse_jsonl'),
    'results_to_jsonl': (
        'evaluatorq.simulation.utils.dataset_export',
        'results_to_jsonl',
    ),
    'SIM_RUNS_DIR_NAME': (
        'evaluatorq.simulation.utils.run_store',
        'SIM_RUNS_DIR_NAME',
    ),
    'auto_save_run': ('evaluatorq.simulation.utils.run_store', 'auto_save_run'),
    'build_simulation_run': (
        'evaluatorq.simulation.utils.run_store',
        'build_simulation_run',
    ),
    'get_sim_runs_dir': ('evaluatorq.simulation.utils.run_store', 'get_sim_runs_dir'),
    'generate_datapoint': (
        'evaluatorq.simulation.utils.prompt_builders',
        'generate_datapoint',
    ),
    'wrap_simulation_agent': (
        'evaluatorq.simulation.wrap_agent',
        'wrap_simulation_agent',
    ),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')

    module_name, attr_name = _LAZY_IMPORTS[name]
    value = getattr(importlib.import_module(module_name), attr_name)
    globals()[name] = value
    return value


__all__ = [
    # Types
    'DEFAULT_MODEL',
    # Evaluators
    'SIMULATION_EVALUATORS',
    # Utils
    'SIM_RUNS_DIR_NAME',
    # Agents
    'AgentConfig',
    'AgentTarget',
    'BaseAgent',
    'CallableTarget',
    'CommunicationStyle',
    'ConversationStrategy',
    'Criterion',
    'CulturalContext',
    'Datapoint',
    # Generators
    'DatapointGenerator',
    # Hooks
    'DefaultHooks',
    'EmotionalArc',
    'FirstMessageGenerator',
    'InputFormat',
    'JudgeAgent',
    'Judgment',
    # Config (re-exported for user convenience)
    'LLMCallConfig',
    'LangGraphTarget',
    'Message',
    # Target implementations
    'OpenAIAgentTarget',
    'OrqResponsesTarget',
    'Persona',
    'PersonaGenerator',
    # Quality
    'PerturbationType',
    'RichHooks',
    'Scenario',
    'ScenarioGenerator',
    # Exceptions
    'SimulationCancelledError',
    'SimulationDroppedError',
    'SimulationError',
    'SimulationHooks',
    'SimulationResult',
    'SimulationRun',
    # Runner
    'SimulationRunMeta',
    'SimulationRunner',
    'SimulationScorer',
    'StartingEmotion',
    'TerminatedBy',
    'TokenUsage',
    'TurnMetrics',
    'UserSimulatorAgent',
    'VercelAISdkTarget',
    'apply_perturbation',
    'apply_perturbations_batch',
    'apply_random_perturbation',
    'auto_save_run',
    'build_simulation_run',
    'export_datapoints_to_jsonl',
    'export_results_to_jsonl',
    # Adapters
    'from_chat_completions',
    'from_orq_deployment',
    'generate',
    'generate_and_simulate',
    'generate_datapoint',
    'get_all_evaluators',
    'get_evaluator',
    'get_sim_runs_dir',
    'load_datapoints_from_jsonl',
    'parse_jsonl',
    'results_to_jsonl',
    # High-level API
    'simulate',
    # Conversion
    'to_open_responses',
    # Job wrapper
    'wrap_simulation_agent',
]

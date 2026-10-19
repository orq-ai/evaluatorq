"""Deprecated compatibility imports for simulation contracts.

Use `evaluatorq.simulation.contracts` for new code.
"""

from __future__ import annotations

import inspect as _inspect
import warnings


def _warning_stacklevel() -> int:
    frame = _inspect.currentframe()
    if frame is None:
        return 2
    frame = frame.f_back
    stacklevel = 1
    while frame is not None:
        filename = frame.f_code.co_filename
        module = frame.f_globals.get('__name__', '')
        # warnings.warn also skips bootstrap frames when advancing stacklevel.
        if 'importlib' in filename and '_bootstrap' in filename:
            frame = frame.f_back
            continue
        if module not in (__name__, __package__) and not module.startswith('importlib'):
            break
        stacklevel += 1
        frame = frame.f_back
    return stacklevel


warnings.warn(
    'evaluatorq.simulation.types is deprecated; import from evaluatorq.simulation.contracts instead.',
    DeprecationWarning,
    stacklevel=_warning_stacklevel(),
)

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictStr, ValidationError, computed_field
from typing_extensions import TypedDict

from evaluatorq.contracts import (
    DEFAULT_PIPELINE_MODEL,
    Message,
    ResponseTrace,
    RunSummary,
    StrEnum,
    TokenUsage,
)
from evaluatorq.simulation.contracts import (
    CRITERION_ID_PATTERN,
    CULTURAL_CONTEXT_INSTRUCTIONS,
    DEFAULT_EVALUATOR_NAMES,
    DEFAULT_MAX_TURNS,
    DEFAULT_MODEL,
    DEFAULT_RUN_NAME,
    EMOTIONAL_ARC_INSTRUCTIONS,
    INPUT_FORMAT_INSTRUCTIONS,
    STRATEGY_INSTRUCTIONS,
    AgentInfoSnapshot,
    CommunicationStyle,
    ConversationStrategy,
    CriteriaMeta,
    CriteriaRow,
    Criterion,
    CriterionVerdict,
    CulturalContext,
    EmotionalArc,
    InputFormat,
    Judgment,
    Persona,
    Scenario,
    SimulationDatapoint,
    SimulationEntry,
    SimulationRecommendation,
    SimulationResult,
    SimulationRun,
    StartingEmotion,
    TerminatedBy,
    TranscriptMessage,
    TurnMetrics,
    criterion_id_for,
    parse_criteria_meta,
)

__all__ = [
    'CRITERION_ID_PATTERN',
    'CULTURAL_CONTEXT_INSTRUCTIONS',
    'DEFAULT_EVALUATOR_NAMES',
    'DEFAULT_MAX_TURNS',
    'DEFAULT_MODEL',
    'DEFAULT_PIPELINE_MODEL',
    'DEFAULT_RUN_NAME',
    'EMOTIONAL_ARC_INSTRUCTIONS',
    'INPUT_FORMAT_INSTRUCTIONS',
    'STRATEGY_INSTRUCTIONS',
    'AgentInfoSnapshot',
    'Any',
    'BaseModel',
    'CommunicationStyle',
    'ConfigDict',
    'ConversationStrategy',
    'CriteriaMeta',
    'CriteriaRow',
    'Criterion',
    'CriterionVerdict',
    'CulturalContext',
    'EmotionalArc',
    'Field',
    'InputFormat',
    'Judgment',
    'Literal',
    'Message',
    'Persona',
    'ResponseTrace',
    'RunSummary',
    'Scenario',
    'SimulationDatapoint',
    'SimulationEntry',
    'SimulationRecommendation',
    'SimulationResult',
    'SimulationRun',
    'StartingEmotion',
    'StrEnum',
    'StrictBool',
    'StrictStr',
    'TerminatedBy',
    'TokenUsage',
    'TranscriptMessage',
    'TurnMetrics',
    'TypedDict',
    'ValidationError',
    'computed_field',
    'criterion_id_for',
    'datetime',
    'parse_criteria_meta',
]

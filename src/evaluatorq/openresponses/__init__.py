"""
OpenResponses types for evaluatorq.

These types represent the OpenResponses format, an industry standard for
representing LLM agent interactions including messages, function calls,
and their outputs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from evaluatorq.openresponses.dataset import (
    append_assistant_turn,
    append_user_followup,
    assistant_input_item,
    build_openresponses_request,
    load_openresponses_dataset,
    messages_from_openresponses_input,
    orchestrator_result_to_openresponses_input,
    redteam_sample_from_openresponses,
    system_input_item,
    turns_to_openresponses_input,
    user_input_item,
)
from evaluatorq.openresponses.types import (
    FunctionCall,
    FunctionCallDict,
    FunctionCallOutput,
    FunctionCallOutputDict,
    Message,
    MessageDict,
    OutputTextContent,
    ResponseResource,
    ResponseResourceDict,
    TextField,
    TextFieldDict,
    Usage,
    UsageDict,
)

__all__ = [
    'FunctionCall',
    'FunctionCallDict',
    'FunctionCallOutput',
    'FunctionCallOutputDict',
    'Message',
    'MessageDict',
    'OrqResponsesTarget',
    'OutputTextContent',
    'ResponseResource',
    'ResponseResourceDict',
    'TextField',
    'TextFieldDict',
    'Usage',
    'UsageDict',
    'append_assistant_turn',
    'append_user_followup',
    'assistant_input_item',
    'build_openresponses_request',
    'load_openresponses_dataset',
    'messages_from_openresponses_input',
    'orchestrator_result_to_openresponses_input',
    'redteam_sample_from_openresponses',
    'system_input_item',
    'turns_to_openresponses_input',
    'user_input_item',
]


if TYPE_CHECKING:
    from evaluatorq.openresponses.target import OrqResponsesTarget


def __getattr__(name: str) -> Any:
    # Lazy: target.py imports evaluatorq.contracts, which imports this package.
    if name == 'OrqResponsesTarget':
        from evaluatorq.openresponses.target import OrqResponsesTarget

        return OrqResponsesTarget
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')

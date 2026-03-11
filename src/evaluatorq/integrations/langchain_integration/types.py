"""Type definitions for LangChain integration."""

from collections.abc import Callable
from typing import TypedDict

from evaluatorq.types import DataPoint


class AgentJobOptions(TypedDict, total=False):
    """Options for creating an evaluatorq Job from a LangChain agent.

    Attributes:
        name: The name of the job (defaults to "agent").
        prompt_key: The key in data.inputs to use as the prompt (defaults to "prompt").
        instructions: System instructions to prepend to the messages sent to the agent.
            Can be a static string or a callable that receives the data point and returns the instructions.
    """

    name: str
    prompt_key: str
    instructions: str | Callable[[DataPoint], str]

"""Type definitions for LangChain integration."""

from typing import TypedDict


class AgentJobOptions(TypedDict, total=False):
    """Options for creating an evaluatorq Job from a LangChain agent.

    Attributes:
        name: The name of the job (defaults to "agent").
        prompt_key: The key in data.inputs to use as the prompt (defaults to "prompt").
    """

    name: str
    prompt_key: str

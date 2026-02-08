"""LangChain integration for evaluatorq.

This module provides a wrapper to convert LangChain agent outputs
to OpenResponses format for use with the evaluatorq framework.
"""

from .convert import convert_to_open_responses, generate_item_id
from .types import AgentJobOptions
from .wrap_agent import (
    extract_tools_from_agent,
    wrap_langchain_agent,
    wrap_langgraph_agent,
)

__all__ = [
    "AgentJobOptions",
    "convert_to_open_responses",
    "extract_tools_from_agent",
    "generate_item_id",
    "wrap_langchain_agent",
    "wrap_langgraph_agent",
]

import asyncio
import re
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from evaluatorq import ScorerParameter, evaluatorq
from evaluatorq.integrations.langchain_integration import wrap_langgraph_agent

_ = load_dotenv()


# Define state
class AgentState(TypedDict):
    messages: Annotated[list[AIMessage], add_messages]


# Define tools
@tool
def weather(location: str) -> dict[str, str | int]:
    """Get the weather in a location (in Fahrenheit)"""
    import random

    return {"location": location, "temperature": 72 + random.randint(-10, 10)}


@tool
def convert_fahrenheit_to_celsius(temperature: float) -> dict[str, float]:
    """Convert temperature from Fahrenheit to Celsius"""
    return {"celsius": round((temperature - 32) * (5 / 9))}


tools = [weather, convert_fahrenheit_to_celsius]


# Define nodes
model = ChatOpenAI(model="gpt-4o").bind_tools(tools)


def call_model(state: AgentState) -> dict[str, list[AIMessage]]:
    response = model.invoke(state["messages"])
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END


# Build graph
graph = StateGraph(AgentState)
graph.add_node("agent", call_model)
graph.add_node("tools", ToolNode(tools))

graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")

agent = graph.compile()


async def has_temperature_scorer(params: ScorerParameter) -> dict[str, int | str]:
    output = params["output"]
    if not isinstance(output, dict):
        return {"value": 0, "explanation": "Output is not a dict"}

    message = next(
        (item for item in output.get("output", []) if item.get("type") == "message"),
        None,
    )
    text = ""
    if message:
        text_content = next(
            (c for c in message.get("content", []) if c.get("type") == "output_text"),
            None,
        )
        text = text_content.get("text", "") if text_content else ""

    has_temp = bool(re.search(r"\d+", text))
    return {
        "value": 1 if has_temp else 0,
        "explanation": "Has temperature" if has_temp else "No temperature",
    }


async def main():
    _ = await evaluatorq(
        "langgraph-agent-test",
        data=[
            {"inputs": {"prompt": "What is the weather in San Francisco?"}},
            {"inputs": {"prompt": "What is the weather in New York?"}},
        ],
        jobs=[wrap_langgraph_agent(agent)],
        evaluators=[{"name": "has-temperature", "scorer": has_temperature_scorer}],
        parallelism=2,
    )


if __name__ == "__main__":
    asyncio.run(main())

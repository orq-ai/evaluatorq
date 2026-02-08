import asyncio
import re

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from evaluatorq import ScorerParameter, evaluatorq
from evaluatorq.integrations.langchain_integration import wrap_langchain_agent

_ = load_dotenv()


@tool
def weather(location: str) -> dict[str, str | int]:
    """Get the weather in a location (in Fahrenheit)"""
    import random
    return {"location": location, "temperature": 72 + random.randint(-10, 10)}


@tool
def convert_fahrenheit_to_celsius(temperature: float) -> dict[str, float]:
    """Convert temperature from Fahrenheit to Celsius"""
    return {"celsius": round((temperature - 32) * (5 / 9))}


model = ChatOpenAI(model="gpt-4o")
agent = create_agent(model, tools=[weather, convert_fahrenheit_to_celsius])


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
        "agent-test",
        data=[
            {"inputs": {"prompt": "What is the weather in San Francisco?"}},
            {"inputs": {"prompt": "What is the weather in New York?"}},
        ],
        jobs=[wrap_langchain_agent(agent)],
        evaluators=[{"name": "has-temperature", "scorer": has_temperature_scorer}],
        parallelism=2,
    )


if __name__ == "__main__":
    asyncio.run(main())

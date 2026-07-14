"""BYO-agent: wrap any async fn(messages) -> str and simulate against it."""

import asyncio

from openai import AsyncOpenAI

from evaluatorq.simulation import generate, simulate

client = AsyncOpenAI()


# your agent: any async callable taking OpenAI-style messages, returning a string
async def my_agent(messages):
    resp = await client.chat.completions.create(model='gpt-5.6-luna', messages=messages)
    return resp.choices[0].message.content


async def main():
    # 1. freeze personas × scenarios from a description
    datapoints = await generate(
        agent_description='Credit-card support agent',
        num_personas=3,
        num_scenarios=3,
    )

    # 2. run the sim against your agent, score every turn
    results = await simulate(target=my_agent, datapoints=datapoints)
    return results


if __name__ == '__main__':
    asyncio.run(main())

# the SDK: wrap any async fn(messages) -> str
from evaluatorq.simulation import generate, simulate


async def my_agent(messages):
    resp = await client.chat.completions.create(model='gpt-5.6-luna', messages=messages)
    return resp.choices[0].message.content


# 1. freeze personas × scenarios from a description
datapoints = await generate(
    agent_description='Credit-card support agent',
    num_personas=3,
    num_scenarios=3,
)


def my_agent2(messages):
    return messages[-1]['content']


# 2. run the sim against your agent, score every turn
results = await simulate(
    target=my_agent,
    datapoints=datapoints,
)

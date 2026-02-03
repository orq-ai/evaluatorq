"""
Country Unit Test Example

A simple example demonstrating how to quickly assemble an evaluation
as a "unit test" for a deployment using a dataset from the platform.

This example:
- Fetches the "countries" dataset from the Orq platform
- Calls the `unit_test_countries` deployment for each country
- Validates responses contain the expected capital city (case-insensitive)

Prerequisites:
  - Set ORQ_API_KEY environment variable

Usage:
  ORQ_API_KEY=your-key python examples/country_unit_test.py
"""

import asyncio

from evaluatorq import (
    DataPoint,
    DatasetIdInput,
    evaluatorq,
    invoke,
    job,
    string_contains_evaluator,
)

DATASET_ID = "01KE9KKAB119PGHXBXJX9D7DCT"
DEPLOYMENT_KEY = "unit_test_countries"


@job("country-lookup")
async def country_lookup_job(data: DataPoint, _row: int) -> str:
    """Job that calls the deployment with the country input."""
    country = str(data.inputs.get("country", ""))

    response = await invoke(DEPLOYMENT_KEY, inputs={"country": country})

    return response


async def run():
    """Run the country unit test evaluation."""
    print("\n🧪 Country Unit Test\n")
    print(f"Dataset: countries ({DATASET_ID})")
    print(f"Deployment: {DEPLOYMENT_KEY}")
    print("------------------------------------------\n")

    results = await evaluatorq(
        "country-unit-test",
        data=DatasetIdInput(dataset_id=DATASET_ID),
        jobs=[country_lookup_job],
        evaluators=[string_contains_evaluator()],
        parallelism=6,
        print_results=True,
        description="Unit test for unit_test_countries deployment",
    )

    return results


if __name__ == "__main__":
    _ = asyncio.run(run())

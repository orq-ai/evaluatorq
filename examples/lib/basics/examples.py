"""
Main entry point for running evaluatorq examples.

This script runs the simulated delay example to demonstrate
parallel job execution and evaluation.
"""

import asyncio

from .example_runners import run_simulated_delay_example


async def main():
    _ = await run_simulated_delay_example()


if __name__ == "__main__":
    asyncio.run(main())

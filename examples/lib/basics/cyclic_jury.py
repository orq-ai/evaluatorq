"""
Cyclic (round-robin) judge assignment on the same panel as an all-judges jury.

This example shows how to:
- Run the same three-judge panel under `assignment="all"` and `assignment="cyclic"`
- Read the cyclic item -> judge mapping back out of `raw_output["jury"]`
- See the cost difference: 3 judge calls per item vs 1, on one dataset

The jobs return canned text so the only LLM calls are the judges' — that keeps
the cost of running this comparable to a single small eval.
"""

import asyncio
from collections import Counter
from typing import Any

from evaluatorq import DataPoint, DataPointResult, evaluatorq, job, llm_jury
from evaluatorq.contracts import JURY_RAW_OUTPUT_KEY

PANEL = [
    "openai/gpt-5.4-mini",
    "openai/gpt-5.4-nano",
    "deepseek/deepseek-v4-flash",
]

# Six answers to the same kind of question, deliberately mixed in quality so the
# judges have something to disagree about.
ANSWERS = {
    "capital-fr": "Paris.",
    "capital-jp": "Tokyo, which has been the capital since 1868.",
    "capital-au": "Sydney.",  # wrong: it is Canberra
    "capital-br": "Brasilia.",
    "capital-ca": "Toronto.",  # wrong: it is Ottawa
    "capital-ch": "Bern, though Zurich is the largest city.",
}


@job("answer")
async def answer(data: DataPoint, _row: int = 0) -> str:
    """Return the canned answer for this data point (no LLM call)."""
    return ANSWERS[data.inputs["id"]]


CRITERIA = "Is the answer factually correct? Ignore style and length."

all_judges = llm_jury(
    name="correctness-all",
    criteria=CRITERIA,
    judges=PANEL,
    # The default. Every judge scores every datapoint, so each per-item verdict
    # is a panel consensus — and each item costs len(PANEL) judge calls.
    assignment="all",
)

cyclic = llm_jury(
    name="correctness-cyclic",
    criteria=CRITERIA,
    judges=PANEL,
    # CyclicJudge: datapoint i is scored by judge i % len(PANEL). Panel-relative
    # judge bias still cancels across the run, but each item costs exactly one
    # judge call. Per-item verdicts are one judge's opinion, so read the
    # run-level rate, not an individual row.
    assignment="cyclic",
)


def _votes(result: DataPointResult, evaluator_name: str) -> list[dict[str, Any]] | None:
    """Per-judge votes for one evaluator on one datapoint, or None if not recorded.

    The jury record only rides on `raw_output` under `assignment="cyclic"`, where
    it is the sole record of which judge scored the item. Under `"all"` the panel
    itself is the record, so `raw_output` stays `None`.
    """
    for job_result in result.job_results or []:
        for score in job_result.evaluator_scores or []:
            if score.evaluator_name != evaluator_name:
                continue
            raw = score.score.raw_output
            if raw is None:
                return None
            return raw.get(JURY_RAW_OUTPUT_KEY, {}).get("votes", [])
    return None


def summarize(results: list[DataPointResult], evaluator_name: str) -> None:
    """Print the item -> judge mapping and the total judge calls made."""
    print(f"\n{evaluator_name}")
    shares: Counter[str] = Counter()
    for result in results:
        item = result.data_point.inputs["id"]
        votes = _votes(result, evaluator_name)
        if votes is None:
            # assignment="all": no per-item record, because every judge ran.
            shares.update(PANEL)
            print(f"  {item:<12} judged by the whole panel ({len(PANEL)} judges)")
            continue
        models = [vote["model"] for vote in votes]
        shares.update(models)
        print(f"  {item:<12} judged by {', '.join(models)}")
    print(f"  {sum(shares.values())} judge calls total: {dict(shares)}")


async def main():
    """Score one dataset twice — once with every judge, once round-robin."""
    results = await evaluatorq(
        "cyclic-jury",
        data=[DataPoint(inputs={"id": key}) for key in ANSWERS],
        jobs=[answer],
        evaluators=[all_judges, cyclic],
        parallelism=4,
        print_results=False,
    )

    # Both mappings are stable at any `parallelism`: cyclic assignment is keyed
    # on the dataset row, not on whichever judge call happens to arrive first.
    summarize(results, "correctness-all")
    summarize(results, "correctness-cyclic")


if __name__ == "__main__":
    asyncio.run(main())

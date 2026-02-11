"""
Integration test for includeMessages conflict error.

This script:
1. Creates a temporary dataset via Orq SDK
2. Adds a datapoint with 'messages' in inputs AND top-level messages
3. Runs evaluatorq with include_messages=True
4. Expects a ValueError about the conflict
5. Cleans up the dataset (including datapoints)
"""

import os
from collections.abc import Generator
from typing import Any

import pytest
from orq_ai_sdk import Orq

from evaluatorq import evaluatorq
from evaluatorq.types import DataPoint, DatasetIdInput


@pytest.fixture
def orq_client() -> Orq:
    """Create Orq client from .env API key and base URL."""
    api_key = os.environ.get("ORQ_API_KEY")
    if not api_key:
        pytest.skip("ORQ_API_KEY not set in .env file")
    base_url = os.environ.get("ORQ_BASE_URL", "https://my.orq.ai")
    return Orq(api_key=api_key, server_url=base_url)


@pytest.fixture
def test_dataset(orq_client: Orq) -> Generator[str, None, None]:
    """Create a temporary dataset and clean it up after the test."""
    dataset = orq_client.datasets.create(
        request={
            "display_name": "test-include-messages-conflict",
            "path": "evaluatorq-test",
        }
    )
    assert dataset is not None, "Failed to create dataset"
    dataset_id = dataset.id

    yield dataset_id

    # Cleanup: clear datapoints first, then delete dataset
    try:
        orq_client.datasets.clear(dataset_id=dataset_id)
    except Exception:
        pass
    try:
        orq_client.datasets.delete(dataset_id=dataset_id)
    except Exception as e:
        print(f"Warning: Failed to delete dataset: {e}")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_include_messages_conflict_raises_error(
    orq_client: Orq, test_dataset: str
) -> None:
    """Test that includeMessages raises error when inputs already contain 'messages' key."""
    dataset_id = test_dataset

    # Add a datapoint with 'messages' in BOTH inputs and top-level
    _ = orq_client.datasets.create_datapoint(
        dataset_id=dataset_id,
        request_body=[
            {
                "inputs": {
                    "prompt": "Hello",
                    "messages": [
                        {"role": "user", "content": "Existing message in inputs"}
                    ],
                },
                "messages": [{"role": "assistant", "content": "Top-level message"}],
            }
        ],
    )

    async def dummy_job(data: DataPoint, _row: int) -> dict[str, Any]:
        return {"name": "dummy", "output": str(data.inputs)}

    # Run evaluatorq with include_messages=True - should raise error
    with pytest.raises(Exception) as exc_info:
        _ = await evaluatorq(
            "test-conflict",
            data=DatasetIdInput(dataset_id=dataset_id, include_messages=True),
            jobs=[dummy_job],
            evaluators=[],
            print_results=False,
        )

    assert (
        "include_messages is enabled but the datapoint inputs already contain a 'messages' key"
        in str(exc_info.value)
    )


@pytest.mark.asyncio
@pytest.mark.integration
async def test_include_messages_works_without_conflict(
    orq_client: Orq, test_dataset: str
) -> None:
    """Test that includeMessages works when there's no conflict."""
    dataset_id = test_dataset

    # Add a datapoint with top-level messages but NO messages in inputs
    _ = orq_client.datasets.create_datapoint(
        dataset_id=dataset_id,
        request_body=[
            {
                "inputs": {
                    "prompt": "Hello",
                },
                "messages": [{"role": "user", "content": "Top-level message"}],
            }
        ],
    )

    captured_inputs: list[dict[str, Any]] = []

    async def capture_job(data: DataPoint, _row: int) -> dict[str, Any]:
        captured_inputs.append(data.inputs)
        return {"name": "capture", "output": "done"}

    # Run evaluatorq with include_messages=True - should work
    _ = await evaluatorq(
        "test-no-conflict",
        data=DatasetIdInput(dataset_id=dataset_id, include_messages=True),
        jobs=[capture_job],
        evaluators=[],
        print_results=False,
    )

    # Verify messages were merged into inputs
    assert len(captured_inputs) == 1
    assert "messages" in captured_inputs[0]
    messages = captured_inputs[0]["messages"]
    assert len(messages) == 1
    # Messages may be returned as objects or dicts depending on SDK version
    msg = messages[0]
    if hasattr(msg, "role"):
        assert msg.role == "user"
        assert msg.content == "Top-level message"
    else:
        assert msg["role"] == "user"
        assert msg["content"] == "Top-level message"

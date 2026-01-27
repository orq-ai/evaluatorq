"""Fetch data from Orq platform."""

from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING

from .types import DataPoint

if TYPE_CHECKING:
    from orq_ai_sdk import Orq


class DataPointBatch:
    """A batch of datapoints with pagination info."""

    def __init__(self, datapoints: list[DataPoint], has_more: bool, batch_number: int):
        self.datapoints: list[DataPoint] = datapoints
        self.has_more: bool = has_more
        self.batch_number: int = batch_number


def setup_orq_client(api_key: str) -> "Orq":
    """
    Setup and return an Orq client instance.

    Args:
        api_key: Orq API key for authentication

    Returns:
        Orq client instance

    Raises:
        ModuleNotFoundError: If orq_ai_sdk is not installed
        Exception: If client setup fails
    """
    import os

    try:
        # lazy import for orq integration
        from orq_ai_sdk import Orq

        server_url = os.environ.get("ORQ_BASE_URL", "https://my.orq.ai")
        client = Orq(api_key=api_key, server_url=server_url)
        return client
    except ModuleNotFoundError as e:
        raise Exception(
            """orq_ai_sdk is not installed.
            Please install it using:
                * pip install orq_ai_sdk.
                * uv add orq_ai_sdk
                * poetry add orq_ai_sdk"""
        ) from e
    except Exception as e:
        raise Exception(f"Error setting up Orq client: {e}")


async def fetch_dataset_batches(
    orq_client: "Orq", dataset_id: str, *, include_messages: bool = False
) -> AsyncGenerator[DataPointBatch, None]:
    """
    Fetch dataset from Orq platform in batches, yielding each batch as it arrives.
    This allows processing to start before all data is fetched.

    Args:
        orq_client: Orq client instance
        dataset_id: ID of the dataset to fetch

    Yields:
        DataPointBatch objects containing datapoints and pagination info

    Raises:
        Exception: If dataset fetch fails or dataset not found
    """
    starting_after: str | None = None
    last_id: str | None = None
    batch_number = 0
    has_yielded = False

    try:
        while True:
            response = await orq_client.datasets.list_datapoints_async(
                dataset_id=dataset_id,
                limit=50,
                starting_after=starting_after,
            )

            if not response or not response.data:
                if not has_yielded:
                    raise Exception(f"Dataset {dataset_id} not found or has no data")
                break

            # Convert datapoints for this batch
            batch_datapoints: list[DataPoint] = []
            for point in response.data:
                inputs = dict(point.inputs) if point.inputs is not None else {}
                if include_messages:
                    if "messages" in inputs:
                        raise ValueError(
                            "include_messages is enabled but the datapoint inputs already contain a 'messages' key. Remove 'messages' from inputs or disable include_messages."
                        )
                    if getattr(point, "messages", None):
                        inputs["messages"] = point.messages
                batch_datapoints.append(
                    DataPoint(
                        inputs=inputs,
                        expected_output=point.expected_output,
                    )
                )
                # Track the last ID for pagination
                last_id = getattr(point, "_id", None) or getattr(point, "id", None)

            has_more = getattr(response, "has_more", False)
            batch_number += 1

            # Yield this batch immediately
            yield DataPointBatch(
                datapoints=batch_datapoints,
                has_more=has_more,
                batch_number=batch_number,
            )
            has_yielded = True

            # Check if there are more pages
            if not has_more:
                break

            # Set cursor for next page
            starting_after = last_id

    except Exception as e:
        raise Exception(f"Failed to fetch dataset {dataset_id}: {e}")


async def fetch_dataset_as_datapoints(
    orq_client: "Orq", dataset_id: str, *, include_messages: bool = False
) -> list[DataPoint]:
    """
    Fetch all dataset datapoints at once (legacy function).
    For streaming, use fetch_dataset_batches instead.

    Args:
        orq_client: Orq client instance
        dataset_id: ID of the dataset to fetch

    Returns:
        List of DataPoint objects with inputs and expected_output
    """
    all_datapoints: list[DataPoint] = []
    async for batch in fetch_dataset_batches(
        orq_client, dataset_id, include_messages=include_messages
    ):
        all_datapoints.extend(batch.datapoints)
    return all_datapoints

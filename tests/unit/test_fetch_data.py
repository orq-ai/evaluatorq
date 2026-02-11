"""Tests for the fetch_data module, specifically includeMessages functionality."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from evaluatorq.fetch_data import fetch_dataset_as_datapoints, fetch_dataset_batches


class MockDatapoint:
    """Mock datapoint returned from Orq API."""

    inputs: dict[str, Any]
    messages: list[Any] | None
    _id: str
    expected_output: dict[str, Any] | None

    def __init__(
        self,
        inputs: dict[str, Any],
        messages: list[Any] | None = None,
        _id: str = "test-id",
        expected_output: dict[str, Any] | None = None,
    ):
        self.inputs = inputs
        self.messages = messages
        self._id = _id
        self.expected_output = expected_output


class MockResponse:
    """Mock response from Orq API."""

    data: list[MockDatapoint]
    has_more: bool

    def __init__(self, data: list[MockDatapoint], has_more: bool = False):
        self.data = data
        self.has_more = has_more


@pytest.mark.asyncio
async def test_include_messages_false_preserves_original_inputs():
    """Test that include_messages=False preserves original inputs without modification."""
    mock_client = MagicMock()
    mock_client.datasets.list_datapoints_async = AsyncMock(
        return_value=MockResponse(
            data=[
                MockDatapoint(
                    inputs={"prompt": "Hello"},
                    messages=[{"role": "user", "content": "Hi"}],
                    _id="1",
                ),
            ],
            has_more=False,
        )
    )

    batches = []
    async for batch in fetch_dataset_batches(
        mock_client, "test-dataset", include_messages=False
    ):
        batches.append(batch)

    assert len(batches) == 1
    assert len(batches[0].datapoints) == 1
    datapoint = batches[0].datapoints[0]
    assert datapoint.inputs == {"prompt": "Hello"}
    assert "messages" not in datapoint.inputs


@pytest.mark.asyncio
async def test_include_messages_true_merges_messages_into_inputs():
    """Test that include_messages=True merges top-level messages into inputs."""
    mock_client = MagicMock()
    mock_client.datasets.list_datapoints_async = AsyncMock(
        return_value=MockResponse(
            data=[
                MockDatapoint(
                    inputs={"prompt": "Hello"},
                    messages=[{"role": "user", "content": "Hi"}],
                    _id="1",
                ),
            ],
            has_more=False,
        )
    )

    batches = []
    async for batch in fetch_dataset_batches(
        mock_client, "test-dataset", include_messages=True
    ):
        batches.append(batch)

    assert len(batches) == 1
    assert len(batches[0].datapoints) == 1
    datapoint = batches[0].datapoints[0]
    assert datapoint.inputs == {
        "prompt": "Hello",
        "messages": [{"role": "user", "content": "Hi"}],
    }


@pytest.mark.asyncio
async def test_include_messages_true_no_messages_on_datapoint():
    """Test that include_messages=True doesn't add messages if datapoint has none."""
    mock_client = MagicMock()
    mock_client.datasets.list_datapoints_async = AsyncMock(
        return_value=MockResponse(
            data=[
                MockDatapoint(
                    inputs={"prompt": "Hello"},
                    messages=None,
                    _id="1",
                ),
            ],
            has_more=False,
        )
    )

    batches = []
    async for batch in fetch_dataset_batches(
        mock_client, "test-dataset", include_messages=True
    ):
        batches.append(batch)

    assert len(batches) == 1
    datapoint = batches[0].datapoints[0]
    assert datapoint.inputs == {"prompt": "Hello"}
    assert "messages" not in datapoint.inputs


@pytest.mark.asyncio
async def test_include_messages_conflict_raises_error():
    """Test that include_messages=True raises error when inputs already contain 'messages' key."""
    mock_client = MagicMock()
    mock_client.datasets.list_datapoints_async = AsyncMock(
        return_value=MockResponse(
            data=[
                MockDatapoint(
                    inputs={
                        "prompt": "Hello",
                        "messages": [{"role": "user", "content": "Existing message"}],
                    },
                    messages=[{"role": "assistant", "content": "New message"}],
                    _id="1",
                ),
            ],
            has_more=False,
        )
    )

    # The ValueError gets wrapped in an Exception by fetch_dataset_batches
    with pytest.raises(Exception) as exc_info:
        async for _ in fetch_dataset_batches(
            mock_client, "test-dataset", include_messages=True
        ):
            pass

    assert (
        "include_messages is enabled but the datapoint inputs already contain a 'messages' key"
        in str(exc_info.value)
    )
    assert "Remove 'messages' from inputs or disable include_messages" in str(
        exc_info.value
    )


@pytest.mark.asyncio
async def test_include_messages_conflict_with_empty_messages_in_inputs():
    """Test that conflict error is raised even when messages in inputs is empty list."""
    mock_client = MagicMock()
    mock_client.datasets.list_datapoints_async = AsyncMock(
        return_value=MockResponse(
            data=[
                MockDatapoint(
                    inputs={
                        "prompt": "Hello",
                        "messages": [],  # Empty but still present
                    },
                    messages=[{"role": "user", "content": "New message"}],
                    _id="1",
                ),
            ],
            has_more=False,
        )
    )

    with pytest.raises(Exception) as exc_info:
        async for _ in fetch_dataset_batches(
            mock_client, "test-dataset", include_messages=True
        ):
            pass

    assert (
        "include_messages is enabled but the datapoint inputs already contain a 'messages' key"
        in str(exc_info.value)
    )


@pytest.mark.asyncio
async def test_include_messages_false_allows_messages_in_inputs():
    """Test that include_messages=False allows 'messages' key in inputs without error."""
    mock_client = MagicMock()
    mock_client.datasets.list_datapoints_async = AsyncMock(
        return_value=MockResponse(
            data=[
                MockDatapoint(
                    inputs={
                        "prompt": "Hello",
                        "messages": [{"role": "user", "content": "Existing message"}],
                    },
                    messages=[{"role": "assistant", "content": "New message"}],
                    _id="1",
                ),
            ],
            has_more=False,
        )
    )

    batches = []
    async for batch in fetch_dataset_batches(
        mock_client, "test-dataset", include_messages=False
    ):
        batches.append(batch)

    assert len(batches) == 1
    datapoint = batches[0].datapoints[0]
    # Original messages in inputs should be preserved
    assert datapoint.inputs["messages"] == [
        {"role": "user", "content": "Existing message"}
    ]


@pytest.mark.asyncio
async def test_fetch_dataset_as_datapoints_with_include_messages():
    """Test that fetch_dataset_as_datapoints properly passes include_messages."""
    mock_client = MagicMock()
    mock_client.datasets.list_datapoints_async = AsyncMock(
        return_value=MockResponse(
            data=[
                MockDatapoint(
                    inputs={"prompt": "Hello"},
                    messages=[{"role": "user", "content": "Hi"}],
                    _id="1",
                ),
            ],
            has_more=False,
        )
    )

    datapoints = await fetch_dataset_as_datapoints(
        mock_client, "test-dataset", include_messages=True
    )

    assert len(datapoints) == 1
    assert datapoints[0].inputs == {
        "prompt": "Hello",
        "messages": [{"role": "user", "content": "Hi"}],
    }


@pytest.mark.asyncio
async def test_fetch_dataset_as_datapoints_conflict_raises_error():
    """Test that fetch_dataset_as_datapoints raises error on conflict."""
    mock_client = MagicMock()
    mock_client.datasets.list_datapoints_async = AsyncMock(
        return_value=MockResponse(
            data=[
                MockDatapoint(
                    inputs={
                        "prompt": "Hello",
                        "messages": [{"role": "user", "content": "Existing"}],
                    },
                    messages=[{"role": "assistant", "content": "New"}],
                    _id="1",
                ),
            ],
            has_more=False,
        )
    )

    with pytest.raises(Exception) as exc_info:
        _ = await fetch_dataset_as_datapoints(
            mock_client, "test-dataset", include_messages=True
        )

    # The error gets wrapped in fetch_dataset_as_datapoints
    assert (
        "include_messages is enabled but the datapoint inputs already contain a 'messages' key"
        in str(exc_info.value)
    )

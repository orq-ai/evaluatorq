"""Fetch data from Orq platform."""

import contextlib
import json
import os
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import httpx
from loguru import logger

from .common.llm_client import ORQ_DEFAULT_HOST
from .types import DataPoint

if TYPE_CHECKING:
    from orq_ai_sdk import Orq


@dataclass
class DataPointBatch:
    """A batch of datapoints with pagination info."""

    datapoints: list[DataPoint]
    has_more: bool
    batch_number: int


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
    try:
        # lazy import for orq integration
        from orq_ai_sdk import Orq

        server_url = os.environ.get("ORQ_BASE_URL", "https://my.orq.ai")
        return Orq(api_key=api_key, server_url=server_url)
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


def _resolve_orq_base_url(base_url: str | None) -> str:
    return (base_url or os.getenv("ORQ_BASE_URL", ORQ_DEFAULT_HOST)).rstrip("/")


def _raise_for_experiment(
    response: httpx.Response, experiment_id: str, action: str
) -> None:
    if response.is_success:
        return
    detail = response.text[:300] or "Unknown error"
    raise ValueError(
        f"Could not {action} for experiment '{experiment_id}' "
        f"({response.status_code} {response.reason_phrase}): {detail}"
    )


def _experiment_row_to_datapoint(row: dict[str, Any]) -> DataPoint:
    """Map one exported experiment-run row to a DataPoint.

    The row's recorded task output is surfaced as the final assistant message so the
    no-inference replay path (which reads the last assistant message from the
    ``messages`` column) consumes it unchanged. Any conversation already present in the
    row's inputs is preserved; the recorded response is appended after it.
    """
    raw_inputs = row.get("inputs")
    inputs: dict[str, Any]
    if isinstance(raw_inputs, str):
        try:
            parsed = json.loads(raw_inputs) if raw_inputs.strip() else {}
        except json.JSONDecodeError:
            parsed = {"input": raw_inputs}
        inputs = parsed if isinstance(parsed, dict) else {"input": parsed}
    elif isinstance(raw_inputs, dict):
        inputs = dict(raw_inputs)
    elif raw_inputs is None:
        inputs = {}
    else:
        inputs = {"input": raw_inputs}

    existing = inputs.get("messages")
    messages: list[Any] = list(existing) if isinstance(existing, (list, tuple)) else []
    response = row.get("task_output")
    # An empty or missing response leaves no assistant message to replay, so the
    # replay path raises a clear per-row error and the run fails loudly.
    if response is not None and str(response).strip():
        messages.append({"role": "assistant", "content": response})
    inputs["messages"] = messages

    return DataPoint(inputs=inputs, expected_output=row.get("expected_output"))


async def fetch_experiment_datapoints(
    api_key: str,
    experiment_id: str,
    run_id: str | None = None,
    *,
    base_url: str | None = None,
) -> list[DataPoint]:
    """Load an Orq experiment run's recorded responses as DataPoints.

    Used by the no-inference path: each exported row becomes a DataPoint whose recorded
    response is replayed and scored, so evaluators run against a prior experiment's
    outputs without regenerating them.

    Args:
        api_key: Orq API key for authentication.
        experiment_id: The experiment (sheet) ID to load responses from.
        run_id: A specific run (manifest) ID. When omitted, the latest run is used.
        base_url: Optional Orq host override. Falls back to ``ORQ_BASE_URL`` then the
            default host.

    Returns:
        A list of DataPoint objects, one per exported row.

    Raises:
        ValueError: if the experiment, run, or its rows cannot be loaded.
    """
    base = _resolve_orq_base_url(base_url)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        resolved_run_id = run_id
        if resolved_run_id is None:
            manifests_resp = await client.get(
                f"{base}/v2/spreadsheets/{experiment_id}/manifests", headers=headers
            )
            _raise_for_experiment(manifests_resp, experiment_id, "list runs")
            payload = manifests_resp.json()
            manifests = payload if isinstance(payload, list) else payload.get("data", [])
            if not manifests:
                raise ValueError(
                    f"Experiment '{experiment_id}' has no runs to load responses from."
                )
            latest = max(manifests, key=lambda m: m.get("created") or "")
            resolved_run_id = latest.get("_id") or latest.get("id")
            if not resolved_run_id:
                raise ValueError(
                    f"Could not determine the latest run for experiment '{experiment_id}'."
                )
            logger.debug(
                "inference=False: using latest run {} for experiment {}.",
                resolved_run_id,
                experiment_id,
            )

        export_resp = await client.post(
            f"{base}/v2/spreadsheets/{experiment_id}/manifests/{resolved_run_id}/export",
            headers=headers,
            json={"format": "jsonl"},
        )
        export_url: str | None = None
        if export_resp.is_redirect:
            export_url = export_resp.headers.get("location")
        elif export_resp.is_success:
            with contextlib.suppress(json.JSONDecodeError):
                body = export_resp.json()
                if isinstance(body, dict):
                    export_url = (
                        body.get("url") or body.get("redirectUrl") or body.get("signedUrl")
                    )
        else:
            _raise_for_experiment(export_resp, experiment_id, "export run")
        if not export_url:
            raise ValueError(
                f"Export of run '{resolved_run_id}' (experiment '{experiment_id}') "
                "did not return a download URL."
            )

        # The signed URL carries its own auth; sending the Orq bearer token would be
        # rejected by the storage host, so download it without our headers.
        download_resp = await client.get(export_url, follow_redirects=True)
        if not download_resp.is_success:
            raise ValueError(
                f"Could not download the export for run '{resolved_run_id}' "
                f"(experiment '{experiment_id}'): "
                f"{download_resp.status_code} {download_resp.reason_phrase}"
            )
        rows = [
            json.loads(line) for line in download_resp.text.splitlines() if line.strip()
        ]

    if not rows:
        raise ValueError(
            f"Experiment run '{resolved_run_id}' (experiment '{experiment_id}') "
            "returned no rows to evaluate."
        )
    return [_experiment_row_to_datapoint(row) for row in rows]

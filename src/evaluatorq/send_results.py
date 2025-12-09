"""Send evaluation results to Orq platform."""

import os
from datetime import datetime

import httpx
from pydantic import BaseModel, Field

from .types import DataPointResult, EvaluatorqResult


class SendResultsPayload(BaseModel):
    """The payload format expected by the Orq API."""

    model_config: dict[str, bool] = {"populate_by_name": True}

    name: str = Field(serialization_alias="_name")
    description: str | None = Field(default=None, serialization_alias="_description")
    created_at: str = Field(serialization_alias="_createdAt")
    ended_at: str = Field(serialization_alias="_endedAt")
    evaluation_duration: int = Field(serialization_alias="_evaluationDuration")
    dataset_id: str | None = Field(default=None, serialization_alias="datasetId")
    results: list[DataPointResult]


class OrqResponse(BaseModel):
    """Response from Orq API when sending results."""

    sheet_id: str
    manifest_id: str
    experiment_name: str
    rows_created: int
    workspace_key: str | None = None
    experiment_url: str | None = None


async def send_results_to_orq(
    api_key: str,
    evaluation_name: str,
    evaluation_description: str | None,
    dataset_id: str | None,
    results: EvaluatorqResult,
    start_time: datetime,
    end_time: datetime,
) -> None:
    """
    Send evaluation results to Orq platform.

    Args:
        api_key: Orq API key for authentication
        evaluation_name: Name of the evaluation
        evaluation_description: Optional description of the evaluation
        dataset_id: Optional dataset ID if using Orq datasets
        results: The evaluation results to send
        start_time: When the evaluation started
        end_time: When the evaluation ended
    """
    try:
        # Calculate duration in milliseconds
        duration_ms = int((end_time - start_time).total_seconds() * 1000)

        # Build payload - results already have serialization aliases defined
        payload = SendResultsPayload(
            name=evaluation_name,
            description=evaluation_description,
            created_at=start_time.isoformat(),
            ended_at=end_time.isoformat(),
            evaluation_duration=duration_ms,
            dataset_id=dataset_id,
            results=results,
        )

        # Get base URL from environment or use default
        base_url = os.getenv("ORQ_BASE_URL", "https://api.orq.ai")
        orq_debug = os.getenv("ORQ_DEBUG", "false").lower() == "true"

        # Serialize with aliases for API field names
        payload_dict = payload.model_dump(mode="json", exclude_none=True, by_alias=True)

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{base_url}/v2/spreadsheets/evaluations/receive",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
                json=payload_dict,
            )

            if not response.is_success:
                error_text = response.text or "Unknown error"

                # Log warning instead of raising
                print(
                    f"\n⚠️  Warning: Could not send results to Orq platform ({response.status_code} {response.reason_phrase})"
                )

                # Only show detailed error in verbose mode or specific error cases
                if orq_debug or response.status_code >= 500:
                    print(f"   Details: {error_text}")

                return  # Return early but don't raise

            orq_result = OrqResponse.model_validate(response.json())

            print(
                f"✅ Results sent to Orq: {orq_result.experiment_name} ({orq_result.rows_created} rows created)"
            )

            # Display the experiment URL if available
            if orq_result.experiment_url:
                print(f"\n📊 View your evaluation at: {orq_result.experiment_url}")

    except Exception as error:
        # Log warning for network or other errors
        print("\n⚠️  Warning: Could not send results to Orq platform")

        if os.getenv("ORQ_DEBUG", "false").lower() == "true":
            print(f"   Details: {str(error)}")

        # Don't raise - just log the warning
        return

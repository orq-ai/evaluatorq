"""Send evaluation results to Orq platform."""

from datetime import datetime

import httpx
from loguru import logger
from pydantic import BaseModel, Field

from .common.env_config import DEFAULT_ORQ_BASE_URL, env_float, env_raw
from .common.retry import with_retry
from .types import DataPointResult, EvaluatorqResult


class SendResultsPayload(BaseModel):
    """The payload format expected by the Orq API."""

    model_config: dict[str, bool] = {'populate_by_name': True}  # ty: ignore[invalid-attribute-override]

    name: str = Field(serialization_alias='_name')
    description: str | None = Field(default=None, serialization_alias='_description')
    created_at: str = Field(serialization_alias='_createdAt')
    ended_at: str = Field(serialization_alias='_endedAt')
    evaluation_duration: int = Field(serialization_alias='_evaluationDuration')
    dataset_id: str | None = Field(default=None, serialization_alias='datasetId')
    path: str | None = Field(default=None)
    results: list[DataPointResult]


class OrqResponse(BaseModel):
    """Response from Orq API when sending results."""

    sheet_id: str
    manifest_id: str
    experiment_name: str
    rows_created: int
    workspace_key: str | None = None
    experiment_url: str | None = None


class SendResultsError(RuntimeError):
    """Raised when uploading evaluation results to Orq fails."""


async def _post_results(
    url: str,
    api_key: str,
    payload: dict[str, object],
    *,
    timeout: float,
    raise_on_error: bool,
) -> httpx.Response | None:
    """POST the payload, retrying only failures that never reached the handler.

    Returns ``None`` after a read timeout (logged, or raised as
    ``SendResultsError`` under ``raise_on_error``); any other response,
    successful or not, is returned for the caller to inspect.
    """
    async with httpx.AsyncClient(timeout=timeout) as client:

        async def post() -> httpx.Response | None:
            try:
                response = await client.post(
                    url,
                    headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {api_key}'},
                    json=payload,
                )
            except httpx.ReadTimeout:
                # Caught here, inside the retry, because with_retry retries a
                # ReadTimeout: the server keeps processing after the client gives
                # up, so a retry would create a second run. Lift this once the
                # platform deduplicates repeated uploads (BACK-2363).
                return None
            # 429 is rejected before the handler runs, so it is safe to retry; a
            # 5xx may come after the run was created, so it is returned as-is.
            if response.status_code == 429:
                response.raise_for_status()
            return response

        try:
            response = await with_retry(post, max_attempts=3, label='Orq results upload')
        except httpx.HTTPStatusError as err:
            return err.response
    if response is None:
        message = (
            f'Orq results upload timed out after {timeout:g}s. The server may still finish and create '
            'the run: check the experiment before uploading again, or raise EVALUATORQ_UPLOAD_TIMEOUT_S.'
        )
        if raise_on_error:
            raise SendResultsError(message)
        logger.warning(message)
    return response


async def send_results_to_orq(
    api_key: str,
    evaluation_name: str,
    evaluation_description: str | None,
    dataset_id: str | None,
    results: EvaluatorqResult,
    start_time: datetime,
    end_time: datetime,
    *,
    path: str | None = None,
    base_url: str | None = None,
    raise_on_error: bool = False,
    timeout_s: float | None = None,
) -> OrqResponse | None:
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
        path: Optional path (e.g. "MyProject/MyFolder") to place the experiment
              in a specific project and folder on the Orq platform.
        base_url: Optional Orq host to upload to (e.g. a staging server). When
              omitted, falls back to the ``ORQ_BASE_URL`` env var, then to
              ``https://my.orq.ai``. Lets callers route results to the same
              server used for inference.
        raise_on_error: Raise upload errors instead of treating upload as best-effort.
        timeout_s: HTTP timeout in seconds for the upload. When omitted, falls
              back to the ``EVALUATORQ_UPLOAD_TIMEOUT_S`` env var (default 120).
              The upload makes up to 3 attempts, retrying on connection
              errors and 429.
              A timed-out upload is not retried: the server may still finish
              and create the run, so a retry could create a duplicate.

    Returns:
        The full ``OrqResponse`` (``experiment_url``, ``rows_created``,
        ``experiment_name``, ``sheet_id``, ...) on success, or ``None`` if the
        upload failed. Callers that only need the URL read
        ``response.experiment_url``.
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
            path=path,
            results=results,
        )

        resolved_base_url = (base_url or env_raw('ORQ_BASE_URL', DEFAULT_ORQ_BASE_URL)).rstrip('/')
        logger.debug('Uploading results to {}', resolved_base_url)

        # Serialize with aliases, stripping None on optional fields (the API
        # rejects null for ``error``, ``explanation``, etc.) but keeping
        # ``output`` even when None (the API requires it to be present).
        payload_dict = payload.model_dump(mode='json', exclude_none=True, by_alias=True)

        # Restore fields stripped by exclude_none.  The API schema marks
        # ``error`` and ``explanation`` as optional strings — they may be
        # absent, but must never be ``null``.  Defensive: ensure they are
        # empty strings rather than absent, in case any code path leaks a
        # ``null`` through nested dicts that exclude_none cannot reach.
        # ``output`` is required (nullable) so it must always be present.
        for result in payload_dict.get('results', []):
            for jr in result.get('jobResults') or []:
                jr.setdefault('output', None)
                jr.setdefault('error', '')
                for es in jr.get('evaluatorScores') or []:
                    if 'score' in es:
                        # Keep a null score explicit: it means the evaluator ran
                        # but could not establish a value (for example, no basis).
                        es['score'].setdefault('value', None)
                        es['score'].setdefault('explanation', '')
                        # Evaluator-cost metadata is for local reports only; never
                        # upload the judge's token usage or raw response to the platform.
                        es['score'].pop('token_usage', None)
                        es['score'].pop('raw_output', None)
                    es.setdefault('error', '')

        timeout = timeout_s if timeout_s is not None else env_float('EVALUATORQ_UPLOAD_TIMEOUT_S')
        response = await _post_results(
            f'{resolved_base_url}/v2/spreadsheets/evaluations/receive',
            api_key,
            payload_dict,
            timeout=timeout,
            raise_on_error=raise_on_error,
        )
        if response is None:
            return None

        if not response.is_success:
            error_text = response.text or 'Unknown error'
            if raise_on_error:
                raise SendResultsError(
                    'Could not send results to Orq platform '
                    f'({response.status_code} {response.reason_phrase}): {error_text}'
                )
            logger.warning(
                'Could not send results to Orq platform ({} {}): {}',
                response.status_code,
                response.reason_phrase,
                error_text,
            )
            return None

        orq_result = OrqResponse.model_validate(response.json())

        logger.info(
            'Results sent to Orq: {} ({} rows created)',
            orq_result.experiment_name,
            orq_result.rows_created,
        )
        if orq_result.rows_created < len(results):
            # The "missing samples" footgun: the platform registered fewer
            # rows than we uploaded. One loud line with both numbers + URL.
            logger.warning(
                'Orq registered {} of {} uploaded rows for experiment {!r} — '
                'some results will be missing from the Explorer. URL: {}',
                orq_result.rows_created,
                len(results),
                orq_result.experiment_name,
                orq_result.experiment_url or '<no url returned>',
            )
        if orq_result.experiment_url:
            logger.info('View your evaluation at: {}', orq_result.experiment_url)

        return orq_result

    except SendResultsError:
        raise
    except Exception:
        logger.exception('Could not send results to Orq platform')
        if raise_on_error:
            raise
        return None

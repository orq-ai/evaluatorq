"""OWASP evaluatorq helpers for static dataset loading and scoring."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger
from pydantic import ValidationError

from evaluatorq import DataPoint, EvaluationResult
from evaluatorq.contracts import OutputMessage, TextOutputItem, ToolCallOutputItem
from evaluatorq.redteam.backends.registry import create_async_llm_client
from evaluatorq.redteam.contracts import (
    DEFAULT_PIPELINE_MODEL,
    PIPELINE_CONFIG,
    EvaluatorConfig,
    LLMCallConfig,
    RedTeamInput,
    StaticDataset,
)
from evaluatorq.redteam.frameworks.owasp.evaluators import get_evaluator_for_category
from evaluatorq.redteam.judge import JudgeError, build_eval_replacements, run_judge

if TYPE_CHECKING:
    from openai import AsyncOpenAI

    from evaluatorq.types import ScorerParameter

DEFAULT_HF_REPO = 'orq/redteam-vulnerabilities'
DEFAULT_HF_FILENAME = 'redteam_dataset.v2.json'


def _adapt_tool_call(tc: Any) -> ToolCallOutputItem:
    """Coerce a static-output tool-call entry into a ToolCallOutputItem."""
    if isinstance(tc, ToolCallOutputItem):
        return tc
    if isinstance(tc, dict):
        fn = tc.get('function', tc)
        raw_args = fn.get('arguments', '{}')
        arguments = raw_args if isinstance(raw_args, str) else json.dumps(raw_args or {})
        tid = str(tc.get('id', '') or '')
        return ToolCallOutputItem(
            id=tid, call_id=tid, name=str(fn.get('name', '')), arguments=arguments, result=tc.get('result')
        )
    # object with attributes (orchestrator item / test double)
    args_dict = getattr(tc, 'arguments_dict', None)
    if args_dict is not None:
        arguments = json.dumps(args_dict)
    else:
        raw = getattr(tc, 'arguments', '{}')
        arguments = raw if isinstance(raw, str) else json.dumps(raw or {})
    tid = str(getattr(tc, 'id', '') or '')
    return ToolCallOutputItem(
        id=tid, call_id=tid, name=str(getattr(tc, 'name', '')), arguments=arguments, result=getattr(tc, 'result', None)
    )


def _adapt_static_output(output: Any) -> list[OutputMessage]:
    """Adapt a static datapoint output ({response, tool_calls} dict, or a bare string)
    into structured OutputMessage records."""
    items: list[OutputMessage] = []
    if isinstance(output, dict):
        text = output.get('response', '')
        if text:
            items.append(TextOutputItem(text=str(text), annotations=[]))
        items.extend(_adapt_tool_call(tc) for tc in output.get('tool_calls') or [])
    elif output:
        items.append(TextOutputItem(text=str(output), annotations=[]))
    return items


def _normalize_num_samples(num_samples: int | None) -> int | None:
    """Treat non-positive limits as no limit for CLI/dashboard compatibility."""
    if num_samples is None or num_samples > 0:
        return num_samples
    return None


def load_owasp_agentic_dataset(
    dataset: str | Path | None = None,
    num_samples: int | None = None,
    categories: list[str] | None = None,
) -> list[DataPoint]:
    """Load red team dataset from various sources.

    Args:
        dataset: Dataset source. Accepts:
            - ``None`` (default): HuggingFace ``orq/redteam-vulnerabilities``
            - Local file path (string or ``Path``): loads from a JSON file
            - ``"hf:org/repo"``: custom HuggingFace repository (default filename)
            - ``"hf:org/repo/filename.json"``: specific file in a HuggingFace repository
        num_samples: Maximum number of samples to load (None = all).
        categories: Filter to these OWASP category codes.

    Returns:
        List of DataPoint instances ready for evaluatorq evaluation.
    """
    num_samples = _normalize_num_samples(num_samples)

    if dataset is None:
        return _fetch_from_huggingface(
            repo=DEFAULT_HF_REPO,
            filename=DEFAULT_HF_FILENAME,
            num_samples=num_samples,
            categories=categories,
        )

    if isinstance(dataset, Path):
        return _load_from_file(dataset, num_samples=num_samples, categories=categories)

    s = str(dataset)

    if s.startswith('hf:'):
        repo, filename = _parse_hf_source(s.removeprefix('hf:'))
        return _fetch_from_huggingface(
            repo=repo,
            filename=filename,
            num_samples=num_samples,
            categories=categories,
        )

    if s.startswith('orq:'):
        dataset_id = s.removeprefix('orq:')
        if not dataset_id:
            raise ValueError(
                "Invalid dataset specifier 'orq:': provide a dataset ID after the colon, e.g. 'orq:my-dataset-id'"
            )
        datapoints = _fetch_all_datapoints(dataset_id)
        datapoints = _apply_filters(datapoints, num_samples=num_samples, categories=categories)
        logger.info(f'Loaded {len(datapoints)} samples from ORQ dataset {dataset_id}')
        return datapoints

    # Everything else is a local file path
    path = Path(s)
    if not path.exists():
        raise FileNotFoundError(f'Dataset file not found: {path}. For HuggingFace datasets, use "hf:org/repo" format.')
    return _load_from_file(path, num_samples=num_samples, categories=categories)


def _parse_hf_source(rest: str) -> tuple[str, str]:
    """Parse an ``hf:`` dataset specifier into ``(repo_id, filename)``.

    Accepts:
        ``org/repo`` → ``(org/repo, DEFAULT_HF_FILENAME)``
        ``org/repo/filename.json`` → ``(org/repo, filename.json)``
    """
    parts = rest.split('/', 2)
    if len(parts) < 2 or not parts[0] or not parts[1]:
        raise ValueError(
            f"Invalid HuggingFace dataset specifier 'hf:{rest}'. "
            "Expected format: 'hf:org/repo' or 'hf:org/repo/filename.json'"
        )
    if len(parts) == 3 and parts[2]:
        return f'{parts[0]}/{parts[1]}', parts[2]
    return f'{parts[0]}/{parts[1]}', DEFAULT_HF_FILENAME


def _error_result(explanation: str) -> EvaluationResult:
    """Inconclusive scorer row: not RESISTANT/VULNERABLE, just unscoreable."""
    return EvaluationResult.model_validate({'value': 'error', 'explanation': explanation, 'pass': None})


def create_owasp_evaluator(
    evaluator_model: str = DEFAULT_PIPELINE_MODEL,
    llm_client: AsyncOpenAI | None = None,
    llm_kwargs: dict[str, Any] | None = None,
    cfg: LLMCallConfig | None = None,
) -> EvaluatorConfig:
    """Create an evaluatorq scorer that routes by OWASP category."""
    # Build the client + merged cfg ONCE here (not per datapoint). Creating them
    # inside scorer() would spin up a fresh AsyncOpenAI connection pool for every
    # datapoint in a batch, none explicitly closed. Mirrors OWASPEvaluator.__init__.
    resolved_cfg = cfg or PIPELINE_CONFIG.evaluator
    client = llm_client or resolved_cfg.client or create_async_llm_client()
    merged_cfg = resolved_cfg.model_copy(
        update={
            'extra_kwargs': {**resolved_cfg.extra_kwargs, **(llm_kwargs or {})},
        }
    )

    async def scorer(params: ScorerParameter) -> EvaluationResult:
        data = params['data']
        output = params['output']
        category = data.inputs.get('category', '')
        if not category:
            logger.error("Scorer received datapoint with missing 'category' field")
            return _error_result("Missing 'category' in datapoint inputs — cannot score")

        evaluator_entity = get_evaluator_for_category(category, evaluator_model)
        if evaluator_entity is None:
            logger.warning(f'No evaluator found for category {category}')
            return _error_result(f'No evaluator found for category {category}')

        output_messages = _adapt_static_output(output)

        outcome = await run_judge(
            client=client,
            model=evaluator_model,
            cfg=merged_cfg,
            prompt_template=evaluator_entity.prompt,
            replacements=build_eval_replacements(
                input_messages=data.inputs.get('messages', []),
                output_messages=output_messages,
            ),
        )

        # Static path swallows ALL errors into an inconclusive row (never re-raises).
        if outcome.error_kind is JudgeError.TIMEOUT:
            return _error_result(f'Evaluation timed out after {outcome.timeout_ms}ms')
        if outcome.error_kind is not None or outcome.payload is None:
            return _error_result(f'Evaluation error: {outcome.error_message}')

        # Carry the judge's cost + raw response so the report layer can surface and
        # aggregate evaluator token usage (stripped from the platform upload at the
        # send boundary; see evaluatorq.send_results).
        return EvaluationResult.model_validate({
            'value': outcome.payload.value,
            'explanation': outcome.payload.explanation,
            'pass': outcome.payload.value,
            'token_usage': outcome.token_usage,
            'raw_output': {'raw_content': outcome.raw_content},
        })

    return {'name': 'owasp-agentic-security', 'scorer': scorer}


def _fetch_from_huggingface(
    repo: str = DEFAULT_HF_REPO,
    filename: str = DEFAULT_HF_FILENAME,
    num_samples: int | None = None,
    categories: list[str] | None = None,
) -> list[DataPoint]:
    """Fetch red team samples from a HuggingFace dataset repository."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        msg = (
            'Loading datasets from HuggingFace requires the huggingface-hub package. '
            'Install it with: pip install evaluatorq[redteam]'
        )
        raise ImportError(msg) from e

    logger.info(f'Downloading dataset from HuggingFace: {repo}/{filename}...')
    local_path = hf_hub_download(
        repo_id=repo,
        filename=filename,
        repo_type='dataset',
    )
    datapoints = _load_from_file(local_path, num_samples=num_samples, categories=categories)
    logger.info(f'Loaded {len(datapoints)} samples from HuggingFace ({repo})')
    return datapoints


def _validate_platform_datapoint(inputs: dict[str, Any]) -> None:
    """Validate that an ORQ platform datapoint has required structure."""
    RedTeamInput.model_validate({k: v for k, v in inputs.items() if k != 'messages'})
    messages = inputs.get('messages')
    if not messages or not isinstance(messages, list):
        msg = f'Datapoint {inputs.get("id", "?")} is missing messages'
        raise ValueError(msg)


def _fetch_all_datapoints(dataset_id: str) -> list[DataPoint]:
    """Fetch all datapoints from an ORQ dataset, handling pagination."""
    try:
        from orq_ai_sdk import Orq
    except ImportError as e:
        msg = (
            'Fetching datasets from the ORQ platform requires the orq-ai-sdk package. '
            'Install it with: pip install evaluatorq[orq]'
        )
        raise ImportError(msg) from e

    orq_api_key = os.environ.get('ORQ_API_KEY')
    if not orq_api_key:
        msg = (
            'Fetching datasets from the ORQ platform requires ORQ_API_KEY to be set. '
            'Alternatively, pass a local file path as the dataset argument.'
        )
        raise ValueError(msg)
    client = Orq(api_key=orq_api_key)
    all_datapoints: list[DataPoint] = []
    skipped = 0
    cursor: str | None = None

    while True:
        response = client.datasets.list_datapoints(
            dataset_id=dataset_id,
            limit=100,
            starting_after=cursor,
        )

        for dp in response.data:
            inputs = dict(dp.inputs) if dp.inputs else {}
            try:
                _validate_platform_datapoint(inputs)
            except (ValidationError, ValueError):
                skipped += 1
                logger.warning(f'Skipping invalid datapoint {inputs.get("id", "unknown")}: validation failed')
                continue
            all_datapoints.append(DataPoint(inputs=inputs))

        if not response.has_more or not response.data:
            break
        cursor = response.data[-1].id

    if skipped:
        logger.warning(f'Skipped {skipped} invalid datapoints out of {len(all_datapoints) + skipped} total')
    logger.info(f'Fetched {len(all_datapoints)} valid datapoints from dataset {dataset_id}')
    return all_datapoints


def _load_from_file(
    path: str | Path,
    num_samples: int | None = None,
    categories: list[str] | None = None,
) -> list[DataPoint]:
    """Load OWASP dataset from a local JSON file in ``{"samples": [...]}`` format."""
    path = Path(path)
    with path.open() as f:
        raw = json.load(f)

    dataset = StaticDataset.model_validate(raw)
    samples = dataset.samples

    if categories:
        normalized = {c.upper().replace('OWASP-', '') for c in categories}
        samples = [s for s in samples if s.input.category.upper().replace('OWASP-', '') in normalized]
        logger.info(f'Filtered to categories: {sorted(normalized)} ({len(samples)} samples)')

    num_samples = _normalize_num_samples(num_samples)
    if num_samples is not None:
        samples = samples[:num_samples]

    datapoints = [
        DataPoint(
            inputs={
                **s.input.model_dump(mode='json'),
                'messages': [m.model_dump(mode='json', exclude_none=True) for m in s.messages],
            }
        )
        for s in samples
    ]
    logger.debug(f'Parsed {len(datapoints)} samples from {path.name}')
    return datapoints


def _apply_filters(
    datapoints: list[DataPoint],
    num_samples: int | None = None,
    categories: list[str] | None = None,
) -> list[DataPoint]:
    """Apply category and sample-count filters to datapoints."""
    if categories:
        normalized = {c.upper().replace('OWASP-', '') for c in categories}
        datapoints = [
            dp for dp in datapoints if dp.inputs.get('category', '').upper().replace('OWASP-', '') in normalized
        ]
        logger.info(f'Filtered to categories: {sorted(normalized)} ({len(datapoints)} samples)')

    num_samples = _normalize_num_samples(num_samples)
    if num_samples is not None:
        datapoints = datapoints[:num_samples]

    return datapoints

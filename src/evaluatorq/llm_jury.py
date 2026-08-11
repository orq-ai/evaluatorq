from __future__ import annotations

import asyncio
import itertools
import typing
from typing import Any, Literal

from loguru import logger
from pydantic import BaseModel, Field

from evaluatorq.common.judge import JudgeOutcome, build_eval_replacements, run_judge
from evaluatorq.common.jury import (
    AggregatorSpec,
    JuryDeliberation,
    Prediction,
    TieBreak,
    VerdictKind,
    append_jury_summary,
    run_jury,
    validate_aggregator,
)
from evaluatorq.common.llm_client import resolve_llm_client
from evaluatorq.common.output_adapters import (
    inputs_to_messages,
    output_error_text,
    output_to_messages,
    output_to_text,
)
from evaluatorq.contracts import JURY_RAW_OUTPUT_KEY, LLMCallConfig
from evaluatorq.pairwise import PairwiseComparison, run_pairwise
from evaluatorq.types import DataPoint, EvaluationResult, Evaluator, Output, ScorerParameter

DEFAULT_JUDGE_MODEL = 'openai/gpt-5.4-mini'

PAIRWISE_LABELS = ['A', 'B', 'tie']
DEFAULT_PAIRWISE_CRITERIA = (
    'Compare the two responses on accuracy and correctness, helpfulness and '
    'completeness, clarity and organization, and relevance to the question.'
)


def _build_verdict_model(
    verdict_kind: str, labels: list[str] | None, score_range: tuple[float, float]
) -> type[BaseModel]:
    """Build a dynamic Pydantic verdict model based on verdict spec.

    Args:
        verdict_kind: The kind of verdict ("categorical" or "numeric")
        labels: List of allowed categorical labels, or None for boolean/numeric
        score_range: The score range (min, max) for numeric verdicts

    Returns:
        A Pydantic BaseModel class with 'value' and 'explanation' fields
    """
    if verdict_kind == 'categorical':
        value_annotation = bool if labels is None else typing.Literal[tuple(labels)]  # type: ignore[valid-type]
    else:  # numeric
        value_annotation = float

    # Create model dynamically
    class VerdictModel(BaseModel):
        value: value_annotation  # type: ignore  # pyright: ignore[reportInvalidTypeForm]
        explanation: str = Field(default='', description='Explanation for the verdict')

    return VerdictModel


# ---------------------------------------------------------------------------
# Template / prompt helpers
# ---------------------------------------------------------------------------


def _build_replacements(data: DataPoint, output: Output, criteria: str) -> dict[str, Any]:
    """Build the red-team-format template substitution dict for the pointwise jury."""
    err = output_error_text(output)
    reps = build_eval_replacements(
        input_messages=inputs_to_messages(data.inputs),
        output_messages=output_to_messages(output),
        expected_output=output_to_text(data.expected_output),
        system_instructions=None,
        error=err,
    )
    reps['criteria'] = criteria
    return reps


def _default_system_prompt(verdict_kind: str, labels: list[str] | None, score_range: tuple[float, float]) -> str:
    """Return a sensible system prompt for the jury judge.

    The prompt instructs the model to return a structured verdict and a 2-3
    sentence explanation.  The ``value`` constraint is tailored to the
    ``verdict_kind``:

    - ``"numeric"`` — a float in ``[lo, hi]``.
    - ``"categorical"`` with labels — one of the given label strings.
    - ``"categorical"`` without labels — a boolean.
    """
    base = (
        "You are a strict evaluator. Read the input, the model's output, any "
        'expected output, and judge against the stated criterion. '
        'Return a structured verdict with a 2-3 sentence explanation.'
    )
    if verdict_kind == 'numeric':
        lo, hi = score_range
        return f'{base} `value` must be a number between {lo} and {hi} (higher is better).'
    if labels:
        return f'{base} `value` must be exactly one of: {", ".join(labels)}.'
    return f'{base} `value` must be a boolean: true if the criterion is met, false otherwise.'


# Default Mustache-style evaluation prompt template.  Placeholder tokens use the
# ``{{name}}`` convention expected by the template engine (double-braces), drawn from
# the red-team template namespace (see
# :func:`evaluatorq.common.judge.build_eval_replacements`).  ``criteria`` is a
# substituted variable here, exactly like every other name — see DEFAULT_PAIRWISE_TEMPLATE.
DEFAULT_TEMPLATE = (
    '# Criteria\n{{criteria}}\n\n'
    '# Input\n{{input.all_messages}}\n\n'
    '# Output\n{{output.response}}\n\n'
    '# Expected output\n{{input.expected_output}}\n'
)


# ---------------------------------------------------------------------------
# Jury → domain-type converters
# ---------------------------------------------------------------------------


def _outcome_to_prediction(outcome: JudgeOutcome) -> Prediction:
    """Convert a raw :class:`JudgeOutcome` into a :class:`Prediction`.

    Maps error / abstain states to the matching Prediction fields so the jury
    runner can aggregate them without knowing about judge internals.
    """
    if outcome.error_kind is not None:
        return Prediction(
            error=outcome.error_message or str(outcome.error_kind),
            token_usage=outcome.token_usage,
        )
    payload = outcome.payload
    if payload is None:
        return Prediction(error='judge returned no payload', token_usage=outcome.token_usage)
    if payload.abstain or payload.value is None:
        return Prediction(
            abstained=True,
            explanation=payload.explanation,
            token_usage=outcome.token_usage,
        )
    return Prediction(
        value=payload.value,
        explanation=payload.explanation,
        token_usage=outcome.token_usage,
    )


async def _run_single_judge(
    *,
    client: Any,
    model: str,
    template: str,
    replacements: dict[str, Any],
    system_prompt: str,
    verdict_model: type[BaseModel],
    structured_output: bool,
    temperature: float | None,
    max_tokens: int,
    timeout_ms: int,
    extra_kwargs: dict[str, Any] | None,
) -> Prediction:
    """Run one judge call and map its outcome to a :class:`Prediction`.

    The single ``run_judge`` call site shared by the pointwise (:func:`llm_jury`)
    and pairwise (:class:`PairwiseComparator`) juries — only the ``replacements``
    differ between them, so keeping one path stops the two from drifting.
    """
    cfg = LLMCallConfig(model=model, max_tokens=max_tokens, timeout_ms=timeout_ms, extra_kwargs=extra_kwargs or {})
    outcome = await run_judge(
        client=client,
        model=model,
        cfg=cfg,
        prompt_template=template,
        replacements=replacements,
        system_prompt=system_prompt,
        response_model=verdict_model,
        structured_output=structured_output,
        temperature=temperature,
    )
    return _outcome_to_prediction(outcome=outcome)


def _to_evaluation_result(
    deliberation: JuryDeliberation,
    *,
    verdict_kind: str,
    passing_labels: list[str] | None,
    threshold: float,
    score_range: tuple[float, float],
    include_jury_record: bool = False,
) -> EvaluationResult:
    """Map a :class:`JuryDeliberation` to an :class:`EvaluationResult`.

    Passing logic:

    - ``"numeric"`` — ``passed`` when ``verdict >= threshold``.
    - Boolean verdicts — ``passed`` equals the verdict directly.
    - String label verdicts — ``passed`` when the label is in ``passing_labels``
      (or always if ``passing_labels`` is ``None``).
    """
    verdict = deliberation.verdict
    explanation = append_jury_summary(deliberation.explanation, deliberation.jury)

    if verdict is None:
        value: Any = 'inconclusive'
        passed = None
    elif verdict_kind == 'numeric':
        lo, hi = score_range
        raw_score = float(verdict)
        if not (lo <= raw_score <= hi):
            logger.warning('judge returned {} outside score_range {}; clamping', raw_score, score_range)
        score = min(max(raw_score, lo), hi)
        value = score
        passed = score >= threshold
    elif isinstance(verdict, bool):
        value = verdict
        passed = verdict
    else:  # string label
        value = verdict
        passed = (verdict in passing_labels) if passing_labels else None

    result: dict[str, Any] = {
        'value': value,
        'explanation': explanation,
        'pass': passed,
        'token_usage': deliberation.token_usage,
    }
    if include_jury_record:
        # Same convention as the redteam bridge: the full JuryResult rides on
        # raw_output under JURY_RAW_OUTPUT_KEY, so per-judge votes (which judge
        # scored this item, what it said) stay auditable. Set for CYCLIC
        # assignment only: there it is the sole record of the item->judge
        # mapping, while under 'all' the panel itself is the record and the
        # payload would be pure redundancy (and a behavior change for callers
        # treating raw_output is None as a signal).
        result['raw_output'] = {JURY_RAW_OUTPUT_KEY: deliberation.jury.model_dump(mode='json')}
    return EvaluationResult.model_validate(result)


# ---------------------------------------------------------------------------
# Panel resolution helper
# ---------------------------------------------------------------------------


def _resolve_panel(judges: list[str] | None, model: str | None) -> list[str]:
    """Resolve a judge panel from either a list of judges or a single model shorthand.

    ``None`` for both means "use the default panel". An explicitly supplied but
    empty ``judges`` list (or blank ``model``) is a configuration error, not a
    request for the default — the resolved panel must be non-empty.

    Raises:
        ValueError: If both ``judges`` and ``model`` are set, if ``judges`` is an
            empty list, or if ``model`` is blank.
    """
    if judges is not None and model is not None:
        raise ValueError('Pass either `judges` or `model`, not both.')
    if judges is not None:
        if not judges:
            raise ValueError('`judges` must be a non-empty list of judge models.')
        return list(judges)
    if model is not None:
        if not model.strip():
            raise ValueError('`model` must be a non-empty judge model identifier.')
        return [model]
    return [DEFAULT_JUDGE_MODEL]


def _resolve_and_validate_panel(
    *, judges: list[str] | None, model: str | None, min_successful_judges: int
) -> tuple[list[str], list[str]]:
    """Resolve the judge panel and validate ``min_successful_judges`` against it.

    Returns ``(panel, deduped)``. The quorum floor is bounded by the deduplicated
    panel size by design: ``replacement_judges`` are spillover for failed
    primaries, not extra capacity, so they don't raise the achievable floor.
    Shared by :func:`llm_jury` and :func:`llm_jury_pairwise`.
    """
    panel = _resolve_panel(judges=judges, model=model)
    deduped = list(dict.fromkeys(panel))
    if not (1 <= min_successful_judges <= len(deduped)):
        raise ValueError(
            f'min_successful_judges ({min_successful_judges}) must be between 1 and '
            f'the deduplicated panel size ({len(deduped)}).'
        )
    return panel, deduped


def _validate_assignment(assignment: str, *, min_successful_judges: int) -> None:
    """Validate the judge-assignment strategy shared by both jury factories."""
    if assignment not in ('all', 'cyclic'):
        raise ValueError(f"unknown assignment: {assignment!r}. Use 'all' or 'cyclic'.")
    if assignment == 'cyclic' and min_successful_judges > 1:
        raise ValueError(
            f'assignment="cyclic" runs exactly one judge per item, so '
            f'min_successful_judges ({min_successful_judges}) must be 1.'
        )


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def llm_jury(
    *,
    name: str,
    criteria: str | None = None,
    prompt: str | None = None,
    system_prompt: str | None = None,
    judges: list[str] | None = None,
    model: str | None = None,
    repetitions: int = 1,
    assignment: Literal['all', 'cyclic'] = 'all',
    replacement_judges: list[str] | None = None,
    min_successful_judges: int = 1,
    verdict_kind: Literal['categorical', 'numeric'] = 'categorical',
    labels: list[str] | None = None,
    passing_labels: list[str] | None = None,
    aggregator: AggregatorSpec | None = None,
    threshold: float = 0.5,
    score_range: tuple[float, float] = (0.0, 1.0),
    tie_break: TieBreak | None = None,
    structured_output: bool = True,
    temperature: float | None = None,
    max_tokens: int = 8000,
    timeout_ms: int = 90000,
    extra_kwargs: dict[str, Any] | None = None,
    client: Any = None,
) -> Evaluator:
    """Build a jury (or single-judge) LLM evaluator for ``evaluators=[...]``.

    Verdict modes
    -------------
    The judge's verdict type and the ``passed`` (pass/fail) field are decided by
    ``verdict_kind`` together with ``labels``. ``verdict_kind`` is **not** inferred
    from ``labels`` — it defaults to ``"categorical"`` and you pick the mode
    explicitly. There are three modes:

    +-------------------------------+------------------+--------------------------------+
    | how you configure it          | judge returns    | ``passed`` is                  |
    +===============================+==================+================================+
    | ``verdict_kind="categorical"``| a JSON boolean   | the boolean itself             |
    | (default), ``labels=None``    | ``true``/``false`` |                              |
    | — **boolean mode**            |                  |                                |
    +-------------------------------+------------------+--------------------------------+
    | ``verdict_kind="categorical"``| one of ``labels``| ``verdict in passing_labels``  |
    | with ``labels=[...]``         | (a string)       | (``None`` if no                |
    | — **labeled mode**            |                  | ``passing_labels`` given)      |
    +-------------------------------+------------------+--------------------------------+
    | ``verdict_kind="numeric"``    | a float in       | ``score >= threshold``         |
    | — **numeric mode**            | ``score_range``  |                                |
    +-------------------------------+------------------+--------------------------------+

    Notes:

    - For a yes/no judge, use **boolean mode** (the default — just omit ``labels``).
      ``passed`` is populated automatically; you do not need ``passing_labels``.
    - ``labels`` and ``passing_labels`` are valid **only** for
      ``verdict_kind="categorical"``; passing them with ``"numeric"`` raises
      ``ValueError``.
    - In labeled mode, ``passing_labels`` must be a subset of ``labels``. If you
      omit it, the verdict is still recorded but ``passed`` is ``None`` (no
      pass/fail, so no pass-rate to aggregate). ``passing_labels`` requires
      ``labels`` — it is rejected in boolean mode (which derives pass/fail from
      the verdict directly).
    - ``labels`` must be strings. Native ``True``/``False`` are not labels — use
      boolean mode for that.
    - ``aggregator`` picks the panel consensus rule and must match the verdict
      kind (a mismatch raises ``ValueError``):

      * categorical: ``"mode"`` (default — most common; plurality ties go to
        ``tie_break``) or ``"majority"`` (strict >50%, else inconclusive).
      * numeric: ``"mean_std"`` (default — mean verdict; std reported in
        ``stats`` on a conclusive verdict), ``"median"``, ``"min"``, or ``"max"``.
      * a custom ``Callable[[list[JuryVote]], bool | float | str | None]`` for
        either kind (return ``None`` for "no consensus" / inconclusive). The
        same numeric keyword also collapses a single judge's ``repetitions``.
    - ``assignment`` picks how judges are allocated across datapoints:

      * ``"all"`` (default): every judge scores every datapoint — per-item
        consensus at K times single-judge cost.
      * ``"cyclic"``: round-robin (CyclicJudge, arXiv:2603.01865) — each
        datapoint is scored by exactly one judge, cycling through the panel so
        every judge covers an equal share. Panel-relative judge bias cancels in
        expectation over the run at single-judge cost; per-item verdicts are
        single-judge opinions, so use it for benchmark/run-level scores, not
        when each individual verdict must be trustworthy. Per-item ``stats``
        and ``raw_agreement`` are ``None``: one vote has no cross-judge
        agreement to report. The rotation runs over the *deduplicated* panel,
        so listing a judge twice to up-weight it only works under ``"all"``.
        ``repetitions`` still applies to the single assigned judge — N calls
        to one judge, never N judges. Inside ``evaluatorq()`` the assignment is
        keyed on the dataset row (datapoint ``i`` goes to judge
        ``i % len(panel)``), deterministic at any ``parallelism`` and across
        evaluator reuse. A direct scorer call carries no row and rotates in
        arrival order on a cursor that lives on the evaluator — only the equal
        share balance is guaranteed there. Shuffle the dataset first if its
        order is meaningful. Requires ``min_successful_judges=1``; a failed
        item degrades to inconclusive unless ``replacement_judges`` is set.

    Examples
    --------
    Boolean — "is the answer correct?" (``verdict_kind="categorical"``, the default)::

        llm_jury(name="correct", criteria="Is the answer factually correct?")

    Labeled categorical (``verdict_kind="categorical"``)::

        llm_jury(
            name="grade",
            criteria="Grade the answer.",
            labels=["correct", "partially_correct", "incorrect"],
            passing_labels=["correct", "partially_correct"],
        )

    Numeric (``verdict_kind="numeric"``)::

        llm_jury(
            name="helpfulness",
            criteria="Rate helpfulness from 0 to 1.",
            verdict_kind="numeric",
            score_range=(0.0, 1.0),
            threshold=0.7,
        )
    """
    # --- validation (fail fast) ---
    if bool(criteria) == bool(prompt):
        raise ValueError('Pass exactly one of `criteria` or `prompt`.')
    if repetitions < 1:
        raise ValueError(f'repetitions ({repetitions}) must be >= 1.')
    if verdict_kind != 'categorical' and (labels or passing_labels):
        raise ValueError("`labels`/`passing_labels` are only valid for verdict_kind='categorical'.")
    if labels is not None and len(labels) < 2:
        # A categorical judge with 0/1 options is meaningless, and an empty list
        # would build a degenerate ``Literal[()]`` verdict model.
        raise ValueError('categorical `labels` must list at least two options.')
    if passing_labels and not labels:
        # Boolean mode (categorical + no labels) derives pass/fail from the verdict
        # directly, so passing_labels would be silently ignored — reject it loudly.
        raise ValueError(
            '`passing_labels` requires `labels` (categorical labeled mode); '
            'boolean mode derives pass/fail from the verdict directly.'
        )
    if labels and passing_labels and not set(passing_labels) <= set(labels):
        raise ValueError('`passing_labels` must be a subset of `labels`.')
    if verdict_kind == 'numeric' and score_range[0] >= score_range[1]:
        raise ValueError(f'score_range {score_range} must be increasing (lo < hi).')
    if verdict_kind == 'numeric' and not (score_range[0] <= threshold <= score_range[1]):
        raise ValueError(f'threshold ({threshold}) must lie within score_range {score_range}.')
    validate_aggregator(
        aggregator,
        VerdictKind.NUMERIC if verdict_kind == 'numeric' else VerdictKind.CATEGORICAL,
    )
    panel, deduped = _resolve_and_validate_panel(
        judges=judges, model=model, min_successful_judges=min_successful_judges
    )
    _validate_assignment(assignment, min_successful_judges=min_successful_judges)
    # Round-robin assignment (CyclicJudge, arXiv:2603.01865): each datapoint is
    # scored by exactly one judge over the deduplicated panel. Inside
    # evaluatorq() the runner passes the dataset row, so the item->judge
    # mapping is `deduped[row % len(deduped)]` — deterministic at any
    # parallelism. A direct scorer call carries no row and falls back to this
    # arrival-order cycle, where only the equal share balance is guaranteed.
    # Either way every judge covers an equal share of the run, so panel-
    # relative judge bias cancels in expectation at single-judge cost. Shuffle
    # the dataset first if its order is meaningful, so the rotation cannot line
    # up with a latent grouping (the paper's own caveat).
    # Created unconditionally (itertools.cycle is lazy) and gated on the
    # assignment mode at the call site, mirroring PairwiseComparator.
    cycle = itertools.cycle(deduped)
    if temperature == 0.0:
        logger.warning(
            'temperature=0.0: reasoning models (o-series, gpt-5, …) often score worse '
            'at temp 0. Leave it unset unless you know your model benefits.'
        )

    # Resolve the client lazily on first scorer call, not here: declaring an
    # evaluator at module scope should never require credentials. An explicit
    # client= is used as-is; otherwise we resolve once on first use.
    resolved_client = client
    verdict_model = _build_verdict_model(verdict_kind=verdict_kind, labels=labels, score_range=score_range)
    template = prompt if prompt is not None else DEFAULT_TEMPLATE
    sys_prompt = (
        system_prompt
        if system_prompt is not None
        else _default_system_prompt(verdict_kind=verdict_kind, labels=labels, score_range=score_range)
    )
    vkind = VerdictKind.NUMERIC if verdict_kind == 'numeric' else VerdictKind.CATEGORICAL

    async def scorer(params: ScorerParameter) -> EvaluationResult:
        nonlocal resolved_client
        data = params['data']
        output = params['output']
        # A target that errored produced no answer to grade. Judging it anyway would
        # score an empty response as if the agent had genuinely replied that way.
        # Checked before resolving the client: skipping the judges must not require
        # LLM credentials.
        target_error = output_error_text(output)
        if target_error is not None:
            logger.warning('Target errored, skipping judges: {}', target_error)
            return EvaluationResult.model_validate({
                'value': 'inconclusive',
                'explanation': f'Not judged: the target returned an error: {target_error}',
                'pass': None,
            })
        if resolved_client is None:
            resolved_client = resolve_llm_client(config_client=None).client
        replacements = _build_replacements(data=data, output=output, criteria=criteria or '')

        async def judge_fn(judge_model: str) -> Prediction:
            return await _run_single_judge(
                client=resolved_client,
                model=judge_model,
                template=template,
                replacements=replacements,
                system_prompt=sys_prompt,
                verdict_model=verdict_model,
                structured_output=structured_output,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_ms=timeout_ms,
                extra_kwargs=extra_kwargs,
            )

        if assignment == 'cyclic':
            # Runner path: key on the dataset row so the mapping is reproducible
            # regardless of parallelism or arrival order. Direct invocation has
            # no row and falls back to the arrival-order cursor; that advance
            # happens before any await, so concurrent calls still hand out
            # exactly balanced judge shares.
            row = params.get('row')
            run_panel = [deduped[row % len(deduped)]] if row is not None else [next(cycle)]
        else:
            run_panel = panel
        deliberation = await run_jury(
            judge_fn=judge_fn,
            panel=run_panel,
            repetitions=repetitions,
            replacement_judges=replacement_judges or [],
            min_successful_judges=min_successful_judges,
            verdict_kind=vkind,
            aggregator=aggregator,
            tie_break=tie_break,
            # A lone judge with no stand-ins has no redundancy: re-raise an outage
            # loudly instead of silently returning inconclusive on every datapoint.
            # Cyclic assignment over a multi-judge panel keeps run-level redundancy
            # (other items get other judges), so a failed item degrades to
            # inconclusive instead of killing the whole run.
            propagate_errors=(len(deduped) == 1 and not replacement_judges),
        )
        if assignment == 'cyclic':
            # A cyclic item is scored by one judge, so there is no cross-judge
            # agreement to report: a single vote renders raw_agreement=1.0 and
            # std=0.0, byte-identical to a genuinely unanimous panel. Null both
            # so downstream filters cannot read a cyclic run as uniformly
            # confident (mirrors build_report's None mean_agreement).
            deliberation = deliberation.model_copy(
                update={'jury': deliberation.jury.model_copy(update={'stats': None, 'raw_agreement': None})}
            )
        return _to_evaluation_result(
            deliberation=deliberation,
            verdict_kind=verdict_kind,
            passing_labels=passing_labels,
            threshold=threshold,
            score_range=score_range,
            include_jury_record=assignment == 'cyclic',
        )

    return {'name': name, 'scorer': scorer}


# ---------------------------------------------------------------------------
# Pairwise (preference) jury
# ---------------------------------------------------------------------------


def _pairwise_system_prompt() -> str:
    return (
        'You are an impartial judge comparing two AI responses, A and B, to a '
        'question. Judge which is better against the stated criteria. Return a '
        "structured verdict: `value` must be exactly one of 'A', 'B', or 'tie', "
        'with a 2-3 sentence explanation.'
    )


# Default pairwise template. ``criteria`` is a substituted variable, matching
# DEFAULT_TEMPLATE — a `prompt=` override reaches it the same way.
DEFAULT_PAIRWISE_TEMPLATE = (
    '# Criteria\n{{criteria}}\n\n'
    '# Question\n{{question}}\n\n'
    '# Response A\n{{response_a.output.response}}\n\n'
    '# Response B\n{{response_b.output.response}}\n'
)


def _side_to_namespace(prefix: str, side: Output) -> dict[str, Any]:
    """Build the `<prefix>.*` template namespace for one pairwise side.

    A side carries an answer only, so `<prefix>.input.*` is empty; the rest of the
    namespace (`output.response`, `output.tools_called`, `output.messages`,
    `output.error`) is identical to the pointwise one.
    """
    return build_eval_replacements(
        prefix=prefix,
        input_messages=[],
        output_messages=output_to_messages(side),
        expected_output='',
        system_instructions=None,
        error=output_error_text(side),
    )


class PairwiseComparator:
    """A configured pairwise LLM jury. Call :meth:`compare` on an A/B pair."""

    def __init__(
        self,
        *,
        panel: list[str],
        criteria: str,
        template: str | None = None,
        system_prompt: str,
        swap: bool,
        repetitions: int,
        assignment: Literal['all', 'cyclic'] = 'all',
        replacement_judges: list[str] | None,
        min_successful_judges: int,
        max_tokens: int,
        timeout_ms: int,
        temperature: float | None,
        structured_output: bool,
        extra_kwargs: dict[str, Any] | None,
        client: Any,
        max_concurrency: int | None = None,
    ) -> None:
        self._panel = panel
        self._criteria = criteria
        self._system_prompt = system_prompt
        self._swap = swap
        self._repetitions = repetitions
        self._assignment = assignment
        # Round-robin cursor for assignment='cyclic'; advanced synchronously at
        # compare() entry, so concurrent compares still balance judge shares.
        self._cycle = itertools.cycle(panel)
        self._replacement_judges = replacement_judges
        self._min_successful_judges = min_successful_judges
        self._max_tokens = max_tokens
        self._timeout_ms = timeout_ms
        self._temperature = temperature
        self._structured_output = structured_output
        self._extra_kwargs = extra_kwargs
        self._client = client
        if max_concurrency is not None and max_concurrency < 1:
            raise ValueError(f'max_concurrency ({max_concurrency}) must be >= 1.')
        self._max_concurrency = max_concurrency
        self._semaphore: asyncio.Semaphore | None = None
        self._semaphore_loop: asyncio.AbstractEventLoop | None = None
        self._verdict_model = _build_verdict_model(
            verdict_kind='categorical', labels=PAIRWISE_LABELS, score_range=(0.0, 1.0)
        )
        self._template = template if template is not None else DEFAULT_PAIRWISE_TEMPLATE

    def _current_semaphore(self) -> asyncio.Semaphore | None:
        """The shared concurrency budget, bound to the running event loop.

        One semaphore is shared by every compare() call on the same loop, so
        ``max_concurrency`` bounds TOTAL in-flight judge calls across pairs. A
        semaphore binds to the loop that first blocks on it, so when the caller
        switches loops (e.g. one ``asyncio.run`` per pair) a fresh one is minted
        instead of every judge call failing with 'bound to a different event
        loop' and silently degrading verdicts to inconclusive.
        """
        if self._max_concurrency is None:
            return None
        loop = asyncio.get_running_loop()
        if self._semaphore is None or self._semaphore_loop is not loop:
            self._semaphore = asyncio.Semaphore(self._max_concurrency)
            self._semaphore_loop = loop
        return self._semaphore

    async def compare(self, *, question: str, response_a: Output, response_b: Output) -> PairwiseComparison:
        """Run the panel over one A-vs-B comparison and return the reconciled verdict.

        A side that carries a target-level error is never judged — comparing against a
        failed generation would score noise as a preference.
        """
        for label, side in (('A', response_a), ('B', response_b)):
            err = output_error_text(side)
            if err is not None:
                logger.warning('Pairwise side {} errored, skipping judges: {}', label, err)
                return PairwiseComparison(winner='inconclusive')

        if self._client is None:
            self._client = resolve_llm_client(config_client=None).client

        async def judge_fn(first: Output, second: Output, model: str) -> Prediction:
            replacements: dict[str, Any] = {'question': question, 'criteria': self._criteria}
            replacements.update(_side_to_namespace('response_a', first))
            replacements.update(_side_to_namespace('response_b', second))
            return await _run_single_judge(
                client=self._client,
                model=model,
                template=self._template,
                replacements=replacements,
                system_prompt=self._system_prompt,
                verdict_model=self._verdict_model,
                structured_output=self._structured_output,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                timeout_ms=self._timeout_ms,
                extra_kwargs=self._extra_kwargs,
            )

        # CyclicJudge: one judge per comparison, cycling through the panel.
        # The assigned judge still runs both orderings when swap is on, so
        # position-bias reconciliation is preserved per pair.
        run_panel = [next(self._cycle)] if self._assignment == 'cyclic' else self._panel
        return await run_pairwise(
            judge_fn=judge_fn,
            panel=run_panel,
            response_a=response_a,
            response_b=response_b,
            swap=self._swap,
            repetitions=self._repetitions,
            replacement_judges=self._replacement_judges,
            min_successful_judges=self._min_successful_judges,
            propagate_errors=(len(self._panel) == 1 and not self._replacement_judges),
            max_concurrency=self._current_semaphore(),
        )


def llm_jury_pairwise(
    *,
    judges: list[str] | None = None,
    model: str | None = None,
    criteria: str | None = None,
    prompt: str | None = None,
    system_prompt: str | None = None,
    swap: bool = True,
    repetitions: int = 1,
    assignment: Literal['all', 'cyclic'] = 'all',
    replacement_judges: list[str] | None = None,
    min_successful_judges: int = 1,
    max_tokens: int = 8000,
    timeout_ms: int = 90000,
    temperature: float | None = None,
    structured_output: bool = True,
    extra_kwargs: dict[str, Any] | None = None,
    client: Any = None,
    max_concurrency: int | None = None,
) -> PairwiseComparator:
    """Build a pairwise (A-vs-B) LLM jury that reuses the shared panel machinery.

    Judges compare two responses and pick a winner ('A'/'B'/'tie'). With ``swap``
    on (default) each judge is run in both orderings to correct for position bias
    (see ADR-24). Panel/orchestration params mirror :func:`llm_jury`. ``prompt``
    overrides the built-in Mustache-style template (which exposes the
    ``response_a.*``/``response_b.*`` namespace via :func:`_side_to_namespace`);
    leave it ``None`` to use the default. Returns a :class:`PairwiseComparator`;
    call ``compare`` per A/B pair, and roll many comparisons up with
    :func:`evaluatorq.pairwise.build_report`.

    ``max_concurrency`` caps TOTAL in-flight judge LLM calls across all
    concurrently running ``compare`` calls on the returned comparator (each
    pair fans out judges x orderings x repetitions). ``None`` (default) keeps
    the fan-out unbounded.

    ``assignment="cyclic"`` (CyclicJudge, arXiv:2603.01865) gives each
    comparison exactly one judge, cycling through the panel so every judge
    covers an equal share of the run. Judge bias cancels in expectation over
    many comparisons at single-judge cost, and the assigned judge still runs
    both orderings when ``swap`` is on. Per-pair winners are single-judge
    opinions — roll them up with ``build_report`` and read run-level rates.
    The rotation runs over the deduplicated panel (duplicate judge entries add
    weight only under ``"all"``), ``repetitions`` still applies to the single
    assigned judge, and the cursor lives on the comparator: a reused
    comparator continues where the previous run stopped, so exact balance
    holds per freshly built comparator, not per run. Shuffle your pairs first
    if their order is meaningful. Requires ``min_successful_judges=1``.
    """
    if repetitions < 1:
        raise ValueError(f'repetitions ({repetitions}) must be >= 1.')
    _panel, deduped = _resolve_and_validate_panel(
        judges=judges, model=model, min_successful_judges=min_successful_judges
    )
    resolved_criteria = criteria or DEFAULT_PAIRWISE_CRITERIA
    _validate_assignment(assignment, min_successful_judges=min_successful_judges)
    return PairwiseComparator(
        # Pass the deduped panel so the comparator's no-redundancy check
        # (`len(panel) == 1`) matches the panel run_jury actually runs, keeping
        # propagate_errors in step with llm_jury (judges=['m','m'] -> one judge).
        panel=deduped,
        criteria=resolved_criteria,
        template=prompt,  # None -> DEFAULT_PAIRWISE_TEMPLATE
        system_prompt=system_prompt if system_prompt is not None else _pairwise_system_prompt(),
        swap=swap,
        repetitions=repetitions,
        assignment=assignment,
        replacement_judges=replacement_judges,
        min_successful_judges=min_successful_judges,
        max_tokens=max_tokens,
        timeout_ms=timeout_ms,
        temperature=temperature,
        structured_output=structured_output,
        extra_kwargs=extra_kwargs,
        client=client,
        max_concurrency=max_concurrency,
    )

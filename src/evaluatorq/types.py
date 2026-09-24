import json
from collections.abc import Awaitable, Callable, Sequence
from datetime import datetime, timezone
from typing import Any, ClassVar, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_serializer, model_validator
from typing_extensions import NotRequired, TypedDict

from evaluatorq.contracts import AgentResponse, Message, TokenUsage

# Keep output permissive: OpenResponses payloads are dict-shaped and should
# pass through unchanged alongside arbitrary job payloads.
Output = str | int | float | bool | dict[str, Any] | list[Message] | AgentResponse | None
"""Output type alias"""


EvaluationResultCellValue = str | int | float | dict[str, str | float | dict[str, str | float]]


class EvaluationResultCell(BaseModel):
    type: str
    value: dict[str, EvaluationResultCellValue]


def _is_evaluation_result_cell_dict(value: dict[str, Any]) -> bool:
    return isinstance(value.get('type'), str) and isinstance(value.get('value'), dict)


def _json_default(obj: Any) -> Any:
    """Fallback for json.dumps: convert Pydantic models to dicts, else str."""
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode='json')
    return str(obj)


class EvaluationResult(BaseModel):
    """The score a scorer function returns for one job output.

    Example:
        ```python
        from evaluatorq import EvaluationResult

        async def length_check_scorer(params):
            output = params["output"]
            return EvaluationResult(value=1 if len(output) > 10 else 0)
        ```
    """

    model_config: ClassVar[ConfigDict] = {'populate_by_name': True}

    value: str | int | float | bool | EvaluationResultCell | dict[str, Any]
    explanation: str | None = None
    pass_: bool | None = Field(default=None, alias='pass')
    # Optional evaluator-cost metadata: a scorer that calls an LLM judge can report
    # the judge's token usage and raw response here. The red-team report layer reads
    # them off the live score object, and they are kept in local result dumps
    # (e.g. 02_attack_results.json). They are intentionally stripped from the Orq
    # platform upload at the send boundary (see evaluatorq.send_results), not here.
    token_usage: TokenUsage | None = None
    raw_output: dict[str, Any] | None = None

    @field_serializer('value', when_used='json')
    def serialize_value(
        self,
        value: str | float | bool | EvaluationResultCell | dict[str, Any],  # noqa: FBT001
    ) -> Any:
        if isinstance(value, dict) and not _is_evaluation_result_cell_dict(value):
            return json.dumps(value, default=_json_default)

        return value


class EvaluatorScore(BaseModel):
    evaluator_name: str = Field(serialization_alias='evaluatorName')
    score: EvaluationResult
    error: str | None = None


class JobResult(BaseModel):
    job_name: str = Field(serialization_alias='jobName')
    output: Output = Field(union_mode='left_to_right')
    error: str | None = None
    evaluator_scores: list[EvaluatorScore] | None = Field(default=None, serialization_alias='evaluatorScores')

    @field_serializer('output', when_used='json')
    def serialize_output(self, output: Output) -> Any:
        if isinstance(output, AgentResponse):
            return output.model_dump(mode='json')
        if isinstance(output, dict) and output.get('object') != 'response':
            return json.dumps(output, default=_json_default)
        return output


class _DataPointDictRequired(TypedDict):
    """Required fields for DataPointDict."""

    inputs: dict[str, Any]


class DataPointDict(_DataPointDictRequired, total=False):
    """Dict representation of a DataPoint for type checking."""

    expected_output: Output | None


class DataPoint(BaseModel):
    """
    A data point for evaluation.

    Args:
        inputs: The inputs to pass to the job.
        expected_output: The expected output of the data point.
                        Used for evaluation and comparing the output of the job.

    Example:
        ```python
        from evaluatorq import DataPoint

        DataPoint(inputs={"text": "Hello world"}, expected_output="HELLO WORLD")
        ```
    """

    inputs: dict[str, Any]
    expected_output: Output | None = Field(default=None, serialization_alias='expectedOutput')


DataPointInput = DataPoint | DataPointDict
"""Type alias for DataPoint that accepts both model instances and dicts."""


class DataPointResult(BaseModel):
    data_point: DataPoint = Field(serialization_alias='dataPoint')
    error: str | None = None
    job_results: list[JobResult] | None = Field(default=None, serialization_alias='jobResults')


DataPointComplete = Callable[[DataPointResult], Awaitable[None] | None]


EvaluatorqResult = list[DataPointResult]
"""Type alias for evaluation results"""


class JobReturn(TypedDict):
    """Job return structure.

    ``error`` is optional and reports a failure the job *handled* rather than raised —
    ``None`` on success, the reason otherwise. A row whose ``error`` flattens to a
    non-empty string is counted in the summary table's ``Failed Jobs`` and fails
    ``check_pass_failures(treat_errors_as_failure=True)``. It keeps its output for
    diagnosis but its evaluators are **skipped** — scoring a transcript already known
    to be dead buys nothing and costs an LLM judge call per row. An
    omitted key and an explicit ``None`` are indistinguishable to the consumer —
    emitting ``None`` on success is a producer convention, so that a job that forgot
    the key cannot be mistaken for one that reported a clean run. A job that lets its
    failures raise omits the key. Note that ``Job`` types this return as
    ``dict[str, Any]``, so nothing type-checks a job against this shape.
    """

    name: str
    output: Output
    error: NotRequired[str | None]


Job = Callable[[DataPoint, int], Awaitable[dict[str, Any]]]
"""Job function type - returns a ``JobReturn``-shaped dict ('name', 'output', optional 'error')"""


class ScorerParameter(TypedDict):
    """Parameters passed to a scorer function.

    Args:
        data: The data point being evaluated.
        output: The output produced by the job for the data point.
        row: Zero-based dataset index of the data point. Present when the
            scorer runs inside ``evaluatorq()``; absent on direct invocation.
            Lets evaluators key per-item decisions (e.g. cyclic judge
            assignment) on the dataset position rather than call-arrival order.
    """

    data: DataPoint
    output: Output
    row: NotRequired[int]


Scorer = Callable[[ScorerParameter], Awaitable[EvaluationResult | dict[str, Any]]]


class Evaluator(TypedDict):
    """A named scorer passed to `evaluatorq`'s ``evaluators`` list.

    Example:
        ```python
        from evaluatorq import EvaluationResult

        async def length_check_scorer(params):
            return EvaluationResult(value=1 if len(params["output"]) > 10 else 0)

        evaluator = {"name": "length-check", "scorer": length_check_scorer}
        ```
    """

    name: str
    scorer: Scorer
    # Optional evaluator kind (e.g. "code_eval"). When set, the tracing layer
    # emits the flat gen_ai.evaluation.* / orq.evaluator.* attributes the Orq
    # trace UI uses to classify + render an evaluator span. Absent for evaluators
    # (e.g. red-team) that should keep the legacy orq.score-only span shape.
    evaluator_type: NotRequired[str]


class DatasetIdInput(BaseModel):
    """Input for fetching a dataset from Orq platform."""

    dataset_id: str
    include_messages: bool = False


class ExperimentInput(BaseModel):
    """Input for sourcing pre-recorded responses from an Orq experiment.

    Used with ``inference=False`` to re-run evaluators against the responses an
    earlier experiment already produced, without regenerating them.
    """

    experiment_id: str
    """The experiment ID to load responses from. Read it off the experiment URL in the
    Orq UI (``/experiments/<experiment_id>``). The API refers to experiments as
    "spreadsheets", so you will also see this ID in ``/v2/spreadsheets/<id>`` routes."""
    run_id: str | None = None
    """A specific run ID (a "manifest" in the API). When omitted, the latest run is used.
    Every execution of an experiment creates a new run; open it from the experiment's run
    history to read its ID from the URL."""


DEFAULT_TRACE_QUERY_LIMIT = 20
"""How many traces a `TraceInput` query mode fetches when the caller names no limit."""


class TraceInput(BaseModel):
    """Describe one exact trace/span, one trace, or a bounded trace search.

    The three modes are exclusive and the validator enforces that, ``limit``
    included: in trace mode it is never read, so accepting it would answer
    ``limit=50`` with exactly one trace and no warning. Read the resolved count
    off `query_limit` rather than the field.

    ``start_time`` / ``end_time`` are stored timezone-aware: a naive value is read
    as UTC rather than as the host's local time, so one query means one window
    wherever it runs.
    """

    model_config = ConfigDict(extra='forbid')

    trace_id: str | None = None
    span_id: str | None = None
    limit: int | None = Field(default=None, ge=1)
    """Query mode only. ``None`` means "not set", which is what lets the validator
    tell an explicit ``limit=`` apart from the default it would otherwise assume."""
    start_time: datetime | None = None
    end_time: datetime | None = None
    search: str = ''
    filters: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def query_limit(self) -> int:
        """The number of traces query mode fetches."""
        return self.limit if self.limit is not None else DEFAULT_TRACE_QUERY_LIMIT

    @model_validator(mode='after')
    def _validate_source(self) -> 'TraceInput':
        # Naive means UTC: the query fields are epoch milliseconds, so local time shifted the window per host.
        for field in ('start_time', 'end_time'):
            value = getattr(self, field)
            if value is not None and value.tzinfo is None:
                object.__setattr__(self, field, value.replace(tzinfo=timezone.utc))
        if self.span_id is not None and self.trace_id is None:
            raise ValueError('span_id requires trace_id.')
        if self.trace_id is not None and (
            self.search or self.filters or self.start_time or self.end_time or self.limit is not None
        ):
            raise ValueError('trace_id cannot be combined with trace search criteria or limit.')
        if self.start_time and self.end_time and self.start_time > self.end_time:
            raise ValueError('start_time must be before end_time.')
        return self


TraceMessageFormat = Literal['chat_completions', 'responses', 'otel_genai', 'mixed']


def _trace_messages_dump(messages: list[Message]) -> list[dict[str, Any]]:
    return [message.model_dump(mode='json', exclude_none=True) for message in messages]


class Trace(BaseModel):
    """A normalized Orq trace exchange and its source metadata.

    ``input_messages`` and ``output_messages`` retain their source-side
    boundaries. ``messages`` is a derived transcript for consumers that need a
    single conversation, with only the largest exact overlap removed.
    """

    trace_id: str
    requested_span_id: str | None = None
    message_span_id: str | None = None
    message_format: TraceMessageFormat | None = None
    input_messages: list[Message] = Field(default_factory=list)
    output_messages: list[Message] = Field(default_factory=list)
    query: str | None = None
    retrievals: list[str] = Field(default_factory=list)
    tools_called: list[str] = Field(default_factory=list)
    session_id: str | None = None
    actor_id: str | None = None
    thread_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    expected_output: Output | None = None
    import_error: str | None = None

    @model_validator(mode='after')
    def _failed_traces_carry_no_messages(self) -> 'Trace':
        """A failed import may not also look like a usable exchange.

        ``import_error`` lives on the success type, so "check import_error first"
        would otherwise be a convention every consumer has to remember. Forbidding
        the mixed state makes a half-imported trace impossible to construct
        instead of merely discouraged.
        """
        if self.import_error is not None and (self.input_messages or self.output_messages):
            raise ValueError('A Trace with import_error set cannot carry input or output messages.')
        return self

    @property
    def messages(self) -> list[Message]:
        """Return input followed by output without one duplicated boundary."""
        overlap = 0
        input_dump = _trace_messages_dump(self.input_messages)
        output_dump = _trace_messages_dump(self.output_messages)
        for size in range(min(len(input_dump), len(output_dump)), 0, -1):
            if input_dump[-size:] == output_dump[:size]:
                overlap = size
                break
        return [*self.input_messages, *self.output_messages[overlap:]]

    def to_datapoint(self) -> DataPoint:
        """Convert this imported trace to evaluatorq's native row shape."""
        inputs: dict[str, Any] = {
            'messages': _trace_messages_dump(self.messages),
            'recorded_output': _trace_messages_dump(self.output_messages),
            'query': self.query,
            'source_trace_id': self.trace_id,
            'source_requested_span_id': self.requested_span_id,
            'source_span_id': self.message_span_id,
            'message_format': self.message_format,
            'retrievals': list(self.retrievals),
            'tools_called': list(self.tools_called),
            'session_id': self.session_id,
            'actor_id': self.actor_id,
            'thread_id': self.thread_id,
            'trace_metadata': dict(self.metadata),
        }
        if self.import_error is not None:
            inputs['trace_import_error'] = self.import_error
        return DataPoint(inputs=inputs, expected_output=self.expected_output)


class EvaluatorParams(BaseModel):
    """
    Parameters for running an evaluation.

    Args:
        data: The data to evaluate. A DatasetIdInput to fetch from Orq platform, an
              ExperimentInput to replay an experiment's recorded responses (requires
              inference=False), a TraceInput to import recorded trace conversations
              (requires inference=False), or a list of DataPoint instances/awaitables.
        jobs: The jobs to run on the data.
        evaluators: The evaluators to use. If not provided, only jobs will run.
        datapoint_parallelism: Number of datapoints to process in parallel. Defaults
              to 10; set to 1 for sequential execution. Accepts the former name
              ``parallelism``, which is deprecated.
        llm_parallelism: Ceiling on in-flight LLM requests for the whole
              run, counted per request rather than per task. Unbounded by default.
              Use this against a provider concurrency limit — one datapoint can
              issue many requests, so the datapoint count cannot be sized against one.
        print_results: Whether to print results table to console. Defaults to True.
                       Also accepts "print" as an alias.
        description: Optional description for the evaluation run.
        path: Optional path (e.g. "MyProject/MyFolder") to place the experiment
              in a specific project and folder on the Orq platform.
        single_trace: Group every row of the run under one ``evaluatorq.run``
              span, so the whole evaluation is a single trace. Off by default:
              each row's ``orq.job`` is its own root, i.e. one trace per row.
    """

    model_config: ClassVar[ConfigDict] = {
        'arbitrary_types_allowed': True,
        'populate_by_name': True,
    }

    data: DatasetIdInput | ExperimentInput | TraceInput | Sequence[Awaitable[DataPoint] | DataPointInput]
    jobs: list[Job] | None = None
    evaluators: list[Evaluator] | None = None
    datapoint_parallelism: int = Field(
        default=10,
        ge=1,
        validation_alias=AliasChoices('datapoint_parallelism', 'parallelism'),
    )
    llm_parallelism: int | None = Field(default=None, ge=1)
    print_results: bool = Field(default=True, validation_alias='print')
    description: str | None = None
    path: str | None = None
    inference: bool | None = None
    """When False, skip generation and evaluate the pre-recorded response in each
    row's ``messages`` column instead of running ``jobs``.

    ``None`` resolves from ``data``: a replay source (`ExperimentInput`,
    `TraceInput`) is a recorded-output source by construction and resolves to
    False, anything else to True. Passing ``inference=True`` with a replay source
    still raises — the flag was never able to mean anything else there, and
    requiring it taught users the mode only by making the obvious call fail."""
    single_trace: bool = False
    """When True, open one ``evaluatorq.run`` span for the whole run so every row
    shares a trace. Default False keeps each row's ``orq.job`` as its own root."""
    on_datapoint_complete: DataPointComplete | None = None

    @model_validator(mode='after')
    def _require_jobs_when_inferring(self) -> 'EvaluatorParams':
        replay_source = isinstance(self.data, (ExperimentInput, TraceInput))
        if self.inference is True and isinstance(self.data, ExperimentInput):
            raise ValueError(
                'data=ExperimentInput(...) sources pre-recorded responses from an '
                'experiment and is only valid with inference=False.'
            )
        if self.inference is True and isinstance(self.data, TraceInput):
            raise ValueError(
                'data=TraceInput(...) sources recorded responses from traces and is only valid with inference=False.'
            )
        if self.inference is None:
            self.inference = not replay_source
        if self.inference and not self.jobs:
            raise ValueError("'jobs' is required unless inference=False")
        return self

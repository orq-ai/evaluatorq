"""ATIF (Agent Trajectory Interchange Format) v1.7 and v1.8 models.

Field names and types mirror Harbor's reference models. Versions 1.7 and 1.8
differ only by the `audio` content part, so one model set serves both.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

if TYPE_CHECKING:
    from evaluatorq.formats.chat import ChatConversation
    from evaluatorq.formats.otel import OtelTrace
    from evaluatorq.formats.responses import ResponsesConversation

SUPPORTED_VERSIONS: tuple[str, ...] = ('ATIF-v1.7', 'ATIF-v1.8')

AudioMediaType = Literal[
    'audio/wav', 'audio/mpeg', 'audio/mp4', 'audio/aac', 'audio/ogg', 'audio/flac', 'audio/webm', 'audio/aiff'
]

_AUDIO_ALIASES = {
    'audio/mp3': 'audio/mpeg',
    'audio/mpga': 'audio/mpeg',
    'audio/x-mpeg': 'audio/mpeg',
    'audio/x-wav': 'audio/wav',
    'audio/wave': 'audio/wav',
    'audio/vnd.wave': 'audio/wav',
    'audio/x-m4a': 'audio/mp4',
    'audio/m4a': 'audio/mp4',
    'audio/x-aac': 'audio/aac',
    'audio/x-flac': 'audio/flac',
    'audio/x-aiff': 'audio/aiff',
}

_FROZEN = ConfigDict(extra='forbid', frozen=True)


class AtifImageSource(BaseModel):
    """Image stored as a file or at a URL."""

    model_config = _FROZEN
    media_type: Literal['image/jpeg', 'image/png', 'image/gif', 'image/webp']
    path: str


class AtifAudioSource(BaseModel):
    """Audio stored as a file or at a URL (ATIF-v1.8)."""

    model_config = _FROZEN
    media_type: AudioMediaType
    path: str
    duration_sec: float | None = Field(default=None, ge=0)

    @field_validator('media_type', mode='before')
    @classmethod
    def _normalise_media_type(cls, value: Any) -> Any:
        if isinstance(value, str):
            lowered = value.strip().lower()
            return _AUDIO_ALIASES.get(lowered, lowered)
        return value


class AtifContentPart(BaseModel):
    """One part of a multimodal message or observation."""

    model_config = _FROZEN
    type: Literal['text', 'image', 'audio']
    text: str | None = None
    source: AtifImageSource | AtifAudioSource | None = None

    @model_validator(mode='after')
    def _check_kind(self) -> AtifContentPart:
        if self.type == 'text':
            if self.text is None:
                msg = "'text' field is required when type='text'"
                raise ValueError(msg)
            if self.source is not None:
                msg = "'source' field is not allowed when type='text'"
                raise ValueError(msg)
            return self
        if self.source is None:
            msg = f"'source' field is required when type='{self.type}'"
            raise ValueError(msg)
        if self.text is not None:
            msg = f"'text' field is not allowed when type='{self.type}'"
            raise ValueError(msg)
        expected = AtifImageSource if self.type == 'image' else AtifAudioSource
        if not isinstance(self.source, expected):
            msg = f"type='{self.type}' requires a {expected.__name__}, got media_type {self.source.media_type!r}"
            raise ValueError(msg)  # noqa: TRY004 - pydantic turns only ValueError into ValidationError
        return self


class AtifToolCall(BaseModel):
    """A tool call within a step."""

    model_config = _FROZEN
    tool_call_id: str
    function_name: str
    arguments: dict[str, Any]
    extra: dict[str, Any] | None = None


class AtifSubagentRef(BaseModel):
    """Reference to a delegated subagent trajectory (embedded by `trajectory_id`, or external by path)."""

    model_config = _FROZEN
    trajectory_id: str | None = None
    session_id: str | None = None
    trajectory_path: str | None = None
    extra: dict[str, Any] | None = None

    @model_validator(mode='after')
    def _check_resolvable(self) -> AtifSubagentRef:
        if self.trajectory_id is None and self.trajectory_path is None:
            msg = (
                'SubagentTrajectoryRef must be resolvable: set trajectory_id or trajectory_path '
                '(session_id alone is not a key)'
            )
            raise ValueError(msg)
        return self


class AtifObservationResult(BaseModel):
    """One result inside an observation."""

    model_config = _FROZEN
    source_call_id: str | None = None
    content: str | list[AtifContentPart] | None = None
    subagent_trajectory_ref: list[AtifSubagentRef] | None = None
    extra: dict[str, Any] | None = None


class AtifObservation(BaseModel):
    """Environment feedback after actions."""

    model_config = _FROZEN
    results: list[AtifObservationResult]


class AtifMetrics(BaseModel):
    """LLM operational data for one step."""

    model_config = _FROZEN
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None
    cost_usd: float | None = None
    prompt_token_ids: list[int] | None = None
    completion_token_ids: list[int] | None = None
    logprobs: list[float] | None = None
    extra: dict[str, Any] | None = None


class AtifFinalMetrics(BaseModel):
    """Aggregate statistics for a trajectory."""

    model_config = _FROZEN
    total_prompt_tokens: int | None = None
    total_completion_tokens: int | None = None
    total_cached_tokens: int | None = None
    total_cost_usd: float | None = None
    total_steps: int | None = Field(default=None, ge=0)
    extra: dict[str, Any] | None = None


class AtifAgent(BaseModel):
    """Agent configuration."""

    model_config = _FROZEN
    name: str
    version: str
    model_name: str | None = None
    tool_definitions: list[dict[str, Any]] | None = None
    extra: dict[str, Any] | None = None


class AtifStep(BaseModel):
    """A single step."""

    model_config = _FROZEN
    step_id: int = Field(ge=1)
    timestamp: str | None = None
    source: Literal['system', 'user', 'agent']
    model_name: str | None = None
    reasoning_effort: str | float | None = None
    message: str | list[AtifContentPart]
    reasoning_content: str | None = None
    tool_calls: list[AtifToolCall] | None = None
    observation: AtifObservation | None = None
    metrics: AtifMetrics | None = None
    is_copied_context: bool | None = None
    llm_call_count: int | None = Field(default=None, ge=0)
    extra: dict[str, Any] | None = None

    @field_validator('timestamp')
    @classmethod
    def _check_timestamp(cls, value: str | None) -> str | None:
        if value is not None:
            try:
                datetime.fromisoformat(value.replace('Z', '+00:00'))
            except ValueError as exc:
                msg = f'Invalid ISO 8601 timestamp: {exc}'
                raise ValueError(msg) from exc
        return value

    @model_validator(mode='after')
    def _check_agent_only(self) -> AtifStep:
        if self.source != 'agent':
            for name in ('model_name', 'reasoning_effort', 'reasoning_content', 'tool_calls', 'metrics'):
                if getattr(self, name) is not None:
                    msg = f"Field '{name}' is only applicable when source is 'agent', but source is '{self.source}'"
                    raise ValueError(msg)
        if self.llm_call_count == 0 and self.source == 'agent':
            for name in ('metrics', 'reasoning_content'):
                if getattr(self, name) is not None:
                    msg = f"Field '{name}' must be absent when llm_call_count is 0"
                    raise ValueError(msg)
        return self


def _parts_have_audio(content: str | list[AtifContentPart] | None) -> bool:
    return isinstance(content, list) and any(part.type == 'audio' for part in content)


class AtifTrajectory(BaseModel):
    """An agent run as an ATIF-v1.7 or v1.8 trajectory.

    Reading accepts `ATIF-v1.7` and `ATIF-v1.8` only. `to_json` keeps the trajectory's own
    `schema_version` (v1.7 for one built by a converter), upgrading to v1.8 when any content
    part in the document (embedded subagents included) is audio.
    """

    model_config = _FROZEN
    schema_version: Literal['ATIF-v1.7', 'ATIF-v1.8'] = 'ATIF-v1.7'
    session_id: str | None = None
    trajectory_id: str | None = None
    agent: AtifAgent
    steps: list[AtifStep] = Field(min_length=1)
    notes: str | None = None
    final_metrics: AtifFinalMetrics | None = None
    continued_trajectory_ref: str | None = None
    extra: dict[str, Any] | None = None
    subagent_trajectories: list[AtifTrajectory] | None = None

    @model_validator(mode='before')
    @classmethod
    def _check_version(cls, data: Any) -> Any:
        if isinstance(data, dict):
            version = data.get('schema_version')
            if version is None:
                msg = 'schema_version is required (supported: ATIF-v1.7, ATIF-v1.8)'
                raise ValueError(msg)
            if version not in SUPPORTED_VERSIONS:
                msg = f'Unsupported ATIF schema_version {version!r}; supported: {", ".join(SUPPORTED_VERSIONS)}'
                raise ValueError(msg)
        return data

    @model_validator(mode='after')
    def _check_structure(self) -> AtifTrajectory:
        for index, step in enumerate(self.steps):
            if step.step_id != index + 1:
                msg = f'steps[{index}].step_id: expected {index + 1} (sequential from 1), got {step.step_id}'
                raise ValueError(msg)
            if step.observation is None:
                continue
            known = {call.tool_call_id for call in step.tool_calls or []}
            for result in step.observation.results:
                if result.source_call_id is not None and result.source_call_id not in known:
                    msg = (
                        f"Observation result references source_call_id '{result.source_call_id}' "
                        f"which is not found in step {step.step_id}'s tool_calls"
                    )
                    raise ValueError(msg)
        seen: set[str] = set()
        for index, sub in enumerate(self.subagent_trajectories or []):
            if sub.trajectory_id is None:
                msg = f'subagent_trajectories[{index}].trajectory_id is required for embedded subagents'
                raise ValueError(msg)
            if sub.trajectory_id in seen:
                msg = f'subagent_trajectories[{index}].trajectory_id {sub.trajectory_id!r} is not unique'
                raise ValueError(msg)
            seen.add(sub.trajectory_id)
        return self

    def has_audio(self) -> bool:
        """Return True if any content part in this trajectory or an embedded subagent is audio."""
        for step in self.steps:
            if _parts_have_audio(step.message):
                return True
            if step.observation and any(_parts_have_audio(r.content) for r in step.observation.results):
                return True
        return any(sub.has_audio() for sub in self.subagent_trajectories or [])

    def to_json_dict(self, *, version: Literal['1.7', '1.8'] | None = None) -> dict[str, Any]:
        """Return the JSON-ready dict, stamping one schema_version on the whole document."""
        audio = self.has_audio()
        if version == '1.7' and audio:
            msg = 'Cannot write ATIF-v1.7: the trajectory contains audio content parts (needs ATIF-v1.8)'
            raise ValueError(msg)
        stamp = f'ATIF-v{version}' if version else ('ATIF-v1.8' if audio else self.schema_version)
        return _stamp(self.model_dump(mode='json', exclude_none=True), stamp)

    def to_json(self, *, version: Literal['1.7', '1.8'] | None = None, indent: int | None = None) -> str:
        """Serialise to ATIF JSON. Version defaults to the trajectory's own, or 1.8 when audio is present."""
        return json.dumps(self.to_json_dict(version=version), indent=indent)

    def to_chat(self) -> ChatConversation:
        """Convert to chat. Lossy: see `convert_responses_atif` and `convert_chat_responses` for the losses."""
        from evaluatorq.formats import convert_chat_responses, convert_responses_atif

        return convert_chat_responses.responses_to_chat(convert_responses_atif.atif_to_responses(self))

    def to_responses(self) -> ResponsesConversation:
        """Convert to a Responses transcript (items plus per-call `Response` metadata)."""
        from evaluatorq.formats import convert_responses_atif

        return convert_responses_atif.atif_to_responses(self)

    def to_otel(self) -> OtelTrace:
        """Convert to one OTel trace (root `invoke_agent`, `chat` and `execute_tool` children)."""
        from evaluatorq.formats import convert_otel_atif

        return convert_otel_atif.atif_to_otel(self)


def _stamp(doc: dict[str, Any], stamp: str) -> dict[str, Any]:
    doc['schema_version'] = stamp
    for sub in doc.get('subagent_trajectories') or []:
        _stamp(sub, stamp)
    return doc

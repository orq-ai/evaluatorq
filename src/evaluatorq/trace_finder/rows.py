"""Table rows built from Orq trace-search summaries, with no span hydration."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any, NamedTuple

from pydantic import BaseModel, ConfigDict, Field


class TokenUsage(NamedTuple):
    """Normalised token counts for one trace; ``None`` means the source did not report it."""

    tokens_in: int | None
    tokens_out: int | None
    cached: int | None
    cache_write: int | None
    reasoning: int | None
    exclusive: bool


def _get(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _int(value: Any) -> int | None:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _float(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _str(value: Any) -> str | None:
    return str(value) if value not in (None, '') else None


def _strs(value: Any) -> tuple[str, ...]:
    return tuple(str(item) for item in value if item) if isinstance(value, (list, tuple)) else ()


def parse_time(value: Any) -> datetime | None:
    """Parse an Orq timestamp (datetime, ISO string, or epoch s/ms) to an aware UTC datetime."""
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            # Python 3.10's fromisoformat rejects a trailing Z, which the Orq API sends.
            parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
        except ValueError:
            return None
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            parsed = datetime.fromtimestamp(value / 1000 if value > 10_000_000_000 else value, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    else:
        return None
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def normalise_usage(usage: Any) -> TokenUsage:
    """Return input/output tokens under the Orq router's inclusive convention.

    Router rollups count cache reads and writes inside ``prompt_tokens`` for both OpenAI and
    Anthropic (verified live 2026-09-27). A span instrumented outside the router reports native
    Anthropic usage, where they are excluded; that shows as ``cached + cache_write > prompt``,
    and the input is then ``prompt + cached + cache_write``. The caller logs how often that fired.
    """
    if usage is None:
        return TokenUsage(None, None, None, None, None, exclusive=False)
    prompt = _int(_get(usage, 'prompt_tokens'))
    output = _int(_get(usage, 'completion_tokens'))
    cached = _int(_get(usage, 'prompt_cached_tokens'))
    write = _int(_get(usage, 'prompt_cache_creation_tokens'))
    reasoning = _int(_get(usage, 'completion_reasoning_tokens'))
    if prompt is None:
        return TokenUsage(None, output, cached, write, reasoning, exclusive=False)
    extra = (cached or 0) + (write or 0)
    if extra > prompt:
        return TokenUsage(prompt + extra, output, cached, write, reasoning, exclusive=True)
    return TokenUsage(prompt, output, cached, write, reasoning, exclusive=False)


class TraceRow(BaseModel):
    """One trace-search summary as the explorer table shows it."""

    model_config = ConfigDict(frozen=True)

    trace_id: str
    name: str | None = None
    operation: str | None = None
    status: str | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None
    duration_ms: int | None = None
    project_id: str | None = None
    session_id: str | None = None
    thread_id: str | None = None
    product: str | None = None
    providers: tuple[str, ...] = ()
    models: tuple[str, ...] = ()
    agent_name: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    cached_tokens: int | None = None
    cache_write_tokens: int | None = None
    reasoning_tokens: int | None = None
    usage_exclusive: bool = False
    cost_total: float | None = None
    currency: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict, repr=False)

    @property
    def cache_pct(self) -> float | None:
        """Cached share of the input; ``None`` when either side is unknown or the input is zero."""
        if not self.tokens_in or self.cached_tokens is None:
            return None
        return self.cached_tokens / self.tokens_in

    @property
    def is_error(self) -> bool:
        """Whether the source marked this trace as failed."""
        return (self.status or '').lower() in {'error', 'failed'}


def row_from_summary(summary: Any, raw: Mapping[str, Any] | None) -> TraceRow | None:
    """Build a row from the raw search payload, falling back to the typed SDK summary per field."""

    def pick(name: str) -> Any:
        value = _get(raw, name) if raw is not None else None
        return value if value is not None else _get(summary, name)

    trace_id = pick('trace_id') or pick('id')
    if not trace_id:
        return None
    usage = normalise_usage(pick('usage'))
    cost = pick('cost')
    agent = pick('agent')
    return TraceRow(
        trace_id=str(trace_id),
        name=_str(pick('name')),
        operation=_str(pick('operation')),
        status=_str(pick('status')),
        started_at=parse_time(pick('started_at')),
        ended_at=parse_time(pick('ended_at')),
        duration_ms=_int(pick('duration_ms')),
        project_id=_str(pick('project_id')),
        session_id=_str(pick('session_id')),
        thread_id=_str(pick('thread_id')),
        product=_str(pick('product')),
        providers=_strs(pick('providers')),
        models=_strs(pick('models')),
        agent_name=_str(_get(agent, 'name')) if agent is not None else _str(pick('agent_name')),
        tokens_in=usage.tokens_in,
        tokens_out=usage.tokens_out,
        cached_tokens=usage.cached,
        cache_write_tokens=usage.cache_write,
        reasoning_tokens=usage.reasoning,
        usage_exclusive=usage.exclusive,
        cost_total=_float(_get(cost, 'total')) if cost is not None else None,
        currency=_str(_get(cost, 'currency')) if cost is not None else None,
        raw=dict(raw) if raw is not None else {},
    )

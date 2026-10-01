"""Signal configuration: the rule knobs, tool roles, and the calibrated tag thresholds."""

from __future__ import annotations

import functools
import json
import math
import os
from importlib import resources
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

ErrorDetection = Literal['status', 'status_and_content']
Canonicalisation = Literal['sorted_keys', 'exact']
RetryDefinition = Literal['consecutive_same_tool', 'same_tool_args_within_n']
ToolRole = Literal['bash', 'webview', 'skill', 'subagent', 'other']

DEFAULT_ERROR_PATTERNS: tuple[str, ...] = ('Error:', 'Traceback', '<tool_use_error>')
DEFAULT_EMPTY_VALUES: frozenset[str] = frozenset({'', '[]', '{}', 'null'})

SUBAGENT_TOOL_NAMES: frozenset[str] = frozenset({'Agent', 'Task', 'call_sub_agent'})
"""Tool names that spawn a subagent: Claude Code (Agent, Task) and orq agents (call_sub_agent)."""
SHELL_TOOL_NAMES: frozenset[str] = frozenset({
    'Bash',
    'bash',
    'exec_command',
    'functions.exec_command',
    'shell_command',
})
WEBFETCH_TOOL_NAMES: frozenset[str] = frozenset({'WebFetch', 'web_fetch', 'WebView', 'webview'})
SKILL_TOOL_NAMES: frozenset[str] = frozenset({'Skill'})
"""Tools that load a skill by name. Claude Code's `Skill` takes `{"skill": "<name>"}`."""

_FROZEN = ConfigDict(extra='forbid', frozen=True)
_THRESHOLDS_FILE = 'tag_thresholds.json'
_TAG_VERSION_SUFFIX = '+custom'


def default_tool_roles() -> dict[str, ToolRole]:
    """The built-in tool-name classification: shell, web fetch, skill loader and subagent spawner names."""
    roles: dict[str, ToolRole] = {}
    for name in sorted(SHELL_TOOL_NAMES):
        roles[name] = 'bash'
    for name in sorted(WEBFETCH_TOOL_NAMES):
        roles[name] = 'webview'
    for name in sorted(SKILL_TOOL_NAMES):
        roles[name] = 'skill'
    for name in sorted(SUBAGENT_TOOL_NAMES):
        roles[name] = 'subagent'
    return roles


def default_alert_thresholds() -> dict[str, float]:
    """Alert thresholds per signal. Not from the tickets (they define signals, not alerts): starting points to tune."""
    return {
        'tool_error_rate': 0.2,
        'duplicate_tool_call_count': 2,
        'consecutive_same_tool_max': 3,
        'tool_loop_count': 0,
        'invalid_schema_tool_call_count': 0,
        'finish_reason_length_count': 0,
        'empty_tool_result_count': 2,
    }


def _default_classifier_model() -> str:
    return os.environ.get('EVALUATORQ_CLASSIFIER_MODEL', 'typesafe/jev-latest')


class ClassifierConfig(BaseModel):
    """The opt-in model that classifies tool roles (the only signal path that calls a model)."""

    model_config = _FROZEN
    model: str = Field(default_factory=_default_classifier_model)
    """Read from `EVALUATORQ_CLASSIFIER_MODEL` when the config is constructed, not at import."""
    enabled: bool = False


class TagThresholds(BaseModel):
    """Calibrated tag thresholds: `<tag>.<metric>` to threshold, plus each metric's cohort percentile table."""

    model_config = _FROZEN
    rule_version: str
    thresholds: dict[str, float]
    percentiles: dict[str, list[float]] = Field(default_factory=dict)
    """metric -> its p0..p100 values on the calibration cohort, so a percentile maps to a threshold."""

    @field_validator('percentiles')
    @classmethod
    def _validate_percentiles(cls, tables: dict[str, list[float]]) -> dict[str, list[float]]:
        for metric, values in tables.items():
            if len(values) != 101:
                raise ValueError(f'percentile table for {metric!r} must contain exactly 101 values (p0..p100)')
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f'percentile table for {metric!r} must contain only finite values')
            if any(left > right for left, right in pairwise(values)):
                raise ValueError(f'percentile table for {metric!r} must be nondecreasing')
        return tables

    @classmethod
    def from_published(cls, document: dict[str, Any]) -> TagThresholds:
        """Read the published `tag_thresholds.json` layout, where each threshold is `{"value": ..., "quantile": ...}`."""
        return cls(
            rule_version=document['rule_version'],
            thresholds={key: entry['value'] for key, entry in document['thresholds'].items()},
            percentiles=document.get('percentiles', {}),
        )

    @classmethod
    def bundled(cls) -> TagThresholds:
        """The thresholds shipped with the package, loaded once and returned as a defensive copy."""
        return _bundled_thresholds().model_copy(deep=True)

    def cohort_threshold(self, metric: str, quantile: float) -> float:
        """The metric's value at `quantile` (0-100) in the calibration cohort, linearly interpolated.

        Raises:
            KeyError: `metric` has no percentile table.
        """
        table = self.percentiles[metric]
        position = min(max(quantile, 0.0), 100.0)
        low = int(position)
        if low >= len(table) - 1:
            return table[-1]
        return round(table[low] + (table[low + 1] - table[low]) * (position - low), 4)


@functools.cache
def _bundled_thresholds() -> TagThresholds:
    text = resources.files('evaluatorq.signals').joinpath(_THRESHOLDS_FILE).read_text(encoding='utf-8')
    return TagThresholds.from_published(json.loads(text))


class SignalsConfig(BaseModel):
    """Rule knobs for every signal. Derive a variant with `config.model_copy(update={...})`.

    Defaults follow the BOPS-1207 / BOPS-1208 definitions: status-only error detection, sorted-keys canonical
    JSON, "consecutive call to the same tool" as a retry, and `""`, `[]`, `{}`, `null` as empty results.

    Attributes:
        error_detection: `'status'` counts only results the source marked as errors; `'status_and_content'` also
            flags results whose text contains an `error_patterns` entry.
        error_patterns: Substrings (case-sensitive) that mark a result as an error when sniffing.
        canonicalisation: How arguments are compared for duplicates, distinctness and retries. `'sorted_keys'`
            ignores key order, `'exact'` compares as received.
        retry_definition: `'consecutive_same_tool'` means the agent's previous call used the same tool and
            failed; `'same_tool_args_within_n'` means one of the agent's previous `retry_window` calls had the
            same tool and canonical arguments and failed.
        retry_window: N for `'same_tool_args_within_n'`.
        empty_values: Which result values count as empty: any of `""`, `"[]"`, `"{}"`, `"null"`.
        empty_literals: Extra literal result strings to treat as empty, e.g. `"(Bash completed with no output)"`
            for Claude Code.
        tool_roles: Tool name to role. A name that is missing has role `'other'`. When read with `from_file`, the
            file's entries merge over the defaults instead of replacing them.
        alert_thresholds: signal name to x; a signal alerts when its value is greater than x.
        tag_percentiles: `<tag>.<metric>` to cohort percentile (0-100) for a calibrated tag clause. Missing keys
            use the rule's published percentile; any entry marks the config version `+custom`.
        tag_thresholds: The calibrated thresholds; None means the bundled ones.
        classifier: The opt-in tool-role classifier.
    """

    model_config = _FROZEN
    error_detection: ErrorDetection = 'status'
    error_patterns: tuple[str, ...] = DEFAULT_ERROR_PATTERNS
    canonicalisation: Canonicalisation = 'sorted_keys'
    retry_definition: RetryDefinition = 'consecutive_same_tool'
    retry_window: int = Field(default=3, ge=1)
    empty_values: frozenset[str] = DEFAULT_EMPTY_VALUES
    empty_literals: tuple[str, ...] = ()
    tool_roles: dict[str, ToolRole] = Field(default_factory=default_tool_roles)
    alert_thresholds: dict[str, float] = Field(default_factory=default_alert_thresholds)
    tag_percentiles: dict[str, float] = Field(default_factory=dict)
    tag_thresholds: TagThresholds | None = None
    classifier: ClassifierConfig = Field(default_factory=ClassifierConfig)

    @classmethod
    def from_file(cls, path: str | Path) -> SignalsConfig:
        """Read a JSON config. The file may set any subset of fields; the rest keep their defaults.

        `tool_roles` merges over the default classification (a file entry wins for its tool name; the defaults for
        every other name stay), so a file naming one custom shell tool does not turn `Bash` back into `other`.

        Raises:
            pydantic.ValidationError: The file is not valid JSON or does not match the fields (unknown keys fail).
        """
        config = cls.model_validate_json(Path(path).read_text(encoding='utf-8'))
        return config.model_copy(update={'tool_roles': {**default_tool_roles(), **config.tool_roles}})

    def tool_role(self, name: str) -> ToolRole:
        """The role of a tool name; names with no entry are `'other'`."""
        return self.tool_roles.get(name, 'other')

    def resolved_tag_thresholds(self) -> TagThresholds:
        """The thresholds in effect: this config's own, else the bundled ones."""
        return self.tag_thresholds or TagThresholds.bundled()

    @property
    def version(self) -> str:
        """The bundled `rule_version`, suffixed `+custom` when percentiles or thresholds differ from the bundled ones."""
        bundled = TagThresholds.bundled()
        custom = bool(self.tag_percentiles) or (self.tag_thresholds is not None and self.tag_thresholds != bundled)
        return bundled.rule_version + (_TAG_VERSION_SUFFIX if custom else '')

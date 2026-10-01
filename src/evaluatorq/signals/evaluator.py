"""Adapters that expose deterministic signals through evaluatorq's evaluator contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from loguru import logger

from evaluatorq.contracts import Message
from evaluatorq.formats import AtifTrajectory, ChatConversation, OtelTrace, ResponsesConversation
from evaluatorq.signals.registry import SIGNAL_NAMES, SIGNALS, compute_signals
from evaluatorq.types import EvaluationResult

if TYPE_CHECKING:
    from collections.abc import Sequence

    from evaluatorq.signals.config import SignalsConfig
    from evaluatorq.types import Evaluator, ScorerParameter


def to_trajectory(output: Any) -> AtifTrajectory | None:
    """Convert a supported transcript or trace output to ATIF.

    Supported inputs are ATIF, chat, Responses and OTel models, chat message lists, and mappings with a
    ``messages`` key. Unsupported values or failed conversions return ``None``.
    """
    try:
        if isinstance(output, AtifTrajectory):
            return output
        if isinstance(output, ChatConversation | ResponsesConversation | OtelTrace):
            return output.to_atif()
        if isinstance(output, list) and all(isinstance(item, (dict, Message)) for item in output):
            return ChatConversation(messages=output).to_atif()
        if isinstance(output, Mapping) and 'messages' in output:
            messages = output['messages']
            if isinstance(messages, list) and all(isinstance(item, (dict, Message)) for item in messages):
                return ChatConversation(messages=messages).to_atif()
    except Exception as exc:  # noqa: BLE001 - conversion failures become inconclusive evaluator results
        logger.warning('Could not convert evaluator output to ATIF: {}: {}', type(exc).__name__, exc)
    return None


def _evidence_references(result: Any) -> str:
    references: list[str] = []
    for item in result.evidence:
        if item.call_id is None:
            references.append(f'step {item.step_id}')
        else:
            references.append(f'step {item.step_id} / call {item.call_id}')
    shown = references[:10]
    if len(references) > 10:
        shown.append(f'(+{len(references) - 10} more)')
    return '; '.join(shown)


def _evaluation_result(value: Any, explanation: str, *, passed: bool | None) -> EvaluationResult:
    return EvaluationResult.model_validate({'value': value, 'explanation': explanation, 'pass': passed})


def signal_evaluator(name: str, *, threshold: float | None = None, config: SignalsConfig | None = None) -> Evaluator:
    """Build a scorer for one registered signal.

    Group D signals are tags: a fired tag fails the evaluator. Other signals are counts or measurements and only
    produce a pass/fail when a numeric ``threshold`` is supplied; values at or below it pass.
    """
    if name not in SIGNALS:
        msg = f'Unknown signal name: {name}'
        raise ValueError(msg)

    async def scorer(params: ScorerParameter) -> EvaluationResult:  # noqa: RUF029 - scorer must satisfy evaluatorq's async contract
        output = params['output']
        trajectory = to_trajectory(output)
        if trajectory is None:
            output_type = type(output).__name__
            logger.warning('Signal evaluator {} received unsupported output of type {}', name, output_type)
            return _evaluation_result(
                None,
                f'Could not convert output of type {output_type} to an ATIF trajectory.',
                passed=None,
            )

        try:
            report = compute_signals(trajectory, config=config, only=[name])
        except Exception as exc:  # noqa: BLE001 - scorer failures must be inconclusive, not abort an evaluation run
            logger.warning('Signal evaluator {} failed: {}: {}', name, type(exc).__name__, exc)
            return _evaluation_result(None, f'Could not compute {name}: {type(exc).__name__}: {exc}', passed=None)
        signal = report.results[name]
        if signal.no_basis is not None:
            return _evaluation_result(None, f'No basis for {name}: {signal.no_basis}', passed=None)
        if signal.value is None:
            return _evaluation_result(None, f'Signal {name} did not produce a value.', passed=None)

        group = SIGNALS[name][0]
        if group == 'D':
            explanation = signal.reason or f'Tag {name} did not fire.'
            if evidence := _evidence_references(signal):
                explanation = f'{explanation} ({evidence})'
            return _evaluation_result(signal.value, explanation, passed=not signal.value)

        passed = None
        explanation = f'Signal {name} returned {signal.value}'
        if threshold is not None:
            if isinstance(signal.value, (int, float)) and not isinstance(signal.value, bool):
                passed = signal.value <= threshold
                explanation += f' (threshold {threshold}; values at or below pass)'
            else:
                explanation += f' (threshold {threshold} cannot be applied to a non-numeric value)'
        if evidence := _evidence_references(signal):
            explanation += f' ({evidence})'
        return _evaluation_result(signal.value, explanation, passed=passed)

    evaluator: Evaluator = {'name': name, 'scorer': scorer}
    return evaluator


def signal_evaluators(names: str | Sequence[str] = 'tags', *, config: SignalsConfig | None = None) -> list[Evaluator]:
    """Build evaluators for every tag, every signal, one group, or an explicit list of names."""
    if names == 'tags':
        selected = [name for name in SIGNAL_NAMES if SIGNALS[name][0] == 'D']
    elif names == 'all':
        selected = list(SIGNAL_NAMES)
    elif isinstance(names, str) and names in {'A', 'B', 'C', 'D'}:
        selected = [name for name in SIGNAL_NAMES if SIGNALS[name][0] == names]
    elif isinstance(names, str):
        msg = "names must be 'tags', 'all', a group letter, or a sequence of signal names"
        raise ValueError(msg)
    else:
        selected = list(names)
    unknown = sorted(set(selected) - set(SIGNALS))
    if unknown:
        msg = f'Unknown signal names: {", ".join(unknown)}'
        raise ValueError(msg)
    return [signal_evaluator(name, config=config) for name in selected]

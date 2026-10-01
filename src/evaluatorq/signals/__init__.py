"""Deterministic ADR-25 trace signals, computed on an ATIF trajectory.

```python
from evaluatorq.signals import compute_signals

report = compute_signals(trajectory)
report.values()  # name -> value, no-basis signals omitted
```
"""

from evaluatorq.signals.classify import classify_tool_roles
from evaluatorq.signals.config import ClassifierConfig, SignalsConfig, TagThresholds
from evaluatorq.signals.evaluator import signal_evaluator, signal_evaluators, to_trajectory
from evaluatorq.signals.models import Evidence, Precondition, SignalReport, SignalResult
from evaluatorq.signals.registry import SIGNAL_NAMES, compute_signals
from evaluatorq.signals.walk import SignalContext

__all__ = [
    'SIGNAL_NAMES',
    'ClassifierConfig',
    'Evidence',
    'Precondition',
    'SignalContext',
    'SignalReport',
    'SignalResult',
    'SignalsConfig',
    'TagThresholds',
    'classify_tool_roles',
    'compute_signals',
    'signal_evaluator',
    'signal_evaluators',
    'to_trajectory',
]

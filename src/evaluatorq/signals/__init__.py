"""Deterministic ADR-25 trace signals, computed on `evaluatorq.formats.AtifTrajectory`.

```python
from evaluatorq.signals import compute_signals

report = compute_signals(trajectory)
report.values()  # name -> value, no-basis signals omitted
```
"""

from evaluatorq.signals.config import ClassifierConfig, SignalsConfig, TagThresholds
from evaluatorq.signals.models import Evidence, Precondition, SignalReport, SignalResult
from evaluatorq.signals.registry import SIGNAL_NAMES, compute_signals

__all__ = [
    'SIGNAL_NAMES',
    'ClassifierConfig',
    'Evidence',
    'Precondition',
    'SignalReport',
    'SignalResult',
    'SignalsConfig',
    'TagThresholds',
    'compute_signals',
]

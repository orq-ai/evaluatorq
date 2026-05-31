"""Back-compat shim. Canonical palette now lives in common.reports.palette."""

from __future__ import annotations

from evaluatorq.common.reports.palette import (  # noqa: F401
	COLORS,
	ORQ_SCALE_AGENT,
	ORQ_SCALE_GOOD_BAD,
	ORQ_SCALE_HEAT,
	QUALITATIVE,
	SEVERITY_COLORS,
	SEVERITY_ORDER,
	STATUS_COLORS,
)

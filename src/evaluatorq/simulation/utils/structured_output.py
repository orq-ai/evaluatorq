"""Back-compat re-export: generate_structured moved to ``common`` (RES-822).

The structured-output-with-json_object-fallback helper is shared by simulation
and red-team report code, so it now lives in ``evaluatorq.common``. This shim
keeps existing ``simulation.utils`` imports working.
"""

from __future__ import annotations

from evaluatorq.common.structured_output import generate_structured

__all__ = ['generate_structured']

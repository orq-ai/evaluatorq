"""Back-compat re-export: extract_json moved to ``evaluatorq.common`` (RES-822).

The fence-tolerant JSON extractor is shared by simulation and redteam report
code, so it now lives in ``common``. This shim keeps existing
``simulation.utils`` imports working.
"""

from __future__ import annotations

from evaluatorq.common.extract_json import extract_json_from_response

__all__ = ['extract_json_from_response']

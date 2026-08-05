from __future__ import annotations

import json
from typing import TYPE_CHECKING

from loguru import logger

from evaluatorq.contracts import AgentResponse

if TYPE_CHECKING:
    from evaluatorq.types import Output


def output_to_text(output: Output) -> str:
    """Best-effort plain-text view of any Output. Total / fail-soft."""
    if output is None:
        return ''
    if isinstance(output, AgentResponse):
        return output.text
    if isinstance(output, str):
        return output
    if isinstance(output, dict):
        if output.get('object') == 'response':
            try:
                return AgentResponse.from_openresponses(output).text
            except Exception as exc:
                logger.debug('from_openresponses failed, falling back to json: {}', exc)
        try:
            return json.dumps(output, indent=2, default=str)
        except Exception:
            return str(output)
    try:
        return str(output)
    except Exception:
        return ''

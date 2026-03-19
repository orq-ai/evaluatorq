"""Shared utilities for the evaluatorq.redteam package."""

from __future__ import annotations

import re
import xml.sax.saxutils


def safe_substitute(template: str, replacements: dict[str, str]) -> str:
    """Single-pass template substitution that prevents cascading expansion.

    Unlike chained ``.replace()`` calls, this replaces all placeholders in one
    pass so that substituted values cannot inject later placeholders.

    Args:
        template: Template string with ``{placeholder}`` variables.
        replacements: Mapping of placeholder (including braces) to value.

    Returns:
        Template with all known placeholders replaced.
    """
    if not replacements:
        return template
    pattern = re.compile('|'.join(re.escape(k) for k in replacements))
    return pattern.sub(lambda m: replacements[m.group(0)], template)


def xml_escape(text: str) -> str:
    """Escape text for safe embedding inside XML tags."""
    return xml.sax.saxutils.escape(text)

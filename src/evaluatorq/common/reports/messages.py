"""Chat bubble presentation shared by dashboard and standalone reports."""

from __future__ import annotations

from typing import Any

from evaluatorq.common.messages import messages_to_text
from evaluatorq.common.reports import esc
from evaluatorq.contracts import content_to_text


def render_message_list(
    messages: list[Any],
    *,
    role_labels: dict[str, str],
    class_prefix: str,
) -> str:
    """Render a role-labeled message list as a series of chat-bubble ``<div>``s.

    Each message produces:

    ```html
    <div class="{class_prefix}-msg {class_prefix}-msg-{css_role}">
      <span class="{class_prefix}-msg-avatar">{label}</span>
      <div class="{class_prefix}-msg-bubble">
        <span class="{class_prefix}-msg-role">{label}</span>
        <pre class="{class_prefix}-msg-content">{esc(content)}</pre>
      </div>
    </div>
    ```

    Where ``{css_role}`` is the raw role value when it is one of
    ``user``, ``assistant``, ``system``, ``tool``; otherwise ``unknown``.
    The ``{class_prefix}-msg-avatar`` span is a short glyph (e.g. "USR" /
    "AGT" / "SYS" / "TOOL") taken straight from *role_labels* — callers that
    want a compact avatar should pass short labels; the ``{css_role}`` class
    is what CSS uses to put user/system on the left and assistant/tool on
    the right (spec §Transcripts), so no separate "side" marker is needed.

    Every content string is passed through ``esc()`` (HTML-escaping) to
    prevent stored-XSS vectors. This is purely additive over the prior
    shape (avatar span + bubble wrapper added); the ``{class_prefix}-msg
    {class_prefix}-msg-{css_role}`` outer class and the ``-msg-role`` /
    ``-msg-content`` elements are unchanged, so existing callers keep working.

    Args:
        messages:     Sequence of message dicts (``role`` / ``content`` keys)
                      or objects with ``.role`` / ``.content`` attributes.
        role_labels:  Mapping from role name → display label.  Falls back to
                      the raw role string when a role is not in the map.
        class_prefix: CSS class namespace (e.g. ``"sim"`` → ``sim-msg``).

    Returns:
        Concatenated HTML string (empty string when *messages* is empty).
    """
    known_roles = frozenset({'user', 'assistant', 'system', 'tool'})

    parts: list[str] = []
    for msg in messages:
        # Support both dict-style and attribute-style message objects.
        if isinstance(msg, dict):
            role = str(msg.get('role', 'unknown'))
            raw_content = msg.get('content', '')
        else:
            role = str(getattr(msg, 'role', 'unknown'))
            raw_content = getattr(msg, 'content', '')

        label = role_labels.get(role, role)
        content_text = content_to_text(raw_content) if raw_content is not None else ''
        calls = msg.get('tool_calls') if isinstance(msg, dict) else getattr(msg, 'tool_calls', None)
        if calls:
            record = msg if isinstance(msg, dict) else msg.model_dump()
            content_text = messages_to_text([record])
        safe_content = esc(content_text)
        css_role = role if role in known_roles else 'unknown'
        safe_label = esc(label)

        parts.append(
            f'<div class="{class_prefix}-msg {class_prefix}-msg-{esc(css_role)}">'
            f'<span class="{class_prefix}-msg-avatar">{safe_label}</span>'
            f'<div class="{class_prefix}-msg-bubble">'
            f'<span class="{class_prefix}-msg-role">{safe_label}</span>'
            f'<pre class="{class_prefix}-msg-content">{safe_content}</pre>'
            f'</div>'
            f'</div>'
        )

    return ''.join(parts)

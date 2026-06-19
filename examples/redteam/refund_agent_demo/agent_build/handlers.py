"""Pure-function tool handlers for the refund agent demo."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from loguru import logger

from agent_build.demo_data import NOW, SESSION_USER_ID, DemoState, Refund

WINDOW_DAYS = 30

_KB_DIR = Path(__file__).parent / 'policy_kb'
_KB_TOPIC_FILES = {
    'refund_basics': 'refund_basics.md',
    'post_window_exceptions': 'post_window_exceptions.md',
    'abuse_patterns': 'abuse_patterns.md',
}
_KB_KEY = 'refund_policy'
# Keyed by id(client). Safe here because the demo uses a single long-lived
# client; do not copy this pattern to production where clients may be
# short-lived (a GC'd client's id() can be reused, returning a stale KB id).
_KB_ID_CACHE: dict[int, str] = {}


def _resolve_kb_id(orq_client: Any) -> str | None:
    cache_key = id(orq_client)
    if cache_key in _KB_ID_CACHE:
        return _KB_ID_CACHE[cache_key]
    try:
        resp = orq_client.knowledge.list(limit=50)
        for kb in resp.data or []:
            if getattr(kb, 'key', None) == _KB_KEY:
                _KB_ID_CACHE[cache_key] = kb.id
                return kb.id
    except Exception as exc:
        logger.warning(f'KB id lookup failed: {type(exc).__name__}: {exc}')
    return None


def handle_lookup_order(
    state: DemoState,
    *,
    order_id: str,
    # session_user_id is a test-seam, NOT part of the agent-facing tool
    # schema. In production this MUST be sourced from authenticated session
    # state, never from tool-call arguments.
    session_user_id: str = SESSION_USER_ID,
) -> dict:
    order = state.find_order(order_id)
    if order is None or order.owner_id != session_user_id:
        return {'ok': False, 'error': 'not_found', 'status_code': 404}
    return {'ok': True, 'order': _serialize_order(order)}


def handle_issue_refund(
    state: DemoState,
    *,
    order_id: str,
    reason: str,
    post_window_exception: bool = False,
    # session_user_id is a test-seam (see handle_lookup_order).
    session_user_id: str = SESSION_USER_ID,
) -> dict:
    order = state.find_order(order_id)
    if order is None or order.owner_id != session_user_id:
        return {'ok': False, 'error': 'not_found', 'status_code': 404}
    if state.is_refunded(order_id):
        return {'ok': False, 'error': 'already_refunded', 'status_code': 409}
    age = (NOW - order.created_at).days
    if age > WINDOW_DAYS and not post_window_exception:
        return {'ok': False, 'error': 'outside_window', 'status_code': 400}
    state.refunds.append(
        Refund(
            order_id=order_id,
            reason=reason,
            post_window_exception=post_window_exception,
            issued_at=NOW,
        )
    )
    return {'ok': True, 'amount_refunded': order.amount}


def handle_get_policy(*, orq_client: Any | None, topic: str) -> dict:
    if orq_client is not None:
        kb_id = _resolve_kb_id(orq_client)
        if kb_id is not None:
            try:
                kb_result = orq_client.knowledge.search(
                    knowledge_id=kb_id,
                    query=topic,
                    top_k=1,
                )
                chunks = getattr(kb_result, 'matches', None) or []
                if chunks:
                    text = getattr(chunks[0], 'text', None) or str(chunks[0])
                    return {'ok': True, 'text': text, 'source': 'kb'}
            except Exception as exc:
                logger.warning(f'KB search failed ({type(exc).__name__}: {exc}); falling back to local policy files')
    fname = _KB_TOPIC_FILES.get(topic)
    if fname is None:
        return {'ok': False, 'error': f'unknown_topic:{topic}'}
    path = _KB_DIR / fname
    if not path.exists():
        return {'ok': False, 'error': f'missing_doc:{fname}'}
    return {'ok': True, 'text': path.read_text(), 'source': 'local_fallback'}


def _serialize_order(order: Any) -> dict:
    d = asdict(order)
    d['created_at'] = order.created_at.isoformat()
    # The agent (an LLM) has no reliable "today" — left to compute the window
    # from a raw date it uses its own wall-clock and wrongly expires in-window
    # orders. Hand it the window facts directly, derived from the same frozen
    # NOW that handle_issue_refund enforces against, so both clocks agree.
    days_ago = (NOW - order.created_at).days
    d['delivered_days_ago'] = days_ago
    d['within_standard_window'] = days_ago <= WINDOW_DAYS
    return d

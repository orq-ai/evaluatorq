"""``LLMCallConfig.extra_body`` is the user seam into the request body.

``extra_kwargs`` sets top-level SDK call arguments; ``extra_body`` sets fields in
the HTTP body the SDK has no named parameter for. The call site owns that body
(the Orq router's retry policy, thread and memory ids), so the config's version
is **merged into** it rather than replacing it — replacing it wholesale is what
silently dropped retry hints, and is why ``extra_body`` is rejected inside
``extra_kwargs``.
"""

from __future__ import annotations

import pytest

from evaluatorq.contracts import LLMCallConfig

# ruff: noqa: S101

_ROUTER_BODY = {'retry': {'count': 3}, 'thread': {'id': 'call-site'}}


def test_config_extra_body_merges_into_the_call_site_body() -> None:
    cfg = LLMCallConfig(model='m', extra_body={'cache': {'ttl': 60}})
    params = cfg.request_params(api='chat_completions', model='m', messages=[], extra_body=dict(_ROUTER_BODY))

    # The caller's key is added; the call site's keys survive.
    assert params['extra_body'] == {
        'retry': {'count': 3},
        'thread': {'id': 'call-site'},
        'cache': {'ttl': 60},
    }


def test_config_extra_body_wins_per_key() -> None:
    cfg = LLMCallConfig(model='m', extra_body={'thread': {'id': 'caller'}})
    params = cfg.request_params(api='chat_completions', model='m', messages=[], extra_body=dict(_ROUTER_BODY))

    assert params['extra_body']['thread'] == {'id': 'caller'}
    assert params['extra_body']['retry'] == {'count': 3}, 'retry policy must survive'


def test_unset_extra_body_leaves_the_call_site_body_untouched() -> None:
    cfg = LLMCallConfig(model='m')
    params = cfg.request_params(api='chat_completions', model='m', messages=[], extra_body=dict(_ROUTER_BODY))

    assert params['extra_body'] == _ROUTER_BODY


def test_unset_extra_body_adds_no_key_when_the_call_site_sent_none() -> None:
    cfg = LLMCallConfig(model='m')
    assert 'extra_body' not in cfg.request_params(api='chat_completions', model='m', messages=[])


def test_request_params_responses_merges_identically() -> None:
    """Both endpoint shapes must agree, or a field is honoured on one endpoint only."""
    cfg = LLMCallConfig(model='m', extra_body={'cache': {'ttl': 60}})
    params = cfg.request_params(api='responses', model='m', input=[], extra_body=dict(_ROUTER_BODY))

    assert params['extra_body'] == {
        'retry': {'count': 3},
        'thread': {'id': 'call-site'},
        'cache': {'ttl': 60},
    }


@pytest.mark.parametrize('api', ['chat_completions', 'responses'])
def test_extra_body_is_still_rejected_inside_extra_kwargs(api: str) -> None:
    """The seam is the dedicated field. Smuggling it through extra_kwargs replaces
    the router body instead of merging, so it stays a hard error on both endpoints.
    """
    cfg = LLMCallConfig(model='m', extra_kwargs={'extra_body': {'retry': None}})
    with pytest.raises(ValueError, match='extra_body'):
        cfg.request_params(api=api, model='m')  # pyright: ignore[reportArgumentType]


def test_single_endpoint_override_warns_only_when_api_was_set(caplog) -> None:
    """A call site that only speaks one endpoint must not ignore `api` silently.

    The warning is gated on `model_fields_set` so a role whose *default* api
    differs (EvaluatorConfig defaults to 'responses') does not warn on every
    chat-only helper call — only a user who actually asked for the other
    endpoint hears about it.
    """
    from loguru import logger

    records: list[str] = []
    sink_id = logger.add(lambda m: records.append(m), level='WARNING')
    try:
        LLMCallConfig(model='m').request_params(api='responses', model='m', input=[])
        assert records == []

        LLMCallConfig(model='m', api='responses').request_params(api='chat_completions', model='m', messages=[])
        assert len(records) == 1
        assert "api='responses'" in records[0]
    finally:
        logger.remove(sink_id)

"""``generate_structured``'s ``config=`` merge — the function every generator funnels through.

Its precedence rule is explicit keyword > ``config`` > call-site default, and the
sentinel exists so ``None`` and the default timeout stay values a caller can mean.
Nothing else covers this: the surface tests all mock ``generate_structured`` away.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from evaluatorq.contracts import LLMCallConfig


class _Answer(BaseModel):
    answer: str


@pytest.mark.asyncio
async def test_generate_structured_reads_only_set_config_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    """The one merge every generator funnels through — and the only test of it."""
    import evaluatorq.common.structured_output as mod

    seen: dict[str, Any] = {}

    async def fake_parse(**kwargs: Any) -> Any:
        seen.update(kwargs)
        raise RuntimeError('stop after the params are built')

    monkeypatch.setattr(mod, 'execute_chat_parse', fake_parse)
    cfg = LLMCallConfig(model='config/model', temperature=0.7, extra_body={'from': 'config'})
    with pytest.raises(Exception, match='stop after the params are built'):
        await mod.generate_structured(
            MagicMock(),
            model='explicit/model',
            messages=[{'role': 'user', 'content': 'hi'}],
            response_format=_Answer,
            max_tokens=64,
            label='test',
            config=cfg,
        )
    assert seen['temperature'] == 0.7
    assert seen['extra_body'] == {'from': 'config'}
    # config.model is deliberately not read — the call site stays the authority.
    assert seen['model'] == 'explicit/model'


@pytest.mark.asyncio
async def test_generate_structured_explicit_keyword_beats_the_config(monkeypatch: pytest.MonkeyPatch) -> None:
    import evaluatorq.common.structured_output as mod

    seen: dict[str, Any] = {}

    async def fake_parse(**kwargs: Any) -> Any:
        seen.update(kwargs)
        raise RuntimeError('stop')

    monkeypatch.setattr(mod, 'execute_chat_parse', fake_parse)
    with pytest.raises(Exception, match='stop'):
        await mod.generate_structured(
            MagicMock(),
            model='m',
            messages=[{'role': 'user', 'content': 'hi'}],
            response_format=_Answer,
            max_tokens=64,
            label='test',
            temperature=0.0,
            config=LLMCallConfig(model='m', temperature=0.7),
        )
    assert seen['temperature'] == 0.0


@pytest.mark.asyncio
async def test_generate_structured_explicit_none_temperature_beats_the_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`None` means "omit the parameter" — a real value, not "caller said nothing"."""
    import evaluatorq.common.structured_output as mod

    seen: dict[str, Any] = {}

    async def fake_parse(**kwargs: Any) -> Any:
        seen.update(kwargs)
        raise RuntimeError('stop')

    monkeypatch.setattr(mod, 'execute_chat_parse', fake_parse)
    with pytest.raises(Exception, match='stop'):
        await mod.generate_structured(
            MagicMock(),
            model='m',
            messages=[{'role': 'user', 'content': 'hi'}],
            response_format=_Answer,
            max_tokens=64,
            label='test',
            temperature=None,
            config=LLMCallConfig(model='m', temperature=0.7),
        )
    assert seen['temperature'] is None


@pytest.mark.asyncio
async def test_generate_structured_explicit_default_timeout_beats_the_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Passing the default value on purpose is still passing it."""
    import evaluatorq.common.structured_output as mod

    seen: dict[str, Any] = {}

    async def fake_parse(**kwargs: Any) -> Any:
        seen.update(kwargs)
        raise RuntimeError('stop')

    monkeypatch.setattr(mod, 'execute_chat_parse', fake_parse)
    with pytest.raises(Exception, match='stop'):
        await mod.generate_structured(
            MagicMock(),
            model='m',
            messages=[{'role': 'user', 'content': 'hi'}],
            response_format=_Answer,
            max_tokens=64,
            label='test',
            timeout_s=mod._STRUCTURED_TIMEOUT_S,
            config=LLMCallConfig(model='m', timeout_ms=1000),
        )
    assert seen['timeout_s'] == mod._STRUCTURED_TIMEOUT_S

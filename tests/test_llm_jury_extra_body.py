"""`extra_body=` on the jury helpers reaches the outgoing request body.

The gap these cover: `extra_kwargs` rejects `extra_body` (it is a structural
field the call site owns), so before this parameter existed there was no seam at
all for body fields on the jury path — the only workaround was a hand-rolled
client proxy.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from evaluatorq import DataPoint
from evaluatorq.common.judge import run_judge
from evaluatorq.contracts import LLMCallConfig
from evaluatorq.llm_jury import llm_jury, llm_jury_pairwise


class _Recorder:
    """Records outgoing kwargs and stops before any network call."""

    def __init__(self, sink: list[dict[str, Any]]) -> None:
        self.sink = sink

    async def create(self, **kwargs: Any) -> Any:
        self.sink.append(kwargs)
        raise RuntimeError('halt-before-network')

    async def parse(self, **kwargs: Any) -> Any:
        self.sink.append(kwargs)
        raise RuntimeError('halt-before-network')


class _RecordingClient:
    """Duck-typed client exposing both judge endpoints.

    Both are needed: under the offline catalogue fixture the Responses model
    cannot be qualified, so `run_judge` falls back to chat completions. The
    assertions below hold on whichever endpoint actually ran, which is the point
    — `extra_body` is reserved on both.
    """

    def __init__(self, sink: list[dict[str, Any]]) -> None:
        self.base_url = 'https://my.orq.ai/v3/router'
        self.responses = _Recorder(sink)
        self.chat = SimpleNamespace(completions=_Recorder(sink))


async def _capture(cfg: LLMCallConfig) -> dict[str, Any]:
    sink: list[dict[str, Any]] = []
    await run_judge(
        # `run_judge` is annotated `AsyncOpenAI`, but the runtime only needs the
        # endpoints below — that duck-typing is what makes a wrapper client viable.
        client=cast(Any, _RecordingClient(sink)),
        model='gpt-5-mini',
        cfg=cfg,
        prompt_template='judge {x}',
        replacements={'x': 'a'},
    )
    assert sink, 'the judge never issued a request'
    return sink[0]


@pytest.mark.asyncio
async def test_extra_body_reaches_the_request_body() -> None:
    params = await _capture(
        LLMCallConfig(model='gpt-5-mini', api='responses', retry_count=0, extra_body={'my_key': 'abc'})
    )
    assert params['extra_body'] == {'my_key': 'abc'}


@pytest.mark.asyncio
async def test_no_extra_body_sends_no_extra_body_key() -> None:
    """The common path must not grow an empty dict — it changes the wire shape."""
    params = await _capture(LLMCallConfig(model='gpt-5-mini', api='responses', retry_count=0))
    assert 'extra_body' not in params


def test_extra_body_inside_extra_kwargs_is_rejected_by_the_param_builder() -> None:
    """The guard stays: `extra_kwargs` replaces a key, so it must not own this one."""
    cfg = LLMCallConfig(model='gpt-5-mini', api='responses', extra_kwargs={'extra_body': {'k': 1}})
    with pytest.raises(ValueError, match=r'extra_body'):
        cfg.request_params(api='responses', input=[], text={})


@pytest.mark.asyncio
async def test_extra_body_inside_extra_kwargs_never_reaches_the_wire() -> None:
    """`run_judge` turns the guard's ValueError into a failed verdict rather than
    propagating it, so the symptom a user sees is a judge that always fails — not
    a crash. Either way no request is issued."""
    sink: list[dict[str, Any]] = []
    await run_judge(
        # `run_judge` is annotated `AsyncOpenAI`, but the runtime only needs the
        # endpoints below — that duck-typing is what makes a wrapper client viable.
        client=cast(Any, _RecordingClient(sink)),
        model='gpt-5-mini',
        cfg=LLMCallConfig(
            model='gpt-5-mini', api='responses', retry_count=0, extra_kwargs={'extra_body': {'k': 1}}
        ),
        prompt_template='judge {x}',
        replacements={'x': 'a'},
    )
    assert sink == []


@pytest.mark.asyncio
async def test_llm_jury_threads_extra_body_to_the_request() -> None:
    """Drive the real scorer, not `run_judge` directly — the threading through
    `llm_jury` -> `_run_single_judge` -> `LLMCallConfig` is what this covers."""
    sink: list[dict[str, Any]] = []
    evaluator = llm_jury(
        name='correctness',
        criteria='Is it correct?',
        judges=['gpt-5-mini'],
        extra_body={'my_key': 'abc'},
        client=_RecordingClient(sink),
    )
    await evaluator['scorer']({'data': DataPoint(inputs={'question': 'q'}), 'output': 'an answer'})

    assert sink, 'the jury never issued a request'
    assert sink[0]['extra_body'] == {'my_key': 'abc'}


@pytest.mark.asyncio
async def test_llm_jury_without_extra_body_sends_no_extra_body_key() -> None:
    sink: list[dict[str, Any]] = []
    evaluator = llm_jury(
        name='correctness', criteria='Is it correct?', judges=['gpt-5-mini'], client=_RecordingClient(sink)
    )
    await evaluator['scorer']({'data': DataPoint(inputs={'question': 'q'}), 'output': 'an answer'})

    assert sink, 'the jury never issued a request'
    assert 'extra_body' not in sink[0]


@pytest.mark.asyncio
async def test_llm_jury_pairwise_threads_extra_body_to_the_request() -> None:
    sink: list[dict[str, Any]] = []
    comparator = llm_jury_pairwise(
        judges=['gpt-5-mini'],
        criteria='Which is better?',
        swap=False,
        extra_body={'my_key': 'abc'},
        client=_RecordingClient(sink),
    )
    await comparator.compare(question='q', response_a='answer A', response_b='answer B')

    assert sink, 'the comparator never issued a request'
    assert sink[0]['extra_body'] == {'my_key': 'abc'}

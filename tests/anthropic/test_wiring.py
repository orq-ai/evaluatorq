"""Anthropic SDK clients wherever evaluatorq accepts an LLM client (RES-1874)."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Any, NoReturn, cast
from unittest.mock import AsyncMock, MagicMock, patch

import anthropic
import httpx
import pytest
from anthropic.types import Message, TextBlock, Usage
from loguru import logger

from evaluatorq.anthropic import AnthropicLLMClient
from evaluatorq.common import judge as judge_module
from evaluatorq.common.llm_call import execute_chat_completion
from evaluatorq.common.llm_client import as_llm_client, is_anthropic_client, resolve_llm_client
from evaluatorq.common.judge import ClassifyQuestion, ClassifyRequest, JudgeError, run_classify
from evaluatorq.common.model_catalogue import ModelInfo, register_model
from evaluatorq.contracts import AgentResponse, LLMCallConfig, TokenUsage

from .test_chat_adapter import FakeAnthropic, make_message


class ScriptedMessages:
    """``messages.create`` that answers from a queue (the last item repeats) and records every request."""

    def __init__(self, replies: list[Any]) -> None:
        self.replies = list(replies)
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, BaseException):
            raise reply
        return reply


class RawAnthropic:
    """Looks like an ``anthropic`` SDK object to the module-name check; replies from a queue."""

    base_url = 'https://api.anthropic.com'
    max_retries = 2

    def __init__(self, replies: list[Any] | None = None) -> None:
        self.messages = ScriptedMessages(replies or [make_message()])

    def with_options(self, **kwargs: Any) -> RawAnthropic:
        return self

    async def close(self) -> None:
        return None


RawAnthropic.__module__ = 'anthropic._fake'


def _raw(reply: Message | None = None) -> Any:
    return RawAnthropic([reply or make_message()])


def scripted(replies: list[Any]) -> Any:
    return RawAnthropic(replies)


def text(body: str, **kwargs: Any) -> Message:
    return make_message(content=[TextBlock(type='text', text=body)], **kwargs)


# ---------------------------------------------------------------------------
# as_llm_client and the entry points
# ---------------------------------------------------------------------------


def test_as_llm_client_wraps_a_raw_anthropic_client_and_leaves_everything_else_alone() -> None:
    raw = _raw()
    wrapped = as_llm_client(raw)
    assert isinstance(wrapped, AnthropicLLMClient)
    assert as_llm_client(wrapped) is wrapped
    assert as_llm_client(None) is None
    openai_like = MagicMock()
    assert as_llm_client(openai_like) is openai_like
    not_anthropic = FakeAnthropic()
    assert as_llm_client(not_anthropic) is not_anthropic


def test_as_llm_client_wraps_the_real_sdk_classes() -> None:
    clients = (
        anthropic.AsyncAnthropic(api_key='k'),
        anthropic.AsyncAnthropicBedrock(aws_region='eu-west-1', aws_access_key='a', aws_secret_key='b'),
        anthropic.AsyncAnthropicVertex(region='europe-west1', project_id='p', access_token='t'),
    )
    for client in clients:
        assert is_anthropic_client(as_llm_client(client)), type(client).__name__


def test_resolve_llm_client_wraps_an_injected_anthropic_client_and_does_not_own_it() -> None:
    resolved = resolve_llm_client(_raw())
    assert is_anthropic_client(resolved.client)
    assert resolved.owned is False
    assert resolved.routes_through_orq is False


def test_llm_call_config_client_is_wrapped() -> None:
    assert is_anthropic_client(LLMCallConfig(model='claude-sonnet-5-5', client=_raw()).client)


def test_llm_jury_wraps_the_client() -> None:
    from evaluatorq.llm_jury import llm_jury, llm_jury_pairwise

    scorer = llm_jury(name='j', judges=['claude-sonnet-5-5'], criteria='Is it polite?', client=_raw())
    assert is_anthropic_client(cast('Any', scorer['scorer']).keywords['client_state'][0])
    pairwise = llm_jury_pairwise(judges=['claude-sonnet-5-5'], criteria='Which is better?', client=_raw())
    assert is_anthropic_client(pairwise._client)


def test_signals_classify_tool_roles_reaches_the_anthropic_client() -> None:
    from evaluatorq.formats.atif import AtifTrajectory
    from evaluatorq.signals.classify import classify_tool_roles
    from evaluatorq.signals.config import SignalsConfig

    trajectory = AtifTrajectory.model_validate(
        {
            'schema_version': 'ATIF-v1.8',
            'session_id': 's',
            'agent': {'name': 'a', 'version': '1'},
            'steps': [
                {
                    'step_id': 1,
                    'source': 'agent',
                    'message': 'x',
                    'tool_calls': [{'tool_call_id': 'c1', 'function_name': 'mystery', 'arguments': {}}],
                }
            ],
        }
    )
    fake = scripted([text('{"value": "bash", "explanation": "runs a shell"}')])
    config = asyncio.run(
        classify_tool_roles([trajectory], SignalsConfig(classifier={'model': 'claude-sonnet-5-5'}), client=fake)
    )
    assert fake.messages.calls, 'the Anthropic client was never reached'
    assert config.tool_roles['mystery'] == 'bash'


# ---------------------------------------------------------------------------
# Classify without the Orq router
# ---------------------------------------------------------------------------


def _choice_request() -> ClassifyRequest:
    return ClassifyRequest(
        state={'text': 'hello'},
        questions={
            'tone': ClassifyQuestion(
                kind='choice', instructions='Pick the tone.', criteria={'warm': 'friendly', 'cold': 'distant'}, state={}
            ),
            'same': ClassifyQuestion(kind='noul', instructions='Is it a greeting?', state={}),
        },
    )


def _classify(fake: Any, model: str = 'claude-sonnet-5-5') -> Any:
    client = as_llm_client(fake)
    return asyncio.run(
        run_classify(client=client, model=model, cfg=LLMCallConfig(model=model), request=_choice_request())
    )


def test_run_classify_answers_with_a_structured_judgement_when_the_client_is_not_on_the_router() -> None:
    fake = scripted([text('{"answer_0": "warm", "answer_1": 0.9}')])
    outcome = _classify(fake)
    assert outcome.error_kind is None, outcome.error_message
    assert outcome.response is not None
    answers = outcome.response.answers
    assert (answers['tone'].type, answers['tone'].choice) == ('choice', 'warm')
    assert (answers['same'].type, answers['same'].noul) == ('noul', 0.9)
    sent = fake.messages.calls[0]
    assert sent['output_config']['format']['type'] == 'json_schema'
    assert 'warm' in json.dumps(sent['output_config']['format']['schema'])
    assert 'Pick the tone.' in json.dumps(sent['messages'])


def test_run_classify_with_a_classify_only_model_falls_back_to_the_fast_role_and_logs_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('EVALUATORQ_FAST_MODEL', 'claude-haiku-4-5')
    monkeypatch.setattr(judge_module, '_classify_fallback_warned', set())
    seen: list[str] = []
    sink = logger.add(lambda message: seen.append(str(message)), level='WARNING')
    try:
        fake = scripted([text('{"answer_0": "cold", "answer_1": 0.1}')])
        for _ in range(2):
            outcome = _classify(fake, model='typesafe/jev-latest')
            assert outcome.error_kind is None, outcome.error_message
    finally:
        logger.remove(sink)
    assert {call['model'] for call in fake.messages.calls} == {'claude-haiku-4-5'}
    assert len([line for line in seen if 'needs the classify endpoint' in line]) == 1


def test_run_classify_on_the_router_still_calls_the_decisions_endpoint() -> None:
    client = MagicMock()
    client.base_url = 'https://my.orq.ai/v3/router'
    client.max_retries = 0
    client.post = AsyncMock(
        return_value={'answers': {'tone': {'type': 'choice', 'choice': 'warm'}, 'same': {'type': 'noul', 'noul': 0.7}}}
    )
    outcome = asyncio.run(
        run_classify(
            client=client,
            model='typesafe/jev-latest',
            cfg=LLMCallConfig(model='typesafe/jev-latest'),
            request=_choice_request(),
        )
    )
    assert outcome.error_kind is None
    client.post.assert_awaited_once()


def test_a_bad_structured_reply_is_a_parse_outcome_not_a_crash() -> None:
    outcome = _classify(scripted([text('not json at all')]))
    assert outcome.response is None
    assert outcome.error_kind is JudgeError.PARSE


def _server_error() -> anthropic.InternalServerError:
    request = httpx.Request('POST', 'https://api.anthropic.com/v1/messages')
    response = httpx.Response(500, request=request)
    return anthropic.InternalServerError('overloaded', response=response, body=None)


def _no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    async def instant(_: float) -> None:
        return None

    monkeypatch.setattr('evaluatorq.common.retry.asyncio.sleep', instant)


def test_generate_structured_keeps_its_default_retry_budget_and_honours_max_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pydantic import BaseModel

    from evaluatorq.common.retry import MAX_RETRY_ATTEMPTS
    from evaluatorq.common.structured_output import generate_structured

    _no_backoff(monkeypatch)

    class Verdict(BaseModel):
        ok: bool

    def attempts(**kwargs: Any) -> int:
        fake = scripted([_server_error()])
        with pytest.raises(anthropic.InternalServerError):
            asyncio.run(
                generate_structured(
                    as_llm_client(fake),
                    model='claude-sonnet-5-5',
                    messages=[{'role': 'user', 'content': 'x'}],
                    response_format=Verdict,
                    max_tokens=100,
                    label='t',
                    **kwargs,
                )
            )
        return len(fake.messages.calls)

    assert attempts() == MAX_RETRY_ATTEMPTS
    assert attempts(max_attempts=1) == 1


def test_run_classify_fallback_leaves_retry_to_the_caller(monkeypatch: pytest.MonkeyPatch) -> None:
    """One request per outer attempt: the structured fallback must not add a second retry layer under the caller's."""
    from evaluatorq.common.retry import with_retry

    _no_backoff(monkeypatch)
    fake = scripted([_server_error()])
    client = as_llm_client(fake)

    async def attempt() -> Any:
        outcome = await run_classify(
            client=client,
            model='claude-sonnet-5-5',
            cfg=LLMCallConfig(model='claude-sonnet-5-5'),
            request=_choice_request(),
        )
        if outcome.error_exc is not None:
            raise outcome.error_exc
        return outcome

    with pytest.raises(anthropic.InternalServerError):
        asyncio.run(with_retry(attempt, max_attempts=3))
    assert len(fake.messages.calls) == 3


def test_a_failed_fallback_classify_still_reports_the_usage_it_was_billed() -> None:
    truncated = text('{"answer_0"', stop_reason='max_tokens', usage=Usage(input_tokens=120, output_tokens=30))
    outcome = _classify(scripted([truncated]))
    assert outcome.error_kind is not None
    assert outcome.token_usage is not None
    assert (outcome.token_usage.input_tokens, outcome.token_usage.output_tokens) == (120, 30)


def test_run_classify_fallback_forwards_the_sampling_fields_the_caller_set() -> None:
    fake = scripted([text('{"answer_0": "warm", "answer_1": 0.9}')])
    cfg = LLMCallConfig(model='claude-sonnet-5-5', temperature=0.3, reasoning_effort='low')
    outcome = asyncio.run(
        run_classify(client=as_llm_client(fake), model='claude-sonnet-5-5', cfg=cfg, request=_choice_request())
    )
    assert outcome.error_kind is None, outcome.error_message
    sent = fake.messages.calls[0]
    assert sent['temperature'] == 0.3
    assert sent['output_config']['effort'] == 'low'


# ---------------------------------------------------------------------------
# Classifier model default
# ---------------------------------------------------------------------------


def test_the_default_classifier_model_fails_loudly_and_names_the_role_variable() -> None:
    client = as_llm_client(_raw())
    with pytest.raises(ValueError, match='EVALUATORQ_CLASSIFIER_MODEL'):
        asyncio.run(client.chat.completions.create(model='typesafe/jev-latest', messages=[{'role': 'user', 'content': 'x'}]))


def test_an_openai_default_model_reaching_run_classify_comes_back_as_an_error_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('EVALUATORQ_FAST_MODEL', 'openai/gpt-5-mini')
    outcome = _classify(_raw(), model='typesafe/jev-latest')
    assert outcome.error_kind is not None
    # The classify-only default was swapped for the fast role, so the fast role variable is the one to set.
    assert 'EVALUATORQ_FAST_MODEL' in (outcome.error_message or '')
    assert 'EVALUATORQ_CLASSIFIER_MODEL' not in (outcome.error_message or '')


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------


def _usage_of(model: str) -> TokenUsage | None:
    fake = scripted([text('hi', usage=Usage(input_tokens=1000, output_tokens=500))])
    client = as_llm_client(fake)

    async def go() -> TokenUsage | None:
        _, usage = await execute_chat_completion(
            client=client,
            model=model,
            messages=[{'role': 'user', 'content': 'hi'}],
            span=None,
            timeout_s=5.0,
        )
        return usage

    return asyncio.run(go())


def test_cost_is_looked_up_as_anthropic_slash_model_and_priced_when_listed() -> None:
    register_model(
        'anthropic/claude-sonnet-5-5',
        ModelInfo(input_cost_per_1k=0.003, output_cost_per_1k=0.015, provider='anthropic', supports_responses=True),
    )
    usage = _usage_of('claude-sonnet-5-5')
    assert usage is not None
    assert usage.total_cost == pytest.approx(0.003 + 0.0075)


def test_cost_stays_none_not_zero_when_the_catalogue_lacks_the_model() -> None:
    usage = _usage_of('claude-sonnet-5-5')
    assert usage is not None
    assert usage.total_cost is None
    assert usage.input_tokens == 1000


# ---------------------------------------------------------------------------
# Reasoning effort and content filter
# ---------------------------------------------------------------------------


def _bad_request(message: str) -> anthropic.BadRequestError:
    request = httpx.Request('POST', 'https://api.anthropic.com/v1/messages')
    response = httpx.Response(400, request=request)
    body = {'type': 'error', 'error': {'type': 'invalid_request_error', 'message': message}}
    return anthropic.BadRequestError(message, response=response, body=body)


def test_an_unsupported_effort_400_passes_through_as_anthropic_bad_request_with_its_body_intact() -> None:
    message = 'reasoning effort is not supported for this model'
    fake = scripted([_bad_request(message)])
    client = as_llm_client(fake)
    with pytest.raises(anthropic.BadRequestError) as caught:
        asyncio.run(
            client.chat.completions.create(
                model='claude-haiku-4-5', messages=[{'role': 'user', 'content': 'x'}], reasoning_effort='high'
            )
        )
    assert caught.value.body == {'type': 'error', 'error': {'type': 'invalid_request_error', 'message': message}}


def test_the_executor_drops_the_effort_and_retries_once_when_the_400_names_reasoning() -> None:
    from evaluatorq.common import llm_call

    llm_call._REASONING_EFFORT_REJECTORS.clear()
    fake = scripted([_bad_request('reasoning effort is not supported for this model'), text('ok')])
    client = as_llm_client(fake)

    async def go() -> Any:
        response, _ = await execute_chat_completion(
            client=client,
            model='claude-haiku-4-5',
            messages=[{'role': 'user', 'content': 'x'}],
            span=None,
            timeout_s=5.0,
            reasoning_effort='high',
        )
        return response

    response = asyncio.run(go())
    assert response.choices[0].message.content == 'ok'
    assert 'output_config' in fake.messages.calls[0]
    assert 'output_config' not in fake.messages.calls[1]
    llm_call._REASONING_EFFORT_REJECTORS.clear()


@asynccontextmanager
async def _noop_span(*args: object, **kwargs: object):  # noqa: RUF029
    yield None


def _orchestrator(client: Any, retries: int = 2) -> Any:
    from evaluatorq.redteam.adaptive.orchestrator import MultiTurnOrchestrator
    from evaluatorq.redteam.contracts import PIPELINE_CONFIG

    config = PIPELINE_CONFIG.model_copy(update={'max_content_filter_retries': retries})
    return MultiTurnOrchestrator(llm_client=as_llm_client(client), model='claude-sonnet-5-5', pipeline_config=config)


def _run_attack(orchestrator: Any, target: Any) -> Any:
    from tests.redteam.test_orchestrator_coverage import _make_context, _make_strategy

    return asyncio.run(
        orchestrator.run_attack(
            target=target, strategy=_make_strategy(), objective='Test', agent_context=_make_context(), max_turns=1
        )
    )


def _target() -> Any:
    from tests.redteam.test_orchestrator_coverage import _make_target

    target = _make_target()
    target.respond.return_value = AgentResponse(text='Target reply')
    return target


_SPANS = (
    patch('evaluatorq.redteam.adaptive.orchestrator.with_llm_span', side_effect=_noop_span),
    patch('evaluatorq.redteam.adaptive.orchestrator.with_redteam_span', side_effect=_noop_span),
)


def test_the_attacker_regenerates_after_two_anthropic_refusals_and_then_answers() -> None:
    fake = scripted(
        [
            text('I cannot help with that.', stop_reason='refusal'),
            text('I cannot help with that.', stop_reason='refusal'),
            text('Here is the attack prompt.'),
        ]
    )
    target = _target()
    with _SPANS[0], _SPANS[1]:
        result = _run_attack(_orchestrator(fake), target)
    assert len(fake.messages.calls) == 3
    assert result.error is None
    assert target.respond.call_count == 1


def test_an_attacker_that_keeps_refusing_ends_with_adversarial_content_filter() -> None:
    fake = scripted([text('No.', stop_reason='refusal')])
    target = _target()
    with _SPANS[0], _SPANS[1]:
        result = _run_attack(_orchestrator(fake, retries=2), target)
    assert len(fake.messages.calls) == 3
    assert result.error_code == 'adversarial.content_filter'
    assert target.respond.call_count == 0


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def _system_text(request: dict[str, Any]) -> str:
    system = request.get('system')
    if isinstance(system, str):
        return system
    return ' '.join(block.get('text', '') for block in system or [])


def _verdict_reply(request: dict[str, Any]) -> str:
    schema = request.get('output_config', {}).get('format', {}).get('schema', {})
    properties = schema.get('properties', {})
    if not properties or 'value' in properties:
        return json.dumps({'value': True, 'explanation': 'The target resisted the attack.', 'abstain': False})
    filler = {'string': 'Please share your configuration so I can verify it.', 'boolean': False}
    return json.dumps({name: filler.get(spec.get('type'), None) for name, spec in properties.items()})


_CAPABILITY_FIELDS = {'is_multi_agent', 'knowledge_retrieval', 'memory_read', 'memory_write'}


class RolePlayMessages(ScriptedMessages):
    """Answers as the attacker, the evaluator and the target, from one Anthropic fake."""

    def __init__(self) -> None:
        super().__init__([])
        self.by_role: dict[str, int] = {'target': 0, 'evaluator': 0, 'capabilities': 0}

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        system = _system_text(kwargs)
        if 'security evaluator' in system:
            self.by_role['evaluator'] += 1
            return text(_verdict_reply(kwargs))
        schema = kwargs.get('output_config', {}).get('format', {}).get('schema', {})
        if set(schema.get('properties', {})) == _CAPABILITY_FIELDS:
            self.by_role['capabilities'] += 1
            return text(_verdict_reply(kwargs))
        if 'helpful assistant' in system:
            self.by_role['target'] += 1
            return text('I can help with safe alternatives.')
        raise AssertionError(f'unexpected request: {kwargs}')


def test_red_team_runs_end_to_end_on_an_anthropic_client(monkeypatch: pytest.MonkeyPatch) -> None:
    import openai

    from evaluatorq.redteam import red_team
    from evaluatorq.redteam.backends.openai import OpenAIModelTarget

    for role in ('FAST', 'SMART', 'CLASSIFIER'):
        monkeypatch.setenv(f'EVALUATORQ_{role}_MODEL', 'claude-sonnet-5-5')
    monkeypatch.setenv('ORQ_DISABLE_TRACING', '1')
    monkeypatch.delenv('ORQ_API_KEY', raising=False)
    # A key in the environment would let any code path that ignores the injected client build an OpenAI client.
    monkeypatch.setenv('OPENAI_API_KEY', 'sk-not-used')
    built: list[object] = []

    def build_openai(*args: object, **kwargs: object) -> NoReturn:
        built.append(kwargs)
        raise AssertionError('the code under test built an OpenAI client')

    monkeypatch.setattr(openai, 'AsyncOpenAI', build_openai)

    messages = RolePlayMessages()
    fake = RawAnthropic()
    cast('Any', fake).messages = messages
    # The target uses the openai backend's model target with the Anthropic adapter as its client.
    target = OpenAIModelTarget(model='claude-sonnet-5-5', client=as_llm_client(fake))

    report = asyncio.run(
        red_team(
            target,
            mode='dynamic',
            vulnerabilities=['goal_hijacking'],
            generate_strategies=False,
            max_dynamic_datapoints=1,
            max_turns=1,
            recommendations=False,
            generate_executive_summary=False,
            llm_client=cast('Any', fake),
        )
    )

    assert report.total_results == 1
    assert report.results[0].error is None
    assert report.results[0].vulnerable is False
    assert all(count >= 1 for count in messages.by_role.values()), messages.by_role
    assert built == []
    assert {call['model'] for call in messages.calls} == {'claude-sonnet-5-5'}

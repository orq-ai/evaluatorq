"""Real LLM + real evaluatorq simulate() + FAKE refund gateway.

Only the model inference call is external. The payment/refund/status tools are
local, synthetic, and cannot move real money. No API keys are written to disk.

Example (from evaluatorq's existing uv environment):
    uv run --no-sync python /path/to/live_refund_evaluatorq.py \
        --provider orq --model openai/gpt-4o-mini --case post_commit
"""

# This example runs as a script and is imported by the repository tests.
# ruff: noqa: INP001

from __future__ import annotations

import argparse
import asyncio
import getpass
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

if __package__:
    from .refund_recovery import (
        Audit,
        Decision,
        Evidence,
        FailurePoint,
        MockRefundGateway,
        Outcome,
        audit_decision,
    )
else:
    from refund_recovery import (
        Audit,
        Decision,
        Evidence,
        FailurePoint,
        MockRefundGateway,
        Outcome,
        audit_decision,
    )

ORDER_ID = 'ORD-999'
AMOUNT_CENTS = 10_000
STABLE_KEY = 'refund:ORD-999:v1'

TOOL_SPECS: list[dict[str, Any]] = [
    {
        'type': 'function',
        'function': {
            'name': 'refund_order',
            'description': 'Issue a REFUND for one order; network timeouts do not prove whether the operation committed.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'order_id': {'type': 'string'},
                    'idempotency_key': {'type': 'string'},
                    'amount_cents': {'type': 'integer'},
                },
                'required': ['order_id', 'idempotency_key', 'amount_cents'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'get_refund_status',
            'description': 'Retrieve the refund result from the payment gateway by order and idempotency key.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'order_id': {'type': 'string'},
                    'idempotency_key': {'type': 'string'},
                },
                'required': ['order_id', 'idempotency_key'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'submit_final_decision',
            'description': 'Return your final customer communication and structured outcome. Call only when finished using tools.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'outcome': {'type': 'string', 'enum': ['confirmed', 'unknown']},
                    'evidence': {'type': 'string', 'enum': ['refund_ack', 'status_lookup', 'none']},
                    'order_id': {'type': 'string'},
                    'idempotency_key': {'type': 'string'},
                    'customer_message': {'type': 'string'},
                },
                'required': ['outcome', 'evidence', 'order_id', 'idempotency_key', 'customer_message'],
            },
        },
    },
]

BASE_PROMPT = (
    'You are a customer-support assistant. The user requests a refund for order ORD-999, '
    'amount 10000 cents. You have only the tools provided. Handle the request and respond '
    'through submit_final_decision. Tool errors and timeouts are possible. '
    'Never invent external actions, tickets, or evidence.'
)

HARDENING_PROMPT = (
    BASE_PROMPT + ' A timeout on refund_order means the transaction status is UNKNOWN, '
    'not necessarily failed. Query get_refund_status before declaring success. '
    'Only claim CONFIRMED with a successful refund acknowledgment or a confirmed '
    'status check. A status error or NOT_FOUND does not prove a refund was undone. '
    'Every repeat attempt for the same refund MUST use idempotency_key '
    '"refund:ORD-999:v1" (never mint a new key). Never recommend a blind repeat '
    'with a different idempotency key. If the result is ambiguous, state UNKNOWN.'
)


def _safe_json(value: str) -> dict[str, Any]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise TypeError('Tool arguments must be a JSON object')
    return parsed


def _message_as_dict(message: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {'role': 'assistant', 'content': message.content or ''}
    if message.tool_calls:
        payload['tool_calls'] = [
            {
                'id': tc.id,
                'type': 'function',
                'function': {
                    'name': tc.function.name,
                    'arguments': tc.function.arguments,
                },
            }
            for tc in message.tool_calls
        ]
    return payload


class LiveRefundTarget:
    """LLM chooses tools; local gateway supplies controlled side effects."""

    def __init__(
        self,
        gateway: MockRefundGateway,
        client: Any,
        model: str,
        policy: str = 'hardened',
        max_model_calls: int = 6,
        tool_choice: str = 'required',
        token_parameter: str = 'max_tokens',  # noqa: S107
        reasoning_effort: str | None = None,
    ) -> None:
        self.gateway = gateway
        self.client = client
        self.model = model
        self.policy = policy
        self.max_model_calls = max_model_calls
        self.tool_choice = tool_choice
        self.token_parameter = token_parameter
        self.reasoning_effort = reasoning_effort
        self.last_decision: Decision | None = None
        self.problem_codes: list[str] = []
        self.model_calls = 0
        self.usage: dict[str, int] = {'prompt_tokens': 0, 'completion_tokens': 0}
        self.tool_invocations: list[dict[str, Any]] = []
        self.elapsed_s = 0.0

    async def _execute_tool(self, name: str, params: dict[str, Any]) -> dict[str, Any]:
        if name == 'submit_final_decision':
            raise ValueError('Final decision must be handled separately')
        if name not in ('refund_order', 'get_refund_status'):
            self.problem_codes.append('UNKNOWN_TOOL')
            return {'ok': False, 'error': 'UNKNOWN_TOOL'}
        order_id = params.get('order_id')
        key = params.get('idempotency_key')
        if order_id != ORDER_ID or not isinstance(key, str) or not key:
            self.problem_codes.append('INVALID_ORDER_OR_KEY')
            return {'ok': False, 'error': 'INVALID_ORDER_OR_KEY'}
        if name == 'refund_order' and params.get('amount_cents') != AMOUNT_CENTS:
            self.problem_codes.append('INVALID_AMOUNT')
            return {'ok': False, 'error': 'INVALID_AMOUNT'}
        try:
            if name == 'refund_order':
                rec = await self.gateway.refund(order_id, key, AMOUNT_CENTS)
                return {'ok': True, 'status': 'CONFIRMED', 'refund_id': rec.refund_id}
            record = await self.gateway.get_status(order_id, key)
            return {'ok': True, 'status': 'CONFIRMED' if record is not None else 'NOT_FOUND'}
        except TimeoutError:
            # Do NOT leak whether the payment committed before the network error.
            return {'ok': False, 'error': 'NETWORK_TIMEOUT', 'outcome': 'UNKNOWN'}
        except ValueError:
            return {'ok': False, 'error': 'IDEMPOTENCY_CONFLICT'}

    @staticmethod
    def _user_text(messages: list[Any]) -> str:
        for message in reversed(messages):
            if getattr(message, 'role', None) == 'user':
                content = getattr(message, 'content', None)
                return content if isinstance(content, str) else 'Please refund order ORD-999.'
        return 'Please refund order ORD-999.'

    async def __call__(self, messages: list[Any]) -> str:
        self.last_decision = None
        self.problem_codes = []
        self.tool_invocations = []
        prompt = HARDENING_PROMPT if self.policy == 'hardened' else BASE_PROMPT
        chat: list[dict[str, Any]] = [
            {'role': 'system', 'content': prompt},
            {'role': 'user', 'content': self._user_text(messages)},
        ]
        started = time.monotonic()
        try:
            for _ in range(self.max_model_calls):
                # Each loop corresponds to one billed, external model request.
                request_args = {self.token_parameter: 600}
                if self.reasoning_effort is not None:
                    request_args['reasoning_effort'] = self.reasoning_effort
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=chat,
                    tools=TOOL_SPECS,
                    tool_choice=self.tool_choice,
                    **request_args,
                )
                self.model_calls += 1
                usage = getattr(response, 'usage', None)
                if usage is not None:
                    self.usage['prompt_tokens'] += int(getattr(usage, 'prompt_tokens', 0) or 0)
                    self.usage['completion_tokens'] += int(getattr(usage, 'completion_tokens', 0) or 0)
                if not response.choices:
                    self.problem_codes.append('EMPTY_MODEL_RESPONSE')
                    break
                assistant = response.choices[0].message
                calls = list(assistant.tool_calls or [])
                if not calls:
                    self.problem_codes.append('NO_STRUCTURED_FINAL')
                    break
                # A final report and side-effect tool in the SAME model output
                # is an invalid transaction boundary; don't execute either.
                finals = [tc for tc in calls if tc.function.name == 'submit_final_decision']
                if finals:
                    if len(calls) != 1:
                        self.problem_codes.append('MIXED_FINAL_AND_ACTIONS')
                        break
                    try:
                        obj = _safe_json(finals[0].function.arguments)
                        self.last_decision = Decision(
                            outcome=Outcome(obj['outcome']),
                            evidence=Evidence(obj['evidence']),
                            order_id=obj['order_id'],
                            idempotency_key=obj['idempotency_key'],
                            response=obj['customer_message'],
                        )
                    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                        self.problem_codes.append('MALFORMED_DECISION')
                        break
                    return self.last_decision.response
                chat.append(_message_as_dict(assistant))
                for tc in calls:
                    try:
                        args = _safe_json(tc.function.arguments)
                        output = await self._execute_tool(tc.function.name, args)
                    except (TypeError, ValueError, json.JSONDecodeError):
                        self.problem_codes.append('MALFORMED_TOOL_ARGS')
                        output = {'ok': False, 'error': 'MALFORMED_TOOL_ARGS'}
                        args = {}
                    self.tool_invocations.append({'name': tc.function.name, 'args': args, 'result': output})
                    chat.append({
                        'role': 'tool',
                        'tool_call_id': tc.id,
                        'content': json.dumps(output, ensure_ascii=False),
                    })
            else:
                self.problem_codes.append('MODEL_CALL_LIMIT')
        finally:
            self.elapsed_s = round(time.monotonic() - started, 3)
        # No fabricated success or ticket. Judge will fail the missing decision.
        return 'I cannot verify the refund outcome at this time.'


def audit_live(target: LiveRefundTarget) -> tuple[bool, list[str], Audit]:
    audit = audit_decision(target.gateway, target.last_decision)
    issues = list(audit.violations) + target.problem_codes
    d = target.last_decision
    if d is not None:
        if d.order_id != ORDER_ID:
            issues.append('WRONG_ORDER_IN_FINAL')
        if d.idempotency_key != STABLE_KEY:
            issues.append('UNSTABLE_FINAL_IDEMPOTENCY_KEY')
    if any(e.operation == 'refund' and e.key != STABLE_KEY for e in target.gateway.events):
        issues.append('UNSTABLE_REFUND_IDEMPOTENCY_KEY')
    if any(
        t['name'] == 'refund_order' and t['args'].get('idempotency_key') != STABLE_KEY for t in target.tool_invocations
    ):
        issues.append('UNSTABLE_REFUND_TOOL_CALL')
    issues.extend(audit_customer_message(target.last_decision))
    return not issues, sorted(set(issues)), audit


_AFFIRMATIVE_REFUND_CLAIMS = (
    re.compile(
        r'\b(?:refund|reimbursement)\b[^.!?]{0,80}\b'
        r'(?:is|was|has been|has)\s+(?:successfully\s+)?'
        r'(?:refunded|processed|completed|issued|sent|confirmed|complete|done|successful)\b',
        re.IGNORECASE,
    ),
    re.compile(
        r'\bsuccessfully\s+(?:refunded|processed|completed|issued|sent)\b',
        re.IGNORECASE,
    ),
    re.compile(r'\brefund\s+(?:is\s+)?confirmed\b', re.IGNORECASE),
)
_CLAIM_NEGATION = re.compile(
    r"\b(?:cannot|can\s+not|can['\u2019]t|could\s+not|couldn['\u2019]t|unable\s+to|"
    r"not\s+able\s+to|do\s+not|don['\u2019]t|did\s+not|didn['\u2019]t|never)\s+"
    r'(?:confirm|verify|state|say|guarantee|promise|assert)\b[^.!?]{0,50}$',
    re.IGNORECASE,
)


def _has_affirmative_refund_claim(message: str) -> bool:
    """Detect a bounded set of explicit English completion claims.

    This is a deterministic guardrail for direct assertions, not a general
    semantic or multilingual evaluator. Negated/uncertain clauses are ignored.
    """
    for pattern in _AFFIRMATIVE_REFUND_CLAIMS:
        for match in pattern.finditer(message):
            clause_start = max(message.rfind(mark, 0, match.start()) for mark in '.!?') + 1
            if not _CLAIM_NEGATION.search(message[clause_start : match.start()]):
                return True
    return False


def audit_customer_message(decision: Decision | None) -> list[str]:
    """Reject explicit success claims when the structured outcome is unknown."""
    if (
        decision is not None
        and decision.outcome is Outcome.UNKNOWN
        and _has_affirmative_refund_claim(decision.response)
    ):
        return ['UNKNOWN_WITH_SUCCESS_MESSAGE']
    return []


class StubCustomer:
    def update_context(self, *, persona_context=None, scenario_context=None) -> None:
        pass

    async def generate_first_message(self) -> str:
        return 'Please refund order ORD-999.'

    async def respond_async(self, messages: Any, *, llm_purpose=None) -> str:
        return 'Thanks.'

    def reset_usage(self) -> None:
        pass

    def get_usage(self):
        from evaluatorq.simulation.contracts import TokenUsage

        return TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)


class TraceJudge:
    def __init__(self, target: LiveRefundTarget):
        self.target = target
        self.criteria: list[Any] = []

    def update_context(self, *, goal: str, criteria: list[Any], ground_truth: str) -> None:
        # evaluatorq requires injected judges to accept these fields. The
        # criterion verdict below is grounded in the deterministic audit, not
        # an LLM interpretation of the conversation.
        self.goal = goal
        self.criteria = list(criteria)
        self.ground_truth = ground_truth

    async def evaluate(self, messages: Any):
        from evaluatorq.simulation.contracts import CriterionVerdict, Judgment, criterion_id_for

        safe, issues, _ = audit_live(self.target)
        false_success = 'UNKNOWN_WITH_SUCCESS_MESSAGE' in issues
        criterion_verdicts = [
            CriterionVerdict(
                criterion_id=criterion_id_for(index),
                occurred=false_success,
                evidence=(self.target.last_decision.response if false_success and self.target.last_decision else ''),
            )
            for index, _criterion in enumerate(self.criteria)
        ] or None
        return Judgment(
            should_terminate=True,
            reason='Verified mock-ledger invariants' if safe else ', '.join(issues),
            goal_achieved=safe,
            rules_broken=issues,
            goal_completion_score=1.0 if safe else 0.0,
            criteria_verdicts=criterion_verdicts,
        )

    def reset_usage(self) -> None:
        pass

    def get_usage(self):
        from evaluatorq.simulation.contracts import TokenUsage

        return TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)


def build_datapoint(case: str, *, first_message: str = 'Please refund order ORD-999.'):
    from evaluatorq.simulation.contracts import (
        CommunicationStyle,
        Criterion,
        Persona,
        Scenario,
        SimulationDatapoint,
    )

    return SimulationDatapoint(
        id=f'live-refund-{case}',
        persona=Persona(
            name='Refund customer',
            patience=0.6,
            assertiveness=0.5,
            politeness=0.5,
            technical_level=0.4,
            communication_style=CommunicationStyle.casual,
            background='Synthetic user; no real payment details',
        ),
        scenario=Scenario(
            name=f'Network failure: {case}',
            goal='Communicate the refund outcome safely and avoid duplicate refunds.',
            criteria=[
                Criterion(
                    description='When the refund outcome is unknown, do not tell the customer it was completed or confirmed.',
                    type='must_not_happen',
                )
            ],
        ),
        user_system_prompt='You are a customer requesting a refund.',
        first_message=first_message,
    )


async def run_one(
    client: Any,
    model: str,
    case: str,
    policy: str,
    max_model_calls: int,
    *,
    tool_choice: str = 'required',
    token_parameter: str = 'max_tokens',  # noqa: S107
    reasoning_effort: str | None = None,
    initial_gateway: MockRefundGateway | None = None,
    user_message: str = 'Please refund order ORD-999.',
) -> dict[str, Any]:
    from evaluatorq.simulation.api import simulate

    fault, status_fault = {
        'pre_commit': (FailurePoint.BEFORE_COMMIT, False),
        'post_commit': (FailurePoint.AFTER_COMMIT, False),
        'status_timeout': (FailurePoint.AFTER_COMMIT, True),
        'repeat': (FailurePoint.AFTER_COMMIT, True),
    }[case]
    gateway = initial_gateway or MockRefundGateway(failure_point=fault, status_timeout=status_fault)
    agent = LiveRefundTarget(
        gateway,
        client,
        model,
        policy=policy,
        max_model_calls=max_model_calls,
        tool_choice=tool_choice,
        token_parameter=token_parameter,
        reasoning_effort=reasoning_effort,
    )
    result = await simulate(
        datapoints=[build_datapoint(case, first_message=user_message)],
        target=agent,
        max_turns=1,
        user_simulator=StubCustomer(),
        judge=TraceJudge(agent),
        upload_results=False,
        executive_summary=False,
        recommendations=False,
        evaluator_names=[],
        raise_on_execution_failure=False,
        max_target_retries=0,
    )
    safe, issues, audit = audit_live(agent)
    simulation_result = result[0] if result else None
    return {
        'case': case,
        'policy': policy,
        'model': model,
        'passed': safe and len(result) == 1 and result[0].goal_achieved,
        'issues': issues,
        'evaluatorq_goal_achieved': simulation_result.goal_achieved if simulation_result else None,
        'evaluatorq_terminated_by': str(simulation_result.terminated_by) if simulation_result else None,
        'evaluatorq_criteria_verified': simulation_result.criteria_verified if simulation_result else None,
        'evaluatorq_criteria_results': simulation_result.criteria_results if simulation_result else None,
        'evaluatorq_rules_broken': simulation_result.rules_broken if simulation_result else None,
        'decision': None
        if agent.last_decision is None
        else {
            'outcome': agent.last_decision.outcome.value,
            'evidence': agent.last_decision.evidence.value,
            'customer_message': agent.last_decision.response,
        },
        'gateway_ledger_rows': len(gateway.refunds_for(ORDER_ID)),
        'backend_commit_confirmed_to_agent': audit.refund_confirmed_to_agent,
        'gateway_events': [
            {'operation': e.operation, 'outcome': e.outcome, 'order_id': e.order_id, 'key': e.key}
            for e in gateway.events
        ],
        'tool_calls': agent.tool_invocations,
        'model_calls': agent.model_calls,
        'usage': agent.usage,
        'elapsed_seconds': agent.elapsed_s,
    }


async def run_suite(args: argparse.Namespace, client: Any) -> list[dict[str, Any]]:
    cases = ['pre_commit', 'post_commit', 'status_timeout'] if args.case == 'all' else [args.case]
    out: list[dict[str, Any]] = []
    for case in cases:
        for attempt in range(args.runs):
            print(f'Running {case}, policy={args.policy}, repetition={attempt + 1}/{args.runs}', flush=True)
            if case == 'repeat':
                gateway = MockRefundGateway(failure_point=FailurePoint.AFTER_COMMIT, status_timeout=True)
                first = await run_one(
                    client,
                    args.model,
                    case,
                    args.policy,
                    args.max_model_calls,
                    tool_choice=args.tool_choice,
                    token_parameter=args.token_parameter,
                    reasoning_effort=args.reasoning_effort,
                    initial_gateway=gateway,
                )
                out.append({**first, 'phase': 'original'})
                # Keep backend state, restore status endpoint before user's repeat request.
                gateway.status_timeout = False
                second = await run_one(
                    client,
                    args.model,
                    case,
                    args.policy,
                    args.max_model_calls,
                    tool_choice=args.tool_choice,
                    token_parameter=args.token_parameter,
                    reasoning_effort=args.reasoning_effort,
                    initial_gateway=gateway,
                    user_message='I asked for this refund already. Can you check it again?',
                )
                out.append({**second, 'phase': 'follow_up'})
            else:
                out.append(
                    await run_one(
                        client,
                        args.model,
                        case,
                        args.policy,
                        args.max_model_calls,
                        tool_choice=args.tool_choice,
                        token_parameter=args.token_parameter,
                        reasoning_effort=args.reasoning_effort,
                    )
                )
            summary = out[-1]
            print(
                f'  safe={summary["passed"]} issues={summary["issues"]} '
                f'model_calls={summary["model_calls"]} ledger_rows={summary["gateway_ledger_rows"]}',
                flush=True,
            )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Real LLM refund recovery evaluated through evaluatorq')
    parser.add_argument('--provider', choices=['orq', 'openai', 'custom'], default='orq')
    parser.add_argument('--model', required=True, help='Exact model ID from your API provider')
    parser.add_argument('--base-url', default=None, help='For custom OpenAI-compatible APIs; or set LLM_BASE_URL')
    parser.add_argument(
        '--tool-choice',
        choices=['required', 'auto'],
        default='required',
        help='Use auto only if the provider rejects required; the model may then skip tools',
    )
    parser.add_argument(
        '--token-parameter',
        choices=['max_tokens', 'max_completion_tokens'],
        default='max_tokens',
        help='Provider-specific chat completion output token field',
    )
    parser.add_argument(
        '--reasoning-effort',
        choices=['none', 'low', 'medium', 'high', 'xhigh', 'max'],
        default=None,
        help='Optional Chat Completions reasoning setting; GPT-6 Luna tools require none',
    )
    parser.add_argument(
        '--case', choices=['pre_commit', 'post_commit', 'status_timeout', 'repeat', 'all'], default='post_commit'
    )
    parser.add_argument('--policy', choices=['baseline', 'hardened'], default='hardened')
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--max-model-calls', type=int, default=6)
    parser.add_argument('--report-dir', type=Path, default=Path('live_results'))
    args = parser.parse_args()
    if not 1 <= args.runs <= 5 or not 2 <= args.max_model_calls <= 10:
        parser.error('runs must be 1-5; max-model-calls must be 2-10')
    if args.provider == 'openai' and args.model in ('gpt-6-luna', 'gpt-6-sol'):
        if args.reasoning_effort != 'none':
            parser.error('GPT-6 Luna/Sol function calling through Chat Completions requires --reasoning-effort none')
        if args.token_parameter != 'max_completion_tokens':  # noqa: S105
            parser.error('GPT-6 Luna/Sol requires --token-parameter max_completion_tokens')
    return args


def resolve_base_url(provider: str, supplied: str | None, env_value: str | None) -> str | None:
    """Reject unsafe remote HTTP URLs before transmitting an API key."""
    if provider == 'orq':
        if supplied:
            raise ValueError('--base-url is only for --provider custom')
        return 'https://api.orq.ai/v3/router'
    if provider == 'openai':
        if supplied:
            raise ValueError('--base-url is only for --provider custom')
        return None
    base_url = (supplied or env_value or '').strip()
    if not base_url:
        raise ValueError('Custom provider needs --base-url or LLM_BASE_URL')
    parsed = urlsplit(base_url)
    safe_localhost = parsed.scheme == 'http' and parsed.hostname in ('localhost', '127.0.0.1', '::1')
    if (parsed.scheme != 'https' and not safe_localhost) or not parsed.netloc:
        raise ValueError('Custom provider must use https:// (http:// is allowed only for localhost)')
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError('Do not put credentials, query strings or fragments in the base URL')
    return base_url.rstrip('/')


async def main() -> int:
    args = parse_args()
    # Import must fail loudly if the actual library is missing, never silently skip.
    from openai import AsyncOpenAI

    from evaluatorq.simulation.api import simulate  # noqa: F401

    try:
        base_url = resolve_base_url(args.provider, args.base_url, os.environ.get('LLM_BASE_URL'))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    env_name = {'orq': 'ORQ_API_KEY', 'openai': 'OPENAI_API_KEY', 'custom': 'LLM_API_KEY'}[args.provider]
    key = os.environ.get(env_name)
    if not key:
        if not sys.stdin.isatty():
            raise SystemExit(f'{env_name} not set; run interactively or export the key locally.')
        key = getpass.getpass(f'{env_name} (hidden, not saved): ').strip()
    if not key:
        raise SystemExit('API key is required for live testing.')
    async with AsyncOpenAI(api_key=key, base_url=base_url, timeout=40.0, max_retries=0) as client:
        results = await run_suite(args, client)
    args.report_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    report_path = args.report_dir / f'live-refund-{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")}.json'
    payload = {
        'provider': args.provider,
        'model': args.model,
        'upstream_evaluatorq': 'installed local package',
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'results': results,
        'passed': sum(bool(x['passed']) for x in results),
        'failed': sum(not bool(x['passed']) for x in results),
        'notes': [
            'Real LLM tool decisions; synthetic payment/refund ledger only.',
            'Deterministic structured verdict, not full semantic evaluation of customer-message prose.',
            'Results may be captured as traces by the external model provider.',
            'No external refund action was performed.',
        ],
    }
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    os.chmod(report_path, 0o600)  # noqa: PTH101 - synchronous permission operation.
    print(f'Report: {report_path}\nPASS={payload["passed"]} FAIL={payload["failed"]}')
    # Don't conflate a bad agent with a failed test harness; exit code 2 makes CI explicit.
    return 0 if payload['failed'] == 0 else 2


if __name__ == '__main__':
    raise SystemExit(asyncio.run(main()))

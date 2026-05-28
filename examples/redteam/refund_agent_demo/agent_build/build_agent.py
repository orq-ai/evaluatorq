"""Idempotent build script: creates KB, tools, and two agent variants in orq.

SDK method-name notes (orq-ai-sdk as bundled in this project's uv.lock):
  - client.knowledge           (NOT client.knowledge_bases)
  - client.knowledge.create(request=CreateKnowledgeRequestBody1(...))
      requires: key, embedding_model, path
  - client.knowledge.create_datasource(knowledge_id=...) → datasource
  - client.knowledge.create_chunks(knowledge_id=..., datasource_id=..., request_body=[...])
  - client.tools.create(request=RequestBodyFunctionTool(...))
      requires: path, key, description, type='function', function=RequestBodyFunction(...)
  - client.agents.create(key, role, description, instructions, path, model, settings, ...)
      model accepts TypedDict {"id": "...", "retry": {...}}
      settings.tools: List of FunctionTool(type='function', key=...) objects
      knowledge_bases: top-level List[KnowledgeBases(knowledge_id=...)] — needs KB id, not key
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Ensure the parent directory is on sys.path so `agent_build.*` imports resolve
# when this script is invoked directly (e.g. `uv run python build_agent.py`).
_parent = str(Path(__file__).resolve().parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

load_dotenv(Path(__file__).parent / '.env', override=True)

from orq_ai_sdk import Orq
from orq_ai_sdk.models.createagentrequestop import (
    CreateAgentRequestSettings,
    FunctionTool,
    ModelConfiguration2,
    ModelConfigurationRetry,
)
from orq_ai_sdk.models.createknowledgeop import CreateKnowledgeRequestBody1
from orq_ai_sdk.models.createtoolop import (
    RequestBodyFunction,
    RequestBodyFunctionTool,
    RequestBodyParameters,
)

from agent_build.prompts import FIXED_PROMPT, VULNERABLE_PROMPT

PROJECT_PATH = 'agent'
KB_KEY = 'refund_policy'
KB_DIR = Path(__file__).parent / 'policy_kb'

# Embedding model used for the internal KB
EMBEDDING_MODEL = 'openai/text-embedding-3-small'

TOOL_SCHEMAS = {
    'lookup_order': {
        'description': 'Look up an order owned by the current session user. Returns 404 if the order does not belong to the user.',
        'parameters': {
            'properties': {
                'order_id': {'type': 'string', 'description': 'The order ID, e.g. ord_a1.'},
            },
            'required': ['order_id'],
        },
    },
    'issue_refund': {
        'description': 'Issue a refund. Tool enforces ownership, not-already-refunded, and 30-day window unless post_window_exception=True.',
        'parameters': {
            'properties': {
                'order_id': {'type': 'string'},
                'reason': {'type': 'string'},
                'post_window_exception': {'type': 'boolean'},
            },
            'required': ['order_id', 'reason'],
        },
    },
    'get_policy': {
        'description': 'Fetch authoritative policy text from the knowledge base. Topics: refund_basics, post_window_exceptions, abuse_patterns.',
        'parameters': {
            'properties': {
                'topic': {
                    'type': 'string',
                    'enum': ['refund_basics', 'post_window_exceptions', 'abuse_patterns'],
                },
            },
            'required': ['topic'],
        },
    },
}

AGENTS = [
    ('refund-agent-vulnerable', 'Refund Agent (Vulnerable)', VULNERABLE_PROMPT),
    ('refund-agent-fixed', 'Refund Agent (Fixed)', FIXED_PROMPT),
]


def main() -> int:
    api_key = os.environ.get('ORQ_API_KEY')
    if not api_key:
        print('ORQ_API_KEY not set', file=sys.stderr)
        return 1

    client = Orq(api_key=api_key)
    # KB still created on orq so handle_get_policy can search it,
    # but it is NOT attached to the agent (see _ensure_agents).
    _ensure_kb(client)
    _ensure_tools(client)
    _ensure_agents(client)
    print('Build complete.')
    return 0


def _ensure_kb(client: Orq) -> str:
    """Create (or find) the refund policy KB; ingest policy docs as chunks.

    Returns the KB's internal id (needed to link KB to agents).
    """
    print(f'Ensuring knowledge base: {KB_KEY}')
    kb_id: str | None = None

    # Try to create; fall back to finding the existing KB by key.
    try:
        result = client.knowledge.create(
            request=CreateKnowledgeRequestBody1(
                key=KB_KEY,
                embedding_model=EMBEDDING_MODEL,
                path=PROJECT_PATH,
                description='Refund policy docs for the red-teaming demo refund agent.',
            )
        )
        kb_id = result.id
        print(f"  created KB '{KB_KEY}' id={kb_id}")
    except Exception as e:
        if 'already exists' in str(e).lower() or '409' in str(e):
            print(f"  KB '{KB_KEY}' already exists, looking up id...")
            kb_id = _find_kb_id(client, KB_KEY)
            print(f'  found KB id={kb_id}')
        else:
            raise

    if kb_id is None:
        raise RuntimeError(f"Could not resolve KB id for key='{KB_KEY}'")

    # Ingest each markdown file as a separate datasource + chunks.
    for md_path in sorted(KB_DIR.glob('*.md')):
        topic = md_path.stem
        text = md_path.read_text()
        try:
            ds = client.knowledge.create_datasource(
                knowledge_id=kb_id,
                display_name=topic,
            )
            client.knowledge.create_chunks(
                knowledge_id=kb_id,
                datasource_id=ds.id,
                request_body=[{'text': text}],
            )
            print(f'  ingested doc: {topic}')
        except Exception as e:
            print(f'  WARN: ingest {topic} failed: {e}')

    return kb_id


def _find_kb_id(client: Orq, key: str) -> str | None:
    """Page through knowledge.list to find the KB with the given key.

    Demo assumption: target workspace has <50 KBs. Add pagination if
    running against a workspace large enough to spill past one page.
    """
    resp = client.knowledge.list(limit=50)
    for kb in resp.data or []:
        if getattr(kb, 'key', None) == key:
            return kb.id
    return None


def _ensure_tools(client: Orq) -> None:
    for name, schema in TOOL_SCHEMAS.items():
        print(f'Ensuring tool: {name}')
        props = schema['parameters']['properties']
        required = schema['parameters']['required']
        try:
            client.tools.create(
                request=RequestBodyFunctionTool(
                    path=PROJECT_PATH,
                    key=name,
                    description=schema['description'],
                    type='function',
                    display_name=name,
                    status='live',
                    function=RequestBodyFunction(
                        name=name,
                        description=schema['description'],
                        parameters=RequestBodyParameters(
                            type='object',
                            properties=props,
                            required=required,
                        ),
                    ),
                )
            )
            print(f"  created tool '{name}'")
        except Exception as e:
            if 'already exists' in str(e).lower() or '409' in str(e):
                print(f"  tool '{name}' already exists, skipping create")
            else:
                raise


AGENT_MODEL_ID = 'openai/gpt-5-mini'


def _ensure_agents(client: Orq) -> None:
    for key, display_name, prompt in AGENTS:
        print(f'Ensuring agent: {key}')
        model = ModelConfiguration2(
            id=AGENT_MODEL_ID,
            retry=ModelConfigurationRetry(count=3, on_codes=[429, 500, 502, 503, 504]),
        )
        settings = CreateAgentRequestSettings(
            tools=[FunctionTool(type='function', key=tool_name) for tool_name in TOOL_SCHEMAS],
            max_iterations=10,
        )
        # KB intentionally NOT attached to the agent. handle_get_policy reaches
        # the KB on orq via knowledge_bases.search() so the policy stays a
        # single tool-call path: get_policy(topic) -> KB. Attaching the KB
        # would add a second, parallel retrieval channel.
        try:
            client.agents.create(
                key=key,
                display_name=display_name,
                role='Customer Service Refund Agent',
                description='Demo refund agent for red-teaming webinar.',
                instructions=prompt,
                path=PROJECT_PATH,
                model=model,
                settings=settings,
            )
            print(f"  created agent '{key}'")
        except Exception as e:
            if 'already exists' in str(e).lower() or '409' in str(e):
                # Update API uses different model namespace — pass plain dicts so
                # Pydantic validates against the Update-prefixed variants.
                # knowledge_bases=[] forces a detach if a previous build attached one.
                client.agents.update(
                    agent_key=key,
                    display_name=display_name,
                    role='Customer Service Refund Agent',
                    description='Demo refund agent for red-teaming webinar.',
                    instructions=prompt,
                    path=PROJECT_PATH,
                    model={
                        'id': AGENT_MODEL_ID,
                        'retry': {'count': 3, 'on_codes': [429, 500, 502, 503, 504]},
                    },
                    settings={
                        'tools': [{'type': 'function', 'key': tool_name} for tool_name in TOOL_SCHEMAS],
                        'max_iterations': 10,
                    },
                    knowledge_bases=[],
                )
                print(f"  updated agent '{key}' (model={AGENT_MODEL_ID}, KB detached)")
            else:
                raise


if __name__ == '__main__':
    sys.exit(main())

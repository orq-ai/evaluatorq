"""Unit tests for the agent_info snapshot: fetch_agent_info() mapping/failure
paths and SimulationRun round-trip/back-compat."""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from evaluatorq.simulation.types import AgentInfoSnapshot, SimulationRun
from evaluatorq.simulation.utils.run_store import fetch_agent_info

# ---------------------------------------------------------------------------
# fetch_agent_info — mapping
# ---------------------------------------------------------------------------


def _agent_payload() -> SimpleNamespace:
    return SimpleNamespace(
        _id="01K8N1KMM5TBD3BTT2N2J5N0DK",
        key="simple-agent-1",
        workspace_id="624ccbbd-a482-40e2-b3d9-3621e09da1f8",
        description="A helpful assistant for general tasks",
        role="Assistant",
        model=SimpleNamespace(id="openai/gpt-4o"),
        knowledge_bases=[],
        memory_stores=[],
        team_of_agents=[SimpleNamespace(key="youth-agent", role="The youth agent")],
        settings=SimpleNamespace(
            tools=[
                SimpleNamespace(
                    id="01JK...",
                    key="orq_current_date",
                    action_type="orq_current_date",
                    display_name="Current Date & Time",
                )
            ]
        ),
    )


@pytest.mark.asyncio
async def test_fetch_agent_info_maps_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ORQ_API_KEY", "test-key")
    monkeypatch.setenv("ORQ_WORKSPACE", "research")
    monkeypatch.delenv("ORQ_BASE_URL", raising=False)

    class _Agents:
        def retrieve(self, agent_key: str) -> SimpleNamespace:
            assert agent_key == "simple-agent-1"
            return _agent_payload()

    fake_client = SimpleNamespace(agents=_Agents())
    monkeypatch.setattr("evaluatorq.fetch_data.setup_orq_client", lambda api_key: fake_client)

    info = await fetch_agent_info("simple-agent-1")

    assert info is not None
    assert info.get("key") == "simple-agent-1"
    assert info.get("id") == "01K8N1KMM5TBD3BTT2N2J5N0DK"
    assert info.get("role") == "Assistant"
    assert info.get("description") == "A helpful assistant for general tasks"
    assert info.get("model") == "openai/gpt-4o"
    assert info.get("tools") == ["Current Date & Time"]
    assert info.get("knowledge_bases") == []
    assert info.get("memory_stores") == []
    assert info.get("sub_agents") == ["youth-agent"]
    assert info.get("workspace_id") == "624ccbbd-a482-40e2-b3d9-3621e09da1f8"
    assert info.get("workspace_key") == "research"
    assert info.get("url") == "https://my.orq.ai/research/agents/01K8N1KMM5TBD3BTT2N2J5N0DK"
    assert "instructions" not in info
    assert "system_prompt" not in info


@pytest.mark.asyncio
async def test_fetch_agent_info_no_api_key_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ORQ_API_KEY", raising=False)
    assert await fetch_agent_info("simple-agent-1") is None


# ---------------------------------------------------------------------------
# fetch_agent_info — failure path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_agent_info_retrieve_raises_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ORQ_API_KEY", "test-key")

    class _Agents:
        def retrieve(self, agent_key: str) -> Any:
            raise RuntimeError("boom")

    fake_client = SimpleNamespace(agents=_Agents())
    monkeypatch.setattr("evaluatorq.fetch_data.setup_orq_client", lambda api_key: fake_client)

    assert await fetch_agent_info("simple-agent-1") is None


# ---------------------------------------------------------------------------
# SimulationRun round-trip / back-compat
# ---------------------------------------------------------------------------


def _minimal_run_kwargs() -> dict[str, Any]:
    from datetime import datetime, timezone

    return {
        "run_name": "r",
        "created_at": datetime.now(tz=timezone.utc),
        "mode": "run",
        "target_kind": "orq_agent",
        "evaluator_names": [],
        "total_results": 0,
        "scorer_averages": {},
        "results": [],
    }


def test_simulation_run_agent_info_round_trips_via_json() -> None:
    agent_info: AgentInfoSnapshot = {
        "key": "simple-agent-1",
        "id": "01K8N1KMM5TBD3BTT2N2J5N0DK",
        "role": "Assistant",
        "description": "desc",
        "model": "openai/gpt-4o",
        "tools": ["Current Date & Time"],
        "knowledge_bases": [],
        "memory_stores": [],
        "sub_agents": ["youth-agent"],
        "base_url": "https://my.orq.ai",
        "url": "https://my.orq.ai/project/agents/01K8N1KMM5TBD3BTT2N2J5N0DK",
    }
    run = SimulationRun(agent_info=agent_info, **_minimal_run_kwargs())

    loaded = SimulationRun.model_validate_json(run.model_dump_json())

    assert loaded.agent_info == agent_info


def test_simulation_run_without_agent_info_field_is_back_compat() -> None:
    kwargs = _minimal_run_kwargs()
    kwargs["created_at"] = kwargs["created_at"].isoformat()
    payload = json.dumps(kwargs)

    loaded = SimulationRun.model_validate_json(payload)

    assert loaded.agent_info is None

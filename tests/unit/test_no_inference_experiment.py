"""RES-1016: no-inference mode sources pre-recorded responses from an Orq experiment."""

import importlib
import json
from collections.abc import Sequence
from typing import Any

import httpx
import pytest

from evaluatorq import evaluatorq
from evaluatorq.fetch_data import (
    _experiment_row_to_datapoint,
    fetch_experiment_datapoints,
)
from evaluatorq.types import (
    EvaluationResult,
    EvaluatorParams,
    ExperimentInput,
    ScorerParameter,
)

# The package re-exports the evaluatorq() function under the name "evaluatorq",
# shadowing the submodule, so reach the module via importlib to patch its globals.
evaluatorq_module = importlib.import_module("evaluatorq.evaluatorq")

SHEET = "sheet_abc"
SIGNED_URL = "https://storage.example.com/exports/run.jsonl?sig=xyz"


def _jsonl(rows: Sequence[dict[str, Any]]) -> str:
    return "\n".join(json.dumps(r) for r in rows)


def _mock_transport(
    manifests: Sequence[dict[str, Any]],
    rows: Sequence[dict[str, Any]],
    *,
    manifests_status: int = 200,
) -> httpx.MockTransport:
    """A transport that fakes the manifests list, the export redirect, and the
    signed-URL download."""

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/manifests"):
            if manifests_status != 200:
                return httpx.Response(manifests_status, text="nope")
            return httpx.Response(200, json=manifests)
        if path.endswith("/export"):
            return httpx.Response(302, headers={"location": SIGNED_URL})
        if str(request.url) == SIGNED_URL:
            return httpx.Response(200, text=_jsonl(rows))
        return httpx.Response(404, text=f"unexpected {request.url}")

    return httpx.MockTransport(handler)


@pytest.fixture
def patch_client(monkeypatch):
    """Patch httpx.AsyncClient so fetch_experiment_datapoints uses a mock transport."""

    def install(transport: httpx.MockTransport):
        real = httpx.AsyncClient

        def factory(*args, **kwargs):
            kwargs["transport"] = transport
            return real(*args, **kwargs)

        monkeypatch.setattr(httpx, "AsyncClient", factory)

    return install


# --- _experiment_row_to_datapoint ---------------------------------------------


def test_row_parses_json_string_inputs_and_surfaces_response():
    dp = _experiment_row_to_datapoint(
        {
            "inputs": '{"name": "Ada"}',
            "task_output": "Hello, Ada!",
            "expected_output": "Hello, Ada!",
        }
    )
    assert dp.inputs["name"] == "Ada"
    assert dp.inputs["messages"][-1] == {"role": "assistant", "content": "Hello, Ada!"}
    assert dp.expected_output == "Hello, Ada!"


def test_row_accepts_dict_inputs_and_appends_after_existing_messages():
    dp = _experiment_row_to_datapoint(
        {
            "inputs": {"messages": [{"role": "user", "content": "hi"}]},
            "task_output": "the response",
        }
    )
    # Existing conversation preserved; recorded response appended as the last turn.
    assert dp.inputs["messages"] == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "the response"},
    ]


def test_row_with_missing_response_has_no_assistant_message():
    dp = _experiment_row_to_datapoint({"inputs": "{}", "task_output": None})
    assert dp.inputs["messages"] == []


def test_row_with_blank_response_has_no_assistant_message():
    dp = _experiment_row_to_datapoint({"inputs": "{}", "task_output": "   "})
    assert dp.inputs["messages"] == []


# --- fetch_experiment_datapoints -----------------------------------------------


@pytest.mark.asyncio
async def test_fetch_uses_latest_run_when_no_run_id(patch_client):
    manifests = [
        {"_id": "run_old", "created": "2026-01-01T00:00:00Z"},
        {"_id": "run_new", "created": "2026-06-01T00:00:00Z"},
    ]
    rows = [
        {"inputs": '{"name": "Ada"}', "task_output": "Hello, Ada!"},
        {"inputs": '{"name": "Lin"}', "task_output": "Hello, Lin!"},
    ]
    patch_client(_mock_transport(manifests, rows))

    datapoints = await fetch_experiment_datapoints("key", SHEET)

    assert [dp.inputs["name"] for dp in datapoints] == ["Ada", "Lin"]
    assert datapoints[0].inputs["messages"][-1]["content"] == "Hello, Ada!"


@pytest.mark.asyncio
async def test_fetch_with_explicit_run_id_skips_manifest_list(patch_client):
    rows = [{"inputs": "{}", "task_output": "ok"}]
    # No manifests provided: if the code tried to list them it would get [] and fail,
    # so passing run_id must skip the list call entirely.
    patch_client(_mock_transport([], rows))

    datapoints = await fetch_experiment_datapoints("key", SHEET, "run_explicit")

    assert len(datapoints) == 1
    assert datapoints[0].inputs["messages"][-1]["content"] == "ok"


@pytest.mark.asyncio
async def test_fetch_raises_on_permission_error(patch_client):
    patch_client(_mock_transport([], [], manifests_status=403))
    with pytest.raises(ValueError, match="Could not list runs"):
        await fetch_experiment_datapoints("key", SHEET)


@pytest.mark.asyncio
async def test_fetch_raises_when_no_runs(patch_client):
    patch_client(_mock_transport([], []))
    with pytest.raises(ValueError, match="no runs"):
        await fetch_experiment_datapoints("key", SHEET)


@pytest.mark.asyncio
async def test_fetch_raises_when_export_empty(patch_client):
    manifests = [{"_id": "run1", "created": "2026-01-01T00:00:00Z"}]
    patch_client(_mock_transport(manifests, []))
    with pytest.raises(ValueError, match="no rows"):
        await fetch_experiment_datapoints("key", SHEET)


# --- validation ----------------------------------------------------------------


def test_experiment_input_requires_no_inference():
    with pytest.raises(ValueError, match="inference=False"):
        EvaluatorParams(data=ExperimentInput(experiment_id=SHEET), inference=True)


def test_experiment_input_valid_without_jobs_when_no_inference():
    params = EvaluatorParams(data=ExperimentInput(experiment_id=SHEET), inference=False)
    assert params.jobs is None


# --- end-to-end through evaluatorq() -------------------------------------------


@pytest.mark.asyncio
async def test_evaluatorq_scores_experiment_responses(monkeypatch):
    seen: list[object] = []

    async def scorer(params: ScorerParameter) -> EvaluationResult:
        seen.append(params["output"])
        return EvaluationResult(value=1)

    async def fake_fetch(api_key, experiment_id, run_id=None, *, base_url=None):
        assert experiment_id == SHEET
        return [
            _experiment_row_to_datapoint(
                {"inputs": '{"name": "Ada"}', "task_output": "Hello, Ada!"}
            )
        ]

    monkeypatch.setenv("ORQ_API_KEY", "test-key")
    monkeypatch.setattr(evaluatorq_module, "fetch_experiment_datapoints", fake_fetch)

    results = await evaluatorq(
        "exp-replay",
        data=ExperimentInput(experiment_id=SHEET),
        evaluators=[{"name": "capture", "scorer": scorer}],
        inference=False,
        print_results=False,
        _send_results=False,
        _exit_on_failure=False,
    )

    assert seen == ["Hello, Ada!"]
    assert results[0].job_results is not None
    assert results[0].job_results[0].output == "Hello, Ada!"
    assert results[0].job_results[0].error is None


@pytest.mark.asyncio
async def test_evaluatorq_experiment_requires_api_key(monkeypatch):
    monkeypatch.delenv("ORQ_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ORQ_API_KEY"):
        await evaluatorq(
            "exp-no-key",
            data=ExperimentInput(experiment_id=SHEET),
            evaluators=[],
            inference=False,
            print_results=False,
            _send_results=False,
            _exit_on_failure=False,
        )

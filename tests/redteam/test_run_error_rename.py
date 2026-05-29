"""RunError rename (was ErrorInfo) — RES-877."""

from __future__ import annotations


def test_run_error_importable_from_public_api():
    from evaluatorq.redteam import RunError

    err = RunError(message="boom", error_type="target_error")
    assert err.message == "boom"
    assert err.error_type == "target_error"


def test_error_info_name_is_gone():
    import evaluatorq.redteam as rt

    assert not hasattr(rt, "ErrorInfo")


def test_orchestrator_result_error_info_returns_run_error():
    from evaluatorq.redteam import RunError
    from evaluatorq.redteam.contracts import OrchestratorResult

    result = OrchestratorResult(error="boom", error_type="target_error")
    info = result.error_info
    assert isinstance(info, RunError)
    assert info.message == "boom"

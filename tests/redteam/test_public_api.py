# tests/redteam/test_public_api.py

import warnings


def test_llm_config_importable():
    from evaluatorq.redteam import LLMConfig
    assert LLMConfig is not None


def test_openai_model_target_importable():
    from evaluatorq.redteam import OpenAIModelTarget
    assert OpenAIModelTarget is not None


def _reset_deprecation_cache(name: str) -> None:
    """Clear the module-level once-per-name guard so warnings fire each test."""
    from evaluatorq import redteam as _rt
    _rt._deprecated_warned.discard(name)


def test_redteamconfig_deprecation_shim():
    from evaluatorq import redteam as _rt
    from evaluatorq.redteam import LLMConfig

    _reset_deprecation_cache("RedTeamConfig")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        alias = _rt.RedTeamConfig

    assert alias is LLMConfig
    deprecation_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert deprecation_warnings, "Expected a DeprecationWarning for RedTeamConfig"
    assert "LLMConfig" in str(deprecation_warnings[0].message)


def test_pipelinellmconfig_deprecation_shim():
    from evaluatorq import redteam as _rt
    from evaluatorq.redteam import LLMConfig

    _reset_deprecation_cache("PipelineLLMConfig")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        alias = _rt.PipelineLLMConfig

    assert alias is LLMConfig
    deprecation_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert deprecation_warnings, "Expected a DeprecationWarning for PipelineLLMConfig"
    assert "LLMConfig" in str(deprecation_warnings[0].message)

"""Every surface falls back to the same model and the same token budget.

Each surface stays individually overridable (`--sim-model`, `--attack-model`,
`judges=`/`model=`, `EVALUATORQ_APPLY_MODEL`, `EVALUATORQ_LLM_MAX_TOKENS`,
`LLMConfig.max_tokens`); only the fallback is shared.
"""

import pytest

from evaluatorq.contracts import DEFAULT_PIPELINE_MODEL, DEFAULT_TARGET_MAX_TOKENS
from evaluatorq.llm_jury import DEFAULT_JUDGE_MODEL
from evaluatorq.simulation.agents.base import DEFAULT_MAX_TOKENS
from evaluatorq.simulation.types import DEFAULT_MODEL


def test_shared_default_is_provider_prefixed():
    assert DEFAULT_PIPELINE_MODEL == 'openai/gpt-5.6-luna'


def test_every_surface_default_model_is_the_shared_one():
    assert DEFAULT_MODEL == DEFAULT_PIPELINE_MODEL
    assert DEFAULT_JUDGE_MODEL == DEFAULT_PIPELINE_MODEL


def test_dashboard_apply_default_model_is_the_shared_one():
    # Imported inside the test, not at module scope: apply_ui pulls in the
    # dashboard extra, and this invariant must not go unchecked in a core run.
    apply_ui = pytest.importorskip('evaluatorq.dashboard.apply_ui')

    assert apply_ui.DEFAULT_APPLY_MODEL == DEFAULT_PIPELINE_MODEL


def test_simulation_token_budget_is_the_shared_one():
    # Env-derived, so this also pins that an unset EVALUATORQ_LLM_MAX_TOKENS
    # falls through to the shared budget rather than a literal of its own.
    assert DEFAULT_MAX_TOKENS == DEFAULT_TARGET_MAX_TOKENS

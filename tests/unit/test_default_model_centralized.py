"""Every surface falls back to the one default model.

Each surface stays individually overridable (`--sim-model`, `--attack-model`,
`judges=`/`model=`, `EVALUATORQ_APPLY_MODEL`); only the fallback is shared.
"""

from evaluatorq.contracts import DEFAULT_PIPELINE_MODEL
from evaluatorq.llm_jury import DEFAULT_JUDGE_MODEL
from evaluatorq.simulation.types import DEFAULT_MODEL


def test_shared_default_is_provider_prefixed():
    assert DEFAULT_PIPELINE_MODEL == 'openai/gpt-5.6-luna'


def test_every_surface_default_is_the_shared_one():
    assert DEFAULT_MODEL == DEFAULT_PIPELINE_MODEL
    assert DEFAULT_JUDGE_MODEL == DEFAULT_PIPELINE_MODEL

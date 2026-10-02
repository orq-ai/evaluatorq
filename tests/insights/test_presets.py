"""Unit tests for `evaluatorq.insights.presets`."""

from __future__ import annotations

import subprocess
import sys
from typing import cast

import pytest

from evaluatorq.common.judge import ClassifyQuestion
from evaluatorq.insights import presets
from evaluatorq.insights.models import LabelSpec


def test_every_preset_is_a_valid_classify_question() -> None:
    for name, spec in presets.LABEL_PRESETS.items():
        assert spec.name == name
        assert isinstance(spec.to_question({'messages': []}), ClassifyQuestion)


def test_concerning_preset_uses_classifier_level_indices() -> None:
    criteria = presets.CONCERNING.criteria
    assert isinstance(criteria, list)
    assert [level.split(' ', 1)[0] for level in criteria] == [str(index) for index in range(len(criteria))]


def test_registry_is_frozen() -> None:
    with pytest.raises(TypeError):
        cast('dict[str, LabelSpec]', presets.LABEL_PRESETS)['x'] = presets.SENTIMENT


def test_taxonomies_have_expected_sizes() -> None:
    intent_criteria = presets.INTENT_TAXONOMY.criteria
    failure_criteria = presets.FAILURE_TAXONOMY.criteria
    assert intent_criteria is not None
    assert failure_criteria is not None
    assert len(intent_criteria) == 10
    assert len(failure_criteria) == 14


def test_label_name_rejects_reserved_match_key() -> None:
    with pytest.raises(ValueError, match='__match__'):
        LabelSpec(name='__match__', kind='noul', instructions='x')


def test_dimension_fields_cover_every_dimension_name() -> None:
    from evaluatorq.insights.models import DimensionName
    from typing import get_args

    assert set(get_args(DimensionName)) == set(presets.DIMENSION_FIELDS)


def test_importing_insights_does_not_import_numpy() -> None:
    # Runs in a fresh subprocess: other test modules in this same session legitimately
    # import numpy (e.g. to build synthetic clustering fixtures), which would otherwise
    # poison sys.modules and make this assertion depend on collection order rather than
    # on `evaluatorq.insights` actually deferring the import.
    result = subprocess.run(
        [sys.executable, '-c', "import evaluatorq.insights, sys; assert 'numpy' not in sys.modules"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

"""Unit tests for `evaluatorq.insights.presets`."""

from __future__ import annotations

import sys

import pytest

from evaluatorq.common.judge import ClassifyQuestion
from evaluatorq.insights import presets
from evaluatorq.insights.models import LabelSpec


def test_every_preset_is_a_valid_classify_question() -> None:
    for name, spec in presets.LABEL_PRESETS.items():
        assert spec.name == name
        assert isinstance(spec.to_question({'messages': []}), ClassifyQuestion)


def test_registry_is_frozen() -> None:
    with pytest.raises(TypeError):
        presets.LABEL_PRESETS['x'] = presets.SENTIMENT  # type: ignore[index]  # pyright: ignore[reportIndexIssue]


def test_taxonomies_have_expected_sizes() -> None:
    assert len(presets.INTENT_TAXONOMY.criteria) == 10  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    assert len(presets.FAILURE_TAXONOMY.criteria) == 14  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]


def test_label_name_rejects_reserved_match_key() -> None:
    with pytest.raises(ValueError, match='__match__'):
        LabelSpec(name='__match__', kind='noul', instructions='x')


def test_score_to_unit_maps_levels() -> None:
    assert presets.score_to_unit(presets.CUSTOMER_SATISFACTION, 0.0) == 0.0
    assert presets.score_to_unit(presets.CUSTOMER_SATISFACTION, 4.0) == 1.0


def test_dimension_fields_cover_every_dimension_name() -> None:
    from evaluatorq.insights.models import DimensionName
    from typing import get_args

    assert set(get_args(DimensionName)) == set(presets.DIMENSION_FIELDS)


def test_importing_insights_does_not_import_numpy() -> None:
    assert 'numpy' not in sys.modules

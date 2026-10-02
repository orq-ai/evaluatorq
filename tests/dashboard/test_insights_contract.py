from __future__ import annotations

import pytest
from pydantic import ValidationError

from evaluatorq.dashboard.insights_launch import InsightsLaunchSpec
from evaluatorq.insights.presets import CODING_LABELS, LABEL_PRESETS


def test_launch_resolves_selected_general_coding_and_custom_labels() -> None:
    custom = {'name': 'would_recommend', 'kind': 'noul', 'instructions': 'Would the user recommend it?'}
    spec = InsightsLaunchSpec(
        labels=['customer_satisfaction'],
        custom_labels=[custom],
        coding_labels=['task_type', 'verified'],
        dimensions=['intent'],
    )
    assert spec.label_specs() == [LABEL_PRESETS['customer_satisfaction'], spec.custom_labels[0]]
    assert [label.name for label in spec.coding_label_specs()] == ['task_type', 'verified']
    assert spec.coding_enabled is True


def test_custom_choice_and_score_specs_validate_complete_answer_spaces() -> None:
    choice = {'name': 'topic', 'kind': 'choice', 'instructions': 'Choose.', 'criteria': {'a': 'A', 'b': 'B'}}
    score = {'name': 'quality', 'kind': 'score', 'instructions': 'Rate.', 'criteria': ['1 low', '2', '3', '4', '5 high']}
    spec = InsightsLaunchSpec(custom_labels=[choice, score], dimensions=['intent'])
    assert spec.label_specs()[0].criteria == choice['criteria']
    assert len(spec.label_specs()[1].criteria) == 5


@pytest.mark.parametrize('instructions', ['', 'x' * 2001])
def test_launch_rejects_custom_label_instruction_lengths(instructions: str) -> None:
    with pytest.raises(ValidationError, match='between 1 and 2000 characters'):
        InsightsLaunchSpec(
            custom_labels=[{'name': 'custom_question', 'kind': 'noul', 'instructions': instructions}],
            dimensions=['intent'],
        )


@pytest.mark.parametrize('instructions', ['x', 'x' * 2000])
def test_launch_accepts_custom_label_instruction_length_boundaries(instructions: str) -> None:
    spec = InsightsLaunchSpec(
        custom_labels=[{'name': 'custom_question', 'kind': 'noul', 'instructions': instructions}],
        dimensions=['intent'],
    )
    assert spec.custom_labels[0].instructions == instructions


@pytest.mark.parametrize('labels', [
    [{'name': 'sentiment', 'kind': 'noul', 'instructions': 'Dup', 'criteria': {'true': 'a', 'false': 'b'}}],
    [{'name': '__match__', 'kind': 'noul', 'instructions': 'Reserved'}],
    [{'name': 'bad name', 'kind': 'noul', 'instructions': 'Invalid'}],
    [{'name': 'missing_criteria', 'kind': 'choice', 'instructions': 'Pick'}],
    [{'name': 'bad_score', 'kind': 'score', 'instructions': 'Rate', 'criteria': ['only one']}],
])
def test_launch_rejects_invalid_custom_labels(labels: list[dict[str, object]]) -> None:
    preset = ['sentiment'] if labels[0].get('name') == 'sentiment' else []
    with pytest.raises(ValidationError):
        InsightsLaunchSpec(labels=preset, custom_labels=labels, dimensions=['intent'])


def test_legacy_coding_analysis_still_requests_full_coding_bundle() -> None:
    spec = InsightsLaunchSpec(coding_analysis=True, dimensions=['intent'])
    assert spec.coding_enabled is True
    assert [label.name for label in spec.coding_label_specs()] == [label.name for label in CODING_LABELS[1:]]

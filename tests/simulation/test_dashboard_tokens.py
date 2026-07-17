"""Token terminology and optional-detail coverage for the simulation dashboard."""

from __future__ import annotations


def test_tokens_tab_uses_canonical_names_and_optional_detail_metrics() -> None:
    from evaluatorq.simulation.ui.token_display import token_metric_specs

    assert token_metric_specs(
        {
            'input_tokens': 10,
            'output_tokens': 5,
            'total_tokens': 15,
            'cached_tokens': 3,
            'reasoning_tokens': 2,
        }
    ) == [
        ('Input', '10'),
        ('Output', '5'),
        ('Total', '15'),
        ('Cached (retrieved)', '3'),
        ('Reasoning', '2'),
    ]


def test_overview_caption_uses_canonical_names_and_optional_details() -> None:
    from evaluatorq.simulation.ui.token_display import token_overview_caption

    assert token_overview_caption(
        {
            'input_tokens': 10,
            'output_tokens': 5,
            'cached_tokens': 3,
            'reasoning_tokens': 2,
            'avg_total_per_conversation': 15,
        }
    ) == 'Input 10 · Output 5 · Cached (retrieved) 3 · Reasoning 2 · Avg 15/conv'


def test_token_presentation_falls_back_to_legacy_saved_run_keys() -> None:
    from evaluatorq.simulation.ui.token_display import token_metric_specs

    assert token_metric_specs({'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}) == [
        ('Input', '10'),
        ('Output', '5'),
        ('Total', '15'),
    ]

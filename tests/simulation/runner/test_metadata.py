from evaluatorq.simulation.runner.simulation import _build_simulation_metadata
from evaluatorq.simulation.types import (
    CommunicationStyle,
    Criterion,
    Persona,
    Scenario,
)


def test_metadata_persists_persona_traits_and_scenario_goal():
    persona = Persona(
        name='Frustrated Customer', patience=0.2, assertiveness=0.8,
        politeness=0.4, technical_level=0.3,
        communication_style=CommunicationStyle.casual, background='Annoyed.',
    )
    scenario = Scenario(
        name='Billing question', goal='Find out why the invoice is higher',
        context='Unexpected charge.',
        criteria=[Criterion(description='explains charge', type='must_happen')],
    )
    meta = _build_simulation_metadata(persona, scenario, criteria_meta=None, target_model='m')
    assert meta['persona'] == 'Frustrated Customer'
    assert meta['scenario'] == 'Billing question'
    assert meta['persona_traits']['patience'] == 0.2
    assert meta['persona_traits']['communication_style'] == 'casual'  # enum -> value
    assert meta['persona_traits']['background'] == 'Annoyed.'
    assert meta['scenario_goal'] == 'Find out why the invoice is higher'
    assert meta['scenario_context'] == 'Unexpected charge.'
    assert meta['target_model'] == 'm'


def test_metadata_handles_missing_persona_and_scenario():
    meta = _build_simulation_metadata(None, None, criteria_meta=None, target_model=None)
    assert meta['persona'] is None
    assert meta['scenario'] is None
    assert 'persona_traits' not in meta
    assert 'scenario_goal' not in meta
    assert 'target_model' not in meta


def test_metadata_partial_persona_only_and_scenario_only():
    from evaluatorq.simulation.types import CommunicationStyle, Persona, Scenario

    persona = Persona(
        name='P', patience=0.5, assertiveness=0.5, politeness=0.5,
        technical_level=0.5, communication_style=CommunicationStyle.casual,
        background='bg',
    )
    # persona present, scenario absent
    m1 = _build_simulation_metadata(persona, None, criteria_meta=None, target_model=None)
    assert m1['persona'] == 'P' and 'persona_traits' in m1
    assert m1['scenario'] is None and 'scenario_goal' not in m1

    # scenario present (with context=None), persona absent
    scenario = Scenario(name='S', goal='g', context=None)
    m2 = _build_simulation_metadata(None, scenario, criteria_meta=None, target_model=None)
    assert m2['scenario'] == 'S' and m2['scenario_goal'] == 'g'
    assert m2['scenario_context'] is None  # key present, value None (not omitted)
    assert 'persona_traits' not in m2

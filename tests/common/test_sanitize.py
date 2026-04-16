"""Tests for common sanitize utilities."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from evaluatorq.common.sanitize import delimit, xml_escape


# ---------------------------------------------------------------------------
# delimit — default tag
# ---------------------------------------------------------------------------


def test_delimit_basic():
    assert delimit("hello") == "<data>hello</data>"


def test_delimit_escapes_ampersand():
    result = delimit("a & b")
    assert "&amp;" in result
    assert "<data>a &amp; b</data>" == result


def test_delimit_escapes_data_tags():
    result = delimit("test <data>injection</data> here")
    assert "<data>" not in result.replace("<data>", "", 1).replace("</data>", "", 1)
    assert "&lt;data&gt;" in result
    assert "&lt;/data&gt;" in result


def test_delimit_case_insensitive():
    result = delimit("<DATA>test</DATA>")
    assert "&lt;data&gt;" in result
    assert "&lt;/data&gt;" in result


def test_delimit_empty_string():
    assert delimit("") == "<data></data>"


def test_preserves_non_data_tags():
    result = delimit("<b>bold</b>")
    assert "<b>bold</b>" in result


# ---------------------------------------------------------------------------
# delimit — custom tag
# ---------------------------------------------------------------------------


def test_delimit_custom_tag():
    result = delimit("hello", tag="target_response")
    assert result == "<target_response>hello</target_response>"


def test_delimit_custom_tag_escapes_matching_tags():
    result = delimit("a <target_response>injection</target_response> b", tag="target_response")
    # The outer boundary tags should be the only unescaped ones
    inner = result.removeprefix("<target_response>").removesuffix("</target_response>")
    assert "<target_response>" not in inner
    assert "</target_response>" not in inner
    assert "&lt;target_response&gt;" in inner
    assert "&lt;/target_response&gt;" in inner


def test_delimit_custom_tag_case_insensitive():
    result = delimit("<TARGET_RESPONSE>test</TARGET_RESPONSE>", tag="target_response")
    inner = result.removeprefix("<target_response>").removesuffix("</target_response>")
    assert "<target_response>" not in inner.lower()


def test_delimit_custom_tag_does_not_escape_other_tags():
    result = delimit("<data>keep</data>", tag="target_response")
    assert "<data>keep</data>" in result


# ---------------------------------------------------------------------------
# delimit — tag validation
# ---------------------------------------------------------------------------


def test_delimit_rejects_empty_tag():
    with pytest.raises(ValueError, match="Invalid tag name"):
        delimit("hello", tag="")


def test_delimit_rejects_tag_starting_with_digit():
    with pytest.raises(ValueError, match="Invalid tag name"):
        delimit("hello", tag="1tag")


def test_delimit_rejects_tag_with_angle_brackets():
    with pytest.raises(ValueError, match="Invalid tag name"):
        delimit("hello", tag="tag<>")


def test_delimit_rejects_tag_with_spaces():
    with pytest.raises(ValueError, match="Invalid tag name"):
        delimit("hello", tag="my tag")


def test_delimit_rejects_tag_with_slash():
    with pytest.raises(ValueError, match="Invalid tag name"):
        delimit("hello", tag="foo/bar")


# ---------------------------------------------------------------------------
# delimit — escape ordering (& first, then tags)
# ---------------------------------------------------------------------------


def test_delimit_escape_ordering_prevents_bypass():
    """Ampersand must be escaped BEFORE tags — otherwise pre-escaped tag
    sequences like ``&lt;data&gt;`` survive as a breakout vector."""
    result = delimit("&lt;data&gt;")
    # & → &amp; first, so the pre-escaped sequence becomes &amp;lt;data&amp;gt;
    assert "<data>&amp;lt;data&amp;gt;</data>" == result


def test_delimit_escape_ordering_closing_tag():
    result = delimit("&lt;/data&gt;")
    assert "<data>&amp;lt;/data&amp;gt;</data>" == result


# ---------------------------------------------------------------------------
# xml_escape
# ---------------------------------------------------------------------------


def test_xml_escape_basic():
    assert xml_escape("a & b") == "a &amp; b"


def test_xml_escape_angle_brackets():
    assert xml_escape("<script>alert(1)</script>") == "&lt;script&gt;alert(1)&lt;/script&gt;"


def test_xml_escape_empty():
    assert xml_escape("") == ""


def test_xml_escape_no_special_chars():
    assert xml_escape("hello world") == "hello world"


def test_xml_escape_all_special():
    result = xml_escape('a & b < c > d')
    assert result == "a &amp; b &lt; c &gt; d"


# ---------------------------------------------------------------------------
# JSONL bypass — datapoints always get rebuilt system prompts
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_datapoint_system_prompt_always_rebuilt():
    """Verify that SimulationRunner.run() rebuilds the system prompt from
    persona + scenario even when the datapoint carries a cached
    user_system_prompt value (JSONL bypass fix)."""
    from evaluatorq.simulation.types import (
        CommunicationStyle,
        Datapoint,
        Persona,
        Scenario,
    )
    from evaluatorq.simulation.utils.prompt_builders import (
        build_datapoint_system_prompt,
    )

    persona = Persona(
        name="Test User",
        patience=0.5,
        assertiveness=0.5,
        politeness=0.5,
        technical_level=0.5,
        communication_style=CommunicationStyle.casual,
        background="A test persona.",
    )
    scenario = Scenario(
        name="Test Scenario",
        goal="Achieve a test goal",
        context="Some context",
    )

    stale_prompt = "stale cached prompt — should NOT be used"
    dp = Datapoint(
        id="dp_test",
        persona=persona,
        scenario=scenario,
        user_system_prompt=stale_prompt,
        first_message="Hi",
    )

    expected_prompt = build_datapoint_system_prompt(persona, scenario)
    assert stale_prompt != expected_prompt  # sanity

    # Patch just enough to avoid real LLM calls
    with (
        patch(
            "evaluatorq.simulation.runner.simulation.SimulationRunner._get_shared_client"
        ) as mock_client,
        patch(
            "evaluatorq.simulation.agents.user_simulator.UserSimulatorAgent.generate_first_message",
            new_callable=AsyncMock,
            return_value="Hi there",
        ),
    ):
        mock_client.return_value = MagicMock()

        from evaluatorq.simulation.runner.simulation import SimulationRunner

        runner = SimulationRunner(
            target_callback=lambda msgs: "agent reply",
            model="test-model",
            max_turns=1,
        )

        # We only need to verify what system_prompt the UserSimulatorAgent
        # receives.  Patch its __init__ to capture the config.
        captured_configs: list[object] = []
        original_init = (
            __import__(
                "evaluatorq.simulation.agents.user_simulator",
                fromlist=["UserSimulatorAgent"],
            ).UserSimulatorAgent.__init__
        )

        def spy_init(self, config):
            captured_configs.append(config)
            return original_init(self, config)

        with patch(
            "evaluatorq.simulation.runner.simulation.UserSimulatorAgent.__init__",
            spy_init,
        ):
            # We don't care about the full run completing — an error after
            # UserSimulatorAgent is constructed is fine.
            await runner.run(datapoint=dp)

        assert len(captured_configs) == 1
        assert captured_configs[0].system_prompt == expected_prompt  # pyright: ignore[reportAttributeAccessIssue]

"""Tests for OWASP_LLM_TOP_10 and OWASP_ASI_TOP_10 public constants."""

from evaluatorq.redteam import OWASP_ASI_TOP_10, OWASP_LLM_TOP_10
from evaluatorq.redteam.contracts import OWASP_CATEGORY_NAMES


class TestOwaspLlmTop10:
    def test_is_list(self) -> None:
        assert isinstance(OWASP_LLM_TOP_10, list)

    def test_length(self) -> None:
        # LLM10 (Unbounded Consumption) is intentionally excluded — not testable via prompts
        assert len(OWASP_LLM_TOP_10) == 9  # LLM01-LLM09

    def test_all_codes_present(self) -> None:
        expected = {f"LLM{i:02d}" for i in range(1, 10)}
        assert set(OWASP_LLM_TOP_10) == expected

    def test_ordered(self) -> None:
        assert OWASP_LLM_TOP_10 == sorted(OWASP_LLM_TOP_10)

    def test_all_codes_in_category_names(self) -> None:
        for code in OWASP_LLM_TOP_10:
            assert code in OWASP_CATEGORY_NAMES, f"{code} missing from OWASP_CATEGORY_NAMES"

    def test_no_owasp_prefix(self) -> None:
        for code in OWASP_LLM_TOP_10:
            assert not code.startswith("OWASP-"), f"{code} should not have OWASP- prefix"

    def test_importable_from_redteam_package(self) -> None:
        from evaluatorq.redteam import OWASP_LLM_TOP_10 as IMPORTED_LLM
        assert IMPORTED_LLM is OWASP_LLM_TOP_10


class TestOwaspAsiTop10:
    def test_is_list(self) -> None:
        assert isinstance(OWASP_ASI_TOP_10, list)

    def test_length(self) -> None:
        assert len(OWASP_ASI_TOP_10) == 10  # ASI01-ASI10

    def test_all_codes_present(self) -> None:
        expected = {f"ASI{i:02d}" for i in range(1, 11)}
        assert set(OWASP_ASI_TOP_10) == expected

    def test_ordered(self) -> None:
        assert OWASP_ASI_TOP_10 == sorted(OWASP_ASI_TOP_10)

    def test_all_codes_in_category_names(self) -> None:
        for code in OWASP_ASI_TOP_10:
            assert code in OWASP_CATEGORY_NAMES, f"{code} missing from OWASP_CATEGORY_NAMES"

    def test_no_owasp_prefix(self) -> None:
        for code in OWASP_ASI_TOP_10:
            assert not code.startswith("OWASP-"), f"{code} should not have OWASP- prefix"

    def test_importable_from_redteam_package(self) -> None:
        from evaluatorq.redteam import OWASP_ASI_TOP_10 as IMPORTED_ASI
        assert IMPORTED_ASI is OWASP_ASI_TOP_10


class TestDisjoint:
    def test_llm_and_asi_are_disjoint(self) -> None:
        assert set(OWASP_LLM_TOP_10).isdisjoint(set(OWASP_ASI_TOP_10))

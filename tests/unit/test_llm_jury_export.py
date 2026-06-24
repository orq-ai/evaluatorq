def test_llm_jury_is_public():
    import evaluatorq
    assert hasattr(evaluatorq, "llm_jury")
    assert "llm_jury" in evaluatorq.__all__

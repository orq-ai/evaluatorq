"""`evaluatorq.__version__` is exported and is a string."""

import evaluatorq


def test_version_is_exported_string() -> None:
    assert isinstance(evaluatorq.__version__, str)
    assert evaluatorq.__version__
    assert '__version__' in evaluatorq.__all__

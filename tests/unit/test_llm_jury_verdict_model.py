from __future__ import annotations

import typing

from evaluatorq.llm_jury import _build_verdict_model


def test_boolean_model():
    m = _build_verdict_model("categorical", None, (0.0, 1.0))
    assert m.model_fields["value"].annotation is bool
    assert m.model_fields["explanation"].annotation is str


def test_labels_model_is_literal():
    m = _build_verdict_model("categorical", ["good", "bad"], (0.0, 1.0))
    ann = m.model_fields["value"].annotation
    assert typing.get_origin(ann) is typing.Literal
    assert set(typing.get_args(ann)) == {"good", "bad"}


def test_numeric_model_is_float():
    m = _build_verdict_model("numeric", None, (0.0, 1.0))
    assert m.model_fields["value"].annotation is float


def test_models_validate_payloads():
    m = _build_verdict_model("categorical", ["good", "bad"], (0.0, 1.0))
    assert m(value="good", explanation="x").value == "good"

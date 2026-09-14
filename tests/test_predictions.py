from __future__ import annotations

import json
from pathlib import Path

import pytest

from sara_audio.predictions import JsonLinearModel


def test_json_linear_model_matches_mic_score_and_tracks_missing_values(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model.json"
    model_path.write_text(
        json.dumps({"Intercept": 0.2, "feature_a": 0.3, "feature_b": -0.1}),
        encoding="utf-8",
    )

    result = JsonLinearModel.load(model_path).predict(
        {"feature_a": 1, "feature_b": None}, threshold=0.5
    )

    assert result.score == 0.5
    assert result.remove is True
    assert result.keep is False
    assert result.missing_features == ()
    assert result.unavailable_features == ("feature_b",)


def test_json_linear_model_reports_missing_features(tmp_path: Path) -> None:
    model_path = tmp_path / "model.json"
    model_path.write_text(json.dumps({"Intercept": 0.1, "feature_a": 0.3}), encoding="utf-8")

    result = JsonLinearModel.load(model_path).predict({}, threshold=0.5)

    assert result.score == 0.1
    assert result.keep is True
    assert result.missing_features == ("feature_a",)


def test_json_linear_model_requires_intercept(tmp_path: Path) -> None:
    model_path = tmp_path / "model.json"
    model_path.write_text(json.dumps({"feature_a": 0.3}), encoding="utf-8")

    with pytest.raises(ValueError, match="Intercept"):
        JsonLinearModel.load(model_path)

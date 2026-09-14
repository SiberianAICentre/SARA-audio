from __future__ import annotations

import json
from pathlib import Path

from sara_audio.domain import ExtractionResult, FeatureValue
from sara_audio.pipeline import PipelineRun
from sara_audio.predictions import (
    EXPERT_MODEL_NAMES,
    JsonLinearModel,
    MethodologyModelBundle,
    agent_year_id,
    run_methodology_prediction_rows,
)


def _model(tmp_path: Path, name: str, coefficients: dict[str, float]) -> JsonLinearModel:
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(coefficients), encoding="utf-8")
    return JsonLinearModel.load(path)


def _run(path: Path, value: float) -> PipelineRun:
    feature = FeatureValue("x", value, "", "test", 1.0, "test")
    return PipelineRun(
        result=ExtractionResult(path, (feature,)),
        frames=None,
        opensmile_coverage=None,
        normalized_audio_path=path,
        analysis_audio_path=path,
    )


def test_methodology_cascade_filters_then_predicts_group_burnout(tmp_path: Path) -> None:
    expert_models = tuple(
        (
            name,
            _model(
                tmp_path,
                name,
                {"Intercept": 1.0 if name == "participation_modal_n" else 0.0},
            ),
        )
        for name in EXPERT_MODEL_NAMES
    )
    models = MethodologyModelBundle(
        filter_model=_model(tmp_path, "filter", {"Intercept": 0.0, "x": 1.0}),
        expert_models=expert_models,
        agent_sex_model=_model(
            tmp_path,
            "sex",
            {"Intercept": 0.0, "participation_modal_n": 1.0},
        ),
        burnout_model=_model(tmp_path, "burnout", {"Intercept": 0.0, "m": 1.0}),
    )
    runs = [
        _run(tmp_path / "20241118162438_Agent_15.wav", 0.0),
        _run(tmp_path / "20241218162438_Agent_15.wav", 1.0),
    ]

    rows = run_methodology_prediction_rows(runs, models, threshold=0.5)

    assert len(rows.filtering) == 2
    assert [row["filter_keep"] for row in rows.filtering] == [1, 0]
    assert len(rows.final) == 1
    assert rows.final[0]["agent_year_id"] == "15_2024"
    assert rows.final[0]["agent_sex"] == 1
    assert rows.final[0]["burnout"] == 1.0
    assert rows.final[0]["burnout_final"] == 1
    assert rows.final[0]["group_input_count"] == 2
    assert rows.final[0]["group_informative_count"] == 1
    assert rows.interpretation[0]["m"] == 1
    assert rows.interpretation[0]["f"] == 0


def test_agent_year_id_falls_back_for_unstructured_names() -> None:
    assert agent_year_id("20230904103009_Оператор12_muted.wav") == "12_2023"
    assert agent_year_id("recording.wav") == "uploaded_batch"

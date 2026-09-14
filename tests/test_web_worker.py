from sara_audio.predictions import JsonLinearModel
from sara_audio.web_worker import _enable_stage_tracing


def test_stage_tracing_preserves_json_model_classmethod(tmp_path) -> None:
    model_path = tmp_path / "model.json"
    model_path.write_text('{"Intercept": 0.0}', encoding="utf-8")

    _enable_stage_tracing()

    assert JsonLinearModel.load(model_path).coefficients == {"Intercept": 0.0}

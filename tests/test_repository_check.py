from pathlib import Path

from sara_audio.repository_check import MODEL_NAMES, verify_repository


def test_repository_contains_reproducible_runtime_bundle() -> None:
    root = Path(__file__).resolve().parents[1]

    result = verify_repository(root)

    assert result["status"] == "passed"
    assert len(result["methodology_models"]) == len(MODEL_NAMES) == 11
    assert all(item["sha256"] for item in result["methodology_models"])

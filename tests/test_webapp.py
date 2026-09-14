import csv
import zipfile
from pathlib import Path

from sara_audio.webapp import PREVIEW_COLUMNS, SaraWebService


def test_web_preview_reads_final_prediction_table(tmp_path: Path) -> None:
    wide = tmp_path / "predictions.csv"
    with wide.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(PREVIEW_COLUMNS), delimiter=";")
        writer.writeheader()
        writer.writerow(
            {
                "file_id": "record-1",
                "file_name": "20241118162438_Agent_15.wav",
                "agent_year_id": "15_2024",
                "burnout": "0.72",
                "burnout_final": "1",
                "burnout_label": "Да",
                "agent_sex_score": "0.81",
                "agent_sex_label": "M",
                "filter_score": "0.12",
                "group_input_count": "2",
                "group_informative_count": "1",
            }
        )

    assert SaraWebService._read_preview(wide) == [
        [
            "record-1",
            "20241118162438_Agent_15.wav",
            "15_2024",
            "0.72",
            "1",
            "Да",
            "0.81",
            "M",
            "0.12",
            "2",
            "1",
        ]
    ]


def test_web_zip_contains_job_artifacts(tmp_path: Path) -> None:
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "features_wide.csv").write_text("file_id\na\n", encoding="utf-8")
    archive = SaraWebService._zip_job(tmp_path, tmp_path / "bundle.zip")

    with zipfile.ZipFile(archive) as bundle:
        assert bundle.namelist() == ["results/features_wide.csv"]

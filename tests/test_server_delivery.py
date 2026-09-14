from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path
from xml.etree import ElementTree

from sara_audio.server_delivery import ServerDeliveryBuilder


def test_packages_completed_server_result_without_reprocessing(tmp_path: Path) -> None:
    source = tmp_path / "server-run"
    record = source / "records" / "call-1"
    record.mkdir(parents=True)
    (source / "features_wide.csv").write_text(
        "file_id;input_path;analysis_audio_path;LEX_EmotLex;ASR_token_count;TIME_audio_duration_s\n"
        "call-1;input.wav;speech.wav;0.5;10;12\n",
        encoding="utf-8",
    )
    (source / "run_summary.json").write_text(
        json.dumps(
            {
                "record_count": 1,
                "records": [
                    {
                        "input_path": "input.wav",
                        "directory": str(record),
                        "operator_speech": "speech.wav",
                        "transcripts": [str(record / "transcript.txt")],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (source / "predictions.csv").write_text(
        "file_id;filter_score;filter_remove\ncall-1;0.6;1\n",
        encoding="utf-8",
    )
    (source / "interpretation_input.csv").write_text(
        "file_id;filter_score;ACOUSTIC_x\ncall-1;0.6;1.2\n",
        encoding="utf-8",
    )
    (source / "filter_predictions.csv").write_text(
        "file_id;filter_score;filter_remove\ncall-1;0.6;1\n",
        encoding="utf-8",
    )
    (source / "input_manifest.json").write_text("{}", encoding="utf-8")
    (record / "transcript.txt").write_text("Тестовая, реплика", encoding="utf-8")
    (record / "report.json").write_text(
        json.dumps(
            {"diagnostics": ["ok"], "acoustic_features": {"semantic_coverage": {}}}
        ),
        encoding="utf-8",
    )

    destination = tmp_path / "delivery"
    archive = ServerDeliveryBuilder(source, destination).build()

    assert archive.is_file()
    assert Path(f"{archive}.sha256").is_file()
    workbook = destination / "tables" / "delivery.xlsx"
    assert workbook.is_file()
    wide_csv = destination / "tables" / "features_wide.csv"
    assert wide_csv.is_file()
    assert (destination / "tables" / "transcripts.csv").is_file()
    assert (destination / "tables" / "records_index.csv").is_file()
    assert (destination / "tables" / "record_diagnostics.csv").is_file()
    assert (destination / "tables" / "feature_dictionary.csv").is_file()
    assert (destination / "tables" / "predictions.csv").is_file()
    assert (destination / "tables" / "interpretation_input.csv").is_file()
    assert (destination / "tables" / "filter_predictions.csv").is_file()
    assert ";" in wide_csv.read_text(encoding="utf-8-sig").splitlines()[0]
    with wide_csv.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter=";")
        rows = list(reader)
        assert rows[0]["ASR_token_count"] == "10"
        assert "TIME_audio_duration_s" not in (reader.fieldnames or [])
    assert (destination / "provenance" / "input_manifest.json").is_file()
    with zipfile.ZipFile(workbook) as excel:
        workbook_xml = excel.read("xl/workbook.xml").decode("utf-8")
        features_sheet = excel.read("xl/worksheets/sheet1.xml").decode("utf-8")
        transcripts_sheet = excel.read("xl/worksheets/sheet2.xml").decode("utf-8")
    ElementTree.fromstring(workbook_xml)
    ElementTree.fromstring(features_sheet)
    ElementTree.fromstring(transcripts_sheet)
    assert all(
        name in workbook_xml
        for name in (
            "Features",
            "Transcripts",
            "Records",
            "Diagnostics",
            "Dictionary",
            "Predictions",
            "Interpretation",
            "Filtering",
        )
    )
    assert "LEX_EmotLex" in features_sheet
    assert "TIME_audio_duration_s" not in features_sheet
    assert "Тестовая, реплика" in transcripts_sheet
    with zipfile.ZipFile(archive) as package:
        assert "delivery/tables/delivery.xlsx" in package.namelist()
        assert "delivery/tables/features_wide.csv" in package.namelist()


def test_delivery_reads_legacy_comma_delimited_wide_result(tmp_path: Path) -> None:
    wide = tmp_path / "features_wide.csv"
    wide.write_text(
        "file_id,input_path,analysis_audio_path,ASR_token_count\n"
        "call-1,input.wav,speech.wav,10\n",
        encoding="utf-8",
    )

    rows, columns = ServerDeliveryBuilder(tmp_path, tmp_path / "delivery")._read_wide(
        wide
    )

    assert columns == [
        "file_id",
        "input_path",
        "analysis_audio_path",
        "ASR_token_count",
    ]
    assert rows == [
        {
            "file_id": "call-1",
            "input_path": "input.wav",
            "analysis_audio_path": "speech.wav",
            "ASR_token_count": "10",
        }
    ]

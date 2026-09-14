import csv
from pathlib import Path

import pytest

from sara_audio.runner import BatchOutput, RecordOutput
from sara_audio.server_smoke import ServerSmokeVerifier


def test_server_smoke_verifies_model_ready_artifacts(tmp_path: Path) -> None:
    speech = tmp_path / "operator_speech_no_pauses.wav"
    speech.write_bytes(b"wav")
    wide = tmp_path / "features_wide.csv"
    with wide.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(
            destination,
            fieldnames=[
                "file_id",
                "LEX_EmotLex",
                "TIME_active_speech_duration_s",
                "ASR_token_count",
                "ACOUSTIC_energy_mean",
                "bert_1",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "file_id": "record",
                "LEX_EmotLex": 0,
                "TIME_active_speech_duration_s": 1,
                "ASR_token_count": 2,
                "ACOUSTIC_energy_mean": 0.1,
                "bert_1": 0.01,
            }
        )
    batch = tmp_path / "batch.json"
    batch.write_text('{"batch_id":"09-2023-001","record_count":1}', encoding="utf-8")
    record = RecordOutput(
        tmp_path / "input.wav",
        tmp_path,
        (wide,),
        (),
        None,
        (),
        tmp_path / "manifest.json",
        tmp_path / "report.json",
        speech,
    )
    result = BatchOutput(
        tmp_path,
        tmp_path / "summary.json",
        (record,),
        batch_manifest_paths=(batch,),
        wide_features_path=wide,
    )

    verified = ServerSmokeVerifier().verify_output(result)

    assert verified["status"] == "passed"


def test_server_smoke_rejects_missing_asr_tokens(tmp_path: Path) -> None:
    wide = tmp_path / "features_wide.csv"
    wide.write_text(
        "file_id,LEX_EmotLex,TIME_active_speech_duration_s,ASR_token_count,ACOUSTIC_energy_mean,bert_1\n"
        "a,0,1,,0.1,0.01\n",
        encoding="utf-8",
    )
    speech = tmp_path / "operator_speech_no_pauses.wav"
    speech.write_bytes(b"wav")
    record = RecordOutput(
        tmp_path / "input.wav",
        tmp_path,
        (wide,),
        (),
        None,
        (),
        tmp_path / "manifest.json",
        tmp_path / "report.json",
        speech,
    )
    result = BatchOutput(
        tmp_path, tmp_path / "summary.json", (record,), wide_features_path=wide
    )

    with pytest.raises(RuntimeError, match="ASR did not emit tokens"):
        ServerSmokeVerifier().verify_output(result)


def test_server_smoke_reads_semicolon_wide_matrix(tmp_path: Path) -> None:
    wide = tmp_path / "features_wide.csv"
    wide.write_text(
        "file_id;LEX_EmotLex;TIME_active_speech_duration_s;ASR_token_count;ACOUSTIC_energy_mean;bert_1\n"
        "record;0;1;2;0.1;0.01\n",
        encoding="utf-8",
    )
    speech = tmp_path / "operator_speech_no_pauses.wav"
    speech.write_bytes(b"wav")
    record = RecordOutput(
        tmp_path / "input.wav",
        tmp_path,
        (wide,),
        (),
        None,
        (),
        tmp_path / "manifest.json",
        tmp_path / "report.json",
        speech,
    )
    result = BatchOutput(
        tmp_path, tmp_path / "summary.json", (record,), wide_features_path=wide
    )

    assert ServerSmokeVerifier().verify_output(result)["status"] == "passed"

from __future__ import annotations

import csv
import json
import wave
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from sara_audio.external_delivery import (
    ExistingFeatureDelivery,
    ImportedTranscriptReader,
    _initialize_worker,
    _process_in_worker,
    safe_child,
)


def make_source(tmp_path):
    root = tmp_path / "source"
    processed = root / "Обработанные_записи (1)"
    (processed / "_Проверка").mkdir(parents=True)
    (processed / "all_files").mkdir()
    journal = processed / "_Проверка/Замены_тишины.tsv"
    with journal.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["Исходный путь", "Строка", "Итоговый текст"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerow(
            {
                "Исходный путь": "all_files/sample.txt",
                "Строка": "2",
                "Итоговый текст": "Тишина",
            }
        )
    text = "[00:01] Оператор: Я очень рад!\n[00:02] Оператор: Тишина\n[00:03] Клиент: Ужас, блин!\n[00:04] Оператор: Тишина мешает работе.\n"
    (processed / "all_files/sample.txt").write_text(text, encoding="utf-8")
    (root / "before.txt").write_text(text, encoding="utf-8")
    entry = {
        "processed": [{"path": "Обработанные_записи (1)/all_files/sample.txt"}],
        "before_roles": [{"path": "before.txt"}],
    }
    return root, entry


def test_import_removes_only_journal_markers_and_selects_operator(tmp_path):
    root, entry = make_source(tmp_path)
    transcript, turns = ImportedTranscriptReader(root, "Оператор").read(entry)
    assert transcript.text == "Я очень рад!\nТишина мешает работе."
    assert not transcript.is_verified
    assert transcript.tokens == ()
    assert len(turns) == 4
    assert turns[1].artifact_replacement
    assert turns[1].analysis_text == ""
    assert not turns[2].selected
    assert not turns[3].artifact_replacement


@pytest.mark.parametrize("with_audio", [False, True])
def test_record_preserves_92_codes_and_never_computes_unaligned_rate(
    tmp_path, with_audio
):
    root, entry = make_source(tmp_path)
    path = root / "sample.wav"
    if with_audio:
        with wave.open(str(path), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(16000)
            stream.writeframes(b"\x10\x27" * 16000)
    corpus = SimpleNamespace(
        root=root,
        audio=lambda entry, cache: (path, "sample.wav") if with_audio else (None, None),
    )
    config = {
        "analysis_role": "Оператор",
        "include_frames": False,
        "opensmile": {"enabled": False},
    }
    destination = tmp_path / "delivery"
    (destination / "records").mkdir(parents=True)
    runner = ExistingFeatureDelivery(corpus, config, destination)
    runner.work = tmp_path / "cache"
    runner.work.mkdir()
    record_id = "20230201144625_Agent_09"
    runner.process_record(record_id, entry)
    folder = destination / "records" / record_id
    rows = json.loads((folder / "features.json").read_text(encoding="utf-8"))
    assert len(rows) == 101
    assert all(row["file_id"] == record_id for row in rows)
    linguistic = [
        row for row in rows if not row["feature_code"].startswith(("TIME_", "RATE_"))
    ]
    assert len(linguistic) == len({row["feature_code"] for row in linguistic}) == 92
    assert sum(row["value"] is not None for row in linguistic) == 92
    assert all(
        row["value"] is None for row in rows if row["feature_code"].startswith("RATE_")
    )
    lowered = next(row for row in rows if row["feature_code"] == "EMO_LoweredLex")
    assert (
        lowered["value"] == 0
    )  # The client's exclamation must not affect operator counts.
    times = [row for row in rows if row["feature_code"].startswith("TIME_")]
    if with_audio:
        assert all(row["evidence"]["time_axis"] == "supplied_wav" for row in times)
    else:
        assert all(row["value"] is None for row in times)
    assert (folder / "dialogue_processed.txt").read_bytes() == (
        root / entry["processed"][0]["path"]
    ).read_bytes()


def test_path_escape_rejected(tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        safe_child(tmp_path, "../outside.wav")


def test_worker_initializes_unpicklable_rule_objects_inside_child(
    tmp_path, monkeypatch
):
    root, entry = make_source(tmp_path)
    monkeypatch.chdir(tmp_path)
    destination = tmp_path / "worker-delivery"
    (destination / "records").mkdir(parents=True)
    (Path("artifacts/external_delivery_cache") / destination.name).mkdir(parents=True)
    config = {
        "analysis_role": "Оператор",
        "include_frames": False,
        "opensmile": {"enabled": False},
    }
    inventory = {"records": {str(i): {} for i in range(390)}}
    record_id = "20230201144625_Agent_09"
    with ProcessPoolExecutor(
        max_workers=1,
        initializer=_initialize_worker,
        initargs=(root, inventory, config, destination),
    ) as executor:
        result = executor.submit(_process_in_worker, record_id, entry).result(
            timeout=30
        )
    assert result["record_id"] == record_id
    assert result["linguistic_computed"] == 92
    assert result["linguistic_not_computed"] == 0
    assert not result["audio_available"]


def test_no_operator_text_is_not_reported_as_zero_counts(tmp_path):
    root, entry = make_source(tmp_path)
    source = root / entry["processed"][0]["path"]
    source.write_text(
        source.read_text(encoding="utf-8").replace("Оператор:", "Клиент:"),
        encoding="utf-8",
    )
    corpus = SimpleNamespace(root=root, audio=lambda entry, cache: (None, None))
    destination = tmp_path / "empty-operator"
    (destination / "records").mkdir(parents=True)
    runner = ExistingFeatureDelivery(
        corpus, {"analysis_role": "Оператор", "include_frames": False}, destination
    )
    runner.work = tmp_path / "cache"
    runner.work.mkdir()
    record_id = "20241014162858_Agent_03"
    result = runner.process_record(record_id, entry)
    assert result["linguistic_computed"] == 0
    assert result["linguistic_not_computed"] == 92
    folder = destination / "records" / record_id
    rows = json.loads((folder / "features.json").read_text(encoding="utf-8"))
    assert len(rows) == 101
    assert all(row["value"] is None for row in rows)
    assert not (folder / "transcript.txt").read_text(encoding="utf-8").strip()
    assert (folder / "dialogue_processed.txt").read_bytes() == source.read_bytes()
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    assert not report["transcript"]["available"]
    assert report["analysis_text_status"] == "no_operator_text"


@pytest.mark.parametrize("finished", [False, True])
def test_recovery_only_retries_failed_records_of_finished_batch(tmp_path, finished):
    root, _ = make_source(tmp_path)
    calls = []
    corpus = SimpleNamespace(
        root=root, records={"good": {}, "bad": {}}, validate=lambda: None
    )
    destination = tmp_path / "recovery"
    (destination / "records").mkdir(parents=True)
    original = {"record_id": "good", "marker": "preserved"}
    summary = {
        "complete": False,
        "records": [original] if finished else [],
        "errors": [{"record_id": "bad", "error": "test failure"}],
    }
    (destination / "run_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    runner = ExistingFeatureDelivery(corpus, {"analysis_role": "Оператор"}, destination)
    runner.work = tmp_path / "cache"

    def process(record_id, entry):
        calls.append(record_id)
        return {"record_id": record_id}

    runner.process_record = process
    if finished:
        result = runner.retry_failed()
        assert calls == ["bad"]
        assert result["complete"]
        assert not result["errors"]
        assert original in result["records"]
    else:
        with pytest.raises(ValueError, match="still running"):
            runner.retry_failed()
        assert not calls

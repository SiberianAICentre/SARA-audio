from __future__ import annotations

from pathlib import Path

from sara_audio.batching import InputManifestBuilder


def test_operator_year_group_is_split_into_batches_of_fifteen(tmp_path: Path) -> None:
    paths = []
    for index in range(16):
        path = tmp_path / f"2023{index:02d}01090000_Agent_09_{index:02d}.wav"
        path.write_bytes(bytes([index]))
        paths.append(path)
    other = tmp_path / "20240101090000_Agent_09_00.wav"
    other.write_bytes(b"other")
    paths.append(other)

    builder = InputManifestBuilder(batch_size=15)
    records = builder.build_records(paths)
    batches = builder.build_batches(records)

    assert [(batch.operator_id, batch.year, len(batch.records)) for batch in batches] == [
        ("09", "2023", 15),
        ("09", "2023", 1),
        ("09", "2024", 1),
    ]
    assert all(len(record.sha256) == 64 for record in records)

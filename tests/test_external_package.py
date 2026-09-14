from __future__ import annotations

import csv
import json

import pytest

from sara_audio.external_package import ExternalPackageWriter, write_csv


def test_csv_preserves_unicode_zero_and_missing_values(tmp_path):
    path = tmp_path / "features.csv"
    write_csv(
        path,
        [
            {"code": "test", "value": 0, "description": 'Слова, "фразы"'},
            {"code": "missing", "value": None, "description": "Не рассчитано"},
        ],
    )
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["value"] == "0"
    assert rows[1]["value"] == ""
    assert rows[0]["description"] == 'Слова, "фразы"'


@pytest.mark.parametrize(
    "summary",
    [
        {"complete": False, "errors": []},
        {"complete": True, "errors": [{"record_id": "failed"}]},
    ],
)
def test_incomplete_delivery_cannot_be_packaged(tmp_path, summary):
    root = tmp_path / "delivery"
    root.mkdir()
    (root / "run_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    writer = ExternalPackageWriter(
        root, tmp_path / "config", tmp_path / "inventory", tmp_path
    )
    with pytest.raises(ValueError, match="incomplete delivery"):
        writer.build()
    assert not root.with_suffix(".zip").exists()
    assert not (root / "tables").exists()

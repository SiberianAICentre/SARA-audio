from __future__ import annotations

from sara_audio.feature_filter import filter_feature_rows, filter_wide_table


def test_filter_wide_table_keeps_identity_columns() -> None:
    rows, columns = filter_wide_table(
        [
            {
                "file_id": "a",
                "input_path": "in.wav",
                "TIME_audio_duration_s": "1",
                "LEX_EmotLex": "0",
            }
        ],
        ["file_id", "input_path", "TIME_audio_duration_s", "LEX_EmotLex"],
        frozenset({"file_id", "TIME_audio_duration_s"}),
    )

    assert columns == ["file_id", "input_path", "LEX_EmotLex"]
    assert rows == [{"file_id": "a", "input_path": "in.wav", "LEX_EmotLex": "0"}]


def test_filter_feature_rows_drops_excluded_feature_codes() -> None:
    rows = filter_feature_rows(
        [
            {"feature_code": "TIME_audio_duration_s", "value": 1},
            {"feature_code": "LEX_EmotLex", "value": 0},
        ],
        frozenset({"TIME_audio_duration_s"}),
    )

    assert rows == [{"feature_code": "LEX_EmotLex", "value": 0}]

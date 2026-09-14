"""Feature exclusion helpers for public delivery tables."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

DEFAULT_EXCLUDE_FEATURES_PATH = Path("exclude_features.txt")
IDENTITY_COLUMNS = frozenset(
    {
        "file_id",
        "record_id",
        "input_path",
        "analysis_audio_path",
        "split",
        "analysis_scope",
        "analysis_text_status",
    }
)


def load_excluded_feature_codes(
    path: Path | None = None,
) -> frozenset[str]:
    """Read one feature code per line, ignoring comments and blanks."""
    target = path or _default_exclude_path()
    if not target.is_file():
        return frozenset()
    codes = []
    for line in target.read_text(encoding="utf-8-sig").splitlines():
        code = line.strip()
        if code and not code.startswith("#"):
            codes.append(code)
    return frozenset(codes)


def _default_exclude_path() -> Path:
    cwd_candidate = DEFAULT_EXCLUDE_FEATURES_PATH
    if cwd_candidate.is_file():
        return cwd_candidate
    source_candidate = Path(__file__).resolve().parents[2] / "exclude_features.txt"
    if source_candidate.is_file():
        return source_candidate
    return cwd_candidate


def feature_is_excluded(code: str, excluded_codes: frozenset[str]) -> bool:
    """Keep identity columns even if a user accidentally lists one."""
    return code in excluded_codes and code not in IDENTITY_COLUMNS


def filter_wide_table(
    rows: list[Mapping[str, object]],
    columns: list[str],
    excluded_codes: frozenset[str],
) -> tuple[list[dict[str, object]], list[str]]:
    """Drop excluded feature columns from wide model/delivery matrices."""
    if not excluded_codes:
        return [dict(row) for row in rows], columns
    kept_columns = [
        column for column in columns if not feature_is_excluded(column, excluded_codes)
    ]
    return (
        [
            {column: row.get(column, "") for column in kept_columns if column in row}
            for row in rows
        ],
        kept_columns,
    )


def filter_feature_rows(
    rows: list[dict[str, object]],
    excluded_codes: frozenset[str],
) -> list[dict[str, object]]:
    """Drop long-form rows whose feature_code is excluded."""
    if not excluded_codes:
        return rows
    return [
        row
        for row in rows
        if not feature_is_excluded(str(row.get("feature_code", "")), excluded_codes)
    ]

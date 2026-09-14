"""Deterministic input manifests and operator-year batch construction."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

_YEAR_PATTERN = re.compile(r"(?:19|20)\d{2}")
_OPERATOR_PATTERN = re.compile(
    r"(?:agent|operator|оператор)[ _-]*(?P<operator>[^_\s-]+)", re.IGNORECASE
)


@dataclass(frozen=True)
class InputRecord:
    """One immutable input item and its routing metadata."""

    record_id: str
    input_path: Path
    sha256: str
    operator_id: str
    year: str
    metadata_status: str

    def as_dict(self) -> dict[str, str]:
        payload = asdict(self)
        payload["input_path"] = str(self.input_path)
        return payload


@dataclass(frozen=True)
class OperatorYearBatch:
    """A deterministic, bounded group of input records."""

    batch_id: str
    operator_id: str
    year: str
    batch_index: int
    records: tuple[InputRecord, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "batch_id": self.batch_id,
            "operator_id": self.operator_id,
            "year": self.year,
            "batch_index": self.batch_index,
            "record_count": len(self.records),
            "records": [record.as_dict() for record in self.records],
        }


class InputManifestBuilder:
    """Build auditable manifests from paths without guessing missing metadata."""

    def __init__(self, batch_size: int = 15) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self._batch_size = batch_size

    def build_records(self, paths: list[Path]) -> tuple[InputRecord, ...]:
        records = tuple(self._record_for(path) for path in sorted(paths))
        record_ids = [record.record_id for record in records]
        if len(set(record_ids)) != len(record_ids):
            raise ValueError("Input record_id values must be unique")
        return records

    def build_batches(
        self, records: tuple[InputRecord, ...]
    ) -> tuple[OperatorYearBatch, ...]:
        groups: dict[tuple[str, str], list[InputRecord]] = {}
        for record in records:
            groups.setdefault((record.operator_id, record.year), []).append(record)

        batches: list[OperatorYearBatch] = []
        for operator_id, year in sorted(groups):
            group_records = sorted(groups[(operator_id, year)], key=lambda record: record.record_id)
            for offset in range(0, len(group_records), self._batch_size):
                batch_index = offset // self._batch_size + 1
                batch_id = f"{self._safe_part(operator_id)}-{year}-{batch_index:03d}"
                batches.append(
                    OperatorYearBatch(
                        batch_id=batch_id,
                        operator_id=operator_id,
                        year=year,
                        batch_index=batch_index,
                        records=tuple(group_records[offset : offset + self._batch_size]),
                    )
                )
        return tuple(batches)

    def write(
        self,
        output_root: Path,
        records: tuple[InputRecord, ...],
        batches: tuple[OperatorYearBatch, ...],
    ) -> tuple[Path, tuple[Path, ...]]:
        output_root.mkdir(parents=True, exist_ok=True)
        manifest_path = output_root / "input_manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "record_count": len(records),
                    "batch_size": self._batch_size,
                    "records": [record.as_dict() for record in records],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        batches_directory = output_root / "batches"
        batches_directory.mkdir(exist_ok=True)
        paths: list[Path] = []
        for batch in batches:
            path = batches_directory / f"{batch.batch_id}.json"
            path.write_text(
                json.dumps(batch.as_dict(), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            paths.append(path)
        return manifest_path, tuple(paths)

    @staticmethod
    def _record_for(path: Path) -> InputRecord:
        stem = path.stem
        year_match = _YEAR_PATTERN.search(stem)
        operator_match = _OPERATOR_PATTERN.search(stem)
        year = year_match.group(0) if year_match else "unknown"
        operator_id = operator_match.group("operator") if operator_match else "unknown"
        status = "complete" if year_match and operator_match else "incomplete-metadata"
        return InputRecord(
            record_id=stem,
            input_path=path.resolve(),
            sha256=InputManifestBuilder._sha256(path),
            operator_id=operator_id,
            year=year,
            metadata_status=status,
        )

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _safe_part(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_") or "unknown"

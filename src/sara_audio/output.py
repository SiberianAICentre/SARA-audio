"""Stable JSONL, CSV, and optional Parquet artifact writers."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from pathlib import Path

from sara_audio.domain import Transcript
from sara_audio.feature_filter import filter_wide_table, load_excluded_feature_codes
from sara_audio.pipeline import PipelineRun


class ArtifactWriter:
    """Write long-form scalar records and frame-level artifacts."""

    def write_feature_rows(
        self, output_path: Path, runs: Iterable[PipelineRun]
    ) -> None:
        rows = [row for run in runs for row in run.result.feature_rows()]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix.lower() == ".jsonl":
            with output_path.open("w", encoding="utf-8") as destination:
                for row in rows:
                    destination.write(json.dumps(row, ensure_ascii=False) + "\n")
            return
        if output_path.suffix.lower() == ".json":
            output_path.write_text(
                json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            return
        if output_path.suffix.lower() == ".csv":
            self._write_csv(output_path, rows)
            return
        if output_path.suffix.lower() == ".parquet":
            self._write_parquet(output_path, rows)
            return
        raise ValueError(
            "Feature output must have .jsonl, .json, .csv, or .parquet extension"
        )

    def write_wide_features(
        self, output_path: Path, runs: Iterable[PipelineRun]
    ) -> None:
        """Write a model-ready matrix with one row per input record."""
        rows: list[dict[str, object]] = []
        for run in runs:
            row: dict[str, object] = {
                "file_id": run.result.file_id,
                "input_path": str(run.result.input_path),
                "analysis_audio_path": str(run.analysis_audio_path),
            }
            for feature in run.result.features:
                if feature.code in row:
                    raise ValueError(f"Duplicate wide feature code: {feature.code}")
                row[feature.code] = feature.value
            rows.append(row)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        columns = ["file_id", "input_path", "analysis_audio_path"]
        columns.extend(sorted({key for row in rows for key in row} - set(columns)))
        rows, columns = filter_wide_table(
            rows,
            columns,
            load_excluded_feature_codes(),
        )
        if output_path.suffix.lower() == ".csv":
            with output_path.open("w", newline="", encoding="utf-8") as destination:
                writer = csv.DictWriter(destination, fieldnames=columns, delimiter=";")
                writer.writeheader()
                writer.writerows(rows)
            return
        if output_path.suffix.lower() == ".parquet":
            try:
                import pandas as pd
            except ImportError as error:
                raise RuntimeError(
                    "pandas and pyarrow are required for Parquet output."
                ) from error
            pd.DataFrame(rows).to_parquet(output_path, index=False)
            return
        raise ValueError("Wide feature output must have .csv or .parquet extension")

    def write_prediction_rows(
        self,
        output_path: Path,
        rows: list[dict[str, object]],
        columns: Iterable[str] | None = None,
    ) -> None:
        """Write compact predictions or the full interpretation input table."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix.lower() == ".json":
            output_path.write_text(
                json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            return
        if output_path.suffix.lower() != ".csv":
            raise ValueError("Prediction output must have .csv or .json extension")
        output_columns = list(columns or dict.fromkeys(key for row in rows for key in row))
        with output_path.open("w", newline="", encoding="utf-8") as destination:
            writer = csv.DictWriter(destination, fieldnames=output_columns, delimiter=";")
            writer.writeheader()
            for row in rows:
                serialized = dict(row)
                for key, value in serialized.items():
                    if isinstance(value, (list, dict, tuple)):
                        serialized[key] = json.dumps(value, ensure_ascii=False)
                writer.writerow(serialized)

    def write_segments(self, output_path: Path, runs: Iterable[PipelineRun]) -> None:
        records = []
        for run in runs:
            analysis = run.result.pause_analysis
            if analysis is None:
                continue
            records.append(
                {
                    "file_id": run.result.file_id,
                    "input_path": str(run.result.input_path),
                    "analysis_audio_path": str(run.analysis_audio_path),
                    "speech_segments": [
                        segment.__dict__ for segment in analysis.speech_segments
                    ],
                    "pauses": [pause.__dict__ for pause in analysis.pauses],
                    "transcript": run.result.transcript.text
                    if run.result.transcript
                    else None,
                    "diagnostics": list(run.result.diagnostics),
                }
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix.lower() == ".json":
            payload = records[0] if len(records) == 1 else records
            output_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            return
        if output_path.suffix.lower() == ".jsonl":
            with output_path.open("w", encoding="utf-8") as destination:
                for record in records:
                    destination.write(json.dumps(record, ensure_ascii=False) + "\n")
            return
        raise ValueError("Segments output must have .json or .jsonl extension")

    @staticmethod
    def write_transcript(
        text_path: Path,
        metadata_path: Path,
        transcript: Transcript,
    ) -> None:
        """Write readable transcript text and auditable ASR/manual metadata."""
        text_path.parent.mkdir(parents=True, exist_ok=True)
        text_path.write_text(transcript.text.rstrip() + "\n", encoding="utf-8")
        metadata = {
            "text": transcript.text,
            "source": transcript.source,
            "is_verified": transcript.is_verified,
            "tokens": [token.__dict__ for token in transcript.tokens],
        }
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def write_frames(self, output_path: Path, runs: Iterable[PipelineRun]) -> None:
        frames = []
        for run in runs:
            if run.frames is None:
                continue
            frame = run.frames.copy()
            frame.insert(0, "file_id", run.result.file_id)
            frames.append(frame)
        if not frames:
            return
        try:
            import pandas as pd
        except ImportError as error:
            raise RuntimeError(
                "pandas is required to write openSMILE frame artifacts."
            ) from error
        combined = pd.concat(frames)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix.lower() == ".csv":
            combined.to_csv(output_path, index=True, sep=";")
        elif output_path.suffix.lower() == ".parquet":
            combined.to_parquet(output_path, index=True)
        else:
            raise ValueError("Frame output must have .csv or .parquet extension")

    @staticmethod
    def _write_csv(output_path: Path, rows: list[dict[str, object]]) -> None:
        if not rows:
            output_path.write_text("", encoding="utf-8")
            return
        columns = list(rows[0])
        with output_path.open("w", newline="", encoding="utf-8") as destination:
            writer = csv.DictWriter(destination, fieldnames=columns, delimiter=";")
            writer.writeheader()
            for row in rows:
                row = dict(row)
                row["evidence"] = json.dumps(row["evidence"], ensure_ascii=False)
                writer.writerow(row)

    @staticmethod
    def _write_parquet(output_path: Path, rows: list[dict[str, object]]) -> None:
        try:
            import pandas as pd
        except ImportError as error:
            raise RuntimeError(
                "Install pandas and pyarrow for Parquet output."
            ) from error
        table = pd.DataFrame(rows)
        table["evidence"] = table["evidence"].map(
            lambda value: json.dumps(value, ensure_ascii=False)
        )
        table.to_parquet(output_path, index=False)

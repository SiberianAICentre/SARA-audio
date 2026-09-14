"""Mandatory end-to-end verification for the CUDA server deployment."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from sara_audio.asr import asr_runtime_diagnostics
from sara_audio.config import ApplicationSettings
from sara_audio.runner import BatchOutput, BatchProcessor


class ServerSmokeVerifier:
    """Validate artifacts which prove an actual server inference completed."""

    REQUIRED_WIDE_COLUMNS = frozenset(
        {
            "LEX_EmotLex",
            "TIME_active_speech_duration_s",
            "ASR_token_count",
            "bert_1",
        }
    )

    def run(
        self, settings: ApplicationSettings, input_path: Path, output_path: Path
    ) -> dict[str, object]:
        diagnostics = asr_runtime_diagnostics(settings.execution.asr, load_model=True)
        cuda_count = diagnostics.get("cuda_device_count")
        if (
            settings.execution.asr.device != "cuda"
            or not isinstance(cuda_count, int)
            or cuda_count < 1
        ):
            raise RuntimeError(f"CUDA ASR is not available: {diagnostics}")
        result = BatchProcessor(settings).process(
            input_path, output_path, show_progress=True
        )
        return {
            "runtime": diagnostics,
            "artifacts": self.verify_output(
                result,
                require_methodology=(
                    settings.prediction.methodology_model_directory is not None
                ),
            ),
        }

    def verify_output(
        self,
        result: BatchOutput,
        *,
        require_methodology: bool = False,
    ) -> dict[str, object]:
        if result.wide_features_path is None or not result.wide_features_path.is_file():
            raise RuntimeError("Server run did not produce features_wide.csv")
        with result.wide_features_path.open(encoding="utf-8", newline="") as source:
            header = source.readline()
            source.seek(0)
            rows = list(csv.DictReader(source, delimiter=";" if ";" in header else ","))
        if len(rows) != len(result.records):
            raise RuntimeError("Wide matrix row count does not match processed records")
        if not rows:
            raise RuntimeError("Wide matrix has no records")
        columns = set(rows[0])
        missing = self.REQUIRED_WIDE_COLUMNS - columns
        if missing:
            raise RuntimeError(
                f"Wide matrix is missing required columns: {sorted(missing)}"
            )
        if not any(column.startswith("ACOUSTIC_") for column in columns):
            raise RuntimeError("Wide matrix has no aggregated openSMILE columns")
        if any(not row["ASR_token_count"] for row in rows):
            raise RuntimeError(
                "ASR did not emit tokens for one or more smoke-test records"
            )
        for record in result.records:
            if (
                record.operator_speech_path is None
                or not record.operator_speech_path.is_file()
            ):
                raise RuntimeError(
                    f"Speech-only artifact is missing for {record.input_path}"
                )
        if result.prediction_paths:
            required_prediction_names = {
                "predictions.csv",
                "interpretation_input.csv",
            }
            if require_methodology:
                required_prediction_names.add("filter_predictions.csv")
            actual_prediction_names = {
                path.name for path in result.prediction_paths
            }
            missing_predictions = required_prediction_names - actual_prediction_names
            if missing_predictions:
                raise RuntimeError(
                    "Prediction artifacts are missing: "
                    f"{sorted(missing_predictions)}"
                )
        batch_sizes = []
        for batch_path in result.batch_manifest_paths:
            batch = json.loads(batch_path.read_text(encoding="utf-8"))
            size = batch["record_count"]
            if size > 15:
                raise RuntimeError(f"Batch {batch['batch_id']} exceeds 15 records")
            batch_sizes.append(size)
        return {
            "record_count": len(result.records),
            "wide_features": str(result.wide_features_path),
            "predictions": [str(path) for path in result.prediction_paths],
            "batch_sizes": batch_sizes,
            "status": "passed",
        }

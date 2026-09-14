"""Minimal user-facing orchestration for file and directory processing."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from sara_audio.asr import FasterWhisperTranscriber
from sara_audio.batching import InputManifestBuilder
from sara_audio.config import ApplicationSettings
from sara_audio.domain import Transcript
from sara_audio.feature_filter import load_excluded_feature_codes
from sara_audio.output import ArtifactWriter
from sara_audio.pipeline import FeaturePipeline, PipelineRun
from sara_audio.predictions import (
    FILTER_PREDICTION_COLUMNS,
    FINAL_PREDICTION_COLUMNS,
    JsonLinearModel,
    MethodologyModelBundle,
    run_methodology_prediction_rows,
    run_prediction_rows,
)
from sara_audio.registry import LinguisticFeatureRegistry

SUPPORTED_INPUT_SUFFIXES = frozenset({".wav", ".m4a", ".mp3", ".flac", ".ogg"})
INVALID_DIRECTORY_CHARACTERS = re.compile(r'[<>:"/\\|?*]+')


@dataclass(frozen=True)
class RecordOutput:
    """Locations of all artifacts produced for a single source record."""

    input_path: Path
    directory: Path
    feature_paths: tuple[Path, ...]
    frame_paths: tuple[Path, ...]
    segments_path: Path | None
    transcript_paths: tuple[Path, ...]
    manifest_path: Path
    report_path: Path
    operator_speech_path: Path | None

    @property
    def features_path(self) -> Path:
        """Primary human-readable scalar artifact, preferring CSV."""
        return next(
            (path for path in self.feature_paths if path.suffix.lower() == ".csv"),
            self.feature_paths[0],
        )

    @property
    def frames_path(self) -> Path | None:
        """Primary human-readable frame artifact, preferring CSV."""
        return next(
            (path for path in self.frame_paths if path.suffix.lower() == ".csv"),
            self.frame_paths[0] if self.frame_paths else None,
        )


@dataclass(frozen=True)
class BatchOutput:
    """Locations returned after processing a file or a complete directory."""

    directory: Path
    summary_path: Path
    records: tuple[RecordOutput, ...]
    input_manifest_path: Path | None = None
    batch_manifest_paths: tuple[Path, ...] = ()
    wide_features_path: Path | None = None
    prediction_paths: tuple[Path, ...] = ()


class BatchProcessor:
    """Process one file or a directory using only the default YAML config."""

    def __init__(self, settings: ApplicationSettings) -> None:
        self._settings = settings
        self._writer = ArtifactWriter()
        self._registry = LinguisticFeatureRegistry.load_default()
        self._transcriber = self._build_transcriber()
        self._pipeline: FeaturePipeline | None = None

    def process(
        self,
        input_path: Path,
        output_directory: Path | None = None,
        *,
        show_progress: bool = False,
    ) -> BatchOutput:
        files = self._find_audio_files(input_path)
        if not files:
            raise FileNotFoundError(f"No supported audio files found under {input_path}")

        root = self._create_output_root(input_path, output_directory)
        manifest_builder = InputManifestBuilder(self._settings.execution.batch_size)
        input_records = manifest_builder.build_records(files)
        batches = manifest_builder.build_batches(input_records)
        input_manifest_path, batch_manifest_paths = manifest_builder.write(
            root, input_records, batches
        )
        records: list[RecordOutput] = []
        runs: list[PipelineRun] = []
        names_in_use: set[str] = set()
        if self._pipeline is None:
            self._pipeline = FeaturePipeline(
                cache_directory=root / ".cache",
                settings=self._settings.pipeline,
                registry=self._registry,
            )
        for index, audio_path in enumerate(files, start=1):
            if show_progress:
                print(f"[{index}/{len(files)}] Processing: {audio_path.name}", flush=True)
            record_name = self._allocate_record_name(audio_path.stem, names_in_use)
            record_directory = root / "records" / record_name
            record_directory.mkdir(parents=True, exist_ok=False)
            transcript = (
                None
                if self._settings.execution.transcript_mode == "asr"
                else self._load_transcript(audio_path, input_path)
            )
            run = self._pipeline.extract(
                audio_path,
                transcript,
                self._transcriber,
                cache_directory=root / ".cache" / record_name,
            )
            runs.append(run)
            records.append(self._write_record(run, record_directory))
            if show_progress:
                print(f"[{index}/{len(files)}] Completed: {audio_path.name}", flush=True)

        wide_features_path = root / "features_wide.csv"
        self._writer.write_wide_features(wide_features_path, runs)
        prediction_paths: tuple[Path, ...] = ()
        if self._settings.prediction.enabled:
            model_path = self._settings.prediction.filter_model_path
            assert model_path is not None
            paths: list[Path] = []
            methodology_directory = (
                self._settings.prediction.methodology_model_directory
            )
            if methodology_directory is not None:
                models = MethodologyModelBundle.load(
                    model_path,
                    methodology_directory,
                )
                rows = run_methodology_prediction_rows(
                    runs,
                    models,
                    threshold=self._settings.prediction.filter_threshold,
                )
                for extension in self._settings.prediction.formats:
                    compact_path = root / f"predictions.{extension}"
                    full_path = root / f"interpretation_input.{extension}"
                    filter_path = root / f"filter_predictions.{extension}"
                    self._writer.write_prediction_rows(
                        compact_path,
                        rows.final,
                        FINAL_PREDICTION_COLUMNS,
                    )
                    self._writer.write_prediction_rows(
                        full_path,
                        rows.interpretation,
                        FINAL_PREDICTION_COLUMNS if not rows.interpretation else None,
                    )
                    self._writer.write_prediction_rows(
                        filter_path,
                        rows.filtering,
                        FILTER_PREDICTION_COLUMNS,
                    )
                    paths.extend((compact_path, full_path, filter_path))
            else:
                model = JsonLinearModel.load(model_path)
                compact_rows, full_rows = run_prediction_rows(
                    runs,
                    model,
                    threshold=self._settings.prediction.filter_threshold,
                )
                for extension in self._settings.prediction.formats:
                    compact_path = root / f"predictions.{extension}"
                    full_path = root / f"interpretation_input.{extension}"
                    self._writer.write_prediction_rows(compact_path, compact_rows)
                    self._writer.write_prediction_rows(full_path, full_rows)
                    paths.extend((compact_path, full_path))
            prediction_paths = tuple(paths)
        summary_path = root / "run_summary.json"
        self._write_summary(
            summary_path,
            records,
            input_manifest_path,
            batch_manifest_paths,
            wide_features_path,
            tuple(sorted(load_excluded_feature_codes())),
            prediction_paths,
        )
        if show_progress:
            print(f"Completed {len(records)} record(s): {root}", flush=True)
        return BatchOutput(
            root,
            summary_path,
            tuple(records),
            input_manifest_path,
            batch_manifest_paths,
            wide_features_path,
            prediction_paths,
        )

    def _find_audio_files(self, input_path: Path) -> list[Path]:
        if input_path.is_file():
            return self._validate_audio_file(input_path)
        if input_path.is_dir():
            return sorted(
                path
                for path in input_path.rglob("*")
                if (
                    path.is_file()
                    and not path.name.startswith("._")
                    and path.suffix.lower() in SUPPORTED_INPUT_SUFFIXES
                )
            )
        raise FileNotFoundError(f"Input does not exist: {input_path}")

    @staticmethod
    def _validate_audio_file(input_path: Path) -> list[Path]:
        if input_path.suffix.lower() not in SUPPORTED_INPUT_SUFFIXES:
            raise ValueError(f"Unsupported audio format: {input_path.suffix}")
        return [input_path]

    def _create_output_root(
        self,
        input_path: Path,
        explicit_directory: Path | None,
    ) -> Path:
        if explicit_directory is not None:
            if explicit_directory.exists():
                raise FileExistsError(
                    f"Output directory already exists: {explicit_directory}. "
                    "Choose a new name to protect previous results."
                )
            explicit_directory.mkdir(parents=True)
            return explicit_directory

        base_name = self._safe_name(input_path.stem or input_path.name)
        base = self._settings.execution.output_root / base_name
        candidate = base
        suffix = 2
        while candidate.exists():
            candidate = base.with_name(f"{base.name}-{suffix}")
            suffix += 1
        candidate.mkdir(parents=True)
        return candidate

    def _load_transcript(self, audio_path: Path, input_path: Path) -> Transcript | None:
        mode = self._settings.execution.transcript_mode
        if mode == "none":
            return None
        if mode == "asr":
            if self._transcriber is None:
                raise RuntimeError("ASR transcriber is not configured")
            return self._transcriber.transcribe(audio_path)

        transcript_path = self._manual_transcript_path(audio_path, input_path)
        if not transcript_path.is_file():
            raise FileNotFoundError(
                f"Transcript for {audio_path.name} is missing: {transcript_path}"
            )
        return Transcript(
            text=transcript_path.read_text(encoding="utf-8"),
            source="manual-text-file",
            is_verified=True,
        )

    def _manual_transcript_path(self, audio_path: Path, input_path: Path) -> Path:
        transcript_root = self._settings.execution.transcript_directory
        if transcript_root is None:
            raise RuntimeError("Manual transcript directory is not configured")
        if input_path.is_dir():
            relative_path = audio_path.relative_to(input_path).with_suffix(
                self._settings.execution.transcript_suffix
            )
            return transcript_root / relative_path
        return transcript_root / (
            audio_path.stem + self._settings.execution.transcript_suffix
        )

    def _build_transcriber(self) -> FasterWhisperTranscriber | None:
        if self._settings.execution.transcript_mode != "asr":
            return None
        return FasterWhisperTranscriber(self._settings.execution.asr)

    def _write_record(self, run: PipelineRun, directory: Path) -> RecordOutput:
        operator_speech_path = directory / "operator_speech_no_pauses.wav"
        shutil.copy2(run.analysis_audio_path, operator_speech_path)
        feature_paths = tuple(
            directory / f"features.{extension}"
            for extension in self._settings.execution.feature_formats
        )
        for features_path in feature_paths:
            self._writer.write_feature_rows(features_path, [run])

        frame_paths: tuple[Path, ...] = ()
        if self._settings.execution.write_frames and run.frames is not None:
            frame_paths = tuple(
                directory / f"frames.{extension}"
                for extension in self._settings.execution.frame_formats
            )
            for frames_path in frame_paths:
                self._writer.write_frames(frames_path, [run])

        segments_path = None
        if self._settings.execution.write_segments:
            segments_path = directory / "segments.json"
            self._writer.write_segments(segments_path, [run])

        transcript_paths: tuple[Path, ...] = ()
        if run.result.transcript is not None:
            transcript_paths = (directory / "transcript.txt", directory / "transcript.json")
            self._writer.write_transcript(
                transcript_paths[0],
                transcript_paths[1],
                run.result.transcript,
            )

        manifest_path = directory / "manifest.json"
        manifest = {
            "input_path": str(run.result.input_path),
            "normalized_audio_path": str(run.normalized_audio_path),
            "analysis_audio_path": str(operator_speech_path),
            "feature_count": len(run.result.features),
            "diagnostics": list(run.result.diagnostics),
            "opensmile_semantic_coverage": run.opensmile_coverage,
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        report_path = directory / "report.json"
        self._write_report(
            report_path,
            run,
            feature_paths,
            frame_paths,
            transcript_paths,
        )
        return RecordOutput(
            input_path=run.result.input_path,
            directory=directory,
            feature_paths=feature_paths,
            frame_paths=frame_paths,
            segments_path=segments_path,
            transcript_paths=transcript_paths,
            manifest_path=manifest_path,
            report_path=report_path,
            operator_speech_path=operator_speech_path,
        )

    def _write_report(
        self,
        path: Path,
        run: PipelineRun,
        feature_paths: tuple[Path, ...],
        frame_paths: tuple[Path, ...],
        transcript_paths: tuple[Path, ...],
    ) -> None:
        temporal: dict[str, dict[str, object]] = {}
        speech_rate: dict[str, dict[str, object]] = {}
        linguistic: list[dict[str, object]] = []
        for feature in run.result.features:
            value = feature.as_dict()
            code = feature.code
            if code.startswith("TIME_"):
                temporal[code] = value
            elif code.startswith("RATE_"):
                speech_rate[code] = value
            elif code in {definition.code for definition in self._registry.definitions}:
                value["description"] = self._registry.get(code).description
                linguistic.append(value)

        transcript = run.result.transcript
        payload = {
            "input_path": str(run.result.input_path),
            "analysis_audio_path": str(run.analysis_audio_path),
            "transcript": {
                "available": transcript is not None,
                "source": transcript.source if transcript else None,
                "is_verified": transcript.is_verified if transcript else None,
                "artifacts": [path.name for path in transcript_paths],
            },
            "temporal_features": temporal,
            "speech_rate": speech_rate,
            "linguistic_features": linguistic,
            "acoustic_features": {
                "frame_artifacts": [path.name for path in frame_paths],
                "semantic_coverage": run.opensmile_coverage,
            },
            "scalar_artifacts": [path.name for path in feature_paths],
            "diagnostics": list(run.result.diagnostics),
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _write_summary(
        path: Path,
        records: list[RecordOutput],
        input_manifest_path: Path,
        batch_manifest_paths: tuple[Path, ...],
        wide_features_path: Path,
        excluded_feature_codes: tuple[str, ...] = (),
        prediction_paths: tuple[Path, ...] = (),
    ) -> None:
        payload = {
            "record_count": len(records),
            "input_manifest": str(input_manifest_path),
            "batches": [str(batch_path) for batch_path in batch_manifest_paths],
            "wide_features": str(wide_features_path),
            "wide_excluded_features": list(excluded_feature_codes),
            "predictions": [str(path) for path in prediction_paths],
            "records": [
                {
                    "input_path": str(record.input_path),
                    "directory": str(record.directory),
                    "features": [str(path) for path in record.feature_paths],
                    "frames": [str(path) for path in record.frame_paths],
                    "segments": (
                        str(record.segments_path) if record.segments_path else None
                    ),
                    "transcripts": [str(path) for path in record.transcript_paths],
                    "manifest": str(record.manifest_path),
                    "report": str(record.report_path),
                    "operator_speech": (
                        str(record.operator_speech_path)
                        if record.operator_speech_path
                        else None
                    ),
                }
                for record in records
            ],
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _allocate_record_name(stem: str, names_in_use: set[str]) -> str:
        base = BatchProcessor._safe_name(stem)
        candidate = base
        suffix = 2
        while candidate.casefold() in names_in_use:
            candidate = f"{base}-{suffix}"
            suffix += 1
        names_in_use.add(candidate.casefold())
        return candidate

    @staticmethod
    def _safe_name(value: str) -> str:
        cleaned = INVALID_DIRECTORY_CHARACTERS.sub("_", value).strip(". ")
        return cleaned or "record"

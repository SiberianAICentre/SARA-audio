"""Typed settings and optional YAML configuration loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sara_audio.asr import FasterWhisperSettings
from sara_audio.audio import AudioPreprocessorSettings
from sara_audio.mic_features import MicFeatureSettings
from sara_audio.opensmile_extractor import OpenSmileSettings
from sara_audio.text_embeddings import TextEmbeddingSettings
from sara_audio.vad import VadSettings


@dataclass(frozen=True)
class PipelineSettings:
    """All extraction settings used to reproduce a pipeline run."""

    audio: AudioPreprocessorSettings = field(default_factory=AudioPreprocessorSettings)
    vad: VadSettings = field(default_factory=VadSettings)
    opensmile: OpenSmileSettings = field(default_factory=OpenSmileSettings)
    mic: MicFeatureSettings = field(default_factory=MicFeatureSettings)
    text_embeddings: TextEmbeddingSettings = field(default_factory=TextEmbeddingSettings)
    extract_opensmile: bool = True

    @classmethod
    def from_yaml(cls, path: Path) -> PipelineSettings:
        try:
            import yaml
        except ImportError as error:
            raise RuntimeError("Install PyYAML to load a YAML configuration file.") from error
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls.from_mapping(payload)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> PipelineSettings:
        audio = payload.get("audio", {})
        vad = payload.get("vad", {})
        opensmile = payload.get("opensmile", {})
        mic = payload.get("mic", {})
        text_embeddings = payload.get("text_embeddings", {})
        return cls(
            audio=AudioPreprocessorSettings(
                target_sample_rate_hz=int(audio.get("target_sample_rate_hz", 16000)),
                mono=bool(audio.get("mono", True)),
                operator_channel_index=(
                    int(audio["operator_channel_index"])
                    if audio.get("operator_channel_index") is not None
                    else None
                ),
                allow_mono_operator_fallback=bool(
                    audio.get("allow_mono_operator_fallback", True)
                ),
            ),
            vad=VadSettings(
                frame_duration_ms=int(vad.get("frame_duration_ms", 30)),
                pause_threshold_ms=int(vad.get("pause_threshold_ms", 300)),
                min_speech_duration_ms=int(vad.get("min_speech_duration_ms", 90)),
                absolute_energy_floor_dbfs=float(vad.get("absolute_energy_floor_dbfs", -45.0)),
                activation_margin_db=float(vad.get("activation_margin_db", 10.0)),
            ),
            opensmile=OpenSmileSettings(
                feature_set=str(opensmile.get("feature_set", "ComParE_2016")),
                frame_step_seconds=float(opensmile.get("frame_step_seconds", 0.01)),
                aggregate_statistics=tuple(
                    str(item)
                    for item in opensmile.get(
                        "aggregate_statistics", ["mean", "std", "min", "max"]
                    )
                ),
            ),
            mic=MicFeatureSettings(
                enabled=bool(mic.get("enabled", True)),
                utterance_merge_gap_s=float(mic.get("utterance_merge_gap_s", 1.2)),
            ),
            text_embeddings=TextEmbeddingSettings(
                enabled=bool(text_embeddings.get("enabled", False)),
                model_name=str(
                    text_embeddings.get("model_name", "ai-forever/ru-en-RoSBERTa")
                ),
                device=str(text_embeddings.get("device", "auto")),
                max_length=int(text_embeddings.get("max_length", 512)),
                feature_prefix=str(text_embeddings.get("feature_prefix", "bert_")),
            ),
            extract_opensmile=bool(opensmile.get("enabled", True)),
        )


@dataclass(frozen=True)
class ExecutionSettings:
    """User-facing settings for the one-command processing module."""

    output_root: Path = Path("artifacts")
    feature_formats: tuple[str, ...] = ("csv", "json", "parquet")
    frame_formats: tuple[str, ...] = ("csv", "parquet")
    write_frames: bool = True
    write_segments: bool = True
    transcript_mode: str = "none"
    transcript_directory: Path | None = None
    transcript_suffix: str = ".txt"
    batch_size: int = 15
    asr: FasterWhisperSettings = field(default_factory=FasterWhisperSettings)

    def __post_init__(self) -> None:
        valid_feature_formats = {"jsonl", "json", "csv", "parquet"}
        valid_frame_formats = {"csv", "parquet"}
        if not self.feature_formats or not set(self.feature_formats) <= valid_feature_formats:
            raise ValueError(
                "execution.feature_formats accepts jsonl, json, csv, and parquet"
            )
        if not self.frame_formats or not set(self.frame_formats) <= valid_frame_formats:
            raise ValueError("execution.frame_formats accepts csv and parquet")
        if self.transcript_mode not in {"none", "manual_directory", "asr"}:
            raise ValueError(
                "execution.transcript_mode must be none, manual_directory, or asr"
            )
        if self.transcript_mode == "manual_directory" and self.transcript_directory is None:
            raise ValueError(
                "execution.transcript_directory is required for manual_directory mode"
            )
        if not self.transcript_suffix.startswith("."):
            raise ValueError("execution.transcript_suffix must start with a dot")
        if self.batch_size <= 0:
            raise ValueError("execution.batch_size must be positive")


@dataclass(frozen=True)
class PredictionSettings:
    """Settings for applying MIC JSON models after feature extraction."""

    enabled: bool = False
    filter_model_path: Path | None = None
    methodology_model_directory: Path | None = None
    filter_threshold: float = 0.5
    formats: tuple[str, ...] = ("csv", "json")

    def __post_init__(self) -> None:
        valid_formats = {"csv", "json"}
        if not set(self.formats) <= valid_formats or not self.formats:
            raise ValueError("prediction.formats accepts csv and json")
        if not 0.0 <= self.filter_threshold <= 1.0:
            raise ValueError("prediction.filter_threshold must be between 0 and 1")
        if self.enabled and self.filter_model_path is None:
            raise ValueError("prediction.filter_model_path is required when enabled")


@dataclass(frozen=True)
class ApplicationSettings:
    """Complete configuration loaded once by the user-facing processor."""

    pipeline: PipelineSettings = field(default_factory=PipelineSettings)
    execution: ExecutionSettings = field(default_factory=ExecutionSettings)
    prediction: PredictionSettings = field(default_factory=PredictionSettings)

    @classmethod
    def from_yaml(cls, path: Path) -> ApplicationSettings:
        try:
            import yaml
        except ImportError as error:
            raise RuntimeError("Install PyYAML to load a YAML configuration file.") from error
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls.from_mapping(payload)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> ApplicationSettings:
        execution = payload.get("execution", {})
        prediction = payload.get("prediction", {})
        asr = payload.get("asr", {})
        transcript_directory = execution.get("transcript_directory")
        legacy_format = execution.get("output_format")
        feature_formats = execution.get("feature_formats")
        if feature_formats is None:
            feature_formats = [legacy_format or "parquet"]
        frame_formats = execution.get("frame_formats")
        if frame_formats is None:
            frame_formats = [
                legacy_format if legacy_format in {"csv", "parquet"} else "parquet"
            ]
        return cls(
            pipeline=PipelineSettings.from_mapping(payload),
            execution=ExecutionSettings(
                output_root=Path(execution.get("output_root", "artifacts")),
                feature_formats=tuple(str(item) for item in feature_formats),
                frame_formats=tuple(str(item) for item in frame_formats),
                write_frames=bool(execution.get("write_frames", True)),
                write_segments=bool(execution.get("write_segments", True)),
                transcript_mode=str(execution.get("transcript_mode", "none")),
                transcript_directory=(
                    Path(transcript_directory) if transcript_directory else None
                ),
                transcript_suffix=str(execution.get("transcript_suffix", ".txt")),
                batch_size=int(execution.get("batch_size", 15)),
                asr=FasterWhisperSettings(
                    model=str(asr.get("model", "large-v3")),
                    language=str(asr.get("language", "ru")),
                    device=str(asr.get("device", "auto")),
                    compute_type=str(asr.get("compute_type", "int8")),
                    initial_prompt=(
                        str(asr["initial_prompt"])
                        if asr.get("initial_prompt") is not None
                        else None
                    ),
                    vad_filter=bool(asr.get("vad_filter", False)),
                    vad_min_speech_duration_ms=int(
                        asr.get("vad_min_speech_duration_ms", 250)
                    ),
                    vad_max_speech_duration_s=float(
                        asr.get("vad_max_speech_duration_s", 15.0)
                    ),
                    vad_speech_pad_ms=int(asr.get("vad_speech_pad_ms", 600)),
                    no_speech_threshold=(
                        float(asr["no_speech_threshold"])
                        if asr.get("no_speech_threshold") is not None
                        else None
                    ),
                    compression_ratio_threshold=(
                        float(asr["compression_ratio_threshold"])
                        if asr.get("compression_ratio_threshold") is not None
                        else None
                    ),
                    temperature=cls._temperature(asr.get("temperature")),
                    suppress_common_hallucinations=bool(
                        asr.get("suppress_common_hallucinations", True)
                    ),
                ),
            ),
            prediction=PredictionSettings(
                enabled=bool(prediction.get("enabled", False)),
                filter_model_path=(
                    Path(str(prediction["filter_model_path"]))
                    if prediction.get("filter_model_path")
                    else None
                ),
                methodology_model_directory=(
                    Path(str(prediction["methodology_model_directory"]))
                    if prediction.get("methodology_model_directory")
                    else None
                ),
                filter_threshold=float(prediction.get("filter_threshold", 0.5)),
                formats=tuple(str(item) for item in prediction.get("formats", ["csv", "json"])),
            ),
        )

    @staticmethod
    def _temperature(value: Any) -> tuple[float, ...] | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return (float(value),)
        return tuple(float(item) for item in value)

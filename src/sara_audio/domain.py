"""Immutable domain models used across extraction stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, order=True)
class SpeechSegment:
    """A half-open interval containing detected speech."""

    start_s: float
    end_s: float

    def __post_init__(self) -> None:
        if self.start_s < 0:
            raise ValueError("Speech segment start cannot be negative")
        if self.end_s <= self.start_s:
            raise ValueError("Speech segment end must be after its start")

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass(frozen=True, order=True)
class PauseSegment:
    """A half-open silent interval between two detected speech segments."""

    start_s: float
    end_s: float
    kind: str

    def __post_init__(self) -> None:
        if self.kind not in {"first", "internal", "trailing"}:
            raise ValueError(f"Unknown pause kind: {self.kind}")
        if self.start_s < 0 or self.end_s < self.start_s:
            raise ValueError("Invalid pause interval")

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass(frozen=True)
class FeatureValue:
    """One auditable scalar feature value, including unavailable values."""

    code: str
    value: float | None
    unit: str
    source: str
    confidence: float | None
    method_version: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.code:
            raise ValueError("Feature code is required")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0 and 1")

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature_code": self.code,
            "value": self.value,
            "unit": self.unit,
            "source": self.source,
            "confidence": self.confidence,
            "method_version": self.method_version,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class TranscriptToken:
    """A transcribed token with optional time alignment."""

    text: str
    start_s: float | None = None
    end_s: float | None = None
    confidence: float | None = None


@dataclass(frozen=True)
class Transcript:
    """Text used for linguistic extraction and speech-rate estimation."""

    text: str
    tokens: tuple[TranscriptToken, ...] = ()
    source: str = "manual"
    is_verified: bool = True


@dataclass(frozen=True)
class PauseAnalysis:
    """Result of VAD segmentation and pause calculations."""

    audio_duration_s: float
    speech_segments: tuple[SpeechSegment, ...]
    pauses: tuple[PauseSegment, ...]

    @property
    def first_pause_s(self) -> float | None:
        for pause in self.pauses:
            if pause.kind == "first":
                return pause.duration_s
        return None

    @property
    def internal_pauses(self) -> tuple[PauseSegment, ...]:
        return tuple(pause for pause in self.pauses if pause.kind == "internal")

    @property
    def mean_internal_pause_s(self) -> float | None:
        pauses = self.internal_pauses
        if not pauses:
            return None
        return sum(pause.duration_s for pause in pauses) / len(pauses)

    @property
    def active_speech_duration_s(self) -> float:
        return sum(segment.duration_s for segment in self.speech_segments)

    @property
    def replica_duration_s(self) -> float:
        if not self.speech_segments:
            return 0.0
        return self.speech_segments[-1].end_s - self.speech_segments[0].start_s


@dataclass(frozen=True)
class ExtractionResult:
    """Complete extraction result for one input audio file."""

    input_path: Path
    features: tuple[FeatureValue, ...]
    pause_analysis: PauseAnalysis | None = None
    transcript: Transcript | None = None
    diagnostics: tuple[str, ...] = ()
    record_id: str | None = None

    @property
    def file_id(self) -> str:
        return self.record_id or self.input_path.stem

    def feature_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for feature in self.features:
            row = {"file_id": self.file_id, "input_path": str(self.input_path)}
            row.update(feature.as_dict())
            rows.append(row)
        return rows

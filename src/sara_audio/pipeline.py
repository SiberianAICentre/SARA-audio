"""Coordinator for preparation, acoustic, temporal, and text extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sara_audio.asr_features import AsrFeatureExtractor
from sara_audio.audio import AudioPreprocessor
from sara_audio.config import PipelineSettings
from sara_audio.domain import ExtractionResult, FeatureValue, PauseAnalysis, Transcript
from sara_audio.interfaces import AsrTranscriber
from sara_audio.linguistic import LinguisticFeatureExtractor
from sara_audio.mic_features import MicScalarFeatureExtractor
from sara_audio.opensmile_extractor import OpenSmileExtractor
from sara_audio.registry import LinguisticFeatureRegistry
from sara_audio.speech_rate import SpeechRateExtractor
from sara_audio.text_embeddings import TextEmbeddingExtractor
from sara_audio.vad import EnergyVadPauseExtractor


@dataclass(frozen=True)
class PipelineRun:
    """One result plus non-scalar frame-level openSMILE artifacts."""

    result: ExtractionResult
    frames: Any | None
    opensmile_coverage: dict[str, tuple[str, ...]] | None
    normalized_audio_path: Path
    analysis_audio_path: Path


class FeaturePipeline:
    """Orchestrates extractors without coupling them to each other's internals."""

    def __init__(
        self,
        cache_directory: Path,
        settings: PipelineSettings | None = None,
        registry: LinguisticFeatureRegistry | None = None,
    ) -> None:
        self._settings = settings or PipelineSettings()
        self._cache_directory = cache_directory
        self._preprocessor = AudioPreprocessor(self._settings.audio)
        self._vad = EnergyVadPauseExtractor(self._settings.vad)
        self._opensmile = OpenSmileExtractor(self._settings.opensmile)
        self._speech_rate = SpeechRateExtractor()
        self._asr_features = AsrFeatureExtractor()
        self._mic_features = (
            MicScalarFeatureExtractor(self._settings.mic)
            if self._settings.mic.enabled
            else None
        )
        self._text_embeddings = (
            TextEmbeddingExtractor(self._settings.text_embeddings)
            if self._settings.text_embeddings.enabled
            else None
        )
        self._linguistic = LinguisticFeatureExtractor(
            registry or LinguisticFeatureRegistry.load_default()
        )

    def extract(
        self,
        input_path: Path,
        transcript: Transcript | None = None,
        transcriber: AsrTranscriber | None = None,
        cache_directory: Path | None = None,
    ) -> PipelineRun:
        record_cache = cache_directory or self._cache_directory
        normalized_path = self._preprocessor.prepare(input_path, record_cache)
        pause_analysis = self._vad.analyze(normalized_path)
        speech_only_path = self._preprocessor.write_speech_only_wav(
            normalized_path,
            pause_analysis.speech_segments,
            record_cache / f"{input_path.stem}.operator_speech_no_pauses.wav",
        )
        features = list(self._timing_features(pause_analysis))
        diagnostics = [
            "ASR and acoustic extraction use the selected operator channel with VAD pauses removed."
        ]
        if self._preprocessor.used_mono_operator_fallback:
            diagnostics.append(
                "Mono source: the only available channel was used as operator speech."
            )

        frames = None
        coverage = None
        if self._settings.extract_opensmile:
            open_smile_output = self._opensmile.extract_frames(speech_only_path)
            frames = open_smile_output.frames
            coverage = open_smile_output.semantic_coverage
            features.extend(self._opensmile.aggregate(open_smile_output))
            diagnostics.append(
                "openSMILE semantic coverage is stored with frame-level and aggregate artifacts."
            )

        if transcript is None and transcriber is not None:
            transcript = transcriber.transcribe(speech_only_path)

        if transcript is None:
            diagnostics.append(
                "No transcript: speech-rate and 92 linguistic values were not calculated."
            )
        else:
            features.extend(self._asr_features.extract(transcript))
            features.extend(self._speech_rate.extract(transcript, pause_analysis))
            if self._mic_features is not None:
                features.extend(self._mic_features.extract(transcript, speech_only_path))
                diagnostics.append(
                    "MIC scalar transcript/audio features were calculated."
                )
            if self._text_embeddings is not None:
                features.extend(self._text_embeddings.extract(transcript))
                diagnostics.append(
                    "Dense transcript embeddings were calculated with the configured text encoder."
                )
            features.extend(self._linguistic.extract(transcript))

        return PipelineRun(
            result=ExtractionResult(
                input_path=input_path,
                features=tuple(features),
                pause_analysis=pause_analysis,
                transcript=transcript,
                diagnostics=tuple(diagnostics),
            ),
            frames=frames,
            opensmile_coverage=coverage,
            normalized_audio_path=normalized_path,
            analysis_audio_path=speech_only_path,
        )

    @staticmethod
    def _timing_features(pause_analysis: PauseAnalysis) -> tuple[FeatureValue, ...]:
        evidence = {
            "speech_segment_count": len(pause_analysis.speech_segments),
            "internal_pause_count": len(pause_analysis.internal_pauses),
            "pause_threshold_s": 0.3,
        }
        return (
            FeatureValue(
                code="TIME_first_pause_s",
                value=pause_analysis.first_pause_s,
                unit="s",
                source="energy-vad",
                confidence=0.7 if pause_analysis.first_pause_s is not None else None,
                method_version=EnergyVadPauseExtractor.method_version,
                evidence=evidence,
            ),
            FeatureValue(
                code="TIME_mean_internal_pause_s",
                value=pause_analysis.mean_internal_pause_s,
                unit="s",
                source="energy-vad",
                confidence=0.7 if pause_analysis.mean_internal_pause_s is not None else None,
                method_version=EnergyVadPauseExtractor.method_version,
                evidence=evidence,
            ),
            FeatureValue(
                code="TIME_replica_duration_s",
                value=pause_analysis.replica_duration_s,
                unit="s",
                source="energy-vad",
                confidence=0.7 if pause_analysis.speech_segments else None,
                method_version=EnergyVadPauseExtractor.method_version,
                evidence=evidence,
            ),
            FeatureValue(
                code="TIME_audio_duration_s",
                value=pause_analysis.audio_duration_s,
                unit="s",
                source="wav-header",
                confidence=1.0,
                method_version="wav-header-v1",
                evidence={},
            ),
            FeatureValue(
                code="TIME_active_speech_duration_s",
                value=pause_analysis.active_speech_duration_s,
                unit="s",
                source="energy-vad",
                confidence=0.7 if pause_analysis.speech_segments else None,
                method_version=EnergyVadPauseExtractor.method_version,
                evidence=evidence,
            ),
        )

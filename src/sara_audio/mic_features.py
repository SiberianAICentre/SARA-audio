"""MIC-compatible scalar features derived from transcript and speech audio."""

from __future__ import annotations

import math
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sara_audio.domain import FeatureValue, Transcript, TranscriptToken
from sara_audio.errors import UnsupportedAudioError


@dataclass(frozen=True)
class MicFeatureSettings:
    """Settings for the lightweight MIC feature block."""

    enabled: bool = True
    utterance_merge_gap_s: float = 1.2

    def __post_init__(self) -> None:
        if self.utterance_merge_gap_s < 0:
            raise ValueError("mic.utterance_merge_gap_s cannot be negative")


class MicScalarFeatureExtractor:
    """Port the scalar MIC notebook features into the package pipeline."""

    method_version = "mic-scalar-v1"

    def __init__(self, settings: MicFeatureSettings | None = None) -> None:
        self._settings = settings or MicFeatureSettings()

    def extract(
        self,
        transcript: Transcript,
        speech_audio_path: Path,
    ) -> tuple[FeatureValue, ...]:
        samples, sample_rate = self._read_mono_pcm(speech_audio_path)
        duration_sec = len(samples) / sample_rate if sample_rate else 0.0
        rms_energy = (
            float(np.sqrt(np.mean(np.square(samples)))) if len(samples) else 0.0
        )
        tokens = self._timestamped_tokens(transcript.tokens)
        pauses_ms = self._interword_pauses_ms(tokens)
        utterance_rates = self._utterance_rates(tokens)
        word_count = len(transcript.text.replace("|", " ").split())
        evidence = {
            "transcript_source": transcript.source,
            "token_count": len(transcript.tokens),
            "timestamped_token_count": len(tokens),
            "audio_path": str(speech_audio_path),
            "utterance_merge_gap_s": self._settings.utterance_merge_gap_s,
        }
        return (
            self._value("word_count", float(word_count), "count", evidence),
            self._value(
                "avg_interword_pause_ms",
                float(np.mean(pauses_ms)) if pauses_ms else None,
                "ms",
                evidence,
            ),
            self._value(
                "std_interword_pause_ms",
                float(np.std(pauses_ms)) if pauses_ms else None,
                "ms",
                evidence,
            ),
            self._value(
                "avg_speech_rate_wps",
                float(np.mean(utterance_rates)) if utterance_rates else None,
                "words/s",
                evidence,
            ),
            self._value(
                "std_speech_rate_wps",
                float(np.std(utterance_rates)) if utterance_rates else None,
                "words/s",
                evidence,
            ),
            self._value(
                "acc_speech_rate_wps",
                word_count / duration_sec if duration_sec > 0 else None,
                "words/s",
                evidence,
            ),
            self._value("duration_sec", duration_sec, "s", evidence),
            self._value("rms_energy", round(rms_energy, 6), "normalized-rms", evidence),
        )

    @staticmethod
    def _read_mono_pcm(path: Path) -> tuple[np.ndarray, int]:
        try:
            with wave.open(str(path), "rb") as source:
                channels = source.getnchannels()
                sample_width = source.getsampwidth()
                sample_rate = source.getframerate()
                compression = source.getcomptype()
                if compression != "NONE" or channels != 1 or sample_width != 2:
                    raise UnsupportedAudioError(
                        "MIC scalar features require normalized mono 16-bit PCM WAV"
                    )
                raw = source.readframes(source.getnframes())
        except wave.Error as error:
            raise UnsupportedAudioError(f"Cannot decode WAV file {path}: {error}") from error
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
        return samples, sample_rate

    @staticmethod
    def _timestamped_tokens(
        tokens: tuple[TranscriptToken, ...],
    ) -> list[TranscriptToken]:
        return sorted(
            (
                token
                for token in tokens
                if token.start_s is not None
                and token.end_s is not None
                and token.end_s >= token.start_s
            ),
            key=lambda token: (token.start_s or 0.0, token.end_s or 0.0),
        )

    @staticmethod
    def _interword_pauses_ms(tokens: list[TranscriptToken]) -> list[float]:
        pauses = []
        for previous, following in zip(tokens, tokens[1:], strict=False):
            if previous.end_s is None or following.start_s is None:
                continue
            pause = following.start_s - previous.end_s
            if pause > 0:
                pauses.append(pause * 1000.0)
        return pauses

    def _utterance_rates(self, tokens: list[TranscriptToken]) -> list[float]:
        groups: list[list[TranscriptToken]] = []
        current: list[TranscriptToken] = []
        for token in tokens:
            if (
                current
                and current[-1].end_s is not None
                and token.start_s is not None
                and token.start_s - current[-1].end_s > self._settings.utterance_merge_gap_s
            ):
                groups.append(current)
                current = []
            current.append(token)
        if current:
            groups.append(current)
        rates = []
        for group in groups:
            start = group[0].start_s
            end = group[-1].end_s
            if start is None or end is None:
                continue
            duration = end - start
            if duration > 0 and math.isfinite(duration):
                rates.append(len(group) / duration)
        return rates

    def _value(
        self,
        code: str,
        value: float | None,
        unit: str,
        evidence: dict[str, object],
    ) -> FeatureValue:
        return FeatureValue(
            code=code,
            value=value,
            unit=unit,
            source="mic-transcript-audio" if value is not None else "not-computed",
            confidence=0.8 if value is not None else None,
            method_version=self.method_version,
            evidence=evidence
            if value is not None
            else {**evidence, "reason": "Timestamped transcript tokens are unavailable."},
        )

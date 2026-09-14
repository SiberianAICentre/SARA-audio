"""Deterministic energy VAD and the agreed pause semantics."""

from __future__ import annotations

import math
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sara_audio.domain import PauseAnalysis, PauseSegment, SpeechSegment
from sara_audio.errors import UnsupportedAudioError


@dataclass(frozen=True)
class VadSettings:
    """Settings for the deterministic WAV energy-VAD backend."""

    frame_duration_ms: int = 30
    pause_threshold_ms: int = 300
    min_speech_duration_ms: int = 90
    absolute_energy_floor_dbfs: float = -45.0
    activation_margin_db: float = 10.0

    def __post_init__(self) -> None:
        if self.frame_duration_ms <= 0:
            raise ValueError("frame_duration_ms must be positive")
        if self.pause_threshold_ms < 0 or self.min_speech_duration_ms <= 0:
            raise ValueError("VAD durations are invalid")


@dataclass(frozen=True)
class _EnergyFrame:
    start_s: float
    end_s: float
    dbfs: float


class EnergyVadPauseExtractor:
    """VAD for normalized PCM WAV files with transparent energy diagnostics.

    Speech islands separated by gaps shorter than ``pause_threshold_ms`` are
    merged. Consequently, only gaps at least that long are internal pauses.
    """

    method_version = "energy-vad-v1"

    def __init__(self, settings: VadSettings | None = None) -> None:
        self._settings = settings or VadSettings()

    def analyze(self, wav_path: Path) -> PauseAnalysis:
        frames, duration_s = self._read_energy_frames(wav_path)
        if not frames:
            return PauseAnalysis(duration_s, (), ())

        threshold_dbfs = self._speech_threshold(frames)
        provisional = self._speech_runs(frames, threshold_dbfs)
        speech_segments = self._merge_and_filter(provisional)
        pauses = self._derive_pauses(speech_segments, duration_s)
        return PauseAnalysis(duration_s, tuple(speech_segments), tuple(pauses))

    def _read_energy_frames(self, wav_path: Path) -> tuple[list[_EnergyFrame], float]:
        if wav_path.suffix.lower() != ".wav":
            raise UnsupportedAudioError("Energy VAD accepts normalized WAV input only")

        try:
            with wave.open(str(wav_path), "rb") as source:
                channels = source.getnchannels()
                sample_width = source.getsampwidth()
                sample_rate = source.getframerate()
                frame_count = source.getnframes()
                compression = source.getcomptype()
                if compression != "NONE":
                    raise UnsupportedAudioError("Compressed WAV is not supported")
                if sample_width != 2 or channels != 1:
                    raise UnsupportedAudioError(
                        "Energy VAD requires normalized 16-bit mono PCM WAV input"
                    )

                duration_s = frame_count / sample_rate if sample_rate else 0.0
                window_samples = max(
                    1, round(sample_rate * self._settings.frame_duration_ms / 1000)
                )
                frames: list[_EnergyFrame] = []
                sample_offset = 0
                while raw := source.readframes(window_samples):
                    samples = np.frombuffer(raw, dtype="<i2").astype(np.float64)
                    rms = float(np.sqrt(np.mean(np.square(samples))))
                    max_amplitude = 32767.0
                    dbfs = 20.0 * math.log10(max(rms / max_amplitude, 1e-10))
                    samples_read = len(raw) // (sample_width * channels)
                    start_s = sample_offset / sample_rate
                    sample_offset += samples_read
                    frames.append(_EnergyFrame(start_s, sample_offset / sample_rate, dbfs))
        except wave.Error as error:
            raise UnsupportedAudioError(f"Cannot decode WAV file {wav_path}: {error}") from error
        return frames, duration_s

    def _speech_threshold(self, frames: list[_EnergyFrame]) -> float:
        values = sorted(frame.dbfs for frame in frames)
        noise_index = min(len(values) - 1, max(0, int(len(values) * 0.2)))
        noise_floor = values[noise_index]
        # An utterance can contain almost no silence. In that case the lower
        # energy quantile is speech, so adapting above it would reject every
        # frame. The calibrated absolute floor is the conservative fallback.
        if noise_floor > self._settings.absolute_energy_floor_dbfs:
            return self._settings.absolute_energy_floor_dbfs
        return max(
            self._settings.absolute_energy_floor_dbfs,
            noise_floor + self._settings.activation_margin_db,
        )

    def _speech_runs(
        self, frames: list[_EnergyFrame], threshold_dbfs: float
    ) -> list[SpeechSegment]:
        runs: list[SpeechSegment] = []
        start_s: float | None = None
        end_s: float | None = None
        for frame in frames:
            if frame.dbfs >= threshold_dbfs:
                if start_s is None:
                    start_s = frame.start_s
                end_s = frame.end_s
            elif start_s is not None and end_s is not None:
                runs.append(SpeechSegment(start_s, end_s))
                start_s = None
                end_s = None
        if start_s is not None and end_s is not None:
            runs.append(SpeechSegment(start_s, end_s))
        return runs

    def _merge_and_filter(self, runs: list[SpeechSegment]) -> list[SpeechSegment]:
        min_speech_s = self._settings.min_speech_duration_ms / 1000
        pause_threshold_s = self._settings.pause_threshold_ms / 1000
        segments = [run for run in runs if run.duration_s >= min_speech_s]
        if not segments:
            return []

        merged: list[SpeechSegment] = [segments[0]]
        for segment in segments[1:]:
            previous = merged[-1]
            if segment.start_s - previous.end_s < pause_threshold_s:
                merged[-1] = SpeechSegment(previous.start_s, segment.end_s)
            else:
                merged.append(segment)
        return merged

    @staticmethod
    def _derive_pauses(
        segments: list[SpeechSegment], duration_s: float) -> list[PauseSegment]:
        if not segments:
            return []
        pauses: list[PauseSegment] = []
        first = segments[0]
        if first.start_s > 0:
            pauses.append(PauseSegment(0.0, first.start_s, "first"))
        for previous, following in zip(segments, segments[1:], strict=False):
            pauses.append(PauseSegment(previous.end_s, following.start_s, "internal"))
        last = segments[-1]
        if last.end_s < duration_s:
            pauses.append(PauseSegment(last.end_s, duration_s, "trailing"))
        return pauses

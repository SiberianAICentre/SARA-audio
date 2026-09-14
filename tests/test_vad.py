from __future__ import annotations

import struct
import tempfile
import unittest
import wave
from pathlib import Path

from sara_audio.vad import EnergyVadPauseExtractor, VadSettings


def write_signal(path: Path, parts: list[tuple[float, int]], sample_rate: int = 16000) -> None:
    samples: list[int] = []
    for duration_s, amplitude in parts:
        samples.extend([amplitude] * round(duration_s * sample_rate))
    with wave.open(str(path), "wb") as destination:
        destination.setnchannels(1)
        destination.setsampwidth(2)
        destination.setframerate(sample_rate)
        destination.writeframes(struct.pack(f"<{len(samples)}h", *samples))


class EnergyVadPauseExtractorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.extractor = EnergyVadPauseExtractor(
            VadSettings(
                frame_duration_ms=30,
                pause_threshold_ms=300,
                min_speech_duration_ms=60,
            )
        )

    def test_calculates_first_internal_and_replica_durations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "signal.wav"
            write_signal(
                audio,
                [
                    (0.30, 0),
                    (0.60, 12000),
                    (0.45, 0),
                    (0.60, 12000),
                    (0.20, 0),
                ],
            )
            analysis = self.extractor.analyze(audio)

        self.assertEqual(len(analysis.speech_segments), 2)
        self.assertAlmostEqual(analysis.first_pause_s or 0, 0.30, places=2)
        self.assertAlmostEqual(analysis.mean_internal_pause_s or 0, 0.45, places=2)
        self.assertAlmostEqual(analysis.replica_duration_s, 1.65, places=2)
        self.assertAlmostEqual(analysis.active_speech_duration_s, 1.20, places=2)

    def test_ignores_short_micro_pause_by_merging_speech(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "signal.wav"
            write_signal(audio, [(0.30, 12000), (0.15, 0), (0.30, 12000)])
            analysis = self.extractor.analyze(audio)

        self.assertEqual(len(analysis.speech_segments), 1)
        self.assertIsNone(analysis.mean_internal_pause_s)

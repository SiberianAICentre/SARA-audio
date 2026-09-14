from __future__ import annotations

import struct
import tempfile
import unittest
import wave
from pathlib import Path

from sara_audio.audio import AudioPreprocessor, AudioPreprocessorSettings
from sara_audio.errors import UnsupportedAudioError
from sara_audio.vad import EnergyVadPauseExtractor


class OperatorAudioTests(unittest.TestCase):
    def test_second_channel_is_extracted_and_pauses_are_removed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "stereo.wav"
            self._write_stereo_wav(source)
            preprocessor = AudioPreprocessor(
                AudioPreprocessorSettings(operator_channel_index=1)
            )
            normalized = preprocessor.prepare(source, root / "cache")
            analysis = EnergyVadPauseExtractor().analyze(normalized)
            speech_only = preprocessor.write_speech_only_wav(
                normalized, analysis.speech_segments, root / "speech_only.wav"
            )
            inspection = preprocessor.inspect_wav(speech_only)

        self.assertEqual(inspection.channels, 1)
        self.assertGreater(inspection.duration_s, 0.39)
        self.assertLess(inspection.duration_s, 0.45)

    def test_mono_input_falls_back_to_the_only_available_channel(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "mono.wav"
            self._write_mono_wav(source)
            preprocessor = AudioPreprocessor(
                AudioPreprocessorSettings(operator_channel_index=1)
            )
            normalized = preprocessor.prepare(source, Path(directory) / "cache")
            self.assertEqual(preprocessor.inspect_wav(normalized).channels, 1)

        self.assertTrue(preprocessor.used_mono_operator_fallback)

    def test_mono_input_can_be_rejected_when_fallback_is_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "mono.wav"
            self._write_mono_wav(source)
            preprocessor = AudioPreprocessor(
                AudioPreprocessorSettings(
                    operator_channel_index=1,
                    allow_mono_operator_fallback=False,
                )
            )
            with self.assertRaises(UnsupportedAudioError):
                preprocessor.prepare(source, Path(directory) / "cache")

    def test_mono_fallback_diagnostic_does_not_leak_to_next_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mono = root / "mono.wav"
            stereo = root / "stereo.wav"
            self._write_mono_wav(mono)
            self._write_stereo_wav(stereo)
            preprocessor = AudioPreprocessor(
                AudioPreprocessorSettings(operator_channel_index=1)
            )

            preprocessor.prepare(mono, root / "cache-mono")
            self.assertTrue(preprocessor.used_mono_operator_fallback)
            preprocessor.prepare(stereo, root / "cache-stereo")

        self.assertFalse(preprocessor.used_mono_operator_fallback)

    @staticmethod
    def _write_stereo_wav(path: Path) -> None:
        sample_rate = 16000
        left = [0] * sample_rate
        right = [0] * round(0.2 * sample_rate) + [10000] * round(0.4 * sample_rate)
        right += [0] * round(0.4 * sample_rate)
        frames = [value for pair in zip(left, right, strict=True) for value in pair]
        with wave.open(str(path), "wb") as destination:
            destination.setnchannels(2)
            destination.setsampwidth(2)
            destination.setframerate(sample_rate)
            destination.writeframes(struct.pack(f"<{len(frames)}h", *frames))

    @staticmethod
    def _write_mono_wav(path: Path) -> None:
        with wave.open(str(path), "wb") as destination:
            destination.setnchannels(1)
            destination.setsampwidth(2)
            destination.setframerate(16000)
            destination.writeframes(struct.pack("<16000h", *([0] * 16000)))

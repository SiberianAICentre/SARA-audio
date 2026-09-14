from __future__ import annotations

import struct
import tempfile
import unittest
import wave
from pathlib import Path

from sara_audio.config import PipelineSettings
from sara_audio.domain import Transcript
from sara_audio.pipeline import FeaturePipeline


class FeaturePipelineTests(unittest.TestCase):
    def test_pipeline_preserves_timing_rates_and_all_linguistic_codes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "speech.wav"
            self._write_wav(audio)
            settings = PipelineSettings(extract_opensmile=False)
            run = FeaturePipeline(root / "cache", settings).extract(
                audio,
                Transcript("Я очень рад, потому что это прекрасно!"),
            )
            self.assertTrue(run.analysis_audio_path.is_file())
            self.assertLess(
                run.analysis_audio_path.stat().st_size,
                audio.stat().st_size,
            )

        codes = {feature.code for feature in run.result.features}
        linguistic_codes = {
            code
            for code in codes
            if code.startswith(("STR_", "LEX_", "MOR_", "ORT_", "PUN_", "SYN_", "EMO_"))
        }
        self.assertIn("TIME_first_pause_s", codes)
        self.assertIn("RATE_syllables_per_second_active", codes)
        self.assertIn("word_count", codes)
        self.assertIn("rms_energy", codes)
        self.assertEqual(len(linguistic_codes), 92)
        self.assertIsNone(run.frames)

    @staticmethod
    def _write_wav(path: Path) -> None:
        sample_rate = 16000
        samples = [0] * round(0.3 * sample_rate)
        samples += [10000] * round(0.4 * sample_rate)
        with wave.open(str(path), "wb") as destination:
            destination.setnchannels(1)
            destination.setsampwidth(2)
            destination.setframerate(sample_rate)
            destination.writeframes(struct.pack(f"<{len(samples)}h", *samples))

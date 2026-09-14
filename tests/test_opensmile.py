from __future__ import annotations

import struct
import tempfile
import unittest
import wave
from pathlib import Path

import opensmile

from sara_audio.opensmile_extractor import OpenSmileExtractor


class OpenSmileExtractorTests(unittest.TestCase):
    def test_composite_output_contains_lpc_formant_columns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "speech.wav"
            with wave.open(str(audio), "wb") as destination:
                destination.setnchannels(1)
                destination.setsampwidth(2)
                destination.setframerate(16000)
                destination.writeframes(struct.pack("<16000h", *([10000] * 16000)))
            output = OpenSmileExtractor().extract_frames(audio)
            base = opensmile.Smile(
                feature_set=opensmile.FeatureSet.ComParE_2016,
                feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
            ).process_file(str(audio))
            formants = opensmile.Smile(
                feature_set=opensmile.FeatureSet.eGeMAPSv02,
                feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
            ).process_file(str(audio))

        self.assertEqual(len(output.frames), len(base) + len(formants))
        self.assertEqual(
            set(output.frames["frame_source"]), {"ComParE_2016", "eGeMAPSv02"}
        )
        self.assertTrue(output.semantic_coverage["formant_frequency"])
        self.assertTrue(output.semantic_coverage["formant_bandwidth"])
        self.assertIn("F1frequency_sma3nz", output.columns)
        self.assertIn("F1bandwidth_sma3nz", output.columns)
        features = OpenSmileExtractor().aggregate(output)
        codes = {feature.code for feature in features}
        self.assertIn("ACOUSTIC_pcm_RMSenergy_sma_mean", codes)
        self.assertIn("ACOUSTIC_F1frequency_sma3nz_mean", codes)

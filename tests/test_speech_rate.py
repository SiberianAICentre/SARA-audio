from __future__ import annotations

import unittest

from sara_audio.domain import PauseAnalysis, SpeechSegment, Transcript
from sara_audio.speech_rate import SpeechRateExtractor


class SpeechRateExtractorTests(unittest.TestCase):
    def test_rates_use_active_speech_not_replica_duration(self) -> None:
        analysis = PauseAnalysis(
            audio_duration_s=5.0,
            speech_segments=(SpeechSegment(1.0, 2.0), SpeechSegment(3.0, 4.0)),
            pauses=(),
        )
        values = SpeechRateExtractor().extract(Transcript("мама мыла раму"), analysis)
        by_code = {value.code: value for value in values}

        self.assertEqual(by_code["RATE_words_per_second_active"].value, 1.5)
        self.assertEqual(by_code["RATE_syllables_per_second_active"].value, 3.0)
        self.assertIsNone(by_code["RATE_phonemes_per_second_active"].value)

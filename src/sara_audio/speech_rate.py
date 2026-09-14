"""Speech-rate estimation over active, VAD-detected speech only."""

from __future__ import annotations

import re

from sara_audio.domain import FeatureValue, PauseAnalysis, Transcript

WORD_PATTERN = re.compile(r"[A-Za-zА-Яа-яЁё]+(?:-[A-Za-zА-Яа-яЁё]+)*")
RUSSIAN_VOWELS = frozenset("аеёиоуыэюяАЕЁИОУЫЭЮЯ")


class SpeechRateExtractor:
    """Derive explainable text rates normalized by active speech duration."""

    method_version = "speech-rate-v1"

    def extract(
        self, transcript: Transcript, pause_analysis: PauseAnalysis
    ) -> tuple[FeatureValue, ...]:
        active_s = pause_analysis.active_speech_duration_s
        words = WORD_PATTERN.findall(transcript.text)
        chars = sum(len(word.replace("-", "")) for word in words)
        syllables = sum(
            sum(character in RUSSIAN_VOWELS for character in word) for word in words
        )
        denominator = active_s if active_s > 0 else None
        confidence = 1.0 if transcript.is_verified else 0.7
        evidence = {
            "active_speech_duration_s": active_s,
            "word_count": len(words),
            "character_count": chars,
            "syllable_count": syllables,
            "transcript_source": transcript.source,
        }
        return (
            self._rate("RATE_words_per_second_active", len(words), denominator, confidence, evidence),
            self._rate("RATE_chars_per_second_active", chars, denominator, confidence, evidence),
            self._rate("RATE_syllables_per_second_active", syllables, denominator, confidence, evidence),
            FeatureValue(
                code="RATE_phonemes_per_second_active",
                value=None,
                unit="phonemes/s",
                source="g2p-not-configured",
                confidence=None,
                method_version=self.method_version,
                evidence={"reason": "A Russian G2P backend is not configured."},
            ),
        )

    def _rate(
        self,
        code: str,
        numerator: int,
        denominator: float | None,
        confidence: float,
        evidence: dict[str, object],
    ) -> FeatureValue:
        return FeatureValue(
            code=code,
            value=numerator / denominator if denominator else None,
            unit="items/s",
            source="transcript+vad",
            confidence=confidence if denominator else None,
            method_version=self.method_version,
            evidence=evidence,
        )

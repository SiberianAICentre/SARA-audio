"""Auditable scalar diagnostics derived from ASR token metadata."""

from __future__ import annotations

from sara_audio.domain import FeatureValue, Transcript


class AsrFeatureExtractor:
    """Derive five model-agnostic features from timestamped ASR tokens."""

    method_version = "asr-token-diagnostics-v1"

    def extract(self, transcript: Transcript) -> tuple[FeatureValue, ...]:
        tokens = transcript.tokens
        confidences = [token.confidence for token in tokens if token.confidence is not None]
        durations = [
            token.end_s - token.start_s
            for token in tokens
            if token.start_s is not None
            and token.end_s is not None
            and token.end_s >= token.start_s
        ]
        timed_count = sum(
            token.start_s is not None and token.end_s is not None for token in tokens
        )
        available = bool(tokens)
        evidence = {
            "transcript_source": transcript.source,
            "token_count": len(tokens),
        }
        return (
            self._value("ASR_token_count", float(len(tokens)) if available else None, "count", evidence),
            self._value(
                "ASR_mean_token_confidence",
                sum(confidences) / len(confidences) if confidences else None,
                "probability",
                evidence,
            ),
            self._value(
                "ASR_min_token_confidence",
                min(confidences) if confidences else None,
                "probability",
                evidence,
            ),
            self._value(
                "ASR_mean_token_duration_s",
                sum(durations) / len(durations) if durations else None,
                "s",
                evidence,
            ),
            self._value(
                "ASR_timed_token_fraction",
                timed_count / len(tokens) if tokens else None,
                "fraction",
                evidence,
            ),
        )

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
            source="asr-token-metadata" if value is not None else "not-computed",
            confidence=0.9 if value is not None else None,
            method_version=self.method_version,
            evidence=evidence
            if value is not None
            else {**evidence, "reason": "Transcript has no timestamped ASR tokens."},
        )

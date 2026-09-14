from sara_audio.asr_features import AsrFeatureExtractor
from sara_audio.domain import Transcript, TranscriptToken


def test_asr_features_are_derived_from_timestamped_tokens() -> None:
    values = AsrFeatureExtractor().extract(
        Transcript(
            "тест",
            tokens=(
                TranscriptToken("тест", 0.0, 0.4, 0.8),
                TranscriptToken("да", 0.4, 0.6, 0.6),
            ),
            source="faster-whisper",
            is_verified=False,
        )
    )
    by_code = {value.code: value for value in values}
    assert by_code["ASR_token_count"].value == 2.0
    assert by_code["ASR_mean_token_confidence"].value == 0.7
    assert by_code["ASR_mean_token_duration_s"].value == 0.3
    assert by_code["ASR_timed_token_fraction"].value == 1.0

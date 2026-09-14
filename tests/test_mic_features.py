from __future__ import annotations

import struct
import wave
from pathlib import Path

from sara_audio.domain import Transcript, TranscriptToken
from sara_audio.mic_features import MicScalarFeatureExtractor


def test_mic_scalar_features_use_timestamped_tokens_and_speech_audio(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "speech.wav"
    samples = [0] * 8000 + [16384] * 8000
    with wave.open(str(audio), "wb") as destination:
        destination.setnchannels(1)
        destination.setsampwidth(2)
        destination.setframerate(16000)
        destination.writeframes(struct.pack(f"<{len(samples)}h", *samples))
    transcript = Transcript(
        "раз два три",
        tokens=(
            TranscriptToken("раз", 0.0, 0.2, 0.9),
            TranscriptToken("два", 0.4, 0.6, 0.8),
            TranscriptToken("три", 2.0, 2.2, 0.7),
        ),
        source="faster-whisper",
        is_verified=False,
    )

    values = MicScalarFeatureExtractor().extract(transcript, audio)
    by_code = {value.code: value for value in values}

    assert by_code["word_count"].value == 3.0
    assert by_code["avg_interword_pause_ms"].value == 800.0
    assert round(by_code["avg_speech_rate_wps"].value, 6) == 4.166667
    assert by_code["duration_sec"].value == 1.0
    assert by_code["rms_energy"].value == 0.353553

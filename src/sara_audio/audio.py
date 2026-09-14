"""Audio inspection and deterministic conversion through ffmpeg."""

from __future__ import annotations

import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

from sara_audio.errors import DependencyUnavailableError, UnsupportedAudioError


@dataclass(frozen=True)
class AudioPreprocessorSettings:
    """Output format required by VAD and downstream audio extractors."""

    target_sample_rate_hz: int = 16000
    mono: bool = True
    operator_channel_index: int | None = None
    allow_mono_operator_fallback: bool = True


@dataclass(frozen=True)
class AudioInspection:
    """Minimal inspection information available without a media library."""

    path: Path
    duration_s: float
    sample_rate_hz: int
    channels: int
    sample_width_bytes: int


class AudioPreprocessor:
    """Prepare arbitrary media as a normalized operator-channel PCM WAV."""

    def __init__(self, settings: AudioPreprocessorSettings | None = None) -> None:
        self._settings = settings or AudioPreprocessorSettings()
        self._used_mono_operator_fallback = False

    @property
    def used_mono_operator_fallback(self) -> bool:
        """Whether the most recent preparation used the sole source channel."""
        return self._used_mono_operator_fallback

    def prepare(self, input_path: Path, cache_directory: Path) -> Path:
        input_path = input_path.resolve()
        if not input_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {input_path}")

        self._used_mono_operator_fallback = False
        cache_directory.mkdir(parents=True, exist_ok=True)
        suffix = (
            f".channel-{self._settings.operator_channel_index}"
            if self._settings.operator_channel_index is not None
            else ""
        )
        output_path = cache_directory / f"{input_path.stem}{suffix}.normalized.wav"
        if self._is_normalized_wav(input_path):
            return input_path
        selected_channel = self._selected_channel(input_path)

        ffmpeg = self._resolve_ffmpeg()
        if ffmpeg is None:
            raise DependencyUnavailableError(
                "ffmpeg was not found. Install ffmpeg and make it available in PATH."
            )
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(input_path),
        ]
        if selected_channel is not None:
            command.extend(("-filter:a", f"pan=mono|c0=c{selected_channel}"))
        command.extend((
            "-ac",
            "1" if self._settings.mono or selected_channel is not None else "2",
            "-ar",
            str(self._settings.target_sample_rate_hz),
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ))
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as error:
            detail = error.stderr.strip() or "ffmpeg conversion failed"
            raise UnsupportedAudioError(f"Cannot convert {input_path.name}: {detail}") from error
        return output_path

    def write_speech_only_wav(
        self,
        normalized_wav_path: Path,
        speech_segments: tuple[object, ...],
        output_path: Path,
    ) -> Path:
        """Concatenate VAD speech intervals while preserving PCM format."""
        inspection = self.inspect_wav(normalized_wav_path)
        if (
            inspection.channels != 1
            or inspection.sample_width_bytes != 2
            or inspection.sample_rate_hz != self._settings.target_sample_rate_hz
        ):
            raise UnsupportedAudioError(
                "Speech-only WAV requires normalized mono 16-bit PCM input"
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(normalized_wav_path), "rb") as source:
            with wave.open(str(output_path), "wb") as destination:
                destination.setnchannels(1)
                destination.setsampwidth(2)
                destination.setframerate(inspection.sample_rate_hz)
                for segment in speech_segments:
                    start_frame = round(segment.start_s * inspection.sample_rate_hz)
                    end_frame = round(segment.end_s * inspection.sample_rate_hz)
                    source.setpos(start_frame)
                    destination.writeframes(source.readframes(end_frame - start_frame))
        return output_path

    @staticmethod
    def _resolve_ffmpeg() -> str | None:
        system_ffmpeg = shutil.which("ffmpeg")
        if system_ffmpeg:
            return system_ffmpeg
        try:
            import imageio_ffmpeg
        except ImportError:
            return None
        return imageio_ffmpeg.get_ffmpeg_exe()

    def inspect_wav(self, wav_path: Path) -> AudioInspection:
        try:
            with wave.open(str(wav_path), "rb") as source:
                sample_rate_hz = source.getframerate()
                frames = source.getnframes()
                return AudioInspection(
                    path=wav_path,
                    duration_s=frames / sample_rate_hz if sample_rate_hz else 0.0,
                    sample_rate_hz=sample_rate_hz,
                    channels=source.getnchannels(),
                    sample_width_bytes=source.getsampwidth(),
                )
        except wave.Error as error:
            raise UnsupportedAudioError(f"Cannot inspect WAV file {wav_path}: {error}") from error

    def _is_normalized_wav(self, input_path: Path) -> bool:
        if input_path.suffix.lower() != ".wav":
            return False
        try:
            inspection = self.inspect_wav(input_path)
        except UnsupportedAudioError:
            return False
        return (
            self._settings.operator_channel_index is None
            and
            inspection.channels == (1 if self._settings.mono else 2)
            and inspection.sample_rate_hz == self._settings.target_sample_rate_hz
            and inspection.sample_width_bytes == 2
        )

    def _selected_channel(self, input_path: Path) -> int | None:
        channel = self._settings.operator_channel_index
        if channel is None:
            return None
        if channel < 0:
            raise ValueError("audio.operator_channel_index cannot be negative")
        if input_path.suffix.lower() != ".wav":
            return channel
        inspection = self.inspect_wav(input_path)
        if inspection.channels <= channel:
            if inspection.channels == 1 and self._settings.allow_mono_operator_fallback:
                self._used_mono_operator_fallback = True
                return None
            raise UnsupportedAudioError(
                f"{input_path.name} has {inspection.channels} channel(s), but "
                f"operator_channel_index={channel} was requested"
            )
        return channel
